#!/usr/bin/env python3
"""External RSS watchdog for campaign runs (C5, block E14; risk R17).

Why this exists (see docs/papers/big_draft/PRE_WRITING_CAMPAIGN.md, R17): an
OOM'd child process must never be recorded as an honest solver negative. The
mitigation is an EXTERNAL sampler that watches the whole process TREE's
resident set size (RSS) and kills it before the OS OOM-killer (or a silent
swap death) can turn a "solver_infeasible" into a fabricated one. Two design
rules are load-bearing and must not be relaxed by a future edit:

  1. NEVER ``RLIMIT_AS`` (or any other rlimit). ``RLIMIT_AS`` caps *virtual*
     address space, which MOSEK/CVXPY/numpy routinely over-commit far beyond
     actual RSS; it kills healthy solves and is unavailable on Windows dev
     boxes at all. External RSS sampling is the only cross-platform,
     precision-matched mechanism.
  2. NEVER trust a piped exit code. ``cmd | tee log; echo $?`` reports
     ``tee``'s exit status, not ``cmd``'s -- a classic way an OOM'd solver
     silently looks like success. This watchdog spawns the child directly as
     an argv list (no ``shell=True``, no pipe), so its own exit_code is
     always the child's own, and its docstring/CLI help say so loudly: do
     not wrap a shell pipeline as the child command.

Windows-importable by construction: process/RSS sampling goes through
``psutil`` exclusively (no ``/proc``, no POSIX-only rlimit calls), so this
module imports and its tests run on the Windows dev box even though the
campaign target machine is Linux aarch64.

CLI:
    python scripts/_watchdog.py --limit-gb 110 --interval 5 \\
        --json-out <path> -- <cmd> [args...]

Programmatic use (e.g. from scripts/scaling_study.py's --measure cells):
    from _watchdog import run_with_watchdog
    result = run_with_watchdog(["python", "worker.py"], limit_gb=100,
                                interval=5.0, json_out="cell.json")
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import psutil
from _artifact_io import atomic_json

# Exit code this watchdog uses for its own OOM kill (distinct from any exit
# code the child itself could have produced; mirrors the Unix SIGKILL (128+9)
# convention without claiming the child was actually signaled that way on
# every platform).
OOM_EXIT_CODE = 137

# Exit code recorded for a child whose tree this watchdog killed and whose real
# wait status could not be recovered. -9 is what CPython's subprocess reports
# for a SIGKILLed child, i.e. the same value the recoverable path yields, and it
# is never 0: a killed child must never be bookkept as a clean exit. Distinct
# from OOM_EXIT_CODE above, which is this watchdog's OWN process exit code.
KILLED_TREE_EXIT_CODE = -9


def _tree_rss_bytes(proc: psutil.Process) -> int:
    """Sum RSS across ``proc`` and every live descendant.

    Processes can exit between listing and querying; every per-process read
    is guarded so a race never crashes the sampler mid-run.
    """
    total = 0
    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


def _kill_tree(proc: psutil.Process, timeout: float = 10.0) -> None:
    """Kill ``proc`` and every descendant. Children first, then the parent."""
    try:
        children = proc.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    all_procs = children + [proc]
    for p in all_procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(all_procs, timeout=timeout)


def run_with_watchdog(
    cmd: Sequence[str],
    limit_gb: Optional[float],
    interval: float = 5.0,
    json_out: Optional[str] = None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    timeout_s: Optional[float] = None,
) -> dict:
    """Spawn ``cmd`` and enforce optional RSS and wall-clock bounds on its tree.

    ``limit_gb=None`` disables the RSS cap; ``timeout_s=None`` disables the
    deadline. A deadline is checked with a monotonic clock and kills the tree.
    Samples the whole process tree's RSS every ``interval`` seconds; kills
    the tree the moment the sample exceeds ``limit_gb``. Returns (and,
    if ``json_out`` is given, writes) a provenance record:

        peak_rss_gb, limit_gb, samples (list of {t_s, rss_gb}), exit_code,
        status ("oom" | "timeout" | "unexpected_error" | None), killed_by_watchdog,
        wall_time_s, command, timestamp_utc_start, timestamp_utc_end.

    ``status`` follows the campaign's typed vocabulary but only ever
    classifies the WATCHDOG's own verdict: "oom" if this watchdog killed the
    tree for exceeding ``limit_gb``, "unexpected_error" if the child exited
    nonzero on its own, else ``None`` -- a clean exit (code 0) is left
    UNTOUCHED here; the child's own JSON output (if any) carries the
    campaign-vocabulary classification (certified / solver_infeasible / ...)
    for what actually happened inside the run. This watchdog never invents
    that classification itself.

    ``exit_code`` is the child's own wait status and is 0 only on a clean
    exit: after a watchdog kill it is the real signal status (-9 on POSIX),
    recovered from psutil, or ``KILLED_TREE_EXIT_CODE`` if even that is gone.
    So an ``"oom"`` record can never carry ``exit_code: 0``. Records written
    before this was fixed (P6-16) do; their ``status`` field is unaffected.
    """
    cmd = list(cmd)
    if not cmd:
        raise ValueError("run_with_watchdog: empty command")

    for label, value in (("limit_gb", limit_gb), ("timeout_s", timeout_s), ("interval", interval)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{label} must be positive and finite")

    t_start_monotonic = time.monotonic()
    ts_start = datetime.now(timezone.utc).isoformat()

    child = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        proc = psutil.Process(child.pid)
    except psutil.NoSuchProcess:
        # The child exited before we could even attach to it; fall through
        # to a zero-sample record below rather than crashing the watchdog.
        proc = None

    limit_bytes = float(limit_gb) * (1024 ** 3) if limit_gb is not None else float("inf")
    samples: List[dict] = []
    peak_rss_bytes = 0
    killed_for_oom = False
    killed_for_timeout = False
    killed_rc = None  # the child's real wait status, recovered from psutil

    try:
        while True:
            ret = child.poll()
            rss_bytes = _tree_rss_bytes(proc) if proc is not None else 0
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            samples.append({
                "t_s": round(time.monotonic() - t_start_monotonic, 3),
                "rss_gb": rss_bytes / (1024 ** 3),
            })
            # Only a child that is still RUNNING can be killed. A child that
            # already exited on its own keeps its own exit code and gets no
            # "oom" verdict, even if this last sample is over the limit:
            # claiming a kill that never happened would be the same
            # bookkeeping dishonesty as the exit code 0 below, reversed.
            over_time = (timeout_s is not None and
                         time.monotonic() - t_start_monotonic >= timeout_s)
            if ret is None and (rss_bytes > limit_bytes or over_time):
                killed_for_oom = rss_bytes > limit_bytes
                killed_for_timeout = over_time and not killed_for_oom
                if proc is not None:
                    _kill_tree(proc)
                    # _kill_tree ends in psutil.wait_procs, which REAPS the
                    # child (a direct child of this process, so psutil calls
                    # waitpid and stores the status on proc.returncode).
                    # Popen's own waitpid below then raises ChildProcessError
                    # and CPython's Popen._try_wait fallback records status 0
                    # -- which is why killed runs used to report exit_code 0
                    # (P6-15, same bug in scripts/solver_comparison.py).
                    # psutil's status is the real one (-9 for the SIGKILL).
                    killed_rc = getattr(proc, "returncode", None)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                break
            if ret is not None:
                break
            remaining = (timeout_s - (time.monotonic() - t_start_monotonic)
                         if timeout_s is not None else interval)
            time.sleep(max(0.0, min(interval, remaining)))
    finally:
        if child.poll() is None:
            # Belt-and-braces: never leave an orphaned tree behind even if
            # something above raised.
            if proc is not None:
                _kill_tree(proc)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    exit_code = child.returncode
    if (killed_for_oom or killed_for_timeout) and exit_code in (None, 0):
        # See the comment above: Popen's 0 here is an artefact of the reap
        # race, never an observed clean exit. Prefer psutil's real status;
        # fall back to the typed sentinel if even that is unavailable.
        exit_code = killed_rc if killed_rc not in (None, 0) else KILLED_TREE_EXIT_CODE

    wall_time_s = time.monotonic() - t_start_monotonic
    ts_end = datetime.now(timezone.utc).isoformat()

    if killed_for_oom:
        status = "oom"
    elif killed_for_timeout:
        status = "timeout"
    elif exit_code != 0:
        status = "unexpected_error"
    else:
        status = None  # clean exit: the child's own result is untouched

    result = {
        "peak_rss_gb": peak_rss_bytes / (1024 ** 3),
        "limit_gb": float(limit_gb) if limit_gb is not None else None,
        "timeout_s": timeout_s,
        "interval_s": float(interval),
        "samples": samples,
        "exit_code": exit_code,
        "status": status,
        "killed_by_watchdog": killed_for_oom or killed_for_timeout,
        "wall_time_s": wall_time_s,
        "command": cmd,
        "timestamp_utc_start": ts_start,
        "timestamp_utc_end": ts_end,
    }

    if json_out is not None:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out_path, result)

    return result


def _split_argv(argv: List[str]):
    """Split ``argv`` at the first bare ``--`` into (watchdog_args, child_cmd)."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def main() -> None:
    argv = sys.argv[1:]
    watchdog_argv, child_cmd = _split_argv(argv)

    parser = argparse.ArgumentParser(
        prog="scripts/_watchdog.py",
        description=(
            "External whole-process-tree RSS watchdog (C5, campaign risk R17). "
            "Samples RSS via psutil (never RLIMIT_AS) and kills the tree if it "
            "exceeds --limit-gb."
        ),
        epilog=(
            "IMPORTANT: never wrap a shell pipeline as the child command "
            "(e.g. `-- bash -c \"solve.py | tee log\"`). A piped command's exit "
            "code reflects the last stage of the pipe (e.g. tee), not the "
            "actual program -- an OOM'd solver can silently look like success "
            "(campaign rule R17). Invoke the target program directly; if you "
            "need a log file, redirect it with the target program's own "
            "flags or with Python's subprocess redirection, not a shell pipe."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit-gb", type=float, required=True,
                        help="RSS limit in GiB for the whole process tree.")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Sampling interval in seconds (default: 5).")
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="Optional hard wall-clock limit; kills the process tree.")
    parser.add_argument("--json-out", required=True,
                        help="Path to write the provenance JSON record.")
    args = parser.parse_args(watchdog_argv)

    if not child_cmd:
        parser.error(
            "no child command given. Usage: "
            "python scripts/_watchdog.py --limit-gb 110 --interval 5 "
            "--json-out <path> -- <cmd> [args...]"
        )

    result = run_with_watchdog(
        child_cmd, limit_gb=args.limit_gb, interval=args.interval,
        json_out=args.json_out, timeout_s=args.timeout_s,
    )

    summary = {k: v for k, v in result.items() if k != "samples"}
    summary["n_samples"] = len(result["samples"])
    print(json.dumps(summary, indent=2))

    if result["status"] == "oom":
        sys.exit(OOM_EXIT_CODE)
    if result["status"] == "timeout":
        sys.exit(124)
    sys.exit(result["exit_code"] if result["exit_code"] is not None else 1)


if __name__ == "__main__":
    main()
