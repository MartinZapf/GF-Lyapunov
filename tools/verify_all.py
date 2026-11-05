#!/usr/bin/env python3
"""Re-run the independent verifier on every certificate of the paper.

STDLIB ONLY (Python >= 3.9). Run from the root of the release:

    python3 tools/verify_all.py                    # every certificate except the long ones
    python3 tools/verify_all.py --only twisting    # one certificate
    python3 tools/verify_all.py --include-long     # also order four (about 10 h each)

For each certificate in MANIFEST.json it

  1. checks the sha256 of the record and of the Gram file against the manifest
     (a mismatch is reported as HASH MISMATCH and the verifier is not run: a
     verdict on other bytes says nothing about the published certificate);
  2. runs ``verification/standalone/verify_certificate.py RECORD --grams GRAMS
     --json logs/.../<id>.verification.json`` in a fresh process, with stdout
     and stderr in ``logs/.../<id>.log``, and measures wall time and peak RSS;
  3. reads the outcome class from the verifier's last ``RESULT:`` line
     (VERIFIED / REJECTED / INVALID INPUT / ERROR) and compares the exact
     certified margin it prints with the one stated in MANIFEST.json.

It writes ``summary.json`` and ``summary.tsv`` into the log folder and exits
0 iff every certificate it ran is VERIFIED with the expected margin and no
file was missing or altered. Nothing here decides a proof; the verifier does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = Path("verification/standalone/verify_certificate.py")
RESULT_RE = re.compile(r"^RESULT: (VERIFIED|REJECTED|INVALID INPUT|ERROR) \(exit code (\d)\)")
EPS_RE = re.compile(r"eps_certified = eps - charge = (-?\d+/\d+)")
BASIS_RE = re.compile(r"^\[(3|4)/5\] .*?(\d+) basis -> (\d+) parity blocks, largest (\d+)")
WORST_RE = re.compile(r"worst charge/eps ([0-9.e+-]+)")
ETA_RE = re.compile(r"psd_shift: (.*)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    env = {"python": platform.python_version(),
           "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count(), "cpu_model": None, "ram_total_kb": None,
           "verifier_sha256": None}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                env["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                env["ram_total_kb"] = int(line.split()[1])
    except OSError:
        pass
    vp = ROOT / VERIFIER
    if vp.exists():
        env["verifier_sha256"] = sha256(vp)
    return env


def run_one(python: str, rec: Path, grams: Path, json_out: Path, log: Path,
            timeout: float | None):
    """Run the verifier in a fresh process; return (exit code, wall s, peak RSS kB)."""
    cmd = [python, str(VERIFIER), str(rec), "--grams", str(grams), "--json", str(json_out)]
    if (ROOT / json_out).exists():
        (ROOT / json_out).unlink()
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        rss = None
        if hasattr(os, "wait4") and timeout is None:
            _, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status) \
                if hasattr(os, "waitstatus_to_exitcode") else (status >> 8)
            rss = ru.ru_maxrss          # kB on Linux, bytes on macOS
            if sys.platform == "darwin":
                rss //= 1024
        else:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                fh.write(f"\nTIMEOUT after {timeout} s (no verdict)\n")
                return None, time.monotonic() - t0, None, cmd
        wall = time.monotonic() - t0
        fh.write(f"\nEXIT {proc.returncode}\nWALL_S {wall:.3f}\n"
                 + (f"PEAK_RSS_KB {rss}\n" if rss is not None else ""))
    return proc.returncode, wall, rss, cmd


def shown(path: Path) -> str:
    """A path relative to the release root when it lies inside it."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_log(text: str) -> dict:
    out = {"result_line": None, "verdict": None, "eps_line": None, "basis": {},
           "worst": [], "psd_shift": None}
    for line in text.splitlines():
        m = RESULT_RE.match(line)
        if m:
            out["result_line"], out["verdict"] = line.strip(), m.group(1)
        m = EPS_RE.search(line)
        if m:
            out["eps_line"] = m.group(1)
        m = BASIS_RE.match(line)
        if m:
            out["basis"]["V" if m.group(1) == "3" else "W"] = {
                "n_basis_max": int(m.group(2)), "n_blocks_min": int(m.group(3)),
                "largest_block": int(m.group(4))}
        m = ETA_RE.search(line)
        if m:
            out["psd_shift"] = m.group(1).strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(ROOT / "MANIFEST.json"))
    ap.add_argument("--only", nargs="+", default=None, metavar="ID")
    ap.add_argument("--include-long", action="store_true",
                    help="also run the certificates marked long_run in MANIFEST.json (order "
                         "four: about 10 h "
                         "each, single-threaded, about 9 GB RAM)")
    ap.add_argument("--logs", default=None,
                    help="log folder (default logs/verify_all_<UTC timestamp>)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the verifier (default: this one)")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per run")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = Path(args.logs).resolve() if args.logs else ROOT / "logs" / f"verify_all_{stamp}"
    logs.mkdir(parents=True, exist_ok=True)
    env = environment()
    env["started_utc"] = stamp
    env["python_used_for_verifier"] = args.python
    if env["verifier_sha256"] != manifest["verifier"]["sha256"]:
        print(f"WARNING: verifier sha256 {env['verifier_sha256']} differs from the manifest's "
              f"{manifest['verifier']['sha256']}", file=sys.stderr)

    results = []
    for cert in manifest["certificates"]:
        cid = cert["id"]
        if args.only and cid not in args.only:
            continue
        row = {"id": cid, "role": cert["role"]}
        files = cert["files"]
        rec, grams = ROOT / files["record"]["path"], ROOT / files["grams"]["path"]
        long_run = bool(cert.get("long_run"))     # marked in MANIFEST.json (order four)
        if long_run and not args.include_long:
            row.update(verdict="SKIPPED (long; use --include-long)",
                       note=cert.get("verify", {}).get("status"))
            results.append(row)
            print(f"{cid:38s} SKIPPED  (long run; --include-long)")
            continue
        missing = [str(p.relative_to(ROOT)) for p in (rec, grams) if not p.exists()]
        if missing:
            row.update(verdict="MISSING FILE", missing=missing)
            results.append(row)
            print(f"{cid:38s} MISSING  {missing}")
            continue
        bad = [k for k, p in (("record", rec), ("grams", grams))
               if sha256(p) != files[k]["sha256"]]
        if bad:
            row.update(verdict="HASH MISMATCH", files=bad)
            results.append(row)
            print(f"{cid:38s} HASH MISMATCH {bad}: not run")
            continue
        json_out, log = logs / f"{cid}.verification.json", logs / f"{cid}.log"
        json_arg = Path(shown(json_out))   # relative to the release root when inside it
        code, wall, rss, cmd = run_one(args.python, rec.relative_to(ROOT),
                                       grams.relative_to(ROOT), json_arg, log, args.timeout)
        parsed = parse_log(log.read_text(errors="replace"))
        expected_eps = cert.get("verify", {}).get("expected_eps_line_prefix")
        row.update({"verdict": parsed["verdict"] or ("TIMEOUT" if code is None else "NO RESULT LINE"),
                    "exit_code": code, "wall_s": round(wall, 3), "peak_rss_kb": rss,
                    "command": " ".join(cmd[1:]), "log": shown(log),
                    "eps_line": parsed["eps_line"],
                    "eps_matches_manifest": (None if not expected_eps
                                             else parsed["eps_line"] == expected_eps),
                    "log_basis_lines": parsed["basis"], "psd_shift_eta": parsed["psd_shift"]})
        if json_out.exists():
            js = json.loads(json_out.read_text())
            row["eps_certified_exact"] = js.get("eps_certified_exact")
            row["worst_charge_over_eps"] = js.get("worst_charge_over_eps")
            row["proof_inputs"] = js.get("proof_inputs")
            row["psd_shift_eta_json"] = js.get("psd_shift_eta")
        results.append(row)
        print(f"{cid:38s} {row['verdict']:8s} exit {code}  {wall:9.2f} s  "
              f"RSS {rss if rss is not None else '?'} kB"
              + ("" if row["eps_matches_manifest"] in (None, True) else "  EPS MISMATCH"))

    ok = all(r["verdict"] == "VERIFIED" or r["verdict"].startswith("SKIPPED")
             for r in results) and all(r.get("eps_matches_manifest") in (None, True)
                                       for r in results)
    env["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (logs / "summary.json").write_text(json.dumps(
        {"environment": env, "all_ok": ok, "results": results}, indent=1) + "\n")
    with open(logs / "summary.tsv", "w") as fh:
        fh.write("id\trole\tverdict\texit\twall_s\tpeak_rss_kb\teps_certified\n")
        for r in results:
            fh.write(f"{r['id']}\t{r['role']}\t{r['verdict']}\t{r.get('exit_code', '')}\t"
                     f"{r.get('wall_s', '')}\t{r.get('peak_rss_kb', '')}\t"
                     f"{r.get('eps_line', '')}\n")
    print(f"\n{'ALL OK' if ok else 'NOT ALL OK'}; logs in {logs}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
