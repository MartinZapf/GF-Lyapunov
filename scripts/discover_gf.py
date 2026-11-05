#!/usr/bin/env python3
"""Paper-machinery GF-SOS discovery for all controllers (full library).

For each controller: build the C¹ Generalized-Form term library, keep the
μ-compatible terms, and solve the standard SOS problem for the coefficients over
the FULL library (associated forms + Gram P ⪰ 0, strict ε-margin; Sanchez-Moreno
2018 §5.3 / Algorithm 8). Degree m is escalated until the strict Lemma-5 gate
certifies V>0 and W=−V̇>0 (with a-posteriori Gram validation). A system that does
not certify at the reachable degrees is reported as an honest negative — there is
no regional/grid fallback.

See ``gf_framework.discovery.gf_discovery``.

``--parity-reduction`` (campaign block S1, OFF by default) poses the DESIGN
SDP over the even-parity sub-library and over one Gram block per orbit of
x ↦ −x (2^(n-1) V and W blocks; the benchmark systems here are nominal, so
the nominal reduction applies — perturbed drivers get the PAIRED one). It is
switched on through ``joint_sos.set_parity_reduction_default`` because the
optimizer is constructed inside ``discovery.gf_discovery._design_full_library``,
not here. DESIGN SIDE ONLY: the strict re-certification gate inside
``discover_gf`` verifies the full 2^n block set either way, which is exactly
the campaign's S1 validation gate. Prerequisites are verified symbolically
per system and the reduction refuses loudly (to stderr) if any fails.
Without the flag every code path is the pre-existing one.

``--gains`` (single controller only) replaces the system's published gain
point with an explicit one, mapped in order onto the spec factory's gain
arguments. It exists because campaign block E6 had to run its candidate gain
sets through an out-of-repo copy of ``run_one`` (now rescued to
``scripts/campaign/e6/discover_e6_variant.py``): the factories always accepted
gain keywords, this driver just never passed any. Records from a gain override
carry a ``gains`` field and land in ``<system>_gains_<tag>.json``, so they can
never be mistaken for a published-gain record.

``--gram-bundle-dir`` (default: a ``design_grams/`` sibling of the row
checkpoints, inside ``--output-dir``) persists the DESIGN Gram bundle of every
certified row, one file per system, in the schema
``gf.design_grams/1`` that ``review/reverify_exact.py --gram-bundle-dir``
consumes. It is the SAME object the design SDP solved: ``discover_gf`` keeps
the ``JointSOSResult`` (P2-11) instead of dropping it, so stage D of the
citable chain is ONE command and X re-proves on D's own Grams with no second
float solve. ``--epsilon`` for stage X must be the design margin recorded in
the bundle (``design_epsilon``, 1e-4 by default).

Usage:
    python -u scripts/discover_gf.py --controllers all --solver MOSEK \
        --output-dir artifacts/publication/results_gf_sos/

    python -u scripts/discover_gf.py --controllers pidsmc4d \
        --gains 3,4.160167646103808,3.0594733539832584,1.1 \
        --solver MOSEK_DIRECT --gate-solver MOSEK --no-design-fallback \
        --parity-reduction --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _artifact_io import atomic_json, atomic_text, atomic_target, file_sha256, result_receipt
from _watchdog import run_with_watchdog

import gf_framework.discovery.universal_discovery as ud
from gf_framework.discovery.gf_discovery import discover_gf
from gf_framework.verification.exact_phase import save_gram_bundle
from gf_framework.verification.joint_sos import (
    GRAM_BUNDLE_SCHEMA, set_parity_reduction_default)

#: Bundle files above this many stored Gram scalars are written as ``.npz``
#: (float64 arrays + one JSON metadata string) instead of ``.json``; below it
#: the human-readable form is worth the bytes. levant4d (32 blocks) is ~5e5
#: scalars, sta (8 blocks of 6x6) is 288.
NPZ_SCALAR_THRESHOLD = 20000


def solver_versions() -> dict:
    """Versions that can move a Gram fingerprint, recorded with every bundle.

    P7-05 measured one MOSEK minor bump (11.1.11 -> 11.2.3) moving
    ``gram_min_eig`` in the 13th significant digit and changing BOTH the alpha
    and the bundle fingerprint at a fixed hash seed. A stored fingerprint is
    therefore only reproducible against the same solver build, so the build is
    part of the record. These keys live in the bundle DOCUMENT metadata, never
    in an entry: ``gram_bundle_fingerprint`` hashes entries only, so recording
    them cannot change the object the exact phase re-proves.
    """
    import platform
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("mosek", "numpy", "cvxpy", "scipy", "sympy", "clarabel", "scs"):
        try:
            m = __import__(mod)
        except Exception as exc:                                  # noqa: BLE001
            out[mod] = f"unavailable: {type(exc).__name__}"
            continue
        if mod == "mosek":
            try:
                out[mod] = ".".join(str(v) for v in m.Env.getversion())
            except Exception as exc:                              # noqa: BLE001
                out[mod] = f"unknown: {type(exc).__name__}"
        else:
            out[mod] = getattr(m, "__version__", "unknown")
    return out


def bundle_path_for(bundle_dir: Path, system: str, entries: list) -> Path:
    """``<dir>/<system>.npz`` for a large bundle, ``<system>.json`` otherwise.

    Both names are in ``review/reverify_exact.py``'s ``BUNDLE_SUFFIXES``, which
    probes ``.npz`` first, so a stale file of the other extension in the same
    directory would shadow this one — the writer removes it.
    """
    n_scalars = sum(len(e["gram"]) ** 2 for e in entries)
    suffix = ".npz" if n_scalars >= NPZ_SCALAR_THRESHOLD else ".json"
    return bundle_dir / f"{system}{suffix}"


def write_gram_bundle(bundle_dir, system: str, result, degree_m: int,
                      solver: str, gate_solver: str, artifact_name=None) -> dict:
    """Persist one row's DESIGN Gram bundle; returns the row's record of it.

    The bundle is ``result.gram_certificates`` verbatim — the Grams the design
    SDP produced for the V this row certified, in the canonical library order
    (``grlex-v1``, P5-13) and at the design margin. Nothing is re-solved.
    """
    record = {"schema": GRAM_BUNDLE_SCHEMA, "path": None, "fingerprint": None,
              "n_blocks": 0, "epsilon": result.design_epsilon,
              "reason": result.gram_bundle_reason,
              "versions": solver_versions()}
    if not result.gram_certificates:
        record["reason"] = result.gram_bundle_reason or "no_gram_certificates"
        return record
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    entries = result.gram_certificates
    path = bundle_path_for(bundle_dir, artifact_name or system, entries)
    with atomic_target(path) as temporary:
        save_gram_bundle(
            temporary, entries,
            system=system, epsilon=result.design_epsilon, degree_m=degree_m,
            solver=solver, gate_solver=gate_solver,
            library_order=entries[0].get("library_order"),
            pythonhashseed=os.environ.get("PYTHONHASHSEED"),
            versions=record["versions"],
            produced_by="scripts/discover_gf.py")
    for other in (".npz", ".json.gz", ".json"):
        stale = bundle_dir / f"{artifact_name or system}{other}"
        if stale != path and stale.exists():
            stale.unlink()
    record.update(path=str(path), n_blocks=len(entries),
                  fingerprint=result.gram_bundle_fingerprint,
                  library_order=entries[0].get("library_order"))
    return record

SPEC_FACTORIES = {
    "fosmc": "create_fosmc_spec",
    "power_rate": "create_power_rate_spec",
    "sta": "create_sta_spec",
    "twisting": "create_twisting_spec",
    "example1": "create_example1_spec",
    "example3": "create_example3_spec",
    "continuous_twisting": "create_cta_spec",
    "levant3d": "create_levant3d_spec",
    "pidsmc": "create_pidsmc_spec",
    "dsta": "create_dsta_spec",
    "fc_cta": "create_fc_cta_spec",
    # n=4 scaling demonstrator (NOT part of the 11-system benchmark matrix).
    "levant4d": "create_levant4d_spec",
    # n=5 scale-boundary probe (opt-in, budgeted runs only).
    "levant5d": "create_levant5d_spec",
    # n=4 best-guess PID-like extension (campaign block E6, C11); claim-hygiene
    # quarantined until Moreno/Sanchez confirm the formulation, so opt-in only.
    "pidsmc4d": "create_pidsmc4d_spec",
}
# The benchmark matrix stays the original 11 systems; levant4d/5d/pidsmc4d are opt-in.
ALL = [k for k in SPEC_FACTORIES if k not in ("levant4d", "levant5d", "pidsmc4d")]


def gain_parameters(name: str) -> list[str]:
    """The spec factory's gain arguments, in declaration order.

    Every factory in SPEC_FACTORIES takes its gains as leading keyword
    arguments and, optionally, a trailing ``delta0`` disturbance bound.
    ``delta0`` is a campaign axis with its own drivers (robust_delta_sweep.py),
    not a gain, so it is excluded here and cannot be set through ``--gains``.
    """
    factory = getattr(ud, SPEC_FACTORIES[name])
    return [p for p in inspect.signature(factory).parameters if p != "delta0"]


def parse_gains(name: str, spec_text: str) -> dict:
    """Map a comma-separated gain list onto ``name``'s factory arguments.

    Refuses a wrong arity loudly. Silently zipping a short list would run a
    DIFFERENT system than the caller asked for while labelling the record with
    the caller's name, which is exactly the kind of mislabelled record the
    campaign's provenance rules exist to prevent.
    """
    params = gain_parameters(name)
    if not params:
        raise SystemExit(
            f"--gains: {name}'s factory {SPEC_FACTORIES[name]} takes no gain "
            f"arguments")
    values = [v.strip() for v in spec_text.split(",") if v.strip()]
    if len(values) != len(params):
        raise SystemExit(
            f"--gains: {name} takes {len(params)} value(s) "
            f"({', '.join(params)}), got {len(values)}: {spec_text!r}")
    out = {}
    for p, v in zip(params, values):
        try:
            out[p] = float(v)
        except ValueError:
            raise SystemExit(f"--gains: {v!r} is not a number") from None
    return out


def gains_tag(gains: dict) -> str:
    """Short deterministic tag for a gain point, for the checkpoint filename.

    A run with overridden gains must not overwrite (or resume from) the
    published-gain checkpoint of the same system: the two are different
    experiments and a shared filename would let one be read as the other.
    """
    payload = ",".join(f"{k}={float(v)!r}" for k, v in sorted(gains.items()))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def run_one(name: str, solver: str, verbose: bool, time_budget: float,
            max_extra_degree: int, mu_filter: bool = True,
            gate_solver: str = None, design_fallback: bool = True,
            gate_fallback: bool = True,
            parity_reduction: bool = False,
            gains: dict = None,
            gram_bundle_dir=None) -> dict:
    # ``gains`` overrides the factory's published gain point (campaign block
    # E6 ran candidates B and C this way, previously through an out-of-repo
    # copy of this function; see scripts/campaign/e6/). Without it the factory
    # is called with no arguments exactly as before.
    factory = getattr(ud, SPEC_FACTORIES[name])
    spec = factory(**gains) if gains else factory()
    # ``discover_gf`` builds the JointSOSOptimizer itself, so the design-time
    # parity reduction is selected through the process-wide default. Restored
    # afterwards so this stays a per-call opt-in, never a sticky global.
    previous_parity = set_parity_reduction_default(parity_reduction)
    try:
        r = discover_gf(
            spec, solver=solver, time_budget=time_budget,
            max_extra_degree=max_extra_degree,
            mu_filter=mu_filter, verbose=verbose,
            gate_solver=gate_solver, design_fallback=design_fallback,
            gate_fallback=gate_fallback,
        )
    finally:
        set_parity_reduction_default(previous_parity)
    d = {
        "system": name,
        "weights_r": list(spec.weights_r),
        "degree_m_published": spec.degree_m,
        "maxr": max(spec.weights_r),
        "degree_used": r.degree_used,
        "degrees_tried": list(r.degrees_tried),
        "certified": r.certified,
        "path": r.path,
        "method": r.method,
        "claim_level": r.claim_level,
        "regularity": r.regularity,
        "surface_scope": r.surface_scope,
        "n_terms": r.n_terms,
        # "cheap" evidence: μ ∝ r is kept minimal (μ-incompatible terms dropped),
        # so the SOS solves over a small library.
        "mu": list(r.mu),
        "c1_library": {"diagonal": r.c1_library_diag, "cross": r.c1_library_cross,
                       "cross_full": r.c1_library_cross_full,
                       "total": r.c1_library_diag + r.c1_library_cross},
        "sdp_solves": r.sdp_solves,
        "total_time": round(r.total_time, 2),
        "message": r.message,
        # Per-degree attempt log — distinguishes an honest SOS/gate negative from
        # an unexpected solver crash at that degree (see gf_discovery._design_full_library).
        "degree_attempts": r.degree_attempts,
        # Numerical-soundness evidence from the strict gate (worst octant).
        "gram_min_eig": r.gram_min_eig,
        "gram_residual_l1": r.gram_residual_l1,
        "epsilon_used": r.epsilon_used,
    }
    # Only present when the flag was passed, so records from runs without it
    # are byte-for-byte what they always were. The gate that produced
    # ``certified`` above ran UNREDUCED regardless.
    if parity_reduction:
        d["parity_reduction_requested"] = True
    # Only present when --gains was passed, so records from published-gain runs
    # stay byte-for-byte what they always were.
    if gains:
        d["gains"] = dict(gains)
    # DESIGN Gram bundle, next to the row: the witness stage X re-proves on.
    # Written only for a certified row, and only when the bundle belongs to
    # the certified V (gf_discovery._adopt_design_bundle); otherwise the
    # reason is recorded and stage X falls back to its re-solve witness.
    if gram_bundle_dir is not None and r.certified:
        d["gram_bundle"] = write_gram_bundle(
            gram_bundle_dir, name, r, r.degree_used, solver,
            gate_solver or solver,
            artifact_name=f"{name}_gains_{gains_tag(gains)}" if gains else name)
    if r.V_gf is not None:
        d["V_repr"] = str(r.V_gf)
        if r.coefficients:
            d["coefficients"] = {k: float(v) for k, v in r.coefficients.items()}
        # Full term structure → independently reconstructable & re-verifiable.
        d["V_terms"] = [
            {"coefficient": float(t.coefficient),
             "exponents": [str(e) for e in t.exponents],
             "term_types": list(t.term_types)}
            for t in r.V_gf.terms
        ]
    return d


def run_configuration(args, controller, gains=None):
    """Bind resume to options, relevant environment, library versions and source bytes."""
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    files = sorted((root / "src" / "gf_framework").rglob("*.py"))
    files += [Path(__file__).resolve(), root / "scripts/_watchdog.py",
              root / "scripts/_artifact_io.py"]
    for path in files:
        source.update(path.relative_to(root).as_posix().encode("utf-8"))
        source.update(path.read_bytes())
    config = {
        "schema": "gf.discovery_run/1", "system": controller, "gains": gains,
        "solver": args.solver, "gate_solver": args.gate_solver or args.solver,
        "design_fallback": args.design_fallback, "gate_fallback": args.design_fallback,
        "mu_filter": args.mu_filter, "parity_reduction": args.parity_reduction,
        "max_extra_degree": args.max_extra_degree, "time_budget": args.time_budget,
        "per_system_timeout": args.per_system_timeout,
        "gram_bundle_dir": (str(Path(args.gram_bundle_dir).resolve())
                            if args.gram_bundle_dir else None),
        "source_sha256": source.hexdigest(), "versions": solver_versions(),
        "environment": {key: os.environ.get(key) for key in
                        ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS")},
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return config, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reusable_checkpoint(path, controller, fingerprint):
    """A corrupt, failed, legacy or differently configured row is retryable work."""
    try:
        raw = Path(path).read_bytes()
        row = json.loads(raw)
        required = {"system", "certified", "path", "claim_level", "total_time",
                    "message", "degrees_tried", "degree_attempts"}
        if not isinstance(row, dict) or not required.issubset(row):
            return None
        if (row["system"] != controller or type(row["certified"]) is not bool or
                row.get("run_fingerprint") != fingerprint or
                row.get("completion_status") != "complete" or
                row["claim_level"] == "worker_failed"):
            return None
        if row["certified"] and (not isinstance(row.get("V_terms"), list) or not row["V_terms"]):
            return None
        bundle = row.get("gram_bundle", {})
        if not isinstance(bundle, dict):
            return None
        if bundle.get("path") and (not Path(bundle["path"]).is_file() or
                file_sha256(bundle["path"]) !=
                bundle.get("file_sha256")):
            return None
        row["resume_provenance"] = {"checkpoint": str(Path(path).resolve()),
                                    "sha256": hashlib.sha256(raw).hexdigest()}
        return row
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _spawn_worker(c: str, args) -> dict:
    """Every public invocation uses an isolated, deadline-bounded process tree."""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
           "--worker", "--controllers", c, "--solver", args.solver,
           "--output-dir", args.output_dir, "--time-budget", str(args.time_budget),
           "--max-extra-degree", str(args.max_extra_degree),
           "--per-system-timeout", str(args.per_system_timeout), "--no-resume"]
    if args.gate_solver:
        cmd += ["--gate-solver", args.gate_solver]
    if args.gains is not None:
        cmd += ["--gains", args.gains]
    if not args.design_fallback:
        cmd.append("--no-design-fallback")
    if not args.mu_filter:
        cmd.append("--no-mu-filter")
    if args.parity_reduction:
        cmd.append("--parity-reduction")
    if args.gram_bundle_dir:
        cmd += ["--gram-bundle-dir", str(args.gram_bundle_dir)]
    else:
        cmd.append("--no-gram-bundle")
    if args.quiet:
        cmd.append("--quiet")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return run_with_watchdog(cmd, limit_gb=None, interval=0.2,
                             timeout_s=args.per_system_timeout, env=env)


def _placeholder_row(c: str, reason: str, gains=None) -> dict:
    spec = getattr(ud, SPEC_FACTORIES[c])(**(gains or {}))
    return {
        "system": c, "weights_r": list(spec.weights_r),
        "degree_m_published": spec.degree_m, "maxr": max(spec.weights_r),
        "degree_used": 0, "degrees_tried": [], "certified": False,
        "path": "none", "method": "none", "claim_level": "worker_failed",
        "regularity": "", "surface_scope": "", "n_terms": 0,
        "c1_library": {"diagonal": 0, "cross": 0, "total": 0},
        "sdp_solves": 0, "total_time": 0.0,
        "message": f"No result: {reason}.",
        "degree_attempts": [],
    }


def _print_bundle_line(d: dict) -> None:
    """One line per row saying whether stage X has a design witness for it."""
    b = d.get("gram_bundle")
    if not b:
        return
    if b.get("path"):
        print(f"  design Gram bundle: {b['n_blocks']} blocks, eps "
              f"{b['epsilon']:g}, fingerprint {b['fingerprint'][:12]} -> "
              f"{b['path']}", flush=True)
    else:
        print(f"  design Gram bundle: NOT persisted ({b.get('reason')})",
              flush=True)


def _bundle_index(rows: list[dict], bundle_dir) -> dict:
    """``<bundle-dir>/design_grams.json``: what stage X will find, per system.

    A directory of ``.npz`` files says nothing about which design point wrote
    it. This index carries the per-system fingerprint, the design margin, the
    library order and the solver versions, so a bundle directory can be
    audited without loading a single Gram.
    """
    systems = []
    for r in rows:
        b = r.get("gram_bundle")
        if not b:
            continue
        systems.append({
            "system": r["system"], "degree_m": r.get("degree_used"),
            "bundle": b.get("path"), "n_gram_blocks": b.get("n_blocks", 0),
            "gram_bundle_fingerprint": b.get("fingerprint"),
            "epsilon": b.get("epsilon"),
            "library_order": b.get("library_order"),
            "reason": b.get("reason", ""),
        })
    versions = next((r["gram_bundle"].get("versions") for r in rows
                     if r.get("gram_bundle")), {})
    return {"schema": "gf.design_gram_index/1",
            "produced_by": "scripts/discover_gf.py",
            "bundle_dir": str(bundle_dir),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "versions": versions, "systems": systems}


def _path_label(d: dict) -> str:
    if d.get("path") == "gf_sos":
        via = {"gf_sos_full_library": "SOS"}.get(d.get("method", ""), "?")
        return f"global·{via}"
    return "—"


def table(rows: list[dict]) -> str:
    lines = [
        "| Controller | Dim | weights | m_used | terms | C¹ lib | Path | claim_level | Time (s) |",
        "|------------|-----|---------|--------|-------|--------|------|-------------|----------|",
    ]
    for r in rows:
        dim = f"{len(r['weights_r'])}D"
        m_used = r["degree_used"] if r["degree_used"] else "-"
        nt = r["n_terms"] if r["n_terms"] else "-"
        lib = r["c1_library"]["total"] if r.get("c1_library") else "-"
        lines.append(
            f"| {r['system']:<18} | {dim} | {str(tuple(r['weights_r'])):<9} | "
            f"{m_used} | {nt} | {lib} | {_path_label(r):<12} | {r['claim_level']:<34} | "
            f"{r['total_time']:>8.1f} |"
        )
    n_global = sum(1 for r in rows if r.get("path") == "gf_sos")
    n_uncert = len(rows) - n_global
    lines += ["",
              f"**Global C¹ GF certificates (Lemma 5, standard SOS): {n_global}/{len(rows)}**",
              f"**Not certified (honest negatives): {n_uncert}/{len(rows)}**"]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["all"])
    ap.add_argument("--solver", default="MOSEK_DIRECT",
                    help="DESIGN solver (default MOSEK_DIRECT, the Task-API "
                         "backend; flipped from MOSEK by the P2-06 validation "
                         "gate under decision D1, which measured identical "
                         "verdict/degree/term count on 12/12 rows at 14.6x "
                         "the speed)")
    ap.add_argument("--gate-solver", default=None,
                    help="solver for the strict gate (defaults to --solver, "
                         "except MOSEK_DIRECT designs default the gate to "
                         "MOSEK, which is the same Task-API encoding under the "
                         "name verify_gf_sos_full knows)")
    ap.add_argument("--no-design-fallback", dest="design_fallback",
                    action="store_false", default=True,
                    help="pin BOTH solver stages (campaign rule 6): the DESIGN "
                         "solve (no CVXPY fallback chain on the joint V+W "
                         "solve; required for MOSEK_DIRECT at large n, where a "
                         "fallback re-enters the canonicalization memory wall) "
                         "AND the strict re-certification gate "
                         "(strict_solver=True in certify_discovered_gf). "
                         "The former C19(a) gap (gate not pinned) is closed "
                         "since 2026-08-02")
    ap.add_argument("--output-dir", default="artifacts/publication/results_gf_sos/")
    ap.add_argument("--time-budget", type=float, default=1800.0)
    ap.add_argument("--max-extra-degree", type=int, default=4)
    ap.add_argument("--no-mu-filter", dest="mu_filter", action="store_false",
                    default=True,
                    help="completeness experiment: keep μ-inflating cross terms in the "
                         "library (expect much larger Grams; measures the cost the "
                         "μ-compatible filter avoids)")
    ap.add_argument("--parity-reduction", action="store_true", default=False,
                    help="S1: pose the DESIGN SDP over the even-parity "
                         "sub-library and one Gram block per orbit of x -> -x "
                         "(2^(n-1) instead of 2^n blocks per stage). "
                         "Prerequisites are verified symbolically per system; "
                         "on failure the reduction refuses loudly (stderr) and "
                         "the full formulation runs. The strict "
                         "re-certification gate is NOT reduced — that is the "
                         "S1 validation gate. Off by default; without it every "
                         "code path is unchanged")
    ap.add_argument("--gains", default=None,
                    help="comma-separated gain point overriding the system's "
                         "published gains, e.g. "
                         "--gains 3,4.160167646103808,3.0594733539832584,1.1 "
                         "for pidsmc4d. The values are mapped in order onto "
                         "the spec factory's gain arguments (delta0 excluded; "
                         "use the robust drivers for that axis), and a wrong "
                         "count is refused. Requires exactly one controller. "
                         "The checkpoint is written to "
                         "<system>_gains_<tag>.json so it can never be "
                         "confused with the published-gain record")
    ap.add_argument("--gram-bundle-dir", default=None,
                    help="directory for the DESIGN Gram bundles (default: a "
                         "design_grams/ sibling of the row checkpoints inside "
                         "--output-dir). One <system>.npz or <system>.json per "
                         "certified row, schema gf.design_grams/1, consumed "
                         "directly by review/reverify_exact.py "
                         "--gram-bundle-dir. It is the design SDP's own Gram, "
                         "not a re-solve, so stage X needs --epsilon equal to "
                         "the design margin the bundle records")
    ap.add_argument("--no-gram-bundle", dest="gram_bundle",
                    action="store_false", default=True,
                    help="do not persist the design Gram bundles (stage X "
                         "then proposes its own witness by re-solving)")
    ap.add_argument("--per-system-timeout", type=float, default=None,
                    help="hard wall-clock cap per isolated worker subprocess "
                         "(default: --time-budget + 600s headroom, so the "
                         "worker's own time budget always expires first; an "
                         "explicit value here is authoritative)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute all systems even if checkpoints exist")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--result-path", help="Write a machine-readable result receipt here.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.design_fallback:
        # Campaign rule 6: one flag pins both stages. Printed at runtime so
        # every campaign log carries the provenance statement.
        print("NOTE: --no-design-fallback pins BOTH the design solve and the "
              "strict re-certification gate (strict_solver=True); no solver "
              "fallback can occur at either stage.",
              file=sys.stderr, flush=True)

    if args.per_system_timeout is None:
        args.per_system_timeout = args.time_budget + 600.0

    for label in ("time_budget", "per_system_timeout"):
        if not math.isfinite(getattr(args, label)) or getattr(args, label) <= 0:
            ap.error(f"--{label.replace(chr(95), chr(45))} must be positive and finite")
    if args.max_extra_degree < 0:
        ap.error("--max-extra-degree must be nonnegative")

    if "all" in args.controllers:
        extra = [c for c in args.controllers if c != "all"]
        controllers = list(ALL) + extra
    else:
        controllers = list(args.controllers)
    # Dedupe while preserving first-seen order (e.g. `--controllers all levant4d`
    # must not run levant4d twice if it were ever added to ALL).
    seen = set()
    controllers = [c for c in controllers if not (c in seen or seen.add(c))]
    unknown = [c for c in controllers if c not in SPEC_FACTORIES]
    if unknown:
        valid = ", ".join(sorted(SPEC_FACTORIES))
        raise SystemExit(f"unknown controller(s): {unknown} (valid: {valid})")

    # --gains names ONE gain point, so it can only address one system. Applying
    # it to a sweep would either mean the same numbers for systems of different
    # dimension (nonsense) or a silent per-system reinterpretation (worse).
    gains = None
    if args.gains is not None:
        if len(controllers) != 1:
            raise SystemExit(
                f"--gains requires exactly one controller, got {controllers}")
        gains = parse_gains(controllers[0], args.gains)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "systems"
    ckpt_dir.mkdir(exist_ok=True)

    # Default the bundle directory to a design_grams/ sibling of the row
    # checkpoints, so one --output-dir gives stage X everything it needs.
    if not args.gram_bundle:
        args.gram_bundle_dir = None
    elif args.gram_bundle_dir is None:
        args.gram_bundle_dir = str(out_dir / "design_grams")

    # Public single-system and matrix calls share the same parent orchestration.
    if args.worker:
        if len(controllers) != 1:
            ap.error("internal --worker requires exactly one controller")
        c = controllers[0]
        stem = f"{c}_gains_{gains_tag(gains)}" if gains else c
        config, fingerprint = run_configuration(args, c, gains)
        d = run_one(c, args.solver, verbose=not args.quiet,
                    time_budget=args.time_budget, max_extra_degree=args.max_extra_degree,
                    mu_filter=args.mu_filter, gate_solver=args.gate_solver,
                    design_fallback=args.design_fallback,
                    gate_fallback=args.design_fallback,
                    parity_reduction=args.parity_reduction, gains=gains,
                    gram_bundle_dir=args.gram_bundle_dir)
        d.update(run_configuration=config, run_fingerprint=fingerprint,
                 completion_status="complete")
        bundle = d.get("gram_bundle", {})
        if bundle.get("path"):
            bundle["file_sha256"] = file_sha256(bundle["path"])
        atomic_json(ckpt_dir / f"{stem}.json", d)
        return

    rows = []
    t0 = time.time()
    for i, c in enumerate(controllers, 1):
        stem = f"{c}_gains_{gains_tag(gains)}" if gains else c
        ckpt = ckpt_dir / f"{stem}.json"
        config, fingerprint = run_configuration(args, c, gains)
        d = reusable_checkpoint(ckpt, c, fingerprint) if args.resume else None
        if d is not None:
            print(f"[{i}/{len(controllers)}] {c} ... RESUMED validated checkpoint", flush=True)
            rows.append(d)
            continue
        print(f"[{i}/{len(controllers)}] {c} ... (isolated worker, "
              f"timeout {args.per_system_timeout:g}s)", flush=True)
        if ckpt.exists():
            # Retain the rejected checkpoint as evidence, outside the active name.
            ckpt.replace(ckpt.with_name(ckpt.stem + ".previous_" +
                         datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"))
        try:
            worker = _spawn_worker(c, args)
        except (OSError, RuntimeError) as exc:
            worker = {"status": "unexpected_error", "exit_code": None,
                      "detail": f"{type(exc).__name__}: {exc}"}
        d = (reusable_checkpoint(ckpt, c, fingerprint)
             if worker.get("exit_code") == 0 else None)
        if d is None:
            d = _placeholder_row(c, worker.get("status") or "missing or invalid worker output", gains)
            d.update(run_configuration=config, run_fingerprint=fingerprint,
                     completion_status="unfinished", worker=worker)
            if gains:
                d["gains"] = gains
            atomic_json(ckpt, d)
        else:
            d.pop("resume_provenance", None)
            d["worker"] = worker
            atomic_json(ckpt, d)
        rows.append(d)
        print(f"  -> {_path_label(d)} | {d['claim_level']} ({d['total_time']:.1f}s) "
              f"| {d['message']}", flush=True)
        _print_bundle_line(d)

    ts = datetime.now(timezone.utc).isoformat()
    summary = {"timestamp": ts, "method": ("gf_sos_mu_filtered_lemma5" if args.mu_filter else "gf_sos_full_library_lemma5"),
               "requested_systems": controllers,
               "solver": args.solver, "total_time_s": round(time.time() - t0, 2),
               "results": rows}
    if args.parity_reduction:
        summary["parity_reduction"] = True
    if args.gram_bundle_dir:
        index = _bundle_index(rows, args.gram_bundle_dir)
        summary["gram_bundle_dir"] = str(args.gram_bundle_dir)
        n_bundles = sum(1 for s in index["systems"] if s["bundle"])
        bdir = Path(args.gram_bundle_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        atomic_json(bdir / "design_grams.json", index)
        print(f"  design Gram bundles: {n_bundles}/"
              f"{sum(1 for r in rows if r.get('certified'))} certified rows "
              f"-> {bdir}", flush=True)
    jpath = out_dir / f"gf_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    atomic_json(jpath, summary)
    library_label = "μ-filtered full library" if args.mu_filter else "full library without μ filter"
    md = (f"# GF-SOS Lyapunov discovery ({library_label}, Lemma 5)\n\n"
          f"- Timestamp: {ts}\n- Solver (SOS oracle): {args.solver}\n\n"
          f"## Results\n\n{table(rows)}\n")
    mpath = jpath.with_suffix(".md")
    atomic_text(mpath, md)
    if args.result_path:
        result_receipt(args.result_path, jpath, controllers)

    n_global = sum(1 for r in rows if r.get("path") == "gf_sos")
    print(f"\n{'='*60}\nGF-SOS DISCOVERY COMPLETE: {time.time()-t0:.1f}s")
    print(f"  Global C¹ GF (Lemma 5, standard SOS): {n_global}/{len(rows)}")
    n_unfinished = sum(r.get("completion_status") != "complete" for r in rows)
    print(f"  Completed without certificate: {len(rows) - n_global - n_unfinished}/{len(rows)}")
    print(f"  Unfinished worker failures: {n_unfinished}/{len(rows)}")
    print(f"  JSON: {jpath}\n  MD:   {mpath}\n{'='*60}")
    print("\n" + table(rows))
    if n_unfinished:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
