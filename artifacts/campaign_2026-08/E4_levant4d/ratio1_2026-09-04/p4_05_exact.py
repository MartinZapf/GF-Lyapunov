#!/usr/bin/env python3
"""P4-05: EXACT rational re-proof (X) of one levant4d ratio-1 cell, on a
PERSISTED design Gram bundle, over the printed-gain bracket corners.

Copied from the freshest resumable exact-run driver
(``artifacts/campaign_2026-08/E5_levant5d/exact_2026-09-04/p4_03_exact.py``)
and extended by the two things the levant4d ratio-1 cells need:

  * a PERTURBED instance (delta0 > 0, two inclusion endpoints), and
  * the exact covariant representative ``p``: the exact rational gain corners
    of the printed instance are mapped by ``lambda_i -> p^i lambda_i`` and
    ``delta0 -> p^4 delta0``, all in Q, never ``Fraction(float(x))``.

Per corner:

  * the exact gains of that bracket corner, scaled by ``p^i`` in Q, are
    injected as ``Fraction`` objects into the system factory,
  * ``run_exact_phase(..., gram_bundle=<npz>, expect_bundle_fingerprint=...)``
    is called, so NO float solve runs on the certifying route,
  * the result is written to ``corner_<k>.json`` IMMEDIATELY (atomic rename).

With ``--no-bundle`` the same driver runs the FIXED-V exact check: no design
witness is supplied, so ``run_exact_phase`` proposes its own witness through
the three-source ladder (P2-05/P2-10: MOSEK primary, one CLARABEL retry) and
the record says which source was adopted. That mode is how a TRANSFERRED
candidate, which has no bundle of its own, reaches X.

RESUMABILITY: re-running the same command skips every corner whose
``corner_<k>.json`` already exists.

Claim hygiene (D6). An exact PASS at every corner is a certificate for the
gain BRACKET BOX at the margin ``eps_certified``, at this degree, for this
system, at THIS delta0 and these gains, and for nothing else. A FAIL is typed
by its reason and its witness path and is NOT a statement that the system
admits no certificate. ``solver_unknown`` decides nothing.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


# --------------------------------------------------------------------- utils

def _peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    v = max(ru, ch)
    return v / (1024.0 ** 2) if sys.platform.startswith("linux") else v / (1024.0 ** 3)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
    except Exception:
        return "unknown"


def _atomic_write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- V reconstruction

def _reconstruct_V(rec: dict, m: int, weights):
    """Rebuild a candidate V as a GeneralizedForm from a design record."""
    from gf_framework.algebra.core import GeneralizedForm, GFTerm

    if "candidate_V_terms" in rec:
        raw = rec["candidate_V_terms"]
        degree_m, w = m, tuple(int(r) for r in weights)
        src = "candidate_V_terms"
    elif isinstance(rec.get("V"), dict):
        V = rec["V"]
        raw = V["terms"]
        degree_m = int(V["degree_m"])
        w = tuple(int(r) for r in V["weights_r"])
        src = "V (gate design record, hex-float coefficients)"
    else:
        raise SystemExit("record carries neither candidate_V_terms nor V")

    terms = []
    for t in raw:
        c = t.get("coefficient_hex", t["coefficient"])
        coeff = float.fromhex(c) if isinstance(c, str) else float(c)
        terms.append(GFTerm(
            coefficient=coeff,
            exponents=tuple(Fraction(str(e)) for e in t["exponents"]),
            term_types=tuple(t["term_types"]),
        ))
    return GeneralizedForm(degree_m=degree_m, weights_r=w, terms=terms), src, len(terms)


# ------------------------------------------------------------------- corners

def _enumerate_corners(p: Fraction, digits: int, key: str = "levant4d"):
    """The printed-gain bracket corner box, mapped to the representative ``p``.

    The printed levant4d point has two irrational gains (lambda2, lambda3), so
    the box is 2**2 = 4 corners, ordered ``sorted(bracketed_names)`` x
    ``itertools.product((lo, hi), ...)`` with corner 0 all-``lo``.

    The covariant map is applied IN Q: ``lambda_i -> p^i lambda_i``. Bracket
    validity is re-verified in Q at the SCALED point by power comparison
    (``(lo p^i)^k < target p^{ik} < (hi p^i)^k``), never through a float.
    """
    from gf_framework.systems.printed_gains import brackets, gain_point

    # WP X10: the base point is FLAGSHIP_GAIN_POINTS[key]. For the default key
    # the flagship entry holds the benchmark entry's own gains object, so the
    # corners are the original ones exactly.
    point = gain_point(key, flagship=True)
    if point.factory != "create_levant4d_spec":
        raise SystemExit(f"gain point {key!r} is not a levant4d-family point")
    bs = brackets(point, digits)
    names = sorted(bs)                       # ('lambda2', 'lambda3')
    idx = {"lambda1": 1, "lambda2": 2, "lambda3": 3, "lambda4": 4}
    scale = {nm: p ** i for nm, i in idx.items()}

    rational = {n: g.value * scale[n]
                for n, g in point.gains.items() if not g.is_bracketed}
    combos = list(itertools.product(*(bs[n] for n in names)))
    out = []
    for i, combo in enumerate(combos):
        assignment = {n: v * scale[n] for n, v in zip(names, combo)}
        label = ",".join("lo" if assignment[n] == bs[n][0] * scale[n] else "hi"
                         for n in names)
        gains = {**rational, **assignment}
        for k, v in gains.items():
            if not isinstance(v, Fraction):
                raise SystemExit(f"gain {k} is {type(v).__name__}, not Fraction")
        out.append({"index": i, "label": label, "names": names, "gains": gains})

    checks = {}
    for n in names:
        lo, hi = bs[n]
        g = point.gains[n]
        k = g.power_k
        s = scale[n]
        tgt = g.power_target * s ** k
        ok = (lo * s) ** k < tgt < (hi * s) ** k
        checks[n] = {"power_k": int(k), "power_target_scaled": str(tgt),
                     "lo_scaled": str(lo * s), "hi_scaled": str(hi * s),
                     "valid_in_Q": bool(ok)}
        if not ok:
            raise SystemExit(f"scaled bracket for {n} is not valid in Q")
    return out, names, checks


# ------------------------------------------------------------------ per corner

def _octant_rows(cap) -> list:
    rows = []
    for c in cap.octants:
        rows.append({
            "signs": list(c.signs),
            "ok": bool(c.ok),
            "reason": c.reason,
            "degree": int(c.degree),
            "n_basis": int(c.n_basis),
            "eta_hat": c.eta_hat,
            "eta_hat_float": c.eta_hat_float,
            "resid_l1_float": c.resid_l1_float,
            "charge_float": c.charge_float,
            "budget_float": c.budget_float,
            "charge_ratio": (c.charge_float / c.budget_float
                             if c.budget_float else None),
            "min_eig_float": c.min_eig_float,
            "eps_certified": c.eps_certified,
            "eps_certified_float": c.eps_certified_float,
            "margin_policy": c.margin_policy,
            "r1": c.r1_provenance(),
        })
    return rows


def _capture_summary(cap) -> dict:
    return {
        "ok": bool(cap.ok),
        "reason": cap.reason,
        "n_octants": len(cap.octants),
        "n_octants_expected": int(cap.n_octants_expected),
        "mu": [int(v) for v in cap.mu],
        "eps": cap.eps,
        "margin_policy": cap.margin_policy,
        "worst_charge_ratio": cap.worst_charge_ratio,
        "eps_certified_float": cap.eps_certified_float,
        "first_failing_octant": next(
            (i for i, c in enumerate(cap.octants) if not c.ok), None),
        "octants": _octant_rows(cap),
    }


def _typed_status(phase) -> str:
    """certified | gate_rejected | solver_unknown (P3-02 vocabulary).

    ``timeout`` / ``oom`` are decided OUTSIDE this process by the watchdog and
    the shell, never here.
    """
    if phase.exact_pass:
        return "certified"
    resolves = ([phase.v_resolve_feasible]
                + [e.resolve_feasible for e in phase.endpoints])
    if not all(resolves):
        return "solver_unknown"
    return "gate_rejected"


# ---------------------------------------------------------------------- README

def _write_status(out_dir: Path, header: dict, rows: list) -> None:
    _atomic_write_json(out_dir / "status.json",
                       {"header": header, "corners": rows})
    lines = [
        "| corner | label | verdict | typed status | worst charge/budget | "
        "eps_certified | witness source | witness solver | attempts | "
        "V reused | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wcr = r.get("worst_charge_ratio")
        ec = r.get("eps_certified")
        f_wcr = f"{wcr:.4e}" if isinstance(wcr, float) else str(wcr)
        f_ec = f"{ec:.4e}" if isinstance(ec, float) else str(ec)
        lines.append(
            f"| {r['index']} | {r['label']} | "
            f"{'**exact-PASS**' if r.get('exact_pass') else '**FAIL**'} | "
            f"`{r.get('typed_status', '')}` | {f_wcr} | {f_ec} | "
            f"`{r.get('witness_source', '')}` | {r.get('witness_solver', '')} | "
            f"{r.get('n_witness_attempts', '')} | {r.get('v_reused')} | "
            f"{r.get('wall_s', 0):.1f} |")
    (out_dir / "STATUS.md").write_text(
        "# P4-05 running status (rewritten after every corner)\n\n"
        f"- {header.get('system', 'levant4d')}, m = {header['m']}, representative p = "
        f"`{header['qscale_p']}`, delta0 = `{header['delta0_exact']}` "
        f"(ratio delta0/L = 1)\n"
        f"- epsilon = {header['epsilon']}, jobs = {header['jobs']}, "
        f"margin_policy = `{header['margin_policy']}`\n"
        f"- witness route: {header['witness_route']}\n"
        f"- bundle fingerprint `{header['expect_fingerprint']}`\n"
        f"- pinned git sha `{header['pinned_git_sha']}`\n"
        f"- last update {datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--p", default="1",
                    help="rational covariant representative (p=1 = printed)")
    ap.add_argument("--v-record", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--no-bundle", action="store_true",
                    help="FIXED-V exact check: no design witness, the "
                         "three-source witness ladder proposes one")
    ap.add_argument("--expect-fingerprint", default=None)
    ap.add_argument("--epsilon", type=float, required=True,
                    help="MUST be the DESIGN margin when a bundle is supplied")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-corners", default="")
    ap.add_argument("--digits", type=int, default=15)
    ap.add_argument("--margin-policy", default="half", choices=["half", "achieved"])
    ap.add_argument("--vproof-cache-dir", default=None)
    ap.add_argument("--gain-point", default="levant4d",
                    help="base point in FLAGSHIP_GAIN_POINTS (a levant4d-family "
                         "key, WP X10); must be the one the design ran at")
    ap.add_argument("--fallback-solver", default="CLARABEL",
                    help="witness retry solver after the bundle/MOSEK attempts; "
                         "'' disables the retry (WP X10; an adopted witness is "
                         "always an exact PASS, so this never weakens a proof)")
    args = ap.parse_args()

    p = Fraction(args.p)
    delta0 = p ** 4
    t_all = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import gf_framework
    from gf_framework.algebra.enumerate import LIBRARY_ORDER
    from gf_framework.discovery import universal_discovery as ud
    from gf_framework.verification.exact_gate import exact_terms_from_gf
    from gf_framework.verification.exact_phase import (
        DiskVProofCache, build_v_cache, run_exact_phase, v_input_hash,
        verifier_code_version)

    src = Path(gf_framework.__file__).resolve().parents[2]
    pinned_repo = src.parent
    _log(f"gf_framework  {gf_framework.__file__}")
    _log(f"pinned repo   {pinned_repo}  sha {_git_sha(pinned_repo)}")

    rec = json.loads(Path(args.v_record).read_text())
    spec0 = ud.create_levant4d_spec()
    weights = tuple(int(r) for r in spec0.weights_r)
    n_vars = len(weights)
    V, v_src, n_v_terms = _reconstruct_V(rec, args.m, weights)
    _log(f"V rebuilt from {v_src}: {n_v_terms} terms, degree_m={V.degree_m}, "
         f"weights={weights}")

    corners, names, bracket_checks = _enumerate_corners(p, args.digits,
                                                         args.gain_point)
    design_system = rec.get("system")
    if design_system is not None and design_system != args.gain_point:
        raise SystemExit(f"--gain-point {args.gain_point!r} contradicts the "
                         f"design record's system {design_system!r}")
    _log(f"corner box: {len(corners)} corners over {names}, p = {p}, "
         f"delta0 = {delta0}")

    want = ([int(x) for x in args.only_corners.split(",") if x.strip() != ""]
            if args.only_corners else [c["index"] for c in corners])

    use_bundle = (not args.no_bundle) and bool(args.bundle)
    witness_route = ("design_bundle (D1: the citable X witness)" if use_bundle
                     else "fixed-V, three-source witness ladder "
                          "(no design bundle exists for this V)")

    cache_root = Path(args.vproof_cache_dir or (out_dir / "vproof_cache"))
    disk = DiskVProofCache(
        cache_root,
        library=f"{args.gain_point}@m{args.m}@p{p}@{args.expect_fingerprint or 'nofp'}",
        extra={"binding": "printed_rationals_scaled", "digits": args.digits,
               "epsilon": repr(args.epsilon), "parity": "none",
               "witness": "design_bundle" if use_bundle else "resolve",
               "wp": "P4-05"},
    )
    v_terms = exact_terms_from_gf(V)
    vih = v_input_hash(V, args.epsilon)

    header = {
        "wp": "P4-05",
        "driver": str(Path(__file__).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "system": args.gain_point, "gain_point": args.gain_point,
        "dynamics_factory": "create_levant4d_spec",
        "m": args.m, "n_vars": n_vars,
        "weights_r": list(weights),
        "qscale_p": str(p),
        "delta0_exact": str(delta0), "delta0_float": float(delta0),
        "ratio_delta0_over_L": "1",
        "epsilon": args.epsilon, "jobs": args.jobs,
        "fallback_solver": args.fallback_solver or None,
        "margin_policy": args.margin_policy,
        "witness_route": witness_route,
        "bundle": str(Path(args.bundle).resolve()) if use_bundle else None,
        "expect_fingerprint": args.expect_fingerprint if use_bundle else None,
        "v_record": str(Path(args.v_record).resolve()),
        "v_source": v_src, "n_v_terms": n_v_terms,
        "v_input_hash": vih,
        "bracket_checks_in_Q": bracket_checks,
        "library_order": LIBRARY_ORDER,
        "pinned_git_sha": _git_sha(pinned_repo),
        "pinned_worktree": str(pinned_repo),
        "gf_framework_file": gf_framework.__file__,
        "verifier_code_version": verifier_code_version(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "command": " ".join([sys.executable] + sys.argv),
        "vproof_cache_root": str(cache_root),
        "claim": (
            "an exact PASS at EVERY corner certifies the gain BRACKET BOX at "
            "eps_certified, at this m, at this delta0 and these gains, and "
            "nothing else; a FAIL is typed by its reason and its witness path "
            "and is NOT a statement that the system admits no certificate"),
    }
    _atomic_write_json(out_dir / "header.json", header)

    rows: list = []
    v_cache = None
    for c in corners:
        pth = out_dir / f"corner_{c['index']}.json"
        if pth.exists():
            try:
                d = json.loads(pth.read_text())
            except ValueError:
                _log(f"corner {c['index']}: unreadable record, will re-run")
                continue
            rows.append(d["summary"])
            _log(f"corner {c['index']}: RESUME, already on disk "
                 f"(exact_pass={d['summary'].get('exact_pass')})")
    done = {r["index"] for r in rows}
    todo = [c for c in corners if c["index"] in want and c["index"] not in done]
    if todo:
        t0 = time.time()
        v_cache, ev = disk.load(V, args.epsilon, n_vars,
                                v_terms=v_terms, input_hash=vih)
        header["vproof_cache_startup"] = ev.as_record()
        _log(f"vproof cache startup: hit={ev.hit} reason={ev.reason} "
             f"{time.time() - t0:.1f}s")
        _atomic_write_json(out_dir / "header.json", header)

    _write_status(out_dir, header, sorted(rows, key=lambda r: r["index"]))

    for c in todo:
        k = c["index"]
        _log(f"=== corner {k} [{c['label']}] : exact phase, jobs={args.jobs}, "
             f"v_cache={'yes' if v_cache is not None else 'no'} ...")
        gains = c["gains"]
        spec = ud.create_levant4d_spec(delta0=delta0, **gains)
        dynamics_list = list(spec.dynamics_gf_vertices or [spec.dynamics_gf])
        kw = {}
        if use_bundle:
            kw["gram_bundle"] = str(Path(args.bundle).resolve())
            kw["expect_bundle_fingerprint"] = args.expect_fingerprint
        t0 = time.time()
        failure = None
        phase = None
        try:
            phase = run_exact_phase(
                V, dynamics_list, n_vars,
                epsilon=args.epsilon,
                v_cache=v_cache,
                v_proof_cache=disk,
                jobs=args.jobs,
                margin_policy=args.margin_policy,
                fallback_solver=args.fallback_solver or None,
                **kw,
            )
        except Exception as exc:   # typed as the corner's OUTCOME, never a bug
            failure = {
                "class": type(exc).__name__,
                "reason": getattr(exc, "reason", ""),
                "detail": str(exc)[:2000],
                "traceback": traceback.format_exc()[-4000:],
            }
        wall = time.time() - t0

        if phase is None:
            summary = {
                "index": k, "label": c["label"],
                "exact_pass": False,
                "typed_status": f"raised:{failure['class']}"
                                + (f":{failure['reason']}" if failure["reason"] else ""),
                "worst_charge_ratio": None, "eps_certified": None,
                "witness_source": "", "witness_solver": "",
                "n_witness_attempts": 0, "v_reused": None,
                "wall_s": round(wall, 1),
            }
            doc = {"header": header, "corner": {
                "index": k, "label": c["label"],
                "exact_gains": {n: str(v) for n, v in sorted(gains.items())},
                "float_image": {n: float(v) for n, v in sorted(gains.items())}},
                "failure": failure, "summary": summary,
                "peak_rss_gb": round(_peak_rss_gb(), 3)}
            _atomic_write_json(out_dir / f"corner_{k}.json", doc)
            rows.append(summary)
            _write_status(out_dir, header, sorted(rows, key=lambda r: r["index"]))
            _log(f"corner {k}: RAISED {summary['typed_status']} after {wall:.1f}s")
            if failure["reason"] in ("gram_bundle_input_mismatch",
                                     "gram_bundle_hash_mismatch",
                                     "gram_bundle_schema_unknown",
                                     "gram_bundle_coverage_incomplete",
                                     "gram_bundle_parity_inconsistent",
                                     "gram_bundle_empty"):
                _log("structural bundle refusal: stopping the remaining corners")
                break
            continue

        summary = {
            "index": k, "label": c["label"],
            "exact_pass": bool(phase.exact_pass),
            "typed_status": _typed_status(phase),
            "worst_charge_ratio": phase.worst_charge_ratio,
            "eps_certified": phase.eps_certified_float,
            "witness_source": phase.witness_source,
            "witness_solver": phase.witness_solver,
            "witness_solver_used": phase.witness_solver_used,
            "n_witness_attempts": len(phase.witness_attempts),
            "clarabel_retry_fired": any(
                str(a.get("solver", "")).upper() == "CLARABEL"
                for a in phase.witness_attempts),
            "v_reused": bool(phase.v_reused),
            "v_fingerprint": phase.v_fingerprint,
            "V_worst_charge_ratio": phase.v_exact.worst_charge_ratio,
            "V_ok": bool(phase.v_exact.ok),
            "V_reason": phase.v_exact.reason,
            "W_worst_charge_ratio": max(
                (e.exact.worst_charge_ratio for e in phase.endpoints
                 if e.exact.worst_charge_ratio is not None), default=None),
            "W_ok": all(bool(e.exact.ok) for e in phase.endpoints),
            "W_reasons": [e.exact.reason for e in phase.endpoints],
            "v_resolve_feasible": bool(phase.v_resolve_feasible),
            "endpoint_resolve_feasible": [bool(e.resolve_feasible)
                                          for e in phase.endpoints],
            "wall_s": round(wall, 1),
            "phase_time_s": phase.time_s,
        }
        doc = {
            "header": header,
            "corner": {
                "index": k, "label": c["label"],
                "bracketed_names_in_corner_order": names,
                "exact_gains": {n: str(v) for n, v in sorted(gains.items())},
                "float_image": {n: float(v) for n, v in sorted(gains.items())},
                "delta0_exact": str(delta0),
            },
            "summary": summary,
            "exact_pass": bool(phase.exact_pass),
            "worst_charge_ratio": phase.worst_charge_ratio,
            "eps_certified_float": phase.eps_certified_float,
            "margin_policy": phase.margin_policy,
            "witness": phase.witness_record(),
            "parallel": dict(phase.parallel),
            "v_proof_cache": dict(phase.v_proof_cache),
            "v_input_hash": phase.v_input_hash,
            "v_reused": bool(phase.v_reused),
            "V": _capture_summary(phase.v_exact),
            "endpoints": [{
                "endpoint": i,
                "resolve_feasible": bool(e.resolve_feasible),
                "lie_crosscheck": e.lie_crosscheck,
                "witness_source": e.witness_source,
                "witness_solver": e.witness_solver,
                "exact": _capture_summary(e.exact),
            } for i, e in enumerate(phase.endpoints)],
            "wall_s": round(wall, 1),
            "peak_rss_gb": round(_peak_rss_gb(), 3),
        }
        _atomic_write_json(out_dir / f"corner_{k}.json", doc)
        rows.append(summary)
        _write_status(out_dir, header, sorted(rows, key=lambda r: r["index"]))
        _log(f"corner {k}: exact_pass={phase.exact_pass} "
             f"typed={summary['typed_status']} "
             f"worst charge/budget={phase.worst_charge_ratio} "
             f"eps_certified={phase.eps_certified_float} "
             f"witness={phase.witness_source}/{phase.witness_solver} "
             f"v_reused={phase.v_reused} {wall:.1f}s "
             f"peakRSS={_peak_rss_gb():.2f}GB")

        if v_cache is None and phase.v_exact.ok and phase.v_grams:
            v_cache = build_v_cache(V, args.epsilon, phase)
            ev = disk.store(V, args.epsilon, n_vars, phase)
            _log(f"V proof frozen for reuse: {v_cache.fingerprint} ; "
                 f"disk store -> {ev.store_reason}")
            header["vproof_cache_store"] = ev.as_record()
            _atomic_write_json(out_dir / "header.json", header)

    rows.sort(key=lambda r: r["index"])
    n_pass = sum(1 for r in rows if r.get("exact_pass"))
    overall = {
        "wp": "P4-05",
        "header": header,
        "n_corners": len(corners),
        "n_corners_recorded": len(rows),
        "n_corners_exact_pass": n_pass,
        "exact_pass_all": bool(len(rows) == len(corners) and n_pass == len(corners)),
        "worst_charge_ratio_over_corners": max(
            (r["worst_charge_ratio"] for r in rows
             if r.get("worst_charge_ratio") is not None), default=None),
        "eps_certified_over_corners": min(
            (r["eps_certified"] for r in rows
             if r.get("eps_certified") is not None), default=None),
        "typed_statuses": sorted({r.get("typed_status", "") for r in rows}),
        "witness_sources": sorted({r.get("witness_source", "") for r in rows}),
        "witness_solvers": sorted({r.get("witness_solver", "") for r in rows}),
        "any_clarabel_retry": any(r.get("clarabel_retry_fired") for r in rows),
        "v_reused_by_corner": {str(r["index"]): r.get("v_reused") for r in rows},
        "corners": rows,
        "total_wall_s": round(time.time() - t_all, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "claim": header["claim"],
    }
    _atomic_write_json(out_dir / "ROW.json", overall)
    _write_status(out_dir, header, rows)
    _log(f"ROW: {n_pass}/{len(corners)} exact-PASS  "
         f"exact_pass_all={overall['exact_pass_all']}  "
         f"worst charge/budget={overall['worst_charge_ratio_over_corners']}  "
         f"eps_certified={overall['eps_certified_over_corners']}  "
         f"total {overall['total_wall_s']}s  peakRSS {overall['peak_rss_gb']}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
