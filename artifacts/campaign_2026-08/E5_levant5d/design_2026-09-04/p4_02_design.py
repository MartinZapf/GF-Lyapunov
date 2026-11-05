#!/usr/bin/env python3
"""P4-02: levant5d nominal re-design on the PACKAGED path (blocked default).

One cell per process. Poses the joint V+W design SDP through
``JointSOSOptimizer.find_lyapunov_with_dynamics`` with:

  * the y-parity Gram block decomposition at its PACKAGED DEFAULT (P5-04),
  * the W epsilon rows (P2-02) and the 2mu odd-y-degree W policy (P2-09),
  * MOSEK_DIRECT only, no CVXPY, no solver fallback,
  * the design Gram bundle captured (P2-01) and persisted as ``.npz`` (P2-03),
  * ``design_screen`` (P2-04) run on that bundle.

The exact SYMBOLIC even-support guard (``algebra.support_guard``) is run in
this driver BEFORE any assembly, in addition to the copy the packaged path
runs internally; a typed ``OddLiftRefusal`` is recorded as the RESULT of the
cell, never as an error.

Claim hygiene (D6): a feasible design is a CANDIDATE, not a certificate.
Nothing here is proved. MOSEK proves nothing. The exact probe at the bottom
is a single-octant spot check on the design bundle, not the X record (P4-03
owns X).

Usage:
  python p4_02_design.py --m 11 --epsilon 1e-4 --out-dir <dir> --tag <tag>
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np


def _peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 ** 2) if sys.platform.startswith("linux") else ru / (1024.0 ** 3)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="levant5d",
                    help="levant5d is the WP cell; other names are SMOKE ONLY "
                         "(the harness is exercised, no E5 claim is made)")
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--design-screen", default="warn",
                    choices=["reject", "warn", "off"])
    ap.add_argument("--no-bundle", action="store_true",
                    help="capture_design_grams=False (escape hatch, P2-01)")
    ap.add_argument("--probe-octant", type=int, default=0,
                    help="V octant index for the single-octant exact probe; "
                         "-1 disables the probe")
    ap.add_argument("--probe-budget-s", type=float, default=600.0)
    ap.add_argument("--mosek-timeout", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()

    src = Path(__import__("gf_framework").__file__).resolve().parents[2]
    pinned_repo = src.parent            # "<worktree>/GF Framework"

    import gf_framework
    import mosek
    from gf_framework.algebra.enumerate import (
        LIBRARY_ORDER, enumerate_diagonal_terms)
    from gf_framework.algebra.support_guard import (
        GuardUnreadable, OddLiftRefusal, check_even_support,
        gf_exponent_support)
    from gf_framework.discovery import universal_discovery as ud
    from gf_framework.discovery.gf_discovery import mu_filtered_cross
    from gf_framework.verification import joint_sos as _js
    from gf_framework.verification import mosek_direct as _md
    from gf_framework.verification.exact_phase import (
        gram_bundle_fingerprint, save_gram_bundle)
    from gf_framework.verification.joint_sos import (
        JointSOSOptimizer, get_y_parity_blocks_default)

    record: dict = {
        "wp": "P4-02",
        "driver": str(Path(__file__).resolve()),
        "driver_note": (
            "packaged path (JointSOSOptimizer.find_lyapunov_with_dynamics), "
            "y-parity blocking at the P5-04 packaged default; supersedes the "
            "August prototype driver artifacts/campaign_2026-08/E4_levant4d/"
            "structure/prototype/yparity_driver.py"),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pinned_git_sha": _git_sha(pinned_repo),
        "pinned_worktree": str(pinned_repo),
        "gf_framework_file": gf_framework.__file__,
        "command": " ".join([sys.executable] + sys.argv),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mosek": ".".join(str(v) for v in mosek.Env.getversion()),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "hash_randomization": int(sys.flags.hash_randomization),
        "system": args.system,
        "m": args.m,
        "delta0": 0.0,
        "design_epsilon": args.epsilon,
        "solver": "MOSEK_DIRECT",
        "y_parity_blocks_default": bool(get_y_parity_blocks_default()),
        "library_order": LIBRARY_ORDER,
        "claim": ("a feasible design is a CANDIDATE, not a certificate; "
                  "nothing in this record is proved"),
    }

    def _dump(status: str) -> int:
        record["typed_status"] = status
        record["peak_rss_gb"] = round(_peak_rss_gb(), 3)
        record["total_s"] = round(time.time() - t_wall0, 1)
        path = out_dir / f"p4_02_{args.tag}.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=False,
                                   default=str), encoding="utf-8")
        print(f"RECORD {path}")
        print(f"TYPED_STATUS {status}")
        return 0

    # ---------------------------------------------------------------- library
    t0 = time.time()
    spec = getattr(ud, f"create_{args.system}_spec")()
    weights = tuple(int(r) for r in spec.weights_r)
    n = len(weights)
    dyn = spec.dynamics_gf
    vertices = spec.dynamics_gf_vertices or [dyn]
    diag = enumerate_diagonal_terms(args.m, weights, c1_only=True)
    mu, _, cross = mu_filtered_cross(args.m, weights, diag, dyn)
    terms = list(diag) + list(cross)
    t_library = time.time() - t0
    record["library"] = {
        "n_terms": len(terms), "n_diag": len(diag), "n_cross": len(cross),
        "mu": list(int(v) for v in mu), "weights_r": list(weights),
        "n_endpoints": len(vertices), "enumeration_s": round(t_library, 1),
    }
    print(f"library: {len(terms)} terms, mu={mu}, {t_library:.1f}s")

    # ------------------------------------------- exact symbolic support guard
    # Run BEFORE assembly, in this driver, at the DESIGN mu (V stage) and at
    # the W stage's own mu (which P2-09 may double). An OddLiftRefusal here is
    # the RESULT of the cell.
    guard: dict = {"stage_v": None, "stage_w": None, "refusal": ""}
    t0 = time.time()
    try:
        lib_exps = [tuple(t.exponents) for t in terms]
        dyn_sym = gf_exponent_support(vertices)
        rep = check_even_support(
            mu=tuple(int(v) for v in mu), weights_r=weights,
            library_exponents=lib_exps, dynamics_vertices=(),
            clearing_power=0)
        guard["stage_v"] = {"ok": True, "report": str(rep)[:400]}
    except OddLiftRefusal as exc:
        guard["stage_v"] = {"ok": False, "detail": str(exc)[:2000]}
        guard["refusal"] = "odd_lift_v"
    except (GuardUnreadable, TypeError) as exc:
        guard["stage_v"] = {"ok": False, "detail": str(exc)[:2000]}
        guard["refusal"] = "guard_unavailable"
    if guard["refusal"] == "":
        try:
            repw = check_even_support(
                mu=tuple(int(v) for v in mu), weights_r=weights,
                library_exponents=lib_exps, dynamics_vertices=dyn_sym,
                clearing_power=0)
            guard["stage_w"] = {"ok": True, "report": str(repw)[:400],
                                "mu": [int(v) for v in mu]}
        except OddLiftRefusal as exc:
            guard["stage_w"] = {"ok": False, "detail": str(exc)[:2000],
                                "mu": [int(v) for v in mu]}
            guard["refusal"] = "odd_lift_w"
        except (GuardUnreadable, TypeError) as exc:
            guard["stage_w"] = {"ok": False, "detail": str(exc)[:2000]}
            guard["refusal"] = "guard_unavailable"
    guard["note"] = (
        "driver-side pre-assembly run of algebra.support_guard at the DESIGN "
        "mu for both stages. The AUTHORITATIVE per-stage guard (W at its own "
        "mu, doubled by P2-09 when the W y-degree is odd) is the packaged "
        "JointSOSOptimizer._plan_y_parity, recorded under 'y_parity'.")
    guard["guard_s"] = round(time.time() - t0, 2)
    record["symbolic_guard_predesign"] = guard
    print(f"symbolic guard (V stage, driver-side): {guard['stage_v']}")
    if guard["refusal"] in ("odd_lift_v", "guard_unavailable"):
        # A V-stage odd lift (or an unreadable guard) means the blocking is
        # not admissible and the dense encoding does not fit at n = 5. This is
        # a RESULT of the cell, not a bug (WP P4-02).
        record["note"] = ("typed refusal before assembly; recorded as the "
                          "cell's outcome, not as an error")
        return _dump(f"refused:{guard['refusal']}")
    if guard["refusal"] == "odd_lift_w":
        # NOT an abort: P2-09's 2mu policy is exactly the handling for this,
        # and the packaged path re-runs the guard at the doubled mu.
        print("note: W stage odd at the DESIGN mu; P2-09's 2mu policy applies")

    # --------------------------------------------------------------- solve
    solve_marks: list = []
    _orig_solve = _md.solve_joint_sos_direct

    def _timed_solve(*a, **kw):
        t = time.time()
        try:
            return _orig_solve(*a, **kw)
        finally:
            solve_marks.append(time.time() - t)

    _md.solve_joint_sos_direct = _timed_solve

    opt = JointSOSOptimizer(
        weights_r=weights,
        epsilon=args.epsilon,
        solver="MOSEK_DIRECT",
        strict_solver=True,
        allow_solver_fallback=False,
        verbose=True,
        capture_design_grams=not args.no_bundle,
        w_epsilon_rows=True,
        design_screen=args.design_screen,
        solver_timeout=args.mosek_timeout,
    )
    t0 = time.time()
    try:
        result = opt.find_lyapunov_with_dynamics(
            degree_m=args.m,
            dynamics_gf=dyn,
            active_terms=terms,
            dynamics_gf_vertices=spec.dynamics_gf_vertices,
        )
    finally:
        _md.solve_joint_sos_direct = _orig_solve
    t_design = time.time() - t0
    t_solve = float(sum(solve_marks))
    record["wall"] = {
        "library_enumeration_s": round(t_library, 1),
        "design_total_s": round(t_design, 1),
        "mosek_solve_s": round(t_solve, 1),
        "assembly_s": round(t_design - t_solve, 1),
        "n_solve_calls": len(solve_marks),
        "note": ("assembly_s = design_total_s - mosek_solve_s, measured by "
                 "wrapping (never editing) mosek_direct.solve_joint_sos_direct; "
                 "library enumeration is counted separately"),
    }
    record["design"] = {
        "feasible": bool(result.feasible),
        "solver_status": result.solver_status,
        "message": getattr(result, "message", "") or "",
        "n_terms": len(terms),
        "solver_attempts": list(getattr(result, "solver_attempts", []) or []),
    }
    record["y_parity"] = dict(result.y_parity or {})
    print(f"design: feasible={result.feasible} status={result.solver_status} "
          f"total={t_design:.1f}s solve={t_solve:.1f}s")
    print(f"y_parity: {record['y_parity']}")

    if not result.feasible:
        record["note"] = ("typed float evidence only; a non-feasible design "
                          "is never a negative result about the system")
        return _dump("design_not_feasible")

    # ----------------------------------------------------- V candidate terms
    record["candidate_V_terms"] = [
        {"exponents": [str(Fraction(e)) for e in t.exponents],
         "term_types": list(t.term_types),
         "coefficient": float(t.coefficient)}
        for t in result.V_gf.terms
    ]

    # ------------------------------------------------------------- the bundle
    entries = result.gram_certificates
    if not entries:
        record["bundle"] = {"captured": False}
        return _dump("feasible_candidate_no_bundle")

    def _psd_scalars(dim: int) -> int:
        return dim * (dim + 1) // 2

    blocks = []
    worst_min_eig = None
    worst_resid = 0.0
    worst_resid_block = ""
    worst_eig_block = ""
    dense_psd = 0
    for e in entries:
        g = np.asarray(e["gram"], dtype=np.float64)
        dim = int(g.shape[0])
        dense_psd += _psd_scalars(dim)
        ev = np.linalg.eigvalsh(0.5 * (g + g.T))
        mn = float(ev[0])
        mx = float(ev[-1])
        tol = max(dim, 1) * float(np.finfo(np.float64).eps) * max(abs(mx), 1.0)
        rank = int((ev > tol).sum())
        # Dense-row reassembly residual: recompute EVERY unreduced dense
        # matching row <A_mono, Q> from the reassembled dense Gram and compare
        # it to the stored (eps-subtracted) target. The union covers the
        # completion rows (P2-09), i.e. products outside the target support.
        B = np.asarray([[int(v) for v in b] for b in e["basis"]],
                       dtype=np.int64)
        base = int(2 * B.max() + 1)
        powers = base ** np.arange(B.shape[1], dtype=np.int64)
        codes = B @ powers                       # linear, so code_i + code_j
        keys = (codes[:, None] + codes[None, :]).ravel()
        uniq, inv = np.unique(keys, return_inverse=True)
        sums = np.bincount(inv, weights=g.ravel(), minlength=uniq.size)
        poly = {int(k): float(v) for k, v in zip(uniq, sums)}
        target = {}
        for k, val in e["target"].items():
            code = int(np.asarray([int(v) for v in k], dtype=np.int64) @ powers)
            target[code] = target.get(code, 0.0) + float(val)
        rmax = 0.0
        for mono in set(poly) | set(target):
            d = abs(poly.get(mono, 0.0) - target.get(mono, 0.0))
            if d > rmax:
                rmax = d
        if worst_min_eig is None or mn < worst_min_eig:
            worst_min_eig = mn
            worst_eig_block = e["name"]
        if rmax > worst_resid:
            worst_resid = rmax
            worst_resid_block = e["name"]
        blocks.append({
            "name": e["name"], "block": e["block"], "dim": dim,
            "octant_index": int(e["octant_index"]),
            "endpoint": e.get("endpoint"),
            "eps": float(e["eps"]), "eps_reason": e.get("eps_reason", ""),
            "mu": [int(v) for v in e["mu"]], "y_degree": int(e["y_degree"]),
            "mu_policy": e.get("mu_policy", ""),
            "library_order": e.get("library_order", ""),
            "clearing_power": int(e.get("clearing_power", 0) or 0),
            "parity_class_sizes": sorted(
                np.bincount((B % 2) @ (2 ** np.arange(B.shape[1]))).tolist(),
                reverse=True),
            "min_eig": mn, "max_eig": mx, "rank": rank,
            "dense_row_residual_linf": rmax,
            "psd_scalars_dense": _psd_scalars(dim),
        })

    yp = record["y_parity"]
    record["sizes"] = {
        "n_bundle_entries": len(entries),
        "dense_psd_scalars": dense_psd,
        "blocked_psd_scalars": yp.get("psd_scalars_emitted"),
        "dense_psd_scalars_from_yparity": yp.get("psd_scalars_dense"),
        "n_psd_blocks": yp.get("n_psd_blocks"),
        "n_gram_blocks": yp.get("n_gram_blocks"),
        "max_block_dim_dense": max(b["dim"] for b in blocks),
        "max_block_dim_blocked": max(max(b["parity_class_sizes"])
                                     for b in blocks),
        "n_parity_classes_V": max(
            (len([c for c in b["parity_class_sizes"] if c])
             for b in blocks if b["block"] == "V"), default=0),
        "n_parity_classes_W": max(
            (len([c for c in b["parity_class_sizes"] if c])
             for b in blocks if b["block"] == "W"), default=0),
        "note": ("dense_psd_scalars is the projection over the reassembled "
                 "dense Grams (the bundle entries); blocked_psd_scalars is "
                 "what the SDP actually posed"),
    }
    record["reassembly"] = {
        "worst_min_eig": worst_min_eig,
        "worst_min_eig_block": worst_eig_block,
        "worst_dense_row_residual": worst_resid,
        "worst_dense_row_residual_block": worst_resid_block,
        "min_rank_over_dim": min(b["rank"] / b["dim"] for b in blocks),
        "n_full_rank_blocks": sum(1 for b in blocks if b["rank"] == b["dim"]),
        "note": ("blocked Grams reassembled block-diagonally onto the full "
                 "basis and re-checked against the UNREDUCED dense matching "
                 "rows, completion rows included"),
    }
    record["blocks"] = blocks
    print(f"reassembly: worst_min_eig={worst_min_eig:.6e} "
          f"worst_dense_row_residual={worst_resid:.6e}")

    # ----------------------------------------------------------- design screen
    ds = result.design_screen
    if ds is not None:
        record["design_screen"] = ds
        gram = (ds or {}).get("gram") or {}
        print(f"design_screen: ok={ds.get('ok')} reason={ds.get('reason')!r} "
              f"n_blocks={gram.get('n_blocks')} "
              f"gateable={gram.get('n_blocks_gateable')} "
              f"failed={gram.get('n_blocks_failed')}")
    else:
        record["design_screen"] = None

    # ------------------------------------------------------------ persist npz
    t0 = time.time()
    npz = out_dir / f"p4_02_{args.tag}_grams.npz"
    save_gram_bundle(
        npz, entries,
        system=args.system, m=args.m, epsilon=args.epsilon, delta0=0.0,
        wp="P4-02", pinned_git_sha=record["pinned_git_sha"],
        library_order=LIBRARY_ORDER,
        mu_policy=(entries[0].get("mu_policy", "") if entries else ""),
        solver="MOSEK_DIRECT",
        mosek_version=record["mosek"], numpy_version=np.__version__,
        python_version=record["python"],
        y_parity=record["y_parity"],
        claim=record["claim"],
    )
    fp = gram_bundle_fingerprint(entries)
    sha = subprocess.check_output(["sha256sum", str(npz)], text=True).split()[0]
    record["bundle"] = {
        "captured": True,
        "path": str(npz),
        "bytes": npz.stat().st_size,
        "sha256": sha,
        "fingerprint": fp,
        "n_entries": len(entries),
        "save_s": round(time.time() - t0, 1),
        "library_order": LIBRARY_ORDER,
        "mu_policy": sorted({e.get("mu_policy", "") for e in entries}),
        "schema": entries[0].get("schema"),
    }
    print(f"bundle: {npz} {npz.stat().st_size} bytes fp={fp[:16]} sha={sha[:16]}")

    # --------------------------------------------- single-octant exact probe
    if args.probe_octant >= 0:
        from gf_framework.verification.exact_gate import (
            exact_terms_from_gf, validate_one_capture)
        cap = None
        for e in entries:
            if e["block"] == "V" and int(e["octant_index"]) == args.probe_octant:
                cap = e
                break
        t0 = time.time()
        probe: dict = {"octant_index": args.probe_octant,
                       "block": cap["name"] if cap else None,
                       "note": ("single-octant spot probe on the design "
                                "bundle; NOT the X record, P4-03 owns X")}
        if cap is None:
            probe["reason"] = "no_such_octant"
        else:
            try:
                exact_terms = exact_terms_from_gf(result.V_gf)
                chk, failed = validate_one_capture(exact_terms, cap, n)
                probe.update({
                    "ok": bool(chk.ok), "reason": chk.reason,
                    "target_assembly_failed": bool(failed),
                    "charge_float": chk.charge_float,
                    "budget_float": chk.budget_float,
                    "charge_ratio": (chk.charge_float / chk.budget_float
                                     if chk.budget_float else None),
                    "eta_hat_float": chk.eta_hat_float,
                    "resid_l1_float": chk.resid_l1_float,
                    "min_eig_float": chk.min_eig_float,
                    "degree": chk.degree, "n_basis": chk.n_basis,
                })
            except Exception as exc:  # typed as a probe outcome, never a bug
                probe["reason"] = f"probe_error:{type(exc).__name__}:{exc}"[:500]
        probe["probe_s"] = round(time.time() - t0, 1)
        record["exact_probe_one_octant"] = probe
        print(f"exact probe: {probe.get('ok')} ratio="
              f"{probe.get('charge_ratio')} {probe['probe_s']}s")

    return _dump("feasible_candidate")


if __name__ == "__main__":
    raise SystemExit(main())
