#!/usr/bin/env python3
"""P4-05: levant4d ratio-1 DESIGN on the packaged blocked path, one cell.

Copied from the P4-03/P4-02 pattern
(``artifacts/campaign_2026-08/E5_levant5d/design_2026-09-04/p4_02_design.py``)
and extended by the only thing the levant4d ratio-1 cells need that the
levant5d nominal cells did not: a **perturbed** instance at an exact covariant
orbit representative.

The representative is the rational ``p`` of the Levant L-family (E4
``solver_diag/README.md``, N12): gains ``lambda_i -> p^i lambda_i``,
``delta0 -> p^4 delta0``. Starting from the printed instance (L = 1, delta0 =
1) every ``p`` names the SAME inclusion question at ratio ``delta0 / L = 1``.
``p = 1`` is the printed instance itself.

Poses the joint V+W design SDP through
``JointSOSOptimizer.find_lyapunov_with_dynamics`` with

  * the y-parity Gram block decomposition at its PACKAGED DEFAULT (P5-04),
  * the W epsilon rows (P2-02) and the 2mu odd-y-degree W policy (P2-09),
  * MOSEK_DIRECT only, no CVXPY, no solver fallback,
  * the design Gram bundle captured (P2-01) and persisted as ``.npz`` (P2-03),
  * ``design_screen`` in ``warn`` (P2-04), so a refused bundle is still
    diagnosable rather than silently dropped.

Claim hygiene (D6): a feasible design is a CANDIDATE, not a certificate.
``solver_infeasible`` here is typed FLOAT evidence and never proof;
``solver_unknown`` decides nothing. Nothing in this record is proved.

RESUMABILITY: the record path is derived from ``--tag``; if it already exists
and carries a typed status, the cell is skipped unless ``--force``.

Usage:
  python p4_05_design.py --p 3/4 --m 9 --epsilon 1e-4 \
      --out-dir <block>/<cell> --tag l4d_m9_p3_4_eps1e-4

``--gain-point KEY`` (WP X10, 2026-09-25) names the base point in
``FLAGSHIP_GAIN_POINTS``. The default ``levant4d`` keeps the original
behaviour bit for bit (the float gains come from the levant4d module
constants). Any other key must be a levant4d-family point (factory
``create_levant4d_spec``, gains lambda1..lambda4); its float design gains are
the registry float images scaled by ``p^i``, and the record, the bundle
document and every downstream artifact carry ``system = KEY``, so the
emitter pins the cell against that registered point.
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


LEVANT4D_GAIN_INDEX = {"lambda1": 1, "lambda2": 2, "lambda3": 3, "lambda4": 4}


def resolve_gain_point(key: str):
    """The registered base point, refused unless it is a levant4d-family point."""
    from gf_framework.systems.printed_gains import FLAGSHIP_GAIN_POINTS
    point = FLAGSHIP_GAIN_POINTS.get(key)
    if point is None:
        raise SystemExit(f"--gain-point {key!r}: not in FLAGSHIP_GAIN_POINTS "
                         f"(known: {sorted(FLAGSHIP_GAIN_POINTS)})")
    if point.factory != "create_levant4d_spec" \
            or set(point.gains) != set(LEVANT4D_GAIN_INDEX):
        raise SystemExit(f"--gain-point {key!r}: not a levant4d-family point "
                         f"(factory {point.factory}, gains {sorted(point.gains)})")
    return point


def scaled_float_gains(p: Fraction, key: str = "levant4d"):
    """The DESIGN point: the module float defaults scaled by ``p^i``.

    Kept in float on purpose. The design solve is float arithmetic either way,
    and the exact phase re-proves at the exact rational bracket corners of the
    SAME scaled point (``p4_05_exact.py``), which is where exactness has to
    live. ``p`` dyadic makes every product below an exact float64 operation;
    ``p = 3/4`` is dyadic, ``p = 1/2`` is dyadic, ``p = 1`` is the identity.

    For a ``key`` other than ``levant4d`` the base floats are the registry
    float images (``float_gain_values``: the rational gains' floats and the
    bracketed gains' float seeds) of that point.
    """
    pf = float(p)
    if key != "levant4d":
        from gf_framework.systems.printed_gains import float_gain_values
        base = float_gain_values(resolve_gain_point(key))
        return {n: base[n] * pf ** i for n, i in LEVANT4D_GAIN_INDEX.items()}
    from gf_framework.systems.levant4d import _L1, _L2, _L3, _L4
    return {
        "lambda1": _L1 * pf,
        "lambda2": _L2 * pf ** 2,
        "lambda3": _L3 * pf ** 3,
        "lambda4": _L4 * pf ** 4,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", default="1",
                    help="rational covariant representative; gains lambda_i -> "
                         "p^i lambda_i, delta0 -> p^4 delta0 (p=1 is the "
                         "printed instance q=1)")
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--design-screen", default="warn",
                    choices=["reject", "warn", "off"])
    ap.add_argument("--probe-octant", type=int, default=0,
                    help="V octant index for the single-octant exact probe; "
                         "-1 disables it")
    ap.add_argument("--mosek-timeout", type=float, default=None)
    ap.add_argument("--delta0-ratio", default="1",
                    help="rational multiplier on delta0 at fixed gains, i.e. "
                         "the ratio delta0/L. 1 is the matched (ratio-1) cell. "
                         "Values > lambda4/L cannot certify and are SOUNDNESS "
                         "controls (E17/N5/N8 shape).")
    ap.add_argument("--gain-point", default="levant4d",
                    help="base point in FLAGSHIP_GAIN_POINTS (a levant4d-family "
                         "key); the default keeps the original behaviour")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    p = Fraction(args.p)
    resolve_gain_point(args.gain_point)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"p4_05_{args.tag}.json"
    if rec_path.exists() and not args.force:
        try:
            done = json.loads(rec_path.read_text())
            if done.get("typed_status"):
                print(f"RESUME design already on disk: {rec_path}")
                print(f"TYPED_STATUS {done['typed_status']}")
                return 0
        except ValueError:
            pass

    t_wall0 = time.time()

    import gf_framework
    import mosek
    from gf_framework.algebra.enumerate import (
        LIBRARY_ORDER, enumerate_diagonal_terms)
    from gf_framework.algebra.support_guard import (
        GuardUnreadable, OddLiftRefusal, check_even_support,
        gf_exponent_support)
    from gf_framework.discovery import universal_discovery as ud
    from gf_framework.discovery.gf_discovery import mu_filtered_cross
    from gf_framework.verification import mosek_direct as _md
    from gf_framework.verification.exact_phase import (
        gram_bundle_fingerprint, save_gram_bundle)
    from gf_framework.verification.joint_sos import (
        JointSOSOptimizer, get_y_parity_blocks_default)

    src = Path(gf_framework.__file__).resolve().parents[2]
    pinned_repo = src.parent

    ratio = Fraction(args.delta0_ratio)
    delta0 = ratio * p ** 4              # exact; the printed delta0 is 1
    gains = scaled_float_gains(p, args.gain_point)

    record: dict = {
        "wp": "P4-05",
        "cell_role": "design (D) at one covariant representative of ratio 1",
        "driver": str(Path(__file__).resolve()),
        "driver_note": (
            "packaged path (JointSOSOptimizer.find_lyapunov_with_dynamics), "
            "y-parity blocking at the P5-04 packaged default; supersedes the "
            "August prototype driver artifacts/campaign_2026-08/E4_levant4d/"
            "structure/prototype/yparity_driver.py for this block"),
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
        "system": args.gain_point,
        "gain_point": args.gain_point,
        "dynamics_factory": "create_levant4d_spec",
        "m": args.m,
        "qscale_p": str(p),
        "delta0_exact": str(delta0),
        "delta0_float": float(delta0),
        "ratio_delta0_over_L": str(ratio),
        "gains_design_float": gains,
        "design_epsilon": args.epsilon,
        "solver": "MOSEK_DIRECT",
        "y_parity_blocks_default": bool(get_y_parity_blocks_default()),
        "library_order": LIBRARY_ORDER,
        "claim": ("a feasible design is a CANDIDATE, not a certificate; "
                  "solver_infeasible is typed float evidence and never proof; "
                  "solver_unknown decides nothing; nothing here is proved"),
    }

    def _dump(status: str) -> int:
        record["typed_status"] = status
        record["peak_rss_gb"] = round(_peak_rss_gb(), 3)
        record["total_s"] = round(time.time() - t_wall0, 1)
        rec_path.write_text(json.dumps(record, indent=1, sort_keys=False,
                                       default=str), encoding="utf-8")
        print(f"RECORD {rec_path}")
        print(f"TYPED_STATUS {status}")
        return 0

    # ---------------------------------------------------------------- library
    t0 = time.time()
    spec = ud.create_levant4d_spec(delta0=float(delta0), **gains)
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
        "mu": [int(v) for v in mu], "weights_r": list(weights),
        "n_endpoints": len(vertices), "enumeration_s": round(t_library, 1),
    }
    print(f"p={p} delta0={delta0} gains={gains}", flush=True)
    print(f"library: {len(terms)} terms, mu={mu}, endpoints={len(vertices)}, "
          f"{t_library:.1f}s", flush=True)

    # ------------------------------------------- exact symbolic support guard
    guard: dict = {"stage_v": None, "stage_w": None, "refusal": ""}
    t0 = time.time()
    lib_exps = [tuple(t.exponents) for t in terms]
    try:
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
        "mu for both stages. The AUTHORITATIVE per-stage guard is the "
        "packaged JointSOSOptimizer._plan_y_parity, recorded under 'y_parity'.")
    guard["guard_s"] = round(time.time() - t0, 2)
    record["symbolic_guard_predesign"] = guard
    print(f"symbolic guard (V stage): {guard['stage_v']}", flush=True)
    if guard["refusal"] in ("odd_lift_v", "guard_unavailable"):
        record["note"] = ("typed refusal before assembly; recorded as the "
                          "cell's outcome, not as an error")
        return _dump(f"refused:{guard['refusal']}")

    # ---------------------------------------------------------------- solve
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
        capture_design_grams=True,
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
          f"total={t_design:.1f}s solve={t_solve:.1f}s", flush=True)

    if not result.feasible:
        # P3-02 vocabulary, applied to a DESIGN outcome.
        st = str(result.solver_status or "")
        if "infeasible" in st:
            typed = "solver_infeasible"
        elif "unknown" in st:
            typed = "solver_unknown"
        else:
            typed = "design_not_feasible"
        record["failure_class"] = typed
        record["note"] = (
            "typed float evidence only. `solver_infeasible` is never proof "
            "(D6, N12: exact covariant scaling has flipped one); "
            "`solver_unknown` decides nothing.")
        return _dump(typed)

    # ----------------------------------------------------- V candidate terms
    record["candidate_V_terms"] = [
        {"exponents": [str(Fraction(e)) for e in t.exponents],
         "term_types": list(t.term_types),
         "coefficient": float(t.coefficient),
         "coefficient_hex": float(t.coefficient).hex()}
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
        B = np.asarray([[int(v) for v in b] for b in e["basis"]],
                       dtype=np.int64)
        base = int(2 * B.max() + 1)
        powers = base ** np.arange(B.shape[1], dtype=np.int64)
        codes = B @ powers
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
          f"worst_dense_row_residual={worst_resid:.6e}", flush=True)

    ds = result.design_screen
    record["design_screen"] = ds
    if ds is not None:
        gram = (ds or {}).get("gram") or {}
        print(f"design_screen: ok={ds.get('ok')} reason={ds.get('reason')!r} "
              f"n_blocks={gram.get('n_blocks')} "
              f"gateable={gram.get('n_blocks_gateable')} "
              f"failed={gram.get('n_blocks_failed')}", flush=True)

    # ------------------------------------------------------------ persist npz
    t0 = time.time()
    npz = out_dir / f"p4_05_{args.tag}_grams.npz"
    save_gram_bundle(
        npz, entries,
        system=args.gain_point, m=args.m, epsilon=args.epsilon,
        delta0=float(delta0),
        wp="P4-05", pinned_git_sha=record["pinned_git_sha"],
        library_order=LIBRARY_ORDER,
        mu_policy=(entries[0].get("mu_policy", "") if entries else ""),
        solver="MOSEK_DIRECT",
        mosek_version=record["mosek"], numpy_version=np.__version__,
        python_version=record["python"],
        y_parity=record["y_parity"],
        qscale_p=str(p),
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
    print(f"bundle: {npz.name} {npz.stat().st_size} B fp={fp[:16]} "
          f"sha={sha[:16]}", flush=True)

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
                                "bundle; NOT the X record")}
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
            except Exception as exc:
                probe["reason"] = f"probe_error:{type(exc).__name__}:{exc}"[:500]
        probe["probe_s"] = round(time.time() - t0, 1)
        record["exact_probe_one_octant"] = probe
        print(f"exact probe: {probe.get('ok')} ratio="
              f"{probe.get('charge_ratio')} {probe['probe_s']}s", flush=True)

    return _dump("feasible_candidate")


if __name__ == "__main__":
    raise SystemExit(main())
