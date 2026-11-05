"""Exact-rational gain certification at the PUBLISHED standard tunings.

Repairs finding F1 of docs/papers/automatica_note/REVIEW_GATE_20260711.md:
the per-endpoint exact phase (Proposition 7) historically read every gain as
Fraction(float(gain)) — the dyadic float64 image — so the exactly certified
system carried gain stand-ins ~1e-16 away from the published values (even
1.1 ≠ 11/10 as a dyadic; float64(0.3) < 3/10). This driver re-runs the exact
phase for the PUBLISHED certificate V (reconstructed from the row JSON, bit
for bit) against dynamics whose gains and δ₀ are injected as exact rationals:

  * rational published values enter exactly (11/10, 3/2, 2, 3, δ₀ ∈ {1, 3/10});
  * an irrational gain g (e.g. λ₂ = (3/2)√2) is covered by a two-point rational
    BRACKET lo < g < hi: with V fixed, the octant target polynomial is affine
    in each gain (the gain multiplies its own disjoint set of Lie-derivative
    terms), so exact-PASS at both bracket endpoints certifies every gain value
    in [lo, hi] — the same convexity mechanism as the disturbance-vertex
    reduction. Bracket validity is asserted in exact arithmetic by power
    comparison (lo² < 9/2 < hi² for (3/2)√2, lo⁶ < 6561/8 < hi⁶ for the
    levant4d λ₃, …); no floating-point step is trusted anywhere.

With k irrational gains the run covers all 2^k bracket corners (a multi-affine
function on a box attains its extrema at the vertices); each corner verifies
W± per inclusion endpoint. V is gain-independent, so its exact gate result is
identical across corners. Since campaign item R2 that identity is EXPLOITED
rather than recomputed: V is proved once at corner 0 and reused, guarded by
the exact content hash of ``verification.exact_phase.v_input_hash`` (V terms,
weights, exact ε), which refuses reuse on any mismatch. ``--no-dedup``
restores the historical per-corner re-proof.

Per-octant Gram captures are persisted to a companion *_grams_*.json (the
standalone verifier consumes them; REVIEW_GATE items F1/M4).

Solver chain (retargeted 2026-09-03, WP P2-12, after the P2-06 validation
gate). This driver IS the exact phase for the flagship published rows: its
verdict is the Proposition-7 proof, so it runs the exact phase's own default
witness policy (PIPELINE.md §4, decision D1) rather than a pinned solver.
``--solver`` defaults to **MOSEK** (the primary witness re-solve) and
``--fallback-solver`` to **CLARABEL** (exactly one labelled retry, taken only
when the primary witness fails or leaves too little charge headroom). No
design happens here, so there is no design-solver knob.

Adoption is never a trust decision about a solver: a witness is adopted only
because the EXACT rational phase verified the Gram it proposed, and every
attempt plus the adopted ``witness_source`` / ``witness_solver`` /
``witness_solver_used`` lands in the record. Before P2-12 this call pinned
its witness solver to CLARABEL — a single attempt through the CVXPY
encoding. G is a
**historical float referee, not a citability condition** from 2026-09
(CANONICAL_STATE.md legend, decision D-g (b)); pass ``--solver CLARABEL`` to
reproduce the historical arm.

**Output location.** ``--out-dir`` defaults to
``artifacts/campaign_2026-08/E1_matrix/standard_gains_2026-09/``. Records used
to be written into ``artifacts/ablation/levant_2003_perturbed``, the frozen
wording root; writing there now requires ``--allow-frozen-root``. The published
row is still READ from that root by default (it is the published certificate
being proved); ``--row`` overrides it.

Gain values and the bracket construction come from
``gf_framework.systems.printed_gains`` (campaign item C20), the single source
of truth shared with the 12-system benchmark re-proof
(``review/reverify_exact.py --printed-rationals``). This driver covers the
flagship PERTURBED rows at the published standard tunings; the benchmark tool
covers the nominal matrix at the module-default gain points. Those are
different gain points for levant3d, deliberately kept apart in the registry.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gf_framework.systems.printed_gains import (
    FLAGSHIP_GAIN_POINTS,
    decimal_bracket,
    legacy_gain_spec,
)
from gf_framework.verification.exact_phase import (
    WITNESS_FALLBACK_SOLVER,
    WITNESS_PRIMARY_SOLVER,
    build_v_cache,
    run_exact_phase,
)

from _frozen_root import add_out_dir_args, resolve_out_dir  # noqa: E402
from levant_2003_perturbed import (  # noqa: E402
    DEFAULT_OUT_DIR,
    OUT,
    _git_sha,
    assert_vertex_encoding,
    reconstruct_V,
)

# Per system: the spec factory, disturbance channel, exactly-rational published
# gains, and the irrational gains as (power k, exact target q = gain^k, float
# approximation for bracket seeding). q pins the algebraic value: for positive
# reals, lo^k < q < hi^k ⟺ lo < gain < hi, decided entirely in ℚ.
#
# The numbers themselves now live in ONE place, the printed-gain registry
# gf_framework.systems.printed_gains (campaign item C20), which the 12-system
# benchmark re-proof (review/reverify_exact.py --printed-rationals) reads as
# well. This dict keeps this driver's historical config shape; the values are
# the same flagship standard tunings as before:
#   sta                 (3/2, 11/10)
#   levant3d            (2, (3/2)√2, 11/10)
#   levant3d_2003print  (3, (3/2)√3, 11/10)
#   levant4d            (3, 2·3^{2/3}, (3/2)√λ₂, 11/10)
# decimal_bracket lives in the registry too, so both drivers share one bracket
# convention.
GAIN_SPECS = {
    name: legacy_gain_spec(point) for name, point in FLAGSHIP_GAIN_POINTS.items()
}


def _serialize_grams(captures) -> list:
    out = []
    for cap in captures or []:
        out.append({
            "signs": list(cap["signs"]),
            "octant_index": cap["octant_index"],
            "eps": cap["eps"],
            "mu": list(cap["mu"]),
            "y_degree": cap["y_degree"],
            "basis": [list(b) for b in cap["basis"]],
            "gram": [[float(x) for x in row] for row in cap["gram"]],
        })
    return out


def _assert_epsilon_matches(captures, expected_eps: float, label: str) -> None:
    """Guard the eps-ladder record: every capture's embedded ``eps`` must equal
    the record's nominal epsilon.

    ``verify_gf_sos_full`` (called by ``run_exact_phase``) walks a descending
    eps-ladder internally (``epsilon``, ``epsilon*1e-3``, ``epsilon*1e-6``) and
    returns whichever rung actually certified — but this driver writes
    ``args.epsilon`` (the top rung) into ``record["epsilon_exact"]``
    unconditionally. If V or one W endpoint/corner only certified at a LOWER
    rung, the record would claim a stronger margin than what was actually
    proved for that component. The standalone verifier's per-capture
    ``eps == epsilon_exact`` check would eventually reject such a record, but
    only at re-verification time, long after the mismatch was silently
    written. Catch it here, at record-writing time, instead.
    """
    for cap in captures or []:
        if cap["eps"] != expected_eps:
            raise AssertionError(
                f"{label}: epsilon-ladder mismatch — this capture certified "
                f"at eps={cap['eps']!r} but the record's epsilon_exact is "
                f"{expected_eps!r}; a lower ladder rung certified while the "
                f"top rung is what would have been recorded"
            )


def _phase_summary(phase) -> dict:
    out = {
        "V": {"exact_ok": bool(phase.v_exact.ok),
              "worst_charge_ratio": phase.v_exact.worst_charge_ratio},
        "exact_pass": phase.exact_pass,
        "worst_charge_ratio": phase.worst_charge_ratio,
        "time_s": phase.time_s,
        # R2: whether this corner's V half was reused from corner 0, and the
        # hashes that make the reuse auditable.
        "v_reused": phase.v_reused,
        "v_input_hash": phase.v_input_hash,
        "v_fingerprint": phase.v_fingerprint,
    }
    for label, ep in zip(("W_plus", "W_minus"), phase.endpoints):
        out[label] = {
            "resolve_feasible": ep.resolve_feasible,
            "exact_ok": bool(ep.exact.ok),
            "worst_charge_ratio": ep.exact.worst_charge_ratio,
            "lie_crosscheck": ep.lie_crosscheck,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, choices=sorted(GAIN_SPECS))
    ap.add_argument("--row", default=None,
                    help="row JSON with the published V (default: the "
                         "canonical published row for the system/δ₀ under "
                         "artifacts/ablation/levant_2003_perturbed/, which is "
                         "READ only — outputs go to --out-dir)")
    ap.add_argument("--delta0", type=Fraction, default=Fraction(1),
                    help="disturbance bound as an EXACT rational, e.g. 1 or 3/10")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--solver", default=WITNESS_PRIMARY_SOLVER,
                    help="X, the exact phase's PRIMARY witness re-solve "
                         "(default: %(default)s). The verdict is the exact "
                         "rational charge against the Proposition-7 budget, "
                         "never a solver status.")
    ap.add_argument("--fallback-solver", default=WITNESS_FALLBACK_SOLVER,
                    help="the exact phase's single labelled FALLBACK witness "
                         "retry, taken only when the primary witness fails or "
                         "leaves too little charge headroom (default: "
                         "%(default)s). Pass an empty string to disable it.")
    ap.add_argument("--digits", type=int, default=15,
                    help="decimal digits of the λ-bracket endpoints")
    ap.add_argument("--no-dedup", action="store_true",
                    help="re-solve and re-prove the (gain-independent) V at "
                         "every bracket corner, as before campaign item R2; "
                         "the default proves V once and reuses it under an "
                         "exact input-hash guard")
    add_out_dir_args(ap, DEFAULT_OUT_DIR, what="the record and its Gram bundle")
    args = ap.parse_args()
    fallback_solver = args.fallback_solver or None
    out_dir = resolve_out_dir(args.out_dir, DEFAULT_OUT_DIR,
                              allow_frozen=args.allow_frozen_root)

    cfg = GAIN_SPECS[args.system]
    row_path = Path(args.row) if args.row else (
        OUT / f"{args.system}_delta{float(args.delta0):g}_row.json")
    row = json.loads(row_path.read_text())
    assert row["system"] == args.system, (row["system"], args.system)
    V = reconstruct_V(row)
    print(f"published V reconstructed from {row_path.name}: "
          f"{len(V.terms)} terms, degree {row['degree_used']}", flush=True)

    brackets = {
        name: decimal_bracket(k, q, approx, args.digits)
        for name, (k, q, approx) in cfg["irrational"].items()
    }
    for name, (lo, hi) in brackets.items():
        k, q, _ = cfg["irrational"][name]
        # Row-pairing sanity only: the row stores the float64 image, which may
        # sit ~half an ulp outside the decimal bracket of the TRUE value.
        row_val = row["gains"][name]
        assert abs(row_val - float(lo)) < 1e-12, (
            f"row gain {name}={row_val} far from bracket [{lo}, {hi}]")
        print(f"bracket {name}: [{lo}, {hi}]  (exact: lo^{k} < {q} < hi^{k})",
              flush=True)
    for name, val in cfg["rational"].items():
        row_val = row["gains"][name]
        assert abs(row_val - float(val)) < 1e-12, (name, row_val, val)

    corner_names = sorted(brackets)
    corners = list(itertools.product(*((brackets[n]) for n in corner_names))) \
        or [()]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "timestamp_utc": ts,
        "git_sha": _git_sha(),
        "system": args.system,
        "row_source": row_path.name,
        "delta0_exact": str(args.delta0),
        "epsilon": args.epsilon,
        "epsilon_exact": str(Fraction(float(args.epsilon))),
        "witness_solver": args.solver,
        "witness_fallback_solver": fallback_solver,
        "out_dir": str(out_dir),
        "rational_gains": {k: str(v) for k, v in cfg["rational"].items()},
        # Published V embedded verbatim (JSON round-trips float64 exactly),
        # so the record + grams bundle is self-contained for the standalone
        # verifier.
        "V": {"degree_m": row["degree_used"],
              "weights_r": row["weights_r"],
              "terms": row["V_terms"]},
        "brackets": {
            n: {"lo": str(brackets[n][0]), "hi": str(brackets[n][1]),
                "power_k": cfg["irrational"][n][0],
                "power_target": str(cfg["irrational"][n][1])}
            for n in corner_names
        },
        "corners": [],
    }
    grams_bundle = {"timestamp_utc": ts, "system": args.system,
                    "V_grams": None, "corners": []}

    record["v_dedup"] = {
        "enabled": not args.no_dedup,
        "protocol": (
            "V is gain-independent (exact_octant_target consumes only the V "
            "terms, mu, octant signs and epsilon), so it is proved once at "
            "corner 0 and reused at the remaining corners under the exact "
            "content hash verification.exact_phase.v_input_hash; a mismatch "
            "raises instead of reusing. --no-dedup re-proves every corner."
        ),
    }

    all_ok = True
    v_cache = None
    for corner in corners:
        assignment = dict(zip(corner_names, corner))
        kwargs = {**cfg["rational"], **assignment, "delta0": args.delta0}
        spec = cfg["factory"](**kwargs)
        assert_vertex_encoding(spec, cfg["channel"], args.delta0)
        # The affinity argument needs BOTH inclusion vertices (G1 mirror).
        assert len(spec.dynamics_gf_vertices) == 2, \
            f"need 2 inclusion vertices, got {len(spec.dynamics_gf_vertices)}"
        label = ", ".join(f"{n}={'lo' if assignment[n] == brackets[n][0] else 'hi'}"
                          for n in corner_names) or "rational gains only"
        print(f"=== corner [{label}]: exact phase ...", flush=True)
        t0 = time.time()
        phase = run_exact_phase(V, spec.dynamics_gf_vertices,
                                len(spec.weights_r),
                                epsilon=args.epsilon, solver=args.solver,
                                fallback_solver=fallback_solver,
                                v_cache=v_cache)
        if (not args.no_dedup and v_cache is None
                and phase.v_resolve_feasible and phase.v_exact.ok):
            v_cache = build_v_cache(V, args.epsilon, phase)
        # Record-integrity guard: the recorded epsilon_exact must be the
        # epsilon that ACTUALLY certified, not just the top of the ladder
        # verify_gf_sos_full started from (see _assert_epsilon_matches).
        _assert_epsilon_matches(phase.v_grams, args.epsilon, f"[{label}] V")
        for ep_label, ep in zip(("W+", "W-"), phase.endpoints):
            _assert_epsilon_matches(ep.grams, args.epsilon, f"[{label}] {ep_label}")
        summary = _phase_summary(phase)
        summary["witness"] = phase.witness_record()
        summary["assignment"] = {n: str(v) for n, v in assignment.items()}
        summary["corner_label"] = label
        record["corners"].append(summary)
        if grams_bundle["V_grams"] is None:
            grams_bundle["V_grams"] = _serialize_grams(phase.v_grams)
        grams_bundle["corners"].append({
            "corner_label": label,
            "assignment": {n: str(v) for n, v in assignment.items()},
            "endpoints": [_serialize_grams(ep.grams) for ep in phase.endpoints],
        })
        ok = phase.exact_pass
        all_ok = all_ok and ok
        worst = summary["worst_charge_ratio"]
        worst_str = f"{worst:.2e}" if worst is not None else "-"
        print(f"=== corner [{label}]: "
              f"{'exact-PASS' if ok else 'FAIL'} "
              f"(worst charge/budget {worst_str}, {time.time()-t0:.0f} s)",
              flush=True)

    # Completeness invariants the standalone verifier re-checks (G1): the
    # record must carry the FULL 2^k corner box and both endpoints per corner.
    assert len(record["corners"]) == 2 ** len(corner_names), "corner box incomplete"
    assert all(len(bc["endpoints"]) == 2 for bc in grams_bundle["corners"]), \
        "endpoint captures incomplete"

    record["exact_pass_all"] = all_ok
    rec_path = out_dir / f"{args.system}_exactgains_{ts}.json"
    grams_path = out_dir / f"{args.system}_exactgains_grams_{ts}.json"
    record["grams_file"] = grams_path.name
    rec_path.write_text(json.dumps(record, indent=1))
    grams_path.write_text(json.dumps(grams_bundle))
    print(f"record: {rec_path}")
    print(f"grams:  {grams_path} "
          f"({grams_path.stat().st_size / 1e6:.1f} MB)")
    print(f"OVERALL: {'exact-PASS at every corner x endpoint' if all_ok else 'FAIL'}",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
