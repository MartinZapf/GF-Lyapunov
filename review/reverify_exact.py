#!/usr/bin/env python3
"""Exact-rational offline re-verification of published GF-SOS certificates.

Implements the Proposition-7 protocol (docs/papers/theory_notes.md §8): the
hardened gate's margin-budget inequality (Proposition 6) is exact mathematics
evaluated in float64; this tool discharges that caveat for the published
artifacts by re-proving it in exact rational arithmetic.

Per certified system in the matrix JSON:
  1. reconstruct V from the stored term structure (float64 coefficients read
     as exact dyadic rationals; exponents are exact Fractions already);
  2. re-solve the per-octant SOS problems for V and W = −dV/dt with an
     independent solver (``--solver``, strict — no in-solve fallback),
     capturing the Gram matrices (the published artifacts store certificates,
     not Grams). Campaign item P2-05 / decision D1: if that witness does not
     prove the certificate, or proves it charging more than a tenth of the
     Proposition-7 budget, the phase is retried EXACTLY once with CLARABEL and
     the better VERIFIED witness is adopted; every attempt is recorded and the
     row names the adopted ``witness_solver``;
  3. rebuild every octant target from scratch in exact arithmetic (exact Lie
     derivative from the rationalized dynamics, exact μ-lifting/association,
     exact margin), so the float pipeline's assembly is not trusted either;
  4. verify each captured Gram exactly: E = target − zᵀQ̄z in ℚ, rational η̂
     with Q̄ + η̂I ⪰ 0 by exact LDLᵀ, and the charge η̂·C₁ + ‖E‖₁ ≤ ε/2 in ℚ.

Floats only PROPOSE (the Gram, the shift η̂); every accepted inequality is
checked in exact rational arithmetic. A cross-check of the exact Lie
derivative against the float pipeline (support + coefficients) is reported
per system as a self-test of the mirror.

Two gain bindings are available (campaign item C20, block E1.3):

  * DEFAULT (unchanged): the spec factories are called with their float
    defaults, so the proved object carries the float64 IMAGES of the gains
    (2.7 as a dyadic, not 27/10). This is the June protocol and stays
    available for comparison.
  * ``--printed-rationals``: the gains are injected as the exact rationals of
    their printed decimals from ``gf_framework.systems.printed_gains``, with
    genuinely irrational gains covered by rational brackets and all 2^j
    bracket corners enumerated. This is what removes footnote 1 of
    CANONICAL_STATE.md from the X column.

Writes ``reverify_exact_<UTC>.json`` + ``.md`` next to the matrix JSON
(``reverify_exact_printed_<UTC>.*`` under ``--printed-rationals``). With
``--dump-grams`` each row additionally gets a gzipped per-system Gram-capture
sidecar (``<stem>_grams_<system>.json.gz``, linked from the row's
``grams_file`` field) that ``scripts/emit_certificate_record.py`` replays
(campaign E3/C9); without it the artifact records per-octant verdicts only
and cannot seed portable records.

Usage:
    python3 review/reverify_exact.py <matrix.json> [--solver MOSEK]
                                     [--systems sta,twisting] [--epsilon 1e-6]
                                     [--printed-rationals [--digits 15]]
                                     [--gram-bundle-dir DIR] [--no-witness-fallback]
                                     [--jobs N]

``--jobs`` (campaign item P3-03) spreads the EXACT rational arithmetic over N
worker processes: a row's 2^n per-octant validations, and a bracketed row's
corners 1..k-1 after corner 0 has produced the shared V proof. It defaults to
1, which creates no pool and is the historical serial path byte for byte. No
float solve is parallelised, and no verdict, margin or octant order depends on
N — results are assembled in canonical order regardless of completion order.

``--solver`` defaults to MOSEK (the Task-API backend, since P2-10) as of the
P2-06 validation gate: decision D1 makes D -> X -> P the citability chain and
the gate measured identical exact verdicts on all twelve rows against the four
stored CLARABEL/MOSEK records, at 15.2 s against 1286.2 s.  Pass
``--solver CLARABEL`` to reproduce the historical strict-gate arm.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _artifact_io import atomic_json, atomic_text, atomic_target, result_receipt

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.systems.printed_gains import (
    BENCHMARK_GAIN_POINTS,
    BRACKET_DIGITS,
    exact_dynamics_corners,
    gain_point,
    gain_point_record,
)
from gf_framework.verification.exact_gate import (
    MARGIN_POLICY_ACHIEVED,
    MARGIN_POLICY_HALF,
)
from gf_framework.verification.exact_phase import (
    DEFAULT_JOBS,
    OCTANT_START_METHOD,
    WITNESS_CHARGE_TRIGGER as CHARGE_TRIGGER,
    WITNESS_FALLBACK_SOLVER as FALLBACK_SOLVER,
    GramBundleError,
    OctantWorkerError,
    build_v_cache,
    resolve_jobs,
    run_exact_phase,
)

#: Typed :class:`CornerWorkerError` reasons (WP P3-03). A bracket corner that
#: ran in a worker process and did not come back fails the ROW with one of
#: these; it is never silently dropped and never a shorter corner box.
CORNER_WORKER_REASONS = frozenset({
    "corner_worker_exception",   # the corner's exact phase raised in the worker
    "corner_worker_died",        # the worker process died (OOM kill, signal)
    "corner_pool_unavailable",   # the pool could not be created / fed
})


class CornerWorkerError(RuntimeError):
    """A bracket corner scheduled in a worker process produced no result.

    A harness failure, never a verdict: the row is recorded as failed with
    ``reason`` from :data:`CORNER_WORKER_REASONS` and the corner index, and the
    remaining rows still run. Re-running with ``--jobs 1`` reproduces the
    corner serially, which is what tells a reader whether the mathematics or
    the scheduling failed.
    """

    def __init__(self, reason: str, detail: str, *,
                 corner: int | None = None) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.corner = corner

SPEC_FACTORIES = {
    "fosmc": "create_fosmc_spec", "power_rate": "create_power_rate_spec",
    "sta": "create_sta_spec", "twisting": "create_twisting_spec",
    "example1": "create_example1_spec", "example3": "create_example3_spec",
    "continuous_twisting": "create_cta_spec", "levant3d": "create_levant3d_spec",
    "pidsmc": "create_pidsmc_spec", "dsta": "create_dsta_spec",
    "fc_cta": "create_fc_cta_spec", "levant4d": "create_levant4d_spec",
}


def _reconstruct_V(row) -> GeneralizedForm:
    terms = [
        GFTerm(coefficient=float(t["coefficient"]),
               exponents=tuple(Fraction(e) for e in t["exponents"]),
               term_types=tuple(t["term_types"]))
        for t in row["V_terms"]
    ]
    return GeneralizedForm(degree_m=row["degree_used"],
                           weights_r=tuple(row["weights_r"]), terms=terms)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _capture_summary(res) -> dict:
    return {
        "ok": res.ok,
        "reason": res.reason,
        "mu": list(res.mu),
        "eps": res.eps,
        "worst_charge_ratio": res.worst_charge_ratio,
        # R1 provenance (campaign item P3-04): which deck group the exact
        # LDLᵀ actually blocked on, rolled up over the form and repeated per
        # octant. Nested under one key so the addition is purely additive —
        # a record written before this existed simply has no
        # "r1_provenance", and every reader that ignores unknown keys is
        # unaffected in both directions.
        "r1_provenance": res.r1_provenance(),
        "octants": [
            {
                "signs": list(c.signs), "ok": c.ok, "reason": c.reason,
                "degree": c.degree, "n_basis": c.n_basis,
                "eta_hat": c.eta_hat, "min_eig_float": c.min_eig_float,
                "resid_l1": c.resid_l1_float, "charge": c.charge_float,
                "budget": c.budget_float,
                "r1_provenance": c.r1_provenance(),
            }
            for c in res.octants
        ],
    }


#: Bundle file names ``--gram-bundle-dir`` looks for, per system, in order.
BUNDLE_SUFFIXES = (".npz", ".json.gz", ".json")


def find_gram_bundle(bundle_dir, system: str):
    """The design Gram bundle for one system, or ``None``.

    Campaign item R3: with a bundle the exact phase consumes the DESIGN Gram
    and runs no float solve. The bundle must belong to the V of the row being
    re-verified — a bundle from another design point is refused by
    ``run_exact_phase`` with ``gram_bundle_input_mismatch``, which is the
    point of naming it here rather than merging bundles by hand.
    """
    if not bundle_dir:
        return None
    base = Path(bundle_dir)
    for suffix in BUNDLE_SUFFIXES:
        candidate = base / f"{system}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _corner_phase(V, dynamics, n_vars, *, epsilon, solver, v_cache,
                  gram_bundle, fallback_solver, jobs):
    """One bracket corner's exact phase. Module-level, so it pickles.

    This is what a corner worker process runs (WP P3-03). It takes the exact
    same arguments the serial loop passes to ``run_exact_phase`` — including
    corner 0's :class:`VProofCache`, which pickles with its proof and its Gram
    captures, so a worker reuses R2's dedup exactly as an in-process corner
    does and the ``v_input_hash`` refusal still guards it inside the worker.
    """
    return run_exact_phase(V, [dynamics], n_vars,
                           epsilon=epsilon, solver=solver, v_cache=v_cache,
                           gram_bundle=gram_bundle,
                           fallback_solver=fallback_solver, jobs=jobs)


def printed_rational_phases(V, system: str, *, epsilon: float, solver: str,
                            digits: int = BRACKET_DIGITS, dedup: bool = True,
                            gram_bundle=None, fallback_solver=FALLBACK_SOLVER,
                            jobs: int = DEFAULT_JOBS,
                            start_method: str = OCTANT_START_METHOD):
    """Exact phase at the PRINTED gains, once per bracket corner (C20 / E1.3).

    The dynamics come from the printed-gain registry, so every dynamics
    coefficient is the exact rational of its printed decimal (or an exact
    rational bracket endpoint of a genuinely irrational gain). Systems without
    an irrational gain have exactly one corner.

    With ``dedup`` (the default, campaign item R2) the gain-independent V is
    proved once at corner 0 and reused for the rest, guarded by the exact
    input hash in ``verification.exact_phase``; ``dedup=False`` (driver flag
    ``--no-dedup``) restores the per-corner re-proof.

    ``gram_bundle`` (R3) makes the design Grams the witness at every corner:
    V is gain-independent and the W target is affine in each gain, so the same
    bundle is checked against each corner's exact target and refused if it
    does not match.

    ``fallback_solver`` (P2-05) is the ONE retry when the primary witness
    fails or leaves less than a factor 10 of the Proposition-7 budget; each
    corner records its own attempt list.

    ``jobs`` (WP P3-03) is one worker budget spent on ONE level at a time,
    never nested:

      * corner 0 always runs IN THIS PROCESS with the full ``jobs`` per-octant
        pool. It has to: it is what produces the V proof every later corner
        reuses (R2), so fanning out before it lands makes every corner
        re-solve V — P3-05 measured that worst case at 2942 MB and 445 s per
        corner against 889 MB and 90 s for a cached one;
      * corners 1..k-1 then fan out over ``min(jobs, k-1)`` worker processes,
        each running its own corner serially (``jobs=1`` inside), with corner
        0's ``VProofCache`` handed over by pickle.

    A single-corner system therefore spends the whole budget on octants, and a
    k-corner system spends it on corners after the first. Results are
    assembled in CORNER ORDER regardless of completion order. A corner worker
    that raises or dies becomes a :class:`CornerWorkerError`, which fails the
    row with a typed reason and leaves the other rows running.

    Returns ``[(ExactDynamics, ExactPhaseResult), ...]``.
    """
    point = gain_point(system)
    corners = list(exact_dynamics_corners(point, digits=digits))
    n_workers = resolve_jobs(jobs)

    # --- corner 0, in this process, with the octant pool -------------------
    phase0 = run_exact_phase(V, [corners[0].dynamics], len(corners[0].weights_r),
                             epsilon=epsilon, solver=solver, v_cache=None,
                             gram_bundle=gram_bundle,
                             fallback_solver=fallback_solver, jobs=n_workers,
                             start_method=start_method)
    v_cache = (build_v_cache(V, epsilon, phase0)
               if dedup and phase0.v_resolve_feasible and phase0.v_exact is not None
               and phase0.v_exact.ok else None)
    out = [(corners[0], phase0)]
    rest = corners[1:]
    if not rest:
        return out

    common = dict(epsilon=epsilon, solver=solver, v_cache=v_cache,
                  gram_bundle=gram_bundle, fallback_solver=fallback_solver)
    if n_workers <= 1 or len(rest) == 1:
        # Serial, byte for byte the pre-P3-03 loop; the remaining corner (if
        # exactly one) is not worth a worker process.
        for ed in rest:
            out.append((ed, _corner_phase(V, ed.dynamics, len(ed.weights_r),
                                          jobs=n_workers, **common)))
        return out

    slots: list = [None] * len(rest)
    ctx = multiprocessing.get_context(start_method)
    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(n_workers, len(rest)), mp_context=ctx) as ex:
            futures = {
                ex.submit(_corner_phase, V, ed.dynamics, len(ed.weights_r),
                          jobs=1, **common): i
                for i, ed in enumerate(rest)
            }
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    slots[i] = fut.result()
                except BrokenProcessPool as exc:
                    raise CornerWorkerError(
                        "corner_worker_died",
                        f"{type(exc).__name__}: {exc}", corner=i + 1) from exc
                except Exception as exc:  # noqa: BLE001 - typed, then raised
                    raise CornerWorkerError(
                        "corner_worker_exception",
                        f"{type(exc).__name__}: {exc}"[:400],
                        corner=i + 1) from exc
    except BrokenProcessPool as exc:
        raise CornerWorkerError("corner_worker_died",
                                f"{type(exc).__name__}: {exc}") from exc
    except CornerWorkerError:
        raise
    except OSError as exc:
        raise CornerWorkerError("corner_pool_unavailable",
                                f"{type(exc).__name__}: {exc}") from exc
    missing = [i + 1 for i, s in enumerate(slots) if s is None]
    if missing:
        # A dropped corner must never read as a smaller, passing corner box.
        raise CornerWorkerError(
            "corner_worker_exception",
            f"corners {missing} produced no result out of {len(corners)}")
    # Canonical order, never completion order.
    out.extend(zip(rest, slots))
    return out


def _corner_summary(ed, phase) -> dict:
    summary = ed.summary()
    summary.update({
        "exact_pass": phase.exact_pass,
        "v_resolve_feasible": phase.v_resolve_feasible,
        "V_exact_ok": bool(phase.v_exact.ok),
        "W_exact_ok": bool(phase.endpoints[0].exact.ok),
        "worst_charge_ratio": phase.worst_charge_ratio,
        "lie_crosscheck": phase.endpoints[0].lie_crosscheck,
        "time_s": phase.time_s,
        # R2 provenance: was this corner's V half reused from corner 0?
        "v_reused": phase.v_reused,
        "v_input_hash": phase.v_input_hash,
        "v_fingerprint": phase.v_fingerprint,
        # P3-05 provenance: what the PERSISTED (cross-process) V-proof cache
        # did for this corner — hit/miss, the typed refusal reason, the path.
        # Empty when no cache was configured on this call.
        "v_proof_cache": dict(phase.v_proof_cache),
        # P3-03 provenance: how THIS corner's octants were scheduled. Corner 0
        # carries the octant pool; corners 1+ under --jobs > 1 carry
        # {"jobs": 1, ...} because the budget was spent on corners instead.
        "parallel": dict(phase.parallel),
    })
    return summary


def _serialize_grams(captures) -> list:
    """JSON-ready per-octant Gram captures.

    Same shape as ``scripts/exact_gate_standard_gains.py::_serialize_grams``;
    ``scripts/emit_certificate_record.py`` consumes both.
    """
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


def _grams_bundle(system: str, phases, epsilon: float):
    """C9-consumable Gram bundle for one nominal matrix row.

    Returns ``(bundle, None)`` or ``(None, refusal_reason)``. The refusal
    mirrors exact_gate_standard_gains.py::_assert_epsilon_matches: if any
    capture certified at a lower rung of the internal eps-ladder than the
    artifact's nominal epsilon, the bundle would claim a stronger margin than
    what was proved, so it is not written.
    """
    for _, p in phases:
        for caps in [p.v_grams] + [ep.grams for ep in p.endpoints]:
            for cap in caps or []:
                if cap["eps"] != epsilon:
                    return None, (
                        "epsilon_ladder_mismatch: capture certified at "
                        f"eps={cap['eps']!r}, artifact epsilon is {epsilon!r}")
    corners = []
    for ed, p in phases:
        corner = {
            "corner_label": ed.corner_label if ed is not None
                            else "float64 default spec",
            "W_grams": _serialize_grams(p.endpoints[0].grams),
        }
        if ed is not None:
            # Bracketed gains only, exact endpoint values; matches the
            # positional corner box emit_certificate_record.py rebuilds from
            # the printed-gain registry (and its named-assignment guard).
            corner["assignment"] = {n: str(ed.gain_values[n])
                                    for n in ed.corner}
        corners.append(corner)
    return {"system": system, "epsilon": epsilon,
            "V_grams": _serialize_grams(phases[0][1].v_grams),
            "corners": corners}, None


def _protocol_threshold(records) -> str:
    """The acceptance inequality the recorded runs actually used (WP P2-14).

    ``"< eps"`` when every row ran under ``margin_policy="achieved"`` (the
    default since P2-14), ``"<= eps/2"`` under ``"half"``, and both spelled
    out when a run mixed them. A record without the key predates P2-13 and is
    ``"half"``.
    """
    policies = {(r.get("witness") or {}).get("margin_policy", MARGIN_POLICY_HALF)
                for r in records} or {MARGIN_POLICY_HALF}
    parts = {MARGIN_POLICY_ACHIEVED: "< eps", MARGIN_POLICY_HALF: "<= eps/2"}
    return " and ".join(parts[p] for p in sorted(policies) if p in parts) \
        or "<= eps/2"


def select_requested_rows(data, systems=None):
    """Require exactly one eligible input for every named proof; never succeed vacuously."""
    candidates = data.get("results") if isinstance(data, dict) else None
    if not isinstance(candidates, list):
        return [], [], [{"reason": "invalid_matrix_results"}]
    if systems is None:
        requested = [r.get("system") for r in candidates if isinstance(r, dict)
                     and r.get("certified") is True and r.get("V_terms")]
    else:
        requested = [part.strip() for part in systems.split(",")]
    errors, rows, seen = [], [], set()
    for name in requested:
        if name in seen:
            errors.append({"system": name, "reason": "duplicate_request"})
            continue
        seen.add(name)
        if name not in SPEC_FACTORIES:
            errors.append({"system": name, "reason": "unknown_system"})
            continue
        matches = [r for r in candidates if isinstance(r, dict) and r.get("system") == name]
        reason = ("missing_requested_system" if not matches else
                  "duplicate_matrix_system" if len(matches) != 1 else
                  "ineligible_requested_system" if not (matches[0].get("certified") is True
                          and isinstance(matches[0].get("V_terms"), list)
                          and matches[0]["V_terms"] and "degree_used" in matches[0]) else None)
        if reason:
            errors.append({"system": name, "reason": reason})
        else:
            rows.append(matches[0])
    if not requested:
        errors.append({"reason": "empty_proof_selection"})
    return rows, requested, errors


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_json")
    ap.add_argument("--solver", default="MOSEK",
                    help="Witness solver for the exact phase's re-solve tier "
                         "(default MOSEK, i.e. the Task-API backend since "
                         "P2-10; flipped from CLARABEL by the P2-06 "
                         "validation gate under decision D1). CLARABEL "
                         "reproduces the historical strict-gate arm and stays "
                         "the measured fallback witness either way")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--systems", default=None,
                    help="comma-separated subset (default: all certified rows)")
    ap.add_argument("--printed-rationals", action="store_true",
                    help="inject the gains as the exact rationals of their "
                         "printed decimals (printed_gains registry) instead of "
                         "rebuilding the float64 default specs; genuinely "
                         "irrational gains are covered by rational brackets and "
                         "every bracket corner is re-proved")
    ap.add_argument("--digits", type=int, default=BRACKET_DIGITS,
                    help="decimal digits of the rational bracket endpoints "
                         "(--printed-rationals only)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="re-solve and re-prove the (gain-independent) V at "
                         "every bracket corner, as before campaign item R2; "
                         "the default proves V once and reuses it under an "
                         "exact input-hash guard (--printed-rationals only; "
                         "single-corner systems are unaffected)")
    ap.add_argument("--gram-bundle-dir", default=None,
                    help="directory of DESIGN Gram bundles, one per system "
                         "(<system>.npz / .json.gz / .json, written by "
                         "verification.exact_phase.save_gram_bundle). When a "
                         "system's bundle is found it becomes the exact "
                         "phase's witness and NO float re-solve runs for that "
                         "row (campaign item R3, decisions D1/D2); --epsilon "
                         "must then be the margin the design subtracted. A "
                         "bundle that does not belong to the row's V is "
                         "refused, not used")
    ap.add_argument("--no-witness-fallback", action="store_true",
                    help="do not retry with " + FALLBACK_SOLVER + " when the "
                         "primary witness fails or leaves a worst exact "
                         f"charge/budget ratio above {CHARGE_TRIGGER:g} "
                         "(campaign item P2-05, decision D1). The retry is "
                         "dropped automatically when --solver already names "
                         "the fallback solver")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help="worker PROCESSES for the exact rational arithmetic "
                         "(campaign item P3-03). 1 (the default) is the "
                         "historical serial path and creates no pool, so "
                         "stored baselines do not move. N > 1 spreads a "
                         "single-corner row's 2^n octants over N spawn "
                         "workers, and a k-corner row's corners 1..k-1 over "
                         "min(N, k-1) workers after corner 0 has proved the "
                         "shared V. Verdicts, margins and octant order are "
                         "independent of N: results are assembled by index, "
                         "and a worker that dies fails the row with a typed "
                         "reason instead of shortening it")
    ap.add_argument("--start-method", default=OCTANT_START_METHOD,
                    choices=("spawn", "forkserver", "fork"),
                    help="multiprocessing start method for --jobs (default "
                         "spawn). fork is UNSAFE here: MOSEK and the BLAS "
                         "thread pool are already loaded in the parent by the "
                         "time the exact phase runs and neither survives a "
                         "fork. Exposed only so the unsafety is measurable, "
                         "never as a recommendation")
    ap.add_argument("--dump-grams", action="store_true",
                    help="write one gzipped per-system Gram-capture bundle "
                         "(<stem>_grams_<system>.json.gz) next to the "
                         "artifact, linked from the row's grams_file field; "
                         "scripts/emit_certificate_record.py replays these "
                         "into portable records (E3/C9)")
    ap.add_argument("--result-path", help="Write a machine-readable result receipt here.")
    args = ap.parse_args()
    data = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))
    rows, requested, coverage_errors = select_requested_rows(data, args.systems)
    if coverage_errors:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = "reverify_exact_printed" if args.printed_rationals else "reverify_exact"
        out = Path(args.matrix_json).parent / f"{prefix}_{stamp}.json"
        atomic_json(out, {"matrix_json": str(Path(args.matrix_json).resolve()),
                         "requested_systems": requested, "coverage_complete": False,
                         "systems": [{"system": error.get("system"), "exact_pass": False,
                                      "request_failure": error} for error in coverage_errors]})
        atomic_text(out.with_suffix(".md"), "# Exact verification request refused\n\n" +
                    "\n".join(f"- {e.get('system')}: {e['reason']}" for e in coverage_errors) + "\n")
        if args.result_path:
            result_receipt(args.result_path, out, requested)
        print(f"Incomplete exact verification request; diagnostics: {out}", file=sys.stderr)
        raise SystemExit(2)

    if args.printed_rationals:
        missing = sorted({r["system"] for r in rows} - set(BENCHMARK_GAIN_POINTS))
        if missing:
            sys.exit(f"no printed-gain registry entry for: {missing}")

    print(f"Exact-rational re-verification (Proposition 7) of {args.matrix_json}")
    print(f"re-solve: {args.solver} (strict, no in-solve fallback), "
          f"ε = {args.epsilon:g}; exact arithmetic: Fraction "
          f"(dyadic-exact coefficients)")
    if args.no_witness_fallback:
        print("witness policy: single attempt (--no-witness-fallback)\n")
    else:
        print(f"witness policy: retry ONCE with {FALLBACK_SOLVER} when the "
              f"witness fails or charges > {CHARGE_TRIGGER:g} of the budget "
              "(P2-05/D1); dropped when it repeats --solver\n")
    if resolve_jobs(args.jobs) > 1:
        print(f"parallelism: --jobs {resolve_jobs(args.jobs)} worker processes "
              f"({args.start_method}) over the EXACT arithmetic only; "
              "verdicts, margins and octant order are unchanged (P3-03)\n")
    if args.printed_rationals:
        print("gain binding: PRINTED RATIONALS from "
              "gf_framework.systems.printed_gains "
              f"(brackets at {args.digits} decimal digits, all corners re-proved)\n")
    print(f"{'system':<20}{'V exact':<10}{'W exact':<10}{'worst charge/budget':<22}"
          f"{'lie mirror':<12}{'time':<8}")
    print("-" * 86)

    records = []
    failures = []
    grams_pending = []
    for row in rows:
        name = row["system"]
        t0 = time.time()
        V = _reconstruct_V(row)

        rec = {"system": name, "degree_used": row["degree_used"],
               "n_terms": row.get("n_terms")}

        # --- Proposition 7 via the shared orchestrator (exact_phase.py):
        # strict re-solve with Gram capture, exact targets rebuilt from
        # scratch, exact LDLᵀ, Lie-mirror cross-check. Under
        # --printed-rationals this runs once per bracket corner; V is
        # gain-independent, so its exact result must agree across corners and
        # the repeat is a consistency check (same reasoning as
        # scripts/exact_gate_standard_gains.py).
        # R3: when a design Gram bundle is named for this system it becomes
        # the witness and no float solve runs. A bundle that does not belong
        # to this row's V raises GramBundleError; that is recorded as a typed
        # row failure rather than aborting the whole matrix.
        bundle_path = find_gram_bundle(args.gram_bundle_dir, name)
        fallback = None if args.no_witness_fallback else FALLBACK_SOLVER
        try:
            if args.printed_rationals:
                phases = printed_rational_phases(V, name, epsilon=args.epsilon,
                                                 solver=args.solver,
                                                 digits=args.digits,
                                                 dedup=not args.no_dedup,
                                                 gram_bundle=bundle_path,
                                                 fallback_solver=fallback,
                                                 jobs=args.jobs,
                                                 start_method=args.start_method)
            else:
                spec = getattr(ud, SPEC_FACTORIES[name])()
                n = len(spec.weights_r)
                phases = [(None, run_exact_phase(V, [spec.dynamics_gf], n,
                                                 epsilon=args.epsilon,
                                                 solver=args.solver,
                                                 gram_bundle=bundle_path,
                                                 fallback_solver=fallback,
                                                 jobs=args.jobs,
                                                 start_method=args.start_method))]
        except GramBundleError as exc:
            rec.update({"exact_pass": False, "time_s": round(time.time() - t0, 2),
                        "gram_bundle_refusal": {"reason": exc.reason,
                                                "detail": exc.detail,
                                                "bundle": str(bundle_path)}})
            records.append(rec)
            failures.append(name)
            print(f"{name:<20}{'REFUSED':<10}{exc.reason}")
            continue
        except (CornerWorkerError, OctantWorkerError) as exc:
            # P3-03: a worker that raised or died fails THIS row with a typed
            # reason and never silently shortens its corner box or its octant
            # set. The remaining rows still run; the process exits 1.
            rec.update({"exact_pass": False, "time_s": round(time.time() - t0, 2),
                        "worker_failure": {
                            "reason": exc.reason,
                            "detail": exc.detail,
                            "corner": getattr(exc, "corner", None),
                            "octant": getattr(exc, "index", None),
                            "jobs": args.jobs,
                            "start_method": args.start_method}})
            records.append(rec)
            failures.append(name)
            print(f"{name:<20}{'WORKER':<10}{exc.reason}")
            continue
        phase = phases[0][1]
        ep = phase.endpoints[0]
        v_exact, w_exact = phase.v_exact, ep.exact
        rec["v_resolve_feasible"] = phase.v_resolve_feasible
        rec["w_resolve_feasible"] = ep.resolve_feasible
        rec["lie_crosscheck"] = ep.lie_crosscheck
        rec["V_exact"] = _capture_summary(v_exact)
        rec["W_exact"] = _capture_summary(w_exact)
        # R3 provenance: WHICH object witnessed this row. "resolve" is the
        # historical independent float re-solve; "design_bundle" means the
        # DESIGN Grams proved it and no float solve ran on the certifying
        # route. The record names exactly one witness either way.
        rec["witness"] = phase.witness_record()
        # P3-03: how this row's exact arithmetic was scheduled. Corner 0's own
        # octant record, plus the corner-level fan-out the row actually used.
        rec["parallel"] = {
            **dict(phase.parallel),
            "jobs_requested": int(args.jobs),
            "start_method_requested": args.start_method,
            "n_corners": len(phases),
            "corner_workers": (min(resolve_jobs(args.jobs), len(phases) - 1)
                               if len(phases) > 1 and resolve_jobs(args.jobs) > 1
                               and len(phases) - 1 > 1 else 0),
        }
        # Verdicts are over ALL corners; with a single corner they coincide
        # with the corner-0 values above.
        v_ok_all = all(p.v_exact.ok for _, p in phases)
        w_ok_all = all(p.endpoints[0].exact.ok for _, p in phases)
        ok = all(p.exact_pass for _, p in phases)
        if args.printed_rationals:
            rec["gain_binding"] = "printed_rationals"
            rec["gain_point"] = gain_point_record(gain_point(name),
                                                  digits=args.digits)
            rec["n_corners"] = len(phases)
            rec["V_exact_ok_all"] = bool(v_ok_all)
            rec["W_exact_ok_all"] = bool(w_ok_all)
            rec["corners"] = [_corner_summary(ed, p) for ed, p in phases]
        rec["exact_pass"] = ok
        rec["time_s"] = round(time.time() - t0, 2)
        records.append(rec)
        if not ok:
            failures.append(name)
        if args.dump_grams:
            bundle, refusal = _grams_bundle(name, phases, args.epsilon)
            grams_pending.append((rec, name, bundle, refusal))

        ratios = [r for _, p in phases
                  for r in (p.v_exact.worst_charge_ratio,
                            p.endpoints[0].exact.worst_charge_ratio)
                  if r is not None]
        ratio_str = f"{max(ratios):.2e}" if ratios else "-"
        lc = rec["lie_crosscheck"]
        lie_str = ("ok" if lc["support_match"] else "SUPPORT!") + \
                  f"/{lc['max_coeff_diff']:.0e}"
        line = (f"{name:<20}"
                f"{'PASS' if v_ok_all else 'FAIL':<10}"
                f"{'PASS' if w_ok_all else 'FAIL':<10}"
                f"{ratio_str:<22}{lie_str:<12}{rec['time_s']:<8.1f}")
        if args.printed_rationals:
            line += f"{len(phases)} corner(s)"
        print(line)

    print("-" * 86)
    n_pass = sum(1 for r in records if r["exact_pass"])
    print(f"exact-PASS: {n_pass}/{len(records)}")
    if failures:
        print(f"FAILURES: {failures}")

    # --- provenance artifact -------------------------------------------------
    matrix_path = Path(args.matrix_json)
    ts = datetime.now(timezone.utc)
    stem = (f"reverify_exact_printed_{ts.strftime('%Y%m%dT%H%M%S%fZ')}"
            if args.printed_rationals
            else f"reverify_exact_{ts.strftime('%Y%m%dT%H%M%S%fZ')}")

    # --- Gram-capture sidecars (E3/C9), one per system --------------------
    # Written before the artifact dict is assembled so the rows can carry
    # their grams_file link (records holds the same dict objects).
    for rec, name, bundle, refusal in grams_pending:
        if bundle is None:
            rec["grams_file"] = None
            rec["grams_refusal"] = refusal
            print(f"[grams] {name}: NOT written ({refusal})")
            continue
        bundle["timestamp_utc"] = ts.isoformat()
        bundle["matrix_json"] = str(matrix_path)
        bundle["git_sha"] = _git_sha()
        gname = f"{stem}_grams_{name}.json.gz"
        with atomic_target(matrix_path.parent / gname) as temporary:
            with gzip.open(temporary, "wt", encoding="utf-8") as fh:
                json.dump(bundle, fh)
        rec["grams_file"] = gname
        size_mb = (matrix_path.parent / gname).stat().st_size / 1e6
        print(f"[grams] {name}: {gname} ({size_mb:.1f} MB)")



    artifact = {
        "matrix_json": str(matrix_path),
        "timestamp_utc": ts.isoformat(),
        "git_sha": _git_sha(),
        "solver": args.solver,
        "strict_solver": True,
        "epsilon": args.epsilon,
        # WP P2-14: the acceptance threshold is the margin policy's, and the
        # policy is read off the phases that actually ran rather than
        # hard-coded, so this string can never outlive a default flip. It is
        # "charge < eps" under "achieved" (the default since P2-14) and
        # "charge <= eps/2" under "half".
        "protocol": ("Proposition 7 (theory_notes.md §8): exact targets, exact "
                     "residual, exact LDL^T PSD with rational eta-shift, "
                     "charge eta*C1 + ||E||_1 " + _protocol_threshold(records)
                     + " in exact rational arithmetic"),
        "margin_policies": sorted({r["witness"]["margin_policy"]
                                   for r in records
                                   if r.get("witness", {}).get("margin_policy")}),
        "n_checked": len(records),
        "n_exact_pass": n_pass,
        "failures": failures,
        "systems": records,
        "requested_systems": requested, "coverage_complete": True,
    }
    artifact["witness"] = {
        "gram_bundle_dir": args.gram_bundle_dir,
        "sources": sorted({r["witness"]["source"] for r in records
                           if "witness" in r}),
        # P2-05: which solver actually proposed each adopted Gram, and how many
        # rows needed the one retry. A row's own numbers are in
        # systems[i].witness.{witness_solver,witness_charge_ratio,witness_attempts}.
        "witness_solvers": sorted({r["witness"]["witness_solver"] for r in records
                                   if r.get("witness", {}).get("witness_solver")}),
        # P2-10's follow-up, landed by P3-05: which ENCODING actually ran.
        # "MOSEK" (the CVXPY interface) and "MOSEK_DIRECT" (the Task API) are
        # the same REQUEST and only this field separates them.
        "witness_solvers_used": sorted({r["witness"]["witness_solver_used"]
                                        for r in records
                                        if r.get("witness", {})
                                            .get("witness_solver_used")}),
        "n_rows_retried": sum(1 for r in records
                              if len(r.get("witness", {})
                                      .get("witness_attempts") or []) > 1),
        "solver_policy": {
            "primary": args.solver,
            "fallback": None if args.no_witness_fallback else FALLBACK_SOLVER,
            "charge_headroom_trigger": CHARGE_TRIGGER,
            "protocol": (
                "The primary solver proposes the witness. When it does not "
                "prove the certificate, or proves it with a worst exact "
                "charge/budget ratio above the trigger, the phase is retried "
                "EXACTLY once with the fallback solver and the better VERIFIED "
                "witness is adopted (decision D1, WP P2-05). A retry is dropped "
                "when it would repeat the primary. Adoption is never a trust "
                "decision about a solver: every adopted witness was verified in "
                "exact rational arithmetic, and a solver status is never proof."
            ),
        },
        "protocol": (
            "Each row names exactly one witness. 'resolve': the Gram was "
            "proposed by an independent strict float re-solve inside the "
            "exact phase. 'design_bundle': the DESIGN-stage Gram bundle was "
            "the witness (campaign item R3, decisions D1/D2), verified by its "
            "per-entry content hash and bundle fingerprint, and no float "
            "solve ran on the certifying route. The exact octant targets are "
            "rebuilt from scratch in exact arithmetic either way."
        ),
    }
    artifact["parallel"] = {
        # P3-03. Recorded on EVERY run, including the serial default, because
        # a wall-clock number without it is not reproducible.
        "jobs": int(args.jobs),
        "jobs_resolved": resolve_jobs(args.jobs),
        "start_method": args.start_method,
        "cpu_count": os.cpu_count(),
        "protocol": (
            "jobs is the number of worker PROCESSES for the EXACT rational "
            "arithmetic only; no float solve and no solver call is "
            "parallelised. A single-corner row spreads its 2^n per-octant "
            "validations over the workers; a k-corner row proves corner 0 in "
            "this process (it produces the shared V proof) and then fans "
            "corners 1..k-1 out, one corner per worker. Results are assembled "
            "in canonical octant/corner order regardless of completion order, "
            "so verdicts, margins, octant order and the truncation at the "
            "first failing octant are independent of jobs; jobs = 1 creates "
            "no pool and is the historical path byte for byte. A worker that "
            "raises or dies fails its ROW with a typed reason "
            "(systems[i].worker_failure) and never shortens a certificate."
        ),
    }
    artifact["v_dedup"] = {
        "enabled": bool(args.printed_rationals and not args.no_dedup),
        "protocol": (
            "V is gain-independent, so it is proved once at bracket corner 0 "
            "and reused at the remaining corners under the exact content hash "
            "verification.exact_phase.v_input_hash (V terms, weights, exact "
            "epsilon); a mismatch raises instead of reusing. --no-dedup "
            "re-proves V at every corner."
        ),
    }
    if args.printed_rationals:
        artifact["gain_binding"] = "printed_rationals"
        artifact["bracket_digits"] = args.digits
        artifact["gain_registry"] = "gf_framework.systems.printed_gains"
        artifact["gain_protocol"] = (
            "Gains injected as the exact rationals of their printed decimals; "
            "genuinely irrational gains enter as rational brackets lo < g < hi "
            "proved in Q by power comparison, with all 2^j bracket corners "
            "re-proved. With V fixed, each octant target is affine in each "
            "gain, so exact-PASS at every corner certifies the whole gain box."
        )
    out_json = matrix_path.parent / f"{stem}.json"
    atomic_json(out_json, artifact)

    if args.printed_rationals:
        object_lines = [
            "Verified object: the rational certificate (stored float64 coefficients as exact",
            "dyadic rationals, exact Fraction exponents, the dyadic-exact margin, and the gains",
            "as the exact rationals of their PRINTED decimals, from the registry",
            "`gf_framework.systems.printed_gains`). A genuinely irrational gain enters as a",
            f"rational bracket lo < g < hi at {args.digits} decimal digits, proved by exact",
            "power comparison; every bracket corner is re-proved, and since each octant target",
            "is affine in each gain at fixed V, the whole gain box is covered.",
        ]
    else:
        object_lines = [
            "Verified object: the rational certificate (stored float64 coefficients as exact",
            "dyadic rationals, exact Fraction exponents, dyadic-exact gains and margin).",
        ]
    lines = [
        "# Exact-rational re-verification (Proposition 7)",
        "",
        f"- matrix: `{matrix_path.name}`  |  solver: {args.solver} (strict)  |  "
        f"ε = {args.epsilon:g}  |  git `{artifact['git_sha'][:12]}`  |  {ts.isoformat()}",
        f"- **exact-PASS: {n_pass}/{len(records)}**"
        + (f"  — failures: {', '.join(failures)}" if failures else ""),
        *([f"- gain binding: **printed rationals** (registry "
           f"`gf_framework.systems.printed_gains`, brackets at {args.digits} "
           f"decimal digits, all corners re-proved)"]
          if args.printed_rationals else []),
        "",
        *object_lines,
        "Floats only propose the Gram and the PSD shift; every accepted inequality is",
        "re-proved in exact rational arithmetic (exact LDLᵀ, exact residual, exact charge).",
        "",
        "| system | m | V exact | W exact | worst charge/budget | witness | max η̂ | Lie mirror Δ | t [s] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if "gram_bundle_refusal" in r:
            refusal = r["gram_bundle_refusal"]
            lines.append(
                f"| {r['system']} | {r['degree_used']} "
                f"| REFUSED: {refusal['reason']} | REFUSED | - | - | - | - "
                f"| {r['time_s']:.1f} |")
            continue
        if "worker_failure" in r:
            wf = r["worker_failure"]
            lines.append(
                f"| {r['system']} | {r['degree_used']} "
                f"| WORKER: {wf['reason']} | WORKER | - | - | - | - "
                f"| {r['time_s']:.1f} |")
            continue
        ratios = [x for x in (r["V_exact"]["worst_charge_ratio"],
                              r["W_exact"]["worst_charge_ratio"]) if x is not None]
        ratios += [c["worst_charge_ratio"] for c in r.get("corners", [])
                   if c["worst_charge_ratio"] is not None]
        etas = [o["eta_hat"] for o in r["V_exact"]["octants"] + r["W_exact"]["octants"]
                if o.get("eta_hat")]
        eta_max = max((float(Fraction(e)) for e in etas), default=0.0)
        # Under --printed-rationals the verdict is over every bracket corner;
        # otherwise (and for single-corner systems) it is the corner-0 verdict.
        v_ok = r.get("V_exact_ok_all", r["V_exact"]["ok"])
        w_ok = r.get("W_exact_ok_all", r["W_exact"]["ok"])
        # P2-05: which witness the row adopted, and how many were tried.
        wit = r.get("witness") or {}
        n_tried = len(wit.get("witness_attempts") or [])
        # P3-05: name the ENCODING when it differs from the request, so a
        # MOSEK_DIRECT witness is not read as a CVXPY-MOSEK one.
        used = wit.get("witness_solver_used") or ""
        witness_str = (f"{wit.get('witness_source', '-')}/"
                       f"{wit.get('witness_solver') or '-'}"
                       + (f" [{used}]" if used and used != wit.get("witness_solver")
                          else "")
                       + (f" ({n_tried} tried)" if n_tried > 1 else ""))
        lines.append(
            f"| {r['system']} | {r['degree_used']} "
            f"| {'PASS' if v_ok else 'FAIL: ' + r['V_exact']['reason']} "
            f"| {'PASS' if w_ok else 'FAIL: ' + r['W_exact']['reason']} "
            f"| {(format(max(ratios), '.2e') if ratios else '-')} "
            f"| {witness_str} "
            f"| {eta_max:.1e} "
            f"| {r['lie_crosscheck']['max_coeff_diff']:.1e} "
            f"| {r['time_s']:.1f} |"
        )
    if args.printed_rationals:
        lines += [
            "",
            "## Injected gains",
            "",
            "| system | printed rationals | rational brackets | corners |",
            "|---|---|---|---|",
        ]
        for r in records:
            if "gain_point" not in r:  # refused before the phase ran
                continue
            gp = r["gain_point"]
            printed = ", ".join(f"{n} = {v}" for n, v in gp["printed_rational"].items())
            brk = ", ".join(
                f"{n} in [{b['lo']}, {b['hi']}] with {n}^{b['power_k']} = {b['power_target']}"
                for n, b in gp["brackets"].items()
            ) or "none"
            lines.append(f"| {r['system']} | {printed} | {brk} | {r['n_corners']} |")
    out_md = matrix_path.parent / f"{stem}.md"
    # Explicit UTF-8: the report carries ε, η̂, LDLᵀ. Without it Python picks
    # the platform default (cp1252 on Windows) and the write raises after the
    # whole re-proof has already run. Byte-identical on any UTF-8 locale.
    atomic_text(out_md, "\n".join(lines) + "\n")
    if args.result_path:
        result_receipt(args.result_path, out_json, requested)
    print(f"provenance artifacts: {out_json}\n                      {out_md}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
