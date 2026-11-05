"""Certify Levant differentiators at the standard 2003 tuning under the
PERTURBED error inclusion |f⁽ⁿ⁾| ≤ δ₀ (matched disturbance, last channel).

Why this exists: in the SMC literature a "convergence proof at gains X" means
the perturbed inclusion — Seeber–Horn 2017 (Automatica 84:241–243) prove
ẋ₂ = −k₂ sign(x₁) + δ with |δ| ≤ L; the nominal case (f⁽ⁿ⁾ ≡ 0) is not the
claim. The nominal records (scripts/levant3d_2003_gains.py and the levant4d
artifacts) therefore do NOT support a "convergence of the differentiator"
claim. This driver measures the perturbed case.

Protocol per degree m (complete μ-lattice slice):
  1. joint V+W design over the full filtered library with the W block
     duplicated per inclusion vertex (f ± δ₀·e_ch; W is affine in δ, so the
     two endpoints cover the interval by convexity — see
     matched_disturbance_vertices);
  2. strict, vertex-aware float gate, no solver fallback;
  3. exact-rational phase PER ENDPOINT: V once, then for each endpoint
     W± = −Lie(V, f ± δ₀·e_ch) is re-solved with Gram capture and re-proved
     in exact arithmetic against that endpoint's dynamics (Proposition 7;
     same flow as review/reverify_exact.py, replicated here because that tool
     rebuilds specs at conservative DEFAULT gains and has no vertex mode).
     Float64 inputs (gains, δ₀) are read as exact dyadic rationals; δ₀ = 1
     is exactly representable.

Solver chain (retargeted 2026-09-03, WP P2-12, after the P2-06 validation
gate). This driver produces CERTIFYING verdicts — the row it writes is a
published certificate and its exact phase is the Proposition-7 proof — so it
runs on the package default chain D -> X -> P (PIPELINE.md §4, decision D1):

  * ``--design-solver`` defaults to **MOSEK_DIRECT** (D, the direct Task-API
    encoding), matching ``discover_gf`` / ``scripts/negative_control.py``
    since commit 4f2d451. No certifying float solve goes through CVXPY.
  * ``--gate-solver`` defaults to **MOSEK** (strict, no fallback). Inside
    ``verify_gf_sos_full`` that name IS the direct poly-SOS backend
    (P2-10/P2-06 contradiction 2). The float gate is a fail-fast filter on a
    proposal, never the proof.
  * ``--exact-solver`` / ``--exact-fallback-solver`` default to the exact
    phase's own witness policy, **MOSEK primary with exactly one CLARABEL
    retry** above the charge-headroom trigger. Adoption is never a trust
    decision about a solver: an attempt is adopted only because the exact
    rational phase verified the Gram it proposed. The adopted witness is
    recorded as ``witness_source`` / ``witness_solver`` /
    ``witness_solver_used``.

Before P2-12 the gate and the exact phase were both pinned to strict CLARABEL
(G). G is a **historical float referee, not a citability condition** from
2026-09 (CANONICAL_STATE.md legend, decision D-g (b)); pass
``--gate-solver CLARABEL --exact-solver CLARABEL`` to reproduce the historical
arm.

**Output location.** ``--out-dir`` (alias ``--output-dir``) defaults to
``artifacts/campaign_2026-08/E1_matrix/standard_gains_2026-09/``. It used to
default to ``artifacts/ablation/levant_2003_perturbed``, the frozen wording
root; writing there now requires ``--allow-frozen-root``.

Systems (channel = the weighted-degree-0 sign channel):
    sta       (k₁,k₂) = (1.5, 1.1), channel 1 — the exact Seeber–Horn
              setting; their Theorem 1 (k₂ > L, k₁ > √(k₂+L): 1.5 vs ≈1.449)
              proves δ₀ = L = 1 converges → validation rung + analytic
              cross-check (WP-5.4).
    levant3d  (2.0, 1.5√2, 1.1) from recursive (1.1, 1.5, 2), channel 2 —
              the order-2 perturbed claim.
    levant4d  (3, 4.1602, 3.0595, 1.1) [module defaults ARE the converted
              2003 gains], channel 3 — the order-3 perturbed claim.

Falsification (WP-5):
  * assert_vertex_encoding: endpoints differ from nominal by EXACTLY one
    constant ±δ₀ term on the disturbance channel, nowhere else (printed).
  * negative control: --delta0 1.2 --expect-fail on levant3d. δ₀ > λ₃ = 1.1
    overpowers the sign channel (ė₃ can be held positive at e₁ > 0), so no
    certificate can exist. If anything certifies, the script exits 2 with a
    STOP-AND-AUDIT banner: the vertex encoding would be proving falsehoods.

Failure ladder: --ladder 0.9 0.75 0.5 retries lower δ₀ (full degree sweep
each) after the primary δ₀ fails everywhere; reports the achieved margin.

``--strict`` (C19(c)): forbids solver fallback on the DESIGN solve for
MOSEK/MOSEK_DIRECT alike (MOSEK_DIRECT is pinned unconditionally; --strict
extends the same no-fallback guarantee to plain MOSEK). The gate runs with
``strict_solver=True`` unconditionally already, so --strict changes nothing
there.

``--parity-reduction`` (campaign block S1, OFF by default): poses the DESIGN
SDP over the even-parity sub-library and over one Gram block per orbit of
x ↦ −x. Perturbed runs get the PAIRED reduction: the W blocks are the
(octant, endpoint) pairs (s, +) ↔ (−s, −), 2ⁿ of 2ⁿ⁺¹, never naive octant
halving at a fixed endpoint (Corollary 10.4 — a fixed endpoint is not odd).
The prerequisites are verified symbolically per run; on any failure the
reduction refuses, the full formulation runs, and the refusal is recorded.
The gate below is UNCHANGED and always verifies the full block set, as does
the exact phase, so this flag can only change how long the design takes and
which candidate comes out — never what is proved. Each degree records
``parity_reduction`` in the run JSON. Without the flag, behavior is
byte-for-byte unchanged.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import enumerate_diagonal_terms
from gf_framework.discovery.gf_discovery import mu_filtered_cross
from gf_framework.verification.exact_phase import (
    WITNESS_FALLBACK_SOLVER,
    WITNESS_PRIMARY_SOLVER,
    run_exact_phase,
)
from gf_framework.verification.joint_sos import JointSOSOptimizer
from gf_framework.verification.verifier import certify_discovered_gf

from _env import collect_env
from _frozen_root import CAMPAIGN_STANDARD_GAINS, add_out_dir_args, resolve_out_dir

# The frozen wording root. Kept as a module constant because it is where the
# PUBLISHED rows are READ from (exact_gate_standard_gains.py, the exact-closure
# drivers, adversarial_sim_perturbed.py all import it). It is no longer the
# default WRITE destination — see DEFAULT_OUT_DIR (WP P2-12).
OUT = REPO_ROOT / "artifacts/ablation/levant_2003_perturbed"

#: Where this driver writes unless --out-dir says otherwise.
DEFAULT_OUT_DIR = CAMPAIGN_STANDARD_GAINS

# Default chain (P2-06 flips, decision D1): MOSEK_DIRECT designs, the strict
# float gate runs on the direct poly-SOS backend that verify_gf_sos_full calls
# "MOSEK", and the exact phase keeps its own witness policy.
DEFAULT_DESIGN_SOLVER = "MOSEK_DIRECT"
DEFAULT_GATE_SOLVER = "MOSEK"

SYSTEMS = {
    "sta": {
        "factory": lambda d0: ud.create_sta_spec(k1=1.5, k2=1.1, delta0=d0),
        "channel": 1,
        "degrees": (3, 4, 5, 6),
        "gains": {"k1": 1.5, "k2": 1.1},
        "note": "Seeber–Horn 2017 setting (validation rung)",
    },
    "levant3d": {
        "factory": lambda d0: ud.create_levant3d_spec(
            lambda1=2.0, lambda2=1.5 * math.sqrt(2.0), lambda3=1.1, delta0=d0),
        "channel": 2,
        "degrees": (5, 6, 7, 8),
        "gains": {"lambda1": 2.0, "lambda2": 1.5 * math.sqrt(2.0),
                  "lambda3": 1.1, "recursive_2003": [1.1, 1.5, 2.0]},
        "note": "order-2 differentiator, 2003-converted gains",
    },
    "levant3d_2003print": {
        # The variant PRINTED in Levant 2003 §7.1 / Levant 2005: recursive
        # (1.1, 1.5, 3) — the sequence without the 2. The "levant3d" system
        # above is the instruction-note / LLY-2017-review / textbook variant
        # (1.1, 1.5, 2). Certifying both covers every published form of the
        # standard tuning (wording repair 2026-07-11, see README).
        "factory": lambda d0: ud.create_levant3d_spec(
            lambda1=3.0, lambda2=1.5 * math.sqrt(3.0), lambda3=1.1, delta0=d0),
        "channel": 2,
        "degrees": (5, 6, 7, 8),
        "gains": {"lambda1": 3.0, "lambda2": 1.5 * math.sqrt(3.0),
                  "lambda3": 1.1, "recursive_2003": [1.1, 1.5, 3.0]},
        "note": "order-2 differentiator, 2003-print variant (1.1, 1.5, 3)",
    },
    "levant4d": {
        # module defaults ARE the 2003-converted gains (3, 4.1602, 3.0595, 1.1)
        "factory": lambda d0: ud.create_levant4d_spec(delta0=d0),
        "channel": 3,
        "degrees": (7, 8),
        "gains": {"lambda1": 3.0, "lambda2": 2.0 * 3.0 ** (2.0 / 3.0),
                  "lambda3": 1.5 * (2.0 * 3.0 ** (2.0 / 3.0)) ** 0.5,
                  "lambda4": 1.1, "recursive_2003": [1.1, 1.5, 2.0, 3.0]},
        "note": "order-3 differentiator, 2003-converted gains",
    },
}


def reconstruct_V(row: dict) -> "GeneralizedForm":
    """Rebuild the published certificate V from a row JSON, bit for bit.

    Shared by adversarial_sim_perturbed.py and exact_gate_standard_gains.py:
    the row's V_terms carry float64 coefficients and exact Fraction exponents;
    the reconstructed GF is exactly the published object.
    """
    from fractions import Fraction
    from gf_framework.algebra.core import GeneralizedForm, GFTerm
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


# Typed trial classification (C19): mirrors robust_delta_sweep.py's vocabulary
# and NEGATIVES.md — certified | solver_infeasible | gate_rejected |
# solver_unknown | timeout | oom | unexpected_error. Additive per-degree field
# (deg["status"]); the existing deg["design"]/["solver_status"]/["certified"]
# fields are unchanged, so this does not alter byte-for-byte behavior.
def _classify_solver_status(status: str) -> str:
    s = (status or "").lower()
    if "infeasible" in s and "unbounded" not in s:
        return "solver_infeasible"
    return "solver_unknown"


def _peak_rss_bytes():
    """Cheap, OS-maintained peak-RSS reading (never an active watchdog).

    Windows: psutil's ``peak_wset`` is the OS-tracked working-set high water
    mark, a genuine peak from a single read. Elsewhere, current RSS is
    reported as a best-effort floor. Process-level OOM-kills above this are
    the external watchdog's job (C5, scripts/_watchdog.py), not this
    driver's.
    """
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)
        return int(peak) if peak else int(mi.rss)
    except Exception:
        return None


def _solver_version(solver_name: str, env: dict) -> str:
    s = (solver_name or "").upper()
    if s.startswith("MOSEK"):
        return env.get("mosek_version", "unknown")
    return env.get("packages", {}).get(s.lower(), "unknown")


def _provenance(argv, design_solver: str, gate_solver: str = DEFAULT_GATE_SOLVER,
                exact_solver: str = WITNESS_PRIMARY_SOLVER,
                exact_fallback_solver=WITNESS_FALLBACK_SOLVER) -> dict:
    """UTC timestamp, git SHA + dirty flag, exact command line, solver
    name+version (design + gate + exact witness), and full env (via
    scripts/_env.py collect_env()); peak RSS if cheaply available. Embedded in
    every emitted JSON (C19(e)).

    ``gate_solver`` used to be hard-wired to ``"CLARABEL"`` here as well as at
    the call site. Both now follow the CLI (WP P2-12), so a record can never
    claim a referee it did not run.
    """
    env = collect_env()
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": env.get("git_sha"),
        "git_dirty": env.get("git_dirty"),
        "command_line": [sys.executable] + list(argv),
        "solver_name": design_solver,
        "solver_version": _solver_version(design_solver, env),
        "gate_solver_name": gate_solver,
        "gate_solver_version": _solver_version(gate_solver, env),
        "exact_witness_solver": exact_solver,
        "exact_witness_fallback_solver": exact_fallback_solver,
        "env": env,
    }
    peak_rss = _peak_rss_bytes()
    if peak_rss is not None:
        prov["peak_rss_bytes"] = peak_rss
    return prov


def assert_vertex_encoding(spec, channel: int, delta0) -> None:
    """WP-5.1: prove (by inspection) the vertices encode f ± δ₀·e_ch and
    nothing else. Prints the differing channel GFs for the record.

    δ₀ may be a float (historical dyadic encoding) or an exact Fraction
    (exact-rational injection runs); the expected coefficient is computed in
    the matching arithmetic so the comparison never crosses a rounding.

    Uses explicit ``raise`` statements rather than bare ``assert``: asserts
    are compiled out under ``python -O``, which would silently disable this
    falsification check instead of failing loudly.
    """
    from fractions import Fraction as _Fr

    def _check(cond, msg):
        if not cond:
            raise AssertionError(msg)

    verts = spec.dynamics_gf_vertices
    _check(verts is not None and len(verts) == 2, "expected 2 endpoint vertices")
    expected_signs = [+1, -1]
    for v, sgn in zip(verts, expected_signs):
        _check(len(v) == len(spec.dynamics_gf), "vertex/nominal channel count mismatch")
        for i, (g, g0) in enumerate(zip(v, spec.dynamics_gf)):
            if i != channel:
                # matched_disturbance_vertices reuses the nominal GF objects
                _check(g is g0, f"non-channel {i} was modified")
        g, g0 = v[channel], spec.dynamics_gf[channel]
        _check(len(g.terms) == len(g0.terms) + 1, "channel must gain exactly 1 term")
        for t, t0 in zip(g.terms[:-1], g0.terms):
            _check(t.coefficient == t0.coefficient
                   and t.exponents == t0.exponents
                   and t.term_types == t0.term_types, "nominal channel terms changed")
        extra = g.terms[-1]
        _check(all(e == 0 for e in extra.exponents), "extra term must be constant")
        # A zero exponent with term_type 'signed' means sign(x) (a genuine
        # discontinuous function), NOT the constant 1 — so checking the
        # exponents are all zero is not enough to prove "extra is a true
        # constant, independent of state"; every position must also be 'abs'.
        _check(all(tt == 'abs' for tt in extra.term_types),
               f"extra term must be a true constant (all 'abs' term_types, "
               f"not 'signed' at exponent 0): got {extra.term_types!r}")
        expected = sgn * delta0 if isinstance(delta0, _Fr) else float(sgn) * float(delta0)
        _check(extra.coefficient == expected,
               f"extra coefficient {extra.coefficient!r} != {expected!r}")
    print(f"vertex encoding OK: channel {channel}, δ₀ = {float(delta0):g}"
          + (f" (exact {delta0})" if isinstance(delta0, _Fr) else ""))
    print(f"  nominal ch{channel}: {spec.dynamics_gf[channel]}")
    print(f"  vertex+  ch{channel}: {verts[0][channel]}")
    print(f"  vertex-  ch{channel}: {verts[1][channel]}", flush=True)


def exact_phase(V, spec, epsilon: float,
                solver: str = WITNESS_PRIMARY_SOLVER,
                fallback_solver=WITNESS_FALLBACK_SOLVER) -> dict:
    """Per-endpoint exact-rational gate (Proposition 7).

    Thin record-building wrapper over the shared orchestrator
    ``gf_framework.verification.exact_phase.run_exact_phase``: V is verified
    once; each endpoint's W = −Lie(V, f ± δ₀·e_ch) is re-solved with Gram
    capture and re-proved exactly against THAT endpoint's dynamics. Both
    endpoints passing certifies the whole inclusion (W affine in δ ⇒
    convexity).

    WP P2-12: this call used to pin its witness solver to CLARABEL, a single
    attempt through the CVXPY encoding. It now runs the exact phase's OWN
    witness policy — MOSEK primary, one CLARABEL retry above the charge
    trigger — and records which witness was adopted. The verdict is unchanged
    in kind either way: only the exact rational phase decides, and a witness
    is adopted only because that phase verified the Gram it proposed.
    """
    n = len(spec.weights_r)
    print(f"exact gate: V re-solve + per-endpoint W (witness {solver}"
          f"{'' if fallback_solver is None else ' + ' + str(fallback_solver) + ' retry'}"
          f", Gram capture) ...", flush=True)
    phase = run_exact_phase(V, spec.dynamics_gf_vertices, n,
                            epsilon=epsilon, solver=solver,
                            fallback_solver=fallback_solver)
    out: dict = {"epsilon": epsilon}
    # Which witness actually carried the proof (P2-03/P2-05/P3-05 vocabulary).
    out["witness"] = phase.witness_record()
    out["V"] = {"resolve_feasible": phase.v_resolve_feasible,
                "exact_ok": bool(phase.v_exact.ok),
                "worst_charge_ratio": phase.v_exact.worst_charge_ratio}
    print(f"exact gate: V {'PASS' if phase.v_exact.ok else 'FAIL'}", flush=True)
    for label, ep in zip(("plus", "minus"), phase.endpoints):
        out[f"W_{label}"] = {
            "resolve_feasible": ep.resolve_feasible,
            "exact_ok": bool(ep.exact.ok),
            "worst_charge_ratio": ep.exact.worst_charge_ratio,
            "lie_crosscheck": ep.lie_crosscheck,
        }
        print(f"exact gate: W ({label}) "
              f"{'PASS' if ep.exact.ok else 'FAIL'} "
              f"(lie mirror Δ={ep.lie_crosscheck['max_coeff_diff']:.1e})",
              flush=True)
    out["worst_charge_ratio"] = phase.worst_charge_ratio
    out["exact_pass"] = phase.exact_pass
    out["time_s"] = phase.time_s
    ratio_str = (f"{out['worst_charge_ratio']:.2e}"
                 if out["worst_charge_ratio"] is not None else "-")
    print(f"exact gate: {'PASS' if out['exact_pass'] else 'FAIL'} "
          f"(worst charge/budget {ratio_str}, {out['time_s']:.0f} s)", flush=True)
    return out


def run_sweep(system: str, delta0: float, degrees, design_solver: str,
              epsilon: float, skip_exact: bool,
              design_epsilon: float = 1e-4, strict: bool = False,
              parity_reduction: bool = False,
              gate_solver: str = DEFAULT_GATE_SOLVER,
              exact_solver: str = WITNESS_PRIMARY_SOLVER,
              exact_fallback_solver=WITNESS_FALLBACK_SOLVER) -> dict:
    """One (system, δ₀) sweep over the degree ladder. Returns the record.

    ``strict`` (C19(c)): forbids design-solve fallback for MOSEK/MOSEK_DIRECT
    alike. MOSEK_DIRECT was already pinned unconditionally
    (``design_solver != "MOSEK_DIRECT"``); strict=False reproduces that exact
    expression, so behavior is byte-for-byte unchanged when the caller does
    not opt in. The gate stays CLARABEL + strict_solver=True regardless.

    ``gate_solver`` (WP P2-12) is the strict float gate's solver, MOSEK by
    default — which inside ``verify_gf_sos_full`` is the direct poly-SOS
    backend, not the CVXPY interface. It used to be pinned to CLARABEL. The
    gate stays ``strict_solver=True`` (no fallback chain) whatever it is.

    ``parity_reduction`` (S1): DESIGN-side only. The reduction never touches
    the strict gate or the exact phase below, both of which keep verifying the
    full block set — that asymmetry is the campaign's S1 gate, not an
    oversight. Each degree's ``parity_reduction`` entry records the mode
    ("paired" here when it applies) or the refusal reason.
    """
    cfg = SYSTEMS[system]
    spec = cfg["factory"](delta0)
    weights = tuple(int(r) for r in spec.weights_r)
    dyn = spec.dynamics_gf
    assert_vertex_encoding(spec, cfg["channel"], delta0)

    record = {
        "system": system, "delta0": delta0, "channel": cfg["channel"],
        "gains": cfg["gains"], "note": cfg["note"], "weights_r": list(weights),
        "design_solver": design_solver, "design_epsilon": design_epsilon,
        "gate_solver": gate_solver, "exact_solver": exact_solver,
        "exact_fallback_solver": exact_fallback_solver,
        "strict": bool(strict), "parity_reduction_requested": bool(parity_reduction),
        "degrees": [], "certified": False,
    }
    for m in degrees:
        diag = enumerate_diagonal_terms(m, weights, c1_only=True)
        # μ from the NOMINAL dynamics is correct for the vertices too: the
        # constant ±δ₀ has integer (zero) exponents, and the vertex-W support
        # adds only ∂V/∂xₙ-shaped terms already present via the sign channel.
        mu, _, cross = mu_filtered_cross(m, weights, diag, dyn)
        terms = list(diag) + list(cross)
        print(f"[{system} δ₀={delta0:g}] m={m}: mu={mu}, library {len(diag)} "
              f"diag + {len(cross)} cross (complete slice)", flush=True)
        t0 = time.time()
        # design inline (not via _design_full_library) so the per-degree
        # solver status lands in the record — negatives must be reportable
        # as the exact MOSEK status (infeasibility certificate vs unknown).
        opt = JointSOSOptimizer(
            weights_r=weights, solver=design_solver, verbose=False,
            epsilon=design_epsilon,
            allow_solver_fallback=(design_solver != "MOSEK_DIRECT") and not strict,
            parity_reduction=parity_reduction,
        )
        deg = {"m": m, "library": {"diag": len(diag), "cross": len(cross)}}
        try:
            res = opt.find_lyapunov_with_dynamics(
                degree_m=m, dynamics_gf=dyn, active_terms=list(terms),
                dynamics_gf_vertices=spec.dynamics_gf_vertices,
            )
            V = res.V_gf if res.feasible else None
            deg["solver_status"] = str(getattr(res, "solver_status", ""))
            # Present only when --parity-reduction was passed, so records from
            # runs without the flag are byte-for-byte what they always were.
            if getattr(res, "parity_reduction", None) is not None:
                deg["parity_reduction"] = res.parity_reduction
        except Exception as e:
            # JointSOSOptimizer._solve_problem catches solver-side failures
            # (SolverError, etc.) internally and returns feasible=False with
            # solver_status set — a genuine solver-reported infeasibility
            # NEVER raises here. An exception escaping this call is therefore
            # an unexpected environment/programming fault (import error, OOM,
            # TypeError, ...), NOT a mathematical negative. Swallowing it as
            # "no feasible candidate" would let --expect-fail pass vacuously
            # on a crashed environment (WP-5 falsification), so mark it
            # distinctly and re-raise instead of continuing the degree ladder
            # — a strict/no-fallback solve that "would fall back" surfaces
            # here as a typed attempt (status=unexpected_error) and aborts
            # loudly, exactly as C19(c) requires, rather than continuing.
            deg["design_time_s"] = round(time.time() - t0, 2)
            deg["design"] = "ERROR (unexpected exception, NOT solver-reported)"
            deg["solver_status"] = f"error: {e!r}"
            deg["status"] = "oom" if isinstance(e, MemoryError) else "unexpected_error"
            record["degrees"].append(deg)
            record["design_errored"] = True
            print(f"[{system} δ₀={delta0:g}] m={m}: UNEXPECTED DESIGN ERROR "
                  f"(re-raising — this is NOT a genuine negative): {e!r}",
                  flush=True)
            raise
        deg["design_time_s"] = round(time.time() - t0, 2)
        if V is None:
            deg["design"] = "no feasible candidate (solver-reported)"
            deg["status"] = _classify_solver_status(deg["solver_status"])
            record["degrees"].append(deg)
            print(f"[{system} δ₀={delta0:g}] m={m}: design infeasible/errored "
                  f"(status: {deg['solver_status']})", flush=True)
            continue
        t1 = time.time()
        cert = certify_discovered_gf(
            V, weights, m, dyn, solver=gate_solver, strict_solver=True,
            dynamics_gf_vertices=spec.dynamics_gf_vertices,
            disturbance_bound=delta0,
        )
        gate_status = (getattr(cert, "solver_status", "") or "").lower()
        gate_message = str(getattr(cert, "message", "") or "")
        non_conclusive = ("solver_error", "unavailable", "illposed", "unknown", "timeout")
        if cert.surface_certified:
            typed_status = "certified"
        elif any(k in gate_status or k in gate_message.lower() for k in non_conclusive):
            typed_status = "solver_unknown"
        else:
            typed_status = "gate_rejected"
        deg.update(design="feasible", n_terms=len(V.terms),
                   certified=bool(cert.surface_certified),
                   gate_time_s=round(time.time() - t1, 2),
                   gate_message=gate_message[:300], status=typed_status)
        record["degrees"].append(deg)
        print(f"[{system} δ₀={delta0:g}] m={m}: {len(V.terms)} terms, "
              f"strict-{gate_solver} (vertex-aware) certified="
              f"{cert.surface_certified}", flush=True)
        if not cert.surface_certified:
            continue
        record["certified"] = True
        record["degree_used"] = m
        record["row"] = {
            "system": system, "certified": True, "degree_used": m,
            "n_terms": len(V.terms),
            "method": f"gf_sos_full_library_2003_gains_perturbed_delta{delta0:g}",
            "delta0": delta0, "channel": cfg["channel"],
            "V_repr": str(V),
            "V_terms": [
                {"coefficient": float(t.coefficient),
                 "exponents": [str(e) for e in t.exponents],
                 "term_types": list(t.term_types)}
                for t in V.terms
            ],
            "weights_r": list(weights),
            "gains": cfg["gains"],
        }
        if not skip_exact:
            record["exact_gate"] = exact_phase(V, spec, epsilon, exact_solver,
                                               exact_fallback_solver)
        break
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--delta0", type=float, default=1.0)
    ap.add_argument("--degrees", type=int, nargs="*", default=None)
    ap.add_argument("--design-solver", default=DEFAULT_DESIGN_SOLVER,
                    help="D, the design stage (default: %(default)s, the "
                         "direct MOSEK Task-API encoding flipped in on "
                         "2026-09-03 after the P2-06 gate). No certifying "
                         "float solve goes through CVXPY.")
    ap.add_argument("--gate-solver", default=DEFAULT_GATE_SOLVER,
                    help="strict float gate, no fallback chain (default: "
                         "%(default)s; inside verify_gf_sos_full that name is "
                         "the direct poly-SOS backend). Pass CLARABEL for the "
                         "historical G referee arm, which is provenance, NOT "
                         "a citability condition.")
    ap.add_argument("--exact-solver", default=WITNESS_PRIMARY_SOLVER,
                    help="X, the exact phase's PRIMARY witness re-solve "
                         "(default: %(default)s). The verdict is the exact "
                         "rational charge, never a solver status.")
    ap.add_argument("--exact-fallback-solver", default=WITNESS_FALLBACK_SOLVER,
                    help="the exact phase's single labelled FALLBACK witness "
                         "retry, used only when the primary witness fails or "
                         "leaves too little charge headroom (default: "
                         "%(default)s). Pass an empty string to disable it.")
    ap.add_argument("--epsilon", type=float, default=1e-6,
                    help="exact-phase re-solve margin")
    ap.add_argument("--design-epsilon", type=float, default=1e-4,
                    help="SOS margin for the design SDP (thin true margins "
                         "may need 1e-5/1e-6)")
    ap.add_argument("--ladder", type=float, nargs="*", default=None,
                    help="descending δ₀ fallbacks if the primary δ₀ fails")
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--expect-fail", action="store_true",
                    help="negative control: exit 2 with audit banner if "
                         "anything certifies")
    ap.add_argument("--strict", action="store_true",
                    help="forbid design-solve fallback for MOSEK/MOSEK_DIRECT "
                         "alike (MOSEK_DIRECT is already pinned unconditionally; "
                         "this extends the same guarantee to plain MOSEK); "
                         "without this flag, behavior is byte-for-byte unchanged")
    ap.add_argument("--parity-reduction", action="store_true",
                    help="S1: pose the DESIGN SDP over the even-parity "
                         "sub-library and the PAIRED (octant, endpoint) block "
                         "set, 2^(n-1) V blocks and 2^n of 2^(n+1) W blocks. "
                         "Prerequisites (odd nominal field, endpoint-swap "
                         "identity) are verified symbolically; on failure the "
                         "reduction refuses loudly and the full formulation "
                         "runs. The strict gate and the exact phase are NOT "
                         "reduced. Off by default; without it behavior is "
                         "byte-for-byte unchanged")
    add_out_dir_args(ap, DEFAULT_OUT_DIR)
    args = ap.parse_args()
    exact_fallback = args.exact_fallback_solver or None

    out_dir = resolve_out_dir(args.out_dir, DEFAULT_OUT_DIR,
                              allow_frozen=args.allow_frozen_root)
    degrees = args.degrees or SYSTEMS[args.system]["degrees"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "timestamp_utc": ts, "git_sha": _git_sha(),
        "expect_fail": bool(args.expect_fail), "strict": bool(args.strict),
        "parity_reduction": bool(args.parity_reduction),
        "out_dir": str(out_dir),
        "sweeps": [],
        "provenance": _provenance(sys.argv[1:], args.design_solver,
                                  args.gate_solver, args.exact_solver,
                                  exact_fallback),
    }

    # incremental record (long ladders can be OOM-killed mid-sweep; the
    # 2026-06-12 δ₀-ladder lost its record exactly this way)
    path = out_dir / f"{args.system}_delta{args.delta0:g}_run_{ts}.json"

    sweep = run_sweep(args.system, args.delta0, degrees, args.design_solver,
                      args.epsilon, args.skip_exact, args.design_epsilon,
                      args.strict, args.parity_reduction, args.gate_solver,
                      args.exact_solver, exact_fallback)
    result["sweeps"].append(sweep)
    path.write_text(json.dumps(result, indent=2))

    if args.expect_fail:
        path = out_dir / f"{args.system}_delta{args.delta0:g}_negcontrol_{ts}.json"
        path.write_text(json.dumps(result, indent=2))
        if sweep["certified"]:
            print("\n" + "!" * 78)
            print("!! NEGATIVE CONTROL CERTIFIED — THE PIPELINE PROVED A FALSE "
                  "STATEMENT.")
            print(f"!! δ₀ = {args.delta0:g} > sign gain: no certificate can "
                  "exist. STOP and audit")
            print("!! the vertex encoding before trusting ANY perturbed result.")
            print("!" * 78, flush=True)
            print(f"[written {path}]")
            sys.exit(2)
        print(f"\nnegative control behaved correctly: no certificate at any "
              f"degree (δ₀ = {args.delta0:g}).")
        print(f"[written {path}]", flush=True)
        return

    if not sweep["certified"] and args.ladder:
        for d0 in args.ladder:
            print(f"\nprimary δ₀ failed; ladder step δ₀ = {d0:g}", flush=True)
            sweep = run_sweep(args.system, d0, degrees, args.design_solver,
                              args.epsilon, args.skip_exact,
                              args.design_epsilon, args.strict,
                              args.parity_reduction, args.gate_solver,
                              args.exact_solver, exact_fallback)
            result["sweeps"].append(sweep)
            path.write_text(json.dumps(result, indent=2))
            if sweep["certified"]:
                break

    certified = [s for s in result["sweeps"] if s["certified"]]
    if certified:
        best = certified[0]
        row_path = out_dir / f"{args.system}_delta{best['delta0']:g}_row.json"
        row_path.write_text(json.dumps(best["row"], indent=2))
        print(f"[row written {row_path}]", flush=True)

    path.write_text(json.dumps(result, indent=2))
    print(f"[written {path}]", flush=True)


if __name__ == "__main__":
    main()
