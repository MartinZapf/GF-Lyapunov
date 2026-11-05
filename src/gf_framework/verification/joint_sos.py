from __future__ import annotations

"""
Joint V + W SOS Optimization via Semidefinite Programming.

This module implements the SOSTOOLS approach for finding Lyapunov coefficients
that satisfy BOTH V > 0 AND W > 0 (where W = -dV/dt) simultaneously.

Theoretical Foundation:
-----------------------
Based on Sanchez & Moreno (2019), Int J Robust Nonlinear Control 29(3):661-681.
Implements Algorithm 7-8 (SOS verification) and extends to joint optimization.

Key insight: Given a fixed term structure, both V and W are LINEAR in the
coefficient vector alpha. This makes the joint optimization a Semidefinite
Program (SDP), which can be solved efficiently.

SOS Formulation:
A polynomial p(y) is Sum-of-Squares (SOS) iff there exists a Positive
Semidefinite matrix Q such that:
    p(y) = z(y)^T Q z(y)
where z(y) is a vector of monomials up to degree ceil(deg(p)/2).

For Lyapunov verification, we require:
1. V(y; alpha) - epsilon * ||y||^d is SOS  (for each octant)
2. W(y; alpha) - epsilon * ||y||^d is SOS  (for each octant)
where d is the polynomial degree.

The constraint p(y) = z(y)^T Q z(y) expands to:
    coef_k(p) = sum_{i,j: deg(z_i) + deg(z_j) = k} Q_{ij}

Since coef_k(p) is LINEAR in alpha, this gives linear constraints on Q and alpha.
Combined with Q >= 0 (PSD), we get an SDP.

Framework Extension: Product-Monomial Clearing
----------------------------------------------
NOT in Sanchez-Moreno paper. For SMC systems, W = -dV/dt often has NEGATIVE
x-exponents (e.g., |x|^{-1/2}), making W non-polynomial in y-space.

Solution: Multiply W by clearing monomial C_k = ∏ᵢ |xᵢ|^{k·ρ/rᵢ} where:
- ρ = max(r) is the maximal weight
- k is chosen so C_k · W has all non-negative y-exponents

Mathematical validity:
- C_k > 0 for all x with all xᵢ ≠ 0 (positive on open orthants)
- C_k = 0 on coordinate hyperplanes {x : xᵢ = 0 for some i}
- For Filippov analysis (Lemma 5), W positivity on open orthants is sufficient
- For continuous W: if C_k·W > 0 on open orthants, then W ≥ 0 on hyperplanes
- C_k · W is polynomial in y-space (SOS-verifiable)

The certificate is established on open orthants. verify_hyperplane_continuity()
is an optional numerical sanity check near coordinate hyperplanes.

Key functions:
- analyze_w_exponents(): Detect negative exponents, compute clearing power k
- build_clearing_monomial_gf(): Construct the clearing monomial C_k
- _compute_w_coefficient_matrix_weighted_sphere(): Handle W with clearing
"""

from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterator, List, Literal, Optional,
                    Sequence, Tuple)
from fractions import Fraction
from math import gcd
import math
import hashlib
import json
import numpy as np
import sympy as sp
from itertools import combinations_with_replacement
import sys
import time

import os
from pathlib import Path

# Configure MOSEK license (portable): only set if user has not configured it.
if 'MOSEKLM_LICENSE_FILE' not in os.environ:
    repo_root = Path(__file__).resolve().parents[1]
    license_path = repo_root / 'mosek.lic'
    if license_path.exists():
        os.environ['MOSEKLM_LICENSE_FILE'] = str(license_path)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

from gf_framework.algebra.core import GeneralizedForm, GFTerm, lcm_list, build_weighted_norm_gf, build_clearing_monomial_gf
from gf_framework.algebra.enumerate import LIBRARY_ORDER, TermSpec, enumerate_all_terms
from gf_framework.algebra.associated import get_hyperoctant_signs, safe_to_int
from gf_framework.algebra.lie_derivative import multiply_gf_by_gf, multiply_gf_terms, collect_terms
from gf_framework.algebra import symmetry as _sym


#: Magnitude at or below which a float target coefficient is kept OUT of the
#: SDP's equality rows (review finding 6, 2026-09-05). It is a solver-encoding
#: floor and NOT an acceptance tolerance: every coefficient dropped by it is
#: charged, in ‖·‖₁, to the a-posteriori residual of
#: ``_validate_gram_certificate``, so the validated Gram is always validated
#: against the TRUE target. Nothing is ever silently discarded.
SOLVER_COEFF_FLOOR = 1e-15

# =============================================================================
# Design-time parity reduction (campaign block S1) — OPT-IN, DEFAULT OFF
# =============================================================================
#
# The reduction is a change of FORMULATION at design time only
# (``docs/papers/theory_notes.md`` §10.6): the same V is proposed, but over the
# even-parity sub-library and over one Gram block per orbit of the reflection
# x ↦ −x. Nothing downstream of the designer is reduced — the strict gate
# (``certify_discovered_gf`` → ``verify_gf_sos_full``) and the exact gate always
# verify the FULL 2ⁿ (nominal) / 2ⁿ⁺¹ (two-endpoint) block set. A certificate
# produced from a reduced design that fails an unreduced gate disqualifies the
# module (campaign gate S1).
#
# Default is OFF everywhere. With ``parity_reduction`` unset every code path is
# the pre-existing one, unchanged.

_PARITY_REDUCTION_DEFAULT = False


def set_parity_reduction_default(enabled: bool) -> bool:
    """Set the process-wide default for the design-time parity reduction.

    Exists so a CLI that does not construct the optimizer itself (e.g.
    ``scripts/discover_gf.py``, which goes through
    ``discovery.gf_discovery.discover_gf``) can still opt in. Returns the
    previous value so callers/tests can restore it.

    An explicit ``parity_reduction=`` argument to :class:`JointSOSOptimizer`
    always wins over this default.
    """
    global _PARITY_REDUCTION_DEFAULT
    previous = _PARITY_REDUCTION_DEFAULT
    _PARITY_REDUCTION_DEFAULT = bool(enabled)
    return previous


def get_parity_reduction_default() -> bool:
    """Current process-wide default for the design-time parity reduction."""
    return _PARITY_REDUCTION_DEFAULT


# ---------------------------------------------------------------------------
# WP P5-04: y-parity Gram BLOCK decomposition.
#
# Orthogonal to the octant/library parity reduction above. That one drops
# blocks and library terms; this one keeps every block and every term and only
# restricts each per-octant Gram to be block-diagonal over the PARITY CLASSES
# of its own monomial basis. The design targets live on the even lattice
# (2Z)^n, so each target is invariant under the sign-flip group G = (Z/2)^n
# acting on y; averaging any PSD Gram over G gives a PSD Gram of the same
# polynomial with every cross-class entry zero. The restriction is therefore
# LOSSLESS in α (see ``verification/yparity.py``), not a relaxation: soundness
# because a block-diagonal Gram is a Gram, completeness because the projection
# of any dense solution is block-diagonal.
#
# S1 discipline is unchanged and non-negotiable: a design produced with blocked
# Grams is still re-gated on the UNREDUCED strict gate and the UNREDUCED exact
# check (``artifacts/campaign_2026-08/S1_validation/README.md``).
#
# The evenness premise is checked SYMBOLICALLY before any float assembly by
# ``algebra.support_guard`` (typed ``OddLiftRefusal``), and again on the
# assembled rows by ``yparity.split_gram_block`` (typed ``YParityRefused``).
# Either refusal falls the whole design back to the dense encoding and is
# recorded; nothing is ever silently blocked.

_Y_PARITY_BLOCKS_DEFAULT = True


def set_y_parity_blocks_default(enabled: bool) -> bool:
    """Set the process-wide default for the y-parity Gram block decomposition.

    Returns the previous value. An explicit ``y_parity_blocks=`` argument to
    :class:`JointSOSOptimizer` always wins over this default.
    """
    global _Y_PARITY_BLOCKS_DEFAULT
    previous = _Y_PARITY_BLOCKS_DEFAULT
    _Y_PARITY_BLOCKS_DEFAULT = bool(enabled)
    return previous


def get_y_parity_blocks_default() -> bool:
    """Current process-wide default for the y-parity block decomposition."""
    return _Y_PARITY_BLOCKS_DEFAULT


# ---------------------------------------------------------------------------
# WP P5-02: the DESIGN-EPSILON SWEEP.
#
# The design margin epsilon is a knob of the SDP, not a property of the system,
# and the escalation loop of ``discovery.gf_discovery.discover_gf`` used to
# hold it fixed at 1e-4 and escalate the DEGREE instead. That confounds two
# different failures: "this degree cannot carry a certificate" and "this margin
# cannot be met at this degree". A design that is primal-infeasible at 1e-4 can
# be feasible at 1e-6 (the margin is a real tightening of the feasible set: the
# SDP must find a strict margin epsilon on every pure-power axis), so escalating
# m on an infeasible 1e-4 solve can walk straight past a lower-degree
# certificate. The sweep tries the ladder below at the CURRENT degree before
# escalating: two extra solves per degree, and none of them at all once a rung
# certifies.
#
# The sweep is paired with the V scale normalisation of
# ``verification/v_normalization.py``, because the two failures it separates are
# coupled: a design at a small epsilon puts its smallest pure-power coefficient
# at about that epsilon, and the strict gate's margin is ABSOLUTE.
#
# Nothing here weakens a verdict. Every rung's proposal is re-gated by the same
# strict ``certify_discovered_gf`` and re-proved by the same exact phase; the
# sweep only changes which candidates get offered.

#: The ladder WP P5-02 measured, coarsest first. The first entry is the shipped
#: design margin, so a swept loop's first rung is byte-identical to the
#: unswept one.
DESIGN_EPSILON_LADDER: Tuple[float, ...] = (1e-4, 1e-6, 1e-8)

#: Process-wide default for the sweep. ``None`` = no sweep (the shipped
#: behaviour: one design per degree, at ``JointSOSOptimizer.epsilon``).
_DESIGN_EPSILON_SWEEP_DEFAULT: Optional[Tuple[float, ...]] = None


def set_design_epsilon_sweep_default(
    ladder: Optional[Sequence[float]],
) -> Optional[Tuple[float, ...]]:
    """Set the process-wide default design-epsilon ladder. Returns the previous.

    ``None`` restores the unswept behaviour. An explicit ``epsilon_ladder=``
    argument to :func:`discovery.gf_discovery.discover_gf` always wins.
    """
    global _DESIGN_EPSILON_SWEEP_DEFAULT
    previous = _DESIGN_EPSILON_SWEEP_DEFAULT
    _DESIGN_EPSILON_SWEEP_DEFAULT = (
        None if ladder is None else resolve_epsilon_ladder(ladder))
    return previous


def get_design_epsilon_sweep_default() -> Optional[Tuple[float, ...]]:
    """Current process-wide default design-epsilon ladder (``None`` = off)."""
    return _DESIGN_EPSILON_SWEEP_DEFAULT


def resolve_epsilon_ladder(ladder: Sequence[float]) -> Tuple[float, ...]:
    """Validate and canonicalize a design-epsilon ladder.

    Every entry must be finite and strictly positive (a zero or negative
    design margin is not a margin), and duplicates are dropped keeping the
    first occurrence, so a caller cannot pay for the same solve twice. Order is
    the caller's: it is the order the rungs are tried in.
    """
    out: List[float] = []
    seen: set = set()
    for raw in ladder:
        value = float(raw)
        if not (value > 0.0) or not math.isfinite(value):
            raise ValueError(
                f"design epsilon ladder entries must be finite and > 0, got {raw!r}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError("design epsilon ladder must have at least one entry")
    return tuple(out)


@dataclass
class EpsilonRung:
    """One rung of the design-epsilon sweep, as handed to the gate (P5-02).

    ``V_gf`` is ``None`` when the design produced no proposal at all. It is NOT
    None when the design screen refused the result: the screen is a filter on
    the BUNDLE, never on the candidate, so the candidate is still offered to
    the gate and ``bundle_reason`` records that its Grams may not be a witness.
    A consumer that gates such a candidate successfully must route the exact
    phase to a re-solve witness and say so; it must never hand X a bundle the
    screen refused.
    """

    index: int = 0
    epsilon: float = 0.0
    #: The design margin AFTER the V normalisation, i.e. the margin the
    #: returned bundle witnesses and the one the exact phase must be given.
    epsilon_effective: float = 0.0
    result: Optional["JointSOSResult"] = None
    V_gf: Optional[GeneralizedForm] = None
    feasible: bool = False
    screen_refused: bool = False
    solver_status: str = ""
    message: str = ""
    seconds: float = 0.0
    gram_certificates: Optional[List[dict]] = None
    bundle_reason: str = ""
    normalization: Optional[dict] = None
    #: Full traceback when ``solver_status == "design_error"``. An unexpected
    #: exception in a design solve is not an SOS negative and must never be
    #: reported as one (the rule ``_design_full_library`` already states).
    traceback: str = ""

    def as_record(self) -> dict:
        """JSON-ready record of the rung (no Gram data, no V object)."""
        return {
            "index": int(self.index),
            "epsilon": float(self.epsilon),
            "epsilon_effective": float(self.epsilon_effective),
            "feasible": bool(self.feasible),
            "screen_refused": bool(self.screen_refused),
            "has_candidate": self.V_gf is not None,
            "solver_status": str(self.solver_status),
            "message": str(self.message),
            "seconds": float(self.seconds),
            "n_bundle_entries": (0 if self.gram_certificates is None
                                 else len(self.gram_certificates)),
            "bundle_reason": str(self.bundle_reason),
            "normalization": self.normalization,
            "traceback": str(self.traceback),
        }


#: Every way a rung's design Gram bundle can fail to be a witness, beyond the
#: reasons ``gf_discovery._adopt_design_bundle`` already types. The screen
#: refusal is the one WP P5-02 adds: it is a real, measured outcome on the
#: lower rungs (NEGATIVES N20) and it must route to a typed re-solve, never to
#: a silently dropped bundle.
EPSILON_RUNG_BUNDLE_REASONS = (
    "",
    "design_screen_refused",   # suffixed with the screen's own detail
    "no_design_grams",
)


#: Every way the y-parity blocking can be declined. Empty string = applied.
Y_PARITY_REFUSAL_CODES = (
    "not_requested",          # the option is off
    "odd_lift_v",             # symbolic guard: a V-stage lift is odd
    "odd_lift_w",             # symbolic guard: a W-stage lift is odd
    "guard_unavailable",      # dynamics objects the symbolic guard cannot read
    "assembly_refused_v",     # yparity runtime guard on the assembled V rows
    "assembly_refused_w",     # yparity runtime guard on the assembled W rows
)


class ParityReductionRefused(RuntimeError):
    """The parity reduction was requested but may not be applied.

    Carries a machine-readable ``reason_code`` (one of
    :data:`PARITY_REFUSAL_CODES`) alongside the human-readable message. The
    designer never propagates this to the caller: it catches it, logs it, runs
    the FULL unreduced formulation, and records
    ``parity_reduction = {"mode": "refused", "reason": ..., "reason_code": ...}``
    in the result.
    """

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.reason = str(message)


#: Every way the reduction can be refused. Kept explicit so a run JSON can be
#: audited without parsing free text.
PARITY_REFUSAL_CODES = (
    "dynamics_not_odd",          # (A1) fails: f(−x) ≠ −f(x) symbolically
    "endpoint_swap_failed",      # Lemma 10.2 fails on the given vertices
    "endpoint_swap_unverified",  # endpoints present but not verifiable objects
    "endpoint_count_unsupported",  # ≠ 1 and ≠ 2 vertices (no proved pairing)
    "asymmetric_constraints",    # (A3) at risk: per-term constraint not R-invariant
    "empty_even_library",        # nothing left to design over
    "diagonal_backbone_changed",  # the PD backbone did not survive the filter
    "library_split_inconsistent",  # filter self-check failed
    "reduction_inconsistent",    # involution/partition self-check failed
    "unexpected_error",          # anything else — never silently reduced
)


def analyze_w_exponents(
    w_terms: List[GFTerm],
    mu: Tuple[int, ...]
) -> Dict:
    """
    Analyze W terms for negative y-exponents.

    FRAMEWORK EXTENSION (not in Sanchez-Moreno 2019):
    This function enables SOS verification for W = -dV/dt with negative exponents,
    which is common in SMC systems but not addressed in the original paper.

    For SMC systems like STA, W = -dV/dt often has negative x-exponents
    (e.g., |x|^{-1/2}), which become negative y-exponents in the lifted
    y-space (e.g., y^{-2}). This makes W non-polynomial.

    The solution is to multiply W by ||x||_r^k to clear all negative exponents.
    This preserves the sign of W (since ||x||_r > 0 for x ≠ 0) while making
    the product polynomial and thus SOS-verifiable.

    Args:
        w_terms: List of GF terms in W
        mu: μ exponents for y-transform (y_i = |x_i|^{1/μ_i})

    Returns:
        dict with:
        - 'has_negative': bool - whether any term has negative y-exponents
        - 'min_y_exp': minimum (most negative) y-exponent across all terms
        - 'clearing_power': minimum k to clear all negative exponents
        - 'negative_terms': list of terms with negative exponents
    """
    min_y_exp = 0
    negative_terms = []

    for term in w_terms:
        term_has_negative = False
        for i, exp in enumerate(term.exponents):
            y_exp = mu[i] * float(exp)
            if y_exp < min_y_exp:
                min_y_exp = y_exp
            if y_exp < 0:
                term_has_negative = True

        if term_has_negative:
            negative_terms.append(term)

    # Calculate clearing power k such that ||x||_r^k · W has all non-negative exponents
    # ||x||_r = Σᵢ |xᵢ|^{ρ/rᵢ} where ρ = max(r)
    # In y-space: ||y||_r = Σᵢ yᵢ^{μᵢ·ρ/rᵢ} = Σᵢ yᵢ^{μᵢ} (since μᵢ = k·rᵢ and ρ = max(r))
    # Each term of ||x||_r^k contributes at least +min(μ) to the y-exponent
    # Need: k * min(μ) >= |min_y_exp|
    if min_y_exp < 0:
        min_mu = min(mu)
        clearing_power = int(np.ceil(-min_y_exp / min_mu))
    else:
        clearing_power = 0

    return {
        'has_negative': min_y_exp < 0,
        'min_y_exp': min_y_exp,
        'clearing_power': clearing_power,
        'negative_terms': negative_terms
    }


def verify_hyperplane_continuity(
    W_gf: GeneralizedForm,
    weights_r: Tuple[int, ...],
    n_samples: int = 1000,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Check W behavior approaching coordinate hyperplanes.

    Numerical sanity check for W behavior near coordinate hyperplanes.

    The SOS certificate itself is built on open orthants (consistent with Lemma 5
    for Filippov trajectories). This helper checks sampled continuity behavior near
    x_i = 0 boundaries for additional practical confidence.

    Args:
        W_gf: The W = -dV/dt generalized form
        weights_r: Homogeneity weights
        n_samples: Number of sample points per hyperplane/epsilon combination
        verbose: Print diagnostics

    Returns:
        Tuple of (passed, message)
    """
    n_vars = len(weights_r)
    epsilons = [1e-3, 1e-6, 1e-9]
    rng = np.random.default_rng(42)

    for i in range(n_vars):
        for epsilon in epsilons:
            # Generate points approaching hyperplane x_i = 0
            # Sample (n_samples) points where x_i = ±epsilon, others random
            X = rng.standard_normal((n_samples, n_vars))

            # Normalize other coordinates to unit weighted sphere
            rho = max(weights_r)
            for j in range(n_vars):
                if j != i:
                    # Scale to keep weighted norm reasonable
                    X[:, j] = np.sign(X[:, j]) * np.abs(X[:, j]) ** (weights_r[j] / rho)

            # Set coordinate i to small epsilon with random sign
            X[:, i] = np.sign(rng.standard_normal(n_samples)) * epsilon

            try:
                W_vals = W_gf.evaluate(X)
                min_val = W_vals.min()

                if min_val < -1e-6:
                    msg = f"W negative ({min_val:.2e}) approaching x_{i}=0 hyperplane (eps={epsilon})"
                    if verbose:
                        print(f"  Hyperplane check FAILED: {msg}")
                    return False, msg
            except Exception as e:
                # Numerical issues near hyperplane - not necessarily a failure
                if verbose:
                    print(f"  Hyperplane check: numerical issue at x_{i}=0, eps={epsilon}: {e}")

    return True, "W continuous to hyperplanes"


@dataclass
class SOSDualInfo:
    """Diagnostic information from SOS solver for infeasible problems."""
    # Which term indices have constraint issues
    blocking_term_indices: List[int] = None
    # Constraint dual values (higher = harder to satisfy)
    constraint_duals: Dict[str, float] = None
    # Which octants are most problematic
    problematic_octants: List[int] = None
    # Raw solver info
    solver_info: Dict = None
    # Suggestion for which terms to swap
    suggested_swaps: List[Tuple[int, int]] = None  # (term_idx, pair_idx)

    def __post_init__(self):
        if self.blocking_term_indices is None:
            self.blocking_term_indices = []
        if self.constraint_duals is None:
            self.constraint_duals = {}
        if self.problematic_octants is None:
            self.problematic_octants = []
        if self.solver_info is None:
            self.solver_info = {}
        if self.suggested_swaps is None:
            self.suggested_swaps = []


# =============================================================================
# Design objectives (WP P5-05)
# =============================================================================
#
# The design SDP is feasibility-only by default: every point of the feasible
# set is equally good to the objective, so an interior-point solver returns an
# analytic-centre-like point, whose Gram has the MAXIMUM rank compatible with
# the constraints. ``min_gram_trace`` adds Σ_blocks tr Q, the nuclear norm of a
# PSD block, which is the standard convex surrogate for rank and moves the
# solve onto a low-rank face of the cone.
#
# It is opt-in. ``"auto"`` is the default and reproduces the historical
# behaviour bit for bit; nothing in the repository selects anything else.
DESIGN_OBJECTIVES = ("auto", "min_gram_trace")


def resolve_design_objective(objective: Optional[str]) -> str:
    """Validate a design-objective name, defaulting ``None`` to ``"auto"``."""
    name = "auto" if objective is None else str(objective).strip().lower()
    if name not in DESIGN_OBJECTIVES:
        raise ValueError(
            f"unknown design objective {objective!r}; "
            f"expected one of {', '.join(DESIGN_OBJECTIVES)}")
    return name


@dataclass
class JointSOSResult:
    """Result of joint V+W SOS optimization."""
    feasible: bool
    coefficients: Optional[Dict[str, float]] = None
    V_gf: Optional[GeneralizedForm] = None
    W_gf: Optional[GeneralizedForm] = None
    solver_status: str = ""
    solver_requested: str = ""
    solver_used: str = ""
    solver_fallback_used: bool = False
    solver_attempts: Optional[List[str]] = None
    message: str = ""
    solve_time: float = 0.0
    # NEW: Dual information for infeasible problems
    dual_info: Optional[SOSDualInfo] = None
    # NEW: Term specs used (for analysis)
    term_specs: Optional[List] = None
    # NEW: exact rational SOS certificates (one per octant/sub-problem) and the
    # strict-positivity shift used. Present only when feasible via the exact path.
    exact_certificates: Optional[List] = None
    epsilon_used: Optional[str] = None
    # A-posteriori Gram validation diagnostics (worst octant of the accepted
    # ε-rung): smallest eigenvalue of the recovered Gram, the ℓ1 coefficient
    # residual ‖target − zᵀQz‖₁, and the Gram condition number λ_max/λ_min
    # (None if a Gram was not strictly PD). See _validate_gram_certificate.
    gram_min_eig: Optional[float] = None
    gram_residual_l1: Optional[float] = None
    gram_cond: Optional[float] = None
    # Per-block Gram certificates. Two producers, one consumed core (see
    # GRAM_BUNDLE_SCHEMA below):
    #   * verify_gf_sos_full(..., capture_grams=True) — the strict VERIFICATION
    #     stage, one entry per octant of the accepted ε-rung (default off);
    #   * find_lyapunov_with_dynamics with solver="MOSEK_DIRECT" — the DESIGN
    #     stage, one entry per emitted Gram block (V and W), on by default and
    #     switchable with JointSOSOptimizer(capture_design_grams=False).
    # Every entry carries {signs, octant_index, eps, mu, y_degree, target,
    # basis, gram}, which is what the offline exact-rational gate
    # (verification.exact_gate.exact_verify_captures / exact_phase /
    # review/reverify_exact.py) consumes. Design-stage entries additionally
    # carry the block identity and a content hash — see GRAM_BUNDLE_SCHEMA.
    gram_certificates: Optional[List[dict]] = None
    # Design-time parity reduction record (campaign block S1). ``None`` when the
    # reduction was not requested — a run without the flag carries NO parity
    # metadata at all, so its results are indistinguishable from before the
    # module existed. When requested it is a dict with at least
    # ``{"mode": "nominal"|"paired"|"refused", ...}``; see
    # ``JointSOSOptimizer._plan_parity_reduction`` for the full schema. The
    # reduction is a DESIGN-side formulation change only: the emitted V is a
    # complete, ordinary candidate and is re-gated UNREDUCED.
    parity_reduction: Optional[dict] = None
    # Fail-fast design screen (WP P2-04, decision D1): the sphere pre-check on
    # V and the a-posteriori Gram validation on every bundle entry, run on the
    # packaged MOSEK_DIRECT design result. ``DesignScreen.as_record()``, or
    # ``None`` when the screen did not run (other backend, mode "off", or a
    # result that was never packaged). It carries per-block (per-octant)
    # ``gram_min_eig`` / ``gram_residual_l1``. The screen can only REJECT: it
    # is a filter on the design output, never the proof — see
    # ``verification.design_screen``.
    design_screen: Optional[dict] = None
    # Which design objective produced this result (WP P5-05): ``"auto"`` (the
    # default and the historical behaviour) or ``"min_gram_trace"``. Set on the
    # packaged DESIGN result only; ``verify_gf_sos_full`` and every non-design
    # producer leave it at ``"auto"``. Provenance only — no consumer branches
    # on it, and it is NOT part of the Gram bundle hash.
    design_objective: str = "auto"
    # y-parity Gram block decomposition record (WP P5-04). ``None`` when the
    # option was off. Otherwise a dict
    # ``{"requested", "v_blocked", "w_blocked", "refusal", "detail",
    #    "psd_scalars_dense", "psd_scalars_blocked", "n_psd_blocks"}``.
    # Design-side encoding only: the bundle entries stay DENSE (block-diagonal
    # Grams reassembled onto the full basis) and every gate downstream is
    # unreduced, exactly as for ``parity_reduction``.
    y_parity: Optional[dict] = None

    def __post_init__(self):
        if self.solver_attempts is None:
            self.solver_attempts = []


# =============================================================================
# Gram bundle schema (design stage) — the proof object X consumes
# =============================================================================
#
# The DESIGN solve already produces the object the exact-rational phase needs:
# ``mosek_direct.DirectSolveOutcome.grams`` holds one recovered Gram per emitted
# PSD block. Persisting it removes the re-solve that used to stand between
# design and the exact re-proof (toolchain D -> X -> P).
#
# One entry per Gram block, as a plain dict:
#
#   consumed by exact_gate.exact_verify_captures (same keys as the
#   verify_gf_sos_full captures, so both producers feed the same consumer):
#     "signs"          tuple[int, ...]     octant signs in {-1, +1}^n
#     "octant_index"   int                 FULL octant index (matches the name)
#     "eps"            float               TRUE-scale margin subtracted from the
#                                          target. ε for V; for W it is ε too
#                                          when the weighted-sphere clearing is
#                                          inactive (P2-02), and 0.0 when it is
#                                          active — the cleared W vanishes on
#                                          the axes, so no axis margin exists
#     "mu"             tuple[int, ...]     y-lift exponents the block was built at
#     "y_degree"       int                 total y-degree of the block's target
#     "target"         dict[tuple, float]  TRUE-scale target coefficients, ε
#                                          already subtracted
#     "basis"          list[tuple[int,...]] Gram basis, in Gram row/column ORDER
#     "gram"           np.ndarray (d, d)   TRUE-scale symmetric Gram, float64
#
#   block identity and provenance (design stage only):
#     "schema"             str    GRAM_BUNDLE_SCHEMA
#     "block"              str    "V" or "W"
#     "name"               str    the MOSEK block name ("V_oct3", "W_oct3_v0")
#     "endpoint"           int|None  inclusion-vertex index for W, None for V
#     "parity_class"       dict|None  None unless the design-time parity
#                          reduction was ACTIVE; then
#                          {"mode": "nominal"|"paired", "role": "representative",
#                           "partner_octant_index": int,
#                           "partner_endpoint": int|None}
#     "clearing_power"     int    weighted-sphere clearing k (0 for V)
#     "eps_reason"         str    "" when the block carries its ε rows, else
#                          the typed reason it carries none:
#                          "weighted_sphere_clearing_active" (the W block is
#                          ∏ᵢ|xᵢ|^{k·ρ/rᵢ}·W and vanishes on the axes),
#                          "w_epsilon_rows_disabled" (opt-out), or a
#                          "w_degree_not_positive_even"/
#                          "pure_power_rows_incomplete" structural refusal.
#                          A consumer routes ``eps == 0.0`` to a re-solve and
#                          reports THIS as the reason, never a bare
#                          ``epsilon_not_positive``. Diagnostic only: it is
#                          derivable from the hashed ``eps`` and
#                          ``clearing_power``, so it is NOT part of the hash.
#     "mu_policy"          str    "" when the block was built at the design μ
#                          (every V block, and every W block whose associated
#                          form already had an even y-degree), or
#                          "w_mu_doubled_odd_y_degree" when the W block's μ was
#                          doubled because its y-degree came out odd (WP P2-09,
#                          ``power_rate``). The block's actual μ is the hashed
#                          ``mu`` key, which every consumer already rebuilds its
#                          exact target at, so this is diagnostic only and NOT
#                          part of the hash.
#     "library_order"      str    the canonical GF-library ordering the design
#                          was built under, ``algebra.enumerate.LIBRARY_ORDER``
#                          ("grlex-v1" since WP P5-13: graded-lex on the exact
#                          exponent tuple, term_types as the tie-break, applied
#                          to the CROSS terms only — the diagonals keep their
#                          axis order because the designer pins
#                          ``alpha[diag_indices[0]]``). Before P5-13 the cross
#                          library came out in set-iteration order, so a design
#                          point was not reproducible across processes without
#                          ``PYTHONHASHSEED=0`` (P2-06 part C: 6 of 12 systems
#                          moved). Diagnostic only: the Gram, basis and target
#                          are already stated in absolute monomials and carry no
#                          library index, so this is NOT part of the hash.
#     "y_parity"           dict   how the DESIGN posed this block (WP P5-04):
#                          {"mode": "blocked"|"dense", "n_classes", "classes"
#                           [[parity key, sub-dim], ...], "dense_dim",
#                           "psd_scalars_dense", "psd_scalars_blocked",
#                           "n_dropped_odd_rows", "n_dropped_cross_pairs"} and,
#                          on a dense block, "reason" (the typed y-parity
#                          refusal, "" when blocking was simply not requested
#                          — see ``Y_PARITY_REFUSAL_CODES``). A BLOCKED design
#                          restricts each Gram to be block-diagonal over the
#                          parity classes of its own basis, which is lossless
#                          in α by group averaging; the sub-Grams are
#                          reassembled onto the full basis before they are
#                          stored here, so the entry — basis, dim, gram,
#                          target, hash — is the same object either way.
#                          Diagnostic only, NOT part of the hash.
#     "coefficient_scale"  float  the scale divided out of gram and target
#     "dim"                int    Gram dimension (== len(basis))
#     "dtype"              str    "float64"
#     "hash"               str    sha256, see gram_bundle_entry_hash
#
# SCALE. The SDP is solved on ``alpha_scaled = coefficient_scale * alpha_true``.
# Both the Gram and the target are divided by ``coefficient_scale`` before they
# are stored, so a bundle entry is stated in the same TRUE scale as ``V_gf`` and
# ``epsilon``: ``target == M·alpha_true − eps·(pure powers)`` and
# ``zᵀ gram z ≈ target``. PSD-ness and the charge/budget ratio are invariant
# under that positive rescaling.
#
# COVERAGE. A design run with the parity reduction ACTIVE emits only the
# representative blocks (2^(n-1) V blocks), so the bundle deliberately does NOT
# cover all 2^n octants; ``exact_verify_captures`` will reject it on coverage.
# ``parity_class`` is what tells a consumer that this is expected.

#: Version tag of the entry schema above. Bump on any incompatible change.
GRAM_BUNDLE_SCHEMA: str = "gf.design_grams/1"

#: Metadata keys that enter the per-entry content hash. ``gram`` is hashed
#: separately from its raw bytes; keys outside this tuple (e.g. ``hash``
#: itself) are not covered.
GRAM_BUNDLE_HASHED_KEYS: Tuple[str, ...] = (
    "schema", "block", "name", "signs", "octant_index", "endpoint",
    "parity_class", "eps", "mu", "y_degree", "clearing_power",
    "coefficient_scale", "dim", "dtype", "basis", "target",
)


def _canonical_hash_value(value: Any) -> Any:
    """JSON-ready, order-stable rendering of one bundle metadata value.

    Tuples become lists, numpy scalars become Python scalars, and a dict keyed
    by monomial tuples (``target``) becomes a list of ``[exponents, value]``
    pairs sorted by exponent so the encoding does not depend on insertion
    order.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, dict):
        if value and all(isinstance(k, tuple) for k in value):
            return [
                [list(k), _canonical_hash_value(v)]
                for k, v in sorted(value.items())
            ]
        return {str(k): _canonical_hash_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical_hash_value(v) for v in value]
    return str(value)


def gram_bundle_entry_hash(entry: Dict[str, Any]) -> str:
    """Content hash of one Gram bundle entry (sha256, hex).

    Recipe, so an independent implementation can reproduce it:

    1. Take ``GRAM_BUNDLE_HASHED_KEYS`` in that order, rendering each value
       through :func:`_canonical_hash_value` (tuples -> lists; ``target`` ->
       a list of ``[exponents, coefficient]`` pairs sorted by exponent).
    2. ``json.dumps(payload, sort_keys=True, separators=(",", ":"),
       ensure_ascii=True)``, encoded UTF-8.
    3. Append a single ``b"\\x00"`` separator, then the Gram as
       little-endian float64 C-order bytes (``astype("<f8").tobytes()``).

    Missing keys are hashed as ``null``, so the digest is defined for entries
    from any producer. The digest covers everything a consumer would verify
    against, which is what lets a later stage refuse an input bundle whose
    hash does not match the one recorded with it.
    """
    payload = {k: _canonical_hash_value(entry.get(k)) for k in GRAM_BUNDLE_HASHED_KEYS}
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True).encode("utf-8"))
    h.update(b"\x00")
    gram = entry.get("gram")
    if gram is not None:
        h.update(np.ascontiguousarray(
            np.asarray(gram, dtype=np.float64)).astype("<f8").tobytes())
    return h.hexdigest()


def gram_bundle_fingerprint(entries: Optional[Sequence[Dict[str, Any]]]) -> str:
    """One hash over a whole bundle: sha256 of the per-entry hashes, in order.

    Order matters (it is the block emission order). ``None``/empty gives the
    digest of the empty byte string, so the value is always well defined.
    """
    h = hashlib.sha256()
    for entry in entries or []:
        digest = entry.get("hash") or gram_bundle_entry_hash(entry)
        h.update(str(digest).encode("ascii"))
    return h.hexdigest()


def _normalize_solver_name(solver: str) -> str:
    """Normalize a solver name to uppercase with sane default."""
    candidate = str(solver).strip().upper()
    return candidate if candidate else "SCS"


def _validate_diagonal_coefficients_true_scale(
    term_specs: List[TermSpec],
    coefficients_true: np.ndarray,
    epsilon: float,
    *,
    positivity_tol: float = 1e-12,
    margin_rel_tol: float = 1e-3,
) -> Tuple[bool, str]:
    """
    Sanity-check diagonal coefficients after solving the joint SOS SDP.

    The full JointSOS path solves on scaled coefficients and relies heavily on SOS
    margin constraints. This helper defensively validates the recovered true-scale
    coefficients before a feasible result is accepted.
    """
    diag_indices = [i for i, spec in enumerate(term_specs) if spec.is_diagonal()]
    if not diag_indices:
        return False, "no diagonal terms present in term_specs"

    coeffs = np.asarray(coefficients_true, dtype=np.float64).reshape(-1)
    margin_tol = max(float(positivity_tol), abs(float(epsilon)) * float(margin_rel_tol))
    bad_nonfinite: List[Tuple[int, float]] = []
    bad_nonpositive: List[Tuple[int, float]] = []
    bad_below_margin: List[Tuple[int, float]] = []

    for idx in diag_indices:
        val = float(coeffs[idx])
        if not np.isfinite(val):
            bad_nonfinite.append((idx, val))
            continue
        if val <= float(positivity_tol):
            bad_nonpositive.append((idx, val))
            continue
        if val + margin_tol < float(epsilon):
            bad_below_margin.append((idx, val))

    if bad_nonfinite or bad_nonpositive or bad_below_margin:
        details: List[str] = []
        if bad_nonfinite:
            details.append(
                "non-finite diagonals "
                + ", ".join(f"alpha{idx}={val!r}" for idx, val in bad_nonfinite)
            )
        if bad_nonpositive:
            details.append(
                "non-positive diagonals "
                + ", ".join(f"alpha{idx}={val:.6g}" for idx, val in bad_nonpositive)
            )
        if bad_below_margin:
            details.append(
                "diagonals below epsilon "
                + ", ".join(f"alpha{idx}={val:.6g}" for idx, val in bad_below_margin)
                + f" (epsilon={float(epsilon):.6g})"
            )
        return False, "; ".join(details)

    return True, ""


def _solver_candidates(
    preferred: str,
    strict_solver: bool,
    allow_solver_fallback: bool,
) -> List[str]:
    """Build ordered solver candidates according to strict/fallback policy."""
    primary = _normalize_solver_name(preferred)
    if strict_solver or not allow_solver_fallback:
        return [primary]

    chain = [primary, "MOSEK", "CLARABEL", "SCS"]
    seen = set()
    ordered: List[str] = []
    for solver_name in chain:
        if solver_name not in seen:
            ordered.append(solver_name)
            seen.add(solver_name)
    return ordered


def get_monomial_basis(n_vars: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Generate monomial basis up to given degree.

    For SOS decomposition p(y) = z(y)^T Q z(y), we need monomials
    up to degree ceil(deg(p)/2).
    """
    monomials = []
    for deg in range(max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), deg):
            exp = [0] * n_vars
            for idx in combo:
                exp[idx] += 1
            monomials.append(tuple(exp))
    return sorted(set(monomials))


def get_homogeneous_monomial_basis(n_vars: int, degree: int) -> List[Tuple[int, ...]]:
    """Monomial basis of total degree EXACTLY ``degree`` (for homogeneous SOS).

    A homogeneous polynomial of even degree ``2h`` is SOS iff it is ``z(y)ᵀ Q z(y)``
    with ``z`` ranging over monomials of total degree exactly ``h`` (the cross-degree
    blocks of a full ``0..h`` basis can never match a purely degree-``2h`` target, so
    they only inflate the SDP). The associated forms of a generalized-homogeneous GF
    are homogeneous, so this is the right — and far smaller — basis: in 2 vars it is
    ``degree+1`` monomials instead of the ``O(degree²)`` of the full basis.
    """
    monomials = []
    for combo in combinations_with_replacement(range(n_vars), degree):
        exp = [0] * n_vars
        for idx in combo:
            exp[idx] += 1
        monomials.append(tuple(exp))
    return sorted(set(monomials))


def _sos_gram_basis(n_vars: int, y_degree: int, homogeneous: bool = True) -> List[Tuple[int, ...]]:
    """SOS Gram basis for a target form of y-degree ``y_degree``.

    When the target is homogeneous of even degree (the GF associated forms are —
    μ ∝ r, so every term has the same y-degree), only the degree-exact half basis
    can match it; the full ``0..⌈y_degree/2⌉`` basis just inflates the per-octant
    Gram from O(d) to O(d²) rows and makes CVXPY's problem compilation blow up
    (a 325×325 vs 25×25 Gram for μ-doubled fractional GFs; the difference between a
    multi-minute hang and ~0.4 s). Monomials of other degrees still appear in the
    coefficient-match constraints, but with an all-zero coefficient row, so dropping
    them from the *basis* changes nothing in the certified form. Falls back to the
    full basis when the form is not known to be homogeneous (e.g. weighted-sphere
    cleared W) or the degree is odd.
    """
    if homogeneous and y_degree > 0 and y_degree % 2 == 0:
        return get_homogeneous_monomial_basis(n_vars, y_degree // 2)
    return get_monomial_basis(n_vars, (y_degree + 1) // 2)


def compute_mu_exponents(
    term_specs: List[TermSpec],
    weights_r: Tuple[int, ...],
    ensure_even_exponents: bool = True
) -> Tuple[int, Tuple[int, ...]]:
    """
    Compute optimal μ per Sanchez-Moreno 2018 Lemma 6.

    For SOS verification (Algorithm 7), y-exponents must be even so
    associated forms can extend to all of ℝⁿ. This requires doubling
    the base μ values.

    The transformation y_i = |x_i|^(1/μ_i) where μ_i = LCM of denominators
    for variable i's exponents (per-variable, not global).

    SOSTOOLS Approach (more efficient):
    Compute μᵢ = LCM of denominators in xᵢ exponents only, not all denominators.
    This minimizes the maximum y-degree and keeps SOS problems tractable.

    Example for PID-SMC (r=(3,2,1), m=7):
    - Base μ = (3, 2, 1) from denominators
    - With ensure_even_exponents=True: μ = (6, 4, 2) for SOS compatibility

    Args:
        term_specs: GF term specifications
        weights_r: Homogeneity weights (r₁, ..., rₙ)
        ensure_even_exponents: If True (default), double μ for SOS-compatible
                              even y-exponents per Algorithm 7 of Sanchez-Moreno 2018.

    Returns:
        (k, mu) where k is kept for backwards compatibility (set to 1),
        and mu guarantees integer (and optionally even) y-exponents.
    """
    n_vars = len(weights_r)

    # Step 1: Compute Lᵢ = LCM of denominators per variable
    per_var_denoms = [set() for _ in range(n_vars)]
    for spec in term_specs:
        for i, exp in enumerate(spec.exponents):
            if isinstance(exp, Fraction) and exp != 0:
                per_var_denoms[i].add(exp.denominator)

    # Step 2: Base μᵢ = LCM(denominators for variable i)
    mu = []
    for i in range(n_vars):
        denoms = list(per_var_denoms[i])
        mu_i = lcm_list(denoms) if denoms else 1
        mu.append(mu_i)

    # Step 3: For SOS, ensure even y-exponents (Algorithm 7)
    if ensure_even_exponents:
        mu = [2 * m for m in mu]

    # k is kept for backwards compatibility, though it's not really used anymore
    k = 1
    return k, tuple(mu)


def _lcm_int(a: int, b: int) -> int:
    """Least common multiple for positive integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def _exp_denominator(exp: Any) -> int:
    """Return exponent denominator as a positive integer."""
    if isinstance(exp, Fraction):
        return int(exp.denominator)
    frac = Fraction(str(exp)).limit_denominator()
    return int(frac.denominator)


def compute_mu_exponents_with_dynamics(
    term_specs: List[TermSpec],
    weights_r: Tuple[int, ...],
    dynamics_gf: Optional[List[GeneralizedForm]] = None,
    ensure_even_exponents: bool = True,
) -> Tuple[int, Tuple[int, ...]]:
    """
    Compute μ for joint V+W SOS so all V and dynamics exponents map to integer y-exponents.

    For joint verification, W = -dV/dt introduces exponents from both V and dynamics.
    If μ is computed from V terms only, W may produce non-integer y-exponents
    (e.g. Levant3D: 5/3 with μ₁=2 -> 10/3), which crashes SOS assembly.

    This helper enforces μᵢ = k * rᵢ with the minimal global k satisfying:
      - k * rᵢ is divisible by every denominator present in variable i across
        V term specs and dynamics terms.
      - optional even-exponent requirement for associated-form/SOS compatibility.
    """
    n_vars = len(weights_r)
    required_denoms = [1] * n_vars

    def _accumulate(exponents: Tuple[Any, ...]) -> None:
        for i, exp in enumerate(exponents):
            if exp == 0:
                continue
            d = _exp_denominator(exp)
            required_denoms[i] = _lcm_int(required_denoms[i], d)

    for spec in term_specs:
        _accumulate(spec.exponents)

    if dynamics_gf is not None:
        for f_i in dynamics_gf:
            if f_i is None:
                continue
            for term in f_i.terms:
                _accumulate(term.exponents)

    # Find smallest global k with μᵢ = k rᵢ meeting all denominator constraints.
    k = 1
    for i, r_i in enumerate(weights_r):
        need = required_denoms[i]
        factor = need // gcd(need, int(r_i))
        k = _lcm_int(k, max(1, factor))

    if ensure_even_exponents:
        k = _lcm_int(k, 2)

    mu = tuple(int(k * r_i) for r_i in weights_r)
    return int(k), mu


def term_to_y_monomial(
    spec: TermSpec,
    mu: Tuple[int, ...],
    signs: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], int]:
    """
    Convert a GF term to y-space monomial with sign factor.

    This function computes the y-space polynomial representation of a GF term
    in a specific octant.

    Sign Factor Convention (MATLAB SOSTOOLS compatible):
    - A 'signed' term type ALWAYS contributes its sign factor sᵢ
    - This matches MATLAB's explicit sign factor formulation where
      sign factors are polynomial coefficients independent of exponents
    - Example: term x₁·x₃⁴ with types ('signed','abs','signed') has
      sign factor s₁·s₃ in all octants

    Mathematical basis:
    - In y-space: yᵢ = |xᵢ|^(1/μᵢ), so |xᵢ| = yᵢ^μᵢ
    - A signed term contributes sign(xᵢ) = sᵢ in octant s
    - The y-exponent is μᵢ·ρᵢ where ρᵢ is the x-space exponent
    - For cross-terms: sign factors MULTIPLY (sᵢ·sⱼ for two signed vars)

    Args:
        spec: Term specification
        mu: μ exponents for y-transform
        signs: Octant signs (+1 or -1 for each variable)

    Returns:
        (y_exponents, sign_factor) where y_exponents are integers
        and sign_factor is the product of all contributing sᵢ

    Example:
        For term x₁·x₃⁴ with types=('signed','abs','signed'), mu=(6,4,2), signs=(+1,+1,-1):
        - x₁: signed → contributes s₁=+1
        - x₂: abs → no contribution
        - x₃: signed → contributes s₃=-1
        - sign_factor = s₁ · s₃ = +1 · -1 = -1
        - y_exponents = (6*1, 4*0, 2*4) = (6, 0, 8)
    """
    n_vars = len(mu)
    y_exp = []
    sign_factor = 1

    for i in range(n_vars):
        exp = spec.exponents[i]
        typ = spec.term_types[i]

        # Compute y-exponent: μ_i * ρ_i (must be an integer)
        y_e = safe_to_int(mu[i] * exp, f"at variable {i}, exp={exp}, mu={mu[i]}")
        y_exp.append(y_e)

        # Handle sign factor for signed terms
        # MATLAB-compatible convention: 'signed' type ALWAYS contributes sᵢ
        # This is independent of whether the exponent is odd/even
        if typ == 'signed':
            sign_factor *= signs[i]

    return tuple(y_exp), sign_factor


def build_v_coefficient_matrix(
    term_specs: List[TermSpec],
    mu: Tuple[int, ...],
    signs: Tuple[int, ...],
    all_monomials: List[Tuple[int, ...]]
) -> np.ndarray:
    """
    Build matrix M such that V(y) coefficients = M @ alpha.

    V(y; alpha) = sum_j alpha_j * term_j(y)

    Each term_j in y-space is: sign_factor_j * y^{exp_j}

    So coefficient of monomial y^k is: sum_{j: exp_j = k} sign_factor_j * alpha_j

    Returns:
        Matrix M of shape (n_monomials, n_terms)
    """
    n_monomials = len(all_monomials)
    n_terms = len(term_specs)
    mono_to_idx = {m: i for i, m in enumerate(all_monomials)}

    M = np.zeros((n_monomials, n_terms))

    for j, spec in enumerate(term_specs):
        y_exp, sign_factor = term_to_y_monomial(spec, mu, signs)
        if y_exp in mono_to_idx:
            M[mono_to_idx[y_exp], j] = sign_factor

    return M


def build_gram_constraints(
    basis: List[Tuple[int, ...]],
    target_monomials: List[Tuple[int, ...]],
    n_vars: int
) -> Dict[Tuple[int, ...], List[Tuple[int, int]]]:
    """
    Build mapping from target monomials to Gram matrix entries.

    For p(y) = z(y)^T Q z(y), coefficient of y^k equals sum of Q_{ij}
    where deg(z_i) + deg(z_j) = k.

    Returns:
        Dict mapping monomial -> list of (i,j) pairs in Q
    """
    n_basis = len(basis)
    mono_to_pairs = {m: [] for m in target_monomials}

    for i in range(n_basis):
        for j in range(n_basis):
            # Product monomial
            prod = tuple(basis[i][k] + basis[j][k] for k in range(n_vars))
            if prod in mono_to_pairs:
                mono_to_pairs[prod].append((i, j))

    return mono_to_pairs


def v_epsilon_row_indices(
    all_v_monomials: Sequence[Tuple[int, ...]],
    v_y_degree: int,
    n_vars: int,
) -> List[int]:
    """Row indices of the V block's pure top-degree powers yᵢ^{d_V} (P5-04).

    The predicate both design encodings already used inline, factored out so
    the CVXPY path and the MOSEK_DIRECT path cannot drift apart on it. Those
    rows carry the design margin ε (``exact_octant_target`` subtracts ε from
    ALL n of them, so the design must emit all n or none).
    """
    rows: List[int] = []
    for idx, mono in enumerate(all_v_monomials):
        if sum(mono) != v_y_degree:
            continue
        nz = [i for i, e in enumerate(mono) if e]
        if len(nz) == 1 and mono[nz[0]] == v_y_degree:
            rows.append(idx)
    return rows


def build_gram_completion_map(
    basis: List[Tuple[int, ...]],
    target_monomials: Sequence[Tuple[int, ...]],
    n_vars: int
) -> Dict[Tuple[int, ...], List[Tuple[int, int]]]:
    """Gram entries the target support does NOT constrain (WP P2-09).

    ``build_gram_constraints`` above emits one coefficient-matching row per
    monomial of the *target support*. Every basis product ``z_i·z_j`` that
    lands OUTSIDE that support is then absent from every row, so the
    corresponding Gram entries are free variables of the SDP: the solver may
    put arbitrary mass there and the block is no longer a certificate of the
    target polynomial, only of ``target + (that mass)``. The design backend's
    own row-wise residual is blind to it; ``_validate_gram_certificate``
    charges it in full (it expands ``zᵀQz`` over ``set(gram_poly) | set(target)``).

    Measured instance before this WP: ``power_rate``'s W block, target
    ``{y³: 3}`` on the non-homogeneous basis ``[1, y, y²]``. The product ``y⁴``
    is outside the support, so ``Q[2,2]`` was unconstrained and MOSEK returned
    ``6.16e+04`` there (P2-04 contradiction 1).

    This map is the fix: one extra row ``Σ_{(i,j)} Q[i,j] == 0`` per outside
    monomial, which turns the block into a genuine SOS certificate of the
    target. It is EMPTY whenever the target support already covers every basis
    product — the homogeneous even-degree case, i.e. every V block and every W
    block once the μ policy below has made its y-degree even — so on the twelve
    nominal systems it adds no rows at all and cannot move a design point. It
    is a structural invariant, not a tuning knob.

    Returns:
        Dict mapping an out-of-support product monomial -> list of (i, j) pairs.
    """
    support = set(target_monomials)
    extra: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    n_basis = len(basis)
    for i in range(n_basis):
        for j in range(n_basis):
            prod = tuple(basis[i][k] + basis[j][k] for k in range(n_vars))
            if prod not in support:
                extra.setdefault(prod, []).append((i, j))
    return extra


#: ``mu_policy`` of a block whose μ is the design μ shared with V.
MU_POLICY_DESIGN: str = ""

#: ``mu_policy`` of a W block whose μ was doubled because the W associated
#: form came out at an ODD y-degree under the design μ (WP P2-09). An odd
#: homogeneous form is sign-indefinite on ℝⁿ and can never be SOS, so the
#: design would either be infeasible (with the completion rows above) or carry
#: an unconstrained Gram entry (without them). Doubling μ is the standard
#: even-exponent lift of Algorithm 7 applied to W alone: y-exponents μᵢ·aᵢ are
#: integers by construction, so 2μᵢ·aᵢ is even, the block becomes homogeneous
#: of even degree, the P2-02 ε rows apply, and the exact gate accepts it —
#: ``exact_octant_target`` rebuilds every target at the BLOCK's own μ, which
#: the bundle carries per entry. This is exactly the μ ``verify_gf_sos_full``
#: computes when it re-solves W on its own (measured: ``power_rate`` W,
#: μ=(4,), y-degree 6, basis [(3,)], Gram [[2.9999]] — identical either way).
MU_POLICY_W_DOUBLED: str = "w_mu_doubled_odd_y_degree"


class JointSOSOptimizer:
    """
    Find GF Lyapunov coefficients via joint SOS optimization.

    This implements the SOSTOOLS approach: find alpha such that
    V(x; alpha) > 0 AND W(x; alpha) = -dV/dt > 0 in all hyperoctants.

    The optimization is formulated as an SDP feasibility problem:

    For each octant s in {-1,+1}^n:
        V_s(y; alpha) - epsilon * ||y||^d_V is SOS
        W_s(y; alpha) - epsilon * ||y||^d_W is SOS

    where SOS means exists Q >= 0 such that p(y) = z(y)^T Q z(y).
    """

    def __init__(
        self,
        weights_r: Tuple[int, ...],
        epsilon: float = 1e-4,
        solver: str = 'MOSEK',
        strict_solver: bool = False,
        allow_solver_fallback: bool = True,
        verbose: bool = False,
        ensure_even_exponents: bool = True,
        coefficient_scale: float = 1.0,
        scs_max_iters: int = 20000,
        solver_timeout: Optional[float] = None,
        use_l1_regularization: bool = False,
        l1_lambda: float = 0.01,
        maximize_diagonal: bool = False,
        coefficient_bounds: Optional[Dict[int, Tuple[Optional[float], Optional[float]]]] = None,
        nonneg_abs_cross: bool = False,
        convexity_cuts: Optional[np.ndarray] = None,
        convexity_cut_margin: float = 0.0,
        domination_pairs: Optional[List[Tuple[int, int]]] = None,
        linear_coefficient_constraints: Optional[Sequence[Tuple[Sequence[float], Optional[float], Optional[float]]]] = None,
        reference_coefficients: Optional[Sequence[float]] = None,
        l1_weights: Optional[Sequence[float]] = None,
        parity_reduction: Optional[bool] = None,
        y_parity_blocks: Optional[bool] = None,
        capture_design_grams: bool = True,
        w_epsilon_rows: bool = True,
        design_screen: str = "reject",
        design_screen_budget: str = "achieved",
        design_screen_residual: str = "dense",
        objective: str = "auto",
    ):
        """
        Initialize Joint SOS Optimizer.

        Args:
            weights_r: Homogeneity weights
            epsilon: SOS margin (default 1e-4)
            solver: SDP solver name (default 'MOSEK')
            strict_solver: If True, fail instead of falling back to other solvers.
            allow_solver_fallback: If False, use requested solver only.
            verbose: Print progress
            ensure_even_exponents: If True (default), double μ values for SOS
                compatibility per Algorithm 7 of Sanchez-Moreno 2018. This ensures
                y-exponents are even so associated forms extend to all of ℝⁿ.
            coefficient_scale: Scale for V coefficients (default 1.0).
                For large systems like PID-SMC where natural coefficients are
                ~1e6-1e7, use coefficient_scale=1e-6 to keep optimization
                numerically stable. The final coefficients will be unscaled.
            scs_max_iters: Maximum iterations for SCS fallback solver (default 20000).
                Prevents SCS from running indefinitely on non-converging problems.
            solver_timeout: Optional timeout in seconds for solver. If None, uses
                solver default.
            use_l1_regularization: DEPRECATED — subsumed by ``l1_weights`` (the
                unweighted special case wⱼ = l1_lambda). If True, add L1 penalty
                on cross-term coefficients to promote sparsity.
            l1_lambda: DEPRECATED with ``use_l1_regularization``.
            domination_pairs: Optional list of (abs_idx, signed_idx) pairs.
                For each pair, enforces alpha[abs_idx] >= |alpha[signed_idx]|
                so abs duplicate always dominates the signed term.
            linear_coefficient_constraints: Optional list of linear constraints on
                the true-scale coefficients. Each entry is `(row, lb, ub)` and is
                interpreted as `lb <= row @ alpha_true <= ub`. Use `None` for
                one-sided bounds.
            reference_coefficients: Optional array of reference coefficient values.
                When provided, the optimizer minimizes ||α - α_ref||² instead of
                feasibility. Useful for finding the closest feasible V to a known
                good candidate.
            l1_weights: Optional per-term weights for a weighted-ℓ₁ objective
                Σⱼ wⱼ·|αⱼ| over the CROSS terms (diagonals stay unpenalized —
                they are pinned by the scale normalization and the ε lower
                bound, which is what makes ℓ₁ values comparable across solves).
                Weights act on TRUE-scale coefficients (they are divided by
                ``coefficient_scale`` internally). One weight per term spec, in
                term order; weights at diagonal positions are ignored. This is
                the reweighted-ℓ₁ (Candès–Wakin–Boyd) hook used by
                ``verification.sparse_sos`` — it supersedes the unweighted
                ``use_l1_regularization``/``l1_lambda`` pair.
            parity_reduction: Opt-in design-time parity/symmetry reduction
                (campaign block S1, ``docs/papers/theory_notes.md`` §10).
                ``None`` (default) uses the process-wide default, which is
                ``False`` unless ``set_parity_reduction_default(True)`` was
                called. When active the DESIGN problem is restricted to the
                even-parity sub-library and to one Gram block per orbit of
                x ↦ −x (2ⁿ⁻¹ V blocks; 2ⁿ⁻¹ W blocks nominally, 2ⁿ PAIRED
                (octant, endpoint) W blocks under a two-endpoint inclusion).
                The prerequisites are VERIFIED symbolically per system; if any
                fails the reduction refuses, the full formulation runs, and the
                refusal is recorded in ``JointSOSResult.parity_reduction``.
                Verification is never reduced.
            capture_design_grams: Persist the design-stage Gram bundle into
                ``JointSOSResult.gram_certificates`` (default True). Applies to
                the ``MOSEK_DIRECT`` backend only, which is the one that
                recovers Grams; the CVXPY design path never captures them.
                The bundle holds one dense float64 Gram per emitted block, so
                on a very large system (levant5d ``m=11``: 32 blocks of
                1365x1365 is about 0.5 GB) pass False to keep only the scalar
                diagnostics. Turning it off changes NO certification outcome —
                the packaged verdict, coefficients and diagnostics are byte
                identical either way.
            y_parity_blocks: Restrict every design Gram (V and every W endpoint
                block) to be block-diagonal over the PARITY CLASSES of its own
                monomial basis (WP P5-04). ``None`` takes the process-wide
                default (``set_y_parity_blocks_default``). This is LOSSLESS in
                α, not a relaxation: the targets live on the even lattice, so
                averaging any PSD Gram over the sign-flip group G = (Z/2)ⁿ
                returns a PSD Gram of the same polynomial with every
                cross-class entry zero (``verification/yparity.py``). Soundness
                is the trivial direction — a block-diagonal Gram IS a Gram.
                What it buys is PSD scalars: a d×d block becomes Σ_c d_c×d_c,
                measured 3.5× fewer scalars at n = 3 and projected 7.2× at
                n = 4, m = 9. Applied only when the SYMBOLIC even-support guard
                (``algebra.support_guard``) passes at the block's own μ; a
                typed refusal (``OddLiftRefusal`` / ``YParityRefused``) falls
                the whole design back to the dense encoding and is recorded in
                ``JointSOSResult.y_parity``. Both backends implement it, so a
                MOSEK_DIRECT → CVXPY fallback changes the encoding and never
                the formulation. The bundle entry stays DENSE: the sub-Grams
                are reassembled block-diagonally onto the full basis, so the
                exact phase, the design screen and the entry hash see exactly
                the object they saw before.
            w_epsilon_rows: Give the design W blocks the same pure-power ε rows
                the V blocks carry, when the weighted-sphere clearing is
                INACTIVE (default True; WP P2-02). This is a real, if small,
                tightening of the design feasible set: the SDP must then find a
                W with a strict margin ε on every axis rather than mere
                non-negativity. It is what makes the design W Gram directly
                exact-gateable — ``exact_gate.exact_verify_captures`` refuses a
                zero margin (``epsilon_not_positive``), because a budget of
                ε/2 = 0 would prove only non-strict positivity. Pass False to
                restore the historical zero-margin W block. Has NO effect when
                the clearing is active: the cleared W carries the factor
                ∏ᵢ|xᵢ|^{k·ρ/rᵢ} and so vanishes on the axes, where an ε shift
                would make the SDP spuriously infeasible.
            design_screen: Fail-fast screen on the packaged design result
                (default ``"reject"``; WP P2-04, decision D1). ``MOSEK_DIRECT``
                only — it is the backend that recovers Grams. Two filters run
                on every feasible design output before it can enter the exact
                phase: ``sphere_check.min_V_on_sphere`` on V (typed
                ``sphere_precheck_failed``) and
                ``_validate_gram_certificate`` on every bundle entry, V and W,
                with the entry's own ε as the budget (typed
                ``gram_validation_failed``, worst block named). Neither is the
                proof; both can only reject. ``"warn"`` records the verdict on
                ``JointSOSResult.design_screen`` and keeps the result feasible;
                ``"off"`` skips the screen entirely. Cost is milliseconds on
                the nominal systems (0.01-0.33 s over all 12). Note that the
                screen consults NO solver status: a design the backend typed
                ``unknown_accepted`` (``mosek_direct.py`` tolerates
                ``solsta.unknown`` at eig_tol 1e-7 / res_tol 1e-5, three to
                five orders looser than the exact budget) still has to pass it.
            design_screen_budget: Which acceptance THRESHOLD the screen's Gram
                half uses (WP P2-13), from
                ``design_screen.BUDGET_POLICIES``. ``"achieved"`` (default
                since P2-14) refuses only a charge at or above ε, the point
                where the certified margin ε − charge stops being positive;
                ``"half"`` (historical) refuses a block whose charge exceeds
                ε/2.
                It must match the ``margin_policy`` the exact phase will run
                under (``exact_phase.run_exact_phase``), because the screen's
                only job is to predict that gate cheaply. Changing it moves no
                design point: the screen runs on an already-solved result.
            design_screen_residual: WHICH residual the screen charges (WP
                P2-13), from ``design_screen.RESIDUAL_MODELS``. ``"dense"``
                (the default) expands ``zᵀQz`` over every basis pair, which
                is the residual BOTH exact implementations charge since
                2026-09-25 (``exact_gate`` adopted the portable verifier's
                dense rule, reports/XVC_RESIDUAL_2026-09-25.md), so it is the
                model that predicts X and P. ``"deck_blocked"`` expands it
                over same-parity-class pairs only (the residual of the
                Reynolds projection, never larger than the dense one); it is
                DEPRECATED and OPTIMISTIC against X and P, and selecting it
                emits ``design_screen.OptimisticResidualModelWarning``. Like
                ``design_screen_budget`` it moves no design point and is never
                a proof.
            objective: Which objective the DESIGN SDP minimizes (WP P5-05).
                ``"auto"`` (default, and the historical behaviour) picks the
                objective from the other flags, in the order
                ``reference_coefficients`` → ``l1_weights`` →
                ``maximize_diagonal`` → ``use_l1_regularization`` →
                feasibility (``Minimize 0``). ``"min_gram_trace"`` ADDS
                Σ_blocks tr Q to whichever of those is active — for a PSD
                block the nuclear norm IS the trace, so this is the standard
                convex low-rank surrogate, and it is bounded below by 0 on the
                PSD cone, so it can never make a feasible problem unbounded.
                Motivation: a feasibility objective hands an interior-point
                solver no reason to prefer any point of the feasible set, and
                it returns the analytic-centre-like MAXIMUM-rank Gram;
                ``min tr Q`` moves the solve onto a low-rank face. Both
                backends implement it (MOSEK_DIRECT through ``putbarcj`` with
                an identity symmetric matrix per block, CVXPY through
                ``cp.trace``), so a MOSEK_DIRECT → CVXPY fallback changes the
                encoding and not the formulation, exactly as the ε-row
                decision of P2-02 does. The flag changes NO default and no
                acceptance criterion; it changes which optimal V the designer
                proposes, and every proposal is still re-gated.
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY not installed. Run: pip install cvxpy")

        self.weights_r = weights_r
        self.n_vars = len(weights_r)
        self.epsilon = epsilon
        self.solver = _normalize_solver_name(solver)
        self.strict_solver = bool(strict_solver)
        self.allow_solver_fallback = bool(allow_solver_fallback and not self.strict_solver)
        self.verbose = verbose
        self.ensure_even_exponents = ensure_even_exponents
        if coefficient_scale <= 0:
            raise ValueError(f"coefficient_scale must be > 0, got {coefficient_scale}")
        self.coefficient_scale = coefficient_scale
        self.scs_max_iters = scs_max_iters
        self.solver_timeout = solver_timeout
        self.use_l1_regularization = use_l1_regularization
        self.l1_lambda = l1_lambda
        self.maximize_diagonal = maximize_diagonal
        self.coefficient_bounds = coefficient_bounds or {}
        self.nonneg_abs_cross = nonneg_abs_cross
        self.convexity_cuts = convexity_cuts
        self.convexity_cut_margin = convexity_cut_margin
        self.domination_pairs = domination_pairs or []
        self.linear_coefficient_constraints = list(linear_coefficient_constraints or [])
        self.reference_coefficients = (
            np.asarray(reference_coefficients, dtype=np.float64)
            if reference_coefficients is not None else None
        )
        self.l1_weights = (
            np.asarray(l1_weights, dtype=np.float64)
            if l1_weights is not None else None
        )
        if self.l1_weights is not None and np.any(self.l1_weights < 0):
            raise ValueError("l1_weights must be non-negative")
        self.parity_reduction = (
            _PARITY_REDUCTION_DEFAULT if parity_reduction is None
            else bool(parity_reduction)
        )
        self.y_parity_blocks = (
            _Y_PARITY_BLOCKS_DEFAULT if y_parity_blocks is None
            else bool(y_parity_blocks)
        )
        self.capture_design_grams = bool(capture_design_grams)
        self.w_epsilon_rows = bool(w_epsilon_rows)
        self.objective = resolve_design_objective(objective)
        from .design_screen import (resolve_budget_policy,
                                    resolve_residual_model,
                                    resolve_screen_mode)
        self.design_screen = resolve_screen_mode(design_screen)
        self.design_screen_budget = resolve_budget_policy(design_screen_budget)
        self.design_screen_residual = resolve_residual_model(
            design_screen_residual)

    def _solve_problem(
        self,
        prob: "cp.Problem",
        *,
        scs_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str], List[str], Optional[str]]:
        """
        Solve a CVXPY problem under the configured solver policy.

        Returns:
            (solved, solver_used, solver_attempts, error_message)
        """
        attempts: List[str] = []
        for solver_name in _solver_candidates(
            self.solver,
            strict_solver=self.strict_solver,
            allow_solver_fallback=self.allow_solver_fallback,
        ):
            solver_obj = getattr(cp, solver_name, None)
            if solver_obj is None:
                attempts.append(f"{solver_name}:unavailable")
                if self.strict_solver:
                    return (
                        False,
                        None,
                        attempts,
                        (
                            "Strict solver mode enabled and requested solver is unavailable: "
                            f"{solver_name}"
                        ),
                    )
                continue

            kwargs: Dict[str, Any] = {"solver": solver_obj, "verbose": self.verbose}
            if solver_name == "SCS" and scs_params:
                kwargs.update(scs_params)
            if solver_name == "MOSEK" and getattr(self, '_mosek_params', None):
                kwargs["mosek_params"] = self._mosek_params

            try:
                prob.solve(**kwargs)
                attempts.append(f"{solver_name}:{prob.status}")
                return True, solver_name, attempts, None
            except cp.error.SolverError as exc:
                attempts.append(f"{solver_name}:solver_error:{exc}")
                if self.strict_solver:
                    return (
                        False,
                        None,
                        attempts,
                        f"Strict solver mode: solver '{solver_name}' failed with SolverError: {exc}",
                    )
            except Exception as exc:  # pragma: no cover - defensive
                attempts.append(f"{solver_name}:error:{exc}")
                if self.strict_solver:
                    return (
                        False,
                        None,
                        attempts,
                        f"Strict solver mode: solver '{solver_name}' failed: {exc}",
                    )

        return (
            False,
            None,
            attempts,
            (
                "No solver succeeded under configured policy "
                f"(requested={self.solver}, strict={self.strict_solver}, "
                f"allow_fallback={self.allow_solver_fallback})"
            ),
        )

    def _extract_dual_info(
        self,
        prob: 'cp.Problem',
        constraints: List,
        n_terms: int,
        n_octants: int,
        term_specs: List[TermSpec]
    ) -> SOSDualInfo:
        """
        Extract diagnostic information from infeasible SOS problem.

        When SOS fails, the dual variables indicate which constraints are
        "blocking" feasibility. High dual values suggest the corresponding
        terms are incompatible with the dynamics.

        Args:
            prob: Solved CVXPY problem (infeasible)
            constraints: List of CVXPY constraints
            n_terms: Number of Lyapunov terms
            n_octants: Number of octants
            term_specs: Term specifications

        Returns:
            SOSDualInfo with diagnostic information
        """
        dual_info = SOSDualInfo()
        dual_info.solver_info = {
            'status': prob.status,
            'solver_stats': str(prob.solver_stats) if hasattr(prob, 'solver_stats') else None,
        }

        # Try to extract constraint dual values
        # For PSD constraints, CVXPY stores dual values differently
        try:
            constraint_duals = {}
            psd_constraint_idx = 0
            blocking_octants = []

            for i, constr in enumerate(constraints):
                try:
                    if hasattr(constr, 'dual_value') and constr.dual_value is not None:
                        dual_val = constr.dual_value
                        if isinstance(dual_val, np.ndarray):
                            # For matrix constraints, use Frobenius norm as "strength"
                            strength = np.linalg.norm(dual_val)
                        else:
                            strength = abs(float(dual_val))

                        constraint_duals[f'constraint_{i}'] = strength

                        # Check if this is a PSD constraint (Q >> 0)
                        # High dual on PSD constraint means that octant is problematic
                        if hasattr(constr, 'args') and ('Q_V_' in str(constr) or 'Q_W_' in str(constr)):
                            octant_idx = psd_constraint_idx // 2  # V and W per octant
                            if strength > 1e-3:
                                if octant_idx not in blocking_octants:
                                    blocking_octants.append(octant_idx)
                            psd_constraint_idx += 1
                except Exception:
                    pass

            dual_info.constraint_duals = constraint_duals
            dual_info.problematic_octants = blocking_octants

            # Identify which terms might be problematic
            # Heuristic: terms that appear in many constraints with high duals
            term_scores = [0.0] * n_terms
            for key, val in constraint_duals.items():
                if val > 1e-4:
                    # Distribute score across all terms (simplified heuristic)
                    for j in range(n_terms):
                        term_scores[j] += val / n_terms

            # Terms with highest scores are most "blocking"
            if any(s > 0 for s in term_scores):
                sorted_indices = sorted(range(n_terms), key=lambda i: term_scores[i], reverse=True)
                dual_info.blocking_term_indices = sorted_indices[:3]  # Top 3

                # Suggest swaps: for each blocking term, suggest swapping
                # with the next available term from the same pair
                for idx in dual_info.blocking_term_indices[:2]:
                    spec = term_specs[idx]
                    # Determine which pair this term belongs to
                    for i in range(self.n_vars):
                        for j in range(i + 1, self.n_vars):
                            if float(spec.exponents[i]) > 0 and float(spec.exponents[j]) > 0:
                                dual_info.suggested_swaps.append((idx, (i, j)))
                                break

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not extract dual info: {e}")

        return dual_info

    def find_lyapunov(
        self,
        degree_m: int,
        dynamics_gf: List[GeneralizedForm],
        max_cross_terms: Optional[int] = None,
        active_terms: Optional[List[TermSpec]] = None,
        max_denominator: int = 2
    ) -> JointSOSResult:
        """
        Find Lyapunov coefficients satisfying V > 0 only.

        Args:
            degree_m: GF homogeneity degree
            dynamics_gf: System dynamics as list of GFs [f_1, ..., f_n]
            max_cross_terms: Limit cross-terms
            active_terms: Use specific terms (overrides enumeration)
            max_denominator: Max denominator for fractional exponents (default 2)
                             IMPORTANT: Large values cause problem explosion!
                             For STA, 2 is sufficient. For PID-SMC, 3 may be needed.

        Returns:
            JointSOSResult with coefficients if feasible
        """
        start_time = time.time()

        # Step 1: Get term structure
        if active_terms is not None:
            term_specs = active_terms
        else:
            diag_terms, cross_terms = enumerate_all_terms(
                degree_m, self.weights_r,
                max_denominator=max_denominator,
                include_signed=True,
                max_cross_terms=max_cross_terms or 50
            )
            term_specs = diag_terms + cross_terms

        n_terms = len(term_specs)

        # Compute μ exponents for y-transform
        k, mu = compute_mu_exponents(term_specs, self.weights_r, self.ensure_even_exponents)

        # Compute degree of V in y-space
        v_y_degree = 0
        for spec in term_specs:
            deg = sum(mu[i] * float(spec.exponents[i]) for i in range(self.n_vars))
            v_y_degree = max(v_y_degree, int(deg))

        if self.verbose:
            print(f"JointSOS: {n_terms} terms, {2**self.n_vars} octants")
            print(f"  μ = {mu}, V y-degree = {v_y_degree}")

        # Step 2: Create CVXPY variables
        alpha = cp.Variable(n_terms, name='alpha')
        constraints = []

        # Track diagonal indices for normalization
        diag_indices = []

        # Diagonal coefficients >= epsilon
        for i, spec in enumerate(term_specs):
            if spec.is_diagonal():
                constraints.append(alpha[i] >= self.epsilon)
                diag_indices.append(i)

        # Normalization: Fix first diagonal coefficient to 1 to avoid trivial solutions
        if diag_indices:
            constraints.append(alpha[diag_indices[0]] == 1.0)

        # Upper bounds to prevent unbounded solutions
        for i in range(n_terms):
            constraints.append(alpha[i] <= 100.0)

        # Step 3: Build monomial basis for SOS. The V associated form is homogeneous
        # (μ ∝ r), so the degree-exact half basis suffices — see _sos_gram_basis.
        basis = _sos_gram_basis(self.n_vars, v_y_degree, homogeneous=True)
        n_basis = len(basis)

        # All monomials that can appear in V (up to v_y_degree)
        all_v_monomials = get_monomial_basis(self.n_vars, v_y_degree)

        if self.verbose:
            print(f"  SOS basis size: {n_basis}, V monomials: {len(all_v_monomials)}")

        # Step 4: For each octant, add SOS constraints for V
        octants = get_hyperoctant_signs(self.n_vars)
        gram_matrices = []

        for oct_idx, signs in enumerate(octants):
            # Build V coefficient matrix for this octant
            # V(y) = M_V @ alpha where M_V maps alpha to polynomial coefficients
            M_V = build_v_coefficient_matrix(term_specs, mu, signs, all_v_monomials)

            # V - epsilon * ||y||^d must be SOS
            # ||y||^d = sum_i y_i^d for homogeneous norm

            # Create Gram matrix Q_V for this octant (must be PSD)
            Q_V = cp.Variable((n_basis, n_basis), symmetric=True, name=f'Q_V_{oct_idx}')
            constraints.append(Q_V >> 0)  # PSD constraint
            gram_matrices.append(Q_V)

            # Build Gram constraints: coef(y^k) = sum Q_ij for matching pairs
            gram_map = build_gram_constraints(basis, all_v_monomials, self.n_vars)

            for mono_idx, mono in enumerate(all_v_monomials):
                # LHS: coefficient of y^mono in V - epsilon*||y||^d
                # This equals (M_V @ alpha)[mono_idx] - epsilon * (1 if mono is ||y||^d term else 0)

                v_coef = M_V[mono_idx, :] @ alpha

                # Subtract epsilon for the ||y||^d terms
                eps_term = 0.0
                if sum(mono) == v_y_degree:
                    # Check if this is a "pure power" monomial like y_i^d
                    for i in range(self.n_vars):
                        if mono[i] == v_y_degree and sum(mono) == v_y_degree:
                            eps_term = self.epsilon
                            break

                # RHS: sum of Q entries for this monomial
                pairs = gram_map.get(mono, [])
                if pairs:
                    q_sum = sum(Q_V[i, j] for i, j in pairs)
                    constraints.append(v_coef - eps_term == q_sum)
                elif abs(eps_term) > 0 or mono_idx < n_terms:
                    # Monomial must have zero coefficient if no Q pairs produce it
                    constraints.append(v_coef - eps_term == 0)

        # Step 5: Solve the SDP
        prob = cp.Problem(cp.Minimize(0), constraints)
        solved, solver_used, solver_attempts, solver_error = self._solve_problem(prob)
        solver_used_str = str(solver_used or "")
        fallback_used = bool(solver_used_str and solver_used_str != self.solver)
        if not solved:
            return JointSOSResult(
                feasible=False,
                solver_status="solver_error",
                solver_requested=self.solver,
                solver_used=solver_used_str,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message=str(solver_error or "Solver error"),
                solve_time=time.time() - start_time,
                term_specs=term_specs,
            )

        # Step 6: Extract result
        # TRUST BOUNDARY: accepting OPTIMAL_INACCURATE here is sound because this
        # is the DESIGN side — the returned V is only a proposal. Every stability
        # claim must pass the strict gate (certify_discovered_gf → verify_gf_sos_full
        # with a-posteriori Gram validation), which never trusts solver status.
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            coef_vals = alpha.value
            coef_dict = {f'alpha{i}': float(coef_vals[i]) for i in range(n_terms)}

            # Build V_gf
            gf_terms = []
            for i, spec in enumerate(term_specs):
                gf_terms.append(GFTerm(
                    coefficient=float(coef_vals[i]),
                    exponents=spec.exponents,
                    term_types=spec.term_types
                ))

            V_gf = GeneralizedForm(
                degree_m=degree_m,
                weights_r=self.weights_r,
                terms=gf_terms
            )

            return JointSOSResult(
                feasible=True,
                coefficients=coef_dict,
                V_gf=V_gf,
                solver_status=f"{prob.status}:{solver_used_str}" if solver_used_str else str(prob.status),
                solver_requested=self.solver,
                solver_used=solver_used_str,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message="Joint SOS feasible (V constraints satisfied)",
                solve_time=time.time() - start_time,
                term_specs=term_specs,
            )
        else:
            return JointSOSResult(
                feasible=False,
                solver_status=f"{prob.status}:{solver_used_str}" if solver_used_str else str(prob.status),
                solver_requested=self.solver,
                solver_used=solver_used_str,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message=f"Infeasible: {prob.status}",
                solve_time=time.time() - start_time,
                term_specs=term_specs,
            )

    # =====================================================================
    # Design-time parity reduction (campaign block S1) — helpers
    # =====================================================================

    def _parity_asymmetric_options(self) -> List[str]:
        """Active options that put assumption (A3) of §10.0 at risk.

        Proposition 10.1 needs the design feasible set to be invariant under
        the coefficient reflection ``Rα = (π(t_j)·α_j)_j``. The options listed
        here attach a constraint or an objective to INDIVIDUAL coefficient
        indices in a way that is not sign-symmetric, or that indexes the
        unfiltered library with no sound re-mapping. With any of them active
        the averaging step of Proposition 10.1 is unavailable, so the reduction
        refuses rather than quietly changing the problem.

        Deliberately NOT listed, because they are R-invariant and re-map
        themselves:

        * ``l1_weights`` / ``use_l1_regularization`` — Σ wⱼ·|αⱼ| depends on the
          magnitudes only, so R leaves it unchanged; the weight vector is
          re-mapped index-wise onto the filtered library (finding 6).
        * ``maximize_diagonal`` — the diagonals are ``abs``-only, hence even and
          fixed by R.
        * ``nonneg_abs_cross`` — the constrained terms are all-``abs``, hence
          even and fixed by R, and their index set is rebuilt from the filtered
          library rather than carried over.
        """
        candidates = {
            "coefficient_bounds": bool(self.coefficient_bounds),
            "domination_pairs": bool(self.domination_pairs),
            "linear_coefficient_constraints": bool(self.linear_coefficient_constraints),
            "convexity_cuts": (self.convexity_cuts is not None
                               and len(self.convexity_cuts) > 0),
            "reference_coefficients": self.reference_coefficients is not None,
        }
        return sorted(name for name, active in candidates.items() if active)

    def _plan_parity_reduction(
        self,
        *,
        dynamics_gf: List[GeneralizedForm],
        dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]],
        term_specs: List[TermSpec],
    ) -> Tuple[Optional[Any], List[TermSpec], List[int], dict]:
        """Verify the §10 prerequisites and build the reduced design plan.

        Returns ``(reduction, term_specs, keep_indices, metadata)``.

        On refusal ``reduction`` is ``None``, ``term_specs`` is the UNFILTERED
        input library, ``keep_indices`` is the identity, and the metadata says
        ``{"mode": "refused", "reason_code": ..., "reason": ...}``. The caller
        then runs the ordinary full formulation. Refusal is never silent: it is
        also printed to stderr.

        Nothing here is assumed. Oddness of the nominal field (A1) and, when
        endpoints are present, the endpoint-swap identity of Lemma 10.2 are
        checked SYMBOLICALLY by ``algebra.symmetry``; the block partition and
        the library split are re-validated by that module's own consistency
        reports before they are used.
        """
        n_full_terms = len(term_specs)
        identity = list(range(n_full_terms))
        try:
            blocked = self._parity_asymmetric_options()
            if blocked:
                raise ParityReductionRefused(
                    "asymmetric_constraints",
                    "per-term options not known to preserve assumption (A3) of "
                    f"theory_notes.md §10.0 are active: {', '.join(blocked)}",
                )

            # Finding 2: endpoints are POSITIONAL. Never trust index order —
            # the swap identity below is what licenses the pairing, and a third
            # vertex (multi-channel inclusion) has no proved pairing at all.
            vertex_arg: Any = dynamics_gf_vertices
            if dynamics_gf_vertices is not None:
                n_vertices = len(dynamics_gf_vertices)
                if n_vertices not in (1, 2):
                    raise ParityReductionRefused(
                        "endpoint_count_unsupported",
                        f"parity reduction is proved for 1 (nominal) or 2 "
                        f"(matched-disturbance endpoints) inclusion vertices; "
                        f"got {n_vertices} — no pairing is proved for this "
                        f"index set (theory_notes.md §10.3)",
                    )
                if n_vertices == 1:
                    # A one-element vertex list is the nominal problem written
                    # in inclusion form; symmetry.endpoint_count reads it as 1.
                    vertex_arg = dynamics_gf_vertices

            plan = _sym.plan_parity_reduction(
                dynamics_gf,
                vertex_arg,
                terms=term_specs,
                weights_r=tuple(self.weights_r),
            )

            if not plan.oddness.holds:
                raise ParityReductionRefused(
                    "dynamics_not_odd",
                    f"assumption (A1) fails: {plan.oddness.message} "
                    f"(residuals f_i(-x)+f_i(x) = {list(plan.oddness.residuals)})",
                )
            if plan.reduction.has_endpoints:
                if plan.endpoint_swap is None:
                    raise ParityReductionRefused(
                        "endpoint_swap_unverified",
                        "inclusion endpoints are present but the swap identity "
                        "f_+(-x) = -f_-(x) could not be verified from the given "
                        "vertex argument (Lemma 10.2)",
                    )
                if not plan.endpoint_swap.holds:
                    raise ParityReductionRefused(
                        "endpoint_swap_failed",
                        f"Lemma 10.2 fails: {plan.endpoint_swap.message} "
                        f"(residuals {list(plan.endpoint_swap.residuals)})",
                    )
            if not plan.reduction_report.ok:
                raise ParityReductionRefused(
                    "reduction_inconsistent",
                    "block partition failed its consistency checks: "
                    f"{plan.reduction_report.failures()}",
                )
            if plan.library_report is not None and not plan.library_report.ok:
                raise ParityReductionRefused(
                    "library_split_inconsistent",
                    "even/odd library split failed its consistency checks: "
                    f"{plan.library_report.failures()}",
                )
            if not plan.usable:  # belt and braces; the above should have fired
                raise ParityReductionRefused(
                    "unexpected_error",
                    "symmetry.ParityReductionPlan reports unusable for a reason "
                    "not covered by the explicit checks above",
                )

            even_terms = list(plan.even_terms)
            if not even_terms:
                raise ParityReductionRefused(
                    "empty_even_library",
                    f"the even-parity sub-library is empty ({n_full_terms} terms "
                    f"in, all odd) — nothing to design over",
                )

            # Finding 6: the filter shifts indices, so the diagonal backbone is
            # RE-MAPPED, never assumed. Diagonals are |x_i|^(m/r_i), abs-only
            # and therefore even, so the full diagonal sub-sequence must survive
            # in its original order; if it does not, the library is not the one
            # §10.0 argues about and the scale normalization
            # (alpha[diag_indices[0]] == coefficient_scale) would silently pin a
            # DIFFERENT term than the unreduced run does.
            diag_full = [s for s in term_specs if s.is_diagonal()]
            diag_even = [s for s in even_terms if s.is_diagonal()]
            if diag_full != diag_even:
                raise ParityReductionRefused(
                    "diagonal_backbone_changed",
                    f"the even filter did not preserve the diagonal backbone: "
                    f"{len(diag_full)} diagonal terms before, {len(diag_even)} "
                    f"after (a signed diagonal would be odd and get dropped)",
                )

            # Positions of the kept terms in the ORIGINAL library. Any caller
            # datum stored against the unfiltered library (l1_weights) is
            # re-mapped through this.
            keep_indices: List[int] = []
            cursor = 0
            for j, spec in enumerate(term_specs):
                if cursor < len(even_terms) and spec is even_terms[cursor]:
                    keep_indices.append(j)
                    cursor += 1
            if cursor != len(even_terms):
                # ``split_library_by_parity`` preserves order and identity, so
                # this cannot fire; if it ever does, the re-map is unsound.
                raise ParityReductionRefused(
                    "library_split_inconsistent",
                    "could not re-map the even sub-library onto the original "
                    "term positions (order or identity not preserved)",
                )

            reduction = plan.reduction
            meta = self._parity_success_meta(
                reduction=reduction,
                n_full_terms=n_full_terms,
                n_even_terms=len(even_terms),
                oddness_ok=bool(plan.oddness.holds),
                endpoint_swap_ok=(
                    bool(plan.endpoint_swap.holds)
                    if plan.endpoint_swap is not None else None
                ),
            )
            if self.verbose:
                print(f"  parity reduction: {reduction.describe()}; library "
                      f"{n_full_terms} → {len(even_terms)} even terms")
            if self.l1_weights is not None and len(even_terms) != n_full_terms:
                # The weights themselves ARE re-mapped below. What this cannot
                # fix is a caller that maps positions in the RETURNED V_gf back
                # onto its own unfiltered library — the reweighted-ℓ₁ driver in
                # ``verification/sparse_sos.py`` does exactly that with its
                # support indices. Make the aliasing visible instead of silent.
                print(f"[joint_sos] parity reduction active WITH l1_weights: the "
                      f"emitted V has {len(even_terms)} terms, not {n_full_terms}. "
                      f"Any caller indexing its own unfiltered library by the "
                      f"position of a returned term MUST re-map through the "
                      f"even-parity filter (finding 6).",
                      file=sys.stderr, flush=True)
            return reduction, even_terms, keep_indices, meta

        except ParityReductionRefused as exc:
            meta = {"mode": "refused", "requested": True,
                    "reason_code": exc.reason_code, "reason": exc.reason}
            print(f"[joint_sos] parity reduction REFUSED ({exc.reason_code}): "
                  f"{exc.reason}; running the FULL unreduced formulation",
                  file=sys.stderr, flush=True)
            return None, list(term_specs), identity, meta
        except Exception as exc:  # symmetry module raised something unexpected
            meta = {"mode": "refused", "requested": True,
                    "reason_code": "unexpected_error",
                    "reason": f"{type(exc).__name__}: {exc}"}
            print(f"[joint_sos] parity reduction REFUSED (unexpected_error): "
                  f"{type(exc).__name__}: {exc}; running the FULL unreduced "
                  f"formulation", file=sys.stderr, flush=True)
            return None, list(term_specs), identity, meta

    def _parity_success_meta(
        self,
        *,
        reduction: Any,
        n_full_terms: int,
        n_even_terms: int,
        oddness_ok: bool,
        endpoint_swap_ok: Optional[bool],
    ) -> dict:
        """The ``parity_reduction`` record of an ACTIVE reduction.

        Schema (all counts are design-side; verification is never reduced):

        ``mode``               "nominal" | "paired"
        ``requested``          always True here
        ``n_v_blocks`` /
        ``n_w_blocks``         Gram blocks actually built
        ``n_v_blocks_full`` /
        ``n_w_blocks_full``    what the unreduced formulation would build
        ``n_octants``          2ⁿ
        ``n_endpoints``        1 (nominal) or 2 (matched-disturbance inclusion)
        ``v_block_octants``    FULL octant indices of the V representatives
        ``w_blocks``           one entry per built W block, each
                               ``{octant_index, endpoint, partner_octant_index,
                               partner_endpoint, label}``. ``partner_endpoint``
                               is what makes naive halving detectable: under the
                               paired reduction it is ALWAYS the other endpoint.
        ``naive_octant_halving`` always False — recorded explicitly so an
                               artifact can be audited without re-deriving the
                               involution (Corollary 10.4).
        ``library``            {n_terms_full, n_terms_even, n_terms_dropped_odd}
        ``checks``             the symbolic evidence that licensed the reduction
        """
        labels = list(reduction.w_blocks())
        w_blocks = []
        for signs, endpoint in labels:
            p_signs, p_endpoint = reduction.partner_label((signs, endpoint))
            ep_label = "" if endpoint is None else _sym.ENDPOINT_LABELS[endpoint]
            w_blocks.append({
                "octant_index": _sym.octant_index(signs),
                "signs": [int(s) for s in signs],
                "endpoint": endpoint,
                "partner_octant_index": _sym.octant_index(p_signs),
                "partner_endpoint": p_endpoint,
                "label": f"oct{_sym.octant_index(signs)}{ep_label}",
            })
        mode = "paired" if reduction.has_endpoints else "nominal"
        return {
            "mode": mode,
            "requested": True,
            "n_v_blocks": int(reduction.n_v_blocks),
            "n_w_blocks": int(reduction.n_w_blocks),
            "n_v_blocks_full": int(reduction.n_octants),
            "n_w_blocks_full": int(reduction.n_octants * reduction.n_endpoints),
            "n_octants": int(reduction.n_octants),
            "n_endpoints": int(reduction.n_endpoints),
            "v_block_octants": [_sym.octant_index(s) for s in reduction.v_blocks()],
            "w_blocks": w_blocks,
            "naive_octant_halving": False,
            "library": {
                "n_terms_full": int(n_full_terms),
                "n_terms_even": int(n_even_terms),
                "n_terms_dropped_odd": int(n_full_terms - n_even_terms),
            },
            "checks": {
                "dynamics_odd": bool(oddness_ok),
                "endpoint_swap": endpoint_swap_ok,
            },
            "describe": reduction.describe(),
            "scope": "design_time_only; strict and exact gates run unreduced",
        }

    # =====================================================================
    # Per-block constraint emission (shared by the reduced and full paths)
    # =====================================================================

    def _plan_y_parity(
        self,
        *,
        term_specs: List[TermSpec],
        mu: Tuple[int, ...],
        w_mu: Tuple[int, ...],
        vertices: List[List[GeneralizedForm]],
        clearing_power: int,
    ) -> dict:
        """Decide whether the design Grams may be split by y-parity (P5-04).

        Runs the SYMBOLIC even-support guard once per stage, at that stage's
        own μ (P2-09 doubles μ for W when the associated form's y-degree comes
        out odd, and the premise must be checked at the μ the block is actually
        built with). The guard is exact and pre-cancellation, so a pair of
        odd-lift terms whose float coefficients happen to cancel still refuses
        — the failure mode the assembled-matrix guard in ``yparity`` cannot
        see on its own.

        Returns the record that rides on ``JointSOSResult.y_parity``. The two
        stages are decided INDEPENDENTLY: ``power_rate`` nominal is the
        measured instance of a system whose W lift is odd at the design μ, and
        nothing forces V to pay for it.
        """
        meta = {
            "requested": bool(self.y_parity_blocks),
            "v_blocked": False,
            "w_blocked": False,
            "refusal": "" if self.y_parity_blocks else "not_requested",
            "detail": "",
            "mu_v": tuple(int(v) for v in mu),
            "mu_w": tuple(int(v) for v in w_mu),
        }
        if not self.y_parity_blocks:
            return meta

        from ..algebra.support_guard import (
            GuardUnreadable,
            OddLiftRefusal,
            check_even_support,
            gf_exponent_support,
        )

        weights = tuple(int(r) for r in self.weights_r)
        try:
            # Exponents are handed over raw: check_even_support converts each
            # one exactly and REFUSES a float that is not integral, which is
            # the behaviour wanted here (a guard that cannot read its input
            # must decline, never assume).
            lib = [tuple(spec.exponents) for spec in term_specs]
            dyn_sym = gf_exponent_support(vertices)
        except (GuardUnreadable, TypeError) as exc:
            meta["refusal"] = "guard_unavailable"
            meta["detail"] = str(exc)
            return meta

        # V stage: library lifts + the pure-power ε rows, at the design μ.
        try:
            check_even_support(
                mu=tuple(int(v) for v in mu), weights_r=weights,
                library_exponents=lib, dynamics_vertices=(),
                clearing_power=0)
            meta["v_blocked"] = True
        except OddLiftRefusal as exc:
            meta["refusal"] = "odd_lift_v"
            meta["detail"] = str(exc)
        except TypeError as exc:
            meta["refusal"] = "guard_unavailable"
            meta["detail"] = str(exc)
            return meta

        # W stage: the same library at the W μ, plus every endpoint's dynamics
        # support and the clearing factor.
        try:
            check_even_support(
                mu=tuple(int(v) for v in w_mu), weights_r=weights,
                library_exponents=lib, dynamics_vertices=dyn_sym,
                clearing_power=int(clearing_power))
            meta["w_blocked"] = True
        except OddLiftRefusal as exc:
            if not meta["refusal"]:
                meta["refusal"] = "odd_lift_w"
                meta["detail"] = str(exc)
        except TypeError as exc:
            meta["v_blocked"] = False
            meta["refusal"] = "guard_unavailable"
            meta["detail"] = str(exc)
        return meta

    def _w_epsilon_rows(
        self,
        w_monomials: List[List[Tuple[int, ...]]],
        clearing_power: int,
    ) -> Tuple[List[List[int]], List[Tuple[Tuple[int, ...], ...]], str]:
        """ε rows of the design W blocks, one row list per inclusion vertex.

        The V block subtracts ε from each pure top-degree power yᵢ^d, so the
        SDP certifies ``V − ε·Σᵢ yᵢ^d`` SOS rather than ``V`` SOS. The design W
        block historically subtracted NOTHING, and a zero margin is exactly
        what ``exact_gate.exact_verify_captures`` refuses
        (``epsilon_not_positive``): with budget ε/2 = 0, Proposition 7's charge
        bound would have to be exactly 0 to pass, which proves only non-strict
        positivity. So the design W Gram could never be re-proved exactly
        without a re-solve (WP P2-02).

        When the weighted-sphere clearing is INACTIVE the cleared W *is* the
        plain associated form: homogeneous of even degree ``d`` over the full
        monomial basis, so the n pure powers yᵢ^d are all present and the same
        ε rows are admissible. ``exact_octant_target`` subtracts ε from ALL n
        pure powers unconditionally, so the design must emit all n or none —
        a partial set would silently disagree with the exact target.

        When the clearing is ACTIVE the block is ``∏ᵢ|xᵢ|^{k·ρ/rᵢ}·W``, which
        VANISHES on every axis. Subtracting ε there asks for a strictly
        negative pure-power coefficient, which no SOS form can have: the SDP
        would be spuriously infeasible. Those blocks keep the zero margin and a
        consumer must route them to a re-solve.

        Returns ``(rows_per_vertex, monomials_per_vertex, reason)``. ``reason``
        is the empty string when the ε rows were emitted, else the typed reason
        they were not.
        """
        n_vertices = len(w_monomials)
        empty_rows: List[List[int]] = [[] for _ in range(n_vertices)]
        empty_monos: List[Tuple[Tuple[int, ...], ...]] = [
            () for _ in range(n_vertices)]
        if not self.w_epsilon_rows:
            return empty_rows, empty_monos, "w_epsilon_rows_disabled"
        if int(clearing_power) > 0:
            return empty_rows, empty_monos, "weighted_sphere_clearing_active"

        rows_per_vertex: List[List[int]] = []
        monos_per_vertex: List[Tuple[Tuple[int, ...], ...]] = []
        for monos_k in w_monomials:
            degree = max((sum(m) for m in monos_k), default=0)
            if degree <= 0 or degree % 2 != 0:
                # Proposition 6/7 need a positive even homogeneity degree; a
                # block that has neither cannot carry a matching exact target.
                return empty_rows, empty_monos, "w_degree_not_positive_even"
            rows, monos = [], []
            for idx, mono in enumerate(monos_k):
                if sum(mono) != degree:
                    continue
                nz = [i for i, e in enumerate(mono) if e]
                if len(nz) == 1:
                    rows.append(idx)
                    monos.append(tuple(int(e) for e in mono))
            if len(rows) != self.n_vars:
                # Defensive: the W monomial support is a full monomial basis of
                # the block degree, so all n pure powers are always present.
                return empty_rows, empty_monos, "pure_power_rows_incomplete"
            rows_per_vertex.append(rows)
            monos_per_vertex.append(tuple(monos))
        return rows_per_vertex, monos_per_vertex, ""

    def _emit_split_block(
        self,
        *,
        constraints: List,
        alpha: Any,
        M: np.ndarray,
        monomials: Sequence[Tuple[int, ...]],
        gram_map: Dict[Tuple[int, ...], List[Tuple[int, int]]],
        zero_pairs: Dict[Tuple[int, ...], List[Tuple[int, int]]],
        eps_rows: Sequence[int],
        scaled_epsilon: float,
        basis: Sequence[Tuple[int, ...]],
        name: str,
        gram_vars: Optional[List] = None,
    ) -> None:
        """CVXPY emission of ONE y-parity-decomposed Gram block (WP P5-04).

        One PSD variable per parity class of ``basis``, in the canonical
        ``SplitBlock.class_order``, and one equality per surviving row over the
        within-class pair sums of every contributing class. The pair lists are
        deduplicated to the lower triangle by ``split_gram_block``, so the
        off-diagonal factor 2 is written out here — the dense branch gets it
        for free from ``build_gram_constraints`` listing both ordered pairs.

        ``YParityRefused`` propagates: the caller decided to block only after
        the symbolic guard passed, so a refusal here is a real disagreement
        between the symbolic support and the assembled one and must not be
        swallowed.
        """
        from .yparity import split_gram_block

        split = split_gram_block(
            np.asarray(M, dtype=np.float64), list(monomials), gram_map,
            list(eps_rows), float(scaled_epsilon), list(basis), name,
            zero_pairs=zero_pairs)
        order = split.class_order
        Qs: Dict[Tuple[int, ...], Any] = {}
        for key in order:
            d = int(split.sub_dims[key])
            suffix = ''.join(str(int(e)) for e in key)
            Q = cp.Variable((d, d), symmetric=True, name=f'{name}#p{suffix}')
            constraints.append(Q >> 0)
            if gram_vars is not None:
                gram_vars.append(Q)
            Qs[key] = Q
        for row in split.rows:
            lhs = (row.a_vals @ alpha[row.a_cols] if row.a_cols.size
                   else cp.Constant(0.0))
            terms = []
            for key in order:
                pair = row.class_pairs.get(key)
                if pair is None or not len(pair[0]):
                    continue
                Q = Qs[key]
                for k, l in zip(pair[0], pair[1]):
                    terms.append(Q[k, l] if k == l else 2.0 * Q[k, l])
            q_sum = sum(terms) if terms else cp.Constant(0.0)
            constraints.append(lhs - row.rhs == q_sum)

    def _emit_v_block(
        self,
        *,
        constraints: List,
        alpha: Any,
        term_specs: List[TermSpec],
        mu: Tuple[int, ...],
        signs: Tuple[int, ...],
        block_index: int,
        all_v_monomials: List[Tuple[int, ...]],
        v_gram_map: Dict[Tuple[int, ...], List[Tuple[int, int]]],
        v_zero_pairs: Dict[Tuple[int, ...], List[Tuple[int, int]]],
        v_basis: List[Tuple[int, ...]],
        n_v_basis: int,
        v_y_degree: int,
        scaled_epsilon: float,
        y_parity_blocks: bool = False,
        gram_vars: Optional[List] = None,
    ) -> None:
        """One V SOS block (associated form of ``signs``, ε on the pure powers).

        ``block_index`` is the FULL octant index (finding 3): a reduced run
        names its representative blocks exactly as the unreduced run names the
        same octants, so ``Q_V_*`` records stay comparable across the two.
        """
        M_V = build_v_coefficient_matrix(term_specs, mu, signs, all_v_monomials)

        # P5-04: y-parity block decomposition. Lossless in α (group averaging),
        # and posed here exactly as the MOSEK_DIRECT path poses it, so a
        # backend fallback changes the encoding and never the formulation.
        if y_parity_blocks:
            self._emit_split_block(
                constraints=constraints, alpha=alpha, M=M_V,
                monomials=all_v_monomials, gram_map=v_gram_map,
                zero_pairs=v_zero_pairs,
                eps_rows=v_epsilon_row_indices(
                    all_v_monomials, v_y_degree, self.n_vars),
                scaled_epsilon=scaled_epsilon, basis=v_basis,
                name=f'Q_V_{block_index}', gram_vars=gram_vars)
            return

        Q_V = cp.Variable((n_v_basis, n_v_basis), symmetric=True, name=f'Q_V_{block_index}')
        constraints.append(Q_V >> 0)
        if gram_vars is not None:
            gram_vars.append(Q_V)

        for mono_idx, mono in enumerate(all_v_monomials):
            v_coef = M_V[mono_idx, :] @ alpha

            # Epsilon for ||y||^d terms (scaled for numerical stability)
            eps_term = 0.0
            if sum(mono) == v_y_degree:
                for i in range(self.n_vars):
                    pure_mono = [0] * self.n_vars
                    pure_mono[i] = v_y_degree
                    if list(mono) == pure_mono:
                        eps_term = scaled_epsilon
                        break

            pairs = v_gram_map.get(mono, [])
            if pairs:
                q_sum = sum(Q_V[i, j] for i, j in pairs)
                constraints.append(v_coef - eps_term == q_sum)
            else:
                constraints.append(v_coef - eps_term == 0)

        # P2-09: every basis product OUTSIDE the target support gets an
        # explicit zero row, so no Gram entry is left free. Empty in the
        # homogeneous even-degree case, which is every V block here.
        for pairs in v_zero_pairs.values():
            constraints.append(sum(Q_V[i, j] for i, j in pairs) == 0)

    def _emit_w_block(
        self,
        *,
        constraints: List,
        alpha: Any,
        term_specs: List[TermSpec],
        mu: Tuple[int, ...],
        signs: Tuple[int, ...],
        block_index: int,
        v_idx: int,
        dyn_k: List[GeneralizedForm],
        w_monomials: List[List[Tuple[int, ...]]],
        w_bases: List[List[Tuple[int, ...]]],
        w_gram_maps: List[Dict[Tuple[int, ...], List[Tuple[int, int]]]],
        w_zero_pairs: List[Dict[Tuple[int, ...], List[Tuple[int, int]]]],
        w_eps_rows: List[List[int]],
        scaled_epsilon: float,
        y_parity_blocks: bool = False,
        gram_vars: Optional[List] = None,
    ) -> None:
        """One W SOS block for octant ``signs`` at inclusion vertex ``v_idx``.

        W = -dV/dt with weighted-sphere clearing for negative exponents.
        Verification: ||x||_r^k · W > 0 ⟺ W > 0 (since ||x||_r^k > 0 for x≠0).
        With vertices, the SAME alpha must make W SOS under every extremal
        dynamics; affinity in the disturbance covers the interval.

        ``block_index`` is the FULL octant index (finding 3), so the Gram name
        is ``Q_W_{octant}_{vertex}`` in reduced and unreduced runs alike.
        """
        M_W, _, _ = self._compute_w_coefficient_matrix_weighted_sphere(
            term_specs, mu, signs, dyn_k
        )
        # Note: clearing power and monomial support are the same for all
        # octants of one vertex, but M_W depends on the sign factors.

        if M_W is None:
            return
        monos_k = w_monomials[v_idx]
        basis_k = w_bases[v_idx]

        # P5-04, same decomposition as the V block above.
        if y_parity_blocks:
            self._emit_split_block(
                constraints=constraints, alpha=alpha, M=np.asarray(M_W),
                monomials=monos_k, gram_map=w_gram_maps[v_idx],
                zero_pairs=w_zero_pairs[v_idx], eps_rows=w_eps_rows[v_idx],
                scaled_epsilon=scaled_epsilon, basis=basis_k,
                name=f'Q_W_{block_index}_{v_idx}', gram_vars=gram_vars)
            return

        Q_W = cp.Variable((len(basis_k), len(basis_k)), symmetric=True,
                          name=f'Q_W_{block_index}_{v_idx}')
        constraints.append(Q_W >> 0)
        if gram_vars is not None:
            gram_vars.append(Q_W)

        w_gram_map = w_gram_maps[v_idx]
        # ε rows decided once in find_lyapunov_with_dynamics (P2-02), so this
        # encoding and the MOSEK_DIRECT one pose the SAME SDP. Empty when the
        # weighted-sphere clearing is active: the cleared W carries a common
        # monomial factor and vanishes on the coordinate axes, so an axis
        # ε-shift would make the SDP spuriously infeasible there.
        eps_set = set(w_eps_rows[v_idx])

        for mono_idx, mono in enumerate(monos_k):
            w_coef = M_W[mono_idx, :] @ alpha

            eps_term = scaled_epsilon if mono_idx in eps_set else 0.0

            pairs = w_gram_map.get(mono, [])
            if pairs:
                q_sum = sum(Q_W[i, j] for i, j in pairs)
                constraints.append(w_coef - eps_term == q_sum)
            else:
                constraints.append(w_coef - eps_term == 0)

        # P2-09, same invariant as the V block: no free Gram entry. ``mu`` here
        # is the W μ (doubled when the associated form's y-degree came out odd),
        # so this set is empty on every reachable design.
        for pairs in w_zero_pairs[v_idx].values():
            constraints.append(sum(Q_W[i, j] for i, j in pairs) == 0)

    def iter_designs_over_epsilons(
        self,
        epsilons: Sequence[float],
        *,
        normalize: bool = True,
        normalization_band_log2: Optional[int] = None,
        stop_on_candidate: bool = False,
        **solve_kwargs: Any,
    ) -> Iterator[EpsilonRung]:
        """Design the SAME problem once per design margin (WP P5-02).

        A generator, deliberately: the caller (the escalation loop of
        ``discovery.gf_discovery.discover_gf``) has the STRICT GATE, which is
        the only thing that can decide whether a rung is good enough, so it
        must be able to stop the ladder after the first rung that gates. Ask
        for three rungs and gate the first successfully and you pay for one
        solve, not three.

        ``self.epsilon`` is set per rung and restored on exit, including on an
        exception; nothing else about the optimizer is touched, so every rung
        poses the same library, the same encoding, the same y-parity blocking
        and the same screen policy as the unswept solve. Rung 0 at
        ``epsilon == self.epsilon`` is therefore byte-identical to
        :meth:`find_lyapunov_with_dynamics` — pinned by
        ``tests/test_epsilon_sweep.py``.

        Each rung is normalised for the gate (``verification.v_normalization``)
        unless ``normalize=False``: when the candidate's coefficient log-centre
        has drifted more than ``2 ** NORMALIZATION_BAND_LOG2`` away from 1, V,
        its design Gram bundle and its design margin are all multiplied by one
        exact power of two, so the gate's ABSOLUTE margin and its equally
        absolute residual budget can both see the certificate. The band is
        deliberately wide: the gate's measured scale window is only about five
        binades, and the design's own scale sits inside it on every row P5-02
        measured, so the step is a no-op on all twelve nominal systems at every
        rung of the shipped ladder.

        A rung whose design was refused by the design screen still yields its
        candidate, with ``screen_refused=True`` and a typed ``bundle_reason``.
        The screen filters the BUNDLE, not the candidate: dropping the
        candidate silently would turn a cheap float filter into a certification
        verdict, which decision D1 forbids. The caller must route such a
        candidate's exact phase to a re-solve witness and record that it did.

        ``stop_on_candidate=True`` stops the generator after the first rung
        that produced a candidate at all, without gating it — for measurement
        drivers that want the ladder, not the loop.
        """
        from .v_normalization import (NORMALIZATION_BAND_LOG2,
                                      normalize_for_gate)

        ladder = resolve_epsilon_ladder(epsilons)
        band = (NORMALIZATION_BAND_LOG2 if normalization_band_log2 is None
                else int(normalization_band_log2))
        saved = self.epsilon
        try:
            for index, eps in enumerate(ladder):
                self.epsilon = float(eps)
                t0 = time.time()
                try:
                    result = self.find_lyapunov_with_dynamics(**solve_kwargs)
                except Exception as exc:  # keep the ladder alive; type the rung
                    import traceback as _traceback
                    tb = _traceback.format_exc()
                    print(f"[joint_sos] design at eps={eps:g}: {exc}\n{tb}",
                          file=sys.stderr, flush=True)
                    yield EpsilonRung(
                        index=index, epsilon=float(eps),
                        epsilon_effective=float(eps), result=None, V_gf=None,
                        feasible=False, screen_refused=False,
                        solver_status="design_error",
                        message=f"{type(exc).__name__}: {exc}",
                        seconds=time.time() - t0, traceback=tb)
                    continue
                seconds = time.time() - t0
                status = str(result.solver_status or "")
                screen_refused = status.startswith("design_screen_refused")
                candidate = result.V_gf if (result.feasible or screen_refused) else None
                entries = list(result.gram_certificates or []) or None
                reason = ""
                if screen_refused:
                    # The screen already stripped the bundle (P2-04); say WHY,
                    # so the consumer routes to a re-solve witness knowingly.
                    entries = None
                    reason = ("design_screen_refused:"
                              + (result.message or "unrecorded"))
                elif candidate is not None and entries is None:
                    reason = ("no_design_grams (not MOSEK_DIRECT, or "
                              "capture_design_grams=False)")
                norm_record = None
                eps_eff = float(eps)
                if candidate is not None:
                    candidate, entries, eps_eff_opt, norm = normalize_for_gate(
                        candidate, gram_bundle=entries, epsilon=float(eps),
                        enabled=bool(normalize), band_log2=band)
                    norm_record = norm.as_record()
                    eps_eff = float(eps if eps_eff_opt is None else eps_eff_opt)
                yield EpsilonRung(
                    index=index, epsilon=float(eps), epsilon_effective=eps_eff,
                    result=result, V_gf=candidate,
                    feasible=bool(result.feasible),
                    screen_refused=screen_refused, solver_status=status,
                    message=str(result.message or ""), seconds=seconds,
                    gram_certificates=entries, bundle_reason=reason,
                    normalization=norm_record)
                if stop_on_candidate and candidate is not None:
                    return
        finally:
            self.epsilon = saved

    def find_lyapunov_with_dynamics(
        self,
        degree_m: int,
        dynamics_gf: List[GeneralizedForm],
        max_cross_terms: Optional[int] = None,
        active_terms: Optional[List[TermSpec]] = None,
        max_denominator: int = 2,
        dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]] = None,
    ) -> JointSOSResult:
        """
        Find Lyapunov coefficients with FULL V > 0 AND W > 0 constraints.

        This is the complete joint SOS optimization from the SOSTOOLS approach.

        Args:
            degree_m: GF homogeneity degree
            dynamics_gf: System dynamics as GF list [f_1, ..., f_n]
            max_cross_terms: Limit cross-terms
            active_terms: Use specific terms
            max_denominator: Max denominator for fractional exponents (default 2)
            dynamics_gf_vertices: Optional differential-inclusion vertices for
                robust design (e.g. matched-disturbance endpoints ±δ₀). The W
                SOS block is duplicated per vertex so the SAME V decreases under
                every extremal dynamics; W is affine in the disturbance, so
                endpoint feasibility covers the interval by convexity.
                None ⇒ nominal only.

        Returns:
            JointSOSResult with coefficients if both V > 0 and W > 0 are SOS

        This is a coefficient *designer*: it solves the standard numeric SDP
        (Sanchez-Moreno 2018 §5.3 / Algorithm 8) and returns the proposed V with
        the diagonal-coefficient margin enforced. A discovered candidate is always
        re-gated strictly by ``certify_discovered_gf`` (which uses the per-octant
        ε-margin SOS of ``verify_gf_sos_full``) before any stability claim.
        """
        start_time = time.time()
        vertices = dynamics_gf_vertices if dynamics_gf_vertices else [dynamics_gf]

        # Step 1: Get term structure
        if active_terms is not None:
            term_specs = active_terms
        else:
            diag_terms, cross_terms = enumerate_all_terms(
                degree_m, self.weights_r,
                max_denominator=max_denominator,
                include_signed=True,
                max_cross_terms=max_cross_terms or 50
            )
            term_specs = diag_terms + cross_terms

        # Step 1b (OPT-IN, default off): design-time parity reduction.
        # ``_plan_parity_reduction`` verifies (A1)/Lemma 10.2 symbolically and
        # either returns the even-parity sub-library plus the block reduction,
        # or refuses and hands back the untouched library. Nothing below this
        # point knows which of the two happened except through ``reduction``.
        parity_reduction = None
        parity_meta: Optional[dict] = None
        l1_weights_eff = self.l1_weights
        if self.parity_reduction:
            n_full_terms = len(term_specs)
            parity_reduction, term_specs, keep_indices, parity_meta = (
                self._plan_parity_reduction(
                    dynamics_gf=dynamics_gf,
                    dynamics_gf_vertices=dynamics_gf_vertices,
                    term_specs=term_specs,
                )
            )
            if parity_reduction is not None and l1_weights_eff is not None:
                # Finding 6: a caller datum indexed against the UNFILTERED
                # library is RE-MAPPED through the kept positions, never
                # truncated. A wrong length is the caller's bug and must not be
                # masked by the reduction, so it raises exactly as before.
                if len(l1_weights_eff) != n_full_terms:
                    raise ValueError(
                        f"l1_weights has length {len(l1_weights_eff)}, expected "
                        f"{n_full_terms}")
                l1_weights_eff = l1_weights_eff[np.asarray(keep_indices, dtype=int)]

        n_terms = len(term_specs)
        k, mu = compute_mu_exponents_with_dynamics(
            term_specs=term_specs,
            weights_r=self.weights_r,
            dynamics_gf=dynamics_gf,
            ensure_even_exponents=self.ensure_even_exponents,
        )

        # Compute V degree in y-space
        v_y_degree = 0
        for spec in term_specs:
            deg = sum(mu[i] * float(spec.exponents[i]) for i in range(self.n_vars))
            v_y_degree = max(v_y_degree, int(deg))

        # Apply coefficient scaling for numerical stability
        # For large systems (e.g., PID-SMC), natural coefficients are ~1e6-1e7
        # Scaling to ~1 makes the optimization numerically tractable
        scaled_epsilon = self.epsilon * self.coefficient_scale

        if self.verbose:
            print(f"JointSOS (full): {n_terms} terms, {2**self.n_vars} octants")
            print(f"  μ = {mu}, V y-degree = {v_y_degree}")
            if self.coefficient_scale != 1.0:
                print(f"  Coefficient scale: {self.coefficient_scale} (ε_scaled = {scaled_epsilon})")

        # Step 2: Create decision variables
        alpha = cp.Variable(n_terms, name='alpha')
        constraints = []

        # Track diagonal indices for scale normalization and add explicit lower
        # bounds in scaled space. The SOS margin is still enforced, but the
        # diagonal bounds provide a cheap, robust guard against solver-tolerance
        # false positives in the recovered coefficients.
        diag_indices = []
        for i, spec in enumerate(term_specs):
            if spec.is_diagonal():
                diag_indices.append(i)
                constraints.append(alpha[i] >= scaled_epsilon)

        # Step 3: Pre-compute W with weighted-sphere clearing (octant-independent
        # analysis), once PER VERTEX of the differential inclusion. The clearing
        # power, monomial support and Gram basis can differ between vertices
        # (e.g. the ±δ₀ constant adds ∂V/∂xₙ·δ₀ monomials absent nominally).
        octants = get_hyperoctant_signs(self.n_vars)
        first_signs = octants[0]  # Use first octant for analysis

        def _w_assembly(mu_w: Tuple[int, ...]):
            """(clearings, monomial supports, Gram bases) of the W blocks at μ_W."""
            clearings, monomials, bases = [], [], []
            for dyn_k in vertices:
                _M, clearing_k, monos_k = (
                    self._compute_w_coefficient_matrix_weighted_sphere(
                        term_specs, mu_w, first_signs, dyn_k))
                clearings.append(clearing_k)
                monomials.append(monos_k)
                deg_k = max(sum(m) for m in monos_k) if monos_k else 0
                bases.append(_sos_gram_basis(
                    self.n_vars, deg_k, homogeneous=(clearing_k == 0)))
            return clearings, monomials, bases

        w_mu = mu
        mu_policy = MU_POLICY_DESIGN
        w_clearing, w_monomials, w_bases = _w_assembly(w_mu)
        # P2-09: W is r-homogeneous, so with μᵢ = k·rᵢ every W monomial has the
        # SAME y-degree; if that degree is odd the associated form is
        # sign-indefinite on ℝⁿ and no SOS certificate exists for it. Doubling μ
        # for the W blocks alone is the even-exponent lift of Algorithm 7 (μᵢ·aᵢ
        # is an integer by construction, so 2μᵢ·aᵢ is even): the block becomes
        # homogeneous of even degree, the P2-02 ε rows apply, and the exact gate
        # rebuilds its target at the block's OWN μ, which the bundle carries per
        # entry. Without it the design silently emitted rows only for the target
        # support and left a Gram entry free (power_rate: Q[2,2] = 6.16e+04,
        # P2-04 contradiction 1). V keeps the design μ: its y-degree is already
        # even by ``ensure_even_exponents``.
        w_degrees = {max((sum(m) for m in monos), default=0)
                     for monos in w_monomials}
        if any(d > 0 and d % 2 for d in w_degrees):
            w_mu = tuple(2 * int(v) for v in mu)
            mu_policy = MU_POLICY_W_DOUBLED
            w_clearing, w_monomials, w_bases = _w_assembly(w_mu)
            if self.verbose:
                print(f"  W y-degree {sorted(w_degrees)} odd -> W μ doubled to "
                      f"{w_mu} ({MU_POLICY_W_DOUBLED})")
        clearing_power = max(w_clearing)
        all_w_monomials = w_monomials[0]
        w_y_degree = max(sum(m) for m in all_w_monomials) if all_w_monomials else 0

        # ε rows of the W blocks (P2-02). Decided ONCE here, before the backend
        # split, so the MOSEK_DIRECT encoding and the CVXPY encoding pose the
        # SAME SDP — a fallback must change the encoding, never the formulation.
        w_eps_rows, w_eps_monomials, w_eps_reason = self._w_epsilon_rows(
            w_monomials, clearing_power)
        if self.verbose and w_eps_reason:
            print(f"  W blocks carry NO ε rows ({w_eps_reason})")

        # P5-04: decided ONCE here, before the backend split, for the same
        # reason the ε rows are — a solver fallback must change the encoding,
        # never the formulation. The symbolic guard runs at each stage's own μ.
        y_parity_meta = self._plan_y_parity(
            term_specs=term_specs, mu=mu, w_mu=w_mu, vertices=vertices,
            clearing_power=clearing_power)
        if self.verbose and y_parity_meta["requested"]:
            print(f"  y-parity blocks: V {y_parity_meta['v_blocked']}, "
                  f"W {y_parity_meta['w_blocked']}"
                  + (f" (refused: {y_parity_meta['refusal']})"
                     if y_parity_meta["refusal"] else ""))

        if clearing_power > 0:
            if self.verbose:
                print(f"  Using weighted-sphere SOS for W (k={clearing_power})")

        # Build monomial bases. The V associated form is homogeneous (μ ∝ r), so its
        # degree-exact half basis suffices. The same holds for W when no weighted-
        # sphere clearing was applied (clearing can break homogeneity). See
        # _sos_gram_basis.
        v_basis = _sos_gram_basis(self.n_vars, v_y_degree, homogeneous=True)
        w_basis = w_bases[0]

        all_v_monomials = get_monomial_basis(self.n_vars, v_y_degree)

        n_v_basis = len(v_basis)
        n_w_basis = len(w_basis)

        if self.verbose:
            print(f"  V basis: {n_v_basis}, W basis: {n_w_basis}, W y-degree: {w_y_degree}"
                  + (f", vertices: {len(vertices)}" if len(vertices) > 1 else ""))
            if parity_reduction is not None:
                print(f"  parity blocks: V {parity_reduction.n_v_blocks}/"
                      f"{parity_reduction.n_octants}, W "
                      f"{parity_reduction.n_w_blocks}/"
                      f"{parity_reduction.n_octants * parity_reduction.n_endpoints}")

        # Direct Task-API backend: encodes the SAME SDP through mosek.Task,
        # bypassing CVXPY canonicalization (which OOMs on the 2^n-octant Gram
        # lifting at n=5). Returns None only when the backend errored/stalled
        # AND fallback is allowed — then the CVXPY path below proceeds (its
        # candidate chain skips the unknown name MOSEK_DIRECT). For very large
        # designs pass allow_solver_fallback=False to fail fast instead of
        # re-entering the canonicalization memory wall.
        if self.solver == 'MOSEK_DIRECT':
            direct_result = self._find_lyapunov_direct(
                degree_m=degree_m,
                term_specs=term_specs,
                mu=mu,
                v_y_degree=v_y_degree,
                v_basis=v_basis,
                all_v_monomials=all_v_monomials,
                vertices=vertices,
                w_monomials=w_monomials,
                w_bases=w_bases,
                w_eps_rows=w_eps_rows,
                w_eps_monomials=w_eps_monomials,
                w_eps_reason=w_eps_reason,
                w_mu=w_mu,
                mu_policy=mu_policy,
                clearing_power=clearing_power,
                diag_indices=diag_indices,
                scaled_epsilon=scaled_epsilon,
                start_time=start_time,
                parity_reduction=parity_reduction,
                parity_meta=parity_meta,
                y_parity_meta=y_parity_meta,
                l1_weights=l1_weights_eff,
            )
            if direct_result is not None:
                return direct_result

        # Step 4: Add the V and W SOS constraints, block by block.
        octants = get_hyperoctant_signs(self.n_vars)

        # The pair maps are octant-independent (only the M-matrix sign factors
        # vary across octants); build them once instead of 2^n times.
        v_gram_map = build_gram_constraints(v_basis, all_v_monomials, self.n_vars)
        w_gram_maps = [
            build_gram_constraints(w_bases[v], w_monomials[v], self.n_vars)
            for v in range(len(vertices))
        ]
        # P2-09: the complement of the two maps above — basis products the
        # target support does not carry, which would otherwise be free Gram
        # entries. Octant-independent for the same reason the pair maps are.
        v_zero_pairs = build_gram_completion_map(
            v_basis, all_v_monomials, self.n_vars)
        w_zero_pairs = [
            build_gram_completion_map(w_bases[v], w_monomials[v], self.n_vars)
            for v in range(len(vertices))
        ]

        # Collected only when the objective needs them (P5-05); the emitted
        # blocks are byte-identical either way.
        gram_vars: Optional[List] = (
            [] if self.objective == "min_gram_trace" else None)

        v_emit_kwargs = dict(
            constraints=constraints, alpha=alpha, term_specs=term_specs, mu=mu,
            all_v_monomials=all_v_monomials, v_gram_map=v_gram_map,
            v_zero_pairs=v_zero_pairs, v_basis=v_basis,
            n_v_basis=n_v_basis, v_y_degree=v_y_degree,
            scaled_epsilon=scaled_epsilon,
            y_parity_blocks=bool(y_parity_meta["v_blocked"]),
            gram_vars=gram_vars,
        )
        w_emit_kwargs = dict(
            constraints=constraints, alpha=alpha, term_specs=term_specs,
            mu=w_mu,
            w_monomials=w_monomials, w_bases=w_bases, w_gram_maps=w_gram_maps,
            w_zero_pairs=w_zero_pairs,
            w_eps_rows=w_eps_rows, scaled_epsilon=scaled_epsilon,
            y_parity_blocks=bool(y_parity_meta["w_blocked"]),
            gram_vars=gram_vars,
        )

        if parity_reduction is None:
            # UNREDUCED (the default): V and the per-vertex W blocks of one
            # octant are emitted together, in the historical order.
            for oct_idx, signs in enumerate(octants):
                # === V SOS Constraint ===
                self._emit_v_block(signs=signs, block_index=oct_idx, **v_emit_kwargs)

                # === W SOS Constraint (one block per inclusion vertex) ===
                for v_idx, dyn_k in enumerate(vertices):
                    self._emit_w_block(signs=signs, block_index=oct_idx,
                                       v_idx=v_idx, dyn_k=dyn_k, **w_emit_kwargs)
        else:
            # REDUCED: the two stages run over DIFFERENT index sets, so the
            # single octant loop is split in two (finding 1).
            #
            #   V stage: 2^(n-1) representative octants. V does not see the
            #     disturbance, so this is the same set nominally and perturbed
            #     (Proposition 10.1 part 2 / Proposition 10.3 part 3).
            #   W stage: iterates PAIRS. Nominally one block per representative
            #     octant; under a two-endpoint inclusion BOTH endpoints of every
            #     representative octant, because the involution is
            #     (s, σ) ↦ (−s, −σ) and identifying (s, σ) with (−s, σ) would
            #     drop a real constraint (Corollary 10.4).
            #
            # Block indices stay FULL octant indices in both stages (finding 3).
            for signs in parity_reduction.v_blocks():
                self._emit_v_block(signs=signs,
                                   block_index=_sym.octant_index(signs),
                                   **v_emit_kwargs)

            for signs, endpoint in parity_reduction.w_blocks():
                # Endpoints are POSITIONAL indices into ``vertices`` and are
                # used as such only because verify_endpoint_swap passed
                # (finding 2). Nominally there is no endpoint index and the
                # single vertex is index 0.
                v_idx = 0 if endpoint is None else int(endpoint)
                self._emit_w_block(signs=signs,
                                   block_index=_sym.octant_index(signs),
                                   v_idx=v_idx, dyn_k=vertices[v_idx],
                                   **w_emit_kwargs)

        # Step 5: Solve
        # Fix first diagonal TRUE coefficient to 1 for scale normalization.
        # Optimization variable alpha is scaled (alpha = coefficient_scale * alpha_true),
        # so the normalized scaled value must equal coefficient_scale.
        if diag_indices:
            constraints.append(alpha[diag_indices[0]] == self.coefficient_scale)

        # Note: We don't add explicit upper/lower bounds on alpha.
        # The scale normalization (alpha[0] == 1) and PSD constraints
        # sufficiently constrain the feasible region.

        # Add user-specified coefficient bounds (e.g., sign constraints)
        for idx, (lb, ub) in self.coefficient_bounds.items():
            if 0 <= idx < n_terms:
                if lb is not None:
                    constraints.append(alpha[idx] >= lb * self.coefficient_scale)
                if ub is not None:
                    constraints.append(alpha[idx] <= ub * self.coefficient_scale)

        # Enforce non-negative coefficients on abs-type cross-terms for quasiconvexity.
        if self.nonneg_abs_cross:
            abs_cross_indices = []
            for i, spec in enumerate(term_specs):
                if not spec.is_diagonal() and all(tt == 'abs' for tt in spec.term_types):
                    abs_cross_indices.append(i)
                    constraints.append(alpha[i] >= 0)
            if self.verbose:
                print(f"  nonneg_abs_cross: constrained {len(abs_cross_indices)} abs cross-terms >= 0 (indices {abs_cross_indices})")

        # Domination constraints: for each (abs_idx, signed_idx) pair,
        # enforce alpha[abs_idx] >= |alpha[signed_idx]| so the abs duplicate
        # always dominates the signed term (sufficient for quasiconvexity).
        if self.domination_pairs:
            for abs_idx, signed_idx in self.domination_pairs:
                if 0 <= abs_idx < n_terms and 0 <= signed_idx < n_terms:
                    constraints.append(alpha[abs_idx] + alpha[signed_idx] >= 0)
                    constraints.append(alpha[abs_idx] - alpha[signed_idx] >= 0)
            if self.verbose:
                print(f"  domination_pairs: {len(self.domination_pairs)} pairs enforced")

        # Linear coefficient constraints on true-scale alpha.
        for row, lb, ub in self.linear_coefficient_constraints:
            row_arr = np.asarray(row, dtype=np.float64).reshape(-1)
            if row_arr.size != n_terms:
                raise ValueError(
                    f"linear coefficient constraint row has length {row_arr.size}, expected {n_terms}"
                )
            expr = row_arr @ alpha
            if lb is not None:
                constraints.append(expr >= float(lb) * self.coefficient_scale)
            if ub is not None:
                constraints.append(expr <= float(ub) * self.coefficient_scale)

        # Add midpoint convexity cuts: Σ αᵢ δᵢ(x,y) ≤ -margin for each sampled violation.
        if self.convexity_cuts is not None and len(self.convexity_cuts) > 0:
            cuts = self.convexity_cuts
            margin = self.convexity_cut_margin
            if cuts.shape[1] == n_terms:
                for k in range(cuts.shape[0]):
                    constraints.append(cuts[k] @ alpha <= -margin)
                if self.verbose:
                    print(f"  convexity_cuts: added {cuts.shape[0]} constraints (margin={margin:.1e})")
            elif self.verbose:
                print(f"  convexity_cuts: shape mismatch ({cuts.shape[1]} vs {n_terms} terms), skipping")

        # Build objective: L1 penalty on cross-terms for sparsity (if enabled)
        if self.reference_coefficients is not None and len(self.reference_coefficients) == n_terms:
            ref_scaled = self.reference_coefficients * self.coefficient_scale
            obj = cp.norm1(alpha - ref_scaled)
            prob = cp.Problem(cp.Minimize(obj), constraints)
            if self.verbose:
                print(f"  Minimizing L1 distance to reference coefficients")
        elif l1_weights_eff is not None:
            # Weighted ℓ₁ over the cross terms (reweighted-ℓ₁ hook; see __init__).
            # alpha is the SCALED variable, so dividing the weights by the scale
            # makes the penalty act on TRUE coefficients: w·|α_true| = (w/s)·|α_scaled|.
            # ``l1_weights_eff`` is ``self.l1_weights`` verbatim unless the parity
            # reduction is active, in which case it was re-mapped onto the even
            # sub-library (finding 6). Σ wⱼ|αⱼ| is invariant under the coefficient
            # reflection, so it does not endanger assumption (A3).
            if len(l1_weights_eff) != n_terms:
                raise ValueError(
                    f"l1_weights has length {len(l1_weights_eff)}, expected {n_terms}")
            cross_indices = [i for i in range(n_terms) if i not in diag_indices]
            if cross_indices:
                w = l1_weights_eff[cross_indices] / self.coefficient_scale
                obj = cp.sum(cp.multiply(w, cp.abs(alpha[cross_indices])))
                prob = cp.Problem(cp.Minimize(obj), constraints)
                if self.verbose:
                    print(f"  Weighted-ℓ₁ objective over {len(cross_indices)} cross terms")
            else:
                prob = cp.Problem(cp.Minimize(0), constraints)
        elif self.maximize_diagonal and diag_indices:
            prob = cp.Problem(cp.Maximize(cp.sum(alpha[diag_indices])), constraints)
            if self.verbose:
                print(f"  Maximizing diagonal dominance ({len(diag_indices)} diag terms)")
        elif self.use_l1_regularization and self.l1_lambda > 0:
            # Cross-term indices: exclude diag and abs duplicates in domination pairs
            dom_abs_indices = {abs_idx for abs_idx, _ in self.domination_pairs}
            cross_indices = [i for i in range(n_terms)
                           if i not in diag_indices and i not in dom_abs_indices]

            if cross_indices:
                # L1 penalty only on cross-term coefficients
                # This encourages small cross-terms to go to zero
                # Note: We use scaled alpha values, so lambda is relative to scaled coefficients
                alpha_cross = alpha[cross_indices]
                l1_penalty = self.l1_lambda * cp.norm1(alpha_cross)
                prob = cp.Problem(cp.Minimize(l1_penalty), constraints)

                if self.verbose:
                    print(f"  Using L1 regularization (lambda={self.l1_lambda}, {len(cross_indices)} cross-terms)")
            else:
                prob = cp.Problem(cp.Minimize(0), constraints)
        else:
            prob = cp.Problem(cp.Minimize(0), constraints)

        # P5-05: the low-rank surrogate, ADDED to whatever the chain above
        # chose. Σ_blocks tr Q is the nuclear norm on the PSD cone and is
        # bounded below by 0 there, so it cannot make a feasible problem
        # unbounded. Implemented here as well as on the MOSEK_DIRECT path so a
        # backend fallback changes the ENCODING and never the formulation
        # (the same invariant the W ε rows of P2-02 are decided under).
        if self.objective == "min_gram_trace" and gram_vars:
            trace_term = sum(cp.trace(Q) for Q in gram_vars)
            prior = prob.objective
            prior_expr = (-prior.expr if isinstance(prior, cp.Maximize)
                          else prior.expr)
            prob = cp.Problem(cp.Minimize(prior_expr + trace_term), constraints)
            if self.verbose:
                print(f"  min Σ tr Q over {len(gram_vars)} Gram blocks "
                      f"(added to the objective above)")

        # Solver parameters for early stopping
        scs_params = {
            'max_iters': self.scs_max_iters,
            'eps_abs': 1e-5,
            'eps_rel': 1e-5,
        }
        if self.solver_timeout is not None:
            scs_params['time_limit_secs'] = self.solver_timeout

        solved, solver_used, solver_attempts, solver_error = self._solve_problem(
            prob,
            scs_params=scs_params,
        )
        solver_used_str = str(solver_used or "")
        fallback_used = bool(solver_used_str and solver_used_str != self.solver)
        if not solved:
            return JointSOSResult(
                feasible=False,
                solver_status="solver_error",
                solver_requested=self.solver,
                solver_used=solver_used_str,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message=str(solver_error or "Solver error"),
                solve_time=time.time() - start_time,
                term_specs=term_specs,
                parity_reduction=parity_meta,
            )

        # Check if SCS hit max iterations (indicates non-convergence)
        if prob.status == 'optimal_inaccurate' and self.verbose:
            print(f"  Warning: Solver returned 'optimal_inaccurate' - solution may be imprecise")

        # TRUST BOUNDARY: accepting OPTIMAL_INACCURATE here is sound because this
        # is the DESIGN side — the returned V is only a proposal. Every stability
        # claim must pass the strict gate (certify_discovered_gf → verify_gf_sos_full
        # with a-posteriori Gram validation), which never trusts solver status.
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            alpha_scaled = (
                np.asarray(alpha.value, dtype=np.float64).reshape(-1)
                if alpha.value is not None else None
            )
            return self._package_design_result(
                alpha_scaled=alpha_scaled,
                term_specs=term_specs,
                degree_m=degree_m,
                clearing_power=clearing_power,
                status_string=str(prob.status),
                solver_used=solver_used_str,
                solver_attempts=solver_attempts,
                fallback_used=fallback_used,
                start_time=start_time,
                parity_meta=parity_meta,
                y_parity_meta=y_parity_meta,
            )
        else:
            # Extract dual info for infeasible problem to guide term swapping
            dual_info = self._extract_dual_info(
                prob, constraints, n_terms, len(octants), term_specs
            )
            return JointSOSResult(
                feasible=False,
                solver_status=f"{prob.status}:{solver_used_str}" if solver_used_str else str(prob.status),
                solver_requested=self.solver,
                solver_used=solver_used_str,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message=f"Infeasible: {prob.status}",
                solve_time=time.time() - start_time,
                dual_info=dual_info,
                term_specs=term_specs,
                parity_reduction=parity_meta,
            )

    def _package_design_result(
        self,
        *,
        alpha_scaled: Optional[np.ndarray],
        term_specs: List[TermSpec],
        degree_m: int,
        clearing_power: int,
        status_string: str,
        solver_used: str,
        solver_attempts: List[str],
        fallback_used: bool,
        start_time: float,
        gram_min_eig: Optional[float] = None,
        gram_residual_l1: Optional[float] = None,
        gram_certificates: Optional[List[dict]] = None,
        parity_meta: Optional[dict] = None,
        y_parity_meta: Optional[dict] = None,
    ) -> JointSOSResult:
        """Shared post-processing of a successful design solve (any backend).

        Unscales the coefficients, runs the diagonal sanity check, and builds
        the proposal ``JointSOSResult``. Backend-neutral: the CVXPY path and
        the MOSEK_DIRECT path both call this, so the trust boundary and the
        validation behavior cannot drift apart.

        ``gram_certificates`` is the design-stage Gram bundle
        (``GRAM_BUNDLE_SCHEMA``); it rides along on the FEASIBLE result only,
        and is ``None`` for the CVXPY path, which recovers no Grams. It is
        data, never a verdict: nothing here or downstream accepts a candidate
        because a bundle is present.
        """
        n_terms = len(term_specs)
        if alpha_scaled is None:
            return JointSOSResult(
                feasible=False,
                solver_status=f"{status_string}:{solver_used}" if solver_used else status_string,
                solver_requested=self.solver,
                solver_used=solver_used,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message="Solver reported optimal status but returned no coefficient values",
                solve_time=time.time() - start_time,
                term_specs=term_specs,
                parity_reduction=parity_meta,
                y_parity=y_parity_meta,
            )
        coef_vals = np.asarray(alpha_scaled, dtype=np.float64).reshape(-1) / self.coefficient_scale
        diag_ok, diag_msg = _validate_diagonal_coefficients_true_scale(
            term_specs,
            coef_vals,
            self.epsilon,
        )
        if not diag_ok:
            return JointSOSResult(
                feasible=False,
                solver_status=f"{status_string}:{solver_used}" if solver_used else status_string,
                solver_requested=self.solver,
                solver_used=solver_used,
                solver_fallback_used=fallback_used,
                solver_attempts=solver_attempts,
                message=f"Post-solve diagonal sanity check failed: {diag_msg}",
                solve_time=time.time() - start_time,
                term_specs=term_specs,
                parity_reduction=parity_meta,
                y_parity=y_parity_meta,
            )
        coef_dict = {f'alpha{i}': float(coef_vals[i]) for i in range(n_terms)}

        gf_terms = [
            GFTerm(
                coefficient=float(coef_vals[i]),
                exponents=spec.exponents,
                term_types=spec.term_types
            )
            for i, spec in enumerate(term_specs)
        ]

        V_gf = GeneralizedForm(
            degree_m=degree_m,
            weights_r=self.weights_r,
            terms=gf_terms
        )

        # Standard SOS certificate (proposal): the numeric SDP returned a PSD
        # Gram for every octant's V and (cleared) W with the diagonal margin
        # enforced — the accepted method (Sanchez-Moreno §5.3 / Algorithm 8),
        # with the SDP solver as the oracle. A discovered candidate is re-gated
        # strictly by certify_discovered_gf before any stability claim.
        message = "Joint SOS feasible (V > 0 AND W > 0), standard numeric SOS"
        if clearing_power > 0:
            message += f" [weighted-sphere k={clearing_power}]"
        # The candidate below is an ORDINARY certificate proposal regardless of
        # the formulation it came from: full coefficient vector, full V_gf. The
        # parity record only says how the SDP was posed (theory_notes.md §10.6).
        if parity_meta is not None and parity_meta.get("mode") in ("nominal", "paired"):
            message += f" [parity reduction: {parity_meta['mode']}]"

        return JointSOSResult(
            feasible=True,
            coefficients=coef_dict,
            V_gf=V_gf,
            solver_status=f"standard_sos:{solver_used}" if solver_used else "standard_sos",
            solver_requested=self.solver,
            solver_used=solver_used,
            solver_fallback_used=fallback_used,
            solver_attempts=solver_attempts,
            message=message,
            solve_time=time.time() - start_time,
            term_specs=term_specs,
            gram_min_eig=gram_min_eig,
            gram_residual_l1=gram_residual_l1,
            gram_certificates=gram_certificates,
            parity_reduction=parity_meta,
            y_parity=y_parity_meta,
            design_objective=self.objective,
        )

    def _find_lyapunov_direct(
        self,
        *,
        degree_m: int,
        term_specs: List[TermSpec],
        mu: Tuple[int, ...],
        v_y_degree: int,
        v_basis: List[Tuple[int, ...]],
        all_v_monomials: List[Tuple[int, ...]],
        vertices: List[List[GeneralizedForm]],
        w_monomials: List[List[Tuple[int, ...]]],
        w_bases: List[List[Tuple[int, ...]]],
        w_eps_rows: List[List[int]],
        w_eps_monomials: List[Tuple[Tuple[int, ...], ...]],
        w_eps_reason: str,
        w_mu: Tuple[int, ...],
        mu_policy: str,
        clearing_power: int,
        diag_indices: List[int],
        scaled_epsilon: float,
        start_time: float,
        parity_reduction: Optional[Any] = None,
        parity_meta: Optional[dict] = None,
        y_parity_meta: Optional[dict] = None,
        l1_weights: Optional[np.ndarray] = None,
    ) -> Optional[JointSOSResult]:
        """MOSEK Task-API encoding of the joint design SDP (no CVXPY).

        Reuses the SAME matrix builders as the CVXPY path
        (``build_v_coefficient_matrix``, ``build_gram_constraints``,
        ``_compute_w_coefficient_matrix_weighted_sphere``) — only the SDP
        encoding differs. Implements exactly the option subset the benchmark
        scripts exercise (feasibility, ``l1_weights``, vertices, diagonal
        pinning/bounds); anything else raises ``NotImplementedError``.

        ``parity_reduction`` (default ``None`` = unreduced) selects the reduced
        block set exactly as the CVXPY path does: the V stage runs over the
        2^(n-1) representative octants and the W stage over the reduction's
        (octant, endpoint) PAIRS. ``term_specs`` is already the even
        sub-library when it is set, and ``l1_weights`` already re-mapped onto
        it. Block names keep the FULL octant index so the ``V_oct*`` /
        ``W_oct*_v*`` records line up with an unreduced run (finding 3).

        Returns ``None`` iff the backend errored or stalled AND solver
        fallback is allowed — the caller then proceeds with the CVXPY path.
        """
        if l1_weights is None:
            l1_weights = self.l1_weights
        unsupported = {
            'maximize_diagonal': self.maximize_diagonal,
            'reference_coefficients': self.reference_coefficients is not None,
            'use_l1_regularization': self.use_l1_regularization,
            'coefficient_bounds': bool(self.coefficient_bounds),
            'nonneg_abs_cross': self.nonneg_abs_cross,
            'domination_pairs': bool(self.domination_pairs),
            'linear_coefficient_constraints': bool(self.linear_coefficient_constraints),
            'convexity_cuts': (self.convexity_cuts is not None
                               and len(self.convexity_cuts) > 0),
        }
        active = sorted(k for k, v in unsupported.items() if v)
        if active:
            raise NotImplementedError(
                f"MOSEK_DIRECT backend does not support: {', '.join(active)} "
                f"(use a CVXPY solver for these options)")

        from .mosek_direct import (
            MOSEK_AVAILABLE,
            blocks_from_assembly,
            blocks_from_split,
            solve_joint_sos_direct,
        )
        from .yparity import (
            YParityRefused,
            dense_record,
            psd_scalars,
            split_gram_block,
        )

        if y_parity_meta is None:
            y_parity_meta = {"requested": False, "v_blocked": False,
                             "w_blocked": False, "refusal": "not_requested",
                             "detail": ""}

        n_terms = len(term_specs)
        attempts: List[str] = []
        if not MOSEK_AVAILABLE:
            attempts.append("MOSEK_DIRECT:unavailable")
            if self.allow_solver_fallback:
                return None
            return JointSOSResult(
                feasible=False,
                solver_status="solver_error",
                solver_requested=self.solver,
                solver_used="",
                solver_fallback_used=False,
                solver_attempts=attempts,
                message="MOSEK python API not importable",
                solve_time=time.time() - start_time,
                term_specs=term_specs,
                parity_reduction=parity_meta,
            )

        # ε rows: pure powers y_i^{v_y_degree} — same predicate as the CVXPY
        # loop, and now literally the same function (P5-04).
        eps_rows = v_epsilon_row_indices(
            all_v_monomials, v_y_degree, self.n_vars)

        v_gram_map = build_gram_constraints(v_basis, all_v_monomials, self.n_vars)
        w_gram_maps = [
            build_gram_constraints(w_bases[v], w_monomials[v], self.n_vars)
            for v in range(len(vertices))
        ]
        # P2-09: the out-of-support basis products, emitted as explicit zero
        # rows so no Gram entry is a free variable. Same object as the CVXPY
        # path builds, so the two encodings keep posing the SAME SDP.
        v_zero_pairs = build_gram_completion_map(
            v_basis, all_v_monomials, self.n_vars)
        w_zero_pairs = [
            build_gram_completion_map(w_bases[v], w_monomials[v], self.n_vars)
            for v in range(len(vertices))
        ]

        # Gram bundle assembly (GRAM_BUNDLE_SCHEMA). The per-block target is
        # M·alpha, and alpha is only known after the solve, so each block keeps
        # the SPARSE nonzero rows of its M instead of M itself: for V every term
        # maps to one monomial, and for W to the few Lie-derivative monomials,
        # so this is O(nnz) and not the O(n_monomials × n_terms) dense matrix
        # (which at n = 5 would be gigabytes if held for all 2^n blocks).
        capture = bool(self.capture_design_grams)
        v_eps_monomials = tuple(all_v_monomials[i] for i in eps_rows)

        def _sparse_rows(M: np.ndarray, monomials: Sequence[Tuple[int, ...]]):
            rows = []
            for r, mono in enumerate(monomials):
                cols = np.nonzero(M[r, :])[0]
                if cols.size:
                    rows.append((tuple(int(e) for e in mono),
                                 cols.astype(np.int32),
                                 np.asarray(M[r, cols], dtype=np.float64)))
            return rows

        # P5-04: block assembly is parameterised by the y-parity decision so
        # a runtime refusal can rebuild the WHOLE design dense, rather than
        # leaving a half-blocked encoding behind.
        def _v_block(signs: Tuple[int, ...], oct_idx: int,
                     parity_class: Optional[dict] = None,
                     blocked: bool = False):
            M_V = build_v_coefficient_matrix(term_specs, mu, signs, all_v_monomials)
            name = f"V_oct{oct_idx}"
            split = None
            if blocked:
                split = split_gram_block(
                    M_V, all_v_monomials, v_gram_map, eps_rows,
                    scaled_epsilon, v_basis, name, zero_pairs=v_zero_pairs)
                emitted = blocks_from_split(split)
            else:
                emitted = [blocks_from_assembly(
                    M_V, all_v_monomials, v_gram_map, eps_rows, scaled_epsilon,
                    dim=len(v_basis), name=name, zero_pairs=v_zero_pairs)]
            meta = None
            if capture:
                meta = {
                    "block": "V",
                    "name": name,
                    "signs": tuple(int(s) for s in signs),
                    "octant_index": int(oct_idx),
                    "endpoint": None,
                    "parity_class": parity_class,
                    # TRUE-scale margin: the SDP subtracts scaled_epsilon from
                    # the scaled target, and the bundle divides both by the
                    # coefficient scale.
                    "eps": float(self.epsilon),
                    "eps_reason": "",
                    "mu_policy": MU_POLICY_DESIGN,
                    "_mu": tuple(int(v) for v in mu),
                    "y_degree": int(v_y_degree),
                    "clearing_power": 0,
                    "basis": [tuple(int(e) for e in b) for b in v_basis],
                    "_rows": _sparse_rows(M_V, all_v_monomials),
                    "_eps_monomials": v_eps_monomials,
                    "y_parity": (split.record() if split is not None
                                 else dense_record(len(v_basis),
                                                   y_parity_meta["refusal"])),
                }
            return emitted, meta, split

        def _w_block(signs: Tuple[int, ...], oct_idx: int, v_idx: int,
                     dyn_k: List[GeneralizedForm],
                     parity_class: Optional[dict] = None,
                     blocked: bool = False):
            M_W, _, _ = self._compute_w_coefficient_matrix_weighted_sphere(
                term_specs, w_mu, signs, dyn_k)
            if M_W is None:
                return None, None, None
            M_W = np.asarray(M_W, dtype=np.float64)
            # ε rows decided once in find_lyapunov_with_dynamics (P2-02): the
            # same pure-power margin the V block carries when the weighted-
            # sphere clearing is inactive, and nothing when it is active (the
            # cleared W vanishes on the axes). ``w_eps_reason`` is the typed
            # reason a block carries no margin, and is what routes a consumer
            # to a re-solve instead of a bare ``epsilon_not_positive``.
            eps_rows_k = w_eps_rows[v_idx]
            eps_k = float(self.epsilon) if eps_rows_k else 0.0
            name = f"W_oct{oct_idx}_v{v_idx}"
            eps_scaled_k = scaled_epsilon if eps_rows_k else 0.0
            split = None
            if blocked:
                split = split_gram_block(
                    M_W, w_monomials[v_idx], w_gram_maps[v_idx], eps_rows_k,
                    eps_scaled_k, w_bases[v_idx], name,
                    zero_pairs=w_zero_pairs[v_idx])
                emitted = blocks_from_split(split)
            else:
                emitted = [blocks_from_assembly(
                    M_W, w_monomials[v_idx],
                    w_gram_maps[v_idx], eps_rows_k, eps_scaled_k,
                    dim=len(w_bases[v_idx]), name=name,
                    zero_pairs=w_zero_pairs[v_idx])]
            meta = None
            if capture:
                meta = {
                    "block": "W",
                    "name": name,
                    "signs": tuple(int(s) for s in signs),
                    "octant_index": int(oct_idx),
                    "endpoint": int(v_idx),
                    "parity_class": parity_class,
                    "eps": eps_k,
                    "eps_reason": w_eps_reason if not eps_rows_k else "",
                    "mu_policy": mu_policy,
                    "_mu": tuple(int(v) for v in w_mu),
                    "y_degree": int(max((sum(m) for m in w_monomials[v_idx]),
                                        default=0)),
                    "clearing_power": int(clearing_power),
                    "basis": [tuple(int(e) for e in b) for b in w_bases[v_idx]],
                    "_rows": _sparse_rows(M_W, w_monomials[v_idx]),
                    "_eps_monomials": w_eps_monomials[v_idx],
                    "y_parity": (split.record() if split is not None
                                 else dense_record(len(w_bases[v_idx]),
                                                   y_parity_meta["refusal"])),
                }
            return emitted, meta, split

        def _assemble(v_blocked: bool, w_blocked: bool):
            """(mosek blocks, groups) for the whole design at one blocking."""
            blocks: List[Any] = []
            groups: List[dict] = []

            def _add(built) -> None:
                emitted, meta, split = built
                if not emitted:
                    return
                groups.append({
                    "meta": meta, "start": len(blocks), "count": len(emitted),
                    "split": split,
                    "dense_dim": (int(split.dense_dim) if split is not None
                                  else int(emitted[0].dim)),
                })
                blocks.extend(emitted)

            if parity_reduction is None:
                for oct_idx, signs in enumerate(get_hyperoctant_signs(self.n_vars)):
                    _add(_v_block(signs, oct_idx, blocked=v_blocked))
                    for v_idx, dyn_k in enumerate(vertices):
                        _add(_w_block(signs, oct_idx, v_idx, dyn_k,
                                      blocked=w_blocked))
            else:
                # Same split as the CVXPY path (finding 1): V over the octant
                # representatives, W over the (octant, endpoint) pairs.
                p_mode = "paired" if parity_reduction.has_endpoints else "nominal"
                for signs in parity_reduction.v_blocks():
                    v_partner = _sym.flip_octant(tuple(signs))
                    _add(_v_block(signs, _sym.octant_index(signs), parity_class={
                        "mode": p_mode, "role": "representative",
                        "partner_octant_index": _sym.octant_index(v_partner),
                        "partner_endpoint": None,
                    }, blocked=v_blocked))
                for signs, endpoint in parity_reduction.w_blocks():
                    v_idx = 0 if endpoint is None else int(endpoint)
                    p_signs, p_endpoint = parity_reduction.partner_label(
                        (tuple(signs), endpoint))
                    _add(_w_block(signs, _sym.octant_index(signs), v_idx,
                                  vertices[v_idx], parity_class={
                                      "mode": p_mode, "role": "representative",
                                      "partner_octant_index": _sym.octant_index(p_signs),
                                      "partner_endpoint": (None if p_endpoint is None
                                                           else int(p_endpoint)),
                                  }, blocked=w_blocked))
            return blocks, groups

        # P5-04: the symbolic guard already passed, so a YParityRefused here is
        # a disagreement between the symbolic support and the ASSEMBLED one.
        # It is never resolved in favour of blocking: the whole design is
        # rebuilt dense and the refusal is recorded on the result.
        try:
            blocks, groups = _assemble(bool(y_parity_meta["v_blocked"]),
                                       bool(y_parity_meta["w_blocked"]))
        except YParityRefused as exc:
            y_parity_meta = dict(y_parity_meta)
            # split_gram_block's message opens with the block name, which is
            # what says whether the V or the W assembly disagreed.
            y_parity_meta["refusal"] = (
                "assembly_refused_v" if str(exc).startswith("V_")
                else "assembly_refused_w")
            y_parity_meta["detail"] = str(exc)
            y_parity_meta["v_blocked"] = False
            y_parity_meta["w_blocked"] = False
            blocks, groups = _assemble(False, False)

        n_dense_scalars = sum(psd_scalars(g["dense_dim"]) for g in groups)
        y_parity_meta = dict(y_parity_meta)
        y_parity_meta.update({
            "n_psd_blocks": len(blocks),
            "n_gram_blocks": len(groups),
            "psd_scalars_dense": int(n_dense_scalars),
            "psd_scalars_emitted": int(sum(psd_scalars(b.dim) for b in blocks)),
        })

        l1 = None
        if l1_weights is not None:
            if len(l1_weights) != n_terms:
                raise ValueError(
                    f"l1_weights has length {len(l1_weights)}, expected {n_terms}")
            cross_indices = [i for i in range(n_terms) if i not in diag_indices]
            if cross_indices:
                l1 = (cross_indices,
                      list(np.asarray(l1_weights)[cross_indices]
                           / self.coefficient_scale))

        min_gram_trace = (self.objective == "min_gram_trace")

        if self.verbose:
            n_rows = sum(len(b.rows) for b in blocks)
            print(f"  MOSEK_DIRECT: {len(blocks)} PSD blocks "
                  f"({len(groups)} Gram blocks, "
                  f"{y_parity_meta['psd_scalars_emitted']}/"
                  f"{y_parity_meta['psd_scalars_dense']} PSD scalars), "
                  f"{n_rows} matching rows"
                  + (f", weighted-ℓ₁ over {len(l1[0])} cross terms" if l1 else "")
                  + (", min Σ tr Q" if min_gram_trace else ""))

        outcome = solve_joint_sos_direct(
            n_terms=n_terms,
            blocks=blocks,
            diag_indices=diag_indices,
            scaled_epsilon=scaled_epsilon,
            coefficient_scale=self.coefficient_scale,
            l1_weights_cross=l1,
            minimize_gram_trace=min_gram_trace,
            timeout=self.solver_timeout,
            mosek_params=getattr(self, '_mosek_params', None),
            verbose=self.verbose,
        )
        attempts.append(f"MOSEK_DIRECT:{outcome.status}")

        if outcome.status in ("optimal", "unknown_accepted"):
            bundle = self._build_gram_bundle(
                groups=groups, grams=outcome.grams,
                alpha_scaled=outcome.alpha, mu=mu,
            ) if capture else None
            # Drop the raw list: the bundle now owns those arrays (they are
            # rescaled in place), and nothing else reads outcome.grams.
            outcome.grams = None
            if self.verbose and bundle is not None:
                print(f"  MOSEK_DIRECT: captured {len(bundle)} design Grams "
                      f"(fingerprint {gram_bundle_fingerprint(bundle)[:16]})")
            packaged = self._package_design_result(
                alpha_scaled=outcome.alpha,
                term_specs=term_specs,
                degree_m=degree_m,
                clearing_power=clearing_power,
                status_string=outcome.status,
                solver_used="MOSEK_DIRECT",
                solver_attempts=attempts,
                fallback_used=False,
                start_time=start_time,
                gram_min_eig=outcome.gram_min_eig,
                gram_residual_l1=outcome.gram_residual_l1,
                gram_certificates=bundle,
                parity_meta=parity_meta,
                y_parity_meta=y_parity_meta,
            )
            return self._apply_design_screen(packaged, start_time=start_time)
        if outcome.status == "infeasible":
            return JointSOSResult(
                feasible=False,
                solver_status="infeasible:MOSEK_DIRECT",
                solver_requested=self.solver,
                solver_used="MOSEK_DIRECT",
                solver_fallback_used=False,
                solver_attempts=attempts,
                message=f"Infeasible: {outcome.message}",
                solve_time=time.time() - start_time,
                dual_info=SOSDualInfo(solver_info={
                    "status": outcome.status, "message": outcome.message}),
                term_specs=term_specs,
                parity_reduction=parity_meta,
                y_parity=y_parity_meta,
                design_objective=self.objective,
            )
        # error / unknown-and-screening-failed: fall back to CVXPY if allowed.
        # NOTE: the CVXPY path then re-poses the SAME (possibly reduced)
        # formulation — ``parity_reduction``/``term_specs`` were decided before
        # the backend split, so a fallback never silently changes the
        # formulation, only the encoding.
        if self.allow_solver_fallback:
            if self.verbose:
                print(f"  MOSEK_DIRECT {outcome.status} ({outcome.message}); "
                      f"falling back to the CVXPY path")
            return None
        return JointSOSResult(
            feasible=False,
            solver_status=f"{outcome.status}:MOSEK_DIRECT",
            solver_requested=self.solver,
            solver_used="MOSEK_DIRECT",
            solver_fallback_used=False,
            solver_attempts=attempts,
            message=outcome.message,
            solve_time=time.time() - start_time,
            term_specs=term_specs,
            parity_reduction=parity_meta,
            y_parity=y_parity_meta,
            design_objective=self.objective,
        )

    def _apply_design_screen(
        self,
        result: JointSOSResult,
        *,
        start_time: float,
    ) -> JointSOSResult:
        """Run the fail-fast design screen on a packaged result (WP P2-04).

        The two cheap filters decision D1 names, on the design output, before
        anything of it can enter the exact phase: ``min_V_on_sphere`` on V and
        ``_validate_gram_certificate`` on every bundle entry (V and W), the
        residual charged against that entry's own ε budget. Neither is the
        proof — both can only reject.

        Acceptance here is independent of the solver: the design backend's own
        policy (``mosek_direct.py``, which tolerates ``solsta.unknown`` at
        ``eig_tol = 1e-7`` / ``res_tol = 1e-5``) is deliberately NOT consulted,
        so an ``unknown_accepted`` solve cannot carry Grams into a certificate
        record without passing this screen.

        Mode ``"reject"`` (default) converts a refusal into a typed
        non-feasible result; ``"warn"`` records the verdict and keeps the
        result; ``"off"`` does nothing. A refusal never returns ``None``, so it
        cannot be mistaken for a backend stall and silently re-solved through
        CVXPY.
        """
        if self.design_screen == "off" or not result.feasible:
            return result
        from .design_screen import screen_design_result

        screen = screen_design_result(
            result, budget_policy=self.design_screen_budget,
            residual_model=self.design_screen_residual)
        record = screen.as_record()
        result.design_screen = record
        if screen.ok:
            if self.verbose:
                print(f"  design screen: ok "
                      f"(min V on sphere {screen.sphere_min_V:.3e}, "
                      f"{screen.n_blocks} Gram blocks, "
                      f"{screen.seconds:.3f} s)")
            return result
        if self.verbose:
            print(f"  design screen: REFUSED — {screen.detail}")
        if self.design_screen == "warn":
            return result
        # The refused result deliberately carries NO ``gram_certificates``: a
        # bundle that failed the screen must not be reachable by the exact
        # phase. The numbers survive in ``design_screen`` (per-block min_eig
        # and residual), which is diagnostics, not a proof object.
        return JointSOSResult(
            feasible=False,
            V_gf=result.V_gf,
            coefficients=result.coefficients,
            solver_status=f"design_screen_refused:{result.solver_used}"
                          if result.solver_used else "design_screen_refused",
            solver_requested=result.solver_requested,
            solver_used=result.solver_used,
            solver_fallback_used=result.solver_fallback_used,
            solver_attempts=list(result.solver_attempts or []),
            message=screen.detail,
            solve_time=time.time() - start_time,
            term_specs=result.term_specs,
            gram_min_eig=result.gram_min_eig,
            gram_residual_l1=result.gram_residual_l1,
            parity_reduction=result.parity_reduction,
            y_parity=result.y_parity,
            design_screen=record,
        )

    def _build_gram_bundle(
        self,
        *,
        groups: List[dict],
        grams: Optional[List[np.ndarray]],
        alpha_scaled: Optional[np.ndarray],
        mu: Tuple[int, ...],
    ) -> Optional[List[dict]]:
        """Package the recovered design Grams as a bundle (GRAM_BUNDLE_SCHEMA).

        ``groups`` and ``grams`` are index aligned through
        ``group["start"]``/``["count"]``: ``solve_joint_sos_direct`` unpacks one
        ``barx`` per emitted PSD block, in emission order, and one GROUP is one
        logical Gram block — a single dense PSD block, or the parity classes of
        one y-parity-decomposed block (WP P5-04).

        A blocked group is REASSEMBLED here: the sub-Grams are placed back on
        their dense basis indices, giving a block-diagonal Gram for the original
        basis ordering. The bundle entry is therefore the same object it was
        before P5-04 — same basis, same dimension, same hash recipe — and the
        exact phase, the design screen and the portable verifier see no
        blocking at all. PSD-ness survives because a block-diagonal matrix is
        permutation-similar to the direct sum of its blocks.

        Everything is converted from the SDP's SCALED space to the TRUE scale
        of ``V_gf``/``epsilon`` by dividing by ``coefficient_scale``: the block
        row is ``M·alpha_scaled − rhs = zᵀ Q_scaled z``, and dividing through
        by the (positive) scale gives ``M·alpha_true − eps = zᵀ Q_true z``
        without touching PSD-ness or the charge/budget ratio.

        Returns ``None`` when there is nothing to package (diagnostics were
        off, so no Gram was recovered, or the solve returned no coefficients).
        """
        if grams is None or alpha_scaled is None:
            return None
        alpha_true = (np.asarray(alpha_scaled, dtype=np.float64).reshape(-1)
                      / self.coefficient_scale)
        scale = float(self.coefficient_scale)
        mu_t = tuple(int(v) for v in mu)
        bundle: List[dict] = []
        for group in groups:
            meta = group["meta"]
            if meta is None:
                continue
            start, count = int(group["start"]), int(group["count"])
            split = group["split"]
            if start + count > len(grams) or any(
                    grams[i] is None for i in range(start, start + count)):
                continue
            if split is None:
                gram = grams[start]
            else:
                from .yparity import reassemble_from_order
                gram = reassemble_from_order(
                    split, [grams[i] for i in range(start, start + count)])
            rows = meta.pop("_rows")
            eps_monomials = meta.pop("_eps_monomials")
            # P2-09: the block's OWN μ. Identical to the design μ everywhere
            # except a W block whose y-degree came out odd, where it is 2μ.
            block_mu = tuple(int(v) for v in meta.pop("_mu", mu_t))
            eps = float(meta["eps"])
            target: Dict[Tuple[int, ...], float] = {}
            for mono, cols, vals in rows:
                target[mono] = target.get(mono, 0.0) + float(vals @ alpha_true[cols])
            for mono in eps_monomials:
                target[mono] = target.get(mono, 0.0) - eps
            g = np.asarray(gram, dtype=np.float64)
            if scale != 1.0:
                # In place: the caller drops ``outcome.grams`` right after, and
                # these arrays are freshly built by unpack_lower_triangle, so
                # nothing else observes them.
                np.divide(g, scale, out=g)
            entry = dict(meta)
            entry.update({
                "schema": GRAM_BUNDLE_SCHEMA,
                # P5-13: which canonical library ordering built this design.
                # Not hashed (see the schema comment): the entry addresses its
                # monomials absolutely, never by library index.
                "library_order": LIBRARY_ORDER,
                "mu": block_mu,
                "coefficient_scale": scale,
                "dim": int(group["dense_dim"]),
                "dtype": "float64",
                "target": target,
                "gram": g,
            })
            entry["hash"] = gram_bundle_entry_hash(entry)
            bundle.append(entry)
        return bundle or None

    def prune_and_reverify(
        self,
        degree_m: int,
        dynamics_gf: List[GeneralizedForm],
        V_gf: GeneralizedForm,
        threshold: float = 1e-4,
        threshold_mode: Literal["absolute", "relative"] = "absolute",
        relative_threshold: float = 0.05,
        min_cross_terms: int = 0,
        max_iterations: int = 5,
    ) -> JointSOSResult:
        """
        Prune small coefficients and re-verify SOS feasibility.

        DEPRECATED as the primary sparsifier — retained as the greedy baseline
        for ``scripts/sparsity_comparison.py``; superseded by the convex
        reweighted-ℓ₁ design in ``verification.sparse_sos.design_sparse_lyapunov``.

        Iteratively:
        1. Remove terms with |coefficient| < threshold (except diagonals)
        2. Re-run SOS verification with remaining terms
        3. Repeat until stable or re-verification fails

        This post-optimization step reduces solution complexity while maintaining
        mathematical correctness through re-verification.

        Args:
            degree_m: GF degree
            dynamics_gf: System dynamics as list of GeneralizedForms
            V_gf: Initial verified V with potentially redundant terms
            threshold: Pruning threshold for absolute mode
            threshold_mode: "absolute" or "relative". In relative mode, a cross
                term is kept when |coef| >= relative_threshold * max(|cross coef|).
            relative_threshold: Relative threshold multiplier for cross terms.
            min_cross_terms: Minimum number of cross terms to retain. If pruning
                would keep fewer, strongest cross terms are restored to this floor.
            max_iterations: Maximum pruning/re-verification rounds

        Returns:
            JointSOSResult with pruned V (or original if pruning fails)
        """
        start_time = time.time()

        # Convert V_gf terms to TermSpecs and extract coefficients
        current_terms = [
            TermSpec(exponents=t.exponents, term_types=t.term_types)
            for t in V_gf.terms
        ]
        current_coeffs = [t.coefficient for t in V_gf.terms]

        # Store best known result
        best_result = JointSOSResult(
            feasible=True,
            V_gf=V_gf,
            coefficients={f"alpha{i}": c for i, c in enumerate(current_coeffs)},
            solver_status="original",
            message="Original solution",
            solve_time=0.0,
            term_specs=current_terms,
        )

        for iteration in range(max_iterations):
            terms_to_keep, coeffs_to_keep, prune_stats = self._select_terms_for_pruning(
                current_terms=current_terms,
                current_coeffs=current_coeffs,
                threshold=threshold,
                threshold_mode=threshold_mode,
                relative_threshold=relative_threshold,
                min_cross_terms=min_cross_terms,
            )

            n_pruned = len(current_terms) - len(terms_to_keep)

            if n_pruned == 0:
                # No terms pruned, we're done
                if self.verbose:
                    print(f"  Pruning converged at {len(current_terms)} terms")
                break

            if self.verbose:
                print(f"  Prune iter {iteration+1}: {len(current_terms)} -> {len(terms_to_keep)} terms (-{n_pruned})")
                if threshold_mode == "relative":
                    eff = float(prune_stats.get("effective_threshold", 0.0))
                    mx = float(prune_stats.get("max_cross_abs", 0.0))
                    print(f"    Relative threshold: {relative_threshold:.3g} * max_cross={mx:.3e} => {eff:.3e}")
                if bool(prune_stats.get("floor_applied", False)):
                    kept = int(prune_stats.get("kept_cross_terms", 0))
                    print(f"    Cross-term floor applied: kept {kept} cross terms")

            # Re-verify with reduced term set
            result = self.find_lyapunov_with_dynamics(
                degree_m=degree_m,
                dynamics_gf=dynamics_gf,
                active_terms=terms_to_keep,
            )

            if not result.feasible:
                if self.verbose:
                    print(f"  Re-verification failed with {len(terms_to_keep)} terms, keeping {len(current_terms)}")
                # Return best known result
                break

            # Update for next iteration
            best_result = result
            current_terms = terms_to_keep

            # Get new coefficients from result
            if result.V_gf:
                current_coeffs = [t.coefficient for t in result.V_gf.terms]
            else:
                current_coeffs = coeffs_to_keep

        best_result.solve_time = time.time() - start_time
        return best_result

    def _select_terms_for_pruning(
        self,
        current_terms: List[TermSpec],
        current_coeffs: List[float],
        threshold: float,
        threshold_mode: Literal["absolute", "relative"],
        relative_threshold: float,
        min_cross_terms: int,
    ) -> Tuple[List[TermSpec], List[float], Dict[str, Any]]:
        """
        Select retained terms under configurable pruning policy.

        Diagonal terms are always kept. Cross terms are filtered by either:
        - absolute threshold: |coef| >= threshold
        - relative threshold: |coef| >= relative_threshold * max(|cross coef|)

        A cross-term floor can be enforced to avoid collapse to near-diagonal forms.
        """
        if threshold_mode not in {"absolute", "relative"}:
            raise ValueError(
                f"Unknown threshold_mode '{threshold_mode}'. "
                "Expected 'absolute' or 'relative'."
            )
        if relative_threshold < 0:
            raise ValueError(
                f"relative_threshold must be >= 0, got {relative_threshold}"
            )
        if min_cross_terms < 0:
            raise ValueError(f"min_cross_terms must be >= 0, got {min_cross_terms}")

        n_terms = len(current_terms)
        keep_mask = [False] * n_terms
        cross_indices: List[int] = []
        cross_abs: Dict[int, float] = {}

        for i, (spec, coef) in enumerate(zip(current_terms, current_coeffs)):
            if spec.is_diagonal():
                keep_mask[i] = True
            else:
                cross_indices.append(i)
                cross_abs[i] = abs(float(coef))

        max_cross_abs = max(cross_abs.values()) if cross_abs else 0.0
        if threshold_mode == "relative":
            effective_threshold = relative_threshold * max_cross_abs
        else:
            effective_threshold = threshold

        # Keep cross terms that pass threshold.
        for idx in cross_indices:
            if cross_abs[idx] >= effective_threshold:
                keep_mask[idx] = True

        # Enforce minimum number of cross terms (by strongest coefficient).
        kept_cross = [i for i in cross_indices if keep_mask[i]]
        floor_applied = False
        if cross_indices and len(kept_cross) < min_cross_terms:
            floor_applied = True
            n_target = min(min_cross_terms, len(cross_indices))
            strongest_cross = sorted(
                cross_indices,
                key=lambda i: cross_abs[i],
                reverse=True,
            )[:n_target]
            for idx in strongest_cross:
                keep_mask[idx] = True
            kept_cross = [i for i in cross_indices if keep_mask[i]]

        terms_to_keep = [spec for i, spec in enumerate(current_terms) if keep_mask[i]]
        coeffs_to_keep = [coef for i, coef in enumerate(current_coeffs) if keep_mask[i]]
        stats: Dict[str, Any] = {
            "threshold_mode": threshold_mode,
            "effective_threshold": float(effective_threshold),
            "max_cross_abs": float(max_cross_abs),
            "total_cross_terms": int(len(cross_indices)),
            "kept_cross_terms": int(len(kept_cross)),
            "floor_applied": bool(floor_applied),
            "min_cross_terms": int(min_cross_terms),
        }
        return terms_to_keep, coeffs_to_keep, stats

    def _compute_w_coefficient_matrix(
        self,
        term_specs: List[TermSpec],
        mu: Tuple[int, ...],
        signs: Tuple[int, ...],
        all_w_monomials: List[Tuple[int, ...]],
        dynamics_gf: List[GeneralizedForm]
    ) -> Optional[np.ndarray]:
        """
        DEPRECATED: Use _compute_w_coefficient_matrix_weighted_sphere() instead.

        This method is kept for backward compatibility but redirects to the
        weighted-sphere method which properly handles negative exponents.

        The old implementation had a critical bug (line 777) where W terms with
        negative x-exponents were silently DROPPED, leading to incomplete
        verification for STA, PID-SMC, and other systems with non-polynomial
        Lie derivatives.

        Returns:
            Matrix M_W of shape (n_w_monomials, n_terms), or None if computation fails
        """
        import warnings
        warnings.warn(
            "_compute_w_coefficient_matrix() is deprecated and redirects to "
            "_compute_w_coefficient_matrix_weighted_sphere() for correct handling "
            "of negative exponents. Use the weighted_sphere method directly.",
            DeprecationWarning,
            stacklevel=2
        )

        # Redirect to the proper implementation that handles negative exponents
        M_W, clearing_power, returned_monomials = self._compute_w_coefficient_matrix_weighted_sphere(
            term_specs, mu, signs, dynamics_gf
        )

        if self.verbose and clearing_power > 0:
            print(f"  Note: W had negative exponents, using clearing power k={clearing_power}")

        return M_W

    def _compute_w_coefficient_matrix_weighted_sphere(
        self,
        term_specs: List[TermSpec],
        mu: Tuple[int, ...],
        signs: Tuple[int, ...],
        dynamics_gf: List[GeneralizedForm]
    ) -> Tuple[Optional[np.ndarray], int, List[Tuple[int, ...]]]:
        """
        Compute W coefficient matrix with weighted-sphere clearing.

        When W has negative y-exponents (e.g., y₁^{-2} from |x₁|^{-1/2}),
        standard SOS cannot verify positivity. The solution is to multiply
        W by ||x||_r^k to clear all negative exponents.

        Verification: ||x||_r^k · W > 0 ⟺ W > 0  (since ||x||_r^k > 0 for x≠0)

        Args:
            term_specs: V term specifications
            mu: μ exponents for y-transform
            signs: Octant signs
            dynamics_gf: System dynamics as GF list

        Returns:
            Tuple of:
            - M_W_cleared: Coefficient matrix for ||x||_r^k · W
            - clearing_power: k used (0 if no clearing needed)
            - all_w_monomials: List of y-monomials in the cleared W
        """
        if dynamics_gf is None:
            return None, 0, []

        from gf_framework.algebra.lie_derivative import gf_partial_derivative_term, multiply_gf_terms

        n_terms = len(term_specs)

        # Check for composite terms in dynamics (requires special handling)
        # Composite terms like ⌈σ(x)⌋^ρ where σ is a GF are non-polynomial in y-space
        # and cannot be directly handled by SOS. We issue a warning and use only
        # the standard term contributions to W.
        has_composite_dynamics = any(
            len(f.composite_terms) > 0 for f in dynamics_gf if f is not None
        )
        if has_composite_dynamics and self.verbose:
            print("  WARNING: Dynamics contains composite terms (nested GF expressions).")
            print("           These are non-polynomial and EXCLUDED from SOS W computation.")
            print("           Numerical verification will validate the full dynamics.")

        # Step 1: Compute all W terms (with unit coefficients) to analyze exponents
        # For each V term j, compute its contribution to W
        all_w_terms_by_v_term = []  # all_w_terms_by_v_term[j] = list of (coef, GFTerm) from V term j
        all_w_exponents = []  # Collect all exponents to find minimum

        for j, spec in enumerate(term_specs):
            unit_term = GFTerm(
                coefficient=1.0,
                exponents=spec.exponents,
                term_types=spec.term_types
            )

            lie_terms = []
            for i in range(self.n_vars):
                deriv_term = gf_partial_derivative_term(unit_term, i)
                if deriv_term is None:
                    continue

                if i < len(dynamics_gf) and dynamics_gf[i] is not None:
                    # Handle standard GF terms
                    for f_term in dynamics_gf[i].terms:
                        product = multiply_gf_terms(deriv_term, f_term)
                        lie_terms.append(product)
                    # Note: composite_terms are NOT included here because they are
                    # non-polynomial in y-space. Numerical verification handles them.

            collected = collect_terms(lie_terms)
            all_w_terms_by_v_term.append(collected)

            # Collect exponents for analysis
            for w_term in collected:
                for i, exp in enumerate(w_term.exponents):
                    y_exp = mu[i] * float(exp)
                    all_w_exponents.append(y_exp)

        # Step 2: Analyze for negative exponents
        min_y_exp = min(all_w_exponents) if all_w_exponents else 0

        if min_y_exp >= 0:
            # No negative exponents, use standard approach
            if self.verbose:
                print("  W has no negative y-exponents, using standard SOS.")

            # Build standard W coefficient matrix.
            # The SOS basis degree must track the TOTAL y-degree of each W term,
            # not just the largest single-variable y-exponent. Otherwise mixed
            # monomials like y1^2 y2^2 y3^8 (total degree 12, max entry 8) are
            # silently omitted from the basis.
            all_w_term_degrees = []
            for w_terms_for_j in all_w_terms_by_v_term:
                for w_term in w_terms_for_j:
                    term_total_degree = sum(mu[i] * float(exp) for i, exp in enumerate(w_term.exponents))
                    all_w_term_degrees.append(term_total_degree)
            max_w_y_degree = int(max(all_w_term_degrees)) if all_w_term_degrees else 0
            all_w_monomials = get_monomial_basis(self.n_vars, int(max_w_y_degree))

            M_W = self._build_w_matrix_from_terms(
                term_specs, all_w_terms_by_v_term, mu, signs, all_w_monomials
            )
            return M_W, 0, all_w_monomials

        # Step 3: Need to multiply by a clearing monomial C_k to clear negative exponents
        #
        # We use a PRODUCT-based monomial C_k = ∏ᵢ |xᵢ|^{k·ρ/rᵢ} instead of the
        # sum-based norm ||x||_r = Σᵢ |xᵢ|^{ρ/rᵢ} because:
        # - Sum norm terms like |x₂|^{2k} don't help clear negative |x₁| exponents
        # - Product monomial adds to ALL variable y-exponents uniformly
        #
        # In y-space, C_k contributes k · k' · ρ to each variable's y-exponent,
        # where k' = μᵢ / rᵢ (same for all i by construction).
        #
        # To clear min_y_exp, we need: k · k' · ρ >= |min_y_exp|
        # So: k = ceil(-min_y_exp / (k' · ρ))
        rho = max(self.weights_r)
        k_prime = mu[0] // self.weights_r[0]

        # Validate assumption: μᵢ/rᵢ must be constant for all i
        for i in range(1, len(mu)):
            k_prime_i = mu[i] // self.weights_r[i]
            if k_prime_i != k_prime:
                raise ValueError(
                    f"μᵢ/rᵢ not constant: μ[0]/r[0]={k_prime}, "
                    f"μ[{i}]/r[{i}]={k_prime_i}. "
                    f"μ={mu}, r={self.weights_r}"
                )

        y_contribution_per_k = k_prime * rho

        clearing_power = int(np.ceil(-min_y_exp / y_contribution_per_k))

        if self.verbose:
            print(f"  W has negative y-exponents (min={min_y_exp:.2f})")
            print(f"  Using clearing monomial C_{clearing_power} = ∏ᵢ |xᵢ|^{{{clearing_power}·ρ/rᵢ}}")
            print(f"  Each variable gets y-contribution: {clearing_power * y_contribution_per_k}")

        # Step 4: Build product-based clearing monomial C_k = ∏ᵢ |xᵢ|^{k·ρ/rᵢ}
        clearing_gf = build_clearing_monomial_gf(self.weights_r, clearing_power)

        # Step 5: Compute C_k · W coefficient matrix
        # For each V term j, we need to compute: clearing_gf * lie_deriv(V_j)
        # Since C_k is a single term, each W term becomes one cleared term
        all_cleared_w_terms_by_v_term = []
        all_cleared_y_exponents = []  # Individual y-exponents for min check
        all_cleared_term_degrees = []  # Total degrees per term for basis size

        clearing_term = clearing_gf.terms[0]  # C_k is a single monomial

        for j, w_terms_for_j in enumerate(all_w_terms_by_v_term):
            cleared_terms = []
            for w_term in w_terms_for_j:
                # Multiply w_term by the clearing monomial (single term)
                product = multiply_gf_terms(w_term, clearing_term)
                cleared_terms.append(product)

            # Collect/simplify (shouldn't combine since product is single)
            collected_cleared = collect_terms(cleared_terms)
            all_cleared_w_terms_by_v_term.append(collected_cleared)

            # Collect y-exponents for monomial basis
            # Track both individual exponents (for min check) and total degrees (for basis size)
            for ct in collected_cleared:
                term_y_exps = []
                for i, exp in enumerate(ct.exponents):
                    y_exp = mu[i] * float(exp)
                    all_cleared_y_exponents.append(y_exp)
                    term_y_exps.append(y_exp)
                # Total degree for this term = sum of y-exponents
                term_total_degree = sum(term_y_exps)
                all_cleared_term_degrees.append(term_total_degree)

        # Sanity net on the clearing construction (cannot fire under the class
        # invariants): cleared y-exp = original + k·k'·ρ ≥ 0 holds exactly by the
        # choice of k, and the float evaluation of mu[i]·exp is exact for the
        # admissible exponent class (y-exponents are integers ≤ 2^53; the
        # clearing ratio's distance from integers is ≥ 1/lcm(denominators) ≫
        # float error, so ceil() cannot under-clear — at worst it over-clears,
        # which only enlarges the basis). The -1e-10 threshold therefore guards
        # against arithmetic bugs, not numerics. Design-side only: published
        # certificates are additionally backstopped by the exact gate, whose
        # independently rebuilt octant targets REFUSE negative/non-integer
        # y-exponents (exact_gate.exact_octant_target → ExactGateError).
        if all_cleared_y_exponents:
            min_cleared = min(all_cleared_y_exponents)
            if min_cleared < -1e-10:
                import warnings
                warnings.warn(
                    f"Clearing power {clearing_power} insufficient: min y-exp = {min_cleared}"
                )

        # Compute max TOTAL degree for monomial basis
        # BUG FIX: Previously used max(individual exponents), but need max(sum of exponents per term)
        if all_cleared_term_degrees:
            max_cleared_degree = int(max(all_cleared_term_degrees))
        else:
            max_cleared_degree = 0

        # Step 6: Build monomial basis for cleared W
        all_w_monomials_cleared = get_monomial_basis(self.n_vars, max_cleared_degree)

        # Step 7: Build coefficient matrix
        M_W_cleared = self._build_w_matrix_from_terms(
            term_specs, all_cleared_w_terms_by_v_term, mu, signs, all_w_monomials_cleared
        )

        return M_W_cleared, clearing_power, all_w_monomials_cleared

    def _build_w_matrix_from_terms(
        self,
        term_specs: List[TermSpec],
        w_terms_by_v_term: List[List[GFTerm]],
        mu: Tuple[int, ...],
        signs: Tuple[int, ...],
        all_w_monomials: List[Tuple[int, ...]]
    ) -> np.ndarray:
        """
        Build W coefficient matrix from pre-computed W terms.

        The W matrix encodes how W polynomial coefficients depend on V term
        coefficients (alpha). For each octant, the sign factors from signed
        terms are incorporated into the matrix.

        Sign factor computation follows Algorithm 1 from Sanchez & Moreno (2018):
        - For each W term (from Lie derivative), track term_types through
          derivative and multiplication operations
        - The final term_types determine which variables contribute sign factors
        - Sign factors MULTIPLY: cross-term derivatives can produce sᵢ·sⱼ products

        Args:
            term_specs: V term specifications
            w_terms_by_v_term: w_terms_by_v_term[j] = list of GFTerms from V term j
            mu: μ exponents for y-transform
            signs: Octant signs
            all_w_monomials: List of y-space monomials

        Returns:
            Matrix M_W such that W(y) coefficients = M_W @ alpha
        """
        n_terms = len(term_specs)
        n_monomials = len(all_w_monomials)
        mono_to_idx = {m: i for i, m in enumerate(all_w_monomials)}

        M_W = np.zeros((n_monomials, n_terms))

        for j, w_terms in enumerate(w_terms_by_v_term):
            for w_term in w_terms:
                # Build TermSpec from w_term
                # CRITICAL: term_types encode which variables have sign dependence
                # These were correctly propagated through gf_partial_derivative_term()
                # and multiply_gf_terms() in the Lie derivative computation
                w_spec = TermSpec(
                    exponents=w_term.exponents,
                    term_types=w_term.term_types
                )

                # Convert to y-monomial with sign factor
                # sign_factor = ∏ᵢ sᵢ for all i where term has 'signed' type
                # with odd numerator exponent
                y_exp, sign_factor = term_to_y_monomial(w_spec, mu, signs)

                if y_exp in mono_to_idx:
                    mono_idx = mono_to_idx[y_exp]
                    # W = -dV/dt, so negate the contribution
                    M_W[mono_idx, j] -= sign_factor * w_term.coefficient

        return M_W

    def diagnose_w_polynomial(
        self,
        term_specs: List[TermSpec],
        coefficients: np.ndarray,
        dynamics_gf: List[GeneralizedForm],
        verbose: bool = True
    ) -> Dict[Tuple[int, ...], Dict]:
        """
        Diagnostic output showing W polynomial structure per octant.

        This is useful for debugging sign factor handling. For each octant,
        it shows the W polynomial with explicit sign factors, matching the
        format used in the MATLAB reference (pidlike_sos_reference.md).

        Args:
            term_specs: V term specifications
            coefficients: V coefficient values
            dynamics_gf: System dynamics as GF list
            verbose: Print detailed output

        Returns:
            Dict mapping octant signs -> dict with:
                'monomials': list of (y_exp, coef, sign_factor_str)
                'w_polynomial': symbolic expression string
        """
        from gf_framework.algebra.lie_derivative import gf_partial_derivative_term, multiply_gf_terms, collect_terms

        k, mu = compute_mu_exponents_with_dynamics(
            term_specs=term_specs,
            weights_r=self.weights_r,
            dynamics_gf=dynamics_gf,
            ensure_even_exponents=self.ensure_even_exponents,
        )
        octants = get_hyperoctant_signs(self.n_vars)

        results = {}

        for oct_idx, signs in enumerate(octants):
            sign_strs = ['s' + str(i+1) + ('=' if signs[i] > 0 else '=') +
                        ('+1' if signs[i] > 0 else '-1')
                        for i in range(self.n_vars)]

            if verbose:
                print(f"\n=== Octant {oct_idx + 1}: ({', '.join(sign_strs)}) ===")

            # Compute W terms for this octant
            w_poly_terms = []

            for j, spec in enumerate(term_specs):
                # Diagnostics only: this routine reports the W polynomial, it
                # never gates acceptance, so a display floor is harmless here
                # (checked under review finding 6, 2026-09-05).
                if abs(coefficients[j]) < 1e-12:
                    continue

                unit_term = GFTerm(
                    coefficient=1.0,
                    exponents=spec.exponents,
                    term_types=spec.term_types
                )

                # Compute Lie derivative contribution from this V term
                lie_terms = []
                for i in range(self.n_vars):
                    deriv_term = gf_partial_derivative_term(unit_term, i)
                    if deriv_term is None:
                        continue
                    if i < len(dynamics_gf) and dynamics_gf[i] is not None:
                        for f_term in dynamics_gf[i].terms:
                            product = multiply_gf_terms(deriv_term, f_term)
                            lie_terms.append(product)

                collected = collect_terms(lie_terms)

                for w_term in collected:
                    w_spec = TermSpec(
                        exponents=w_term.exponents,
                        term_types=w_term.term_types
                    )
                    y_exp, sign_factor = term_to_y_monomial(w_spec, mu, signs)

                    # Build sign factor string (e.g., "(s1*s2)" or "1")
                    sign_vars = []
                    for i in range(self.n_vars):
                        exp = w_spec.exponents[i]
                        typ = w_spec.term_types[i]
                        if typ == 'signed':
                            if exp == 0 or (isinstance(exp, Fraction) and exp.numerator % 2 == 1):
                                sign_vars.append(f's{i+1}')
                            elif isinstance(exp, (int, float)):
                                exp_int = int(exp)
                                if exp_int == exp and exp_int % 2 == 1:
                                    sign_vars.append(f's{i+1}')

                    if sign_vars:
                        sign_str = '(' + '*'.join(sign_vars) + ')'
                    else:
                        sign_str = '1'

                    # W = -dV/dt coefficient
                    w_coef = -coefficients[j] * w_term.coefficient * sign_factor

                    y_str = '*'.join([f'y{i+1}^{e}' for i, e in enumerate(y_exp) if e != 0])
                    if not y_str:
                        y_str = '1'

                    w_poly_terms.append({
                        'y_exp': y_exp,
                        'coef': w_coef,
                        'sign_factor': sign_factor,
                        'sign_str': sign_str,
                        'y_str': y_str,
                        'v_term_idx': j,
                        'term_types': w_spec.term_types
                    })

            if verbose:
                print(f"  W polynomial terms:")
                for t in w_poly_terms:
                    print(f"    {t['coef']:+.4f} * {t['sign_str']} * {t['y_str']}  "
                          f"[from α_{t['v_term_idx']+1}, types={t['term_types']}]")

            results[signs] = {
                'monomials': w_poly_terms,
                'octant_idx': oct_idx
            }

        return results


def _validate_gram_certificate(
    target: Dict[Tuple[int, ...], float],
    gram: np.ndarray,
    basis: List[Tuple[int, ...]],
    n_vars: int,
    eps_budget: float,
    psd_tol: float,
    budget_policy: str = "achieved",
    residual_model: str = "dense",
    diag: Optional[Dict[str, float]] = None,
    extra_residual_l1: float = 0.0,
) -> Tuple[bool, float, float, str]:
    """A-posteriori validation of a numeric SOS Gram certificate.

    The SDP solver — even on ``OPTIMAL``, and especially on ``OPTIMAL_INACCURATE``
    — returns a Gram Q that satisfies the equality constraints only to solver
    precision. Accepting it on status alone is exactly the false-positive class
    documented in review/REVIEW.md (non-PD V's certified). This helper makes
    acceptance independent of solver status. Write

        target = zᵀQ̄z + E,   Q̄ = sym(Q),   η = max(0, −λ_min(Q̄)),

    so ``Q̄ + ηI ⪰ 0`` gives a genuine SOS and ``E`` is the per-monomial
    coefficient residual. For a homogeneous even-degree-d target with the
    ``ε·Σᵢ y_iᵈ`` margin already subtracted by the caller (``eps_budget`` = ε),
    weighted AM-GM (``|y^a| ≤ Σᵢ (aᵢ/d)·y_iᵈ`` for |a| = d) bounds both error
    terms by multiples of ``Σᵢ y_iᵈ``:

        F := target + ε·Σᵢ y_iᵈ  ≥  SOS + (ε − η·C₁ − ‖E‖₁)·Σᵢ y_iᵈ,
        C₁ = maxᵢ Σ_b 2β_{b,i}/d   (β_b the half-degree basis exponents).

    Acceptance requires ``charge = η·C₁ + ‖E‖₁ < ε`` (WP P2-14, owner decision
    D-l), which certifies ``F ≥ (ε − charge)·Σᵢ y_iᵈ > 0`` off the origin — the
    solver's inaccuracy is charged against the strictness margin instead of
    being ignored. Without a margin budget (legacy callers) or for
    non-homogeneous targets, a conservative absolute policy applies:
    ``λ_min ≥ −psd_tol`` and ``‖E‖₁ ≤ 1e-7``.

    ``budget_policy`` (WP P2-13) chooses the acceptance THRESHOLD, and only
    that: ``"achieved"`` (the default since P2-14) demands ``charge < ε``
    strictly, which is the weakest demand under which the proof line above
    still yields a strictly positive certified margin ``ε − charge``;
    ``"half"`` (historical, and the inequality the frozen Automatica note
    prints) demands ``charge ≤ ε/2``. It mirrors
    ``exact_gate.MARGIN_POLICIES`` exactly, so the cheap float screen and the
    exact gate keep agreeing about which bundles are worth proving; running
    them under DIFFERENT policies is what turns the screen from a conservative
    filter into a source of false negatives (or, the other way round, of
    wasted exact work). This function is never the proof either way.

    ``residual_model`` (WP P2-13) chooses WHICH residual is charged.
    ``"dense"`` (default) expands ``zᵀQ̄z`` over every basis pair. That is the
    residual ``exact_gate.exact_validate_octant`` (X) charges since 2026-09-25
    and the one the portable verifier (P) always charged (fix (a) of
    reports/XVC_RESIDUAL_2026-09-25.md), so ``"dense"`` is the model that
    predicts both exact implementations. ``"deck_blocked"`` expands it over
    same-parity-class pairs only, i.e. the residual of the Reynolds
    projection ``Q̃``: ``P`` is invariant under the target's deck group ``H``,
    so the deleted pairs contribute only to non-invariant monomials, and
    ``‖Ẽ‖₁ ≤ ‖E‖₁`` always. X charged that residual until 2026-09-25; it no
    longer does. ``"deck_blocked"`` is therefore DEPRECATED and OPTIMISTIC
    against X and P (it can accept a block they refuse, by exactly the
    cross-class mass) and selecting it emits
    ``design_screen.OptimisticResidualModelWarning``. The λ_min proposal stays
    DENSE under both, exactly as the exact gate proposes it (pinching gives
    λ_min(Q̃) ≥ λ_min(Q̄), so a dense shift is valid and at worst
    conservative). The deck group is read off the FLOAT target support here,
    not the exact one; that is admissible only because this function is a
    filter and never a proof.

    ``extra_residual_l1`` (review finding 6, 2026-09-05) is coefficient mass of
    the TRUE target that the caller could not hand to the solver — e.g. a
    coefficient dropped below the solver-encoding floor. It is ‖·‖₁ mass of
    ``target_true − target_solved``, so adding it to ‖E‖₁ makes the charge an
    upper bound for the residual of the true target, exactly as the triangle
    inequality requires. Silently truncating instead (the pre-fix behaviour)
    left that mass uncharged and could accept a target the budget does not
    cover.

    ``diag``, when a dict is supplied, is filled with ``eta``, ``c1``,
    ``charge``, ``budget``, ``eps_certified`` (= ``ε − charge``),
    ``residual_model``, ``resid_l1`` (the one charged) and ``resid_l1_dense``
    for the margin policy branch, so a caller can record the numbers without
    recomputing them. Untouched on the legacy absolute branch.

    Returns ``(ok, min_eig, residual_l1, reject_reason)``. ``residual_l1`` is
    the residual the policy CHARGED, so it follows ``residual_model``.
    """
    if budget_policy not in ("half", "achieved"):
        raise ValueError(
            f"unknown budget_policy {budget_policy!r}; "
            "expected 'half' or 'achieved'")
    if residual_model not in ("dense", "deck_blocked"):
        raise ValueError(
            f"unknown residual_model {residual_model!r}; "
            "expected 'dense' or 'deck_blocked'")
    extra_residual_l1 = abs(float(extra_residual_l1))
    gram_sym = (gram + gram.T) / 2.0
    min_eig = float(np.linalg.eigvalsh(gram_sym).min())

    # Per-monomial residual E = target − zᵀQ̄z, over every basis pair.
    gram_poly: Dict[Tuple[int, ...], float] = {}
    n_basis = len(basis)
    for i in range(n_basis):
        for j in range(n_basis):
            mono = tuple(basis[i][k] + basis[j][k] for k in range(n_vars))
            gram_poly[mono] = gram_poly.get(mono, 0.0) + float(gram_sym[i, j])
    resid_l1_dense = extra_residual_l1
    for mono in set(gram_poly) | set(target):
        resid_l1_dense += abs(target.get(mono, 0.0) - gram_poly.get(mono, 0.0))
    resid_l1 = resid_l1_dense

    if residual_model == "deck_blocked" and n_basis:
        # The residual of the Reynolds projection: same-parity-class pairs
        # only. DEPRECATED and OPTIMISTIC since 2026-09-25: the exact gate and
        # the portable verifier both charge the dense residual above
        # (reports/XVC_RESIDUAL_2026-09-25.md). Imported lazily; exact_gate is
        # numpy-light and pulls in nothing from this module.
        from .design_screen import warn_optimistic_residual_model
        warn_optimistic_residual_model(residual_model)
        from .exact_gate import deck_group_basis, parity_blocks
        group_basis = deck_group_basis(list(target), n_vars)
        blocked_poly: Dict[Tuple[int, ...], float] = {}
        for idxs in parity_blocks(basis, group_basis):
            for i in idxs:
                for j in idxs:
                    mono = tuple(basis[i][k] + basis[j][k]
                                 for k in range(n_vars))
                    blocked_poly[mono] = (blocked_poly.get(mono, 0.0)
                                          + float(gram_sym[i, j]))
        resid_l1 = extra_residual_l1
        for mono in set(blocked_poly) | set(target):
            resid_l1 += abs(target.get(mono, 0.0) - blocked_poly.get(mono, 0.0))
        gram_poly = blocked_poly

    degrees = {sum(m) for m in target} | {sum(m) for m in gram_poly}
    d = next(iter(degrees)) if len(degrees) == 1 else 0
    if eps_budget > 0.0 and d > 0 and d % 2 == 0:
        eta = max(0.0, -min_eig)
        c1 = max(
            (sum(2.0 * b[i] for b in basis) / float(d) for i in range(n_vars)),
            default=0.0,
        )
        charge = eta * c1 + resid_l1
        budget = (0.5 * float(eps_budget) if budget_policy == "half"
                  else float(eps_budget))
        if diag is not None:
            diag.update({
                "eta": float(eta), "c1": float(c1), "charge": float(charge),
                "budget": float(budget),
                "eps_certified": float(eps_budget) - float(charge),
                "resid_l1": float(resid_l1),
                "resid_l1_dense": float(resid_l1_dense),
            })
        ok = (charge <= budget if budget_policy == "half" else charge < budget)
        if ok:
            return True, min_eig, resid_l1, ""
        return False, min_eig, resid_l1, (
            f"gram_margin_exceeded(eta={eta:.2e},C1={c1:.1f},"
            f"resid_l1={resid_l1:.2e},budget={budget:.2e})"
        )

    if min_eig < -psd_tol:
        return False, min_eig, resid_l1, f"gram_not_psd(min_eig={min_eig:.2e})"
    if resid_l1 > 1e-7:
        return False, min_eig, resid_l1, f"gram_residual(resid_l1={resid_l1:.2e})"
    return True, min_eig, resid_l1, ""


def _numeric_poly_is_sos(
    target: Dict[Tuple[int, ...], float],
    n_vars: int,
    y_degree: int,
    solver: str = 'MOSEK',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
    psd_tol: float = 1e-7,
    eps_budget: float = 0.0,
    use_mosek_direct: bool = True,
    extra_residual_l1: float = 0.0,
) -> Tuple[bool, Optional[np.ndarray], Optional[str], List[str], Optional[Dict[str, float]]]:
    """Standard (numeric) SOS feasibility for a polynomial ``target``.

    Solves the LMI feasibility problem: does there exist a PSD Gram matrix Q with
    ``z(y)ᵀ Q z(y) == target(y)``, where z(y) is the monomial basis up to degree
    ⌈y_degree/2⌉?  By Choi-Lam-Reznick this is exactly "target is SOS", the
    certification path of Sanchez-Moreno 2018 §5.3 / Algorithm 7. The SDP solver
    proposes the Gram, but acceptance is gated by the a-posteriori
    ``_validate_gram_certificate`` check (PSD margin + equality residual), so an
    ``OPTIMAL_INACCURATE`` status can never smuggle in an invalid certificate.

    Strict positive-definiteness is obtained by the caller subtracting an
    ``ε·Σ_i y_i^{y_degree}`` margin from ``target`` before calling this; passing
    that ε as ``eps_budget`` lets the validator charge solver inaccuracy against
    the margin rigorously (see ``_validate_gram_certificate``).

    ``use_mosek_direct`` (WP P2-10, default ON): when the REQUESTED solver is
    MOSEK, pose this same feasibility problem through the MOSEK **Task API**
    (``mosek_direct.solve_poly_sos_direct``) instead of CVXPY's MOSEK interface.
    The MOSEK_DIRECT rung SUBSTITUTES the CVXPY MOSEK rung (it does not precede
    it), so no CVXPY solve is added anywhere; the remaining rungs of
    ``_solver_candidates`` (CLARABEL, SCS) are untouched and still CVXPY. Basis,
    Gram block structure, monomial-matching rows, ε/margin semantics and the
    a-posteriori ``_validate_gram_certificate`` acceptance are literally the same
    objects on both branches — only the encoding differs. ``used`` is then
    ``"MOSEK_DIRECT"``. Set ``False`` to force the historical CVXPY-only ladder.
    N7/N9/N10 are the recorded motivation (CVXPY-encoding artefacts, false
    NEGATIVES, that vanished under the direct path on the design side), but see
    **N19**: on the WITNESS ladder the measured win comes from
    ``mosek_direct.POLY_SOS_MOSEK_PARAMS`` (the tightened interior-point accuracy
    target), not from the encoding. At MOSEK's default tolerances this branch
    produces MORE infeasible witnesses than the CVXPY one, not fewer.

    ``extra_residual_l1`` is target coefficient mass the CALLER already dropped
    before handing the target over (see ``_validate_gram_certificate``); it is
    charged to the residual on top of whatever this function drops itself.

    **Coefficient truncation (review finding 6, 2026-09-05).** Coefficients at
    or below ``SOLVER_COEFF_FLOOR`` are still kept out of the solver's equality
    rows — an equality row at 1e-16 is noise to an interior-point method and
    the floor also keeps the "monomial outside the basis" structural test from
    firing on rounding dust. But the dropped ‖·‖₁ mass is now CHARGED to the
    a-posteriori residual instead of vanishing, so the accepted Gram is
    validated against the true target. Before this fix the mass was simply
    deleted, and a target that was empty only after truncation was reported SOS
    as ``"trivial_zero"`` with no Gram at all.

    **Zero SOS remainder (follow-up review 2026-09-05, A).** An EXACTLY zero
    target — no coefficients, or only exact zeros, before any truncation — is
    the zero polynomial, which is SOS with the zero Gram ``Q = 0`` of the
    ordinary half-basis: zero residual, zero PSD shift, charge 0. That is a
    legitimate outcome of the ε-subtraction and NOT a lost margin: for
    ``F(y) = ε·Σᵢ y_iᵈ`` the margin-subtracted target ``P = F − ε·Σᵢ y_iᵈ`` is
    identically zero while ``F`` carries exactly the requested strict margin.
    The repair returns the zero Gram (with its basis, so Gram-capturing callers
    stay well-formed) and runs the NORMAL budget calculation, so any separately
    dropped mass ``extra_residual_l1`` is still charged: with
    ``dropped_l1 ≥ ε`` the zero Gram is refused as
    ``target_below_solver_floor(...)``. ``used`` is ``"zero_remainder"`` on that
    branch, distinct from the Gram-free ``"trivial_zero"`` that the legacy
    marginless callers (``eps_budget == 0`` and nothing dropped) still get.
    Until this repair the branch returned ``empty_strict_target`` for every
    zero target with a margin, on the false premise that a strict target always
    carries the ``−ε·Σᵢ y_iᵈ`` pure powers as NONZERO coefficients.

    Returns ``(is_sos, gram_float, solver_used, attempts, gram_diagnostics)``
    where ``gram_diagnostics`` is ``{"min_eig": …, "residual_l1": …}`` for the
    last validated Gram (None if no Gram was recovered).
    """
    raw_target = {m: float(c) for m, c in target.items()}
    nonzero_target = {m: c for m, c in raw_target.items() if c != 0.0}
    dropped_l1 = abs(float(extra_residual_l1))

    if not nonzero_target:
        # The GENUINELY zero polynomial: no coefficients at all, or only exact
        # zeros, BEFORE any truncation. Nothing was truncated away here, so the
        # only mass to account for is what the caller dropped (dropped_l1).
        if eps_budget <= 0.0 and dropped_l1 == 0.0:
            # Legacy marginless callers: no Gram, no budget, nothing to charge.
            return True, None, "trivial_zero", [], None
        # Zero SOS REMAINDER (follow-up review 2026-09-05, A). P ≡ 0 is SOS
        # with Q = 0 over the ordinary half-basis; the strict margin lives in
        # F = P + ε·Σᵢ y_iᵈ, which is exactly ε-positive. Exhibit that Gram and
        # let the normal a-posteriori validator charge the dropped mass.
        d_zero = int(y_degree)
        if d_zero > 0 and d_zero % 2 == 0:
            zero_basis = get_homogeneous_monomial_basis(n_vars, d_zero // 2)
        else:
            # No even homogeneity degree to state a margin against. The zero
            # polynomial is still SOS, but the ε·Σᵢ y_iᵈ bookkeeping below is
            # only defined for a positive even d, so refuse instead of
            # pretending a margin was certified.
            if eps_budget > 0.0:
                return False, None, None, [
                    f"zero_target_degree_not_positive_even(y_degree={d_zero})"
                ], None
            zero_basis = get_monomial_basis(n_vars, (d_zero + 1) // 2)
        if not zero_basis:
            return False, None, None, ["empty_basis"], None
        zero_gram = np.zeros((len(zero_basis), len(zero_basis)), dtype=float)
        ok_z, min_eig_z, resid_z, reason_z = _validate_gram_certificate(
            {}, zero_gram, zero_basis, n_vars, eps_budget, psd_tol,
            extra_residual_l1=dropped_l1,
        )
        diag_z = {"min_eig": min_eig_z, "residual_l1": resid_z, "cond": None,
                  "basis": zero_basis}
        if ok_z:
            return True, zero_gram, "zero_remainder", [], diag_z
        # The zero remainder itself is unimpeachable; what does not fit is the
        # mass the caller dropped. Same typed refusal as the truncation branch.
        return False, None, None, [
            f"target_below_solver_floor(dropped_l1={dropped_l1:.2e},"
            f"{reason_z})"], diag_z

    target = {m: c for m, c in nonzero_target.items()
              if abs(c) > SOLVER_COEFF_FLOOR}
    dropped_l1 += sum(abs(c) for c in nonzero_target.values()
                      if abs(c) <= SOLVER_COEFF_FLOOR)
    if not target:
        # Non-empty before truncation, empty after. This is NOT the zero
        # polynomial and must never be reported SOS on the trivial_zero branch.
        # Refusing is the conservative side: the caller can re-scale the target
        # if it genuinely wants such a bundle certified.
        return False, None, None, [
            f"target_below_solver_floor(dropped_l1={dropped_l1:.2e})"], None

    # If the target is homogeneous of even total degree (the associated forms of a
    # generalized-homogeneous GF always are), only the degree-exact half-basis can
    # produce it — using the full 0..h basis just inflates the Gram from O(h) to
    # O(h²) rows and makes CVXPY's problem compilation blow up (a 325×325 vs 25×25
    # Gram for μ-doubled fractional GFs). Restrict the basis accordingly.
    degrees = {sum(m) for m in target}
    if len(degrees) == 1:
        total_deg = degrees.pop()
        if total_deg % 2 == 0:
            basis = get_homogeneous_monomial_basis(n_vars, total_deg // 2)
        else:
            # Odd-degree homogeneous form can never be SOS (would be sign-indefinite).
            return False, None, None, ["odd_homogeneous_degree"], None
    else:
        half_degree = (int(y_degree) + 1) // 2
        basis = get_monomial_basis(n_vars, half_degree)
    n_basis = len(basis)
    if n_basis == 0:
        return False, None, None, ["empty_basis"], None

    # The Gram pair map is the SHARED object: both encodings index the same
    # basis and the same monomial-matching rows off it, so "same PSD block
    # structure" is by construction, not by convention.
    gram_map: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    for i in range(n_basis):
        for j in range(n_basis):
            prod = tuple(basis[i][k] + basis[j][k] for k in range(n_vars))
            gram_map.setdefault(prod, []).append((i, j))

    for mono in set(gram_map) | set(target):
        if not gram_map.get(mono) and target.get(mono, 0.0) != 0.0:
            # target needs a monomial the basis cannot produce -> not SOS
            return False, None, None, ["monomial_outside_basis"], None

    attempts: List[str] = []
    last_diag: Optional[Dict[str, float]] = None

    def _screen(gram: np.ndarray) -> Tuple[bool, Dict[str, Any], str]:
        """Shared a-posteriori acceptance. Identical on both encodings."""
        ok_, min_eig_, resid_, reason_ = _validate_gram_certificate(
            target, gram, basis, n_vars, eps_budget, psd_tol,
            extra_residual_l1=dropped_l1,
        )
        eigs_ = np.linalg.eigvalsh((gram + gram.T) / 2.0)
        cond_ = float(eigs_[-1] / eigs_[0]) if eigs_[0] > 0 else None
        diag_ = {"min_eig": min_eig_, "residual_l1": resid_, "cond": cond_,
                 "basis": basis}
        if ok_:
            return True, diag_, (
                f"postcheck_ok(min_eig={min_eig_:.2e},resid_l1={resid_:.2e})"
            )
        return False, diag_, reason_

    # --- rung 0: MOSEK through the Task API (WP P2-10) ----------------------
    # Substitutes the CVXPY MOSEK rung when the REQUESTED solver is MOSEK. The
    # solver status is not the certificate; ``_screen`` is.
    use_direct = False
    if use_mosek_direct and _normalize_solver_name(solver) in ("MOSEK", "MOSEK_DIRECT"):
        from .mosek_direct import MOSEK_AVAILABLE as _MOSEK_DIRECT_AVAILABLE
        if _MOSEK_DIRECT_AVAILABLE:
            use_direct = True
            from .mosek_direct import solve_poly_sos_direct
            outcome = solve_poly_sos_direct(
                target=target, gram_map=gram_map, n_basis=n_basis)
            attempts.append(f"MOSEK_DIRECT:{outcome.status}")
            if outcome.gram is not None:
                gram = np.asarray(outcome.gram, dtype=float)
                ok, diag, detail = _screen(gram)
                attempts[-1] += f":{detail}"
                if ok:
                    return True, gram, "MOSEK_DIRECT", attempts, diag
                last_diag = diag
        else:
            attempts.append("MOSEK_DIRECT:unavailable")
    if use_direct and strict_solver:
        return False, None, None, attempts, last_diag

    if not CVXPY_AVAILABLE:
        attempts.append("cvxpy_unavailable")
        return False, None, None, attempts, last_diag

    Q = cp.Variable((n_basis, n_basis), symmetric=True, name="Q")
    constraints = [Q >> 0]
    for mono in set(gram_map) | set(target):
        pairs = gram_map.get(mono, [])
        if pairs:
            constraints.append(sum(Q[i, j] for i, j in pairs)
                               == target.get(mono, 0.0))

    prob = cp.Problem(cp.Minimize(0), constraints)
    candidates = _solver_candidates(solver, strict_solver, allow_solver_fallback)
    if use_direct:
        # The MOSEK rung already ran, natively. Never re-run it through CVXPY:
        # that would ADD a CVXPY solve on the certifying path.
        candidates = [c for c in candidates if c != "MOSEK"]
    for solver_name in candidates:
        solver_obj = getattr(cp, solver_name, None)
        if solver_obj is None:
            attempts.append(f"{solver_name}:unavailable")
            if strict_solver:
                break
            continue
        try:
            prob.solve(solver=solver_obj)
            attempts.append(f"{solver_name}:{prob.status}")
        except Exception as exc:  # SolverError or numeric failure
            attempts.append(f"{solver_name}:error:{exc}")
            if strict_solver:
                break
            continue
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and Q.value is not None:
            gram = np.asarray(Q.value, dtype=float)
            # The solver status is NOT the certificate; the recovered Gram is.
            # Validate it a-posteriori (PSD shift + equality residual charged
            # against the ε-margin) regardless of OPTIMAL vs OPTIMAL_INACCURATE.
            # ``_screen`` also carries the Gram conditioning (λ_max/λ_min) and
            # the monomial "basis", which Gram-capturing callers (exact gate)
            # need to index the Gram.
            ok, diag, detail = _screen(gram)
            attempts[-1] += f":{detail}"
            if ok:
                return True, gram, solver_name, attempts, diag
            last_diag = diag
        if strict_solver:
            break
    return False, None, None, attempts, last_diag


def verify_sos_polynomial(
    poly_coeffs: Dict[Tuple[int, ...], float],
    n_vars: int,
    epsilon: float = 1e-6,
    solver: str = 'MOSEK',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
    use_mosek_direct: bool = True,
) -> Tuple[bool, Optional[np.ndarray], str, List[str], bool, str]:
    """
    Verify that a polynomial is SOS via standard (numeric) SOS.

    Args:
        poly_coeffs: Dict mapping exponent tuple -> coefficient
        n_vars: Number of variables
        epsilon: kept for signature stability (callers bake the margin into the poly)
        solver: Preferred CVXPY solver.
        strict_solver: If True, fail instead of trying fallback solvers.
        allow_solver_fallback: If False, use the preferred solver only.

    Returns:
        (is_sos, gram_matrix, solver_status, solver_attempts, solver_fallback_used, error_message)

    The SDP solver is the accepted oracle (standard SOS, Sanchez-Moreno 2018 §5.3).
    Callers that need strict positivity must bake the ε-margin into ``poly_coeffs``.
    """
    if not poly_coeffs:
        return True, None, "trivial_zero", [], False, ""
    y_degree = max((sum(m) for m in poly_coeffs), default=0)
    # eps_budget=0: callers of this legacy entry point bake the margin into the
    # poly without telling us its size, so the validator applies the conservative
    # absolute-tolerance policy instead of margin charging.
    ok, gram, used, attempts, _diag = _numeric_poly_is_sos(
        poly_coeffs, n_vars, y_degree, solver=solver,
        strict_solver=strict_solver, allow_solver_fallback=allow_solver_fallback,
        use_mosek_direct=use_mosek_direct,
    )
    if ok:
        return True, gram, f"standard_sos:{used or solver}", attempts, False, ""
    return False, None, "standard_sos_infeasible", attempts, False, "not SOS (numeric)"


def verify_gf_sos_full(
    V_gf: GeneralizedForm,
    epsilon: float = 1e-6,
    solver: str = 'MOSEK',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
    verbose: bool = False,
    ensure_even_exponents: bool = True,
    capture_grams: bool = False,
    use_mosek_direct: bool = True,
) -> JointSOSResult:
    """
    Verify that a GF is positive definite via STANDARD (numeric) SOS in ALL octants.

    For each sign combination ``s ∈ {−1,+1}ⁿ`` the associated form ``V(y; s)`` is
    built in y-space (even exponents via μ), and we certify that
    ``V(y; s) − ε·Σ_i y_i^{d}`` is SOS (a PSD Gram, found numerically). This gives
    the strict bound ``V ≥ ε·Σ y_i^{d} > 0`` off the origin — a standard,
    widely-accepted SOS certificate (Sanchez-Moreno 2018 §5.3 / Algorithm 7) with
    the SDP solver as the oracle.

    A fast NECESSARY pre-check (``min_V_on_sphere``) refutes candidates whose V is
    not even positive-definite (the CTA / Levant3D failure class) before any SDP.

    Args:
        V_gf: Generalized Form to verify
        epsilon: Strictness margin (top of a descending ε-ladder)
        solver: CVXPY solver
        strict_solver: If True, fail instead of trying fallback solvers.
        allow_solver_fallback: If False, use requested solver only.
        verbose: Print progress
        ensure_even_exponents: If True (default), double μ for SOS-compatible
                              even y-exponents per Algorithm 7 of Sanchez-Moreno 2018.

    Returns:
        JointSOSResult with verification status
    """
    if not CVXPY_AVAILABLE:
        return JointSOSResult(feasible=False, message="CVXPY not available")

    from .sphere_check import min_V_on_sphere

    start_time = time.time()
    n_vars = len(V_gf.weights_r)

    # Convert terms to TermSpecs
    term_specs = [
        TermSpec(exponents=t.exponents, term_types=t.term_types)
        for t in V_gf.terms
    ]

    # (A) Dynamics-independent NECESSARY pre-check: a (generalized-)homogeneous V
    #     that goes negative on the unit sphere cannot be positive definite, so no
    #     SOS certificate can exist. This catches the "certified V is not even PD"
    #     failure class (e.g. CTA / Levant3D) before any SDP work.
    sc = min_V_on_sphere(V_gf)
    if sc.refutes_pd:
        return JointSOSResult(
            feasible=False,
            solver_status="sphere_refutation",
            solver_requested=_normalize_solver_name(solver),
            message=(
                f"V is NOT positive definite: min V on unit sphere = "
                f"{sc.min_V:.4e} at {sc.argmin} (no SOS certificate can exist)"
            ),
            solve_time=time.time() - start_time,
            term_specs=term_specs,
        )

    coeffs = [float(t.coefficient) for t in V_gf.terms]

    # Compute μ and the y-space degree of V (even, by ensure_even_exponents).
    k, mu = compute_mu_exponents(term_specs, V_gf.weights_r, ensure_even_exponents)
    # μ-preservation guard (P5-03). This μ comes from the V TERMS ALONE, so a
    # sparsified V whose support dropped the terms pinning one axis's exponent
    # denominator lands on a μ that is not proportional to r. The form is then
    # not homogeneous, the Gram basis silently changes here, and the exact gate
    # later refuses target assembly (exact_gate.py:215-217). Typed, not silent.
    from .mu_guard import is_proportional_to_r
    mu_homogeneous = is_proportional_to_r(mu, V_gf.weights_r)
    if not mu_homogeneous:
        return JointSOSResult(
            feasible=False,
            solver_status="mu_not_proportional_to_r",
            solver_requested=_normalize_solver_name(solver),
            message=(
                f"V-only mu {tuple(mu)} is not proportional to "
                f"r={tuple(V_gf.weights_r)}: the associated forms are not "
                f"homogeneous, so no exact re-proof of this candidate can "
                f"exist (see mu_guard, P5-03)"
            ),
            solve_time=time.time() - start_time,
            term_specs=term_specs,
        )
    v_y_degree = 0
    for spec in term_specs:
        deg = sum(mu[i] * float(spec.exponents[i]) for i in range(n_vars))
        v_y_degree = max(v_y_degree, int(round(deg)))

    octants = get_hyperoctant_signs(n_vars)
    oracle = _normalize_solver_name(solver)

    # Strict-positivity margin ε: certify V(y;s) − ε·Σ y_i^{d} SOS, giving
    # V ≥ ε·Σ y_i^{d} > 0. Descend so a "tight" but genuinely PD V is still
    # certified at a smaller (still positive) margin.
    eps_ladder = [float(epsilon), float(epsilon) * 1e-3, float(epsilon) * 1e-6]

    all_attempts: List[str] = []
    solver_used_str = ""
    worst_min_eig: Optional[float] = None
    worst_resid: Optional[float] = None
    worst_cond: Optional[float] = None
    for eps in eps_ladder:
        all_ok = True
        eps_min_eig: Optional[float] = None
        eps_resid: Optional[float] = None
        eps_cond: Optional[float] = None
        eps_captures: List[dict] = []
        for oct_idx, signs in enumerate(octants):
            if verbose:
                print(f"  [ε={eps:.0e}] octant {oct_idx + 1}/{len(octants)}: {signs}")

            # Octant target = Σ_j sign_factor_j · coeff_j · y^{exp_j}
            target: Dict[Tuple[int, ...], float] = {}
            for j, spec in enumerate(term_specs):
                y_exp, sign_factor = term_to_y_monomial(spec, mu, signs)
                target[y_exp] = target.get(y_exp, 0.0) + float(sign_factor) * coeffs[j]
            # subtract ε from each pure top-degree power for strict positivity
            for i in range(n_vars):
                pure = tuple(v_y_degree if t == i else 0 for t in range(n_vars))
                target[pure] = target.get(pure, 0.0) - eps

            ok, gram, used, attempts, diag = _numeric_poly_is_sos(
                target, n_vars, v_y_degree, solver=solver,
                strict_solver=strict_solver, allow_solver_fallback=allow_solver_fallback,
                eps_budget=eps, use_mosek_direct=use_mosek_direct,
            )
            if capture_grams and ok and gram is not None and diag is not None:
                eps_captures.append({
                    "signs": tuple(signs),
                    "octant_index": oct_idx,
                    "eps": float(eps),
                    "mu": tuple(int(v) for v in mu),
                    "y_degree": int(v_y_degree),
                    "target": dict(target),
                    "basis": list(diag.get("basis") or []),
                    "gram": gram,
                })
            all_attempts.extend(f"oct{oct_idx}:{a}" for a in attempts)
            if used:
                solver_used_str = used
            if diag is not None:
                me, rl = diag.get("min_eig"), diag.get("residual_l1")
                cd = diag.get("cond")
                if me is not None:
                    eps_min_eig = me if eps_min_eig is None else min(eps_min_eig, me)
                    worst_min_eig = me if worst_min_eig is None else min(worst_min_eig, me)
                if rl is not None:
                    eps_resid = rl if eps_resid is None else max(eps_resid, rl)
                    worst_resid = rl if worst_resid is None else max(worst_resid, rl)
                if cd is not None:
                    eps_cond = cd if eps_cond is None else max(eps_cond, cd)
                    worst_cond = cd if worst_cond is None else max(worst_cond, cd)
            if not ok:
                all_ok = False
                break

        if all_ok:
            return JointSOSResult(
                feasible=True,
                V_gf=V_gf,
                solver_status=f"standard_sos:{solver_used_str or oracle}",
                solver_requested=oracle,
                solver_used=solver_used_str or oracle,
                solver_attempts=all_attempts,
                message=(
                    f"V verified SOS in all {len(octants)} octants via standard "
                    f"numeric SOS (ε={eps:.0e})"
                ),
                solve_time=time.time() - start_time,
                term_specs=term_specs,
                epsilon_used=str(eps),
                gram_min_eig=eps_min_eig,
                gram_residual_l1=eps_resid,
                gram_cond=eps_cond,
                gram_certificates=eps_captures if capture_grams else None,
            )

    return JointSOSResult(
        feasible=False,
        solver_status="standard_sos_infeasible",
        solver_requested=oracle,
        solver_used=solver_used_str,
        solver_attempts=all_attempts,
        message="V not SOS: no standard SOS certificate found in some octant",
        solve_time=time.time() - start_time,
        term_specs=term_specs,
        gram_min_eig=worst_min_eig,
        gram_residual_l1=worst_resid,
        gram_cond=worst_cond,
    )


def verify_w_given_numeric(
    V_gf: GeneralizedForm,
    dynamics_gf: List[GeneralizedForm],
    weights_r: Tuple[int, ...],
    degree_m: int,
    solver: str = 'CLARABEL',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
    epsilon: float = 1e-6,
    use_mosek_direct: bool = True,
) -> JointSOSResult:
    """Certify W = −dV/dt > 0 on every OPEN orthant for the GIVEN V via standard SOS.

    Unlike ``find_lyapunov_with_dynamics`` this does NOT re-optimize coefficients —
    it checks the cleared-W octant polynomials of the presented V exactly as stored.

    Strictness: the cleared-W octant polynomial typically carries a common monomial
    factor (it vanishes on the coordinate axes / switching surfaces), so we factor it
    out and certify the QUOTIENT is *strictly* positive on the open orthant via an
    ε-margin SOS. This rejects both weak Lyapunov functions (V̇ ≤ 0 but not < 0, e.g.
    a low-gain STA) and indefinite ones (V̇ > 0 somewhere, e.g. the published PID-SMC
    certificate). The SDP solver is the accepted oracle.
    """
    term_specs = [TermSpec(exponents=t.exponents, term_types=t.term_types) for t in V_gf.terms]
    coeffs = [float(t.coefficient) for t in V_gf.terms]
    _k, mu = compute_mu_exponents_with_dynamics(
        term_specs=term_specs, weights_r=weights_r, dynamics_gf=dynamics_gf,
        ensure_even_exponents=True,
    )
    opt = JointSOSOptimizer(
        weights_r=weights_r, solver=solver, allow_solver_fallback=allow_solver_fallback,
    )
    octants = get_hyperoctant_signs(opt.n_vars)
    attempts: List[str] = []
    used = ""
    worst_min_eig: Optional[float] = None
    worst_resid: Optional[float] = None
    worst_cond: Optional[float] = None
    for signs in octants:
        M_W, _cp, w_monos = opt._compute_w_coefficient_matrix_weighted_sphere(
            term_specs, mu, signs, dynamics_gf
        )
        if M_W is None:
            return JointSOSResult(
                feasible=False, V_gf=V_gf, solver_status="w_assembly_failed",
                solver_requested=solver, term_specs=term_specs,
                message="W assembly failed on an octant",
            )
        target: Dict[Tuple[int, ...], float] = {}
        # Coefficient mass kept out of the target by the solver-encoding floor.
        # It is handed to ``_numeric_poly_is_sos`` as ``extra_residual_l1`` and
        # charged against the ε budget instead of being discarded (review
        # finding 6, 2026-09-05).
        w_dropped_l1 = 0.0
        for r_idx, mono in enumerate(w_monos):
            c = sum(float(M_W[r_idx, j]) * coeffs[j] for j in range(len(coeffs)))
            if abs(c) > SOLVER_COEFF_FLOOR:
                target[mono] = c
            else:
                w_dropped_l1 += abs(c)
        if not target:
            return JointSOSResult(
                feasible=False, V_gf=V_gf, solver_status="w_identically_zero",
                solver_requested=solver, term_specs=term_specs,
                message="cleared W identically zero on an octant",
            )

        # Factor the common monomial (cleared W vanishes on the axes) and certify
        # the quotient is STRICTLY positive on the open orthant. With even
        # y-exponents (ensure_even_exponents) the quotient is an even-degree form.
        monos = list(target.keys())
        common = tuple(min(m[k] for m in monos) for k in range(opt.n_vars))
        quotient = {
            tuple(m[k] - common[k] for k in range(opt.n_vars)): c
            for m, c in target.items()
        }
        q_deg = max((sum(m) for m in quotient), default=0)
        for i in range(opt.n_vars):
            pure = tuple(q_deg if t == i else 0 for t in range(opt.n_vars))
            quotient[pure] = quotient.get(pure, 0.0) - float(epsilon)

        ok, _g, su, att, diag = _numeric_poly_is_sos(
            quotient, opt.n_vars, q_deg, solver=solver,
            strict_solver=strict_solver,
            allow_solver_fallback=allow_solver_fallback,
            eps_budget=float(epsilon), use_mosek_direct=use_mosek_direct,
            extra_residual_l1=w_dropped_l1,
        )
        attempts.extend(att)
        if su:
            used = su
        if diag is not None:
            me, rl = diag.get("min_eig"), diag.get("residual_l1")
            cd = diag.get("cond")
            if me is not None:
                worst_min_eig = me if worst_min_eig is None else min(worst_min_eig, me)
            if rl is not None:
                worst_resid = rl if worst_resid is None else max(worst_resid, rl)
            if cd is not None:
                worst_cond = cd if worst_cond is None else max(worst_cond, cd)
        if not ok:
            return JointSOSResult(
                feasible=False, V_gf=V_gf,
                solver_status="standard_sos_infeasible", solver_requested=solver,
                solver_used=used, solver_attempts=attempts, term_specs=term_specs,
                message="W not strictly positive on some open orthant (standard SOS)",
                gram_min_eig=worst_min_eig, gram_residual_l1=worst_resid,
                gram_cond=worst_cond,
            )
    return JointSOSResult(
        feasible=True, V_gf=V_gf,
        solver_status=f"standard_sos:{used or solver}", solver_requested=solver,
        solver_used=used or solver, solver_attempts=attempts, term_specs=term_specs,
        message="W > 0 on all open orthants (standard SOS)",
        gram_min_eig=worst_min_eig, gram_residual_l1=worst_resid,
        gram_cond=worst_cond,
    )
