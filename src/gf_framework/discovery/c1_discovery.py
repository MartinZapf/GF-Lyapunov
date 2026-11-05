"""Deterministic C¹-restricted Lyapunov discovery (Sanchez-Moreno, Lemma 5).

The continuous-time stability bridge is Lemma 5 (Filippov), which requires V ∈ C¹
(restriction R2: abs-exponents ∈ {0}∪(1,∞), signed-exponents ≥ 1, and m > maxᵢ rᵢ
for the diagonal terms). A C¹ V makes ∇V continuous across the switching surfaces
and W = -∇V·f free of negative exponents — so no clearing, no surface analysis,
no Clarke.

This module searches **only the C¹ (R2) term class** (enumerating classical,
often non-C¹, homogeneous structures would propose candidates the sound verifier
must reject). The search is deterministic: enumerate the C¹ term library, then
let the exact joint-SOS engine find certifying coefficients for increasing term
subsets. Every accepted result is
re-checked by the authoritative exact-rational gate (``certify_discovered_gf``).

If a system's published homogeneity has m = maxᵢ rᵢ (e.g. STA/twisting at m=2, or
the 1-D reaching laws at m=1), the max-weight diagonal term is |xᵢ|^1 — non-C¹ —
so no C¹ term spans that variable and **no C¹ GF Lyapunov function exists at that
homogeneity**. That is reported honestly rather than worked around.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..algebra.core import GeneralizedForm, GFTerm
from ..algebra.enumerate import enumerate_diagonal_terms, enumerate_cross_terms
from ..interfaces import TermSpec
from ..verification.joint_sos import JointSOSOptimizer
from ..verification.verifier import certify_discovered_gf, classify_smoothness


def is_c1_term(term: TermSpec, weights_r: Tuple[int, ...], degree_m: int) -> bool:
    """True iff a single GF term is C¹ (restriction R2).

    Uses the canonical classifier on the one-term GF so the definition cannot
    drift from the verifier's surface-scope logic.
    """
    gf = GeneralizedForm(
        degree_m=degree_m, weights_r=weights_r,
        terms=[GFTerm(coefficient=1.0, exponents=term.exponents, term_types=term.term_types)],
    )
    return classify_smoothness(gf)["regularity"] == "C1"


@dataclass
class C1DiscoveryResult:
    system: str = ""
    certified: bool = False                # full Lemma-5 asymptotic stability
    c1_lf_exists: bool = False             # a C¹ PD LF with V̇<0 was found
    V_gf: Optional[GeneralizedForm] = None
    coefficients: Optional[dict] = None
    degree_used: int = 0                   # LF degree m at which it certified
    n_terms: int = 0
    regularity: str = ""
    surface_scope: str = "open_orthant_only"
    claim_level: str = "candidate"
    candidates_tried: int = 0
    degrees_tried: list = field(default_factory=list)
    total_time: float = 0.0
    message: str = ""
    c1_diag: int = 0
    c1_cross: int = 0


def _spans_all_vars(terms: List[TermSpec], n: int) -> bool:
    seen = set()
    for t in terms:
        for i, e in enumerate(t.exponents):
            if float(e) != 0.0:
                seen.add(i)
    return len(seen) == n


def _degree_candidates(degree_m: int, weights_r: Tuple[int, ...],
                       max_extra_degree: int) -> List[int]:
    """LF degrees to try, smallest first. C¹ (R2) needs m > maxᵢ rᵢ for the
    diagonal terms, so the smallest C¹-admissible degree is maxr+1 — start there
    (the cheapest certificate) and escalate, regardless of the system's published
    homogeneity. Higher m (with fractional exponents) is exactly how Sanchez-Moreno
    obtain smooth GF Lyapunov functions for systems whose classical LF is non-smooth.

    The ``degree_m`` argument is accepted for signature stability but no longer
    pins the base (a published m > maxr+1 would skip cheaper lower-degree certs)."""
    maxr = max(weights_r)
    base = maxr + 1
    return [base + k for k in range(0, max_extra_degree + 1)]


def _search_at_degree(
    system, weights_r, m, dynamics_gf, solver,
    max_cross_singles, max_exact_attempts, time_budget, t0, out, verbose,
):
    """Search C¹ term subsets at a fixed LF degree m. Returns
    ('certified', res, cert, n) | ('partial', res, cert, n) | ('none', diag, cross)."""
    n = len(weights_r)
    diag = [t for t in enumerate_diagonal_terms(m, weights_r)
            if is_c1_term(t, weights_r, m)]
    # Legacy subset-search path: no μ-filter, so cap-4 here is a search-breadth
    # bound, NOT a lattice-completeness claim (the paper pipeline is discover_gf,
    # whose cap is derived from μ via gf_discovery.complete_lattice_cap).
    cross_all = enumerate_cross_terms(
        m, weights_r, max_denominator=4,
        include_signed=True, minimal=False, include_abs_variants=True,
    )
    cross = _prioritize([t for t in cross_all if is_c1_term(t, weights_r, m)])
    out.c1_diag, out.c1_cross = len(diag), len(cross)
    if verbose:
        print(f"[{system}] m={m}: C¹ library {len(diag)} diagonal, {len(cross)} cross "
              f"(maxr={max(weights_r)})")
    if not _spans_all_vars(diag, n):
        return ("none", diag, cross)

    opt = JointSOSOptimizer(weights_r=weights_r, solver=solver, verbose=False)

    def _screen(cand):
        ts = time.time()
        try:
            r = opt.find_lyapunov_with_dynamics(
                degree_m=m, dynamics_gf=dynamics_gf, active_terms=cand)
        except Exception as e:
            if verbose:
                print(f"  m={m} numeric-screen ({len(cand)} terms): solver error {e}")
            return None
        if verbose:
            print(f"  m={m} numeric-screen ({len(cand)} terms): "
                  f"feasible={r.feasible} [{time.time()-ts:.1f}s]", flush=True)
        return r

    # KEY: the full C¹ library is the superset of every subset, so a SINGLE numeric
    # SDP solve decides feasibility at this degree — if infeasible, escalate.
    full = list(diag) + list(cross)
    out.candidates_tried += 1
    screen_full = _screen(full)
    if screen_full is None or not screen_full.feasible:
        if verbose:
            print(f"  m={m}: full C¹ library ({len(full)} terms) numerically "
                  f"infeasible → escalate degree")
        return ("none", diag, cross)

    # Feasible. The numeric solution already sparsifies — most cross coefficients
    # are ~0. PRUNE to the significant terms (diagonals always kept, they carry
    # positive-definiteness) so the expensive exact certification runs on a small
    # candidate. Exact-certify the pruned structure first, full library as fallback.
    ndiag = len(diag)
    coefs = screen_full.coefficients or {}
    vals = [abs(float(coefs.get(f"alpha{i}", 0.0))) for i in range(len(full))]
    mx = max(vals, default=0.0)
    tol = 1e-4 * mx if mx > 0 else 0.0
    pruned = list(diag) + [cross[j] for j in range(len(cross)) if vals[ndiag + j] > tol]

    candidates: List[List[TermSpec]] = []
    if len(pruned) < len(full):
        candidates.append(pruned)
    candidates.append(full)

    partial = None
    attempts = 0
    for cand in candidates:
        if attempts >= max_exact_attempts or time.time() - t0 > time_budget:
            break
        # The pruned structure needs its own coefficients; the gate re-optimizes
        # via joint SOS, so re-screen the pruned candidate to get a starting V.
        screen = screen_full if cand is full else _screen(cand)
        if screen is None or not screen.feasible:
            continue
        attempts += 1
        tsc = time.time()
        cert = certify_discovered_gf(screen.V_gf, weights_r, m, dynamics_gf, solver=solver)
        if verbose:
            print(f"  m={m} exact-cert ({len(cand)} terms) [{time.time()-tsc:.1f}s]: "
                  f"V={cert.V_verified} W={cert.W_verified} reg={cert.regularity} "
                  f"surface_certified={cert.surface_certified}", flush=True)
        if cert.surface_certified:
            return ("certified", screen, cert, len(cand))
        if cert.V_verified and cert.W_verified and partial is None:
            partial = (screen, cert, len(cand))
    if partial is not None:
        return ("partial",) + partial
    return ("none", diag, cross)


def discover_c1_lyapunov(
    system: str,
    weights_r: Tuple[int, ...],
    degree_m: int,
    dynamics_gf: List[GeneralizedForm],
    solver: str = "MOSEK",
    max_cross_singles: int = 12,
    max_exact_attempts: int = 5,
    max_extra_degree: int = 4,
    time_budget: float = 1200.0,
    verbose: bool = False,
) -> C1DiscoveryResult:
    """Search the C¹ (R2) term class for a Lyapunov function and certify it,
    escalating the LF degree m (with fractional exponents) until one certifies.

    Returns the first degree at which a complete Lemma-5 certificate is obtained.
    """
    t0 = time.time()
    out = C1DiscoveryResult(system=system)
    degrees = _degree_candidates(degree_m, weights_r, max_extra_degree)
    partial_best = None

    for m in degrees:
        if time.time() - t0 > time_budget:
            break
        out.degrees_tried.append(m)
        r = _search_at_degree(
            system, weights_r, m, dynamics_gf, solver,
            max_cross_singles, max_exact_attempts, time_budget, t0, out, verbose)
        if r[0] == "certified":
            _res, cert, ncand = r[1], r[2], r[3]
            out.certified = True
            out.c1_lf_exists = True
            out.V_gf = cert.V_gf if cert.V_gf is not None else _res.V_gf
            out.coefficients = _res.coefficients
            out.degree_used = m
            out.n_terms = ncand
            out.regularity = cert.regularity
            out.surface_scope = cert.surface_scope
            out.claim_level = "lemma5_asymptotic_stability"
            out.message = (f"Certified C¹ Lyapunov function ({ncand} terms, m={m}) "
                           f"via Lemma 5.")
            out.total_time = time.time() - t0
            return out
        if r[0] == "partial" and partial_best is None:
            partial_best = (m, r[1], r[2], r[3])

    if partial_best is not None:
        m, res, cert, ncand = partial_best
        out.c1_lf_exists = False
        out.V_gf = res.V_gf
        out.coefficients = res.coefficients
        out.degree_used = m
        out.n_terms = ncand
        out.regularity = cert.regularity
        out.surface_scope = cert.surface_scope
        out.claim_level = "open_orthant_decrease_non_c1"
        out.message = ("V>0 ∧ V̇<0 on open orthants, but the certified structure is "
                       f"not C¹ (regularity={cert.regularity}, m={m}).")
        out.total_time = time.time() - t0
        return out

    out.claim_level = "no_c1_lf_found"
    out.message = (f"No certifying C¹ Lyapunov function found over degrees "
                   f"{out.degrees_tried} ({out.candidates_tried} subsets tried).")
    out.total_time = time.time() - t0
    return out


def _prioritize(cross_terms: List[TermSpec]) -> List[TermSpec]:
    """Prefer signed, low-denominator, integer-exponent cross terms first."""
    def key(t: TermSpec):
        n_signed = sum(1 for ty in t.term_types if ty == "signed")
        denom_sum = sum(getattr(e, "denominator", 1) for e in t.exponents)
        n_int = sum(1 for e in t.exponents
                    if float(e) != 0 and getattr(e, "denominator", 1) == 1)
        return (-n_signed, denom_sum, -n_int)
    return sorted(cross_terms, key=key)
