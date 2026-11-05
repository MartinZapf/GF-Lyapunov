"""Exact-rational gate (Proposition 7) — unit and end-to-end tests.

Pins four things:

1. The exact LDLᵀ PSD test (``ldl_psd``) on hand-built PSD / indefinite /
   rank-deficient / zero-diagonal matrices, plus randomized AᵀA instances.

2. The exact Lie-derivative mirror reproduces the float pipeline
   (``compute_lie_derivative_gf``) on published certificates: identical term
   support, coefficients to float accuracy.

3. End-to-end Proposition-7 verification (re-solve with CLARABEL, capture
   Grams, exact-check) passes on a small published certificate.

4. Soundness: a corrupted Gram and a corrupted certificate coefficient are
   rejected by the exact check (the budget/PSD chain, not solver status, is
   the authority).
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf, negate_gf
from gf_framework.verification.exact_gate import (
    ExactGateError,
    exact_neg_lie_derivative,
    exact_octant_target,
    exact_terms_from_gf,
    exact_validate_octant,
    exact_verify_captures,
    ldl_psd,
    validate_mu,
)
from gf_framework.verification.joint_sos import verify_gf_sos_full

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "artifacts/publication/results_gf_sos_clean/gf_matrix_20260610_230219.json"

SPEC_FACTORIES = {
    "sta": "create_sta_spec", "twisting": "create_twisting_spec",
    "levant3d": "create_levant3d_spec", "fosmc": "create_fosmc_spec",
}


def _published_row(system):
    if not MATRIX.exists():
        pytest.skip("published matrix artifact not present")
    data = json.load(open(MATRIX))
    for row in data["results"]:
        if row["system"] == system and row.get("certified") and row.get("V_terms"):
            return row
    pytest.skip(f"no published certified row for {system}")


def _reconstruct(row):
    terms = [
        GFTerm(coefficient=float(t["coefficient"]),
               exponents=tuple(Fraction(e) for e in t["exponents"]),
               term_types=tuple(t["term_types"]))
        for t in row["V_terms"]
    ]
    return GeneralizedForm(degree_m=row["degree_used"],
                           weights_r=tuple(row["weights_r"]), terms=terms)


def _F(rows):
    return [[Fraction(x) for x in row] for row in rows]


# ---------------------------------------------------------------------------
# 1. Exact LDLᵀ PSD test
# ---------------------------------------------------------------------------

def test_ldl_accepts_identity_and_pd():
    assert ldl_psd(_F([[1, 0], [0, 1]]))
    assert ldl_psd(_F([[2, 1], [1, 2]]))


def test_ldl_accepts_psd_rank_deficient():
    # rank-1 PSD
    assert ldl_psd(_F([[1, 1], [1, 1]]))
    # zero matrix
    assert ldl_psd(_F([[0, 0], [0, 0]]))


def test_ldl_rejects_indefinite_and_negative():
    assert not ldl_psd(_F([[1, 0], [0, -1]]))
    assert not ldl_psd(_F([[-1, 0], [0, 1]]))
    # zero diagonal with nonzero off-diagonal is NOT PSD
    assert not ldl_psd(_F([[0, 1], [1, 0]]))
    # PD diagonal but indefinite overall (det < 0)
    assert not ldl_psd(_F([[1, 2], [2, 1]]))


def test_ldl_randomized_gram_instances():
    rng = np.random.default_rng(7)
    for _ in range(5):
        A = rng.integers(-3, 4, size=(5, 5))
        G = A.T @ A  # PSD by construction, integer entries
        M = _F(G.tolist())
        assert ldl_psd(M)
        # subtracting slightly more than λ_min breaks PSD
        lam = float(np.linalg.eigvalsh(G.astype(float)).min())
        shift = Fraction(float(lam)) + Fraction(1, 100)
        M2 = [[M[i][j] - (shift if i == j else 0) for j in range(5)] for i in range(5)]
        assert not ldl_psd(M2)


# ---------------------------------------------------------------------------
# 2. Exact Lie mirror vs float pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", ["sta", "levant3d"])
def test_exact_lie_matches_float_pipeline(system):
    row = _published_row(system)
    V = _reconstruct(row)
    spec = getattr(ud, SPEC_FACTORIES[system])()
    W_float = negate_gf(compute_lie_derivative_gf(V, spec.dynamics_gf, validate=False))
    w_exact = exact_neg_lie_derivative(
        exact_terms_from_gf(V),
        [exact_terms_from_gf(f_i) for f_i in spec.dynamics_gf],
    )
    fmap = {(t.exponents, t.term_types): float(t.coefficient) for t in W_float.terms}
    emap = {(e, t): float(c) for c, e, t in w_exact}
    assert set(fmap) == set(emap), "term support differs between exact and float Lie"
    scale = max(abs(v) for v in fmap.values())
    for k in fmap:
        assert abs(fmap[k] - emap[k]) <= 1e-12 * max(1.0, scale)


def test_exact_octant_target_rejects_non_integer_lift():
    # x^(1/3) with μ = 2 → y-exponent 2/3: must raise, not round
    terms = [(Fraction(1), (Fraction(1, 3),), ("abs",))]
    with pytest.raises(ExactGateError):
        exact_octant_target(terms, (2,), (1,), Fraction(1, 10**6))


def test_exact_octant_target_margin_and_homogeneity():
    # V = |x1|^2 + |x2|^2 with μ=(2,2): associated form y1^4 + y2^4
    terms = [
        (Fraction(3), (Fraction(2), Fraction(0)), ("abs", "abs")),
        (Fraction(5), (Fraction(0), Fraction(2)), ("abs", "abs")),
    ]
    eps = Fraction(1, 1000)
    target, d = exact_octant_target(terms, (2, 2), (1, -1), eps)
    assert d == 4
    assert target[(4, 0)] == Fraction(3) - eps
    assert target[(0, 4)] == Fraction(5) - eps


# ---------------------------------------------------------------------------
# 3. End-to-end Proposition 7 on a published certificate (CLARABEL)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", ["fosmc", "sta"])
def test_exact_pass_end_to_end(system):
    row = _published_row(system)
    V = _reconstruct(row)
    spec = getattr(ud, SPEC_FACTORIES[system])()
    n = len(spec.weights_r)

    res_v = verify_gf_sos_full(V, epsilon=1e-6, solver="CLARABEL",
                               strict_solver=True, allow_solver_fallback=False,
                               capture_grams=True)
    assert res_v.feasible and res_v.gram_certificates
    v_exact = exact_verify_captures(exact_terms_from_gf(V), res_v.gram_certificates, n)
    assert v_exact.ok, v_exact.reason

    W = negate_gf(compute_lie_derivative_gf(V, spec.dynamics_gf, validate=False))
    res_w = verify_gf_sos_full(W, epsilon=1e-6, solver="CLARABEL",
                               strict_solver=True, allow_solver_fallback=False,
                               capture_grams=True)
    assert res_w.feasible and res_w.gram_certificates
    w_terms = exact_neg_lie_derivative(
        exact_terms_from_gf(V), [exact_terms_from_gf(f_i) for f_i in spec.dynamics_gf])
    w_exact = exact_verify_captures(w_terms, res_w.gram_certificates, n)
    assert w_exact.ok, w_exact.reason


# ---------------------------------------------------------------------------
# 4. Soundness: corrupted inputs must be rejected
# ---------------------------------------------------------------------------

def _sta_v_captures():
    row = _published_row("sta")
    V = _reconstruct(row)
    res = verify_gf_sos_full(V, epsilon=1e-6, solver="CLARABEL",
                             strict_solver=True, allow_solver_fallback=False,
                             capture_grams=True)
    assert res.feasible and res.gram_certificates
    return V, res.gram_certificates


def test_corrupted_gram_rejected():
    V, captures = _sta_v_captures()
    cap = captures[0]
    bad = dict(cap)
    g = np.array(cap["gram"], dtype=float).copy()
    g[0, 0] -= 1e-3  # large enough to break the residual/PSD budget
    bad["gram"] = g
    terms = exact_terms_from_gf(V)
    target, d = exact_octant_target(
        terms, tuple(bad["mu"]), tuple(bad["signs"]), Fraction(float(bad["eps"])))
    check = exact_validate_octant(target, d, bad["gram"], bad["basis"],
                                  Fraction(float(bad["eps"])), tuple(bad["signs"]))
    assert not check.ok


def test_corrupted_coefficient_rejected():
    V, captures = _sta_v_captures()
    # Perturb a V coefficient by 1%: the exact target no longer matches the Gram.
    terms = exact_terms_from_gf(V)
    c0, e0, t0 = terms[0]
    terms_bad = [(c0 * Fraction(101, 100), e0, t0)] + list(terms[1:])
    cap = captures[0]
    target, d = exact_octant_target(
        terms_bad, tuple(cap["mu"]), tuple(cap["signs"]), Fraction(float(cap["eps"])))
    check = exact_validate_octant(target, d, cap["gram"], cap["basis"],
                                  Fraction(float(cap["eps"])), tuple(cap["signs"]))
    assert not check.ok
    assert check.reason == "budget_exceeded"


# ---------------------------------------------------------------------------
# Per-endpoint exact phase on a perturbed inclusion (the flagship pattern):
# V once, W± per endpoint vertex, interval covered by affinity+convexity.
# Also pins the shared orchestrator run_exact_phase (exact_phase.py).
# ---------------------------------------------------------------------------

def test_exact_phase_per_endpoint_fosmc():
    from gf_framework.discovery.universal_discovery import create_fosmc_spec
    from gf_framework.verification.exact_phase import run_exact_phase

    spec = create_fosmc_spec(k=1.0, delta0=0.5)
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])
    res = run_exact_phase(V, spec.dynamics_gf_vertices, 1,
                          epsilon=1e-6, solver="CLARABEL")
    assert res.v_resolve_feasible and res.v_exact.ok
    assert len(res.endpoints) == 2
    for ep in res.endpoints:
        assert ep.resolve_feasible and ep.exact.ok, ep.exact.reason
        assert ep.lie_crosscheck["support_match"]
        assert ep.lie_crosscheck["max_coeff_diff"] < 1e-12
    assert res.exact_pass
    assert res.worst_charge_ratio is not None and res.worst_charge_ratio <= 1.0


def test_exact_verify_captures_rejects_invalid_sign_label():
    """M9: a fabricated/corrupted capture with a sign label outside
    {-1,+1}^n (e.g. (0,)) must be rejected outright, not merely by count/
    distinctness — mirrors the standalone verifier's `seen != full` check."""
    captures = [
        {"signs": (1,), "eps": 1e-6},
        {"signs": (0,), "eps": 1e-6},  # invalid: 0 not in {-1, +1}
    ]
    res = exact_verify_captures([], captures, 1)
    assert not res.ok
    assert "invalid_sign_label" in res.reason


def test_exact_verify_captures_rejects_zero_epsilon():
    """epsilon = 0 makes the budget eps/2 = 0, silently weakening (or
    trivializing) the strict-positivity conclusion; must be rejected."""
    captures = [
        {"signs": (1,), "eps": 0.0},
        {"signs": (-1,), "eps": 0.0},
    ]
    res = exact_verify_captures([], captures, 1)
    assert not res.ok
    assert "epsilon_not_positive" in res.reason


def test_exact_verify_captures_rejects_nonuniform_epsilon():
    """Every octant of one verification set must share the same margin —
    a non-uniform epsilon across captures would let one octant's proof rest
    on a smaller margin than the caller believes was certified overall."""
    captures = [
        {"signs": (1,), "eps": 1e-6},
        {"signs": (-1,), "eps": 1e-9},
    ]
    res = exact_verify_captures([], captures, 1)
    assert not res.ok
    assert "epsilon_not_uniform_across_captures" in res.reason


def test_exact_phase_fails_above_gain():
    """δ₀ = 1.5 > k = 1: W(+endpoint) = 2|s| - 3s is negative for s > 0 — the
    endpoint W re-solve (or its exact proof) must fail, and exact_pass with it."""
    from gf_framework.discovery.universal_discovery import create_fosmc_spec
    from gf_framework.verification.exact_phase import run_exact_phase

    spec = create_fosmc_spec(k=1.0, delta0=1.5)
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])
    res = run_exact_phase(V, spec.dynamics_gf_vertices, 1,
                          epsilon=1e-6, solver="CLARABEL")
    assert not res.exact_pass


# ---------------------------------------------------------------------------
# Regressions from the 2026-09-05 external scientific review
#
# Findings 2 (Laurent basis accepted by the exact block validator) and 5 (mu
# coerced instead of validated). The reviewer drove both through the portable
# verifier AND through this module; the instances are pinned on both sides.
# ---------------------------------------------------------------------------

REVIEW_V_TERMS = [(Fraction(-1), (Fraction(2), Fraction(0)), ("abs", "abs")),
                  (Fraction(4), (Fraction(0), Fraction(4)), ("abs", "abs"))]
REVIEW_MU = (4, 2)
REVIEW_GRAM = np.array([[0.5, -1.0], [-1.0, 3.0]])


def _review_target():
    """V = -x1^2 + 4 x2^4, r = (2,1), m = 4, mu = (4,2), eps = 1."""
    return exact_octant_target(REVIEW_V_TERMS, REVIEW_MU, (1, 1), Fraction(1))


def test_review_finding2_laurent_basis_is_refused():
    """Regression from the 2026-09-05 external scientific review, finding 2.

    V(1, 0) = -1, so nothing may certify its positivity. The basis
    {(8, -4), (0, 4)} passes ``2*sum(beta) == d`` and, with the positive
    definite Gram [[1/2, -1], [-1, 3]], used to be ACCEPTED with a certified
    margin of 1/2. Proposition 7's two bounds (the residual 1-norm and the
    eta*C1 shift) are inequalities about polynomial monomials; a Laurent
    exponent breaks both, and the residual -1/2 y1^16 y2^-8 is unbounded as
    y2 -> 0. Every raw exponent must be a nonnegative integer."""
    target, d = _review_target()
    check = exact_validate_octant(target, d, REVIEW_GRAM, [(8, -4), (0, 4)],
                                  Fraction(1))
    assert not check.ok
    assert "invalid_basis_exponent" in check.reason
    assert "violates >= 0" in check.reason


def test_review_finding2_basis_dimension_duplicates_and_degree_are_refused():
    """Regression from the 2026-09-05 external scientific review, finding 2."""
    target, d = _review_target()
    for basis, needle in (([(4,), (0,)], "invalid_basis_exponent"),
                          ([(3.5, 0.5), (0, 4)], "not an integer"),
                          ([(0, 4), (0, 4)], "invalid_basis_duplicate"),
                          ([], "empty_basis"),
                          ([(8, 0), (0, 8)], "basis_degree_mismatch")):
        check = exact_validate_octant(target, d, REVIEW_GRAM, basis, Fraction(1))
        assert not check.ok and needle in check.reason, (basis, check.reason)


def test_review_finding2_non_finite_gram_is_refused():
    """Regression from the 2026-09-05 external scientific review, finding 2.

    Fraction(nan) raises and an infinity poisons every exact comparison, so a
    non-finite Gram is refused as a malformed capture."""
    target, d = _review_target()
    bad = np.array([[float("nan"), 0.0], [0.0, 1.0]])
    check = exact_validate_octant(target, d, bad, [(4, 0), (0, 4)], Fraction(1))
    assert not check.ok and check.reason == "gram_not_finite"


def test_review_finding5_mu_is_validated_not_coerced():
    """Regression from the 2026-09-05 external scientific review, finding 5.

    ``tuple(int(v) for v in capture["mu"])`` read a declared lift of 2.9 as 2.
    x_i = s_i y_i^{mu_i} is a chart only for integral mu_i >= 1, so fractional,
    zero, negative and wrong-arity lifts are refused on the RAW value."""
    for mu, needle in (((4, 2.9), "not an integer"),
                       ((4, 0), "violates >= 1"),
                       ((4, -2), "violates >= 1"),
                       ((4, True), "boolean")):
        with pytest.raises(ExactGateError) as exc:
            exact_octant_target(REVIEW_V_TERMS, mu, (1,) * len(mu), Fraction(1))
        assert needle in str(exc.value), (mu, str(exc.value))
    # A lift of the wrong arity cannot address the form at all: the terms
    # carry two exponents, so a 1- or 3-variable lift is refused outright.
    for mu in ((4,), (4, 2, 1)):
        with pytest.raises(ExactGateError) as exc:
            exact_octant_target(REVIEW_V_TERMS, mu, (1,) * len(mu), Fraction(1))
        assert f"the lift has {len(mu)} variables" in str(exc.value)
    # And against a capture's declared arity, where n_vars is known.
    with pytest.raises(ExactGateError):
        validate_mu((4, 2, 1), 2, "mu")


def test_review_finding5_captures_must_declare_one_and_the_same_mu():
    """Regression from the 2026-09-05 external scientific review, finding 5.

    One form is proved under ONE association: 2^n octants under different
    lifts are charts of different substitutions and do not cover R^n."""
    caps = [{"signs": (1,), "eps": 1e-6, "mu": (2,), "gram": [[1.0]],
             "basis": [(1,)]},
            {"signs": (-1,), "eps": 1e-6, "mu": (4,), "gram": [[1.0]],
             "basis": [(1,)]}]
    res = exact_verify_captures([], caps, 1)
    assert not res.ok and "mu_not_uniform_across_captures" in res.reason


def test_review_finding5_a_fractional_mu_in_a_capture_is_refused():
    """Regression from the 2026-09-05 external scientific review, finding 5."""
    caps = [{"signs": (1,), "eps": 1e-6, "mu": (2.9,), "gram": [[1.0]],
             "basis": [(1,)]},
            {"signs": (-1,), "eps": 1e-6, "mu": (2.9,), "gram": [[1.0]],
             "basis": [(1,)]}]
    res = exact_verify_captures([], caps, 1)
    assert not res.ok and "invalid_mu" in res.reason


def test_review_finding5_a_fractional_sign_label_is_refused():
    """Regression from the 2026-09-05 external scientific review, finding 5.

    Sign labels were coerced too: int(1.5) == 1 would have turned a corrupted
    label into a valid-looking octant."""
    caps = [{"signs": (1.5,), "eps": 1e-6, "mu": (2,), "gram": [[1.0]],
             "basis": [(1,)]},
            {"signs": (-1,), "eps": 1e-6, "mu": (2,), "gram": [[1.0]],
             "basis": [(1,)]}]
    res = exact_verify_captures([], caps, 1)
    assert not res.ok and "invalid_sign_label" in res.reason
