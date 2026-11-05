"""Review finding 6 (2026-09-05): the float SOS gate must not lose target mass.

``joint_sos._numeric_poly_is_sos`` used to open with

    target = {m: float(c) for m, c in target.items() if abs(float(c)) > 1e-15}

which had two consequences, both fixed here and pinned below:

1. every coefficient at or below the ABSOLUTE floor 1e-15 disappeared before
   ``_validate_gram_certificate`` ever saw the target, so its mass was never
   charged against the strictness margin ε. At the bottom of the ε ladder that
   omitted mass can exceed the budget it was supposed to be measured against;
2. if the truncation emptied the target, the function returned
   ``(True, None, "trivial_zero", …)`` — a strict target reported SOS with no
   Gram at all.

The fix keeps the floor as a SOLVER-ENCODING floor only
(``joint_sos.SOLVER_COEFF_FLOOR``) and charges the dropped ‖·‖₁ mass to the
a-posteriori residual via ``extra_residual_l1``, and it splits the three cases
that used to collapse onto ``trivial_zero``: the genuinely zero polynomial with
no margin (``trivial_zero``, no Gram), an exactly zero margin-subtracted target
(``zero_remainder``, the zero Gram of the half-basis), and a target that is
empty only after truncation (``target_below_solver_floor``, refused).

The middle case is the follow-up review's finding A (2026-09-05). The first
repair refused every zero target under a margin as ``empty_strict_target``, on
the premise that the ε·Σᵢ y_iᵈ margin always leaves nonzero pure powers. It
does not: for ``F(y) = ε·Σᵢ y_iᵈ`` the margin-subtracted target is identically
zero, its SOS remainder is the zero polynomial with Gram ``Q = 0``, and the
exact/portable validators accept it with charge 0. The float path now agrees.
Dropped mass is still charged on that branch, so a zero target whose caller
lost more than the budget is refused.

Every test here is solver-free. The one test that needs a Gram injects it by
stubbing the MOSEK Task-API rung, so what is pinned is this module's own
arithmetic and never a solver's behaviour. The exact D→X→P route rebuilds its
targets in rational arithmetic and is untouched by all of this.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from gf_framework.verification import joint_sos
from gf_framework.verification.joint_sos import (
    SOLVER_COEFF_FLOOR,
    _numeric_poly_is_sos,
    _validate_gram_certificate,
)

EPS = 1e-6

# One variable, d = 2, basis [(1,)], target {y²: 1}, Gram [[1]]: the exact fit,
# so ‖E‖₁ = 0 and η = 0 and the whole charge is whatever we hand in as dropped
# mass. Same instance as tests/test_margin_policy.py.
ONE_VAR_BASIS: List[Tuple[int, ...]] = [(1,)]
ONE_VAR_TARGET: Dict[Tuple[int, ...], float] = {(2,): 1.0}
ONE_VAR_GRAM = np.array([[1.0]])


# --------------------------------------------------------------------------
# 1. the dropped mass reaches _validate_gram_certificate
# --------------------------------------------------------------------------

def test_exact_fit_without_dropped_mass_is_accepted():
    """Baseline: charge 0, so both policies accept and the residual is 0."""
    for policy in ("half", "achieved"):
        ok, min_eig, resid, why = _validate_gram_certificate(
            ONE_VAR_TARGET, ONE_VAR_GRAM, ONE_VAR_BASIS, 1, EPS, 1e-7,
            budget_policy=policy)
        assert ok, (policy, why)
        assert resid == pytest.approx(0.0, abs=1e-18)
        assert min_eig == pytest.approx(1.0)


@pytest.mark.parametrize("residual_model", ["dense", "deck_blocked"])
def test_dropped_mass_is_added_to_the_charged_residual(residual_model):
    """``extra_residual_l1`` is ‖·‖₁ mass and is charged as such, under BOTH
    residual models — the projection can only delete Gram pairs, never the
    caller's truncation."""
    dropped = 0.25 * EPS
    diag: Dict[str, float] = {}
    ok, _min_eig, resid, _why = _validate_gram_certificate(
        ONE_VAR_TARGET, ONE_VAR_GRAM, ONE_VAR_BASIS, 1, EPS, 1e-7,
        residual_model=residual_model, diag=diag, extra_residual_l1=dropped)
    assert ok
    assert resid == pytest.approx(dropped, rel=1e-12)
    assert diag["charge"] == pytest.approx(dropped, rel=1e-12)
    assert diag["eps_certified"] == pytest.approx(EPS - dropped, rel=1e-9)


def test_dropped_mass_over_the_budget_flips_acceptance_to_refusal():
    """The defect in one line: without the charge this Gram is a perfect fit
    and is accepted; with 1.5·ε of dropped target mass it must be refused."""
    dropped = 1.5 * EPS
    ok_uncharged, _, _, _ = _validate_gram_certificate(
        ONE_VAR_TARGET, ONE_VAR_GRAM, ONE_VAR_BASIS, 1, EPS, 1e-7)
    assert ok_uncharged
    ok, _min_eig, resid, why = _validate_gram_certificate(
        ONE_VAR_TARGET, ONE_VAR_GRAM, ONE_VAR_BASIS, 1, EPS, 1e-7,
        extra_residual_l1=dropped)
    assert not ok
    assert resid == pytest.approx(dropped, rel=1e-12)
    assert "gram_margin_exceeded" in why


def test_dropped_mass_is_charged_on_the_legacy_absolute_policy_too():
    """Without a margin (``eps_budget = 0``) acceptance needs ‖E‖₁ ≤ 1e-7, and
    the dropped mass counts towards that too."""
    ok, _min_eig, _resid, _why = _validate_gram_certificate(
        ONE_VAR_TARGET, ONE_VAR_GRAM, ONE_VAR_BASIS, 1, 0.0, 1e-7,
        extra_residual_l1=1e-9)
    assert ok
    ok, _min_eig, resid, why = _validate_gram_certificate(
        ONE_VAR_TARGET, ONE_VAR_GRAM, ONE_VAR_BASIS, 1, 0.0, 1e-7,
        extra_residual_l1=1e-5)
    assert not ok
    assert "gram_residual" in why
    assert resid == pytest.approx(1e-5, rel=1e-12)


# --------------------------------------------------------------------------
# 2. _numeric_poly_is_sos actually charges what it truncates
# --------------------------------------------------------------------------

class _StubOutcome:
    def __init__(self, gram: Optional[np.ndarray], status: str = "stub"):
        self.gram = gram
        self.status = status


def _stub_mosek_direct(monkeypatch, gram: np.ndarray) -> Dict[str, Any]:
    """Replace the MOSEK Task-API rung by a fixed Gram proposal.

    ``_numeric_poly_is_sos`` imports both names from ``.mosek_direct`` at CALL
    time, so patching the module attributes is enough. Nothing about the
    acceptance path (``_screen`` → ``_validate_gram_certificate``) is stubbed:
    only the proposal is. Records the target the solver was handed.

    Callers below pass ``strict_solver=True`` so the CVXPY rungs never run: the
    stubbed proposal is then the only Gram the acceptance path ever sees.
    """
    from gf_framework.verification import mosek_direct

    seen: Dict[str, Any] = {}

    def _fake(target, gram_map, n_basis):
        seen["target"] = dict(target)
        seen["n_basis"] = n_basis
        return _StubOutcome(np.asarray(gram, dtype=float))

    monkeypatch.setattr(mosek_direct, "MOSEK_AVAILABLE", True, raising=False)
    monkeypatch.setattr(mosek_direct, "solve_poly_sos_direct", _fake,
                        raising=False)
    return seen


#: 2 variables, degree 2. Basis is the homogeneous half-basis {(1,0), (0,1)}
#: in whichever order, and ``I₂`` reproduces ``y₁² + y₂²`` under either order.
TWO_VAR_GRAM = np.eye(2)
#: Below ``SOLVER_COEFF_FLOOR`` (so it is truncated) but far above the ε
#: budgets used below.
TINY = 5e-16


def test_truncated_coefficient_enters_the_residual(monkeypatch):
    """A coefficient under the solver floor is kept out of the equality rows
    but shows up, in full, in the charged residual."""
    seen = _stub_mosek_direct(monkeypatch, TWO_VAR_GRAM)
    target = {(2, 0): 1.0, (0, 2): 1.0, (1, 1): TINY}
    assert abs(TINY) <= SOLVER_COEFF_FLOOR

    ok, gram, used, attempts, diag = _numeric_poly_is_sos(
        target, n_vars=2, y_degree=2, eps_budget=1e-15,
        use_mosek_direct=True, strict_solver=True)

    # accepted, because 5e-16 < 1e-15 — but only because it was charged
    assert ok, attempts
    assert used == "MOSEK_DIRECT"
    assert gram is not None
    # the solver never saw the tiny coefficient ...
    assert (1, 1) not in seen["target"]
    # ... but the validator did
    assert diag["residual_l1"] == pytest.approx(TINY, rel=1e-9)


def test_truncated_coefficient_above_the_budget_is_refused(monkeypatch):
    """The regression the review asked for: a coefficient below 1e-15 but
    ABOVE the applicable budget must not be accepted as SOS.

    Pre-fix the same call returned ``True`` with residual 0, because the
    coefficient had been deleted before validation.
    """
    _stub_mosek_direct(monkeypatch, TWO_VAR_GRAM)
    target = {(2, 0): 1.0, (0, 2): 1.0, (1, 1): TINY}

    ok, gram, used, attempts, diag = _numeric_poly_is_sos(
        target, n_vars=2, y_degree=2, eps_budget=4e-16,
        use_mosek_direct=True, strict_solver=True)

    assert not ok
    assert gram is None
    assert used is None
    assert diag is not None
    assert diag["residual_l1"] == pytest.approx(TINY, rel=1e-9)
    assert any("gram_margin_exceeded" in a for a in attempts), attempts


def test_caller_supplied_dropped_mass_is_forwarded(monkeypatch):
    """``extra_residual_l1`` lets a caller that truncated its own assembly
    (``verify_gf_sos_full_with_dynamics``' cleared-W target) charge that mass
    here rather than lose it."""
    _stub_mosek_direct(monkeypatch, TWO_VAR_GRAM)
    target = {(2, 0): 1.0, (0, 2): 1.0}

    ok, _gram, _used, _attempts, diag = _numeric_poly_is_sos(
        target, n_vars=2, y_degree=2, eps_budget=1e-9,
        use_mosek_direct=True, strict_solver=True, extra_residual_l1=2e-9)
    assert not ok
    assert diag["residual_l1"] == pytest.approx(2e-9, rel=1e-9)


# --------------------------------------------------------------------------
# 3. the three cases that used to collapse onto "trivial_zero"
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target", [
    {},
    {(2, 0): 0.0},
    {(2, 0): 0.0, (1, 1): 0.0, (0, 2): 0.0},
])
def test_genuinely_zero_polynomial_keeps_trivial_zero(target):
    """No coefficients at all (or only EXACT zeros) really is the zero
    polynomial: nothing was truncated, so nothing is unaccounted for."""
    ok, gram, used, attempts, diag = _numeric_poly_is_sos(
        target, n_vars=2, y_degree=2, eps_budget=0.0)
    assert ok
    assert used == "trivial_zero"
    assert gram is None and diag is None and attempts == []


@pytest.mark.parametrize("eps", [1.0, 1e-6])
@pytest.mark.parametrize("target", [
    {},
    {(2, 0): 0.0},
    {(2, 0): 0.0, (1, 1): 0.0, (0, 2): 0.0},
])
def test_zero_margin_subtracted_target_is_sos_with_the_zero_gram(target, eps):
    """Follow-up review finding A: ``F = ε·Σᵢ y_iᵈ`` has the requested strict
    margin and the margin-subtracted target ``P = F − ε·Σᵢ y_iᵈ`` is EXACTLY
    zero. Zero is SOS with ``Q = 0``: zero residual, zero PSD shift, charge 0.

    Pre-fix this returned ``empty_strict_target``, disagreeing with the exact
    and portable validators, which accept the same zero remainder.
    """
    ok, gram, used, attempts, diag = _numeric_poly_is_sos(
        target, n_vars=2, y_degree=2, eps_budget=eps)
    assert ok, attempts
    assert used == "zero_remainder"
    assert gram is not None
    # the zero Gram of the ordinary half-basis, so a Gram-capturing caller
    # (design bundle, exact phase) gets a well-formed (gram, basis) pair
    assert diag is not None
    basis = diag["basis"]
    assert len(basis) == gram.shape[0] == gram.shape[1]
    assert all(sum(b) == 1 for b in basis)  # degree d/2 = 1, nonnegative
    assert np.all(gram == 0.0)
    assert diag["min_eig"] == pytest.approx(0.0, abs=0.0)
    assert diag["residual_l1"] == pytest.approx(0.0, abs=0.0)


def test_zero_target_still_charges_caller_dropped_mass():
    """The zero remainder is free, the caller's truncation is not: mass inside
    the budget passes, mass above it is refused with the typed reason."""
    ok, gram, used, _attempts, diag = _numeric_poly_is_sos(
        {(2, 0): 0.0}, n_vars=2, y_degree=2, eps_budget=EPS,
        extra_residual_l1=0.25 * EPS)
    assert ok
    assert used == "zero_remainder"
    assert gram is not None
    assert diag["residual_l1"] == pytest.approx(0.25 * EPS, rel=1e-12)

    ok, gram, used, attempts, diag = _numeric_poly_is_sos(
        {(2, 0): 0.0}, n_vars=2, y_degree=2, eps_budget=EPS,
        extra_residual_l1=1.5 * EPS)
    assert not ok
    assert gram is None
    assert used is None
    assert len(attempts) == 1
    assert attempts[0].startswith("target_below_solver_floor")
    assert diag["residual_l1"] == pytest.approx(1.5 * EPS, rel=1e-12)


def test_zero_target_without_an_even_degree_is_refused_under_a_margin():
    """The ε·Σᵢ y_iᵈ bookkeeping is defined for a positive EVEN d only. Zero is
    still SOS, but no margin can be stated against an odd degree, so the
    branch refuses instead of reporting a certified margin it cannot support.
    """
    ok, gram, used, attempts, _diag = _numeric_poly_is_sos(
        {}, n_vars=2, y_degree=3, eps_budget=EPS)
    assert not ok
    assert gram is None and used is None
    assert attempts == ["zero_target_degree_not_positive_even(y_degree=3)"]


@pytest.mark.parametrize("eps_budget", [0.0, 1e-18, EPS])
def test_target_empty_only_after_truncation_is_never_sos(eps_budget):
    """The second half of the defect: a target that is non-empty before the
    floor and empty after it is NOT the zero polynomial and must not be
    reported SOS."""
    target = {(2, 0): 1e-16, (0, 2): -3e-16}
    ok, gram, used, attempts, _diag = _numeric_poly_is_sos(
        target, n_vars=2, y_degree=2, eps_budget=eps_budget)
    assert not ok
    assert used != "trivial_zero"
    assert used is None
    assert gram is None
    assert len(attempts) == 1
    assert attempts[0].startswith("target_below_solver_floor")


def test_the_solver_floor_is_a_named_encoding_floor_not_a_tolerance():
    """Guard against the floor drifting back into an inline literal."""
    assert SOLVER_COEFF_FLOOR == 1e-15
    src = joint_sos._numeric_poly_is_sos.__doc__ or ""
    assert "SOLVER_COEFF_FLOOR" in src
    assert "CHARGED" in src or "charged" in src
