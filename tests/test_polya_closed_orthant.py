"""Closed-orthant Pólya positivity and weighted-sphere V re-verification.

Regression from the 2026-09-05 external scientific review, findings 1 and 3
(``review/scientific_2026-09-05/REVIEW.md``, reproducer
``review/scientific_2026-09-05/reproduce_polya_gap.py``).

Finding 1 (critical). ``polya.check_all_coefficients_positive`` iterated only
over the monomials PRESENT in the sparse polynomial, so a form positive on the
OPEN orthant but vanishing on a face (a missing pure power) was accepted as
positive definite on the CLOSED orthant. The public gate
``certify_discovered_gf`` then reported a complete stability proof
(``surface_certified=True``, ``surface_scope="lemma5_filippov"``) for an
UNSTABLE system.

Finding 3 (major). The weighted-sphere branch may return a redesigned V from the
joint SOS optimizer. That V replaced ``V_gf`` in the payload while
``V_verified=True`` still referred to the original stage-1 candidate. The
invariant pinned here: the V in the returned Certificate is the V whose
positivity was established.
"""
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification.failure_typing import FailureClass
from gf_framework.verification.polya import (
    STATUS_MISSING_PURE_POWER,
    STATUS_NONPOSITIVE_COEFFICIENT,
    STATUS_NOT_POLYNOMIAL,
    STATUS_OK,
    STATUS_ZERO_POLYNOMIAL,
    analyze_closed_orthant_positivity,
    check_all_coefficients_positive,
    verify_gf_polya,
)
from gf_framework.verification.verifier import certify_discovered_gf, verify_lyapunov

try:  # CVXPY gates only the SOS fallback, not the Pólya rule itself.
    from gf_framework.verification.joint_sos import CVXPY_AVAILABLE
except Exception:  # pragma: no cover - defensive
    CVXPY_AVAILABLE = False


Y = sp.symbols("y1 y2", positive=True)


# ---------------------------------------------------------------------------
# Finding 1 — the pure-power rule itself
# ---------------------------------------------------------------------------
class TestPurePowerRule:
    """``all present coefficients > 0`` is NOT positive definiteness."""

    def test_missing_pure_power_is_rejected(self):
        """y1*y2 is positive on the OPEN orthant and zero on both axes."""
        ok, status, failure = analyze_closed_orthant_positivity(Y[0] * Y[1], list(Y))
        assert not ok
        assert status == STATUS_MISSING_PURE_POWER
        assert failure is not None
        ok_compat, _ = check_all_coefficients_positive(Y[0] * Y[1], list(Y))
        assert not ok_compat, "the historical entry point must inherit the rule"

    def test_full_pure_power_coverage_passes(self):
        poly = Y[0] ** 2 + Y[0] * Y[1] + Y[1] ** 2
        ok, status, failure = analyze_closed_orthant_positivity(poly, list(Y))
        assert ok and status == STATUS_OK and failure is None

    def test_non_homogeneous_form_with_coverage_passes(self):
        """μ is the per-variable LCM, so associated forms need not be homogeneous.

        CTA at m=8, r=(3,2,1) produces exactly this shape. Requiring standard
        homogeneity here would reject flagship certificates; the closed-orthant
        rule does not need it.
        """
        poly = Y[0] ** 2 + Y[1] ** 4
        ok, status, _ = analyze_closed_orthant_positivity(poly, list(Y))
        assert ok and status == STATUS_OK

    def test_positive_constant_covers_every_variable(self):
        """FOSMC: W = -dV/dt = k is a positive constant and stays admissible."""
        ok, status, _ = analyze_closed_orthant_positivity(sp.Float(1.5), list(Y))
        assert ok and status == STATUS_OK

    def test_zero_polynomial_is_rejected(self):
        ok, status, _ = analyze_closed_orthant_positivity(sp.Integer(0), list(Y))
        assert not ok and status == STATUS_ZERO_POLYNOMIAL

    def test_negative_coefficient_is_rejected(self):
        poly = Y[0] ** 2 - 3 * Y[1] ** 2
        ok, status, failure = analyze_closed_orthant_positivity(poly, list(Y))
        assert not ok and status == STATUS_NONPOSITIVE_COEFFICIENT
        assert failure[0] == (0, 2) and failure[1] < 0

    def test_laurent_monomial_is_not_a_polynomial(self):
        poly = Y[0] ** -2 + Y[1] ** 2
        ok, status, failure = analyze_closed_orthant_positivity(poly, list(Y))
        assert not ok and status == STATUS_NOT_POLYNOMIAL
        assert failure[0] == (-1, -1)


class TestPolyaGFLevel:
    """``verify_gf_polya`` must not report verified for degenerate forms."""

    def test_face_vanishing_gf_is_rejected_with_axis_witness(self):
        # V = x2^2 on R^2: positive off the x1 axis, identically zero on it.
        V = GeneralizedForm(
            degree_m=2, weights_r=(1, 1),
            terms=[GFTerm(1.0, (Fraction(0), Fraction(2)), ("abs", "abs"))],
        )
        result = verify_gf_polya(V, max_p=6)
        assert not result.verified
        assert "pure power" in result.message
        ce = result.counterexample
        assert ce is not None
        value = np.atleast_1d(V.evaluate(ce.reshape(1, -1)))[0]
        assert value <= 1e-12, "the witness must actually annihilate V"

    def test_empty_gf_is_rejected(self):
        V = GeneralizedForm(degree_m=2, weights_r=(1, 1), terms=[])
        result = verify_gf_polya(V, max_p=3)
        assert not result.verified
        assert "no terms" in result.message


# ---------------------------------------------------------------------------
# Finding 1 — through the PUBLIC gate, exactly like the reproducer
# ---------------------------------------------------------------------------
def _unstable_linear_case():
    """x1' = x1, x2' = -x2, V = x2^2, r = (1,1), m = 2.

    The origin is the unique equilibrium and is unstable: the solution from
    (1, 0) is (exp(t), 0), along which V and W both vanish.
    """
    r = (1, 1)
    V = GeneralizedForm(2, r, [GFTerm(1, (Fraction(0), Fraction(2)), ("abs", "abs"))])
    f = [
        GeneralizedForm(1, r, [GFTerm(1, (Fraction(1), Fraction(0)), ("signed", "abs"))]),
        GeneralizedForm(1, r, [GFTerm(-1, (Fraction(0), Fraction(1)), ("abs", "signed"))]),
    ]
    return V, r, 2, f


def _non_attractive_case():
    """x1' = -|x2|sign(x1), x2' = -sign(x2), V = x1^2 + x2^4, r = (2,1), m = 4.

    W = 2|x1||x2| + 4|x2|^3 vanishes on the whole x1 axis, and every (c, 0) is
    stationary.
    """
    r = (2, 1)
    V = GeneralizedForm(4, r, [
        GFTerm(1, (Fraction(2), Fraction(0)), ("abs", "abs")),
        GFTerm(1, (Fraction(0), Fraction(4)), ("abs", "abs")),
    ])
    f = [
        GeneralizedForm(1, r, [GFTerm(-1, (Fraction(0), Fraction(1)), ("signed", "abs"))]),
        GeneralizedForm(0, r, [GFTerm(-1, (Fraction(0), Fraction(0)), ("abs", "signed"))]),
    ]
    return V, r, 4, f


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not installed")
class TestPublicGateRejectsReviewExamples:

    def test_unstable_linear_system_is_not_certified(self):
        V, r, m, f = _unstable_linear_case()
        cert = certify_discovered_gf(V, r, m, f, solver="CLARABEL", strict_solver=True)

        assert not cert.V_verified, "V = x2^2 vanishes on the x1 axis"
        assert not cert.W_verified
        assert not cert.fully_verified
        assert not cert.surface_certified
        assert cert.surface_scope == "open_orthant_only"
        # Typed refusal, not an exception and not a bare "failed".
        assert cert.failure_class not in ("", FailureClass.NONE)

    def test_non_attractive_inclusion_is_not_certified(self):
        V, r, m, f = _non_attractive_case()
        cert = certify_discovered_gf(V, r, m, f, solver="CLARABEL", strict_solver=True)

        # V = x1^2 + x2^4 IS positive definite, so V_verified may legitimately be
        # True. What must never happen again is the complete stability verdict.
        assert not cert.W_verified, "W vanishes on the entire x1 axis"
        assert not cert.fully_verified
        assert not cert.surface_certified
        assert cert.surface_scope == "open_orthant_only"
        assert cert.failure_class not in ("", FailureClass.NONE)

    def test_w_of_non_attractive_case_fails_polya_on_the_axis(self):
        """Pin the mechanism, independent of any solver."""
        from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf

        V, _r, _m, f = _non_attractive_case()
        dVdt = compute_lie_derivative_gf(V, f)
        W = GeneralizedForm(
            degree_m=dVdt.degree_m, weights_r=dVdt.weights_r,
            terms=[GFTerm(-t.coefficient, t.exponents, t.term_types) for t in dVdt.terms],
        )
        result = verify_gf_polya(W, max_p=6)
        assert not result.verified
        assert "pure power" in result.message


# ---------------------------------------------------------------------------
# Finding 3 — the payload V is the V whose positivity was established
# ---------------------------------------------------------------------------
def _redesign_fixture():
    """One-dimensional stand-in that reaches the joint-SOS redesign branch."""
    from gf_framework.interfaces import SelectionResult, SystemSpec

    seed_v = GeneralizedForm(
        degree_m=1, weights_r=(1,),
        terms=[GFTerm(coefficient=1.0, exponents=(Fraction(1),), term_types=("abs",))],
    )
    redesigned_v = GeneralizedForm(
        degree_m=1, weights_r=(1,),
        terms=[GFTerm(coefficient=2.0, exponents=(Fraction(1),), term_types=("abs",))],
    )
    selection = SelectionResult(
        active_terms=[], coefficients={0: 1.0}, V_gf=seed_v,
        numerical_valid=True, max_V_violation=-0.1, max_W_violation=-0.1,
    )
    dynamics = [GeneralizedForm(
        degree_m=0, weights_r=(1,),
        terms=[GFTerm(coefficient=-1.0, exponents=(Fraction(0),), term_types=("signed",))],
    )]
    spec = SystemSpec(weights_r=(1,), degree_m=1, dynamics_gf=dynamics)
    return seed_v, redesigned_v, selection, spec


class TestRedesignedVIsReVerified:
    """The weighted-sphere path is monkeypatched.

    Reaching it naturally needs W with negative lifted exponents AND a solver
    that returns a redesigned V, which cannot be forced deterministically in a
    unit test; the monkeypatch exercises exactly the verifier branch the review
    flagged.
    """

    def test_failed_re_verification_refuses_the_certificate(self, monkeypatch):
        import gf_framework.verification.verifier as verifier_mod

        seed_v, redesigned_v, selection, spec = _redesign_fixture()
        seen = []

        def fake_positivity(gf, config, verbose):
            seen.append(gf)
            # Stage 1 accepts the seed; the redesigned candidate is NOT PD.
            ok = gf is not redesigned_v
            return (ok, "mock_v" if ok else "failed", None,
                    verifier_mod._default_solver_metadata(config))

        monkeypatch.setattr(verifier_mod, "_verify_positivity", fake_positivity)
        monkeypatch.setattr(
            verifier_mod, "_verify_w_positivity",
            lambda W_gf, V_gf, system_spec, config, w_analysis, verbose: (
                True, "sos_weighted_sphere", None, 0, redesigned_v,
                verifier_mod._default_solver_metadata(config),
            ),
        )

        cert = verify_lyapunov(selection, spec, verbose=False)

        assert redesigned_v in seen, "the redesigned V must be re-gated"
        assert not cert.fully_verified
        assert not cert.W_verified
        assert not cert.surface_certified
        # The payload keeps the V that was actually proven positive definite.
        assert cert.V_gf is seed_v
        assert cert.W_failure_class == FailureClass.POLICY_REJECTED
        redesign = [r for r in (cert.stage_reports or [])
                    if r["stage"] == "V_redesigned"]
        assert redesign and redesign[0]["verified"] is False

    def test_passed_re_verification_keeps_the_redesigned_v(self, monkeypatch):
        import gf_framework.verification.verifier as verifier_mod

        seed_v, redesigned_v, selection, spec = _redesign_fixture()
        seen = []

        def fake_positivity(gf, config, verbose):
            seen.append(gf)
            return True, "mock_v", None, verifier_mod._default_solver_metadata(config)

        monkeypatch.setattr(verifier_mod, "_verify_positivity", fake_positivity)
        monkeypatch.setattr(
            verifier_mod, "_verify_w_positivity",
            lambda W_gf, V_gf, system_spec, config, w_analysis, verbose: (
                True, "sos_weighted_sphere", None, 0, redesigned_v,
                verifier_mod._default_solver_metadata(config),
            ),
        )

        cert = verify_lyapunov(selection, spec, verbose=False)

        assert cert.fully_verified
        assert cert.V_gf is redesigned_v
        assert seen[-1] is redesigned_v, "payload V must be the re-verified V"
        redesign = [r for r in (cert.stage_reports or [])
                    if r["stage"] == "V_redesigned"]
        assert redesign and redesign[0]["verified"] is True
        assert redesign[0]["v_redesigned"] is True
