"""
Tests for GF Algebra Module.

Validates Generalized Form operations:
- Term creation, multiplication, evaluation
- GeneralizedForm construction, evaluation, serialization
- Lie derivative computation
- Term enumeration
- Associated forms
"""

import pytest
import numpy as np
import sympy as sp
from fractions import Fraction

from gf_framework.algebra.core import (
    GFTerm,
    GeneralizedForm,
    create_diagonal_gf,
    sgnpow_numpy,
    lcm_list,
)
from gf_framework.algebra.enumerate import (
    TermSpec,
    enumerate_diagonal_terms,
    enumerate_cross_terms,
    enumerate_all_terms,
    validate_degree_compatibility,
    suggest_compatible_degrees,
)
from gf_framework.algebra.associated import (
    gf_to_associated_forms,
    get_hyperoctant_signs,
    inverse_transform_point,
    forward_transform_point,
)
from gf_framework.algebra.lie_derivative import (
    compute_lie_derivative_gf,
    negate_gf,
    multiply_gf_terms,
    gf_partial_derivative_term,
    collect_terms,
)
from gf_framework.systems.fosmc import (
    get_fosmc_dynamics_gf,
    get_fosmc_known_lyapunov,
    FOSMC_KAPPA,
)
from gf_framework.systems.sta import (
    get_sta_dynamics_gf,
    get_sta_known_lyapunov,
    STA_KAPPA,
)
from gf_framework.systems.pidsmc import (
    get_pidsmc_dynamics_gf,
    get_pidsmc_known_lyapunov_structure,
    PIDSMC_KAPPA,
)


# =========================================================================
# GFTerm
# =========================================================================

class TestGFTerm:
    """Test GFTerm dataclass."""

    def test_diagonal_term_1d(self):
        """Test 1D diagonal term |s|."""
        term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1),),
            term_types=('abs',),
        )
        assert term.n_vars == 1

        x = np.array([[1.0], [-2.0], [0.5]])
        vals = term.evaluate(x)
        expected = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(vals, expected)

    def test_diagonal_term_2d(self):
        """Test 2D diagonal terms for STA (r=(2,1), m=2)."""
        term1 = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1), Fraction(0)),
            term_types=('abs', 'abs'),
        )
        term2 = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(2)),
            term_types=('abs', 'abs'),
        )

        x = np.array([[1.0, 2.0], [-1.0, -3.0]])
        np.testing.assert_allclose(term1.evaluate(x), [1.0, 1.0])
        np.testing.assert_allclose(term2.evaluate(x), [4.0, 9.0])

    def test_signed_power_term(self):
        """Test signed power term ceil(x)^(1/2)."""
        term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1, 2), Fraction(0)),
            term_types=('signed', 'abs'),
        )

        x = np.array([[4.0, 0.0], [-4.0, 0.0]])
        vals = term.evaluate(x)
        expected = np.array([2.0, -2.0])
        np.testing.assert_allclose(vals, expected)

    def test_weighted_degree(self):
        """Test weighted degree computation."""
        term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1), Fraction(0)),
            term_types=('abs', 'abs'),
        )
        assert term.weighted_degree((2, 1)) == 2

        cross_term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1, 2), Fraction(1)),
            term_types=('signed', 'abs'),
        )
        assert cross_term.weighted_degree((2, 1)) == 2


# =========================================================================
# GeneralizedForm
# =========================================================================

class TestGeneralizedForm:
    """Test GeneralizedForm dataclass."""

    def test_fosmc_lyapunov(self, fosmc_v_gf):
        """Test FOSMC Lyapunov V(s) = |s|."""
        gf = fosmc_v_gf
        assert gf.n_vars == 1
        assert gf.n_terms == 1

        x = np.array([[0.0], [1.0], [-2.0], [3.5]])
        vals = gf.evaluate(x)
        np.testing.assert_allclose(vals, [0.0, 1.0, 2.0, 3.5])

    def test_sta_diagonal_lyapunov(self):
        """Test STA diagonal Lyapunov V = a1|s| + a2|v|^2."""
        gf = create_diagonal_gf(
            degree_m=2, weights_r=(2, 1), coefficients=[1.0, 0.5],
        )
        assert gf.n_vars == 2
        assert gf.n_terms == 2

        # V(1, 2) = 1*|1| + 0.5*|2|^2 = 1 + 2 = 3
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(gf.evaluate(x), 3.0)

    def test_homogeneity_validation(self):
        """Test that invalid terms are rejected."""
        valid_term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1), Fraction(0)),
            term_types=('abs', 'abs'),
        )
        gf = GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=[valid_term])
        assert gf.n_terms == 1

        invalid_term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(2), Fraction(0)),
            term_types=('abs', 'abs'),
        )
        with pytest.raises(AssertionError):
            GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=[invalid_term])

    def test_mu_exponents(self):
        """Test μ exponents for y-transform."""
        gf_fosmc = create_diagonal_gf(1, (1,))
        assert gf_fosmc.mu_exponents == (1,)

        gf_sta = create_diagonal_gf(2, (2, 1))
        assert gf_sta.mu_exponents == (1, 1)

        gf_pidsmc = create_diagonal_gf(7, (3, 2, 1))
        assert gf_pidsmc.mu_exponents == (3, 2, 1)

    def test_sympy_conversion(self):
        """Test conversion to SymPy expression."""
        gf = create_diagonal_gf(1, (1,), [2.0])
        expr = gf.to_sympy()
        assert expr.has(sp.Abs)


# =========================================================================
# Enumeration
# =========================================================================

class TestEnumeration:
    """Test term enumeration."""

    def test_fosmc_terms(self):
        """FOSMC should have exactly 1 diagonal term."""
        diag, cross = enumerate_all_terms(1, (1,))
        assert len(diag) == 1
        assert len(cross) == 0

    def test_sta_diagonal_terms(self):
        """STA should have 2 diagonal terms."""
        diag = enumerate_diagonal_terms(2, (2, 1))
        assert len(diag) == 2

        exp_set = {d.exponents for d in diag}
        assert (Fraction(1), Fraction(0)) in exp_set
        assert (Fraction(0), Fraction(2)) in exp_set

    def test_pidsmc_diagonal_terms(self):
        """PID-SMC should have 3 diagonal terms."""
        diag = enumerate_diagonal_terms(7, (3, 2, 1))
        assert len(diag) == 3

        exp_set = {d.exponents for d in diag}
        assert (Fraction(7, 3), Fraction(0), Fraction(0)) in exp_set
        assert (Fraction(0), Fraction(7, 2), Fraction(0)) in exp_set
        assert (Fraction(0), Fraction(0), Fraction(7)) in exp_set

    def test_degree_compatibility(self):
        """Test degree compatibility checking."""
        assert validate_degree_compatibility(2, (2, 1))
        assert validate_degree_compatibility(6, (3, 2, 1))

    def test_suggest_degrees(self):
        """Test degree suggestion for different systems."""
        sta_degs = suggest_compatible_degrees((2, 1), min_degree=2, max_degree=10)
        assert 2 in sta_degs

        pidsmc_degs = suggest_compatible_degrees((3, 2, 1), min_degree=4, max_degree=10)
        assert 6 in pidsmc_degs or 7 in pidsmc_degs


# =========================================================================
# Associated Forms
# =========================================================================

class TestAssociatedForms:
    """Test associated forms transformation."""

    def test_hyperoctant_signs(self):
        """Test hyperoctant sign generation."""
        signs_1d = get_hyperoctant_signs(1)
        assert len(signs_1d) == 2
        assert (1,) in signs_1d
        assert (-1,) in signs_1d

        signs_2d = get_hyperoctant_signs(2)
        assert len(signs_2d) == 4

        signs_3d = get_hyperoctant_signs(3)
        assert len(signs_3d) == 8

    def test_inverse_transform(self):
        """Test y-space to x-space transformation."""
        y = np.array([2.0, 3.0])
        signs = (1, -1)
        mu = (1, 2)

        x = inverse_transform_point(y, signs, mu)
        expected = np.array([2.0, -9.0])
        np.testing.assert_allclose(x, expected)

    def test_forward_transform(self):
        """Test x-space to y-space transformation."""
        x = np.array([2.0, -9.0])
        mu = (1, 2)

        y, signs = forward_transform_point(x, mu)
        np.testing.assert_allclose(y, [2.0, 3.0])
        assert signs == (1, -1)

    def test_roundtrip_transform(self):
        """Test that forward + inverse is identity."""
        mu = (2, 3, 6)
        x_orig = np.array([1.5, -2.0, 0.5])

        y, signs = forward_transform_point(x_orig, mu)
        x_back = inverse_transform_point(y, signs, mu)

        np.testing.assert_allclose(x_back, x_orig, rtol=1e-10)

    def test_associated_forms_fosmc(self):
        """Test associated forms for FOSMC."""
        gf = create_diagonal_gf(1, (1,), [1.0])
        forms = gf_to_associated_forms(gf)

        assert len(forms) == 2

        y1 = sp.Symbol('y1', positive=True)
        for signs, poly in forms.items():
            assert poly.has(y1)


# =========================================================================
# GFTerm Operations (Lie Derivative helpers)
# =========================================================================

class TestGFTermOperations:
    """Test low-level GF term operations."""

    def test_multiply_abs_terms(self):
        """Test multiplication of absolute value terms."""
        t1 = GFTerm(coefficient=2.0, exponents=(Fraction(2),), term_types=('abs',))
        t2 = GFTerm(coefficient=3.0, exponents=(Fraction(3),), term_types=('abs',))
        result = multiply_gf_terms(t1, t2)

        assert result.coefficient == 6.0
        assert result.exponents == (Fraction(5),)
        assert result.term_types == ('abs',)

    def test_multiply_signed_terms(self):
        """⌈x⌋^2 * ⌈x⌋^3 = |x|^5 (sign(x)^2 = 1)."""
        t1 = GFTerm(coefficient=2.0, exponents=(Fraction(2),), term_types=('signed',))
        t2 = GFTerm(coefficient=3.0, exponents=(Fraction(3),), term_types=('signed',))
        result = multiply_gf_terms(t1, t2)

        assert result.coefficient == 6.0
        assert result.exponents == (Fraction(5),)
        assert result.term_types == ('abs',)

    def test_multiply_mixed_terms(self):
        """|x|^2 * ⌈x⌋^3 = ⌈x⌋^5."""
        t1 = GFTerm(coefficient=2.0, exponents=(Fraction(2),), term_types=('abs',))
        t2 = GFTerm(coefficient=3.0, exponents=(Fraction(3),), term_types=('signed',))
        result = multiply_gf_terms(t1, t2)

        assert result.coefficient == 6.0
        assert result.exponents == (Fraction(5),)
        assert result.term_types == ('signed',)

    def test_multiply_2d_terms(self):
        """Test multiplication of 2D terms."""
        t1 = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1, 2), Fraction(0)),
            term_types=('abs', 'abs'),
        )
        t2 = GFTerm(
            coefficient=2.0,
            exponents=(Fraction(1, 2), Fraction(1)),
            term_types=('signed', 'abs'),
        )
        result = multiply_gf_terms(t1, t2)

        assert result.coefficient == 2.0
        assert result.exponents == (Fraction(1), Fraction(1))
        assert result.term_types == ('signed', 'abs')

    def test_partial_derivative_abs(self):
        """d/dx |x|^3 = 3⌈x⌋^2."""
        term = GFTerm(coefficient=1.0, exponents=(Fraction(3),), term_types=('abs',))
        deriv = gf_partial_derivative_term(term, 0)

        assert deriv is not None
        assert deriv.coefficient == 3.0
        assert deriv.exponents == (Fraction(2),)
        assert deriv.term_types == ('signed',)

    def test_partial_derivative_signed(self):
        """d/dx ⌈x⌋^3 = 3|x|^2."""
        term = GFTerm(coefficient=1.0, exponents=(Fraction(3),), term_types=('signed',))
        deriv = gf_partial_derivative_term(term, 0)

        assert deriv is not None
        assert deriv.coefficient == 3.0
        assert deriv.exponents == (Fraction(2),)
        assert deriv.term_types == ('abs',)

    def test_partial_derivative_zero_exp(self):
        """Zero exponent gives zero derivative."""
        term = GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(2)),
            term_types=('abs', 'abs'),
        )
        deriv = gf_partial_derivative_term(term, 0)
        assert deriv is None

    def test_collect_terms(self):
        """Terms with same signature are collected."""
        terms = [
            GFTerm(coefficient=2.0, exponents=(Fraction(1),), term_types=('abs',)),
            GFTerm(coefficient=3.0, exponents=(Fraction(1),), term_types=('abs',)),
            GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=('abs',)),
        ]
        collected = collect_terms(terms)

        assert len(collected) == 2
        coefs = {t.exponents[0]: t.coefficient for t in collected}
        assert coefs[Fraction(1)] == 5.0
        assert coefs[Fraction(2)] == 1.0


# =========================================================================
# FOSMC Lie Derivative
# =========================================================================

class TestFOSMCLieDerivative:
    """Test Lie derivative for FOSMC (1D)."""

    def test_dynamics_gf_structure(self):
        """FOSMC dynamics are correctly represented as GF."""
        k = 1.0
        f = get_fosmc_dynamics_gf(k)

        assert len(f) == 1
        assert f[0].n_vars == 1
        assert f[0].degree_m == 0
        assert len(f[0].terms) == 1
        assert f[0].terms[0].coefficient == -k

    def test_known_lyapunov(self):
        """Known Lyapunov V = |s| is correctly represented."""
        V = get_fosmc_known_lyapunov()

        assert V.n_vars == 1
        assert V.degree_m == 1
        assert len(V.terms) == 1
        assert V.terms[0].exponents == (Fraction(1),)

    def test_lie_derivative_computation(self):
        """dV/dt = sign(s)·(-k·sign(s)) = -k (constant)."""
        k = 1.0
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)

        assert dVdt.degree_m == 0
        assert len(dVdt.terms) == 1

        test_points = np.array([[1.0], [-1.0], [0.5], [-2.0]])
        values = dVdt.evaluate(test_points)
        np.testing.assert_allclose(values, -k, rtol=1e-10)

    def test_w_is_positive(self):
        """W = -dV/dt = k > 0."""
        k = 1.5
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        test_points = np.array([[1.0], [-1.0], [0.5]])
        w_values = W.evaluate(test_points)

        assert np.all(w_values > 0)
        np.testing.assert_allclose(w_values, k, rtol=1e-10)


# =========================================================================
# STA Lie Derivative
# =========================================================================

class TestSTALieDerivative:
    """Test Lie derivative for STA (2D)."""

    def test_dynamics_gf_structure(self):
        """STA dynamics are correctly represented as GF."""
        k1, k2 = 1.5, 1.1
        f = get_sta_dynamics_gf(k1, k2)

        assert len(f) == 2
        assert f[0].degree_m == 1
        assert len(f[0].terms) == 2
        assert f[1].degree_m == 0
        assert len(f[1].terms) == 1

    def test_known_lyapunov(self):
        """Moreno-Osorio Lyapunov is correctly represented."""
        V = get_sta_known_lyapunov()

        assert V.n_vars == 2
        assert V.degree_m == 2
        assert len(V.terms) == 3

    def test_lie_derivative_degree(self):
        """Lie derivative has correct degree."""
        k1, k2 = 1.5, 1.1
        V = get_sta_known_lyapunov(k1, k2)
        f = get_sta_dynamics_gf(k1, k2)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)

        assert dVdt.n_vars == 2
        assert len(dVdt.terms) > 0

    def test_lie_derivative_numerical(self):
        """Lie derivative matches numerical computation."""
        k1, k2 = 1.5, 1.1
        V = get_sta_known_lyapunov(k1, k2)
        f = get_sta_dynamics_gf(k1, k2)

        dVdt_gf = compute_lie_derivative_gf(V, f, validate=False)

        test_points = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [2.0, -1.0],
        ])

        gf_values = dVdt_gf.evaluate(test_points)

        numerical_values = []
        for x in test_points:
            grad = V.gradient(x)
            f_eval = np.array([
                -k1 * np.sign(x[0]) * np.abs(x[0])**0.5 + x[1],
                -k2 * np.sign(x[0]),
            ])
            dVdt_num = np.dot(grad, f_eval)
            numerical_values.append(dVdt_num)

        numerical_values = np.array(numerical_values)
        np.testing.assert_allclose(gf_values, numerical_values, rtol=0.1, atol=0.1)


# =========================================================================
# PID-SMC Lie Derivative
# =========================================================================

class TestPIDSMCLieDerivative:
    """Test Lie derivative for PID-SMC (3D)."""

    def test_dynamics_gf_structure(self):
        """PID-SMC dynamics are correctly represented as GF."""
        f = get_pidsmc_dynamics_gf()

        assert len(f) == 3
        assert f[0].degree_m == 2
        assert len(f[0].terms) == 1
        assert f[1].degree_m == 1
        assert len(f[1].terms) == 3
        assert f[2].degree_m == 0
        assert len(f[2].terms) == 1

    def test_known_lyapunov_structure(self):
        """Known Lyapunov structure is correct."""
        V = get_pidsmc_known_lyapunov_structure()

        assert V.n_vars == 3
        assert V.degree_m == 7
        assert len(V.terms) == 9

    def test_lie_derivative_computes(self):
        """Lie derivative can be computed for PID-SMC."""
        V = get_pidsmc_known_lyapunov_structure()
        f = get_pidsmc_dynamics_gf()

        dVdt = compute_lie_derivative_gf(V, f, validate=False)

        assert dVdt.n_vars == 3
        assert len(dVdt.terms) > 0


# =========================================================================
# Negate GF
# =========================================================================

class TestNegateGF:
    """Test GF negation for W = -dV/dt."""

    def test_negate_single_term(self):
        """Negation of single-term GF."""
        gf = GeneralizedForm(
            degree_m=1,
            weights_r=(1,),
            terms=[GFTerm(coefficient=2.0, exponents=(Fraction(1),), term_types=('abs',))],
        )
        neg_gf = negate_gf(gf)
        assert neg_gf.terms[0].coefficient == -2.0

    def test_negate_multi_term(self):
        """Negation of multi-term GF."""
        gf = create_diagonal_gf(2, (2, 1), [1.0, 2.0])
        neg_gf = negate_gf(gf)

        assert neg_gf.terms[0].coefficient == -1.0
        assert neg_gf.terms[1].coefficient == -2.0


# =========================================================================
# Numerical Stability
# =========================================================================

class TestNumericalStability:
    """Test numerical stability of GF operations."""

    def test_near_origin(self):
        """Evaluation near origin."""
        gf = create_diagonal_gf(2, (2, 1), [1.0, 1.0])

        x_small = np.array([[1e-10, 1e-10], [1e-15, 1e-15]])
        vals = gf.evaluate(x_small)

        assert np.all(vals >= 0)
        assert np.all(np.isfinite(vals))

    def test_large_values(self):
        """Evaluation at large values."""
        gf = create_diagonal_gf(2, (2, 1), [1.0, 1.0])

        x_large = np.array([[1e5, 1e5], [1e10, 1e3]])
        vals = gf.evaluate(x_large)

        assert np.all(vals > 0)
        assert np.all(np.isfinite(vals))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
