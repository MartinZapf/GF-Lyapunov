"""Solver-free regressions for model fidelity and exact regularity boundaries."""
from fractions import Fraction
import importlib

import numpy as np
import pytest

from gf_framework.algebra.enumerate import classify_term_regularity
from gf_framework.discovery.universal_discovery import create_power_rate_spec
from gf_framework.systems.power_rate import PowerRateSystem, get_power_rate_dynamics_gf
from gf_framework.systems.sta import get_sta_known_lyapunov, moreno_osorio_lyapunov_matrix
from gf_framework.systems.twisting import (
    get_twisting_known_lyapunov, get_twisting_lyapunov_seed,
)


@pytest.mark.parametrize("k1,k2", [(Fraction(3, 2), Fraction(11, 10)),
                                   (Fraction(1, 10), Fraction(7, 3)),
                                   (Fraction(3), Fraction(1))])
def test_sta_known_matrix_has_exact_nominal_decay(k1, k2):
    # Exact Lyapunov identity checks the returned fixed matrix, not a reoptimized V.
    P = moreno_osorio_lyapunov_matrix(k1, k2)
    A = np.array([[-k1 / 2, Fraction(1, 2)], [-k2, Fraction(0)]], dtype=object)
    Q = -(A.T @ P + P @ A)
    expected = np.array([[k1 * (2 * k2 + k1**2), -k1**2],
                         [-k1**2, k1]], dtype=object)
    assert np.array_equal(Q, expected)
    for M in (P, Q):
        assert M[0, 0] > 0
        assert M[0, 0] * M[1, 1] - M[0, 1]**2 > 0
    V = get_sta_known_lyapunov(k1, k2)
    assert V.terms[1].coefficient == 2 * P[0, 1] == -2 * k1
    # The old positive cross term had Vdot > 0 at (1,10).
    xi = np.array([Fraction(1), Fraction(10)], dtype=object)
    assert -(xi @ Q @ xi) < 0


def test_twisting_old_name_is_an_explicit_unverified_seed():
    seed = get_twisting_lyapunov_seed()
    with pytest.warns(DeprecationWarning, match="unverified seed"):
        legacy = get_twisting_known_lyapunov()
    assert legacy == seed
    # Preserve the seed, but never call this fixed polynomial a decay certificate.
    assert [term.coefficient for term in seed.terms] == [6.25, 1.5, 1.0]
    assert 13.75 * 10 + 21.5 * (-3.5) > 0


@pytest.mark.parametrize("alpha,exact", [(0, Fraction(0)), (0.5, Fraction(1, 2)),
                                        (0.99, Fraction(99, 100)),
                                        (Fraction(1, 3), Fraction(1, 3)),
                                        (1, Fraction(1)), (1.2, Fraction(6, 5))])
def test_power_rate_keeps_one_exponent_in_gf_numeric_and_spec(alpha, exact):
    system = PowerRateSystem(alpha=alpha)
    gf, = get_power_rate_dynamics_gf(alpha=alpha)
    spec = create_power_rate_spec(alpha=alpha)
    assert system.alpha_exact == gf.degree_m == gf.terms[0].exponents[0] == exact
    assert spec.dynamics_gf[0].degree_m == exact
    assert system.kappa == spec.kappa == float(exact - 1)
    points = np.array([[-0.25], [0.0], [1e-20], [0.25]])
    np.testing.assert_allclose(system.dynamics_nominal(points).ravel(),
                               gf.evaluate(points), rtol=1e-14, atol=0)


def test_power_rate_near_one_keeps_negative_kappa_before_float_conversion():
    alpha = Fraction(10**20 - 1, 10**20)
    assert PowerRateSystem(alpha=alpha).kappa < 0
    assert create_power_rate_spec(alpha=alpha).kappa < 0
    assert get_power_rate_dynamics_gf(alpha=alpha)[0].degree_m == alpha


@pytest.mark.parametrize("factory", [PowerRateSystem, get_power_rate_dynamics_gf,
                                     create_power_rate_spec])
def test_power_rate_rejects_negative_exponent_consistently(factory):
    with pytest.raises(ValueError, match="nonnegative"):
        factory(alpha=Fraction(-1, 3))


@pytest.mark.parametrize("module", ["continuous_twisting", "dsta", "example1", "example3",
                                     "fc_cta", "levant3d", "levant4d", "levant5d",
                                     "pidsmc", "pidsmc4d", "power_rate", "sta", "twisting"])
@pytest.mark.parametrize("p", [0, 1 / 3, 0.5, 2])
def test_signed_power_preserves_values_below_old_floor(module, p):
    helper = importlib.import_module("gf_framework.systems." + module).sgnpow
    x = np.array([-1e-20, 0, 1e-20])
    expected = np.sign(x) if p == 0 else np.sign(x) * np.abs(x)**p
    np.testing.assert_array_equal(helper(x, p), expected)


def test_continuous_twisting_numpy_alias_has_no_additive_floor():
    from gf_framework.systems.continuous_twisting import sgnpow_numpy
    np.testing.assert_array_equal(sgnpow_numpy(np.array([-1e-20, 0, 1e-20]), 0.5),
                                  np.array([-1e-10, 0, 1e-10]))


@pytest.mark.parametrize("module,class_name,n", [("levant3d", "Levant3DSystem", 3),
                                                 ("levant4d", "Levant4DSystem", 4)])
def test_levant_one_sided_boundary_preserves_only_the_relay(module, class_name, n):
    cls = getattr(importlib.import_module("gf_framework.systems." + module), class_name)
    system = cls()
    plus = system.dynamics_plus(np.zeros(n))
    minus = system.dynamics_minus(np.zeros(n))
    np.testing.assert_array_equal(plus[:-1], np.zeros(n - 1))
    np.testing.assert_array_equal(minus[:-1], np.zeros(n - 1))
    assert plus[-1] < 0 < minus[-1]
    np.testing.assert_array_equal(system.dynamics_nominal(np.zeros(n)), np.zeros(n))


def test_power_rate_sign_boundary_and_continuous_boundary():
    for alpha in (0, Fraction(1, 3)):
        system = PowerRateSystem(alpha=alpha)
        x = np.zeros(1)
        expected = system.k if alpha == 0 else 0
        np.testing.assert_array_equal(system.dynamics_plus(x), [[-expected]])
        np.testing.assert_array_equal(system.dynamics_minus(x), [[expected]])
        np.testing.assert_array_equal(system.dynamics_nominal(x), [[0]])


def test_cta_orthant_limits_have_zero_continuous_injection_at_origin():
    from gf_framework.systems.continuous_twisting import CTASystem
    system = CTASystem()
    for signs in ("plus_plus", "plus_minus", "minus_plus", "minus_minus"):
        value = getattr(system, "dynamics_" + signs)(np.zeros(3))
        np.testing.assert_array_equal(value[:2], np.zeros(2))


def test_regularity_uses_exact_rational_boundary():
    below = Fraction(10**20 - 1, 10**20)
    above = Fraction(10**20 + 1, 10**20)
    assert float(below) == float(above) == 1.0
    for kind in ("signed", "abs"):
        assert classify_term_regularity(below, kind) == "sub_lipschitz"
        assert classify_term_regularity(above, kind) == "C1"
    assert classify_term_regularity(Fraction(1), "abs") == "lipschitz_kink"
    assert classify_term_regularity(Fraction(1), "signed") == "C1"
