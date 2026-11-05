"""Levant-5D (fourth-order differentiator) system definition checks.

The n=5 scale-boundary probe: these tests pin the GF encoding (weighted
degrees, exponent/type structure) and the non-recursive gain conversion
before any budgeted discovery run trusts them.
"""
from __future__ import annotations

import numpy as np
import pytest

from gf_framework.systems.levant5d import (
    Levant5DSystem,
    get_levant5d_dynamics_gf,
    LEVANT5D_WEIGHTS,
    LEVANT5D_KAPPA,
    _L1, _L2, _L3, _L4, _L5,
)
from gf_framework.discovery.universal_discovery import create_levant5d_spec


class TestLevant5DGF:
    def test_weighted_degrees(self):
        """Each fᵢ is a GF of degree κ + rᵢ and every term matches it."""
        dynamics = get_levant5d_dynamics_gf()
        assert len(dynamics) == 5
        for i, f in enumerate(dynamics):
            expected = LEVANT5D_KAPPA + LEVANT5D_WEIGHTS[i]
            assert f.degree_m == expected
            for t in f.terms:
                wdeg = sum(float(r * e)
                           for r, e in zip(LEVANT5D_WEIGHTS, t.exponents))
                assert wdeg == pytest.approx(expected)

    def test_term_structure(self):
        """−λᵢ⌈e₁⌋^((5−i)/5) plus the chain term ⌈e_{i+1}⌋ (none for f₅)."""
        dynamics = get_levant5d_dynamics_gf()
        gains = (_L1, _L2, _L3, _L4, _L5)
        for i, f in enumerate(dynamics):
            e1_terms = [t for t in f.terms if t.exponents[0] != 0 or i == 4]
            assert len(e1_terms) == 1
            t = e1_terms[0]
            assert t.coefficient == pytest.approx(-gains[i])
            assert t.term_types[0] == "signed"
            assert len(f.terms) == (1 if i == 4 else 2)

    def test_gain_conversion(self):
        """Non-recursive conversion of Levant-2003 (1.1, 1.5, 2, 3, 5), L=1."""
        assert _L1 == 5.0
        assert _L2 == pytest.approx(3.0 * 5.0 ** 0.75)
        assert _L3 == pytest.approx(2.0 * _L2 ** (2.0 / 3.0))
        assert _L4 == pytest.approx(1.5 * _L3 ** 0.5)
        assert _L5 == 1.1


class TestLevant5DDynamics:
    def test_homogeneity(self):
        """fᵢ(Λ_ε x) = ε^(κ+rᵢ) fᵢ(x) for the numerical dynamics."""
        sys5 = Levant5DSystem()
        rng = np.random.default_rng(7)
        x = rng.uniform(0.5, 2.0, size=(4, 5)) * rng.choice([-1, 1], size=(4, 5))
        for eps in (0.5, 2.0):
            scale = np.array([eps ** r for r in LEVANT5D_WEIGHTS])
            fx = np.atleast_2d(sys5.dynamics_nominal(x))
            fsx = np.atleast_2d(sys5.dynamics_nominal(x * scale))
            for i in range(5):
                expected = eps ** (LEVANT5D_KAPPA + LEVANT5D_WEIGHTS[i]) * fx[:, i]
                np.testing.assert_allclose(fsx[:, i], expected, rtol=1e-9)


class TestLevant5DSpec:
    def test_spec(self):
        spec = create_levant5d_spec()
        assert spec.n_vars == 5
        assert tuple(spec.weights_r) == (5, 4, 3, 2, 1)
        assert spec.degree_m == 6  # cheapest C¹ degree (R2: m > max rᵢ)
        assert len(spec.dynamics_gf) == 5
