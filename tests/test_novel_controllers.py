"""Smoke tests for novel controllers DSTA and FC-CTA.

Verifies:
- System modules import and construct without errors
- Spec factories produce valid SystemSpec objects
- Dynamics callables produce correct-shape outputs
- GF representations have expected weighted degree
- Homogeneity sanity check on a single sample
"""
from __future__ import annotations

import sys
from pathlib import Path
from fractions import Fraction

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


class TestDSTA:
    def test_module_imports(self):
        from gf_framework.systems.dsta import (
            DSTASystem, create_dsta_dynamics, get_dsta_dynamics_gf,
            DSTA_WEIGHTS, DSTA_KAPPA, DSTA_DEGREE,
        )
        assert DSTA_WEIGHTS == (2, 1)
        assert DSTA_KAPPA == -1.0
        assert DSTA_DEGREE == 2

    def test_dynamics_shape(self):
        from gf_framework.systems.dsta import DSTASystem
        sys_ = DSTASystem()
        x = np.array([[0.5, -0.3], [-0.2, 0.4]])
        y = sys_.dynamics_nominal(x)
        assert y.shape == (2, 2)

    def test_spec_factory(self):
        from gf_framework.discovery.universal_discovery import create_dsta_spec
        spec = create_dsta_spec()
        assert spec.name == "DSTA"
        assert spec.weights_r == (2, 1)
        assert spec.kappa == -1.0
        assert spec.degree_m == 2
        assert len(spec.dynamics_gf) == 2

    def test_gf_homogeneity_degree(self):
        """f_i must have weighted degree κ + r_i."""
        from gf_framework.systems.dsta import get_dsta_dynamics_gf
        f = get_dsta_dynamics_gf()
        assert f[0].degree_m == 1  # κ + r₁ = -1 + 2 = 1
        assert f[1].degree_m == 0  # κ + r₂ = -1 + 1 = 0

    def test_known_v_homogeneous(self):
        from gf_framework.systems.dsta import get_dsta_known_lyapunov
        v = get_dsta_known_lyapunov()
        assert v.degree_m == 2
        assert v.weights_r == (2, 1)


class TestFCCTA:
    def test_module_imports(self):
        from gf_framework.systems.fc_cta import (
            FCCTASystem, create_fc_cta_dynamics, get_fc_cta_dynamics_gf,
            FC_CTA_WEIGHTS, FC_CTA_KAPPA, FC_CTA_DEGREE,
        )
        assert FC_CTA_WEIGHTS == (3, 2, 1)
        assert FC_CTA_KAPPA == -1.0
        assert FC_CTA_DEGREE == 5

    def test_dynamics_shape(self):
        from gf_framework.systems.fc_cta import FCCTASystem
        sys_ = FCCTASystem()
        x = np.array([[0.5, -0.3, 0.1], [-0.2, 0.4, -0.05]])
        y = sys_.dynamics_nominal(x)
        assert y.shape == (2, 3)

    def test_spec_factory(self):
        from gf_framework.discovery.universal_discovery import create_fc_cta_spec
        spec = create_fc_cta_spec()
        assert spec.name == "FC-CTA"
        assert spec.weights_r == (3, 2, 1)
        assert spec.kappa == -1.0
        assert spec.degree_m == 5
        assert len(spec.dynamics_gf) == 3

    def test_gf_homogeneity_degree(self):
        from gf_framework.systems.fc_cta import get_fc_cta_dynamics_gf
        f = get_fc_cta_dynamics_gf()
        assert f[0].degree_m == 2  # κ + r₁ = -1 + 3
        assert f[1].degree_m == 1  # κ + r₂ = -1 + 2
        assert f[2].degree_m == 0  # κ + r₃ = -1 + 1

    def test_novel_relay_present(self):
        """The k₅·sign(x₃) term distinguishes FC-CTA from CTA."""
        from gf_framework.systems.fc_cta import get_fc_cta_dynamics_gf
        f_list = get_fc_cta_dynamics_gf(k5=2.0)
        f3 = f_list[2]
        signed_x3_terms = [t for t in f3.terms
                           if t.term_types == ('abs', 'abs', 'signed')
                           and t.exponents == (Fraction(0), Fraction(0), Fraction(0))]
        assert len(signed_x3_terms) == 1
        assert signed_x3_terms[0].coefficient == pytest.approx(-2.0)


class TestCLIRegistry:
    def test_dsta_in_spec_factories(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from discover_gf import SPEC_FACTORIES
        assert "dsta" in SPEC_FACTORIES
        assert "fc_cta" in SPEC_FACTORIES

    def test_dsta_in_matrix_controllers(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from discover_gf import ALL
        assert "dsta" in ALL
        assert "fc_cta" in ALL
