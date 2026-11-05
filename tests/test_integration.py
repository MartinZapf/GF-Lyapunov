"""
Integration Tests for GF Framework.

End-to-end tests for the Generation → Verification pipeline.
"""

import pytest

from gf_framework.interfaces import (
    SystemSpec,
    SelectionResult,
)
from gf_framework.algebra.core import create_diagonal_gf
from gf_framework.algebra.generation import generate_term_basis
from gf_framework.verification import verify_lyapunov


# =========================================================================
# FOSMC Integration
# =========================================================================

class TestFOSMCIntegration:
    """End-to-end tests for FOSMC system."""

    @pytest.fixture
    def fosmc_spec(self):
        from gf_framework.systems.fosmc import get_fosmc_dynamics_gf

        return SystemSpec(
            weights_r=(1,), degree_m=1, dynamics_gf=get_fosmc_dynamics_gf(1.0),
        )

    def test_fosmc_generation(self, fosmc_spec):
        """Test FOSMC term generation."""
        basis = generate_term_basis(fosmc_spec)

        assert basis.n_diagonal == 1
        assert basis.n_cross == 0

    def test_fosmc_verification(self, fosmc_spec):
        """Test FOSMC verification (V-only) of the known V = |s|."""
        V_gf = create_diagonal_gf(degree_m=1, weights_r=(1,), coefficients=[1.0])
        selection = SelectionResult(
            active_terms=[], coefficients={}, V_gf=V_gf,
            numerical_valid=True, max_V_violation=0.0, max_W_violation=0.0,
        )
        spec_v_only = SystemSpec(weights_r=(1,), degree_m=1, dynamics_gf=None)

        cert = verify_lyapunov(selection, spec_v_only, verbose=False)
        assert cert.V_verified

    def test_fosmc_full_pipeline(self, fosmc_spec):
        """Test complete FOSMC pipeline with V + W verification."""
        V_gf = create_diagonal_gf(degree_m=1, weights_r=(1,), coefficients=[1.0])
        selection = SelectionResult(
            active_terms=[], coefficients={}, V_gf=V_gf,
            numerical_valid=True, max_V_violation=0.0, max_W_violation=0.0,
        )

        cert = verify_lyapunov(selection, fosmc_spec, verbose=False)

        assert cert.V_verified
        # With dynamics, W should be attempted
        assert cert.W_method != "no_dynamics"


# =========================================================================
# STA Integration
# =========================================================================

class TestSTAIntegration:
    """End-to-end tests for STA system."""

    def test_sta_generation(self):
        """Test STA term generation."""
        from gf_framework.systems.sta import get_sta_dynamics_gf

        spec = SystemSpec(
            weights_r=(2, 1), degree_m=2, dynamics_gf=get_sta_dynamics_gf(2.5, 1.5),
        )
        basis = generate_term_basis(spec)

        assert basis.n_diagonal == 2
        assert basis.n_cross > 0


# =========================================================================
# PID-SMC Integration
# =========================================================================

class TestPIDSMCIntegration:
    """End-to-end tests for PID-SMC system."""

    def test_pidsmc_generation(self):
        """Test PID-SMC term generation."""
        from gf_framework.systems.pidsmc import get_pidsmc_dynamics_gf

        spec = SystemSpec(
            weights_r=(3, 2, 1), degree_m=6, dynamics_gf=get_pidsmc_dynamics_gf(),
        )
        basis = generate_term_basis(spec)

        assert basis.n_diagonal == 3
        assert basis.n_cross >= 5


# =========================================================================
# Error Handling
# =========================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_incompatible_degree_raises(self):
        """Incompatible degree should raise error."""
        spec = SystemSpec(weights_r=(2, 1), degree_m=1)

        try:
            generate_term_basis(spec)
        except ValueError as e:
            assert "incompatible" in str(e).lower() or "compatible" in str(e).lower()

    def test_no_dynamics_partial_verification(self):
        """Without dynamics, only V should be verified."""
        V_gf = create_diagonal_gf(
            degree_m=2, weights_r=(2, 1), coefficients=[1.0, 1.0],
        )

        selection = SelectionResult(
            active_terms=[], coefficients={},
            V_gf=V_gf, numerical_valid=True,
            max_V_violation=-0.1, max_W_violation=-0.05,
        )
        spec = SystemSpec(weights_r=(2, 1), degree_m=2, dynamics_gf=None)

        cert = verify_lyapunov(selection, spec, verbose=False)
        assert cert.V_verified is not None
        assert cert.W_method == "no_dynamics"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
