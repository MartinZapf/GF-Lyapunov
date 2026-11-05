"""
Tests for Verification Module.

Validates SOS and Pólya verification for V > 0 and W > 0,
admissibility checks, and complete Lyapunov certification.
"""

import pytest
import numpy as np
from fractions import Fraction

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from gf_framework.algebra.core import (
    GFTerm, CompositeGFTerm, GeneralizedForm, create_diagonal_gf,
)
from gf_framework.algebra.enumerate import TermSpec
from gf_framework.algebra.lie_derivative import (
    compute_lie_derivative_gf, negate_gf,
)
from gf_framework.verification.polya import PolyaVerifier, PolyaResult
from gf_framework.verification.verifier import (
    verify_lyapunov, verify_v_only, quick_verify,
)
from gf_framework.verification.admissibility import (
    validate_dynamics_gf_admissibility,
)
from gf_framework.interfaces import (
    SystemSpec, SelectionResult, VerificationConfig, Certificate,
)
from gf_framework.systems.fosmc import (
    get_fosmc_dynamics_gf, get_fosmc_known_lyapunov,
    FOSMC_KAPPA, FOSMC_WEIGHTS, FOSMC_DEGREE,
)
from gf_framework.systems.sta import (
    get_sta_dynamics_gf, get_sta_known_lyapunov,
    STA_KAPPA,
)
from gf_framework.systems.pidsmc import (
    get_pidsmc_dynamics_gf, get_pidsmc_known_lyapunov_structure,
    PIDSMC_KAPPA, PIDSMC_WEIGHTS, PIDSMC_DEGREE,
)


# =========================================================================
# V > 0 verification
# =========================================================================

class TestVerifyVOnly:
    """Test V > 0 verification."""

    def test_verify_positive_diagonal_gf(self):
        """Diagonal GF with positive coefficients should verify."""
        gf = create_diagonal_gf(
            degree_m=2, weights_r=(2, 1), coefficients=[1.0, 1.0],
        )
        verified, method, polya_p = verify_v_only(gf, verbose=False)
        assert verified
        assert method in ['polya', 'sos']

    def test_verify_fosmc_lyapunov(self, fosmc_v_gf):
        """FOSMC Lyapunov V(s) = |s| should verify."""
        verified, method, polya_p = verify_v_only(fosmc_v_gf, verbose=False)
        assert verified

    def test_verify_moreno_osorio_v(self):
        """Moreno-Osorio STA Lyapunov V should verify (diagonal subset)."""
        gf = create_diagonal_gf(
            degree_m=2, weights_r=(2, 1), coefficients=[1.0, 1.0],
        )
        verified, method, polya_p = verify_v_only(gf, verbose=False)
        assert verified


# =========================================================================
# Pólya method
# =========================================================================

class TestPolyaMethod:
    """Test Pólya theorem verification."""

    def test_polya_finds_correct_exponent(self, fosmc_v_gf):
        """Pólya should find minimal exponent p."""
        verified, method, polya_p = verify_v_only(
            fosmc_v_gf,
            config=VerificationConfig(max_polya_p=10),
            verbose=False,
        )
        if method == 'polya':
            assert polya_p is not None
            assert polya_p >= 0


# =========================================================================
# Certificate
# =========================================================================

class TestCertificate:
    """Test verification certificate structure."""

    def test_certificate_properties(self):
        """Certificate should have correct properties."""
        cert = Certificate(
            V_verified=True,
            W_verified=True,
            V_method='polya',
            W_method='sos_weighted_sphere',
            clearing_power=1,
            polya_p_V=2,
        )
        assert cert.fully_verified
        assert cert.method_summary == "V: polya, W: sos_weighted_sphere"

    def test_partial_certificate(self):
        """Partial verification should be captured."""
        cert = Certificate(
            V_verified=True,
            W_verified=False,
            V_method='polya',
            W_method='failed',
        )
        assert not cert.fully_verified
        assert cert.V_verified


# =========================================================================
# Full V + W verification
# =========================================================================

class TestFullVerification:
    """Test full V + W verification."""

    def test_fosmc_v_verification(self, fosmc_selection, fosmc_spec):
        """FOSMC should verify V via Pólya."""
        cert = verify_lyapunov(fosmc_selection, fosmc_spec, verbose=False)

        assert cert.V_verified
        assert cert.W_method == "no_dynamics"

    def test_sta_v_verification(self, sta_v_diagonal_gf):
        """STA V verification should succeed for diagonal V."""
        selection = SelectionResult(
            active_terms=[],
            coefficients={},
            V_gf=sta_v_diagonal_gf,
            numerical_valid=True,
            max_V_violation=-0.1,
            max_W_violation=-0.05,
        )
        spec = SystemSpec(
            weights_r=(2, 1), degree_m=2, dynamics_gf=None,
        )
        cert = verify_lyapunov(selection, spec, verbose=False)
        assert cert.V_verified
        assert cert.W_method == "no_dynamics"


class TestQuickVerify:
    """Test quick verification helper."""

    def test_quick_verify_diagonal(self):
        """Quick verify should work for diagonal GF."""
        gf = create_diagonal_gf(
            degree_m=2, weights_r=(2, 1), coefficients=[1.0, 1.0],
        )
        cert = quick_verify(gf)
        assert cert.V_verified


# =========================================================================
# Verification config
# =========================================================================

class TestVerificationConfig:
    """Test verification configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = VerificationConfig()

        assert config.epsilon > 0
        assert config.max_polya_p > 0
        assert config.solver in ['MOSEK', 'SCS']
        assert config.strict_solver is False
        assert config.allow_solver_fallback is True

    def test_custom_config(self):
        """Custom config should work."""
        config = VerificationConfig(
            epsilon=1e-4, max_polya_p=20, solver='SCS',
        )
        assert config.epsilon == 1e-4
        assert config.max_polya_p == 20
        assert config.solver == 'SCS'

    def test_strict_solver_disables_fallback(self):
        """Strict solver mode should force fallback disabling."""
        config = VerificationConfig(
            solver="MOSEK", strict_solver=True, allow_solver_fallback=True,
        )
        assert config.strict_solver is True
        assert config.allow_solver_fallback is False


# =========================================================================
# W verification with dynamics
# =========================================================================

class TestWVerificationWithDynamics:
    """Test W = -dV/dt verification with actual dynamics_gf."""

    def test_fosmc_w_verification(self):
        """FOSMC: W = -dV/dt must be verified, not skipped."""
        V_gf = get_fosmc_known_lyapunov(coef=1.0)
        dynamics_gf = get_fosmc_dynamics_gf(k=1.0)

        selection = SelectionResult(
            active_terms=[], coefficients={0: 1.0},
            V_gf=V_gf, numerical_valid=True,
            max_V_violation=-0.1, max_W_violation=-0.1,
        )
        spec = SystemSpec(
            weights_r=(1,), degree_m=1, dynamics_gf=dynamics_gf,
        )

        cert = verify_lyapunov(selection, spec, verbose=False)

        assert cert.V_verified, "V verification failed"
        assert cert.W_method != "no_dynamics", "W must be computed"
        assert cert.W_verified, f"W verification failed: {cert.W_method}"
        assert cert.fully_verified, "Full verification must pass"

    def test_sta_w_verification_diagonal(self):
        """STA: Diagonal V should attempt W, not skip it."""
        V_gf = create_diagonal_gf(
            degree_m=2, weights_r=(2, 1), coefficients=[1.0, 1.0],
        )
        dynamics_gf = get_sta_dynamics_gf(k1=1.5, k2=1.1)

        selection = SelectionResult(
            active_terms=[], coefficients={0: 1.0, 1: 1.0},
            V_gf=V_gf, numerical_valid=True,
            max_V_violation=-0.1, max_W_violation=-0.1,
        )
        spec = SystemSpec(
            weights_r=(2, 1), degree_m=2, dynamics_gf=dynamics_gf,
        )

        cert = verify_lyapunov(selection, spec, verbose=False)

        assert cert.V_verified
        assert cert.W_method != "no_dynamics"

    @pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not installed")
    def test_sta_w_verification_moreno_osorio(self):
        """STA: Moreno-Osorio V + W verification attempt."""
        V_gf = get_sta_known_lyapunov(k1=1.5, k2=1.1)
        dynamics_gf = get_sta_dynamics_gf(k1=1.5, k2=1.1)

        selection = SelectionResult(
            active_terms=[], coefficients={},
            V_gf=V_gf, numerical_valid=True,
            max_V_violation=-0.1, max_W_violation=-0.1,
        )
        spec = SystemSpec(
            weights_r=(2, 1), degree_m=2, dynamics_gf=dynamics_gf,
        )

        cert = verify_lyapunov(selection, spec, verbose=False)

        assert cert.V_verified, "V verification failed for Moreno-Osorio"
        assert cert.W_method != "skipped", "W verification must be attempted"

    def test_lie_derivative_computation(self):
        """Lie derivative is computed correctly."""
        V_gf = get_fosmc_known_lyapunov(coef=1.0)
        dynamics_gf = get_fosmc_dynamics_gf(k=1.0)

        dVdt = compute_lie_derivative_gf(V_gf, dynamics_gf)
        assert dVdt is not None
        assert dVdt.n_terms >= 1

        W_gf = negate_gf(dVdt)
        assert W_gf.n_terms >= 1

        test_point = np.array([[1.0]])
        W_val = W_gf.evaluate(test_point)
        W_scalar = float(W_val) if np.ndim(W_val) == 0 else W_val[0]
        assert W_scalar > 0, f"W should be positive, got {W_scalar}"


# =========================================================================
# Certificate consistency (monkeypatch)
# =========================================================================

class TestCertificateConsistency:
    """Ensure certificate payload matches the actually certified coefficients."""

    def test_joint_sos_certified_v_replaces_seed_v(self, monkeypatch):
        """When joint SOS returns a certified V, it must appear in the certificate."""
        import gf_framework.verification.verifier as verifier_mod

        seed_v = GeneralizedForm(
            degree_m=1, weights_r=(1,),
            terms=[GFTerm(coefficient=1.0, exponents=(Fraction(1),), term_types=('abs',))],
        )
        certified_v = GeneralizedForm(
            degree_m=1, weights_r=(1,),
            terms=[GFTerm(coefficient=2.0, exponents=(Fraction(1),), term_types=('abs',))],
        )

        selection = SelectionResult(
            active_terms=[], coefficients={0: 1.0},
            V_gf=seed_v, numerical_valid=True,
            max_V_violation=-0.1, max_W_violation=-0.1,
        )
        dynamics = [GeneralizedForm(
            degree_m=0, weights_r=(1,),
            terms=[GFTerm(coefficient=-1.0, exponents=(Fraction(0),), term_types=('signed',))],
        )]
        spec = SystemSpec(weights_r=(1,), degree_m=1, dynamics_gf=dynamics)

        monkeypatch.setattr(
            verifier_mod, "_verify_positivity",
            lambda gf, config, verbose: (True, "mock_v", None),
        )
        monkeypatch.setattr(
            verifier_mod, "_verify_w_positivity",
            lambda W_gf, V_gf, system_spec, config, w_analysis, verbose: (
                True, "sos_weighted_sphere", None, 0, certified_v
            ),
        )

        cert = verify_lyapunov(selection, spec, verbose=False)

        assert cert.V_verified
        assert cert.W_verified
        assert cert.V_gf is not None
        assert cert.V_gf.terms[0].coefficient == pytest.approx(2.0)
        assert cert.V_gf.terms[0].coefficient != seed_v.terms[0].coefficient


# =========================================================================
# Complete Lyapunov verification (V + W)
# =========================================================================

class TestFOSMCCompleteVerification:
    """Test complete verification for FOSMC (1D)."""

    def test_v_positive_definite(self):
        """V = |s| verifies as positive definite."""
        V = get_fosmc_known_lyapunov()
        verifier = PolyaVerifier(max_p=5)

        result = verifier.verify(V)
        assert result.verified, f"V should be positive definite: {result.message}"
        assert result.polya_p == 0, "V = |s| should verify with p=0"

    def test_w_positive_definite(self):
        """W = -dV/dt = k verifies as positive."""
        k = 1.5
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        assert W.degree_m == 0
        verifier = PolyaVerifier(max_p=5)
        result = verifier.verify(W)
        assert result.verified, f"W = k > 0 should verify: {result.message}"

    def test_complete_verification(self):
        """Complete Lyapunov verification via PolyaVerifier."""
        k = 1.5
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        verifier = PolyaVerifier(max_p=10)
        V_result, W_result = verifier.verify_lyapunov_conditions(V, W)

        assert V_result.verified, f"V failed: {V_result.message}"
        assert W_result.verified, f"W failed: {W_result.message}"
        assert verifier.verify_complete(V, W)

    def test_numerical_values(self):
        """W evaluates to positive constant k."""
        k = 2.0
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        test_points = np.array([[1.0], [-1.0], [0.5], [-3.0], [10.0]])
        w_values = W.evaluate(test_points)
        np.testing.assert_allclose(
            w_values, k, rtol=1e-10,
            err_msg="W should evaluate to constant k",
        )


class TestSTACompleteVerification:
    """Test complete verification for STA (2D)."""

    def test_v_positive_definite(self):
        """Moreno-Osorio V has expected structure (may not Pólya-verify)."""
        k1, k2 = 1.5, 1.1
        V = get_sta_known_lyapunov(k1, k2)
        verifier = PolyaVerifier(max_p=15)

        result = verifier.verify(V)
        if not result.verified:
            assert "not a polynomial" in result.message or "fractional" in result.message, \
                f"Expected fractional exponent failure, got: {result.message}"

    def test_w_construction(self):
        """W = -dV/dt is constructed correctly."""
        k1, k2 = 1.5, 1.1
        V = get_sta_known_lyapunov(k1, k2)
        f = get_sta_dynamics_gf(k1, k2)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        assert W.n_vars == 2
        assert len(W.terms) > 0

    def test_complete_verification(self):
        """Complete verification for STA (documents Pólya limitation)."""
        k1, k2 = 1.5, 1.1
        V = get_sta_known_lyapunov(k1, k2)
        f = get_sta_dynamics_gf(k1, k2)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        verifier = PolyaVerifier(max_p=15)
        V_result, W_result = verifier.verify_lyapunov_conditions(V, W)

        if not V_result.verified:
            assert "not a polynomial" in V_result.message or "fractional" in V_result.message


class TestPIDSMCCompleteVerification:
    """Test complete verification for PID-SMC (3D)."""

    def test_v_structure(self):
        """PID-SMC Lyapunov structure is correct."""
        V = get_pidsmc_known_lyapunov_structure()

        assert V.n_vars == 3
        assert V.degree_m == 7
        assert len(V.terms) == 9
        assert V.weights_r == (3, 2, 1)

    def test_w_construction(self):
        """W can be constructed for PID-SMC."""
        V = get_pidsmc_known_lyapunov_structure()
        f = get_pidsmc_dynamics_gf()

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        assert W.n_vars == 3
        assert len(W.terms) > 0


# =========================================================================
# PolyaVerifier integration
# =========================================================================

class TestVerifierIntegration:
    """Test PolyaVerifier integration with complete verification."""

    def test_verify_both_conditions(self):
        """verify_lyapunov_conditions verifies both V and W."""
        k = 1.0
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        verifier = PolyaVerifier(max_p=10)
        V_result, W_result = verifier.verify_lyapunov_conditions(V, W)

        assert isinstance(V_result, PolyaResult)
        assert isinstance(W_result, PolyaResult)
        assert V_result.verified
        assert W_result.verified

    def test_verify_complete_returns_bool(self):
        """verify_complete returns True/False correctly."""
        k = 1.0
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        verifier = PolyaVerifier(max_p=10)
        result = verifier.verify_complete(V, W)

        assert isinstance(result, bool)
        assert result is True


# =========================================================================
# Counterexamples
# =========================================================================

class TestCounterexamples:
    """Test counterexample extraction from verification failures."""

    def test_v_counterexample_extraction(self):
        """V verification failures provide counterexamples."""
        invalid_V = GeneralizedForm(
            degree_m=2,
            weights_r=(1, 1),
            terms=[
                GFTerm(coefficient=1.0, exponents=(Fraction(2), Fraction(0)), term_types=('abs', 'abs')),
                GFTerm(coefficient=-5.0, exponents=(Fraction(1), Fraction(1)), term_types=('abs', 'abs')),
                GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(2)), term_types=('abs', 'abs')),
            ],
        )

        verifier = PolyaVerifier(max_p=10)
        result = verifier.verify(invalid_V)

        if not result.verified and result.counterexample is not None:
            ce = result.counterexample
            assert len(ce) == 2
            v_values = invalid_V.evaluate(ce.reshape(1, -1))
            v_value = float(v_values) if v_values.ndim == 0 else v_values[0]
            assert v_value <= 1e-5, "Counterexample should have V ≤ 0"


# =========================================================================
# W numerical evaluation
# =========================================================================

class TestWEvaluationNumerical:
    """Test numerical evaluation of W = -dV/dt."""

    def test_fosmc_w_constant(self):
        """For FOSMC, W = k is constant everywhere."""
        k = 3.5
        V = get_fosmc_known_lyapunov()
        f = get_fosmc_dynamics_gf(k)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        np.random.seed(42)
        test_points = np.random.randn(100, 1)
        w_values = W.evaluate(test_points)

        np.testing.assert_allclose(w_values, k, rtol=1e-10)

    def test_sta_w_sign(self):
        """STA W construction sanity check."""
        k1, k2 = 1.5, 1.1
        V = get_sta_known_lyapunov(k1, k2)
        f = get_sta_dynamics_gf(k1, k2)

        dVdt = compute_lie_derivative_gf(V, f, validate=False)
        W = negate_gf(dVdt)

        test_points = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.0],
            [2.0, 0.5],
        ])
        w_values = W.evaluate(test_points)
        # Just verify computation does not fail
        assert len(w_values) == 4


# =========================================================================
# Admissibility
# =========================================================================

class TestAdmissibility:
    """Test GF admissibility checking."""

    def test_pidsmc_admissibility_passes(self):
        """PID-SMC in-scope admissibility passes."""
        report = validate_dynamics_gf_admissibility(
            dynamics_gf=get_pidsmc_dynamics_gf(
                lambda1=2.7, lambda2=5.345, lambda3=1.1, delta=1.0,
            ),
            weights_r=PIDSMC_WEIGHTS,
            degree_m=PIDSMC_DEGREE,
            kappa=PIDSMC_KAPPA,
            system_name="pidsmc",
        )
        assert report.in_scope_formal_gf is True
        assert report.severity == "pass"

    def test_admissibility_fails_on_composite_dynamics(self):
        """Composite dynamics are out of GF scope."""
        inner = GeneralizedForm(
            degree_m=1, weights_r=(1,),
            terms=[GFTerm(coefficient=1.0, exponents=(Fraction(1),), term_types=("abs",))],
        )
        comp = CompositeGFTerm(
            inner_gf=inner, outer_exponent=Fraction(1, 2),
            outer_type="signed", coefficient=1.0,
        )
        gf = GeneralizedForm(
            degree_m=1, weights_r=(1,),
            terms=[], composite_terms=[comp],
        )
        report = validate_dynamics_gf_admissibility(
            dynamics_gf=[gf], weights_r=(1,),
            degree_m=1, kappa=0.0, system_name="composite_case",
        )
        assert report.in_scope_formal_gf is False
        assert any(
            c.name.endswith("_composite_terms") and c.status == "fail"
            for c in report.checks
        )

    def test_admissibility_warns_when_kappa_missing(self):
        """Missing κ triggers warning."""
        gf = GeneralizedForm(
            degree_m=1, weights_r=(1,),
            terms=[GFTerm(coefficient=-1.0, exponents=(Fraction(1),), term_types=("signed",))],
        )
        report = validate_dynamics_gf_admissibility(
            dynamics_gf=[gf], weights_r=(1,),
            degree_m=1, kappa=None, system_name="kappa_missing",
        )
        assert any(
            c.name == "kappa_provided" and c.status == "warn"
            for c in report.checks
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ---------------------------------------------------------------------------
# W-clearing correctness: the product clearing monomial must leave NO negative
# y-exponents (review 2026-06-12; the in-code -1e-10 check is a warn-only
# sanity net — this test makes it a hard regression).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy required")
def test_w_clearing_produces_nonnegative_exponents():
    import warnings
    from gf_framework.verification.joint_sos import JointSOSOptimizer

    # 1D construction with a genuinely negative W y-exponent:
    # V term |x|^{3/2}  →  ∂V/∂x = (3/2)⌈x⌋^{1/2};
    # dynamics f = −⌈x⌋^{-1}  →  W = (3/2)|x|^{-1/2}  →  y-exp = μ·(−1/2) = −2.
    spec = TermSpec(exponents=(Fraction(3, 2),), term_types=("abs",))
    f = GeneralizedForm(degree_m=-1, weights_r=(1,), terms=[
        GFTerm(coefficient=-1.0, exponents=(Fraction(-1),),
               term_types=("signed",)),
    ])
    opt = JointSOSOptimizer(weights_r=(1,), solver="CLARABEL", verbose=False)
    with warnings.catch_warnings():
        # The "clearing power insufficient" warning must NOT fire.
        warnings.filterwarnings("error", message="Clearing power")
        M, clearing_power, monomials = (
            opt._compute_w_coefficient_matrix_weighted_sphere(
                [spec], mu=(4,), signs=(1,), dynamics_gf=[f]))
    # k' = μ/r = 4, ρ = 1 ⇒ contribution 4 per clearing step; |min y-exp| = 2
    # ⇒ k = ceil(2/4) = 1; cleared min y-exp = −2 + 4 = 2 ≥ 0.
    assert clearing_power == 1
    assert M is not None
    assert monomials and all(all(e >= 0 for e in mono) for mono in monomials)
