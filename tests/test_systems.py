"""
Tests for the Systems Module.

Validates that all 9 benchmark controllers instantiate correctly,
provide proper dynamics, and integrate with the GF framework.
"""

import pytest
import numpy as np

from gf_framework.systems import (
    CONTROLLER_REGISTRY,
    get_system,
    FOSMCSystem,
    PowerRateSystem,
    STASystem,
    TwistingSystem,
    Example1System,
    Example3System,
    CTASystem,
    Levant3DSystem,
    PIDSMCSystem,
)
from gf_framework.systems.fosmc import (
    get_fosmc_dynamics_gf, get_fosmc_known_lyapunov,
    FOSMC_WEIGHTS, FOSMC_DEGREE, FOSMC_KAPPA,
)
from gf_framework.systems.sta import (
    get_sta_dynamics_gf, get_sta_known_lyapunov,
    STA_WEIGHTS, STA_DEGREE, STA_KAPPA,
)
from gf_framework.systems.pidsmc import (
    get_pidsmc_dynamics_gf, get_pidsmc_known_lyapunov_structure,
    PIDSMC_WEIGHTS, PIDSMC_DEGREE, PIDSMC_KAPPA,
)
from gf_framework.systems.twisting import TWISTING_WEIGHTS, TWISTING_DEGREE
from gf_framework.algebra.core import GeneralizedForm
from gf_framework.interfaces import SystemSpec


# Expected system metadata
SYSTEM_METADATA = {
    "fosmc":      {"weights": (1,),    "n_vars": 1},
    "power_rate": {"weights": (1,),    "n_vars": 1},
    "sta":        {"weights": (2, 1),  "n_vars": 2},
    "twisting":   {"weights": (2, 1),  "n_vars": 2},
    "example1":   {"weights": (1, 3),  "n_vars": 2},
    "example3":   {"weights": (3, 2),  "n_vars": 2},
    "cta":        {"weights": (3, 2, 1), "n_vars": 3},
    "levant3d":   {"weights": (3, 2, 1), "n_vars": 3},
    "pidsmc":     {"weights": (3, 2, 1), "n_vars": 3},
}


# =========================================================================
# Registry
# =========================================================================

class TestControllerRegistry:
    """Test CONTROLLER_REGISTRY completeness."""

    def test_registry_has_9_entries(self):
        """CONTROLLER_REGISTRY has exactly 9 entries."""
        assert len(CONTROLLER_REGISTRY) == 9

    def test_registry_keys(self):
        """All expected controller names are present."""
        expected = {
            "fosmc", "power_rate", "sta", "twisting",
            "example1", "example3", "cta", "levant3d", "pidsmc",
        }
        assert set(CONTROLLER_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        """All registry values are callable classes."""
        for name, cls in CONTROLLER_REGISTRY.items():
            assert callable(cls), f"{name} is not callable"


# =========================================================================
# get_system() factory
# =========================================================================

class TestGetSystem:
    """Test get_system() factory function."""

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_get_system_creates_instance(self, name):
        """get_system() creates an instance for each name."""
        system = get_system(name)
        assert system is not None

    def test_get_system_invalid_name(self):
        """get_system() raises KeyError for unknown name."""
        with pytest.raises(KeyError):
            get_system("nonexistent_system")


# =========================================================================
# All systems instantiate
# =========================================================================

class TestAllSystemsInstantiate:
    """Test that all 9 controllers instantiate with default parameters."""

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_system_instantiation(self, name):
        """Each system instantiates without error."""
        system = get_system(name)
        assert system is not None

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_system_has_weights(self, name):
        """Each system has a .weights property returning a tuple."""
        system = get_system(name)
        weights = system.weights
        assert isinstance(weights, tuple)
        assert all(isinstance(w, int) for w in weights)
        assert all(w > 0 for w in weights)
        expected = SYSTEM_METADATA[name]["weights"]
        assert weights == expected, f"{name}: expected {expected}, got {weights}"

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_system_has_kappa(self, name):
        """Each system has a .kappa property returning a float."""
        system = get_system(name)
        kappa = system.kappa
        assert isinstance(kappa, (int, float))

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_system_n_vars(self, name):
        """Number of state variables matches weights length."""
        system = get_system(name)
        expected_n_vars = SYSTEM_METADATA[name]["n_vars"]
        assert len(system.weights) == expected_n_vars


# =========================================================================
# Dynamics evaluation
# =========================================================================

class TestDynamicsEvaluation:
    """Test that each system's dynamics callables work."""

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_dynamics_nominal(self, name):
        """dynamics_nominal() returns tensor of correct shape."""
        system = get_system(name)
        n = SYSTEM_METADATA[name]["n_vars"]
        x = np.random.randn(10, n)
        dx = system.dynamics_nominal(x)
        assert dx.shape[-1] == n

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_dynamics_plus(self, name):
        """dynamics_plus() returns tensor of correct shape (where available)."""
        system = get_system(name)
        if not hasattr(system, 'dynamics_plus'):
            pytest.skip(f"{name} is a continuous system (no Filippov split)")
        n = SYSTEM_METADATA[name]["n_vars"]
        x = np.random.randn(10, n)
        dx = system.dynamics_plus(x)
        assert dx.shape[-1] == n

    @pytest.mark.parametrize("name", list(CONTROLLER_REGISTRY.keys()))
    def test_dynamics_minus(self, name):
        """dynamics_minus() returns tensor of correct shape (where available)."""
        system = get_system(name)
        if not hasattr(system, 'dynamics_minus'):
            pytest.skip(f"{name} is a continuous system (no Filippov split)")
        n = SYSTEM_METADATA[name]["n_vars"]
        x = np.random.randn(10, n)
        dx = system.dynamics_minus(x)
        assert dx.shape[-1] == n

    def test_fosmc_dynamics_sign(self):
        """FOSMC: dynamics = -k*sign(s)."""
        system = FOSMCSystem(k=2.0)
        x_pos = np.array([[1.0]])
        x_neg = np.array([[-1.0]])

        dx_pos = system.dynamics_nominal(x_pos)
        dx_neg = system.dynamics_nominal(x_neg)

        assert dx_pos.item() < 0, "ṡ should be negative for s > 0"
        assert dx_neg.item() > 0, "ṡ should be positive for s < 0"

    def test_sta_dynamics_structure(self):
        """STA: f1 depends on sqrt(s), f2 = -k2*sign(s)."""
        system = STASystem(k1=1.5, k2=1.1)
        x = np.array([[1.0, 0.0]])
        dx = system.dynamics_nominal(x)

        # At (s=1, v=0): ds = -k1*1^{1/2} + 0 = -1.5, dv = -k2 = -1.1
        assert dx.shape[-1] == 2
        np.testing.assert_allclose(dx[0].item(), -1.5, atol=0.01)
        np.testing.assert_allclose(dx[1].item(), -1.1, atol=0.01)


# =========================================================================
# GF dynamics
# =========================================================================

class TestGFDynamics:
    """Test module-level get_*_dynamics_gf() functions."""

    def test_fosmc_dynamics_gf(self):
        """FOSMC dynamics as GF list."""
        f = get_fosmc_dynamics_gf(k=1.0)
        assert isinstance(f, list)
        assert len(f) == 1
        assert all(isinstance(g, GeneralizedForm) for g in f)

    def test_sta_dynamics_gf(self):
        """STA dynamics as GF list."""
        f = get_sta_dynamics_gf(k1=1.5, k2=1.1)
        assert isinstance(f, list)
        assert len(f) == 2
        assert all(isinstance(g, GeneralizedForm) for g in f)

    def test_pidsmc_dynamics_gf(self):
        """PID-SMC dynamics as GF list."""
        f = get_pidsmc_dynamics_gf()
        assert isinstance(f, list)
        assert len(f) == 3
        assert all(isinstance(g, GeneralizedForm) for g in f)


# =========================================================================
# Known Lyapunov functions
# =========================================================================

class TestKnownLyapunov:
    """Test known Lyapunov function constructors."""

    def test_fosmc_known_lyapunov(self):
        """FOSMC V = |s| structure."""
        V = get_fosmc_known_lyapunov()
        assert V.n_vars == 1
        assert V.degree_m == FOSMC_DEGREE
        assert V.weights_r == FOSMC_WEIGHTS

    def test_sta_known_lyapunov(self):
        """STA Moreno-Osorio V structure."""
        V = get_sta_known_lyapunov()
        assert V.n_vars == 2
        assert V.degree_m == STA_DEGREE
        assert V.weights_r == STA_WEIGHTS
        assert len(V.terms) == 3  # diagonal + cross

    def test_pidsmc_known_lyapunov_structure(self):
        """PID-SMC V structure (9 terms)."""
        V = get_pidsmc_known_lyapunov_structure()
        assert V.n_vars == 3
        assert V.degree_m == PIDSMC_DEGREE
        assert V.weights_r == PIDSMC_WEIGHTS
        assert len(V.terms) == 9


# =========================================================================
# SystemSpec creation
# =========================================================================

class TestSystemSpec:
    """Test SystemSpec creation for each controller."""

    def test_fosmc_spec(self):
        """FOSMC SystemSpec."""
        spec = SystemSpec(
            weights_r=FOSMC_WEIGHTS,
            degree_m=FOSMC_DEGREE,
            dynamics_gf=get_fosmc_dynamics_gf(),
        )
        assert spec.n_vars == 1
        assert spec.weights_r == (1,)

    def test_sta_spec(self):
        """STA SystemSpec."""
        spec = SystemSpec(
            weights_r=STA_WEIGHTS,
            degree_m=STA_DEGREE,
            dynamics_gf=get_sta_dynamics_gf(),
        )
        assert spec.n_vars == 2

    def test_pidsmc_spec(self):
        """PID-SMC SystemSpec."""
        spec = SystemSpec(
            weights_r=PIDSMC_WEIGHTS,
            degree_m=PIDSMC_DEGREE,
            dynamics_gf=get_pidsmc_dynamics_gf(),
        )
        assert spec.n_vars == 3

    def test_spec_validation_rejects_bad_degree(self):
        """SystemSpec rejects non-positive degree."""
        with pytest.raises(ValueError):
            SystemSpec(weights_r=(2, 1), degree_m=0)

    def test_spec_validation_rejects_bad_weights(self):
        """SystemSpec rejects non-positive weights."""
        with pytest.raises(ValueError):
            SystemSpec(weights_r=(0, 1), degree_m=2)


# =========================================================================
# Module constants
# =========================================================================

class TestModuleConstants:
    """Test module-level constants match expected values."""

    def test_fosmc_constants(self):
        assert FOSMC_WEIGHTS == (1,)
        assert FOSMC_DEGREE == 1
        assert FOSMC_KAPPA == -1.0

    def test_sta_constants(self):
        assert STA_WEIGHTS == (2, 1)
        assert STA_DEGREE == 2
        assert STA_KAPPA == -1.0

    def test_pidsmc_constants(self):
        assert PIDSMC_WEIGHTS == (3, 2, 1)
        assert PIDSMC_DEGREE == 7
        assert PIDSMC_KAPPA == -1.0

    def test_twisting_constants(self):
        assert TWISTING_WEIGHTS == (2, 1)
        assert TWISTING_DEGREE == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
