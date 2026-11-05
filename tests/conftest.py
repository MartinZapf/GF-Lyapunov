"""Pytest configuration for GF Framework tests."""

import sys
from pathlib import Path

# Ensure src/ is on the path for editable installs
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import numpy as np
from fractions import Fraction

from gf_framework.algebra.core import (
    GFTerm, GeneralizedForm, create_diagonal_gf,
)
from gf_framework.interfaces import (
    SystemSpec, SelectionResult, VerificationConfig, Certificate,
)


# ---------------------------------------------------------------------------
# MOSEK availability check
# ---------------------------------------------------------------------------
def _mosek_available() -> bool:
    try:
        import mosek  # noqa: F401
        return True
    except ImportError:
        return False


MOSEK_AVAILABLE = _mosek_available()


# ---------------------------------------------------------------------------
# CVXPY availability check
# ---------------------------------------------------------------------------
def _cvxpy_available() -> bool:
    try:
        import cvxpy  # noqa: F401
        return True
    except ImportError:
        return False


CVXPY_AVAILABLE = _cvxpy_available()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fosmc_v_gf():
    """FOSMC known Lyapunov: V(s) = |s|."""
    return create_diagonal_gf(degree_m=1, weights_r=(1,), coefficients=[1.0])


@pytest.fixture
def sta_v_diagonal_gf():
    """STA diagonal Lyapunov: V = |x1| + |x2|^2."""
    return create_diagonal_gf(degree_m=2, weights_r=(2, 1), coefficients=[1.0, 1.0])


@pytest.fixture
def fosmc_spec():
    """FOSMC SystemSpec (V-only, no dynamics)."""
    return SystemSpec(weights_r=(1,), degree_m=1, dynamics_gf=None)


@pytest.fixture
def fosmc_selection(fosmc_v_gf):
    """FOSMC SelectionResult with V = |s|."""
    return SelectionResult(
        active_terms=[],
        coefficients={0: 1.0},
        V_gf=fosmc_v_gf,
        numerical_valid=True,
        max_V_violation=-0.1,
        max_W_violation=-0.05,
    )
