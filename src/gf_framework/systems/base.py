from __future__ import annotations
from typing import Protocol, Tuple, List, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class System(Protocol):
    """Protocol for GF-compatible dynamical systems."""

    @property
    def name(self) -> str: ...

    @property
    def weights_r(self) -> Tuple[int, ...]: ...

    @property
    def kappa(self) -> int: ...

    @property
    def degree_m(self) -> int: ...

    @property
    def n_vars(self) -> int: ...

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray: ...

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray: ...

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray: ...

    def known_lyapunov_terms(self) -> Optional[List]: ...
