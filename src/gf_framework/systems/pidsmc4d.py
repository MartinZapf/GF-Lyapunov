from __future__ import annotations

"""
Fourth-order PID-like SMC system (campaign block E6, code item C11).

Natural relative-degree extension of the third-order PID-SMC benchmark
(``pidsmc.py``) to r = (4, 3, 2, 1), exactly as recorded in
``docs/papers/big_draft/PRE_WRITING_CAMPAIGN.md`` (block "E6 — 4th-order
PID-like SMC"):

    x1' = x2
    x2' = x3
    x3' = -k1|x1|^(1/4)sign(x1) - k2|x2|^(1/3)sign(x2) - k3|x3|^(1/2)sign(x3) + x4
    x4' = -k4 sign(x1)          (+ matched delta on channel 3, weighted degree 0)

This is r-homogeneous with:
    - weights r = (4, 3, 2, 1)
    - degree kappa = -1
    - cheapest C1 Lyapunov degree: m = 5 (R2: m > max r_i, max r_i = 4)

Weighted degrees (kappa + r_i), checked term by term:
    f1: -1 + 4 = 3   (x2 has weight 3) ok
    f2: -1 + 3 = 2   (x3 has weight 2) ok
    f3: -1 + 2 = 1   (k1|x1|^(1/4): 4*(1/4)=1; k2|x2|^(1/3): 3*(1/3)=1;
                      k3|x3|^(1/2): 2*(1/2)=1; x4 has weight 1) ok
    f4: -1 + 1 = 0   (sign(x1) has weighted degree 0) ok

The "channel 3, weighted degree 0" disturbance the campaign card describes is
the x4' equation (0-indexed channel 3, the same convention ``levant4d.py``
uses for its own top channel): it is the unique weighted-degree-0 component,
which is exactly the structural requirement
``discovery.universal_discovery.matched_disturbance_vertices`` enforces for a
constant matched disturbance |delta| <= delta0 to preserve homogeneity.

CLAIM-HYGIENE QUARANTINE (campaign review finding; see the E6 section of
PRE_WRITING_CAMPAIGN.md). This formulation and its gains are a best-guess,
natural extension of the third-order benchmark, not a confirmed reproduction
of any specific prior construction. Until Moreno/Sanchez confirm that this is
the formulation and gain set they mean, no sentence here or downstream may tie
this system to "the Sanchez example" or describe it as "previously attempted
by hand": it is described only as "a fourth-order PID-like extension". If the
confirmed formulation differs, this best-guess run moves to NEGATIVES.md
context and the module is re-run against the confirmed version.

PROVISIONAL GAINS (review finding 28: must be pre-registered in the E6 block
README before the first discovery run is trusted). Default
(k1, k2, k3, k4) = (1.1, 2.7, 5.345, 1.1). The derivation, two alternative
candidates, and the recommendation rationale are in the C11 implementation
report accompanying this module (not duplicated here to avoid drift between
two copies); Moreno's answer confirms or replaces this set.
"""

from typing import Tuple, List
from fractions import Fraction
import numpy as np

from gf_framework.algebra.core import GeneralizedForm, GFTerm


def sgnpow(x: np.ndarray, p: float) -> np.ndarray:
    """Signed power: sign(x) * |x|^p"""
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


# Default configuration
PIDSMC4D_WEIGHTS = (4, 3, 2, 1)
PIDSMC4D_KAPPA = -1.0
PIDSMC4D_DEGREE = 5  # cheapest C1 degree (m > max r_i); discovery escalates

# Provisional gains (BEST-GUESS, quarantined; see module docstring and the C11
# report). k2, k3, k4 reuse the published Perez-Ventura pidsmc triple
# (lambda1, lambda2, lambda3) = (2.7, 5.345, 1.1) unchanged, since they play the
# SAME term role after the index shift (k2 multiplies |x2|^(1/3), the same
# exponent pidsmc's k1 used on x1; k3 multiplies |x3|^(1/2), the same exponent
# pidsmc's k2 used on x2; k4 multiplies sign(x1), exactly pidsmc's k3 role,
# same variable). k1 is the genuinely new leading term (exponent 1/4, no
# pidsmc counterpart); the provisional choice reuses Levant's smallest
# published universal SMC constant (lambda_hat_0 = 1.1) rather than an
# invented number.
_K1 = 1.1
_K2 = 2.7
_K3 = 5.345
_K4 = 1.1


class PIDSMC4DSystem:
    """
    Fourth-order PID-like SMC closed-loop system (best-guess extension).

    x1' = x2
    x2' = x3
    x3' = -k1|x1|^(1/4)sign(x1) - k2|x2|^(1/3)sign(x2) - k3|x3|^(1/2)sign(x3) + x4
    x4' = -k4 sign(x1)
    """

    def __init__(
        self,
        k1: float = _K1,
        k2: float = _K2,
        k3: float = _K3,
        k4: float = _K4,
    ):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

    @property
    def weights(self) -> Tuple[int, ...]:
        return PIDSMC4D_WEIGHTS

    @property
    def kappa(self) -> float:
        return PIDSMC4D_KAPPA

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics (no disturbance).

        Args:
            x: State tensor of shape (batch, 4) or (4,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        dx1 = x2
        dx2 = x3
        dx3 = (
            -self.k1 * sgnpow(x1, 1 / 4)
            - self.k2 * sgnpow(x2, 1 / 3)
            - self.k3 * sgnpow(x3, 1 / 2)
            + x4
        )
        dx4 = -self.k4 * np.sign(x1)

        return np.stack([dx1, dx2, dx3, dx4], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """
        Dynamics when x1 > 0 (sign(x1) = +1).

        Only the genuinely discontinuous term (bare sign(x1) in x4') needs a
        fixed endpoint value for Filippov analysis; the sgnpow(x1, ...) terms
        in x3' are continuous and vanish at x1 = 0, so they keep the general
        formula (same convention as ``pidsmc.py``).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        dx1 = x2
        dx2 = x3
        dx3 = (
            -self.k1 * sgnpow(x1, 1 / 4)
            - self.k2 * sgnpow(x2, 1 / 3)
            - self.k3 * sgnpow(x3, 1 / 2)
            + x4
        )
        dx4 = -self.k4 * np.ones_like(x1)  # sign(x1) = +1

        return np.stack([dx1, dx2, dx3, dx4], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """
        Dynamics when x1 < 0 (sign(x1) = -1).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        dx1 = x2
        dx2 = x3
        dx3 = (
            -self.k1 * sgnpow(x1, 1 / 4)
            - self.k2 * sgnpow(x2, 1 / 3)
            - self.k3 * sgnpow(x3, 1 / 2)
            + x4
        )
        dx4 = self.k4 * np.ones_like(x1)  # sign(x1) = -1

        return np.stack([dx1, dx2, dx3, dx4], axis=1).squeeze()


def create_pidsmc4d_dynamics(
    k1: float = _K1,
    k2: float = _K2,
    k3: float = _K3,
    k4: float = _K4,
):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = PIDSMC4DSystem(k1, k2, k3, k4)
    return system.dynamics_plus, system.dynamics_minus


def get_pidsmc4d_dynamics_gf(
    k1: float = _K1,
    k2: float = _K2,
    k3: float = _K3,
    k4: float = _K4,
) -> List[GeneralizedForm]:
    """
    Fourth-order PID-like SMC dynamics as Generalized Forms.

    Weighted degrees (kappa + r_i), see the module docstring for the term-by-term
    check.
    """
    weights_r = PIDSMC4D_WEIGHTS

    def _z():
        return Fraction(0)

    # f1 = x2
    f1 = GeneralizedForm(degree_m=3, weights_r=weights_r, terms=[
        GFTerm(coefficient=1.0,
               exponents=(_z(), Fraction(1), _z(), _z()),
               term_types=('abs', 'signed', 'abs', 'abs')),
    ])

    # f2 = x3
    f2 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=[
        GFTerm(coefficient=1.0,
               exponents=(_z(), _z(), Fraction(1), _z()),
               term_types=('abs', 'abs', 'signed', 'abs')),
    ])

    # f3 = -k1|x1|^(1/4)sign(x1) - k2|x2|^(1/3)sign(x2) - k3|x3|^(1/2)sign(x3) + x4
    f3 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=[
        GFTerm(coefficient=-k1,
               exponents=(Fraction(1, 4), _z(), _z(), _z()),
               term_types=('signed', 'abs', 'abs', 'abs')),
        GFTerm(coefficient=-k2,
               exponents=(_z(), Fraction(1, 3), _z(), _z()),
               term_types=('abs', 'signed', 'abs', 'abs')),
        GFTerm(coefficient=-k3,
               exponents=(_z(), _z(), Fraction(1, 2), _z()),
               term_types=('abs', 'abs', 'signed', 'abs')),
        GFTerm(coefficient=1.0,
               exponents=(_z(), _z(), _z(), Fraction(1)),
               term_types=('abs', 'abs', 'abs', 'signed')),
    ])

    # f4 = -k4 sign(x1)
    f4 = GeneralizedForm(degree_m=0, weights_r=weights_r, terms=[
        GFTerm(coefficient=-k4,
               exponents=(_z(), _z(), _z(), _z()),
               term_types=('signed', 'abs', 'abs', 'abs')),
    ])

    return [f1, f2, f3, f4]
