from __future__ import annotations

"""
Dynamics-independent necessary pre-check for positive-definiteness.

A Generalized-Form Lyapunov candidate ``V`` is positive definite only if
``V(x) > 0`` for every ``x ≠ 0``. Finding *any* point with ``V(x) < 0`` therefore
refutes ``V > 0`` outright — no SOS certificate can exist. Because ``V`` is
(generalized-)homogeneous, its sign is constant along dilation orbits, so a dense
sample of directions on the unit sphere reliably exposes negativity if present.

This check can only ever *reject*; it never accepts. It is cheap (float, no
solver) and catches the whole "certified V is not even PD" failure class (e.g.
the published CTA / Levant3D certificates, ``min V ≈ -3e-3 / -5e-3`` on the
sphere) before any SDP is attempted.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gf_framework.algebra.core import GeneralizedForm


@dataclass
class SphereCheck:
    min_V: float
    argmin: list
    n_samples: int
    is_possibly_pd: bool  # False ⇒ definitely NOT positive definite

    @property
    def refutes_pd(self) -> bool:
        return not self.is_possibly_pd


def min_V_on_sphere(
    V_gf: GeneralizedForm,
    n_samples: int = 200_000,
    tol: float = 1e-9,
    seed: int = 0,
) -> SphereCheck:
    """Sample ``V`` on the Euclidean unit sphere and report its minimum.

    Args:
        V_gf: the Lyapunov candidate (its ``evaluate`` consumes ``(N, n)`` arrays).
        n_samples: number of random directions.
        tol: a point counts as a refutation when ``V < -tol``.
        seed: RNG seed (deterministic).

    Returns:
        SphereCheck with ``is_possibly_pd = False`` iff a point with ``V < -tol``
        was found.
    """
    n = len(V_gf.weights_r)
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_samples, n))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Vs = np.asarray(V_gf.evaluate(D), dtype=float)
    finite = np.isfinite(Vs)
    Vs = Vs[finite]
    Df = D[finite]
    if Vs.size == 0:
        return SphereCheck(float("nan"), [0.0] * n, 0, True)
    idx = int(np.argmin(Vs))
    mn = float(Vs[idx])
    return SphereCheck(
        min_V=mn,
        argmin=[float(v) for v in Df[idx]],
        n_samples=int(Vs.size),
        is_possibly_pd=bool(mn >= -tol),
    )
