"""
System-spec registry for GF Lyapunov discovery.

Defines ``SystemSpec`` (the universal system description consumed by
``discovery.gf_discovery.discover_gf``), the ``create_*_spec`` factories for
every benchmark controller, the matched-disturbance inclusion vertices for
robust certification, and numerical sanity-check utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from fractions import Fraction

from gf_framework.algebra.core import GeneralizedForm, GFTerm

def numerical_verification(
    V_gf: GeneralizedForm,
    dynamics_gf: List[GeneralizedForm],
    n_samples: int = 100000,
    domain_scale: float = 2.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify Lyapunov function numerically via dense sampling.

    This is a fallback when formal SOS verification fails (which can happen
    for well-conditioned 3D problems like PID-SMC).

    NOTE: This function supports dynamics with composite terms via
    evaluate_gf_with_composites, though such systems (like CTA) are
    outside the formal GF framework and not officially supported.

    Args:
        V_gf: Lyapunov candidate
        dynamics_gf: System dynamics (may contain composite terms)
        n_samples: Number of random samples
        domain_scale: Sample domain is [-scale, scale]^n

    Returns:
        (verified, details) where verified=True if V>0 and dV/dt<0 everywhere
    """
    import numpy as np
    from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf, evaluate_gf_with_composites

    n_vars = len(dynamics_gf)

    # Check if dynamics have composite terms
    has_composite = any(
        hasattr(gf, 'composite_terms') and len(gf.composite_terms) > 0
        for gf in dynamics_gf if gf is not None
    )

    # Sample random points (exclude origin)
    np.random.seed(42)
    points = np.random.uniform(-domain_scale, domain_scale, (n_samples, n_vars))
    # Filter out points too close to origin
    norms = np.linalg.norm(points, axis=1)
    points = points[norms > 0.01]

    # Evaluate V
    v_values = np.array([V_gf.evaluate(p) for p in points])
    v_min = v_values.min()
    v_positive = (v_values > 0).all()

    # Compute and evaluate W = dV/dt
    # NOTE: For composite dynamics, compute_lie_derivative_gf stores product
    # composite terms in _product_composite_terms, which must be included
    # via evaluate_gf_with_composites for correct numerical evaluation.
    W = compute_lie_derivative_gf(V_gf, dynamics_gf, validate=False)

    # Use evaluate_gf_with_composites to include all contributions
    w_values = np.array([evaluate_gf_with_composites(W, p) for p in points])
    w_max = w_values.max()
    w_negative = (w_values < 0).all()

    verified = v_positive and w_negative

    details = {
        'n_samples': len(points),
        'v_min': float(v_min),
        'v_positive': bool(v_positive),
        'w_max': float(w_max),
        'w_negative': bool(w_negative),
        'v_negative_count': int((v_values <= 0).sum()),
        'w_positive_count': int((w_values >= 0).sum()),
        'has_composite_dynamics': has_composite,
    }

    return verified, details


def has_composite_dynamics(dynamics_gf: List[GeneralizedForm]) -> bool:
    """
    Check if dynamics contain composite GF terms.

    Composite terms (e.g., ⌈σ(x)⌋^ρ where σ is a GF) are non-polynomial
    in lifted y-space and cannot be directly handled by SOS verification.

    For such systems:
    - SOS verification will be partial (excluding composite contributions)
    - Numerical verification is required for complete validation

    Args:
        dynamics_gf: System dynamics as list of GeneralizedForm

    Returns:
        True if any dynamics component has composite terms
    """
    return any(
        hasattr(gf, 'composite_terms') and len(gf.composite_terms) > 0
        for gf in dynamics_gf if gf is not None
    )


@dataclass
class SystemSpec:
    """Universal system specification."""
    dynamics_gf: List[GeneralizedForm]
    weights_r: Tuple[int, ...]
    degree_m: int
    kappa: float  # homogeneity degree
    name: str = "unnamed"
    # Domain bounds for sampling (per variable)
    domain: Optional[List[Tuple[float, float]]] = None
    # Differential-inclusion vertices for robust certification under a bounded
    # matched disturbance (see matched_disturbance_vertices). None ⇒ nominal.
    dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]] = None
    # Disturbance bound the vertices encode (informational, for artifacts).
    disturbance_bound: Optional[float] = None

    @property
    def n_vars(self) -> int:
        return len(self.weights_r)

    @property
    def has_composite_dynamics(self) -> bool:
        """Check if dynamics contain composite (nested GF) terms."""
        return has_composite_dynamics(self.dynamics_gf)


def matched_disturbance_vertices(
    dynamics_gf: List[GeneralizedForm],
    channel: int,
    delta0: float,
) -> List[List[GeneralizedForm]]:
    """Endpoint vertices of the inclusion ẋ ∈ f(x) + [-δ₀, δ₀]·e_channel.

    A constant matched disturbance is only admissible on a channel whose GF has
    weighted degree 0 (κ + r_channel = 0): the constant ±δ₀ is itself a GF of
    weighted degree 0, so homogeneity is preserved — exactly the structure of
    the sign-discontinuous channel of every SMC benchmark (FOSMC ṡ, STA v̇,
    PID-SMC/CTA/Levant ẋ₃/ė₃). Since W = −∇V·f is affine in δ, certifying both
    endpoint vertices certifies the whole interval by convexity.

    Returns [f + δ₀·e_ch, f − δ₀·e_ch] as two full dynamics lists.
    """
    if delta0 < 0:
        raise ValueError(f"delta0 must be ≥ 0, got {delta0}")
    chan = dynamics_gf[channel]
    if Fraction(chan.degree_m) != 0:
        raise ValueError(
            f"matched disturbance needs a weighted-degree-0 channel "
            f"(κ + r_ch = 0); channel {channel} has GF degree {chan.degree_m}"
        )
    n = len(chan.weights_r)
    vertices = []
    for sign in (+1, -1):
        # Preserve an exact Fraction δ₀ (exact-rational gain-injection runs);
        # float inputs keep the historical float64 encoding bit for bit.
        coeff = sign * delta0 if isinstance(delta0, Fraction) else float(sign) * float(delta0)
        const = GFTerm(
            coefficient=coeff,
            exponents=tuple(Fraction(0) for _ in range(n)),
            term_types=tuple('abs' for _ in range(n)),  # ∏|xᵢ|⁰ = 1
        )
        chan_k = GeneralizedForm(
            degree_m=chan.degree_m,
            weights_r=chan.weights_r,
            terms=list(chan.terms) + [const],
        )
        vertices.append([
            chan_k if i == channel else f for i, f in enumerate(dynamics_gf)
        ])
    return vertices


# Convenience functions for specific systems
def create_fosmc_spec(k: float = 1.0, delta0: float = 0.0) -> SystemSpec:
    """Create spec for FOSMC system (matched disturbance |δ| ≤ delta0 on ṡ)."""
    from gf_framework.systems.fosmc import (
        get_fosmc_dynamics_gf, FOSMC_WEIGHTS, FOSMC_DEGREE, FOSMC_KAPPA,
    )


    dynamics_gf = get_fosmc_dynamics_gf(k)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=FOSMC_WEIGHTS,
        degree_m=FOSMC_DEGREE,
        kappa=FOSMC_KAPPA,
        name="FOSMC",
        domain=[(-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 0, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_sta_spec(k1: float = 1.5, k2: float = 1.1,
                    delta0: float = 0.0) -> SystemSpec:
    """Create spec for STA system (matched disturbance |δ| ≤ delta0 on v̇)."""
    from gf_framework.systems.sta import (
        get_sta_dynamics_gf, STA_WEIGHTS, STA_DEGREE, STA_KAPPA,
    )


    dynamics_gf = get_sta_dynamics_gf(k1, k2)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=STA_WEIGHTS,
        degree_m=STA_DEGREE,
        kappa=STA_KAPPA,
        name="STA",
        domain=[(-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 1, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_example1_spec(k1: float = 1.0, k2: float = 1.0) -> SystemSpec:
    """Create spec for Example1 (polynomial) system."""
    from gf_framework.systems.example1 import (
        get_example1_dynamics_gf, EXAMPLE1_WEIGHTS, EXAMPLE1_DEGREE, EXAMPLE1_KAPPA,
    )


    return SystemSpec(
        dynamics_gf=get_example1_dynamics_gf(k1, k2),
        weights_r=EXAMPLE1_WEIGHTS,
        degree_m=EXAMPLE1_DEGREE,
        kappa=EXAMPLE1_KAPPA,
        name="Example1",
        domain=[(-2, 2), (-2, 2)],
    )


def create_example3_spec(k1: float = 1.8, k2: float = 4.2) -> SystemSpec:
    """Create spec for Example3 (double integrator) system."""
    from gf_framework.systems.example3 import (
        get_example3_dynamics_gf, EXAMPLE3_WEIGHTS, EXAMPLE3_DEGREE, EXAMPLE3_KAPPA,
    )


    return SystemSpec(
        dynamics_gf=get_example3_dynamics_gf(k1, k2),
        weights_r=EXAMPLE3_WEIGHTS,
        degree_m=EXAMPLE3_DEGREE,
        kappa=EXAMPLE3_KAPPA,
        name="Example3",
        domain=[(-2, 2), (-2, 2)],
    )


def create_pidsmc_spec(
    lambda1: float = 2.7,
    lambda2: float = 5.345,
    lambda3: float = 1.1,
    delta: float = 1.0,
    delta0: float = 0.0,
) -> SystemSpec:
    """Create spec for PID-SMC system (matched disturbance |δ| ≤ delta0 on ẋ₃)."""
    from gf_framework.systems.pidsmc import (
        get_pidsmc_dynamics_gf,
        PIDSMC_WEIGHTS, PIDSMC_DEGREE, PIDSMC_KAPPA,
    )


    dynamics_gf = get_pidsmc_dynamics_gf(lambda1, lambda2, lambda3, delta)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=PIDSMC_WEIGHTS,
        degree_m=PIDSMC_DEGREE,
        kappa=PIDSMC_KAPPA,
        name="PID-SMC",
        domain=[(-2, 2), (-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 2, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_power_rate_spec(k: float = 1.5, alpha: float = 0.5) -> SystemSpec:
    """Create spec for Power-Rate Reaching Law system."""
    from gf_framework.systems.power_rate import (
        get_power_rate_dynamics_gf, POWER_RATE_WEIGHTS, POWER_RATE_DEGREE,
    )


    dynamics = get_power_rate_dynamics_gf(k, alpha)
    kappa = float(dynamics[0].degree_m - 1)
    return SystemSpec(
        dynamics_gf=dynamics,
        weights_r=POWER_RATE_WEIGHTS,
        degree_m=POWER_RATE_DEGREE,
        kappa=kappa,
        name="PowerRate",
        domain=[(-2, 2)],
    )


def create_twisting_spec(r1: float = 2.0, r2: float = 1.5,
                         delta0: float = 0.0) -> SystemSpec:
    """Create spec for Twisting Algorithm (matched disturbance |δ| ≤ delta0 on ẋ₂)."""
    from gf_framework.systems.twisting import (
        get_twisting_dynamics_gf, TWISTING_WEIGHTS, TWISTING_DEGREE, TWISTING_KAPPA,
    )
    dynamics_gf = get_twisting_dynamics_gf(r1, r2)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=TWISTING_WEIGHTS,
        degree_m=TWISTING_DEGREE,
        kappa=TWISTING_KAPPA,
        name="Twisting",
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 1, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_levant3d_spec(
    lambda1: float = 3.0,
    lambda2: float = 2.0,
    lambda3: float = 1.1,
    delta0: float = 0.0,
) -> SystemSpec:
    """Create spec for Levant 3D Differentiator (|f⁽³⁾| ≤ delta0 enters ė₃)."""
    from gf_framework.systems.levant3d import (
        get_levant3d_dynamics_gf,
        LEVANT3D_WEIGHTS, LEVANT3D_DEGREE, LEVANT3D_KAPPA,
    )


    dynamics_gf = get_levant3d_dynamics_gf(lambda1, lambda2, lambda3)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=LEVANT3D_WEIGHTS,
        degree_m=LEVANT3D_DEGREE,
        kappa=LEVANT3D_KAPPA,
        name="Levant-3D",
        domain=[(-2, 2), (-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 2, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_levant4d_spec(
    lambda1: float = None,
    lambda2: float = None,
    lambda3: float = None,
    lambda4: float = None,
    delta0: float = 0.0,
) -> SystemSpec:
    """Create spec for Levant 4D (third-order) differentiator (|f⁽⁴⁾| ≤ delta0 enters ė₄).

    The n=4 scaling demonstrator: first system where the μ-compatible C¹ cross
    library is the largest on the benchmark (≈97 terms at m=8).
    """
    from gf_framework.systems.levant4d import (
        get_levant4d_dynamics_gf,
        LEVANT4D_WEIGHTS, LEVANT4D_DEGREE, LEVANT4D_KAPPA,
        _L1, _L2, _L3, _L4,
    )
    l1 = _L1 if lambda1 is None else lambda1
    l2 = _L2 if lambda2 is None else lambda2
    l3 = _L3 if lambda3 is None else lambda3
    l4 = _L4 if lambda4 is None else lambda4


    dynamics_gf = get_levant4d_dynamics_gf(l1, l2, l3, l4)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=LEVANT4D_WEIGHTS,
        degree_m=LEVANT4D_DEGREE,
        kappa=LEVANT4D_KAPPA,
        name="Levant-4D",
        domain=[(-2, 2), (-2, 2), (-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 3, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_pidsmc4d_spec(
    k1: float = None,
    k2: float = None,
    k3: float = None,
    k4: float = None,
    delta0: float = 0.0,
) -> SystemSpec:
    """Create spec for the fourth-order PID-like SMC system (C11, campaign block E6).

    BEST-GUESS / PROVISIONAL formulation and gains: natural relative-degree
    extension of ``pidsmc`` to r = (4, 3, 2, 1). Not yet confirmed against any
    specific prior hand attempt — see ``systems/pidsmc4d.py`` for the full
    claim-hygiene quarantine. Matched disturbance |delta| <= delta0 enters x4'
    (the unique weighted-degree-0 channel, index 3, same convention as
    ``create_levant4d_spec``).
    """
    from gf_framework.systems.pidsmc4d import (
        get_pidsmc4d_dynamics_gf,
        PIDSMC4D_WEIGHTS, PIDSMC4D_DEGREE, PIDSMC4D_KAPPA,
        _K1, _K2, _K3, _K4,
    )
    kk1 = _K1 if k1 is None else k1
    kk2 = _K2 if k2 is None else k2
    kk3 = _K3 if k3 is None else k3
    kk4 = _K4 if k4 is None else k4

    dynamics_gf = get_pidsmc4d_dynamics_gf(kk1, kk2, kk3, kk4)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=PIDSMC4D_WEIGHTS,
        degree_m=PIDSMC4D_DEGREE,
        kappa=PIDSMC4D_KAPPA,
        name="PID-SMC-4D",
        domain=[(-2, 2), (-2, 2), (-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 3, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_levant5d_spec(
    lambda1: float = None,
    lambda2: float = None,
    lambda3: float = None,
    lambda4: float = None,
    lambda5: float = None,
) -> SystemSpec:
    """Create spec for Levant 5D (fourth-order) differentiator.

    The n=5 scale-boundary probe (NOT part of the benchmark matrix): 32
    octants per solve, dense per-octant Gram bound C(m+4,4) rows.
    """
    from gf_framework.systems.levant5d import (
        get_levant5d_dynamics_gf,
        LEVANT5D_WEIGHTS, LEVANT5D_DEGREE, LEVANT5D_KAPPA,
        _L1, _L2, _L3, _L4, _L5,
    )
    l1 = _L1 if lambda1 is None else lambda1
    l2 = _L2 if lambda2 is None else lambda2
    l3 = _L3 if lambda3 is None else lambda3
    l4 = _L4 if lambda4 is None else lambda4
    l5 = _L5 if lambda5 is None else lambda5

    return SystemSpec(
        dynamics_gf=get_levant5d_dynamics_gf(l1, l2, l3, l4, l5),
        weights_r=LEVANT5D_WEIGHTS,
        degree_m=LEVANT5D_DEGREE,
        kappa=LEVANT5D_KAPPA,
        name="Levant-5D",
        domain=[(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)],
    )


# =============================================================================
# CTA (Continuous Twisting Algorithm) - Torres-González et al. 2017
# =============================================================================
# CTA is a valid GF system per Sanchez-Moreno Definition 3.
#
# From Torres-González, Fridman & Moreno (2017), "Design of Continuous
# Twisting Algorithm", Automatica 80:119-126, equation (4):
#
#   ẋ₁ = x₂
#   ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
#   ẋ₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0
#
# ALL terms are products of single-variable signed/absolute powers - NO
# composite terms. Full SOS verification is possible.
#
# NOTE: Some literature describes a different "continuous twisting" variant
# with composite terms like ⌈σ⌋^(1/2) where σ = ⌈x₁⌋^(2/3) + x₂. That variant
# is NOT the Torres-González CTA and is NOT supported.
# =============================================================================


def create_cta_spec(
    k1: float = 54.8,
    k2: float = 27.0,
    k3: float = 5.7,
    k4: float = 3.0,
    delta0: float = 0.0,
) -> SystemSpec:
    """
    Create spec for Continuous Twisting Algorithm (Torres-González 2017).

    CTA is a 3D r-homogeneous system with:
        - weights r = (3, 2, 1)
        - degree κ = -1 (finite-time stable)
        - Lyapunov degree m = 5

    The third equation has a Twisting structure: -k₃·sign(x₁) - k₄·sign(x₂)

    Stability requires k₃ > k₄ (necessary for unique equilibrium).

    Args:
        k1, k2, k3, k4: Controller gains (defaults from Lemma 1 for μ=1)

    Returns:
        SystemSpec for CTA discovery
    """
    from gf_framework.systems.continuous_twisting import (
        get_cta_dynamics_gf,
        CTA_WEIGHTS, CTA_DEGREE, CTA_KAPPA,
    )



    dynamics_gf = get_cta_dynamics_gf(k1, k2, k3, k4)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=CTA_WEIGHTS,
        degree_m=CTA_DEGREE,
        kappa=CTA_KAPPA,
        name="CTA",
        domain=[(-2, 2), (-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 2, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_dsta_spec(k1: float = 1.5, k2: float = 1.1, k3: float = 0.5,
                     delta0: float = 0.0) -> SystemSpec:
    """
    Create spec for the Damped Super-Twisting Algorithm (DSTA).

    NOVEL controller: STA + sign(x₂) damping in the second equation.
        ẋ₁ = -k₁⌈x₁⌋^{1/2} + x₂
        ẋ₂ = -k₂·sign(x₁) - k₃·sign(x₂)

    r = (2, 1), κ = -1, m = 2. Stability requires k₂ > k₃ > 0.
    Matched disturbance |δ| ≤ delta0 enters ẋ₂.
    """
    from gf_framework.systems.dsta import (
        get_dsta_dynamics_gf,
        DSTA_WEIGHTS, DSTA_DEGREE, DSTA_KAPPA,
    )



    dynamics_gf = get_dsta_dynamics_gf(k1, k2, k3)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=DSTA_WEIGHTS,
        degree_m=DSTA_DEGREE,
        kappa=DSTA_KAPPA,
        name="DSTA",
        domain=[(-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 1, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


def create_fc_cta_spec(k1: float = 54.8, k2: float = 27.0, k3: float = 5.7,
                       k4: float = 3.0, k5: float = 2.0,
                       delta0: float = 0.0) -> SystemSpec:
    """
    Create spec for the Fully-Connected Continuous Twisting Algorithm (FC-CTA).

    NOVEL controller: CTA + sign(x₃) self-correcting relay.
        ẋ₁ = x₂
        ẋ₂ = -k₁⌈x₁⌋^{1/3} - k₂⌈x₂⌋^{1/2} + x₃
        ẋ₃ = -k₃·sign(x₁) - k₄·sign(x₂) - k₅·sign(x₃)

    r = (3, 2, 1), κ = -1, m = 5. Stability requires k₃ > k₄ > 0, k₅ > 0.
    """
    from gf_framework.systems.fc_cta import (
        get_fc_cta_dynamics_gf,
        FC_CTA_WEIGHTS, FC_CTA_DEGREE, FC_CTA_KAPPA,
    )



    dynamics_gf = get_fc_cta_dynamics_gf(k1, k2, k3, k4, k5)
    return SystemSpec(
        dynamics_gf=dynamics_gf,
        weights_r=FC_CTA_WEIGHTS,
        degree_m=FC_CTA_DEGREE,
        kappa=FC_CTA_KAPPA,
        name="FC-CTA",
        domain=[(-2, 2), (-2, 2), (-2, 2)],
        dynamics_gf_vertices=(matched_disturbance_vertices(dynamics_gf, 2, delta0)
                              if delta0 > 0 else None),
        disturbance_bound=delta0 if delta0 > 0 else None,
    )


if __name__ == "__main__":
    from .gf_discovery import discover_gf

    print("GF-SOS Lyapunov Discovery — STA smoke test")
    print("=" * 70)
    result = discover_gf(create_sta_spec(), verbose=True)
    print(f"\n{'✓' if result.certified else '✗'} {result.message}")
