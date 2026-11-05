"""Robust certificates: W > 0 for every vertex of the disturbance inclusion.

Ground truth: FOSMC ṡ = -k·sign(s) + δ, |δ| ≤ δ₀ with V = s² gives
V̇ = -2k|s| + 2sδ ≤ -2(k - δ₀)|s|, so the certificate must exist iff δ₀ < k.
The vertex machinery (endpoints ±δ₀; W affine in δ ⇒ convexity covers the
interval) must reproduce exactly that threshold — and reject above it.

MOSEK-free: CLARABEL everywhere.
"""
from fractions import Fraction

import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.discovery.universal_discovery import (
    create_fosmc_spec,
    create_levant4d_spec,
    create_sta_spec,
    matched_disturbance_vertices,
)
from gf_framework.verification.joint_sos import JointSOSOptimizer
from gf_framework.verification.verifier import certify_discovered_gf


def _fosmc_V():
    """V = s² for the 1D FOSMC (C¹: m = 2 > r = 1)."""
    return GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])


# ---------------------------------------------------------------------------
# Vertex construction
# ---------------------------------------------------------------------------

def test_vertices_require_degree_zero_channel():
    """A constant disturbance on a channel of nonzero weighted degree breaks
    homogeneity and must be rejected (admissibility κ + r_ch = 0)."""
    spec = create_sta_spec()
    with pytest.raises(ValueError):
        matched_disturbance_vertices(spec.dynamics_gf, channel=0, delta0=0.1)


def test_vertices_structure():
    spec = create_sta_spec()
    verts = matched_disturbance_vertices(spec.dynamics_gf, channel=1, delta0=0.3)
    assert len(verts) == 2
    for vert, sign in zip(verts, (+1.0, -1.0)):
        const = vert[1].terms[-1]
        assert float(const.coefficient) == pytest.approx(sign * 0.3)
        assert all(e == 0 for e in const.exponents)
        # Channel 0 untouched
        assert vert[0] is spec.dynamics_gf[0]


def test_spec_factory_wires_vertices():
    spec = create_fosmc_spec(k=1.0, delta0=0.5)
    assert spec.dynamics_gf_vertices is not None
    assert len(spec.dynamics_gf_vertices) == 2
    assert spec.disturbance_bound == 0.5
    nominal = create_fosmc_spec(k=1.0)
    assert nominal.dynamics_gf_vertices is None


def test_levant4d_spec_factory_wires_vertices():
    """delta0 plumbing for the 4D differentiator (added for the perturbed-case
    runs): disturbance on channel 3 (ė₄, the weighted-degree-0 sign channel),
    nominal channels untouched, exactly one constant ±δ₀ term added."""
    spec = create_levant4d_spec(delta0=1.0)
    nominal = create_levant4d_spec()
    assert nominal.dynamics_gf_vertices is None
    assert spec.disturbance_bound == 1.0
    assert len(spec.dynamics_gf_vertices) == 2
    for vert, sign in zip(spec.dynamics_gf_vertices, (+1.0, -1.0)):
        for i in range(3):
            assert vert[i] is spec.dynamics_gf[i]
        chan, chan0 = vert[3], spec.dynamics_gf[3]
        assert len(chan.terms) == len(chan0.terms) + 1
        const = chan.terms[-1]
        assert float(const.coefficient) == pytest.approx(sign * 1.0)
        assert all(e == 0 for e in const.exponents)


# ---------------------------------------------------------------------------
# FOSMC analytic threshold: certificate iff δ₀ < k
# ---------------------------------------------------------------------------

def test_fosmc_robust_certifies_below_gain():
    spec = create_fosmc_spec(k=1.0, delta0=0.5)
    cert = certify_discovered_gf(
        _fosmc_V(), (1,), 2, spec.dynamics_gf, solver="CLARABEL",
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
        disturbance_bound=spec.disturbance_bound,
    )
    assert cert.surface_certified, cert.message
    assert cert.n_vertices == 2
    assert cert.disturbance_bound == 0.5
    assert "robust" in cert.message


def test_fosmc_robust_rejected_above_gain():
    """δ₀ = 1.5 > k = 1: V̇ > 0 near the worst-case endpoint — must fail."""
    spec = create_fosmc_spec(k=1.0, delta0=1.5)
    cert = certify_discovered_gf(
        _fosmc_V(), (1,), 2, spec.dynamics_gf, solver="CLARABEL",
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
    )
    assert not cert.surface_certified
    assert cert.W_verified is False


def test_fosmc_nominal_unaffected():
    """No vertices ⇒ identical nominal behavior (n_vertices = 1)."""
    spec = create_fosmc_spec(k=1.0)
    cert = certify_discovered_gf(
        _fosmc_V(), (1,), 2, spec.dynamics_gf, solver="CLARABEL",
    )
    assert cert.surface_certified
    assert cert.n_vertices == 1


# ---------------------------------------------------------------------------
# STA end-to-end: robust design + robust gate
# ---------------------------------------------------------------------------

def test_sta_robust_design_and_gate():
    """The designer with vertex W-blocks must produce a V that the vertex-aware
    gate certifies for a small δ₀ (STA, k₁ = 1.5, k₂ = 1.1, δ₀ = 0.3)."""
    spec = create_sta_spec(delta0=0.3)
    from gf_framework.algebra.enumerate import TermSpec

    # The certified nominal STA structure at m = 3 (diag + the x₁x₂ cross term).
    terms = [
        TermSpec(exponents=(Fraction(3, 2), Fraction(0)), term_types=("abs", "abs")),
        TermSpec(exponents=(Fraction(0), Fraction(3)), term_types=("abs", "abs")),
        TermSpec(exponents=(Fraction(1), Fraction(1)), term_types=("signed", "signed")),
    ]
    opt = JointSOSOptimizer(weights_r=(2, 1), solver="CLARABEL", verbose=False)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=terms,
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
    )
    assert res.feasible, res.message
    cert = certify_discovered_gf(
        res.V_gf, (2, 1), 3, spec.dynamics_gf, solver="CLARABEL",
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
        disturbance_bound=0.3,
    )
    assert cert.surface_certified, cert.message
    assert cert.n_vertices == 2


def test_sta_robust_rejected_for_excessive_disturbance():
    """δ₀ = 2.0 > k₂ = 1.1: the integrator channel cannot dominate — no design
    and no certificate may exist."""
    spec = create_sta_spec(delta0=2.0)
    from gf_framework.algebra.enumerate import TermSpec

    terms = [
        TermSpec(exponents=(Fraction(3, 2), Fraction(0)), term_types=("abs", "abs")),
        TermSpec(exponents=(Fraction(0), Fraction(3)), term_types=("abs", "abs")),
        TermSpec(exponents=(Fraction(1), Fraction(1)), term_types=("signed", "signed")),
    ]
    opt = JointSOSOptimizer(weights_r=(2, 1), solver="CLARABEL", verbose=False)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=terms,
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
    )
    if res.feasible and res.V_gf is not None:
        cert = certify_discovered_gf(
            res.V_gf, (2, 1), 3, spec.dynamics_gf, solver="CLARABEL",
            dynamics_gf_vertices=spec.dynamics_gf_vertices,
        )
        assert not cert.surface_certified, (
            "δ₀ above the STA gain bound must not certify"
        )


# ---------------------------------------------------------------------------
# Convexity premise: W is AFFINE in the matched disturbance δ
# (the two-endpoint design covers the whole interval [-δ₀, +δ₀] only because
# of this affinity — pin it symbolically, not numerically)
# ---------------------------------------------------------------------------

def _w_term_map(V, dyn):
    from gf_framework.algebra.lie_derivative import (
        compute_lie_derivative_gf, negate_gf,
    )
    W = negate_gf(compute_lie_derivative_gf(V, dyn, validate=False))
    return {(t.exponents, t.term_types): t.coefficient for t in W.terms}


def test_w_affine_in_delta():
    """Term-wise: W(+δ₀) + W(−δ₀) = 2·W(0), and the δ-dependent part is exactly
    ∓δ₀·∂V/∂x_ch — i.e. W is affine in δ with slope −∂V/∂x_ch."""
    delta0 = 0.3
    spec = create_sta_spec(delta0=delta0)
    nominal = create_sta_spec()
    # Any GF with x₂-dependence exercises the channel; use a 3-term STA shape.
    from gf_framework.algebra.core import GeneralizedForm
    V = GeneralizedForm(degree_m=3, weights_r=(2, 1), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(3, 2), Fraction(0)),
               term_types=("abs", "abs")),
        GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(3)),
               term_types=("abs", "abs")),
        GFTerm(coefficient=-0.5, exponents=(Fraction(1), Fraction(1)),
               term_types=("signed", "signed")),
    ])
    w_plus = _w_term_map(V, spec.dynamics_gf_vertices[0])
    w_minus = _w_term_map(V, spec.dynamics_gf_vertices[1])
    w_zero = _w_term_map(V, nominal.dynamics_gf)

    # Affinity: the midpoint of the endpoint Lie derivatives is the nominal one.
    keys = set(w_plus) | set(w_minus) | set(w_zero)
    for k in keys:
        mid = (w_plus.get(k, 0.0) + w_minus.get(k, 0.0)) / 2.0
        assert mid == pytest.approx(w_zero.get(k, 0.0), abs=1e-12), k

    # Slope: W(+δ₀) − W(0) = −δ₀·∂V/∂x₂ (channel 1), term-for-term.
    from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf
    one = GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(0)),
                 term_types=("abs", "abs"))
    e_ch = [GeneralizedForm(degree_m=0, weights_r=(2, 1), terms=[]),
            GeneralizedForm(degree_m=0, weights_r=(2, 1), terms=[one])]
    dVdx2 = {(t.exponents, t.term_types): t.coefficient
             for t in compute_lie_derivative_gf(V, e_ch, validate=False).terms}
    for k in set(w_plus) | set(w_zero) | set(dVdx2):
        diff = w_plus.get(k, 0.0) - w_zero.get(k, 0.0)
        assert diff == pytest.approx(-delta0 * dVdx2.get(k, 0.0), abs=1e-12), k


def test_fosmc_midpoint_delta_also_certifies():
    """A δ₀-certified V must also gate-certify at every smaller bound (here the
    midpoint δ₀/2) — the interval-coverage consequence of affinity+convexity."""
    V = _fosmc_V()
    for d0 in (0.5, 0.25):
        spec = create_fosmc_spec(k=1.0, delta0=d0)
        cert = certify_discovered_gf(
            V, (1,), 2, spec.dynamics_gf, solver="CLARABEL",
            dynamics_gf_vertices=spec.dynamics_gf_vertices,
            disturbance_bound=d0,
        )
        assert cert.surface_certified, f"failed at δ₀={d0}: {cert.message}"
