"""Exact-rational gain injection + bracketing (REVIEW_GATE_20260711 F1).

Pins the repair of finding F1: the exact gate historically read every dynamics
coefficient as Fraction(float(x)) — the dyadic float64 image — so certified
gains were stand-ins ~1e-16 away from the published rationals (11/10 is not a
dyadic). These tests pin:

  1. Fraction coefficients pass through exact_terms_from_gf untouched;
  2. matched_disturbance_vertices preserves an exact Fraction δ₀;
  3. decimal_bracket constructs exactly-verified rational brackets of the
     irrational standard-tuning gains;
  4. end-to-end: run_exact_phase on a Fraction-gain system produces exact
     W coefficients in the gain's arithmetic (denominator 10, not 2^52) and
     retains the per-octant Gram captures for --dump-grams persistence.
"""
from fractions import Fraction
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.discovery.universal_discovery import (
    create_fosmc_spec,
    matched_disturbance_vertices,
)
from gf_framework.verification.exact_gate import (
    exact_neg_lie_derivative,
    exact_terms_from_gf,
)
from gf_framework.verification.exact_phase import run_exact_phase

from exact_gate_standard_gains import GAIN_SPECS, decimal_bracket


def test_fraction_coefficient_pass_through():
    """An exact Fraction coefficient must NOT be routed through float()."""
    gf = GeneralizedForm(degree_m=0, weights_r=(1,), terms=[
        GFTerm(coefficient=Fraction(11, 10), exponents=(Fraction(0),),
               term_types=("abs",)),
    ])
    (coeff, _, _), = exact_terms_from_gf(gf)
    assert coeff == Fraction(11, 10)
    assert coeff != Fraction(1.1)  # the dyadic float64 image differs


def test_float_coefficient_unchanged_encoding():
    """Float coefficients keep the historical dyadic encoding, bit for bit."""
    gf = GeneralizedForm(degree_m=0, weights_r=(1,), terms=[
        GFTerm(coefficient=1.1, exponents=(Fraction(0),), term_types=("abs",)),
    ])
    (coeff, _, _), = exact_terms_from_gf(gf)
    assert coeff == Fraction(1.1) and coeff != Fraction(11, 10)


def test_matched_vertices_exact_delta0():
    spec = create_fosmc_spec(k=1.0, delta0=Fraction(3, 10))
    for vertex, sign in zip(spec.dynamics_gf_vertices, (+1, -1)):
        extra = vertex[0].terms[-1]
        assert isinstance(extra.coefficient, Fraction)
        assert extra.coefficient == sign * Fraction(3, 10)
        assert extra.coefficient != sign * 0.3  # dyadic float64(0.3) < 3/10
    # float path unchanged
    spec_f = create_fosmc_spec(k=1.0, delta0=0.3)
    assert spec_f.dynamics_gf_vertices[0][0].terms[-1].coefficient == 0.3


@pytest.mark.parametrize("k,q,approx", [
    (2, Fraction(9, 2), 1.5 * math.sqrt(2.0)),           # (3/2)√2
    (2, Fraction(27, 4), 1.5 * math.sqrt(3.0)),          # (3/2)√3
    (3, Fraction(72), 2.0 * 3.0 ** (2.0 / 3.0)),         # 2·3^{2/3}
    (6, Fraction(6561, 8), 1.5 * (2.0 * 3.0 ** (2.0 / 3.0)) ** 0.5),
    # levant5d (P4-01): λ₂ = 3·5^{3/4}, λ₃ = 2·λ₂^{2/3}, λ₄ = (3/2)·√λ₃
    (4, Fraction(10125), 3.0 * 5.0 ** (3.0 / 4.0)),
    (6, Fraction(648000), 2.0 * (3.0 * 5.0 ** (3.0 / 4.0)) ** (2.0 / 3.0)),
    (12, Fraction(5380840125, 64),
     1.5 * (2.0 * (3.0 * 5.0 ** (3.0 / 4.0)) ** (2.0 / 3.0)) ** 0.5),
])
def test_decimal_bracket_exact(k, q, approx):
    lo, hi = decimal_bracket(k, q, approx)
    assert lo > 0
    assert lo ** k < q < hi ** k            # exact, in ℚ — brackets the TRUE gain
    assert hi - lo <= Fraction(3, 10 ** 15)  # tight
    # The float64 image may sit up to ~half an ulp OUTSIDE the decimal bracket
    # (it approximates the true value, which is inside) — only proximity holds.
    assert abs(float(lo) - approx) < 1e-12 and abs(float(hi) - approx) < 1e-12


def test_gain_specs_power_targets():
    """The GAIN_SPECS algebra: each (k, q, approx) is self-consistent.

    The tolerance is relative, not absolute: approx**k is a float, so its
    error scales with k and with |q|. levant5d's λ₄ has k = 12 and
    q = 5380840125/64 ≈ 8.4e7, where an absolute 1e-9 is smaller than one ulp
    of q. 1e-12 relative leaves about three orders of magnitude of headroom
    over the observed error on every registered gain (worst measured: 1.1e-15
    relative, λ₄).
    """
    for system, cfg in GAIN_SPECS.items():
        for name, (k, q, approx) in cfg["irrational"].items():
            tol = max(1e-9, 1e-12 * float(q))
            assert abs(approx ** k - float(q)) <= tol, (system, name)


def test_exact_phase_fraction_gains_end_to_end():
    """FOSMC with k = 11/10 injected exactly, δ₀ = 1: the exact W coefficients
    live in the gain's arithmetic (denominators dividing 10), the phase
    passes, and the Gram captures are retained for persistence."""
    spec = create_fosmc_spec(k=Fraction(11, 10), delta0=Fraction(1))
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])
    # Exact algebra check first (no solver): W₊ = -∇V·f₊ coefficients are
    # exact combinations of 11/10 and 1 — denominator 10 — not 2^52 dyadics.
    v_terms = exact_terms_from_gf(V)
    for dyn in spec.dynamics_gf_vertices:
        f_terms = [exact_terms_from_gf(f_i) for f_i in dyn]
        w_terms = exact_neg_lie_derivative(v_terms, f_terms)
        assert w_terms, "W must be non-empty"
        for coeff, _, _ in w_terms:
            assert coeff.denominator in (1, 2, 5, 10), coeff

    res = run_exact_phase(V, spec.dynamics_gf_vertices, 1,
                          epsilon=1e-6, solver="CLARABEL")
    assert res.exact_pass
    # Gram retention (REVIEW_GATE M4 / --dump-grams)
    assert res.v_grams, "V Gram captures must be retained"
    for cap in res.v_grams:
        assert {"signs", "basis", "gram"} <= set(cap)
    assert all(ep.grams for ep in res.endpoints)
