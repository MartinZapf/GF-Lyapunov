"""WP P5-02: the V scale normalisation paired with the design-epsilon sweep.

Every test here is MOSEK-free and solver-free. What is pinned is the
ARITHMETIC and its soundness argument:

  * the factor is an exact power of two, so scaling introduces no rounding and
    ``Fraction(c*x) == c*Fraction(x)``;
  * V, its design Gram bundle and its design margin scale TOGETHER, so the
    bundle still witnesses the V the gate is about to see and the exact
    phase's margin equality check still holds;
  * the exact Proposition-7 charge/budget ratio is INVARIANT under the
    rescaling -- the certificate claim does not move, only the units of the
    margin do;
  * nothing is scaled while the design's coefficients sit inside the measured
    band, and a no-op returns the SAME objects.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.interfaces import VerificationConfig
from gf_framework.verification.exact_gate import (
    MARGIN_POLICY_HALF,
    exact_validate_octant,
)
from gf_framework.verification.joint_sos import gram_bundle_entry_hash
from gf_framework.verification.v_normalization import (
    MAX_LOG2_FACTOR,
    NORMALIZATION_BAND_LOG2,
    NORMALIZATION_REASONS,
    normalize_for_gate,
    plan_normalization,
    pure_power_indices,
    scale_gf,
    scale_gram_bundle,
    scale_statistic,
)


def _gf(*coefficients, cross=()):
    """A 2-variable r-homogeneous GF: pure powers plus optional cross terms.

    ``coefficients`` are the pure powers ``a_i * |x_i|^2``; ``cross`` are
    ``(coefficient,)`` entries for the mixed term ``|x_0| * |x_1|``.
    """
    terms = []
    for i, a in enumerate(coefficients):
        exps = [Fraction(0), Fraction(0)]
        exps[i] = Fraction(2)
        terms.append(GFTerm(coefficient=float(a), exponents=tuple(exps),
                            term_types=("abs", "abs")))
    for a in cross:
        terms.append(GFTerm(coefficient=float(a),
                            exponents=(Fraction(1), Fraction(1)),
                            term_types=("abs", "abs")))
    return GeneralizedForm(degree_m=2, weights_r=(1, 1), terms=terms)


# ---------------------------------------------------------------------------
# The statistic and the band
# ---------------------------------------------------------------------------

def test_pure_power_indices_are_the_terms_with_one_nonzero_exponent():
    V = _gf(1.0, 2.0, cross=(0.5,))
    assert pure_power_indices(V) == [0, 1]


def test_the_statistic_is_the_geometric_mean_of_the_two_ends():
    # Lower end: the smallest PURE POWER (what the gate's absolute margin has
    # to fit under). Upper end: the largest coefficient of any kind (what the
    # gate's absolute residual budget tracks).
    V = _gf(1e-6, 1.0, cross=(100.0,))
    assert scale_statistic(V) == pytest.approx(math.sqrt(1e-6 * 100.0))


def test_a_form_with_no_pure_power_term_has_no_statistic():
    V = GeneralizedForm(degree_m=2, weights_r=(1, 1), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(1), Fraction(1)),
               term_types=("abs", "abs"))])
    assert math.isnan(scale_statistic(V))
    assert plan_normalization(V).reason == "no_pure_power_terms"


def test_the_band_is_wide_enough_for_every_shipped_design_point():
    # The twelve nominal rows at eps = 1e-4 have a log-centre between
    # 2**-6.09 (pidsmc) and 2**+0.70 (example1); the widest sweep rung that
    # certified is 2**-13.02 (continuous_twisting m=6 at eps=1e-8). The band
    # must contain all of them, or the shipped path would start rescaling
    # certificates the measured scale window says are already well placed.
    assert NORMALIZATION_BAND_LOG2 >= 14
    for centre_log2 in (-6.09, 0.70, -13.02, -1.18, -9.82, -2.67):
        V = _gf(2.0 ** (2 * centre_log2), 1.0)   # log-centre = 2**centre_log2
        assert plan_normalization(V).applied is False


def test_the_gate_reference_epsilon_is_where_the_absolute_thresholds_come_from():
    # The mechanism the module documents: the gate seeds its descending ladder
    # from this config default and charges the residual against eps/2. If this
    # moves, the scale window moves and the band is a re-measurement.
    assert VerificationConfig().epsilon == 1e-6


@pytest.mark.parametrize("reason", NORMALIZATION_REASONS)
def test_every_declared_reason_is_a_string(reason):
    assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# The factor
# ---------------------------------------------------------------------------

def test_a_drifted_design_is_scaled_by_an_exact_power_of_two():
    V = _gf(2.0 ** -60, 1.0)                      # log-centre 2**-30
    plan = plan_normalization(V, epsilon=1e-8)
    assert plan.applied is True
    assert plan.reason == ""
    assert plan.factor == 2.0 ** plan.log2_factor
    assert math.log2(plan.factor).is_integer()
    assert plan.log2_factor == 30
    # ... and it lands the log-centre on 1.
    assert plan.statistic_after == pytest.approx(1.0, rel=1e-12)


def test_the_margin_is_scaled_with_the_form_and_never_alone():
    V = _gf(2.0 ** -60, 1.0)
    plan = plan_normalization(V, epsilon=1e-8)
    assert plan.epsilon == 1e-8
    assert plan.epsilon_scaled == 1e-8 * plan.factor


def test_an_absurd_factor_is_refused_rather_than_applied():
    V = _gf(2.0 ** -60, 1.0)
    plan = plan_normalization(V, band_log2=1, max_log2=4)
    assert plan.applied is False
    assert plan.reason == "factor_out_of_range"
    assert plan.factor == 1.0


def test_a_form_far_ABOVE_the_band_is_scaled_down():
    # The mechanism is symmetric: too large a coefficient blows the gate's
    # absolute residual budget exactly as too small a one falls under its
    # absolute margin.
    V = _gf(2.0 ** 40, 2.0 ** 60)                 # log-centre 2**50
    plan = plan_normalization(V)
    assert plan.applied is True
    assert plan.log2_factor == -50
    assert plan.factor == 2.0 ** -50


def test_non_finite_coefficients_are_refused_not_scaled():
    V = _gf(float("nan"), 1.0)
    assert plan_normalization(V).reason == "non_finite_coefficients"
    V = _gf(0.0, 1.0)
    assert plan_normalization(V).reason == "non_positive_statistic"


def test_composite_terms_are_refused_because_a_flat_multiply_is_not_a_rescaling():
    V = _gf(1e-40, 1.0)
    V.composite_terms = ["not-a-real-composite-term"]
    assert plan_normalization(V).reason == "composite_terms"
    with pytest.raises(ValueError):
        scale_gf(V, 2.0)


# ---------------------------------------------------------------------------
# Scaling the form
# ---------------------------------------------------------------------------

def test_a_unit_factor_returns_the_same_object_so_a_no_op_is_literal():
    # gf_discovery._adopt_design_bundle compares ``V is V_proposal``; a copy
    # here would silently drop every design witness.
    V = _gf(1.0, 1.0)
    assert scale_gf(V, 1.0) is V
    out, bundle, eps, rec = normalize_for_gate(V, epsilon=1e-4)
    assert out is V and rec.applied is False and rec.reason == "within_band"
    assert eps == 1e-4 and bundle is None


def test_scaling_preserves_the_support_the_degree_and_the_term_types():
    V = _gf(3.0, 5.0, cross=(-7.0,))
    W = scale_gf(V, 2.0 ** 8)
    assert W is not V
    assert W.degree_m == V.degree_m and W.weights_r == V.weights_r
    assert [t.exponents for t in W.terms] == [t.exponents for t in V.terms]
    assert [t.term_types for t in W.terms] == [t.term_types for t in V.terms]
    assert V.terms[0].coefficient == 3.0            # the input is untouched


def test_scaling_by_a_power_of_two_is_exact_in_float_and_in_Q():
    # This is why the factor is a power of two: the exact-rational objects the
    # X stage rebuilds are EXACTLY c times the unscaled ones, so the bundle's
    # margin equality check (a Fraction comparison) still holds.
    V = _gf(0.1, 1.0 / 3.0, cross=(1e-9,))
    c = 2.0 ** 17
    W = scale_gf(V, c)
    for a, b in zip(V.terms, W.terms):
        assert Fraction(b.coefficient) == Fraction(c) * Fraction(a.coefficient)
    assert Fraction(1e-8 * c) == Fraction(c) * Fraction(1e-8)


# ---------------------------------------------------------------------------
# Scaling the bundle
# ---------------------------------------------------------------------------

def _entry(q=1.0, eps=1e-6, target=None):
    entry = {
        "schema": "gf.design_grams/1", "block": "V", "name": "V_oct0",
        "signs": (1,), "octant_index": 0, "endpoint": None,
        "parity_class": None, "eps": float(eps), "eps_reason": "",
        "mu": (1,), "y_degree": 2, "clearing_power": 0,
        "coefficient_scale": 1.0, "dim": 1, "dtype": "float64",
        "basis": [(1,)],
        "target": dict(target if target is not None else {(2,): 1.0 - eps}),
        "gram": np.array([[q]], dtype=float),
    }
    entry["hash"] = gram_bundle_entry_hash(entry)
    return entry


def test_the_bundle_scales_gram_target_and_margin_and_nothing_else():
    c = 2.0 ** 12
    before = _entry()
    after = scale_gram_bundle([before], c)[0]
    assert after["gram"][0, 0] == before["gram"][0, 0] * c
    assert after["target"][(2,)] == before["target"][(2,)] * c
    assert after["eps"] == before["eps"] * c
    for key in ("basis", "mu", "signs", "y_degree", "clearing_power", "dim",
                "block", "name", "octant_index", "eps_reason"):
        assert after[key] == before[key]
    # The input is not mutated.
    assert before["eps"] == 1e-6


def test_a_zero_margin_block_keeps_a_zero_margin():
    # Scaling must not manufacture a margin where the design proved none: the
    # typed P2-02 zero-eps blocks stay unwitnessable.
    entry = _entry(eps=0.0, target={(2,): 1.0})
    entry["eps_reason"] = "w_degree_not_positive_even"
    after = scale_gram_bundle([entry], 2.0 ** 20)[0]
    assert after["eps"] == 0.0
    assert after["eps_reason"] == "w_degree_not_positive_even"


def test_the_scaled_entry_hash_is_recomputed_and_self_consistent():
    after = scale_gram_bundle([_entry()], 2.0 ** 6)[0]
    stored = after.pop("hash")
    assert stored == gram_bundle_entry_hash(after)
    assert stored != _entry()["hash"]        # a different proof object


def test_scaling_an_absent_bundle_is_none_not_an_empty_list():
    assert scale_gram_bundle(None, 2.0) is None


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_factor_is_refused_everywhere(bad):
    with pytest.raises(ValueError):
        scale_gf(_gf(1.0, 1.0), bad)
    with pytest.raises(ValueError):
        scale_gram_bundle([_entry()], bad)


# ---------------------------------------------------------------------------
# The soundness claim, in the exact arithmetic itself
# ---------------------------------------------------------------------------

#: The P2-13 instance: one variable, d = 2, basis [(1,)], target
#: ``(1-eps)*y^2``, Gram ``[[q]]``.  With ``q = 1 - eps - f*eps`` the exact
#: charge is ``f*eps``, so ``f = 1/4`` is accepted (charge under eps/2) and
#: ``f = 3/2`` is refused by both policies.
def _q(f, eps=1e-6):
    return 1.0 - eps - f * eps


def _octant_check(c, q=None, eps=1e-6, policy=MARGIN_POLICY_HALF):
    """Proposition 7 on the one-variable instance, scaled by ``c``.

    target = (1-eps)*y^2 scaled by c, Gram = [[q]] scaled by c, margin eps*c.
    """
    if q is None:
        q = _q(0.25, eps)
    cf = Fraction(float(c))
    target = {(2,): cf * (Fraction(1) - Fraction(float(eps)))}
    return exact_validate_octant(
        target, 2, np.array([[q]], dtype=float) * float(c), [(1,)],
        cf * Fraction(float(eps)), signs=(1,), margin_policy=policy)


def test_the_exact_charge_and_budget_both_scale_by_c():
    base = _octant_check(1.0)
    up = _octant_check(2.0 ** 20)
    assert base.ok and up.ok
    # Exactly, in binary: a power-of-two factor moves the exponent and leaves
    # the mantissa alone, so these are equalities and not tolerances.
    assert Fraction(up.charge_float) == Fraction(2 ** 20) * Fraction(base.charge_float)
    assert Fraction(up.budget_float) == Fraction(2 ** 20) * Fraction(base.budget_float)
    assert Fraction(up.resid_l1_float) == Fraction(2 ** 20) * Fraction(base.resid_l1_float)


def test_the_exact_charge_over_budget_ratio_is_invariant_under_rescaling():
    # THE soundness statement of this module: rescaling V by a positive
    # constant proves the same theorem with the same headroom. The margin
    # number moves (it is not scale-invariant), the verdict and the ratio do
    # not.
    ratios = []
    for k in (-20, -8, 0, 8, 20):
        chk = _octant_check(2.0 ** k)
        assert chk.ok
        ratios.append(Fraction(chk.charge_float) / Fraction(chk.budget_float))
    assert len(set(ratios)) == 1


def test_the_certified_margin_scales_with_c_and_is_not_scale_invariant():
    base = _octant_check(1.0)
    up = _octant_check(2.0 ** 20)
    assert Fraction(up.eps_certified) == Fraction(2 ** 20) * Fraction(base.eps_certified)
    assert Fraction(up.eps_certified) != Fraction(base.eps_certified)


def test_a_refusal_stays_a_refusal_under_rescaling():
    # Rescaling cannot rescue a Gram that does not support the margin, which
    # is the other half of "the claim does not move".
    q_over = _q(1.5)               # charge above eps, refused by both policies
    for k in (-20, 0, 20):
        assert _octant_check(2.0 ** k, q=q_over).ok is False


def test_max_log2_factor_is_inside_the_float64_exponent_range():
    # A factor near the exponent limit would overflow the Gram rather than fix
    # anything; 2**120 squared is still finite, 2**600 is not.
    assert 0 < MAX_LOG2_FACTOR < 500
    assert math.isfinite(2.0 ** MAX_LOG2_FACTOR)
