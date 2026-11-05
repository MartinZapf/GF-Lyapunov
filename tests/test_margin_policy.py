"""WP P2-13/P2-14: the acceptance-threshold policy shared by the screen and X.

P2-13 added the ``margin_policy`` vocabulary with ``"half"`` as the default;
P2-14 (owner decision D-l, 2026-09-04) flipped every default to
``"achieved"``. The tests below therefore pin ``"achieved"`` as the default of
every entry point and keep pinning that ``"half"`` still behaves exactly as it
did when it was the default.


Every test here is MOSEK-free and solver-free: the Grams are injected by hand
with a residual computed in closed form, so what is pinned is the THRESHOLD
arithmetic and its provenance, never a solver's behaviour.

The instance used throughout is the smallest one that exhibits the whole
mechanism. One variable, ``d = 2``, basis ``[(1,)]``, target ``{y²: 1}`` and a
Gram ``[[q]]``. Then

    zᵀQz = q·y²,   E = (1 − q)·y²,   ‖E‖₁ = |1 − q|,
    C₁ = Σ_β 2β₀/d = 1,   η̂ = 0 for q ≥ 0,   charge = |1 − q| .

Choosing ``q = 1 − 3ε/4`` puts the charge strictly between ε/2 and ε: refused
under ``"half"``, accepted under ``"achieved"`` with certified margin ε/4.
That is exactly the band the measured CTA m = 5 (ε = 1e-7) and levant4d E8
32-term (ε = 1e-6) cases fell into.
"""
from fractions import Fraction

import numpy as np
import pytest

from gf_framework.verification.design_screen import (
    BUDGET_POLICIES,
    OptimisticResidualModelWarning,
    resolve_budget_policy,
    screen_design_candidate,
    screen_gram_bundle,
)
from gf_framework.verification.exact_gate import (
    MARGIN_POLICIES,
    MARGIN_POLICY_ACHIEVED,
    MARGIN_POLICY_HALF,
    ExactGateError,
    exact_validate_octant,
    exact_verify_captures,
    margin_budget,
    resolve_margin_policy,
    validate_one_capture,
)
from gf_framework.verification.joint_sos import (
    JointSOSOptimizer,
    _validate_gram_certificate,
)

EPS = 1e-6
#: q such that the charge |1 − q| lands strictly inside (ε/2, ε).
Q_IN_BAND = 1.0 - 0.75 * EPS
#: q such that the charge lands under ε/2 (both policies accept).
Q_CLEAN = 1.0 - 0.25 * EPS
#: q such that the charge exceeds ε (both policies refuse).
Q_OVER = 1.0 - 1.5 * EPS


def _octant(q: float, eps: float = EPS):
    """The one-variable instance above, as (target, d, gram, basis, eps)."""
    return (
        {(2,): Fraction(1)},
        2,
        np.array([[q]], dtype=float),
        [(1,)],
        Fraction(float(eps)),
    )


def _exact_charge(q: float) -> Fraction:
    """‖E‖₁ of the instance, in the same exact arithmetic the gate uses."""
    return abs(Fraction(1) - Fraction(float(q)))


#: The same instance as a FORM: |x|² at μ = (1,), whose exact octant target
#: is ``(1 − ε)·y²`` because the gate subtracts the margin itself. So a Gram
#: ``[[q]]`` charges ``|1 − ε − q|``, and ``_form_q(f)`` puts that at ``f·ε``.
FORM_TERMS = [(Fraction(1), (Fraction(2),), ("abs",))]


def _form_q(fraction_of_eps: float, eps: float = EPS) -> float:
    return 1.0 - eps - fraction_of_eps * eps


def _capture(q: float, signs=(1,), eps: float = EPS) -> dict:
    """One Gram capture of the instance, in the shape the gate consumes."""
    return {
        "signs": tuple(signs),
        "octant_index": 0 if signs == (1,) else 1,
        "eps": float(eps),
        "mu": (1,),
        "y_degree": 2,
        "target": {(2,): 1.0 - eps},
        "basis": [(1,)],
        "gram": np.array([[q]], dtype=float),
    }


def _bundle_entry(q: float, name: str = "V_oct0", signs=(1,),
                  eps: float = EPS) -> dict:
    entry = _capture(q, signs=signs, eps=eps)
    entry.update({"block": "V", "name": name, "endpoint": None,
                  "dim": 1, "eps_reason": ""})
    return entry


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

def test_the_two_policy_vocabularies_are_the_same_strings():
    # The screen only works as a predictor of the gate while the two name the
    # same policies; a divergence here is the bug the WP was written about.
    assert tuple(MARGIN_POLICIES) == tuple(BUDGET_POLICIES) == ("half", "achieved")


def test_an_unknown_margin_policy_raises_and_never_falls_back_to_half():
    with pytest.raises(ExactGateError):
        resolve_margin_policy("loose")
    with pytest.raises(ExactGateError):
        margin_budget(Fraction(1, 10), "loose")


def test_an_unknown_screen_budget_policy_raises():
    with pytest.raises(ValueError):
        resolve_budget_policy("absolute")
    with pytest.raises(ValueError):
        JointSOSOptimizer([1.0], design_screen_budget="absolute")


def test_the_budget_is_eps_over_two_under_half_and_eps_under_achieved():
    eps = Fraction(1, 1000)
    assert margin_budget(eps, MARGIN_POLICY_HALF) == eps / 2
    assert margin_budget(eps, MARGIN_POLICY_ACHIEVED) == eps


# ---------------------------------------------------------------------------
# The exact gate
# ---------------------------------------------------------------------------

def test_achieved_is_the_default_of_every_entry_point():
    """WP P2-14: the DEFAULT is now ``achieved`` — budget ε, strict."""
    check = exact_validate_octant(*_octant(Q_CLEAN))
    assert check.ok and check.margin_policy == MARGIN_POLICY_ACHIEVED
    assert check.budget_float == pytest.approx(EPS)
    cap = exact_validate_octant(*_octant(Q_IN_BAND))
    assert cap.ok, "the default must now PROVE the in-band charge"
    assert cap.eps_certified_float == pytest.approx(EPS / 4, rel=1e-9)
    # ... and asking for the historical policy still refuses it.
    half = exact_validate_octant(*_octant(Q_IN_BAND),
                                 margin_policy=MARGIN_POLICY_HALF)
    assert not half.ok and half.budget_float == pytest.approx(EPS / 2)


def test_the_policy_free_ratio_names_eps_as_its_denominator():
    """WP P2-14: ``charge_over_eps`` is the same number under both policies."""
    achieved = exact_validate_octant(*_octant(Q_IN_BAND))
    half = exact_validate_octant(*_octant(Q_IN_BAND),
                                 margin_policy=MARGIN_POLICY_HALF)
    assert achieved.charge_over_eps == pytest.approx(0.75, rel=1e-9)
    assert half.charge_over_eps == pytest.approx(0.75, rel=1e-9)
    # while charge/budget halves with the policy, which is why the field name
    # has to say which denominator it used.
    assert achieved.charge_float / achieved.budget_float == \
        pytest.approx(0.75, rel=1e-9)
    assert half.charge_float / half.budget_float == pytest.approx(1.5, rel=1e-9)


def test_the_note_half_margin_flag_is_reported_and_never_enforced():
    """The frozen note prints ``charge ≤ ε/2``; P2-14 reports it, never gates
    on it. An accepted octant may report False, and that is the whole point of
    recording it."""
    in_band = exact_validate_octant(*_octant(Q_IN_BAND))
    assert in_band.ok and in_band.note_half_margin_holds is False
    clean = exact_validate_octant(*_octant(Q_CLEAN))
    assert clean.ok and clean.note_half_margin_holds is True


def test_a_charge_in_the_band_is_refused_by_half_and_proved_by_achieved():
    half = exact_validate_octant(*_octant(Q_IN_BAND),
                                 margin_policy=MARGIN_POLICY_HALF)
    achieved = exact_validate_octant(*_octant(Q_IN_BAND),
                                     margin_policy=MARGIN_POLICY_ACHIEVED)
    assert not half.ok and half.reason == "budget_exceeded"
    assert achieved.ok and achieved.reason == ""
    # Same Gram, same residual, same η̂: only the threshold moved.
    assert half.resid_l1_float == achieved.resid_l1_float
    assert half.charge_float == achieved.charge_float
    assert achieved.eta_hat == "0"


def test_the_certified_margin_is_eps_minus_charge_exactly_under_both_policies():
    eps = Fraction(float(EPS))
    expected = eps - _exact_charge(Q_IN_BAND)
    for policy in MARGIN_POLICIES:
        check = exact_validate_octant(*_octant(Q_IN_BAND), margin_policy=policy)
        assert Fraction(check.eps_certified) == expected
        assert check.eps_float == pytest.approx(EPS)
    # And it is a quarter of ε for this q, by construction.
    assert float(expected) == pytest.approx(EPS / 4, rel=1e-9)


def test_a_charge_over_eps_is_refused_by_both_policies():
    for policy in MARGIN_POLICIES:
        check = exact_validate_octant(*_octant(Q_OVER), margin_policy=policy)
        assert not check.ok and check.reason == "budget_exceeded"
        # The refusal still records how far short it fell: negative margin.
        assert float(Fraction(check.eps_certified)) < 0.0


def test_achieved_is_strict_so_a_charge_of_exactly_eps_is_refused():
    # charge == ε exactly: certified margin would be 0, i.e. non-strict
    # positivity, which is not what the certificate claims. ε and q are
    # dyadic here so the charge is ε on the nose in ℚ, not near it.
    eps = 2.0 ** -20
    q = 1.0 - eps
    assert _exact_charge(q) == Fraction(float(eps))
    check = exact_validate_octant(*_octant(q, eps),
                                  margin_policy=MARGIN_POLICY_ACHIEVED)
    assert not check.ok and check.reason == "budget_exceeded"
    assert Fraction(check.eps_certified) == 0
    # ... and the same charge under "half" is refused as well.
    assert not exact_validate_octant(*_octant(q, eps)).ok


def test_half_accepts_a_strict_subset_of_what_achieved_accepts():
    # ε > 0 makes ε/2 < ε, so charge ≤ ε/2 implies charge < ε. Pinned over a
    # sweep so a future edit to either branch cannot invert the containment.
    for k in range(1, 40):
        q = 1.0 - k * EPS / 20.0
        half = exact_validate_octant(*_octant(q), margin_policy=MARGIN_POLICY_HALF)
        achieved = exact_validate_octant(*_octant(q),
                                         margin_policy=MARGIN_POLICY_ACHIEVED)
        assert not (half.ok and not achieved.ok), f"containment broken at k={k}"


def test_a_non_psd_gram_is_refused_under_achieved_too():
    # The policy moves the threshold, never the PSD test: a negative Gram
    # needs η̂ > 0, and here η̂·C₁ alone blows any budget.
    check = exact_validate_octant(
        {(2,): Fraction(1)}, 2, np.array([[-1.0]]), [(1,)],
        Fraction(float(EPS)), (1,), MARGIN_POLICY_ACHIEVED)
    assert not check.ok


# ---------------------------------------------------------------------------
# Whole-form aggregation
# ---------------------------------------------------------------------------

def test_the_worker_takes_the_policy_positionally_so_it_pickles_with_the_item():
    # exact_verify_captures hands the mapper (terms, cap, n_vars, policy)
    # tuples and the pool calls fn(*item); a keyword-only policy would be
    # silently dropped into a worker's default.
    check, failed = validate_one_capture(
        FORM_TERMS, _capture(_form_q(0.75)), 1, MARGIN_POLICY_ACHIEVED)
    assert failed is False
    assert check.margin_policy == MARGIN_POLICY_ACHIEVED and check.ok
    half, _ = validate_one_capture(FORM_TERMS, _capture(_form_q(0.75)), 1,
                                   MARGIN_POLICY_HALF)
    assert not half.ok
    # and the DEFAULT of the worker is "achieved" since P2-14.
    default, _ = validate_one_capture(FORM_TERMS, _capture(_form_q(0.75)), 1)
    assert default.ok and default.margin_policy == MARGIN_POLICY_ACHIEVED


def test_the_form_level_verdict_and_roll_ups_follow_the_policy():
    q = _form_q(0.75)
    caps = [_capture(q, signs=(1,)), _capture(q, signs=(-1,))]
    half = exact_verify_captures(FORM_TERMS, caps, 1,
                                 margin_policy=MARGIN_POLICY_HALF)
    achieved = exact_verify_captures(FORM_TERMS, caps, 1)
    assert half.margin_policy == MARGIN_POLICY_HALF and not half.ok
    assert achieved.margin_policy == MARGIN_POLICY_ACHIEVED and achieved.ok
    # The certified margin is policy-free; the charge RATIO is not, because
    # its denominator is the policy's own budget.
    assert achieved.eps_certified_float == pytest.approx(EPS / 4, rel=1e-9)
    assert achieved.worst_charge_ratio == pytest.approx(0.75, rel=1e-9)
    assert half.worst_charge_ratio == pytest.approx(1.5, rel=1e-9)
    # WP P2-14: the policy-free roll-up and the reported note inequality.
    assert achieved.worst_charge_over_eps == pytest.approx(0.75, rel=1e-9)
    assert half.worst_charge_over_eps == pytest.approx(0.75, rel=1e-9)
    assert achieved.note_half_margin_holds is False
    assert half.note_half_margin_holds is False


def test_the_octant_coverage_and_epsilon_rules_are_untouched_by_the_policy():
    short = exact_verify_captures(FORM_TERMS, [_capture(_form_q(0.25))], 1,
                                  margin_policy=MARGIN_POLICY_ACHIEVED)
    assert not short.ok and short.reason.startswith("octant_coverage_incomplete")
    zero = [_capture(_form_q(0.25), signs=(1,), eps=0.0),
            _capture(_form_q(0.25), signs=(-1,), eps=0.0)]
    res = exact_verify_captures(FORM_TERMS, zero, 1,
                                margin_policy=MARGIN_POLICY_ACHIEVED)
    assert not res.ok and res.reason.startswith("epsilon_not_positive")


# ---------------------------------------------------------------------------
# Proposition 6 (float) — the design screen
# ---------------------------------------------------------------------------

def test_validate_gram_certificate_moves_the_same_threshold():
    target = {(2,): 1.0}
    gram = np.array([[Q_IN_BAND]])
    half = _validate_gram_certificate(target, gram, [(1,)], 1, EPS, 1e-7,
                                      budget_policy="half")
    achieved = _validate_gram_certificate(target, gram, [(1,)], 1, EPS, 1e-7)
    assert half[0] is False and "gram_margin_exceeded" in half[3]
    assert achieved[0] is True and achieved[3] == ""


def test_validate_gram_certificate_reports_its_numbers_through_diag():
    diag: dict = {}
    _validate_gram_certificate({(2,): 1.0}, np.array([[Q_IN_BAND]]), [(1,)],
                               1, EPS, 1e-7, budget_policy="achieved",
                               diag=diag)
    assert diag["eta"] == 0.0 and diag["c1"] == pytest.approx(1.0)
    assert diag["budget"] == pytest.approx(EPS)
    assert diag["charge"] == pytest.approx(0.75 * EPS, rel=1e-6)
    assert diag["eps_certified"] == pytest.approx(0.25 * EPS, rel=1e-6)


def test_validate_gram_certificate_rejects_an_unknown_policy():
    with pytest.raises(ValueError):
        _validate_gram_certificate({(2,): 1.0}, np.array([[1.0]]), [(1,)], 1,
                                   EPS, 1e-7, budget_policy="eps_over_ten")


def test_the_screen_refuses_or_passes_the_same_bundle_by_policy():
    entries = [_bundle_entry(_form_q(0.75), "V_oct0", (1,)),
               _bundle_entry(_form_q(0.75), "V_oct1", (-1,))]
    half = screen_design_candidate(None, entries, run_sphere=False,
                                   budget_policy="half")
    achieved = screen_design_candidate(None, entries, run_sphere=False)
    assert not half.ok and half.reason == "gram_validation_failed"
    assert achieved.ok and achieved.n_blocks_failed == 0
    assert half.budget_policy == "half" and achieved.budget_policy == "achieved"
    assert achieved.eps_certified == pytest.approx(0.25 * EPS, rel=1e-6)


def test_the_screen_record_carries_the_policy_and_the_numbers_per_block():
    blocks = screen_gram_bundle([_bundle_entry(_form_q(0.75))],
                                budget_policy="achieved")
    rec = blocks[0].as_record()
    assert rec["budget_policy"] == "achieved"
    assert rec["budget"] == pytest.approx(EPS)
    assert rec["charge"] == pytest.approx(0.75 * EPS, rel=1e-6)
    assert rec["eps_certified"] == pytest.approx(0.25 * EPS, rel=1e-6)
    # And the record is still JSON-serializable end to end.
    import json
    json.dumps(screen_design_candidate(None, [_bundle_entry(_form_q(0.75))],
                                       run_sphere=False).as_record())


def test_the_screen_default_is_achieved_and_the_optimizer_stores_the_flag():
    """WP P2-14: the screen's default moved with the gate's, in lock step —
    a screen running a different policy than the gate is the one thing P2-13
    measured as strictly harmful."""
    assert screen_gram_bundle(
        [_bundle_entry(_form_q(0.75))])[0].budget_policy == "achieved"
    assert JointSOSOptimizer([1.0]).design_screen_budget == "achieved"
    assert JointSOSOptimizer(
        [1.0], design_screen_budget="half").design_screen_budget == "half"


def test_run_exact_phase_refuses_an_unknown_policy_before_it_solves_anything():
    from gf_framework.verification.exact_phase import run_exact_phase
    with pytest.raises(ExactGateError):
        run_exact_phase(None, [], 1, margin_policy="whatever_passes")


def test_the_screen_and_the_exact_gate_agree_on_the_same_bundle_per_policy():
    # The property the whole design rests on: run both under ONE policy and
    # the cheap float filter predicts the exact verdict on this instance.
    for fraction in (0.25, 0.75, 1.5):
        q = _form_q(fraction)
        for policy in MARGIN_POLICIES:
            screen = screen_design_candidate(
                None, [_bundle_entry(q, "V_oct0", (1,)),
                       _bundle_entry(q, "V_oct1", (-1,))],
                run_sphere=False, budget_policy=policy)
            exact = exact_verify_captures(
                FORM_TERMS, [_capture(q, (1,)), _capture(q, (-1,))], 1,
                margin_policy=policy)
            assert screen.ok == exact.ok, (fraction, policy)


# ---------------------------------------------------------------------------
# The residual model (WP P2-13, the second half)
# ---------------------------------------------------------------------------

def _cross_class_instance():
    """A 2-variable instance whose Gram carries CROSS-parity-class mass.

    Target ``y0^4 + y1^4`` (all-even, so the deck group is the full F2^2),
    basis ``[(2,0), (1,1), (0,2)]``. The classes of the basis exponents under
    H = F2^2 are (0,0), (1,1), (0,0), so index 1 sits alone: ``Q[0,1]`` and
    ``Q[1,2]`` are CROSS-class and the Reynolds projection deletes them. They
    land on the monomials ``y0^3 y1`` and ``y0 y1^3``, which the invariant
    target cannot contain, so the projection removes exactly that residual.
    """
    target = {(4, 0): Fraction(1), (2, 2): Fraction(0), (0, 4): Fraction(1)}
    target = {k: v for k, v in target.items() if v != 0}
    gram = np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)
    basis = [(2, 0), (1, 1), (0, 2)]
    return target, 4, gram, basis


def test_the_projected_residual_is_never_larger_than_the_dense_one():
    target, d, gram, basis = _cross_class_instance()
    # Put mass on the cross-class entry (0,1): it contributes only to y0^3 y1.
    gram = gram.copy()
    gram[0, 1] = gram[1, 0] = 1e-3
    tgt = {k: float(v) for k, v in target.items()}
    dense: dict = {}
    blocked: dict = {}
    _validate_gram_certificate(tgt, gram, basis, 2, 1e-2, 1e-7,
                               residual_model="dense", diag=dense)
    with pytest.warns(OptimisticResidualModelWarning):
        _validate_gram_certificate(tgt, gram, basis, 2, 1e-2, 1e-7,
                                   residual_model="deck_blocked", diag=blocked)
    assert dense["resid_l1_dense"] == blocked["resid_l1_dense"]
    assert blocked["resid_l1"] < dense["resid_l1"]
    # The deleted mass is exactly the two cross entries, 2 x 1e-3.
    assert dense["resid_l1"] - blocked["resid_l1"] == pytest.approx(2e-3)


def test_the_dense_screen_reproduces_the_exact_gates_own_residual():
    # Since XVC_RESIDUAL_2026-09-25 (fix (a)) the exact gate charges the DENSE
    # residual (the portable verifier's rule); "dense" predicts it. This
    # replaces the pre-fix pin "deck_blocked == exact gate", which encoded the
    # projected accounting X no longer uses.
    target, d, gram, basis = _cross_class_instance()
    gram = gram.copy()
    gram[0, 1] = gram[1, 0] = 1e-3
    check = exact_validate_octant(target, d, gram, basis, Fraction(1, 100))
    assert check.n_blocks == 2          # the blocking still runs
    dense: dict = {}
    _validate_gram_certificate({k: float(v) for k, v in target.items()}, gram,
                               basis, 2, 1e-2, 1e-7,
                               residual_model="dense", diag=dense)
    assert dense["resid_l1"] == pytest.approx(check.resid_l1_float, rel=1e-12)
    # The exact dense residual is the two cross entries, 2 x 1e-3 (as floats).
    assert Fraction(check.resid_l1_float) == 2 * Fraction(1e-3)


def test_the_projected_screen_is_optimistic_against_the_exact_gate():
    # "deck_blocked" charges strictly LESS than X on a cross-class Gram, by
    # exactly the cross-class mass, and warns loudly when selected.
    target, d, gram, basis = _cross_class_instance()
    gram = gram.copy()
    gram[0, 1] = gram[1, 0] = 1e-3
    check = exact_validate_octant(target, d, gram, basis, Fraction(1, 100))
    blocked: dict = {}
    with pytest.warns(OptimisticResidualModelWarning):
        _validate_gram_certificate({k: float(v) for k, v in target.items()},
                                   gram, basis, 2, 1e-2, 1e-7,
                                   residual_model="deck_blocked", diag=blocked)
    assert blocked["resid_l1"] < check.resid_l1_float
    assert check.resid_l1_float - blocked["resid_l1"] == pytest.approx(2e-3)


def test_the_dense_residual_model_is_the_default_everywhere():
    import warnings
    from gf_framework.verification.design_screen import RESIDUAL_MODELS
    assert RESIDUAL_MODELS == ("dense", "deck_blocked")
    with warnings.catch_warnings():
        # the default path must stay silent
        warnings.simplefilter("error", OptimisticResidualModelWarning)
        assert screen_gram_bundle(
            [_bundle_entry(_form_q(0.75))])[0].residual_model == "dense"
        assert JointSOSOptimizer([1.0]).design_screen_residual == "dense"
    with pytest.warns(OptimisticResidualModelWarning):
        assert JointSOSOptimizer(
            [1.0], design_screen_residual="deck_blocked"
        ).design_screen_residual == "deck_blocked"


def test_an_unknown_residual_model_raises_at_every_entry_point():
    from gf_framework.verification.design_screen import resolve_residual_model
    with pytest.raises(ValueError):
        resolve_residual_model("projected")
    with pytest.raises(ValueError):
        JointSOSOptimizer([1.0], design_screen_residual="projected")
    with pytest.raises(ValueError):
        _validate_gram_certificate({(2,): 1.0}, np.array([[1.0]]), [(1,)], 1,
                                   EPS, 1e-7, residual_model="projected")


def test_the_block_record_reports_both_residuals():
    entry = _bundle_entry(_form_q(0.75))
    with pytest.warns(OptimisticResidualModelWarning):
        rec = screen_gram_bundle([entry],
                                 residual_model="deck_blocked")[0].as_record()
    assert rec["residual_model"] == "deck_blocked"
    assert rec["gram_residual_l1_dense"] == pytest.approx(0.75 * EPS, rel=1e-6)


# ---------------------------------------------------------------------------
# WP P2-14: the persisted V-proof cache key carries the policy
# ---------------------------------------------------------------------------
#
# The flip makes "achieved" the default, so the SEPARATION rule stops being an
# opt-in detail and becomes the thing that keeps a looser proof out of a
# stricter run's cache. It lives in one exported function.


def test_the_cache_key_separates_the_policies_and_leaves_half_alone(tmp_path):
    from gf_framework.verification.exact_phase import (
        DiskVProofCache,
        vproof_cache_for_margin_policy,
    )
    cache = DiskVProofCache(tmp_path, library="L")

    # "half" is the historical key: the same object, no extra component, so
    # every entry written before the policy existed is still addressable.
    half = vproof_cache_for_margin_policy(cache, MARGIN_POLICY_HALF)
    assert half is cache
    assert half.extra == {}

    # "achieved" keys under its own name and never mutates the caller's cache.
    achieved = vproof_cache_for_margin_policy(cache, MARGIN_POLICY_ACHIEVED)
    assert achieved is not cache
    assert achieved.extra == {"margin_policy": MARGIN_POLICY_ACHIEVED}
    assert cache.extra == {}
    assert achieved.root == cache.root and achieved.library == cache.library

    assert vproof_cache_for_margin_policy(None, MARGIN_POLICY_ACHIEVED) is None
    with pytest.raises(ExactGateError):
        vproof_cache_for_margin_policy(cache, "nonesuch")


def test_a_proof_accepted_under_achieved_is_unreachable_from_a_half_run(tmp_path):
    # Same V, same epsilon, same n_vars: only the policy differs, and that is
    # enough for the two runs to address different files.
    from gf_framework.algebra.core import GeneralizedForm, GFTerm
    from gf_framework.verification.exact_phase import (
        DiskVProofCache,
        vproof_cache_for_margin_policy,
    )
    V = GeneralizedForm(degree_m=2, weights_r=(1,),
                        terms=[GFTerm(1.0, (Fraction(2),), ("abs",))])
    cache = DiskVProofCache(tmp_path, library="L")
    k_half = vproof_cache_for_margin_policy(
        cache, MARGIN_POLICY_HALF).key_for(V, EPS, 1)
    k_ach = vproof_cache_for_margin_policy(
        cache, MARGIN_POLICY_ACHIEVED).key_for(V, EPS, 1)
    assert k_half != k_ach
    assert cache.key_for(V, EPS, 1) == k_half
