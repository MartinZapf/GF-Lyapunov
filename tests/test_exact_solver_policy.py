"""Witness policy inside the exact phase (WP P2-05, decision D1).

Three witness sources, in order: the DESIGN Gram bundle, a MOSEK re-solve,
and EXACTLY ONE retry with CLARABEL. A source is left behind when it does not
prove the certificate or when it proves it with a worst exact charge/budget
ratio above 0.1 (within a factor 10 of the Proposition-7 cliff).

What is pinned here:

* ``run_exact_phase`` defaults to ``solver="MOSEK"`` with ``CLARABEL`` as the
  one retry;
* a clean primary witness costs exactly ONE attempt (no speculative retry);
* a deliberately degraded MOSEK witness triggers EXACTLY ONE recorded retry,
  under both triggers: an unusable witness (``ok == False``) and a usable one
  with too little headroom (charge ratio > 0.1);
* the retry is never repeated: a second degraded witness ends the ladder at
  two attempts and the better VERIFIED witness is adopted;
* every attempt is recorded (source, solver, ok, worst charge ratio, typed
  shortfall) and the adopted one is named by ``witness_solver`` /
  ``witness_source`` / ``witness_charge_ratio``;
* adoption is never a trust decision about a solver: the adopted witness is
  adopted because the exact rational phase verified it.

The tests are MOSEK-free by construction: the "MOSEK" arm is a stub that runs
the real CLARABEL solve and then degrades the Gram it returns, so the policy
is exercised identically on a box with or without MOSEK.
"""
import inspect
from fractions import Fraction

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.discovery.universal_discovery import create_fosmc_spec
from gf_framework.verification import exact_phase as ep
from gf_framework.verification.exact_phase import (
    BUNDLE_FALLTHROUGH_REASONS,
    WITNESS_CHARGE_TRIGGER,
    WITNESS_DESIGN_BUNDLE,
    WITNESS_FALLBACK_SOLVER,
    WITNESS_PRIMARY_SOLVER,
    WITNESS_RESOLVE,
    run_exact_phase,
)

EPS = 1e-6
#: Proposition 7's budget is ε/2, so a residual of 2e-7 against ε = 1e-6 is a
#: charge ratio of about 0.4: above the 0.1 trigger, below the 1.0 cliff.
DEGRADE_RESIDUAL = 2e-7


def _fosmc():
    """δ₀ = 0.5 < k = 1: the smallest system that certifies (n = 1, 2 endpoints)."""
    spec = create_fosmc_spec(k=1.0, delta0=0.5)
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])
    return V, spec.dynamics_gf_vertices


def _stub_solver(monkeypatch, degrade, *, degraded="MOSEK", base="CLARABEL"):
    """Route ``degraded`` through a real ``base`` solve, then damage the Gram.

    Returns the list of solver names the exact phase asked for, in order, so a
    test can count how many re-solves each arm actually cost.
    """
    real = ep.verify_gf_sos_full
    asked = []

    def stub(form, **kwargs):
        solver = kwargs.get("solver")
        asked.append(solver)
        result = real(form, **{**kwargs, "solver": base})
        if solver == degraded:
            degrade(result)
        return result

    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    return asked


def _degrade_unusable(result):
    """The witness the solver proposed is gone: nothing can be verified."""
    result.feasible = False
    result.gram_certificates = None


def _degrade_headroom(result, residual=DEGRADE_RESIDUAL):
    """A VALID but sloppy witness: still PSD, charged against the margin.

    Adding ``residual`` to a diagonal Gram entry perturbs zᵀQz by exactly that
    amount on one monomial, so the exact residual ‖E‖₁ grows by ``residual``
    while the Gram stays positive semidefinite. Nothing about the proof is
    weakened: the exact gate charges the whole discrepancy.
    """
    for capture in result.gram_certificates or []:
        gram = np.array(capture["gram"], dtype=np.float64)
        gram[0, 0] += residual
        capture["gram"] = gram


# ---------------------------------------------------------------------------
# 1. The defaults and the ladder
# ---------------------------------------------------------------------------

def test_default_solver_is_mosek_with_one_clarabel_retry():
    params = inspect.signature(run_exact_phase).parameters
    assert params["solver"].default == "MOSEK" == WITNESS_PRIMARY_SOLVER
    assert params["fallback_solver"].default == "CLARABEL" == WITNESS_FALLBACK_SOLVER
    assert params["charge_headroom_trigger"].default == 0.1 == WITNESS_CHARGE_TRIGGER


def test_witness_plan_is_bundle_then_mosek_then_one_clarabel():
    plan = ep._witness_plan("MOSEK", "CLARABEL", True)
    assert plan == [(WITNESS_DESIGN_BUNDLE, ""),
                    (WITNESS_RESOLVE, "MOSEK"),
                    (WITNESS_RESOLVE, "CLARABEL")]
    assert ep._witness_plan("MOSEK", "CLARABEL", False) == [
        (WITNESS_RESOLVE, "MOSEK"), (WITNESS_RESOLVE, "CLARABEL")]


def test_a_caller_pinning_clarabel_keeps_exactly_one_attempt():
    """The retry is dropped when it would repeat the primary."""
    assert ep._witness_plan("CLARABEL", "CLARABEL", False) == [
        (WITNESS_RESOLVE, "CLARABEL")]
    assert ep._witness_plan("MOSEK", None, False) == [(WITNESS_RESOLVE, "MOSEK")]


def test_fallthrough_reasons_are_the_two_design_side_limits():
    assert BUNDLE_FALLTHROUGH_REASONS == {"gram_bundle_margin_missing",
                                          "gram_bundle_clearing_active"}


# ---------------------------------------------------------------------------
# 2. A clean primary witness costs one attempt
# ---------------------------------------------------------------------------

def test_clean_primary_witness_is_adopted_without_a_retry(monkeypatch):
    V, dyn = _fosmc()
    asked = _stub_solver(monkeypatch, lambda r: None)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)

    assert phase.exact_pass
    assert len(phase.witness_attempts) == 1
    only = phase.witness_attempts[0]
    assert only["source"] == WITNESS_RESOLVE
    assert only["solver"] == "MOSEK"
    assert only["ok"] is True and only["adopted"] is True and only["shortfall"] == ""
    assert phase.witness_solver == "MOSEK"
    assert phase.witness_source == WITNESS_RESOLVE
    assert phase.witness_charge_ratio == phase.worst_charge_ratio
    assert phase.witness_charge_ratio <= WITNESS_CHARGE_TRIGGER
    # V once + one W per endpoint, and nothing more: no speculative retry.
    assert asked == ["MOSEK", "MOSEK", "MOSEK"]


# ---------------------------------------------------------------------------
# 3. The WP acceptance: a degraded MOSEK witness triggers EXACTLY one retry
# ---------------------------------------------------------------------------

def test_degraded_mosek_witness_triggers_exactly_one_recorded_retry(monkeypatch):
    """Trigger 1: the MOSEK witness is unusable (``ExactCaptureResult.ok`` false)."""
    V, dyn = _fosmc()
    asked = _stub_solver(monkeypatch, _degrade_unusable)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)

    assert len(phase.witness_attempts) == 2, phase.witness_attempts
    first, second = phase.witness_attempts
    assert first["index"] == 0 and first["solver"] == "MOSEK"
    assert first["ok"] is False and first["adopted"] is False
    assert first["shortfall"] == "v_resolve_infeasible"
    assert second["index"] == 1 and second["solver"] == "CLARABEL"
    assert second["ok"] is True and second["adopted"] is True
    assert second["shortfall"] == ""
    # Exactly one retry: the MOSEK arm runs to completion (V and both
    # endpoints, unchanged from the historical single-arm behaviour), then one
    # CLARABEL arm. No third arm.
    assert asked == ["MOSEK"] * 3 + ["CLARABEL"] * 3

    assert phase.exact_pass
    assert phase.witness_solver == "CLARABEL"
    assert phase.witness_source == WITNESS_RESOLVE
    assert phase.witness_charge_ratio == phase.worst_charge_ratio


def test_low_headroom_mosek_witness_triggers_exactly_one_recorded_retry(monkeypatch):
    """Trigger 2: the MOSEK witness PROVES it, but within a factor 10 of the
    acceptance cliff. WP P2-14 re-expressed the ratio against ε (it was
    against the margin policy's own budget), so the typed shortfall names
    ``charge_over_eps`` and the trigger no longer moves with the policy."""
    V, dyn = _fosmc()
    asked = _stub_solver(monkeypatch, _degrade_headroom)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)

    assert len(phase.witness_attempts) == 2, phase.witness_attempts
    first, second = phase.witness_attempts
    assert first["solver"] == "MOSEK"
    # The degraded witness is a VALID proof; it is retried for headroom only.
    assert first["ok"] is True and first["adopted"] is False
    assert first["worst_charge_over_eps"] > WITNESS_CHARGE_TRIGGER
    assert first["shortfall"].startswith("charge_over_eps_")
    assert first["shortfall"].endswith("_above_0.1")
    # The two ratios coincide under the default "achieved" policy (budget = ε)
    # and would differ by a factor 2 under "half" — which is exactly why the
    # trigger reads the one whose name states its denominator.
    assert first["worst_charge_over_eps"] == first["worst_charge_ratio"]
    assert second["solver"] == "CLARABEL" and second["adopted"] is True
    assert second["worst_charge_ratio"] < first["worst_charge_ratio"]
    assert asked.count("MOSEK") == 3 and asked.count("CLARABEL") == 3
    assert len(asked) == 6

    assert phase.exact_pass
    assert phase.witness_solver == "CLARABEL"
    assert phase.witness_charge_ratio <= WITNESS_CHARGE_TRIGGER


def test_the_retry_happens_once_even_when_the_fallback_is_also_degraded(monkeypatch):
    """Both witnesses degraded: two attempts, no third, best verified adopted."""
    V, dyn = _fosmc()

    real = ep.verify_gf_sos_full
    asked = []

    def stub(form, **kwargs):
        solver = kwargs.get("solver")
        asked.append(solver)
        result = real(form, **{**kwargs, "solver": "CLARABEL"})
        # MOSEK is charged twice as much as the fallback, so the fallback is
        # the better of two witnesses that both trip the trigger.
        _degrade_headroom(result, 4e-7 if solver == "MOSEK" else 2e-7)
        return result

    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)

    assert len(phase.witness_attempts) == 2
    first, second = phase.witness_attempts
    assert first["ok"] is True and second["ok"] is True
    assert first["shortfall"] and second["shortfall"]
    assert second["adopted"] is True and first["adopted"] is False
    assert phase.witness_solver == "CLARABEL"
    assert phase.witness_charge_ratio == second["worst_charge_ratio"]
    assert asked == ["MOSEK"] * 3 + ["CLARABEL"] * 3


def test_the_better_verified_witness_is_adopted_when_both_trip_the_trigger(monkeypatch):
    """Order does not decide: the SMALLER exact charge does."""
    V, dyn = _fosmc()

    real = ep.verify_gf_sos_full
    asked = []

    def stub(form, **kwargs):
        solver = kwargs.get("solver")
        asked.append(solver)
        result = real(form, **{**kwargs, "solver": "CLARABEL"})
        _degrade_headroom(result, 2e-7 if solver == "MOSEK" else 4e-7)
        return result

    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)

    first, second = phase.witness_attempts
    assert first["adopted"] is True and second["adopted"] is False
    assert phase.witness_solver == "MOSEK"
    assert phase.witness_charge_ratio == first["worst_charge_ratio"]
    assert first["worst_charge_ratio"] < second["worst_charge_ratio"]


def test_no_retry_when_the_fallback_is_disabled(monkeypatch):
    V, dyn = _fosmc()
    asked = _stub_solver(monkeypatch, _degrade_headroom)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS, fallback_solver=None)

    assert len(phase.witness_attempts) == 1
    only = phase.witness_attempts[0]
    assert only["solver"] == "MOSEK" and only["adopted"] is True
    assert only["shortfall"].startswith("charge_over_eps_")
    assert "CLARABEL" not in asked
    # A trigger that cannot be acted on still leaves an honest record: the
    # adopted witness is the low-headroom one and the record says so.
    assert phase.witness_charge_ratio > WITNESS_CHARGE_TRIGGER
    assert phase.exact_pass


def test_a_failing_certificate_records_both_attempts_and_still_fails(monkeypatch):
    """δ₀ = 1.5 > k = 1: no witness can prove it. The ladder does not hide that."""
    spec = create_fosmc_spec(k=1.0, delta0=1.5)
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])
    phase = run_exact_phase(V, spec.dynamics_gf_vertices, 1, epsilon=EPS,
                            solver="CLARABEL", fallback_solver="SCS")

    assert not phase.exact_pass
    assert len(phase.witness_attempts) == 2
    assert [a["solver"] for a in phase.witness_attempts] == ["CLARABEL", "SCS"]
    assert all(a["ok"] is False for a in phase.witness_attempts)
    assert phase.witness_attempts[-1]["adopted"] is True


# ---------------------------------------------------------------------------
# 4. The record
# ---------------------------------------------------------------------------

def test_witness_record_names_the_adopted_witness_and_every_attempt(monkeypatch):
    V, dyn = _fosmc()
    _stub_solver(monkeypatch, _degrade_unusable)
    record = run_exact_phase(V, dyn, 1, epsilon=EPS).witness_record()

    # P2-03's key and semantics are unchanged; P2-05's names are additive.
    assert record["source"] == record["witness_source"] == WITNESS_RESOLVE
    assert record["witness_solver"] == "CLARABEL"
    assert isinstance(record["witness_charge_ratio"], float)
    assert record["n_witness_attempts"] == len(record["witness_attempts"]) == 2
    assert sum(1 for a in record["witness_attempts"] if a["adopted"]) == 1
    # "solver_used" was added by P3-05 (P2-10's follow-up): the ENCODING that
    # actually ran, beside the solver that was REQUESTED. Additive.
    for attempt in record["witness_attempts"]:
        # "worst_charge_over_eps" was added by P2-14 beside the
        # policy-relative "worst_charge_ratio". Additive, same discipline.
        assert set(attempt) == {"index", "source", "solver", "solver_used",
                                "ok", "worst_charge_ratio",
                                "worst_charge_over_eps", "shortfall",
                                "detail", "adopted", "time_s"}


def test_the_record_is_json_serializable(monkeypatch):
    import json

    V, dyn = _fosmc()
    _stub_solver(monkeypatch, _degrade_headroom)
    record = run_exact_phase(V, dyn, 1, epsilon=EPS).witness_record()
    assert json.loads(json.dumps(record))["witness_solver"] == "CLARABEL"


def test_endpoints_carry_the_adopted_solver(monkeypatch):
    V, dyn = _fosmc()
    _stub_solver(monkeypatch, _degrade_unusable)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)
    assert [e.witness_solver for e in phase.endpoints] == ["CLARABEL"] * 2


# ---------------------------------------------------------------------------
# 4b. The design bundle is the FIRST witness source
# ---------------------------------------------------------------------------

def test_an_accepted_bundle_is_adopted_as_one_attempt_and_no_solve_runs(monkeypatch):
    """Source 1 of 3. The bundle wins outright; sources 2 and 3 never run."""
    pytest.importorskip("gf_framework.verification.mosek_direct")
    from gf_framework.algebra.enumerate import (
        enumerate_cross_terms,
        enumerate_diagonal_terms,
    )
    from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
    from gf_framework.discovery.universal_discovery import create_sta_spec
    from gf_framework.verification.joint_sos import JointSOSOptimizer
    from gf_framework.verification.mosek_direct import MOSEK_AVAILABLE

    if not MOSEK_AVAILABLE:
        pytest.skip("MOSEK not installed")

    design_eps = 1e-4
    spec = create_sta_spec()
    diag = enumerate_diagonal_terms(3, spec.weights_r, c1_only=True)
    cross = enumerate_cross_terms(3, spec.weights_r, max_denominator=4,
                                  include_signed=True, minimal=False,
                                  include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    terms = list(diag) + [t for t in cross if mu_compatible(t, mu)]
    res = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                            allow_solver_fallback=False, epsilon=design_eps
                            ).find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=terms,
        dynamics_gf_vertices=spec.dynamics_gf_vertices)
    assert res.feasible and res.gram_certificates

    def no_solve(*_a, **_k):
        raise AssertionError("no float solve may run once the bundle is adopted")

    monkeypatch.setattr(ep, "verify_gf_sos_full", no_solve)
    phase = run_exact_phase(res.V_gf, [spec.dynamics_gf], 2,
                            epsilon=design_eps, gram_bundle=res)
    assert phase.exact_pass
    assert len(phase.witness_attempts) == 1
    assert phase.witness_attempts[0]["source"] == WITNESS_DESIGN_BUNDLE
    assert phase.witness_attempts[0]["adopted"] is True
    assert phase.witness_source == WITNESS_DESIGN_BUNDLE
    assert phase.witness_solver == ep.DESIGN_BUNDLE_SOLVER == "MOSEK_DIRECT"
    assert phase.witness_charge_ratio == phase.worst_charge_ratio


# ---------------------------------------------------------------------------
# 5. Soundness guards
# ---------------------------------------------------------------------------

def test_a_retry_never_turns_a_refutation_into_a_pass(monkeypatch):
    """The fallback proposes a Gram; it cannot manufacture a certificate.

    ``k = 1, δ₀ = 1.5`` has W(+) = 2|s| − 3s, negative for s > 0. Every arm of
    the ladder must fail on it, whatever the solvers report.
    """
    spec = create_fosmc_spec(k=1.0, delta0=1.5)
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])

    real = ep.verify_gf_sos_full

    def lying_stub(form, **kwargs):
        """A solver that claims feasibility on everything it is handed."""
        result = real(form, **{**kwargs, "solver": "CLARABEL"})
        result.feasible = True
        return result

    monkeypatch.setattr(ep, "verify_gf_sos_full", lying_stub)
    phase = run_exact_phase(V, spec.dynamics_gf_vertices, 1, epsilon=EPS)
    assert not phase.exact_pass


def test_an_exception_in_the_primary_arm_is_typed_and_retried(monkeypatch):
    V, dyn = _fosmc()
    real = ep.verify_gf_sos_full

    def stub(form, **kwargs):
        if kwargs.get("solver") == "MOSEK":
            raise MemoryError("simulated MOSEK allocation failure")
        return real(form, **kwargs)

    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS)

    assert phase.witness_attempts[0]["shortfall"] == "oom"
    assert "simulated MOSEK" in phase.witness_attempts[0]["detail"]
    assert phase.witness_solver == "CLARABEL" and phase.exact_pass


def test_an_exception_in_the_last_arm_propagates(monkeypatch):
    """Nothing proved anything: a silent None would be a false negative."""
    V, dyn = _fosmc()

    def stub(form, **kwargs):
        raise RuntimeError("both arms down")

    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    with pytest.raises(RuntimeError, match="both arms down"):
        run_exact_phase(V, dyn, 1, epsilon=EPS)


def test_a_reused_v_half_does_not_cause_a_retry(monkeypatch):
    """R2 dedup: a retry cannot re-solve a cached V, so it must not trigger one."""
    V, dyn = _fosmc()
    _stub_solver(monkeypatch, lambda r: None)
    first = run_exact_phase(V, dyn, 1, epsilon=EPS)
    cache = ep.build_v_cache(V, EPS, first)
    # Pretend corner 0 adopted a V with no headroom left: the cached exact
    # result is what corner 1 reuses, and corner 1 cannot improve it.
    for octant in cache.exact.octants:
        octant.charge_float = 0.9 * octant.budget_float

    asked = _stub_solver(monkeypatch, lambda r: None)
    phase = run_exact_phase(V, dyn, 1, epsilon=EPS, v_cache=cache)
    assert phase.v_reused
    assert phase.worst_charge_ratio > WITNESS_CHARGE_TRIGGER
    assert len(phase.witness_attempts) == 1
    assert asked == ["MOSEK", "MOSEK"]  # both endpoints, V not re-solved
