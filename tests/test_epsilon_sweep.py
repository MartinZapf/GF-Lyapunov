"""WP P5-02: the design-epsilon sweep in the degree-escalation loop.

Two objects are pinned here.

* ``JointSOSOptimizer.iter_designs_over_epsilons`` — the ladder generator. Its
  contract is that it is LAZY (a caller that gates the first rung pays for one
  solve), that it restores ``self.epsilon`` whatever happens, and that a rung
  whose design was refused by the design screen still offers its candidate with
  a typed reason instead of vanishing.
* ``discovery.gf_discovery._design_rungs`` / ``discover_gf(epsilon_ladder=...)``
  — the wiring. Its contract is that with no ladder the loop is the historical
  one, call for call.

Most tests are solver-free: the designer is replaced by a stub that records the
margin it was asked for. The few that need a real design are marked
``needs_mosek``.
"""
import json

import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from fractions import Fraction

from gf_framework.discovery import gf_discovery as gd
from gf_framework.verification.joint_sos import (
    DESIGN_EPSILON_LADDER,
    EpsilonRung,
    JointSOSOptimizer,
    JointSOSResult,
    get_design_epsilon_sweep_default,
    resolve_epsilon_ladder,
    set_design_epsilon_sweep_default,
)

try:  # the same gate the rest of the suite uses
    import mosek  # noqa: F401
    HAVE_MOSEK = True
except Exception:  # pragma: no cover
    HAVE_MOSEK = False

needs_mosek = pytest.mark.skipif(not HAVE_MOSEK, reason="MOSEK not installed")


def _V(coeff=1.0):
    """A one-variable pure power, enough to carry an identity through."""
    return GeneralizedForm(
        degree_m=2, weights_r=(1,),
        terms=[GFTerm(coefficient=coeff, exponents=(Fraction(2),),
                      term_types=("abs",))])


class _StubOptimizer(JointSOSOptimizer):
    """A JointSOSOptimizer whose design step is a scripted lookup.

    ``script`` maps a design margin to the ``JointSOSResult`` to return, so a
    test can state exactly which rung is feasible without a solver.
    """

    def __init__(self, script, **kw):
        super().__init__(weights_r=(1,), **kw)
        self._script = script
        self.calls = []

    def find_lyapunov_with_dynamics(self, **kwargs):  # type: ignore[override]
        self.calls.append(self.epsilon)
        outcome = self._script(self.epsilon)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _feasible(V=None, entries=None):
    return JointSOSResult(feasible=True, V_gf=V if V is not None else _V(),
                          solver_status="standard_sos:MOSEK_DIRECT",
                          gram_certificates=entries)


def _infeasible():
    return JointSOSResult(feasible=False, V_gf=None,
                          solver_status="infeasible:MOSEK_DIRECT")


def _screen_refused(V=None):
    return JointSOSResult(feasible=False, V_gf=V if V is not None else _V(),
                          solver_status="design_screen_refused:MOSEK_DIRECT",
                          message="gram_validation_failed(block=V_oct0,...)",
                          gram_certificates=None)


# ---------------------------------------------------------------------------
# The ladder vocabulary
# ---------------------------------------------------------------------------

def test_the_shipped_ladder_starts_at_the_shipped_design_margin():
    # Rung 0 must be the historical margin, otherwise a swept run is not a
    # superset of an unswept one and the acceptance comparison is meaningless.
    assert DESIGN_EPSILON_LADDER[0] == 1e-4
    assert DESIGN_EPSILON_LADDER == (1e-4, 1e-6, 1e-8)


def test_resolve_epsilon_ladder_keeps_order_and_drops_duplicates():
    assert resolve_epsilon_ladder([1e-4, 1e-6, 1e-4, 1e-8]) == (1e-4, 1e-6, 1e-8)


@pytest.mark.parametrize("bad", [[0.0], [-1e-6], [float("inf")], [float("nan")], []])
def test_a_non_positive_or_empty_design_margin_ladder_raises(bad):
    # A zero design margin is not a margin: the exact gate would have budget
    # zero and prove only non-strict positivity.
    with pytest.raises(ValueError):
        resolve_epsilon_ladder(bad)


def test_the_process_default_ships_off_and_round_trips():
    assert get_design_epsilon_sweep_default() is None
    previous = set_design_epsilon_sweep_default(DESIGN_EPSILON_LADDER)
    try:
        assert previous is None
        assert get_design_epsilon_sweep_default() == DESIGN_EPSILON_LADDER
    finally:
        set_design_epsilon_sweep_default(previous)
    assert get_design_epsilon_sweep_default() is None


def test_the_rung_record_is_json_serializable():
    rung = EpsilonRung(index=1, epsilon=1e-6, epsilon_effective=1e-6,
                       solver_status="unknown:MOSEK_DIRECT")
    json.dumps(rung.as_record())


# ---------------------------------------------------------------------------
# The generator's contract
# ---------------------------------------------------------------------------

def test_the_ladder_is_lazy_so_gating_the_first_rung_costs_one_solve():
    opt = _StubOptimizer(lambda eps: _feasible())
    it = opt.iter_designs_over_epsilons(DESIGN_EPSILON_LADDER)
    first = next(it)
    it.close()
    assert first.epsilon == 1e-4
    assert opt.calls == [1e-4]


def test_every_rung_is_solved_when_none_produces_a_candidate():
    opt = _StubOptimizer(lambda eps: _infeasible())
    rungs = list(opt.iter_designs_over_epsilons(DESIGN_EPSILON_LADDER))
    assert opt.calls == [1e-4, 1e-6, 1e-8]
    assert [r.epsilon for r in rungs] == [1e-4, 1e-6, 1e-8]
    assert all(r.V_gf is None and not r.feasible for r in rungs)


def test_the_optimizer_margin_is_restored_after_the_ladder():
    opt = _StubOptimizer(lambda eps: _infeasible(), epsilon=1e-4)
    list(opt.iter_designs_over_epsilons([1e-6, 1e-8]))
    assert opt.epsilon == 1e-4


def test_the_optimizer_margin_is_restored_when_the_caller_abandons_the_ladder():
    opt = _StubOptimizer(lambda eps: _feasible(), epsilon=1e-4)
    it = opt.iter_designs_over_epsilons([1e-6, 1e-8])
    next(it)
    assert opt.epsilon == 1e-6      # inside the ladder
    it.close()                       # generator closed -> finally runs
    assert opt.epsilon == 1e-4


def test_a_design_crash_is_typed_as_a_crash_and_never_as_an_sos_negative():
    # The rule ``_design_full_library`` states: an unexpected exception can
    # silently poison a publication-grade "not certified" claim.
    def script(eps):
        return RuntimeError("backend exploded") if eps == 1e-6 else _infeasible()

    opt = _StubOptimizer(script)
    rungs = list(opt.iter_designs_over_epsilons(DESIGN_EPSILON_LADDER))
    crashed = [r for r in rungs if r.solver_status == "design_error"]
    assert len(crashed) == 1
    assert "backend exploded" in crashed[0].message
    assert "RuntimeError" in crashed[0].traceback
    # ... and the ladder kept going.
    assert opt.calls == [1e-4, 1e-6, 1e-8]


def test_a_screen_refused_rung_still_offers_its_candidate_without_a_bundle():
    opt = _StubOptimizer(lambda eps: _screen_refused())
    rung = next(iter(opt.iter_designs_over_epsilons([1e-8])))
    assert rung.screen_refused is True
    assert rung.feasible is False
    assert rung.V_gf is not None                    # the candidate survives
    assert rung.gram_certificates is None           # the bundle does not
    assert rung.bundle_reason.startswith("design_screen_refused:")
    assert "gram_validation_failed" in rung.bundle_reason


def test_stop_on_candidate_ends_the_ladder_at_the_first_proposal():
    def script(eps):
        return _feasible() if eps == 1e-6 else _infeasible()

    opt = _StubOptimizer(script)
    rungs = list(opt.iter_designs_over_epsilons(DESIGN_EPSILON_LADDER,
                                                stop_on_candidate=True))
    assert [r.epsilon for r in rungs] == [1e-4, 1e-6]
    assert opt.calls == [1e-4, 1e-6]


def test_normalisation_off_leaves_the_candidate_object_identical():
    V = _V(1e-9)                      # far under the gate floor
    opt = _StubOptimizer(lambda eps: _feasible(V=V))
    rung = next(iter(opt.iter_designs_over_epsilons([1e-8], normalize=False)))
    assert rung.V_gf is V
    assert rung.epsilon_effective == 1e-8
    assert rung.normalization["applied"] is False
    assert rung.normalization["reason"] == "disabled"


def test_normalisation_on_scales_the_margin_with_the_candidate():
    V = _V(2.0 ** -40)                # deep under the band
    opt = _StubOptimizer(lambda eps: _feasible(V=V))
    rung = next(iter(opt.iter_designs_over_epsilons([1e-8], normalize=True)))
    c = rung.normalization["factor"]
    assert rung.normalization["applied"] is True
    assert rung.V_gf is not V
    # The margin travels with V: it is the only margin the scaled Grams can
    # witness (exact_phase._check_bundle_group compares them as Fractions).
    assert rung.epsilon_effective == 1e-8 * c
    assert float(rung.V_gf.terms[0].coefficient) == float(V.terms[0].coefficient) * c


# ---------------------------------------------------------------------------
# The wiring in the escalation loop
# ---------------------------------------------------------------------------

def test_no_ladder_calls_the_historical_design_helper_once(monkeypatch):
    seen = {}

    def fake(weights_r, m, terms, dynamics_gf, solver, verbose, **kw):
        seen["kw"] = kw
        seen["m"] = m
        kw["design_out"]["result"] = _feasible()
        kw["design_out"]["epsilon"] = 1e-4
        return _V()

    monkeypatch.setattr(gd, "_design_full_library", fake)
    rungs = list(gd._design_rungs((1,), 3, [], [], "MOSEK_DIRECT", False,
                                  epsilon_ladder=None))
    assert len(rungs) == 1
    assert seen["m"] == 3
    assert "design_out" in seen["kw"]
    assert rungs[0].epsilon == 1e-4
    assert rungs[0].normalization is None       # the unswept record is unchanged
    assert rungs[0].bundle_reason == ""


def test_a_ladder_bypasses_the_historical_helper(monkeypatch):
    monkeypatch.setattr(gd, "_design_full_library",
                        lambda *a, **k: pytest.fail("must not be called"))
    monkeypatch.setattr(
        gd, "JointSOSOptimizer",
        lambda **kw: _StubOptimizer(lambda eps: _infeasible(), **{
            k: v for k, v in kw.items() if k not in ("weights_r",)}))
    rungs = list(gd._design_rungs((1,), 3, [], [], "MOSEK_DIRECT", False,
                                  epsilon_ladder=(1e-4, 1e-6)))
    assert [r.epsilon for r in rungs] == [1e-4, 1e-6]


def test_a_swept_rung_crash_is_recorded_in_degree_attempts(monkeypatch):
    monkeypatch.setattr(
        gd, "JointSOSOptimizer",
        lambda **kw: _StubOptimizer(lambda eps: RuntimeError("boom"), **{
            k: v for k, v in kw.items() if k not in ("weights_r",)}))
    attempts: list = []
    list(gd._design_rungs((1,), 3, [], [], "MOSEK_DIRECT", False,
                          degree_attempts=attempts, epsilon_ladder=(1e-4,)))
    assert len(attempts) == 1
    assert attempts[0]["status"] == "unexpected_error"
    assert attempts[0]["epsilon"] == 1e-4
    assert "boom" in attempts[0]["message"]


def test_the_bundle_adoption_takes_a_caller_supplied_refusal():
    V = _V()
    entries, fp, reason = gd._adopt_design_bundle(
        _feasible(V=V), V, V, entries=None,
        refusal="design_screen_refused:gram_validation_failed(block=V_oct0)")
    assert entries is None and fp is None
    assert reason.startswith("design_screen_refused:")


def test_the_bundle_adoption_prefers_caller_supplied_entries():
    # This is what the normalisation needs: the entries that witness the
    # SCALED V are the scaled ones, not the ones the solver returned.
    V = _V()
    original = [{"name": "V_oct0", "eps": 1e-4, "dim": 1}]
    scaled = [{"name": "V_oct0", "eps": 1e-2, "dim": 1}]
    result = _feasible(V=V, entries=original)
    entries, _fp, reason = gd._adopt_design_bundle(result, V, V, entries=scaled)
    assert reason == ""
    assert entries[0]["eps"] == 1e-2


# ---------------------------------------------------------------------------
# End to end, with a solver
# ---------------------------------------------------------------------------

@needs_mosek
def test_a_swept_run_reproduces_the_unswept_row_when_the_first_rung_certifies():
    import gf_framework.discovery.universal_discovery as ud

    spec = ud.create_sta_spec()
    kw = dict(solver="MOSEK_DIRECT", gate_solver="MOSEK",
              design_fallback=False, gate_fallback=False)
    plain = gd.discover_gf(spec, **kw)
    swept = gd.discover_gf(spec, epsilon_ladder=DESIGN_EPSILON_LADDER, **kw)
    assert plain.certified and swept.certified
    assert plain.degree_used == swept.degree_used
    assert plain.n_terms == swept.n_terms
    # Rung 0 IS the unswept solve, so the design point must be the same object.
    assert plain.gram_bundle_fingerprint == swept.gram_bundle_fingerprint
    assert swept.design_epsilon_requested == 1e-4
    assert swept.design_epsilon == plain.design_epsilon
    # ... and the unswept run records nothing about a ladder it did not walk.
    assert plain.design_epsilon_ladder == ()
    assert plain.epsilon_rungs == []
    assert swept.design_epsilon_ladder == DESIGN_EPSILON_LADDER


@needs_mosek
def test_the_process_default_ladder_is_honoured_and_an_argument_overrides_it():
    import gf_framework.discovery.universal_discovery as ud

    spec = ud.create_sta_spec()
    kw = dict(solver="MOSEK_DIRECT", gate_solver="MOSEK",
              design_fallback=False, gate_fallback=False)
    previous = set_design_epsilon_sweep_default(DESIGN_EPSILON_LADDER)
    try:
        res = gd.discover_gf(spec, **kw)
        assert res.design_epsilon_ladder == DESIGN_EPSILON_LADDER
        pinned = gd.discover_gf(spec, epsilon_ladder=(1e-4,), **kw)
        assert pinned.design_epsilon_ladder == (1e-4,)
    finally:
        set_design_epsilon_sweep_default(previous)


@needs_mosek
def test_the_screen_refusal_route_is_reachable_on_a_real_design():
    """continuous_twisting m = 5 at eps = 1e-8: the N20 instance.

    The design solves, the screen refuses its Grams, and the candidate is
    still offered to the gate with a typed reason. That is the route WP P5-02
    adds; if it ever stops firing here the test is telling you the design
    point moved, not that the route is dead.
    """
    import gf_framework.discovery.universal_discovery as ud
    from gf_framework.algebra.enumerate import enumerate_diagonal_terms
    from gf_framework.discovery.gf_discovery import mu_filtered_cross

    spec = ud.create_cta_spec()
    weights = tuple(int(r) for r in spec.weights_r)
    diag = enumerate_diagonal_terms(5, weights, c1_only=True)
    _mu, _cf, cross = mu_filtered_cross(5, weights, diag, spec.dynamics_gf)
    opt = JointSOSOptimizer(weights_r=weights, solver="MOSEK_DIRECT",
                            verbose=False, allow_solver_fallback=False)
    rungs = list(opt.iter_designs_over_epsilons(
        (1e-8,), normalize=False, degree_m=5, dynamics_gf=spec.dynamics_gf,
        active_terms=list(diag) + list(cross)))
    assert len(rungs) == 1
    rung = rungs[0]
    assert rung.screen_refused is True
    assert rung.V_gf is not None
    assert rung.gram_certificates is None
    assert rung.bundle_reason.startswith("design_screen_refused:")
