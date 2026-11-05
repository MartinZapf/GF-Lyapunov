"""Tests for the optional backward-elimination pass (verification.sparse_sos).

MOSEK-free. The exact charge of a real candidate needs a MOSEK-direct design
Gram bundle, so every test here injects a ``charge_fn`` instead: the point under
test is the GUARD LOGIC (what is refused, with which typed reason, and that the
pass is off unless asked for), not the charge measurement itself, which is
``exact_phase``'s own tested surface.

System under test: Twisting (r = (2,1), m = 5), the same 10-term μ-filtered
library the reweighted-ℓ₁ tests use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import (enumerate_cross_terms,
                                            enumerate_diagonal_terms)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.verification.sparse_sos import (
    BWD_CHARGE_UNAVAILABLE,
    BWD_GATE_REJECTED,
    BWD_KEPT,
    BWD_REFUSED_HEADROOM,
    BWD_REFUSED_MU,
    ChargeProbe,
    SparseDesignConfig,
    backward_eliminate,
    design_sparse_lyapunov,
)
from gf_framework.verification.verifier import certify_discovered_gf

SOLVER = "CLARABEL"


def _twisting_library(m=5):
    spec = ud.create_twisting_spec()
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return spec, m, list(diag) + list(cross)


def _gate(spec, m):
    def gate_fn(V):
        return certify_discovered_gf(V, spec.weights_r, m, spec.dynamics_gf,
                                     solver=SOLVER, strict_solver=True)
    return gate_fn


def _dense_support(spec, m, terms):
    """Start the pass from the FULL library, so there is room to eliminate."""
    return list(range(len(terms)))


def _outcomes(res):
    return [s.outcome for s in res.steps]


# --------------------------------------------------------------------------
# 1. Off by default.
# --------------------------------------------------------------------------

def test_backward_pass_is_off_by_default():
    cfg = SparseDesignConfig()
    assert cfg.backward_elimination is False
    assert cfg.backward_max_charge_ratio == pytest.approx(1e-3)
    assert cfg.resolved_backward_order() == "abs_coef"


def test_design_sparse_lyapunov_does_not_run_the_pass_by_default():
    spec, m, terms = _twisting_library()
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver=SOLVER,
                                 config=SparseDesignConfig(max_reweight_iters=3))
    assert res.feasible
    assert res.backward_elimination is False
    assert res.backward_steps == []
    assert res.backward_solves == 0
    assert res.backward_charge_ratio is None
    assert "backward elimination" not in res.message


def test_unknown_backward_order_raises():
    with pytest.raises(ValueError):
        SparseDesignConfig(backward_order="l0_greedy").resolved_backward_order()


# --------------------------------------------------------------------------
# 2. The headroom guard refuses a candidate above the bound.
# --------------------------------------------------------------------------

def test_headroom_guard_refuses_a_candidate_above_the_bound():
    """Same instance, two bounds: generous keeps drops, tight keeps none."""
    spec, m, terms = _twisting_library()
    gate_fn = _gate(spec, m)

    # A charge that is comfortably inside the generous bound and outside the
    # tight one. Constant, so the ONLY difference between the two runs is the
    # configured bound.
    def charge_fn(V, design_result):
        return ChargeProbe(1e-4, True)

    generous = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(backward_max_charge_ratio=1e-3),
        gate_fn=gate_fn, charge_fn=charge_fn)
    assert BWD_KEPT in _outcomes(generous)
    assert generous.n_dropped > 0
    assert len(generous.support) < len(terms)
    assert generous.charge_ratio == pytest.approx(1e-4)

    tight = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(backward_max_charge_ratio=1e-6),
        gate_fn=gate_fn, charge_fn=charge_fn)
    assert BWD_KEPT not in _outcomes(tight)
    assert BWD_REFUSED_HEADROOM in _outcomes(tight)
    assert tight.n_dropped == 0
    assert list(tight.support) == _dense_support(spec, m, terms)
    # The refused ratio is RECORDED, with the bound it was charged against.
    refused = [s for s in tight.steps if s.outcome == BWD_REFUSED_HEADROOM]
    assert all(s.charge_ratio == pytest.approx(1e-4) for s in refused)
    assert all(s.charge_bound == pytest.approx(1e-6) for s in refused)


def test_exact_failure_is_refused_even_below_the_bound():
    """A small ratio with a FAILED exact verdict is still a refusal."""
    spec, m, terms = _twisting_library()

    def charge_fn(V, design_result):
        return ChargeProbe(1e-9, False, "octant3:residual_over_budget")

    res = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(backward_max_charge_ratio=1e-3),
        gate_fn=_gate(spec, m), charge_fn=charge_fn)
    assert res.n_dropped == 0
    assert BWD_REFUSED_HEADROOM in _outcomes(res)
    assert any("residual_over_budget" in s.detail for s in res.steps)


def test_unmeasurable_charge_is_refused_not_waved_through():
    spec, m, terms = _twisting_library()

    def charge_fn(V, design_result):
        return ChargeProbe(None, False, "no_design_gram_bundle", "CLARABEL design")

    res = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(), gate_fn=_gate(spec, m), charge_fn=charge_fn)
    assert res.n_dropped == 0
    assert BWD_CHARGE_UNAVAILABLE in _outcomes(res)
    assert BWD_KEPT not in _outcomes(res)


def test_no_charge_fn_on_a_non_mosek_design_refuses_every_drop():
    """The shipped default charge probe on a CLARABEL design: no bundle, no drop."""
    spec, m, terms = _twisting_library()
    res = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(backward_max_passes=1),
        gate_fn=_gate(spec, m))
    assert res.n_dropped == 0
    assert set(_outcomes(res)) <= {BWD_CHARGE_UNAVAILABLE, BWD_REFUSED_MU,
                                   BWD_GATE_REJECTED, "infeasible"}
    assert BWD_CHARGE_UNAVAILABLE in _outcomes(res)


# --------------------------------------------------------------------------
# 3. The μ guard refuses a μ-breaking candidate.
# --------------------------------------------------------------------------

def test_mu_guard_refuses_a_mu_breaking_candidate_without_a_solve():
    """A reference μ no candidate can match ⇒ every drop is refused_mu, no solve.

    The μ of a support is a function of its term exponents alone, so the guard
    runs BEFORE the design re-solve. Forcing the reference to an unreachable
    value is the cheap way to pin that ordering: if a single design solve had
    run, ``n_solves`` would not be zero.
    """
    spec, m, terms = _twisting_library()
    calls = {"charge": 0}

    def charge_fn(V, design_result):
        calls["charge"] += 1
        return ChargeProbe(0.0, True)

    res = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(), gate_fn=_gate(spec, m),
        charge_fn=charge_fn, mu_reference=(99, 98))
    assert res.n_dropped == 0
    assert res.n_solves == 0          # refused before any solve was spent
    assert calls["charge"] == 0       # and before any charge was measured
    assert set(_outcomes(res)) == {BWD_REFUSED_MU}
    assert all("mu_not_preserved" in s.detail for s in res.steps)


def test_mu_guard_off_lets_the_same_candidate_through_to_the_solve():
    """Control for the test above: with the guard off the pass reaches the solve."""
    spec, m, terms = _twisting_library()
    res = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(mu_guard="off", backward_max_passes=1),
        gate_fn=_gate(spec, m),
        charge_fn=lambda V, r: ChargeProbe(1e-5, True),
        mu_reference=(99, 98))
    assert BWD_REFUSED_MU not in _outcomes(res)
    assert res.n_solves > 0


# --------------------------------------------------------------------------
# 4. Bookkeeping: typed outcomes, recorded ratios, wiring through the designer.
# --------------------------------------------------------------------------

def test_every_step_carries_a_typed_outcome_and_a_json_record():
    spec, m, terms = _twisting_library()
    res = backward_eliminate(
        spec.weights_r, m, spec.dynamics_gf, terms,
        _dense_support(spec, m, terms), solver=SOLVER,
        config=SparseDesignConfig(backward_max_charge_ratio=1e-3,
                                  backward_max_passes=2),
        gate_fn=_gate(spec, m),
        charge_fn=lambda V, r: ChargeProbe(2e-4, True))
    assert res.steps
    for rec in res.records():
        assert rec["outcome"] in {BWD_KEPT, BWD_REFUSED_MU, BWD_REFUSED_HEADROOM,
                                  BWD_GATE_REJECTED, BWD_CHARGE_UNAVAILABLE,
                                  "infeasible", "budget_exhausted"}
        assert rec["support_size_after"] == rec["support_size_before"] - 1
        assert isinstance(rec["term"], str) and rec["term"]
        import json
        json.dumps(rec)  # the record must survive an artifact dump


def test_designer_runs_the_pass_when_asked_and_records_both_ratios():
    spec, m, terms = _twisting_library()
    cfg = SparseDesignConfig(max_reweight_iters=3, backward_elimination=True,
                             backward_max_charge_ratio=1e-3,
                             backward_max_passes=2)
    res = design_sparse_lyapunov(
        spec.weights_r, m, spec.dynamics_gf, terms, solver=SOLVER, config=cfg,
        gate_fn=_gate(spec, m))
    assert res.feasible and res.gated
    assert res.backward_elimination is True
    assert res.backward_max_charge_ratio == pytest.approx(1e-3)
    assert res.backward_start_support
    # No MOSEK bundle on a CLARABEL design ⇒ nothing can be priced, so the pass
    # adopts nothing and the rl1 support survives untouched.
    assert res.backward_n_dropped == 0
    assert list(res.support) == list(res.backward_start_support)
    assert res.n_terms_sparse == len(res.backward_start_support)
    assert "backward elimination" in res.message
    assert res.V_gf is not None and len(res.V_gf.terms) == res.n_terms_sparse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
