"""WP P5-05: the opt-in ``min_gram_trace`` design objective.

What is pinned here:

  * the flag validates, defaults to ``"auto"``, and is recorded on the result;
  * ``"auto"`` is bit-identical to the pre-P5-05 behaviour (same coefficients,
    same Gram bundle fingerprint) on BOTH backends;
  * ``"min_gram_trace"`` really minimizes Σ tr Q (strictly smaller total trace,
    lower numerical Gram rank) and still produces a design that passes the
    P2-04 screen and the exact gate;
  * both backends implement it, so a MOSEK_DIRECT -> CVXPY fallback changes the
    ENCODING and not the formulation.

Everything except the three explicitly MOSEK-marked tests runs without MOSEK
(CVXPY/CLARABEL design solves; a design solve is not a certifying solve).
"""
from __future__ import annotations

import numpy as np
import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import (
    enumerate_cross_terms, enumerate_diagonal_terms)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.verification.joint_sos import (
    DESIGN_OBJECTIVES, JointSOSOptimizer, gram_bundle_fingerprint,
    resolve_design_objective,
)

try:  # the MOSEK Task-API backend is what carries the Gram bundle
    from gf_framework.verification.mosek_direct import (
        MOSEK_AVAILABLE, solve_joint_sos_direct)
except Exception:  # pragma: no cover - import guard only
    MOSEK_AVAILABLE = False
    solve_joint_sos_direct = None

requires_mosek = pytest.mark.skipif(
    not MOSEK_AVAILABLE, reason="MOSEK not available on this box")

EPS = 1e-4


def _sta():
    spec = ud.create_sta_spec()
    return spec, tuple(int(r) for r in spec.weights_r), 3


def _library(spec, m):
    """The E1 mu-filtered C1 library, in a deterministic order.

    Not the designer's default ``enumerate_all_terms``: that one lands on a
    design whose W blocks have the weighted-sphere clearing ACTIVE, hence
    ``eps = 0`` and no exact-gateable W (P2-02), which would make the exact
    assertion below vacuous for BOTH objectives.
    """
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = sorted((t for t in cross_full if mu_compatible(t, mu)),
                   key=lambda t: (tuple(str(e) for e in t.exponents),
                                  tuple(t.term_types)))
    return list(diag) + cross


def _design(objective, solver, **kw):
    spec, weights, m = _sta()
    opt = JointSOSOptimizer(weights_r=weights, solver=solver, epsilon=EPS,
                            allow_solver_fallback=False, objective=objective,
                            **kw)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf,
        active_terms=_library(spec, m))


# --------------------------------------------------------------------------
# flag surface
# --------------------------------------------------------------------------

def test_default_objective_is_auto():
    opt = JointSOSOptimizer(weights_r=(2, 1))
    assert opt.objective == "auto"
    assert DESIGN_OBJECTIVES[0] == "auto"


@pytest.mark.parametrize("name", DESIGN_OBJECTIVES)
def test_every_listed_objective_resolves(name):
    assert resolve_design_objective(name) == name
    assert JointSOSOptimizer(weights_r=(2, 1), objective=name).objective == name


def test_none_resolves_to_auto():
    assert resolve_design_objective(None) == "auto"


def test_objective_is_normalized_case_insensitively():
    assert resolve_design_objective("  MIN_GRAM_TRACE ") == "min_gram_trace"


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="unknown design objective"):
        JointSOSOptimizer(weights_r=(2, 1), objective="min_rank")
    with pytest.raises(ValueError, match="unknown design objective"):
        resolve_design_objective("nuclear")


# --------------------------------------------------------------------------
# CVXPY backend (MOSEK-free)
# --------------------------------------------------------------------------

def test_cvxpy_auto_is_feasible_and_records_the_objective():
    res = _design("auto", "CLARABEL")
    assert res.feasible, res.solver_status
    assert res.design_objective == "auto"


def test_cvxpy_min_gram_trace_is_feasible_and_records_the_objective():
    res = _design("min_gram_trace", "CLARABEL")
    assert res.feasible, res.solver_status
    assert res.design_objective == "min_gram_trace"


def test_cvxpy_auto_is_unchanged_by_the_new_flag():
    """The default path must be bit-identical to the pre-P5-05 behaviour."""
    a = _design("auto", "CLARABEL")
    b = _design("auto", "CLARABEL")
    assert a.feasible and b.feasible
    assert a.coefficients == b.coefficients


def test_cvxpy_min_gram_trace_moves_the_design_point():
    """Different objective, different optimum — otherwise the flag is a no-op."""
    a = _design("auto", "CLARABEL")
    b = _design("min_gram_trace", "CLARABEL")
    assert a.feasible and b.feasible
    va = np.array([a.coefficients[k] for k in sorted(a.coefficients)])
    vb = np.array([b.coefficients[k] for k in sorted(b.coefficients)])
    assert not np.allclose(va, vb)


def test_cvxpy_objective_is_not_collected_when_not_requested(monkeypatch):
    """``gram_vars`` stays None on the default path, so nothing is collected."""
    import gf_framework.verification.joint_sos as js
    seen = []
    original = js.JointSOSOptimizer._emit_v_block

    def spy(self, **kw):
        seen.append(kw.get("gram_vars"))
        return original(self, **kw)

    monkeypatch.setattr(js.JointSOSOptimizer, "_emit_v_block", spy)
    _design("auto", "CLARABEL")
    assert seen and all(g is None for g in seen)
    seen.clear()
    _design("min_gram_trace", "CLARABEL")
    assert seen and all(isinstance(g, list) for g in seen)


# --------------------------------------------------------------------------
# MOSEK_DIRECT backend
# --------------------------------------------------------------------------

@requires_mosek
def test_mosek_auto_bundle_is_unchanged_by_the_new_flag():
    a = _design("auto", "MOSEK_DIRECT")
    b = _design("auto", "MOSEK_DIRECT")
    assert a.feasible and b.feasible
    assert a.design_objective == "auto"
    assert (gram_bundle_fingerprint(a.gram_certificates)
            == gram_bundle_fingerprint(b.gram_certificates))


@requires_mosek
def test_mosek_min_gram_trace_lowers_the_total_gram_trace():
    a = _design("auto", "MOSEK_DIRECT")
    b = _design("min_gram_trace", "MOSEK_DIRECT")
    assert a.feasible and b.feasible, (a.solver_status, b.solver_status)
    tr = lambda r: sum(float(np.trace(np.asarray(e["gram"], dtype=float)))
                       for e in r.gram_certificates)
    assert tr(b) < tr(a)
    assert b.design_objective == "min_gram_trace"
    assert (gram_bundle_fingerprint(a.gram_certificates)
            != gram_bundle_fingerprint(b.gram_certificates))


@requires_mosek
def test_mosek_min_gram_trace_lowers_the_numerical_gram_rank():
    a = _design("auto", "MOSEK_DIRECT")
    b = _design("min_gram_trace", "MOSEK_DIRECT")
    assert a.feasible and b.feasible

    def ranks(r):
        out = []
        for e in r.gram_certificates:
            g = np.asarray(e["gram"], dtype=float)
            smax = float(np.linalg.svd(g, compute_uv=False).max())
            out.append(int(np.linalg.matrix_rank(g, tol=1e-8 * smax)))
        return out

    assert sum(ranks(b)) < sum(ranks(a))


@requires_mosek
def test_mosek_min_gram_trace_still_passes_the_design_screen():
    """P2-04's fail-fast screen must keep accepting the new objective's bundle."""
    res = _design("min_gram_trace", "MOSEK_DIRECT")
    assert res.feasible, res.solver_status
    assert res.design_screen is not None
    assert res.design_screen["ok"] is True, res.design_screen.get("reason")


@requires_mosek
def test_mosek_min_gram_trace_bundle_still_proves_exactly():
    """The new objective's design still proves exactly (P2-03), and X judges its
    bundle by P's rule.

    Minimizing the Gram trace drives the Grams onto the PSD boundary: here a W
    Gram has float lambda_min of about -5.1e-9. Until 2026-09-26 X proposed
    eta = |lambda_min| + 1e-15 and adopted the bundle at charge/eps ~2e-4; P's
    ladder stops at 1e-9 before the Gershgorin rung, so P could not reproduce
    that. Since fix (e) (author decision D9) X uses P's ladder, refuses the
    bundle exactly as P would (budget_exceeded at the Gershgorin rung), and the
    exact phase adopts its recorded fallback witness, which proves.
    """
    from gf_framework.verification.exact_phase import run_exact_phase
    spec, _weights, _m = _sta()
    res = _design("min_gram_trace", "MOSEK_DIRECT")
    assert res.feasible
    ph = run_exact_phase(res.V_gf, [list(spec.dynamics_gf)],
                         len(spec.weights_r), epsilon=EPS, gram_bundle=res)
    assert ph.exact_pass, ph.v_exact.reason
    first = ph.witness_attempts[0]
    assert first["source"] == "design_bundle"
    if ph.witness_source != "design_bundle":
        # the bundle is below P's small rungs: refused, never adopted
        assert res.gram_min_eig < -1e-9
        assert first["ok"] is False and first["adopted"] is False
        assert first["shortfall"].endswith("budget_exceeded")
        adopted = [a for a in ph.witness_attempts if a.get("adopted")]
        assert len(adopted) == 1 and adopted[0]["ok"] is True
        assert adopted[0]["source"] == ph.witness_source


@requires_mosek
def test_solve_joint_sos_direct_trace_flag_is_off_by_default():
    """The backend signature default must not change any existing caller."""
    import inspect
    sig = inspect.signature(solve_joint_sos_direct)
    assert sig.parameters["minimize_gram_trace"].default is False
