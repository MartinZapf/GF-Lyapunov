"""MOSEK_DIRECT branch of the single-polynomial SOS feasibility solve (WP P2-10).

Before this WP ``joint_sos._numeric_poly_is_sos`` was CVXPY-only, so P2-05's
witness sources 2 (MOSEK re-solve) and 3 (CLARABEL retry) both ran through the
CVXPY encoding, and ``solver="MOSEK"`` there meant CVXPY's MOSEK *interface*,
not the MOSEK Task API the design side uses. N7, N9 and N10 record that the
CVXPY encoding manufactures false NEGATIVES that vanish under the direct path,
which is exactly the risk on a witness route.

What is pinned here:

* when the REQUESTED solver is MOSEK and the direct backend is importable, the
  Gram comes from ``mosek_direct.solve_poly_sos_direct`` (Task API) and
  ``used`` is ``"MOSEK_DIRECT"``;
* the MOSEK_DIRECT rung SUBSTITUTES the CVXPY MOSEK rung — it never adds a
  CVXPY solve, and CLARABEL/SCS stay exactly where ``_solver_candidates`` puts
  them;
* the attempt trace is typed per P3-02 (``failure_typing`` parses
  ``MOSEK_DIRECT:<status>[:detail]`` with no producer change);
* a monkeypatched-unavailable MOSEK direct backend falls back to the typed
  CVXPY path and still certifies;
* both encodings pose the SAME problem: same basis, same Gram dimension, same
  ε/margin semantics, same a-posteriori ``_validate_gram_certificate``
  acceptance, and both produce Grams the exact rational phase accepts on two
  nominal systems;
* the solver status is never the certificate. A non-SOS target is refused on
  both branches, and a Gram that fails the post-check is not returned.

Every MOSEK-dependent test skips cleanly when MOSEK is not importable.
"""
from fractions import Fraction

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.discovery.universal_discovery import (
    create_fosmc_spec,
    create_sta_spec,
)
from gf_framework.verification import exact_phase as ep
from gf_framework.verification import joint_sos as js
from gf_framework.verification import mosek_direct as md
from gf_framework.verification.exact_phase import run_exact_phase
from gf_framework.verification.failure_typing import (
    FailureClass,
    parse_octant_diagnostics,
)
from gf_framework.verification.joint_sos import (
    _numeric_poly_is_sos,
    verify_gf_sos_full,
)

MOSEK = pytest.mark.skipif(not md.MOSEK_AVAILABLE, reason="MOSEK not installed")

EPS = 1e-6

#: A strictly positive definite quartic form in 2 variables, well inside the
#: SOS cone, with the ε margin subtracted from the pure powers exactly the way
#: ``verify_gf_sos_full`` subtracts it.
def _pd_quartic(eps=1e-4):
    target = {(4, 0): 1.0, (0, 4): 1.0, (2, 2): -0.5}
    target[(4, 0)] -= eps
    target[(0, 4)] -= eps
    return target, eps


def _fosmc():
    """δ₀ = 0.5 < k = 1: the smallest system that certifies (n = 1)."""
    spec = create_fosmc_spec(k=1.0, delta0=0.5)
    V = GeneralizedForm(degree_m=2, weights_r=(1,), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(2),), term_types=("abs",)),
    ])
    return V, spec.dynamics_gf_vertices, 1


def _sta():
    """A designed STA certificate (n = 2), CLARABEL-designed so the DESIGN side
    of the fixture is independent of the backend under test."""
    from gf_framework.algebra.enumerate import (
        enumerate_cross_terms,
        enumerate_diagonal_terms,
    )
    from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
    from gf_framework.verification.joint_sos import JointSOSOptimizer

    spec = create_sta_spec()
    diag = enumerate_diagonal_terms(3, spec.weights_r, c1_only=True)
    cross = enumerate_cross_terms(3, spec.weights_r, max_denominator=4,
                                  include_signed=True, minimal=False,
                                  include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    terms = list(diag) + [t for t in cross if mu_compatible(t, mu)]
    res = JointSOSOptimizer(weights_r=spec.weights_r, solver="CLARABEL",
                            allow_solver_fallback=False, epsilon=1e-4
                            ).find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=terms,
        dynamics_gf_vertices=spec.dynamics_gf_vertices)
    assert res.feasible, res.message
    return res.V_gf, [spec.dynamics_gf], 2


def _spy_solver_used(monkeypatch):
    """Record ``JointSOSResult.solver_used`` for every re-solve the phase runs."""
    real = ep.verify_gf_sos_full
    seen = []

    def spy(form, **kwargs):
        result = real(form, **kwargs)
        seen.append(result.solver_used)
        return result

    monkeypatch.setattr(ep, "verify_gf_sos_full", spy)
    return seen


# ---------------------------------------------------------------------------
# 1. The branch exists, is named, and is typed
# ---------------------------------------------------------------------------

@MOSEK
def test_the_direct_branch_names_mosek_direct_and_records_it_in_the_trace():
    target, eps = _pd_quartic()
    ok, gram, used, attempts, diag = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps)
    assert ok is True
    assert used == "MOSEK_DIRECT"
    assert gram is not None and gram.shape == (3, 3)
    assert len(attempts) == 1
    assert attempts[0].startswith("MOSEK_DIRECT:")
    assert "postcheck_ok" in attempts[0]
    assert diag["min_eig"] > 0.0


@MOSEK
def test_the_attempt_trace_types_per_p3_02():
    """``failure_typing`` parses the new producer with no change of its own."""
    target, eps = _pd_quartic()
    _ok, _g, _u, attempts, _d = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps)
    parsed = parse_octant_diagnostics(attempts)
    assert len(parsed) == 1
    assert parsed[0]["solver"] == "MOSEK_DIRECT"
    assert parsed[0]["solver_status"] == "optimal"
    assert parsed[0]["accepted"] is True
    assert parsed[0]["failure_class"] == FailureClass.NONE
    assert parsed[0]["gram_min_eig"] is not None
    assert parsed[0]["gram_residual_l1"] is not None


@MOSEK
def test_the_octant_prefix_still_parses_through_verify_gf_sos_full():
    V, _dyn, _n = _fosmc()
    res = verify_gf_sos_full(V, epsilon=EPS, solver="MOSEK")
    assert res.feasible
    assert res.solver_used == "MOSEK_DIRECT"
    assert all(a.startswith("oct") for a in res.solver_attempts)
    parsed = parse_octant_diagnostics(res.solver_attempts)
    assert {p["solver"] for p in parsed} == {"MOSEK_DIRECT"}
    assert {p["octant_index"] for p in parsed} == {0, 1}


# ---------------------------------------------------------------------------
# 2. Substitution, not addition: no new CVXPY solve
# ---------------------------------------------------------------------------

@MOSEK
def test_the_direct_rung_substitutes_the_cvxpy_mosek_rung():
    """A refuted target walks the whole ladder. MOSEK appears exactly once,
    as MOSEK_DIRECT; the CVXPY rungs after it are unchanged."""
    not_sos = {(4, 0): 1.0, (0, 4): 1.0, (2, 2): -3.0}
    ok, gram, used, attempts, _d = _numeric_poly_is_sos(
        not_sos, 2, 4, solver="MOSEK")
    assert ok is False and gram is None and used is None
    solvers = [a.split(":")[0] for a in attempts]
    assert solvers == ["MOSEK_DIRECT", "CLARABEL", "SCS"]
    assert "MOSEK" not in solvers  # never re-posed through CVXPY


def test_a_clarabel_request_never_reaches_the_direct_backend(monkeypatch):
    """The branch is keyed on the REQUESTED solver, so every CLARABEL-pinned
    caller (levant_2003_perturbed, exact_gate_standard_gains, the drivers and
    most tests) is bit-for-bit unaffected."""
    def boom(**_kw):
        raise AssertionError("the direct backend must not run for CLARABEL")

    monkeypatch.setattr(md, "solve_poly_sos_direct", boom)
    target, eps = _pd_quartic()
    ok, _g, used, attempts, _d = _numeric_poly_is_sos(
        target, 2, 4, solver="CLARABEL", eps_budget=eps)
    assert ok is True and used == "CLARABEL"
    assert not any("MOSEK_DIRECT" in a for a in attempts)


@MOSEK
def test_use_mosek_direct_false_restores_the_historical_cvxpy_ladder():
    target, eps = _pd_quartic()
    ok, _g, used, attempts, _d = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps, use_mosek_direct=False)
    assert ok is True and used == "MOSEK"
    assert attempts[0].startswith("MOSEK:")
    assert not any("MOSEK_DIRECT" in a for a in attempts)


@MOSEK
def test_strict_solver_keeps_the_direct_rung_alone():
    """``strict_solver`` means one rung. With MOSEK requested that rung is now
    MOSEK_DIRECT and no CVXPY solve runs at all."""
    not_sos = {(4, 0): 1.0, (0, 4): 1.0, (2, 2): -3.0}
    ok, _g, _u, attempts, _d = _numeric_poly_is_sos(
        not_sos, 2, 4, solver="MOSEK", strict_solver=True)
    assert ok is False
    assert attempts == ["MOSEK_DIRECT:infeasible"]


# ---------------------------------------------------------------------------
# 3. Unavailable direct backend -> the typed CVXPY path
# ---------------------------------------------------------------------------

def test_unavailable_mosek_direct_falls_back_to_the_typed_cvxpy_path(monkeypatch):
    monkeypatch.setattr(md, "MOSEK_AVAILABLE", False)
    target, eps = _pd_quartic()
    ok, gram, used, attempts, _d = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps)
    assert attempts[0] == "MOSEK_DIRECT:unavailable"
    parsed = parse_octant_diagnostics(attempts)
    assert parsed[0]["solver"] == "MOSEK_DIRECT"
    assert parsed[0]["failure_class"] == FailureClass.SOLVER_UNAVAILABLE
    # ... and the CVXPY ladder still runs, with the MOSEK rung back in place.
    assert ok is True and gram is not None
    assert used in ("MOSEK", "CLARABEL", "SCS")
    assert [a.split(":")[0] for a in attempts[1:]][0] == "MOSEK"


def test_unavailable_mosek_direct_still_certifies_through_the_phase(monkeypatch):
    monkeypatch.setattr(md, "MOSEK_AVAILABLE", False)
    V, dyn, n = _fosmc()
    phase = run_exact_phase(V, dyn, n, epsilon=EPS, solver="MOSEK")
    assert phase.exact_pass is True


# ---------------------------------------------------------------------------
# 4. Same problem, both encodings
# ---------------------------------------------------------------------------

@MOSEK
def test_both_encodings_pose_the_same_gram_block():
    target, eps = _pd_quartic()
    _ok_d, g_d, u_d, _a, d_d = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps)
    _ok_c, g_c, u_c, _a, d_c = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps, use_mosek_direct=False)
    assert u_d == "MOSEK_DIRECT" and u_c == "MOSEK"
    assert g_d.shape == g_c.shape
    assert list(d_d["basis"]) == list(d_c["basis"])
    # Same target, so both Grams reproduce it to within the post-check budget.
    assert d_d["residual_l1"] < 0.5 * eps
    assert d_c["residual_l1"] < 0.5 * eps


@MOSEK
@pytest.mark.parametrize("system", ["fosmc", "sta"])
def test_both_branches_yield_grams_the_exact_phase_accepts(monkeypatch, system):
    """Acceptance (a): two nominal systems, both encodings, exact-PASS on both.

    The exact rational phase is the authority in both arms; the float solve
    only PROPOSES the Gram, so this is a statement about the proposer, not
    about the certificate.
    """
    V, dyn, n = _fosmc() if system == "fosmc" else _sta()

    seen_direct = _spy_solver_used(monkeypatch)
    direct = run_exact_phase(V, dyn, n, epsilon=EPS, solver="MOSEK")
    assert direct.exact_pass is True, direct.message
    assert "MOSEK_DIRECT" in seen_direct

    monkeypatch.setattr(md, "MOSEK_AVAILABLE", False)
    seen_cvxpy = _spy_solver_used(monkeypatch)
    cvxpy = run_exact_phase(V, dyn, n, epsilon=EPS, solver="MOSEK")
    assert cvxpy.exact_pass is True, cvxpy.message
    assert "MOSEK_DIRECT" not in seen_cvxpy

    assert direct.exact_pass == cvxpy.exact_pass


# ---------------------------------------------------------------------------
# 5. Soundness: the status is never the certificate
# ---------------------------------------------------------------------------

@MOSEK
def test_a_damaged_direct_gram_is_refused_and_the_ladder_continues(monkeypatch):
    """``solve_poly_sos_direct`` returning ``optimal`` proves nothing. Damage
    the Gram past the ε budget and the post-check must reject it, type the
    rejection, and hand the problem to the next rung."""
    real = md.solve_poly_sos_direct

    def damaged(**kwargs):
        out = real(**kwargs)
        if out.gram is not None:
            out.gram = np.asarray(out.gram, dtype=float).copy()
            out.gram[0, 0] += 1.0
        return out

    monkeypatch.setattr(md, "solve_poly_sos_direct", damaged)
    target, eps = _pd_quartic()
    ok, gram, used, attempts, _d = _numeric_poly_is_sos(
        target, 2, 4, solver="MOSEK", eps_budget=eps)
    assert attempts[0].startswith("MOSEK_DIRECT:optimal:gram_")
    assert parse_octant_diagnostics(attempts)[0]["failure_class"] == \
        FailureClass.GRAM_REJECTED
    # The next rung (CLARABEL, CVXPY) is untouched and still certifies.
    assert ok is True and used == "CLARABEL" and gram is not None


@MOSEK
def test_the_direct_backend_refuses_a_monomial_outside_the_basis():
    outcome = md.solve_poly_sos_direct(
        target={(6, 0): 1.0}, gram_map={(4, 0): [(0, 0)]}, n_basis=1)
    assert outcome.status == "infeasible"
    assert outcome.gram is None
    assert outcome.message == "monomial_outside_basis"


@MOSEK
def test_the_direct_backend_matches_the_bara_doubling_convention():
    """``gram_map`` lists BOTH ordered pairs for i != j, and MOSEK's <A, X>
    doubles a lower-triangle off-diagonal, so the barA weight is 1.0. A wrong
    convention would show up as an O(1) residual on the cross term."""
    # (y0 + y1)^2 = y0^2 + 2 y0 y1 + y1^2 over the basis [y0, y1].
    gram_map = {(2, 0): [(0, 0)], (1, 1): [(0, 1), (1, 0)], (0, 2): [(1, 1)]}
    target = {(2, 0): 1.0, (1, 1): 2.0, (0, 2): 1.0}
    outcome = md.solve_poly_sos_direct(
        target=target, gram_map=gram_map, n_basis=2)
    assert outcome.status in ("optimal", "unknown")
    Q = outcome.gram
    assert abs(Q[0, 0] - 1.0) < 1e-7
    assert abs(Q[1, 1] - 1.0) < 1e-7
    assert abs(Q[0, 1] - 1.0) < 1e-7  # 2 * Q[0,1] == 2.0


def test_the_direct_backend_reports_unavailable_without_mosek(monkeypatch):
    monkeypatch.setattr(md, "MOSEK_AVAILABLE", False)
    outcome = md.solve_poly_sos_direct(
        target={(2, 0): 1.0}, gram_map={(2, 0): [(0, 0)]}, n_basis=1)
    assert outcome.status == "unavailable"
    assert outcome.gram is None


def test_structural_refusals_are_unchanged_by_the_new_branch():
    """The pre-solve structural verdicts are encoding-independent and still
    short-circuit before any backend is touched."""
    assert _numeric_poly_is_sos({}, 2, 4, solver="MOSEK")[2] == "trivial_zero"
    assert _numeric_poly_is_sos({(3, 0): 1.0}, 2, 4, solver="MOSEK")[3] == \
        ["odd_homogeneous_degree"]
    assert _numeric_poly_is_sos({(4, 0): 1.0, (0, 2): 1.0, (5, 0): 1.0},
                                2, 4, solver="MOSEK")[3] == \
        ["monomial_outside_basis"]


def test_the_flag_is_threaded_through_every_public_entry_point():
    import inspect
    for fn in (js._numeric_poly_is_sos, js.verify_sos_polynomial,
               js.verify_gf_sos_full, js.verify_w_given_numeric):
        params = inspect.signature(fn).parameters
        assert "use_mosek_direct" in params, fn.__name__
        assert params["use_mosek_direct"].default is True, fn.__name__
