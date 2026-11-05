"""WP P2-09: the design encoding of an ODD-y-degree W block.

Two independent defects were fixed together, and this file pins both.

1. **Free Gram entries.** ``build_gram_constraints`` emits one
   coefficient-matching row per monomial of the *target support*. Any basis
   product landing outside that support therefore appeared in NO row, so the
   corresponding Gram entries were free variables of the SDP: the block was a
   certificate of ``target + (that mass)``, not of ``target``. Measured on
   ``power_rate``'s W block (target ``{y³: 3}``, basis ``[1, y, y²]``, the
   product ``y⁴`` outside the support): ``Q[2,2] = 6.16e+04`` and
   ``‖E‖₁ = 6.16e+04`` (P2-04 contradiction 1; the same mass made the
   ``min_gram_trace`` objective stall, P5-05 contradiction 3).
   ``build_gram_completion_map`` closes it with an explicit ``⟨A, Q⟩ = 0`` row
   per outside monomial, in BOTH encodings.

2. **Why the block was odd at all.** ``W`` is r-homogeneous, so at μᵢ = k·rᵢ
   every W monomial shares one y-degree; when that degree is odd the
   associated form is sign-indefinite on ℝⁿ and no SOS certificate exists —
   adding the completion rows alone would have made the design INFEASIBLE.
   The design now doubles μ for the W blocks in that case (μᵢ·aᵢ is an integer
   by construction, so 2μᵢ·aᵢ is even), which is exactly the μ
   ``verify_gf_sos_full`` picks when it re-solves W on its own. The block
   becomes homogeneous of even degree, P2-02's ε rows apply, and the exact
   gate accepts the DESIGN Gram with no re-solve.

Nothing here is a certifying run: the CVXPY arms are small synthetic
feasibility problems posed only to show the two encodings agree.
"""
import numpy as np
import pytest

import cvxpy as cp

from gf_framework.algebra.enumerate import (
    enumerate_cross_terms,
    enumerate_diagonal_terms,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.discovery.universal_discovery import (
    create_levant3d_spec,
    create_power_rate_spec,
    create_sta_spec,
)
from gf_framework.verification.exact_phase import run_exact_phase
from gf_framework.verification.joint_sos import (
    MU_POLICY_DESIGN,
    MU_POLICY_W_DOUBLED,
    JointSOSOptimizer,
    build_gram_completion_map,
    build_gram_constraints,
    get_monomial_basis,
)
from gf_framework.verification.mosek_direct import (
    MOSEK_AVAILABLE,
    blocks_from_assembly,
    solve_joint_sos_direct,
)

needs_mosek = pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not installed")

#: The pre-P2-09 ``power_rate`` W block, in miniature: target 3·y³ on the
#: non-homogeneous basis [1, y, y²]. ``y⁴`` is reachable and outside the
#: support, which is where the 6.16e+04 sat.
ODD_BASIS = [(0,), (1,), (2,)]
ODD_SUPPORT = get_monomial_basis(1, 3)          # [(0,), (1,), (2,), (3,)]
ODD_M = np.array([[0.0], [0.0], [0.0], [3.0]])  # coefficient rows vs alpha


def _library(spec, m):
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    return list(diag) + [t for t in cross_full if mu_compatible(t, mu)]


def _design(spec, m, **kwargs):
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                            allow_solver_fallback=False, **kwargs)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf,
        active_terms=_library(spec, m),
        dynamics_gf_vertices=spec.dynamics_gf_vertices)


# ---------------------------------------------------------------------------
# The completion map itself
# ---------------------------------------------------------------------------

def test_the_completion_map_finds_the_free_entry_of_the_odd_block():
    extra = build_gram_completion_map(ODD_BASIS, ODD_SUPPORT, 1)
    assert set(extra) == {(4,)}
    assert extra[(4,)] == [(2, 2)]


def test_the_completion_map_is_empty_on_a_homogeneous_even_block():
    """Half-basis of degree d/2 -> every product has degree exactly d."""
    basis = [(2, 0), (1, 1), (0, 2)]
    assert build_gram_completion_map(basis, get_monomial_basis(2, 4), 2) == {}


def test_the_completion_map_is_the_exact_complement_of_the_pair_map():
    """Together the two maps cover every ordered basis pair, once."""
    basis = [(0, 0), (1, 0), (0, 1), (2, 0)]
    support = get_monomial_basis(2, 3)
    pairs = build_gram_constraints(basis, support, 2)
    extra = build_gram_completion_map(basis, support, 2)
    covered = [p for v in pairs.values() for p in v]
    covered += [p for v in extra.values() for p in v]
    assert sorted(covered) == sorted(
        (i, j) for i in range(len(basis)) for j in range(len(basis)))
    assert not (set(pairs) & set(extra))


# ---------------------------------------------------------------------------
# The two encodings agree on the small odd block
# ---------------------------------------------------------------------------

def _cvxpy_odd_block(with_completion: bool):
    """CVXPY encoding of the miniature odd block. Returns the status."""
    alpha = cp.Variable(1, name="alpha")
    Q = cp.Variable((3, 3), symmetric=True, name="Q")
    gram_map = build_gram_constraints(ODD_BASIS, ODD_SUPPORT, 1)
    cons = [Q >> 0, alpha[0] == 1.0]
    for r, mono in enumerate(ODD_SUPPORT):
        coef = ODD_M[r, :] @ alpha
        pairs = gram_map.get(mono, [])
        if pairs:
            cons.append(coef == sum(Q[i, j] for i, j in pairs))
        else:
            cons.append(coef == 0)
    if with_completion:
        for pairs in build_gram_completion_map(
                ODD_BASIS, ODD_SUPPORT, 1).values():
            cons.append(sum(Q[i, j] for i, j in pairs) == 0)
    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver="CLARABEL")
    return prob.status, (None if Q.value is None else np.asarray(Q.value))


def _mosek_odd_block(with_completion: bool):
    """MOSEK Task-API encoding of the same block. Returns the status."""
    gram_map = build_gram_constraints(ODD_BASIS, ODD_SUPPORT, 1)
    zero_pairs = (build_gram_completion_map(ODD_BASIS, ODD_SUPPORT, 1)
                  if with_completion else None)
    block = blocks_from_assembly(
        ODD_M, ODD_SUPPORT, gram_map, eps_rows=(), scaled_epsilon=0.0,
        dim=3, name="W_odd", zero_pairs=zero_pairs)
    out = solve_joint_sos_direct(
        n_terms=1, blocks=[block], diag_indices=[0], scaled_epsilon=0.0,
        coefficient_scale=1.0)
    return out.status, (None if out.grams is None else out.grams[0])


@needs_mosek
def test_both_encodings_leave_the_same_entry_free_without_the_completion_rows():
    """The defect reproduces identically under CVXPY and under MOSEK direct."""
    cvx_status, cvx_Q = _cvxpy_odd_block(with_completion=False)
    msk_status, msk_Q = _mosek_odd_block(with_completion=False)
    assert cvx_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert msk_status in ("optimal", "unknown_accepted")
    # Both certify 3y³ + Q[2,2]·y⁴, i.e. NOT the target: the y⁴ coefficient is
    # whatever the solver put there, and it is the only unmatched monomial.
    for Q in (cvx_Q, msk_Q):
        assert Q is not None
        # 2·Q[1,2] == 3 is the only row that touches the target coefficient.
        assert 2.0 * Q[1, 2] == pytest.approx(3.0, abs=1e-6)


@needs_mosek
def test_both_encodings_refuse_the_odd_block_once_no_entry_is_free():
    """With every product constrained, 3y³ SOS is infeasible in both encodings.

    This is the reason the μ policy exists: an odd homogeneous form is
    sign-indefinite, so closing the encoding is necessary but not sufficient.
    """
    cvx_status, _ = _cvxpy_odd_block(with_completion=True)
    msk_status, _ = _mosek_odd_block(with_completion=True)
    assert cvx_status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
    assert msk_status in ("infeasible", "unknown")


@needs_mosek
def test_the_mosek_encoding_adds_exactly_one_row_per_outside_monomial():
    gram_map = build_gram_constraints(ODD_BASIS, ODD_SUPPORT, 1)
    zero_pairs = build_gram_completion_map(ODD_BASIS, ODD_SUPPORT, 1)
    base = blocks_from_assembly(ODD_M, ODD_SUPPORT, gram_map, (), 0.0,
                                dim=3, name="a")
    closed = blocks_from_assembly(ODD_M, ODD_SUPPORT, gram_map, (), 0.0,
                                  dim=3, name="a", zero_pairs=zero_pairs)
    assert len(closed.rows) == len(base.rows) + len(zero_pairs)
    extra_row = closed.rows[-1]
    assert extra_row.a_cols.size == 0 and extra_row.rhs == 0.0
    assert list(zip(extra_row.pairs_k.tolist(),
                    extra_row.pairs_l.tolist())) == [(2, 2)]


def test_the_cvxpy_encoding_adds_exactly_one_constraint_per_outside_monomial():
    n_without = len(_cvxpy_constraints(with_completion=False))
    n_with = len(_cvxpy_constraints(with_completion=True))
    assert n_with - n_without == len(
        build_gram_completion_map(ODD_BASIS, ODD_SUPPORT, 1))


def _cvxpy_constraints(with_completion: bool):
    """The CVXPY constraint list of the miniature block (no solve)."""
    alpha = cp.Variable(1)
    Q = cp.Variable((3, 3), symmetric=True)
    gram_map = build_gram_constraints(ODD_BASIS, ODD_SUPPORT, 1)
    cons = [Q >> 0]
    for r, mono in enumerate(ODD_SUPPORT):
        pairs = gram_map.get(mono, [])
        coef = ODD_M[r, :] @ alpha
        cons.append(coef == (sum(Q[i, j] for i, j in pairs) if pairs else 0))
    if with_completion:
        for pairs in build_gram_completion_map(
                ODD_BASIS, ODD_SUPPORT, 1).values():
            cons.append(sum(Q[i, j] for i, j in pairs) == 0)
    return cons


# ---------------------------------------------------------------------------
# The μ policy on the real design
# ---------------------------------------------------------------------------

@needs_mosek
def test_power_rate_w_blocks_are_designed_at_the_doubled_mu():
    res = _design(create_power_rate_spec(), 2)
    assert res.feasible, res.message
    w = [e for e in res.gram_certificates if e["block"] == "W"]
    v = [e for e in res.gram_certificates if e["block"] == "V"]
    assert len(w) == 2 and len(v) == 2
    assert {e["mu"] for e in v} == {(2,)} and {e["mu"] for e in w} == {(4,)}
    assert {e["mu_policy"] for e in w} == {MU_POLICY_W_DOUBLED}
    assert {e["mu_policy"] for e in v} == {MU_POLICY_DESIGN}
    assert {e["y_degree"] for e in w} == {6}
    assert {e["basis"][0] for e in w} == {(3,)}
    assert {e["eps"] for e in w} == {1e-4}


@needs_mosek
def test_both_backends_lift_the_odd_w_block_to_the_same_mu():
    """The CVXPY encoding and the MOSEK one pose the SAME W formulation.

    A fallback may change the encoding, never the formulation, so the μ the
    per-octant W assembly is built at must be identical on both backends.
    """
    spec = create_power_rate_spec()
    seen = {}
    for backend in ("MOSEK_DIRECT", "CLARABEL"):
        opt = JointSOSOptimizer(weights_r=spec.weights_r, solver=backend,
                                allow_solver_fallback=False)
        mus = []
        original = opt._compute_w_coefficient_matrix_weighted_sphere

        def spy(term_specs, mu, signs, dyn, _o=original, _m=mus):
            _m.append(tuple(int(v) for v in mu))
            return _o(term_specs, mu, signs, dyn)

        opt._compute_w_coefficient_matrix_weighted_sphere = spy
        res = opt.find_lyapunov_with_dynamics(
            degree_m=2, dynamics_gf=spec.dynamics_gf,
            active_terms=_library(spec, 2))
        assert res.feasible, f"{backend}: {res.message}"
        seen[backend] = mus
        # (2,) appears once: the probe that DETECTS the odd y-degree. Every
        # per-octant emission after it is at the doubled μ.
        assert mus.count((2,)) == 1
        assert set(mus[1:]) == {(4,)}
    assert seen["MOSEK_DIRECT"] == seen["CLARABEL"]


@needs_mosek
@pytest.mark.parametrize("factory,m", [
    (create_power_rate_spec, 2), (create_sta_spec, 3), (create_levant3d_spec, 5),
], ids=["power_rate", "sta", "levant3d"])
def test_no_design_block_leaves_a_monomial_outside_the_support(factory, m):
    """The structural invariant, on real designs: no free Gram entry anywhere.

    The support the design emits rows for is ``get_monomial_basis(n, y_degree)``
    (every monomial of total degree <= the block's y-degree), so any basis
    product outside it would carry no row.
    """
    res = _design(factory(), m)
    assert res.feasible, res.message
    for entry in res.gram_certificates:
        basis = [tuple(int(e) for e in b) for b in entry["basis"]]
        n = len(basis[0])
        support = get_monomial_basis(n, int(entry["y_degree"]))
        assert build_gram_completion_map(basis, support, n) == {}, entry["name"]


@needs_mosek
def test_the_power_rate_design_bundle_is_now_the_exact_witness():
    """The whole point: no re-solve, and no ``gram_bundle_margin_missing``."""
    spec = create_power_rate_spec()
    res = _design(spec, 2, epsilon=1e-4)
    assert res.feasible, res.message
    out = run_exact_phase(res.V_gf, [spec.dynamics_gf], 1, epsilon=1e-4,
                          gram_bundle=res)
    assert out.exact_pass is True
    assert out.witness_source == "design_bundle"
    assert len(out.witness_attempts) == 1
    assert out.witness_attempts[0]["shortfall"] == ""
    assert out.worst_charge_ratio < 1e-3


@needs_mosek
def test_the_design_screen_reports_the_mu_policy_per_block():
    res = _design(create_power_rate_spec(), 2, epsilon=1e-4)
    blocks = res.design_screen["gram"]["blocks"]
    assert res.design_screen["ok"] is True
    assert all(b["gateable"] and b["ok"] for b in blocks)
    policies = {b["block"]: {x["mu_policy"] for x in blocks
                             if x["block"] == b["block"]}
                for b in blocks}
    assert policies["V"] == {MU_POLICY_DESIGN}
    assert policies["W"] == {MU_POLICY_W_DOUBLED}
    assert {tuple(b["mu"]) for b in blocks if b["block"] == "W"} == {(4,)}
