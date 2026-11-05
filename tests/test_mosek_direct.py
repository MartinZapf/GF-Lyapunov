"""Direct MOSEK Task-API backend: convention pins + parity with the CVXPY path.

Parity philosophy: interior-point feasibility solutions are NOT unique, so
parity never compares coefficient vectors. Parity = both backends reach the
same verdict AND both candidates pass the same strict gate
(``certify_discovered_gf``). The one place a numeric value IS unique — the
optimal value of the weighted-ℓ₁ program — is compared directly.
"""
from fractions import Fraction

import numpy as np
import pytest

from gf_framework.algebra.enumerate import (
    TermSpec,
    enumerate_cross_terms,
    enumerate_diagonal_terms,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.discovery.universal_discovery import (
    create_example3_spec,
    create_levant3d_spec,
    create_sta_spec,
)
from gf_framework.verification.joint_sos import (
    JointSOSOptimizer,
    build_gram_constraints,
)
from gf_framework.verification.mosek_direct import (
    MOSEK_AVAILABLE,
    GramBlock,
    GramRow,
    blocks_from_assembly,
    solve_joint_sos_direct,
    unpack_lower_triangle,
)
from gf_framework.verification.sparse_sos import atom_scales, design_sparse_lyapunov
from gf_framework.verification.verifier import certify_discovered_gf

needs_mosek = pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not installed")


def _library(spec, m):
    """The same μ-filtered C¹ library discover_gf solves over (scripts/_sparsity_common)."""
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return list(diag) + list(cross)


def _design(spec, m, solver, library, **opt_kwargs):
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver=solver,
                            allow_solver_fallback=False, verbose=False,
                            **opt_kwargs)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf, active_terms=library,
        dynamics_gf_vertices=spec.dynamics_gf_vertices)


def _gate(spec, m, V_gf):
    cert = certify_discovered_gf(
        V_gf, spec.weights_r, m, spec.dynamics_gf, solver="CLARABEL",
        strict_solver=True,
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
        disturbance_bound=spec.disturbance_bound)
    return bool(cert.surface_certified)


# ---------------------------------------------------------------------------
# Convention pins
# ---------------------------------------------------------------------------

def test_unpack_lower_triangle():
    """MOSEK packs the lower triangle column-major: (0,0),(1,0),(2,0),(1,1),..."""
    packed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Q = unpack_lower_triangle(packed, 3)
    expected = np.array([[1.0, 2.0, 3.0],
                         [2.0, 4.0, 5.0],
                         [3.0, 5.0, 6.0]])
    np.testing.assert_allclose(Q, expected)


@needs_mosek
def test_bara_convention_tiny_sos():
    """(x²+x+1)² = x⁴+2x³+3x²+2x+1 with basis [1, x, x²].

    Encodes 0·α − ⟨A,Q⟩ = −t_k per monomial (no alpha), so a feasible solve
    must reproduce the target coefficients from the recovered Gram. The
    independent re-expansion below catches any barA doubling/dedup error: the
    diagnostics residual uses our assumed convention, and the explicit
    z^T Q z convolution uses no MOSEK convention at all.
    """
    basis = [(0,), (1,), (2,)]
    target = {(0,): 1.0, (1,): 2.0, (2,): 3.0, (3,): 2.0, (4,): 1.0}
    monomials = list(target)
    gram_map = build_gram_constraints(basis, monomials, 1)
    rows = []
    for mono in monomials:
        pairs = gram_map[mono]
        dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
        rows.append(GramRow(
            a_cols=np.empty(0, dtype=np.int32),
            a_vals=np.empty(0, dtype=float),
            rhs=-target[mono],
            pairs_k=np.array([p[0] for p in dedup], dtype=np.int32),
            pairs_l=np.array([p[1] for p in dedup], dtype=np.int32)))
    block = GramBlock(dim=3, name="tiny", rows=rows)
    out = solve_joint_sos_direct(
        n_terms=0, blocks=[block], diag_indices=[], scaled_epsilon=0.0,
        coefficient_scale=1.0)
    assert out.status == "optimal", out.message
    assert out.gram_residual_l1 < 1e-6
    assert out.gram_min_eig > -1e-8
    # Independent check: expand z^T Q z by convolution and compare with target.
    Q = out.grams[0]
    achieved = {}
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            mono = (bi[0] + bj[0],)
            achieved[mono] = achieved.get(mono, 0.0) + Q[i, j]
    for mono, coef in target.items():
        assert achieved[mono] == pytest.approx(coef, abs=1e-6)


def test_blocks_from_assembly_skips_trivial_rows():
    """All-zero M row + no pairs + zero rhs is the vacuous 0 == 0 constraint."""
    M = np.array([[0.0, 0.0], [1.0, -1.0]])
    monomials = [(5, 0), (2, 0)]                  # (5,0) unreachable by pairs
    gram_map = {(2, 0): [(0, 0)], (5, 0): []}
    block = blocks_from_assembly(M, monomials, gram_map, [], 0.0, dim=1, name="t")
    assert len(block.rows) == 1
    assert block.rows[0].rhs == 0.0
    np.testing.assert_array_equal(block.rows[0].a_vals, [1.0, -1.0])


# ---------------------------------------------------------------------------
# Parity with the CVXPY path (verdict + strict gate, never coefficients)
# ---------------------------------------------------------------------------

@needs_mosek
@pytest.mark.parametrize("make_spec,m", [
    (create_sta_spec, 3),
    (create_example3_spec, 5),
    (create_levant3d_spec, 5),
], ids=["sta_m3", "example3_m5", "levant3d_m5"])
def test_parity_feasible_and_certified(make_spec, m):
    spec = make_spec()
    library = _library(spec, m)
    res_cvxpy = _design(spec, m, "MOSEK", library)
    res_direct = _design(spec, m, "MOSEK_DIRECT", library)
    assert res_cvxpy.feasible, res_cvxpy.message
    assert res_direct.feasible, res_direct.message
    assert res_direct.solver_used == "MOSEK_DIRECT"
    assert "MOSEK_DIRECT" in res_direct.solver_status
    # The runtime convention assertion: residual recomputed from the Grams.
    assert res_direct.gram_residual_l1 is not None
    assert res_direct.gram_residual_l1 < 1e-5
    assert _gate(spec, m, res_cvxpy.V_gf)
    assert _gate(spec, m, res_direct.V_gf)


@needs_mosek
def test_parity_weighted_l1_objective_value():
    """The ℓ₁ optimum VALUE is unique even when the argmin is not."""
    spec = create_sta_spec()
    m = 6
    library = _library(spec, m)
    w = atom_scales(library, spec.weights_r, spec.dynamics_gf)
    vals = {}
    for solver in ("MOSEK", "MOSEK_DIRECT"):
        res = _design(spec, m, solver, library, l1_weights=w)
        assert res.feasible, f"{solver}: {res.message}"
        coefs = np.array([float(t.coefficient) for t in res.V_gf.terms])
        cross = [i for i, s in enumerate(library) if not s.is_diagonal()]
        vals[solver] = float(np.sum(w[cross] * np.abs(coefs[cross])))
    ref = max(abs(vals["MOSEK"]), 1.0)
    assert abs(vals["MOSEK"] - vals["MOSEK_DIRECT"]) <= 1e-3 * ref, vals


@needs_mosek
def test_parity_robust_vertices():
    """±δ endpoint vertices: per-(octant, vertex) W blocks on both backends."""
    spec = create_sta_spec(delta0=0.3)
    m = 3
    assert spec.dynamics_gf_vertices is not None
    library = _library(spec, m)
    for solver in ("MOSEK", "MOSEK_DIRECT"):
        res = _design(spec, m, solver, library)
        assert res.feasible, f"{solver}: {res.message}"
        assert _gate(spec, m, res.V_gf), solver


@needs_mosek
def test_parity_infeasible():
    """Diagonal-only STA library is known infeasible (the cross term is needed)."""
    spec = create_sta_spec()
    m = 3
    diag_only = [t for t in _library(spec, m) if t.is_diagonal()]
    res_cvxpy = _design(spec, m, "MOSEK", diag_only)
    res_direct = _design(spec, m, "MOSEK_DIRECT", diag_only)
    assert not res_cvxpy.feasible
    assert not res_direct.feasible
    assert "infeasible" in res_direct.solver_status.lower()


@needs_mosek
def test_not_implemented_options():
    spec = create_sta_spec()
    library = _library(spec, 3)
    for kwargs in ({"maximize_diagonal": True},
                   {"domination_pairs": [(0, 1)]}):
        opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                                allow_solver_fallback=False, **kwargs)
        with pytest.raises(NotImplementedError):
            opt.find_lyapunov_with_dynamics(
                degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=library)


@needs_mosek
def test_sparse_design_end_to_end_direct():
    """design_sparse_lyapunov drives the direct backend through the rl1 loop.

    Support size may legitimately differ from the CVXPY run by a term
    (solution non-uniqueness); the assertion is certification, not equality.
    """
    spec = create_sta_spec()
    m = 6
    library = _library(spec, m)
    res = design_sparse_lyapunov(
        weights_r=spec.weights_r, degree_m=m, dynamics_gf=spec.dynamics_gf,
        active_terms=library, solver="MOSEK_DIRECT",
        gate_fn=lambda V: certify_discovered_gf(
            V, spec.weights_r, m, spec.dynamics_gf, solver="CLARABEL",
            strict_solver=True))
    assert res.feasible, res.message
    assert res.gated, res.message
    assert 0 < res.n_terms_sparse < len(library)
