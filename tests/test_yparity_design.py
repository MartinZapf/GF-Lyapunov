"""WP P5-04: the y-parity Gram block decomposition on the DESIGN path.

``tests/test_yparity.py`` pins the decomposition itself (partition, refusal,
row bookkeeping, exact reassembly, group averaging) MOSEK-free.
``tests/test_support_guard.py`` pins the symbolic even-support premise. This
file pins the WIRING: that both design encodings pose the SAME formulation with
it, that the Gram bundle it produces is the same object a dense design
produces, that the P2-09 completion invariant survives the split, and that the
evenness refusal is still reachable and still falls back to the dense encoding.

S1 discipline (``artifacts/campaign_2026-08/S1_validation/README.md``): a
design produced with blocked Grams is re-gated UNREDUCED. The system-level half
of that is measured in the WP record, not here; what is pinned here is that the
bundle the unreduced gate consumes is unchanged by the blocking.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import enumerate_diagonal_terms
from gf_framework.algebra.support_guard import (
    GuardUnreadable,
    OddLiftRefusal,
    check_even_support,
    gf_exponent_support,
)
from gf_framework.discovery.gf_discovery import mu_filtered_cross
from gf_framework.verification import joint_sos as js
from gf_framework.verification.joint_sos import (
    JointSOSOptimizer,
    build_gram_completion_map,
    build_gram_constraints,
    build_v_coefficient_matrix,
    gram_bundle_entry_hash,
    get_y_parity_blocks_default,
    set_y_parity_blocks_default,
    v_epsilon_row_indices,
)
from gf_framework.verification.mosek_direct import (
    MOSEK_AVAILABLE,
    blocks_from_assembly,
    blocks_from_split,
)
from gf_framework.verification.yparity import (
    YParityRefused,
    parity_of,
    psd_scalars,
    split_gram_block,
)

mosek_only = pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not available")


def _same_class(basis) -> np.ndarray:
    """Boolean matrix: do basis monomials i and j share a parity class?"""
    keys = sorted({parity_of(b) for b in basis})
    ids = np.array([keys.index(parity_of(b)) for b in basis])
    return np.equal.outer(ids, ids)


def library(name, m, **kw):
    spec = getattr(ud, f"create_{name}_spec")(**kw)
    weights = tuple(int(r) for r in spec.weights_r)
    diag = enumerate_diagonal_terms(m, weights, c1_only=True)
    _mu, _k, cross = mu_filtered_cross(m, weights, diag, spec.dynamics_gf)
    return spec, weights, list(diag) + list(cross)


def design(name, m, *, blocked, solver="MOSEK_DIRECT", epsilon=1e-4, **kw):
    spec, weights, terms = library(name, m, **kw)
    opt = JointSOSOptimizer(
        weights_r=weights, epsilon=epsilon, solver=solver,
        allow_solver_fallback=False, y_parity_blocks=blocked)
    res = opt.find_lyapunov_with_dynamics(
        m, spec.dynamics_gf, active_terms=terms,
        dynamics_gf_vertices=spec.dynamics_gf_vertices)
    return spec, res


# ---------------------------------------------------------------------------
# The default and the opt-out
# ---------------------------------------------------------------------------

def test_the_process_default_is_a_settable_flag_and_the_argument_wins():
    previous = get_y_parity_blocks_default()
    try:
        set_y_parity_blocks_default(True)
        assert JointSOSOptimizer(weights_r=(1,)).y_parity_blocks is True
        assert JointSOSOptimizer(
            weights_r=(1,), y_parity_blocks=False).y_parity_blocks is False
        set_y_parity_blocks_default(False)
        assert JointSOSOptimizer(weights_r=(1,)).y_parity_blocks is False
        assert JointSOSOptimizer(
            weights_r=(1,), y_parity_blocks=True).y_parity_blocks is True
    finally:
        set_y_parity_blocks_default(previous)


# ---------------------------------------------------------------------------
# Block assembly: the split emits fewer PSD scalars and the same rows
# ---------------------------------------------------------------------------

def _v_assembly(name, m, **kw):
    """(M, monomials, gram_map, zero_pairs, eps_rows, basis) of one V block."""
    from gf_framework.algebra.associated import get_hyperoctant_signs
    from gf_framework.verification.joint_sos import (
        _sos_gram_basis, compute_mu_exponents_with_dynamics, get_monomial_basis)

    spec, weights, terms = library(name, m, **kw)
    _k, mu = compute_mu_exponents_with_dynamics(
        term_specs=terms, weights_r=weights, dynamics_gf=spec.dynamics_gf,
        ensure_even_exponents=True)
    n = len(weights)
    v_y_degree = max(int(sum(mu[i] * float(t.exponents[i]) for i in range(n)))
                     for t in terms)
    monos = get_monomial_basis(n, v_y_degree)
    basis = _sos_gram_basis(n, v_y_degree, homogeneous=True)
    signs = get_hyperoctant_signs(n)[0]
    M = build_v_coefficient_matrix(terms, mu, signs, monos)
    gram_map = build_gram_constraints(basis, monos, n)
    zero_pairs = build_gram_completion_map(basis, monos, n)
    eps_rows = v_epsilon_row_indices(monos, v_y_degree, n)
    return M, monos, gram_map, zero_pairs, eps_rows, basis


@pytest.mark.parametrize("name,m", [("levant3d", 5), ("pidsmc", 6)])
def test_the_split_keeps_every_within_class_entry_in_at_least_one_row(name, m):
    M, monos, gram_map, zero_pairs, eps_rows, basis = _v_assembly(name, m)
    split = split_gram_block(M, monos, gram_map, eps_rows, 1e-4, basis,
                             "V_oct0", zero_pairs=zero_pairs)
    # every entry of every sub-Gram is touched by some row: this is the P2-09
    # invariant ("no free Gram entry") transported through the decomposition.
    covered = {key: set() for key in split.class_order}
    for row in split.rows:
        for key, (pk, pl) in row.class_pairs.items():
            for k, l in zip(pk, pl):
                covered[key].add((int(k), int(l)))
    for key in split.class_order:
        d = split.sub_dims[key]
        assert covered[key] == {(k, l) for k in range(d) for l in range(k + 1)}


@pytest.mark.parametrize("name,m", [("levant3d", 5), ("pidsmc", 6)])
def test_the_split_strictly_reduces_psd_scalars(name, m):
    M, monos, gram_map, zero_pairs, eps_rows, basis = _v_assembly(name, m)
    split = split_gram_block(M, monos, gram_map, eps_rows, 1e-4, basis,
                             "V_oct0", zero_pairs=zero_pairs)
    assert split.blocked_psd_scalars < split.dense_psd_scalars
    assert split.dense_psd_scalars == psd_scalars(len(basis))
    assert sum(len(v) for v in split.classes.values()) == len(basis)


def test_blocks_from_split_names_and_offsets_are_self_consistent():
    M, monos, gram_map, zero_pairs, eps_rows, basis = _v_assembly("levant3d", 5)
    split = split_gram_block(M, monos, gram_map, eps_rows, 1e-4, basis,
                             "V_oct0", zero_pairs=zero_pairs)
    emitted = blocks_from_split(split)
    assert [b.dim for b in emitted] == [split.sub_dims[k]
                                        for k in split.class_order]
    assert [b.name for b in emitted] == [
        "V_oct0#p" + "".join(str(int(e)) for e in k) for k in split.class_order]
    # every row is owned by the FIRST sub-block, and every offset lands inside
    # the group.
    assert all(not b.rows for b in emitted[1:])
    for row in emitted[0].rows:
        for off, pk, pl in row.extra:
            assert 0 < off < len(emitted)
            assert int(max(pk)) < emitted[off].dim


def test_the_dense_and_split_encodings_agree_row_for_row():
    """On a block-diagonal Q the split row values equal the dense ones.

    This is the reduction stated as an identity: dropping the cross-class
    entries from a row changes nothing when those entries are zero, and the
    odd-monomial rows the split drops are identically 0 = 0 there.
    """
    from gf_framework.verification.yparity import row_value_dense, row_value_split

    M, monos, gram_map, zero_pairs, eps_rows, basis = _v_assembly("levant3d", 5)
    split = split_gram_block(M, monos, gram_map, eps_rows, 1e-4, basis,
                             "V_oct0", zero_pairs=zero_pairs)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(len(basis), len(basis)))
    Qb = np.where(_same_class(basis), A @ A.T, 0.0)
    sub = {k: Qb[np.ix_(split.classes[k], split.classes[k])]
           for k in split.class_order}

    by_mono = {}
    for row in split.rows:
        by_mono[row.monomial] = by_mono.get(row.monomial, 0.0) + row_value_split(
            row, sub)

    checked = 0
    for mono in sorted(set(gram_map) | set(zero_pairs)):
        pairs = gram_map.get(mono) or zero_pairs.get(mono) or []
        if not pairs:
            continue
        dense = row_value_dense(Qb, pairs)
        assert by_mono.get(mono, 0.0) == pytest.approx(dense, abs=1e-9), mono
        checked += 1
    assert checked > 0
    # and the odd rows really were vacuous on the assembled problem
    assert split.n_dropped_odd_rows > 0
    # A cross-class pair (b, c) has b + c ODD, so it can only ever appear on an
    # odd monomial, whose whole row is dropped. No pair is therefore ever
    # dropped from a SURVIVING row: the counter is structurally 0, and that is
    # why "drop the cross-class entries" costs no even-row coefficient.
    assert split.n_dropped_cross_pairs == 0


# ---------------------------------------------------------------------------
# The evenness refusal (still needed, still typed)
# ---------------------------------------------------------------------------

def test_power_rate_W_is_odd_at_the_design_mu_and_even_at_the_doubled_mu():
    """The refusal is not dead code: it is exactly what P2-09's 2μ removes."""
    spec, weights, terms = library("power_rate", 2)
    dyn = gf_exponent_support(spec.dynamics_gf_vertices or [spec.dynamics_gf])
    lib = [tuple(t.exponents) for t in terms]
    with pytest.raises(OddLiftRefusal) as exc:
        check_even_support(mu=(2,), weights_r=weights, library_exponents=lib,
                           dynamics_vertices=dyn)
    assert exc.value.violations[0].stage == "W"
    # at the W μ the design actually uses (2μ, WP P2-09) the premise holds
    rep = check_even_support(mu=(4,), weights_r=weights, library_exponents=lib,
                             dynamics_vertices=dyn)
    assert rep.ok


def test_the_symbolic_refusal_turns_the_whole_design_dense(monkeypatch):
    import gf_framework.algebra.support_guard as sg
    from gf_framework.algebra.support_guard import OddLift

    def refuse(**kwargs):
        raise OddLiftRefusal([OddLift(stage="W", obj="fake", variable=0,
                                      exponent=Fraction(1, 2),
                                      lift=Fraction(1))])

    monkeypatch.setattr(sg, "check_even_support", refuse)
    opt = JointSOSOptimizer(weights_r=(2, 1), y_parity_blocks=True)
    meta = opt._plan_y_parity(
        term_specs=library("sta", 3)[2], mu=(4, 2), w_mu=(4, 2),
        vertices=[library("sta", 3)[0].dynamics_gf], clearing_power=0)
    assert meta["v_blocked"] is False and meta["w_blocked"] is False
    assert meta["refusal"] == "odd_lift_v"


def test_an_unreadable_dynamics_object_declines_rather_than_assuming():
    class Term:
        pass

    class GF:
        terms = [Term()]

    with pytest.raises(GuardUnreadable):
        gf_exponent_support([[GF()]])


def test_the_assembled_row_guard_still_refuses_an_odd_support():
    basis = [(0,), (1,)]
    monos = [(0,), (1,), (2,)]
    gram_map = {(0,): [(0, 0)], (1,): [(0, 1), (1, 0)], (2,): [(1, 1)]}
    M = np.zeros((3, 1))
    M[1, 0] = 1.0          # a nonzero matching row on the ODD monomial y^1
    with pytest.raises(YParityRefused):
        split_gram_block(M, monos, gram_map, [], 0.0, basis, "odd")


# ---------------------------------------------------------------------------
# End to end: the bundle a blocked design produces
# ---------------------------------------------------------------------------

@mosek_only
@pytest.mark.parametrize("name,m", [("sta", 3), ("levant3d", 5), ("pidsmc", 6)])
def test_a_blocked_design_emits_the_same_bundle_shape_as_a_dense_one(name, m):
    _spec, dense = design(name, m, blocked=False)
    _spec, blocked = design(name, m, blocked=True)
    assert dense.feasible and blocked.feasible
    assert blocked.y_parity["v_blocked"] and blocked.y_parity["w_blocked"]
    assert (blocked.y_parity["psd_scalars_emitted"]
            < blocked.y_parity["psd_scalars_dense"])
    assert (blocked.y_parity["psd_scalars_dense"]
            == dense.y_parity["psd_scalars_dense"])
    d_bundle, b_bundle = dense.gram_certificates, blocked.gram_certificates
    assert len(d_bundle) == len(b_bundle)
    for d, b in zip(d_bundle, b_bundle):
        assert d["name"] == b["name"]
        assert d["dim"] == b["dim"]
        assert d["basis"] == b["basis"]
        assert np.asarray(b["gram"]).shape == np.asarray(d["gram"]).shape
        assert b["hash"] == gram_bundle_entry_hash(b)
        assert b["y_parity"]["mode"] == "blocked"
        assert d["y_parity"]["mode"] == "dense"


@mosek_only
def test_the_blocked_bundle_gram_is_block_diagonal_over_the_parity_classes():
    _spec, res = design("levant3d", 5, blocked=True)
    for entry in res.gram_certificates:
        Q = np.asarray(entry["gram"], dtype=float)
        same = _same_class(entry["basis"])
        assert np.all(Q[~same] == 0.0)
        # and it is a genuine Gram of the entry's target, which is what the
        # design screen and the exact phase consume.
        assert float(np.linalg.eigvalsh(Q).min()) > -1e-9


@mosek_only
def test_a_blocked_design_still_passes_the_design_screen_on_every_block():
    for name, m in [("power_rate", 2), ("levant3d", 5), ("pidsmc", 6)]:
        _spec, res = design(name, m, blocked=True)
        screen = res.design_screen
        assert screen["ok"], (name, screen["detail"])
        gram = screen["gram"]
        assert gram["n_blocks_failed"] == 0
        assert gram["n_blocks_gateable"] == gram["n_blocks"]
        assert all(b["y_parity_mode"] == "blocked" for b in gram["blocks"])


@mosek_only
def test_power_rate_is_blocked_now_that_its_W_mu_is_doubled():
    _spec, res = design("power_rate", 2, blocked=True)
    assert res.y_parity["v_blocked"] and res.y_parity["w_blocked"]
    assert res.y_parity["refusal"] == ""
    w = [e for e in res.gram_certificates if e["block"] == "W"]
    assert w and all(e["mu_policy"] == "w_mu_doubled_odd_y_degree" for e in w)


@mosek_only
def test_the_unreduced_exact_check_passes_on_a_blocked_design():
    """S1 in miniature: reduced design -> UNREDUCED exact check."""
    from gf_framework.verification.exact_phase import run_exact_phase

    for name, m in [("sta", 3), ("levant3d", 5)]:
        spec, res = design(name, m, blocked=True)
        verts = spec.dynamics_gf_vertices or [spec.dynamics_gf]
        exact = run_exact_phase(res.V_gf, verts, len(spec.weights_r),
                                epsilon=1e-4, gram_bundle=res)
        assert exact.exact_pass, name
        assert exact.witness_source == "design_bundle"
        assert exact.worst_charge_ratio < 1.0


@mosek_only
def test_both_backends_pose_the_same_blocked_formulation(monkeypatch):
    """A MOSEK_DIRECT -> CVXPY fallback changes the ENCODING, not the SDP.

    The two backends solve the same feasibility problem and land on different
    interior points, so their alphas differ. What must agree is the FORMULATION:
    the same blocks, split into the same parity classes, carrying the same
    number of rows. That is what is compared here, by spying on the single
    function both encodings route the decomposition through.
    """
    import gf_framework.verification.yparity as yp

    seen = {}
    original = yp.split_gram_block

    def record(tag):
        def spy(*args, **kwargs):
            split = original(*args, **kwargs)
            # NOT the block name: the two encodings have always named their
            # blocks differently ("V_oct0" vs "Q_V_0"). What must match is the
            # decomposition itself.
            seen.setdefault(tag, []).append(
                (tuple(split.class_order),
                 tuple(split.sub_dims[k] for k in split.class_order),
                 len(split.rows), split.n_dropped_odd_rows))
            return split
        return spy

    monkeypatch.setattr(yp, "split_gram_block", record("direct"))
    _spec, direct = design("sta", 3, blocked=True, solver="MOSEK_DIRECT")
    monkeypatch.setattr(yp, "split_gram_block", record("cvxpy"))
    _spec, cvxpy = design("sta", 3, blocked=True, solver="MOSEK")

    assert direct.feasible and cvxpy.feasible
    assert cvxpy.y_parity["v_blocked"] and cvxpy.y_parity["w_blocked"]
    assert seen["direct"] and seen["direct"] == seen["cvxpy"]
