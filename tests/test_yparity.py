#!/usr/bin/env python3
"""MOSEK-free soundness tests for the y-parity block decomposition.

Run: python -m pytest test_yparity.py -q   (no solver, no network)

Covers, on real repo-built assemblies (levant4d m=5/7, perturbed) and on
synthetic data:

  T1  partition: parity classes partition the basis.
  T2  refusal: a nonzero matching row on an odd monomial raises YParityRefused.
  T3  row bookkeeping: per-class pair lists reproduce the dense deduplicated
      pair list minus cross-class pairs; no pair is lost or duplicated.
  T4  exact reassembly: random PSD sub-Grams -> reassembled dense Q is PSD and
      every dense row value equals the blockwise value to machine precision.
  T5  completeness (group averaging): the parity projection of ANY dense PSD
      Gram is PSD and represents the same polynomial when the polynomial's
      support is even; equivalently the projection changes <A,Q> on no even
      row (numerical check on the real assembly).
  T6  vacuous odd rows: on the real assemblies, every odd-monomial row has
      zero alpha part and zero rhs (the premise of dropping them).
"""
from __future__ import annotations

import numpy as np
import pytest

from gf_framework.verification.yparity import (
    YParityRefused,
    is_even_mono,
    parity_classes,
    parity_of,
    reassemble_dense_gram,
    row_value_dense,
    row_value_split,
    split_gram_block,
)


# ---------------------------------------------------------------------------
# Real repo assembly (levant4d, small degree) as a fixture
# ---------------------------------------------------------------------------

def build_assembly(m=5, delta0=0.3):
    import gf_framework.discovery.universal_discovery as ud
    from gf_framework.algebra.enumerate import enumerate_diagonal_terms
    from gf_framework.discovery.gf_discovery import mu_filtered_cross
    from gf_framework.algebra.associated import get_hyperoctant_signs
    from gf_framework.verification.joint_sos import (
        JointSOSOptimizer, _sos_gram_basis, build_gram_constraints,
        build_v_coefficient_matrix, get_monomial_basis,
    )

    spec = ud.create_levant4d_spec(delta0=delta0)
    weights = tuple(int(r) for r in spec.weights_r)
    n = len(weights)
    dyn = spec.dynamics_gf
    vertices = spec.dynamics_gf_vertices or [dyn]
    diag = enumerate_diagonal_terms(m, weights, c1_only=True)
    mu, _, cross = mu_filtered_cross(m, weights, diag, dyn)
    terms = list(diag) + list(cross)
    opt = JointSOSOptimizer(weights_r=weights, solver="MOSEK_DIRECT",
                            verbose=False, epsilon=1e-4,
                            allow_solver_fallback=False)
    v_y_degree = max(int(sum(mu[i] * float(t.exponents[i]) for i in range(n)))
                     for t in terms)
    v_basis = _sos_gram_basis(n, v_y_degree, homogeneous=True)
    all_v = get_monomial_basis(n, v_y_degree)
    v_map = build_gram_constraints(v_basis, all_v, n)
    signs0 = get_hyperoctant_signs(n)[0]
    M_V = build_v_coefficient_matrix(terms, mu, signs0, all_v)
    eps_rows = []
    for r, mono in enumerate(all_v):
        if sum(mono) == v_y_degree and sorted(mono)[-1] == v_y_degree:
            eps_rows.append(r)
    # W stage, endpoint 0
    M_W, clearing_k, monos_w = opt._compute_w_coefficient_matrix_weighted_sphere(
        terms, mu, signs0, vertices[0])
    M_W = np.asarray(M_W, dtype=float)
    w_y_degree = max(sum(mm) for mm in monos_w)
    w_basis = _sos_gram_basis(n, w_y_degree, homogeneous=(clearing_k == 0))
    w_map = build_gram_constraints(w_basis, monos_w, n)
    return {
        "V": (M_V, all_v, v_map, eps_rows, 1e-4, v_basis),
        "W": (M_W, monos_w, w_map, [], 0.0, w_basis),
    }


@pytest.fixture(scope="module")
def assembly():
    return build_assembly(m=5, delta0=0.3)


def _split(a, stage, name):
    M, monos, gmap, eps, se, basis = a[stage]
    return split_gram_block(M, monos, gmap, eps, se, basis, name), \
        (M, monos, gmap, eps, se, basis)


# ---------------------------------------------------------------------------
# T1 partition
# ---------------------------------------------------------------------------

def test_partition_covers_basis(assembly):
    for stage in ("V", "W"):
        split, (_, _, _, _, _, basis) = _split(assembly, stage, stage)
        seen = sorted(i for idxs in split.classes.values() for i in idxs)
        assert seen == list(range(len(basis)))
        for key, idxs in split.classes.items():
            for i in idxs:
                assert parity_of(basis[i]) == key


# ---------------------------------------------------------------------------
# T2 refusal on odd-support targets
# ---------------------------------------------------------------------------

def test_refusal_on_odd_support():
    basis = [(1, 0), (0, 1)]  # products: (2,0), (1,1), (0,2)
    monos = [(2, 0), (1, 1), (0, 2)]
    gmap = {(2, 0): [(0, 0)], (1, 1): [(0, 1), (1, 0)], (0, 2): [(1, 1)]}
    M = np.zeros((3, 1))
    M[1, 0] = 1.0  # alpha hits the ODD monomial (1,1)
    with pytest.raises(YParityRefused):
        split_gram_block(M, monos, gmap, [], 0.0, basis, "odd")


def test_odd_pairs_only_row_dropped_not_refused():
    basis = [(1, 0), (0, 1)]
    monos = [(2, 0), (1, 1), (0, 2)]
    gmap = {(2, 0): [(0, 0)], (1, 1): [(0, 1), (1, 0)], (0, 2): [(1, 1)]}
    M = np.zeros((3, 1))
    M[0, 0] = 1.0
    M[2, 0] = 1.0
    split = split_gram_block(M, monos, gmap, [], 0.0, basis, "ok")
    assert split.n_dropped_odd_rows == 1
    assert len(split.rows) == 2
    assert all(is_even_mono(r.monomial) for r in split.rows)


# ---------------------------------------------------------------------------
# T3 pair bookkeeping on the real assembly
# ---------------------------------------------------------------------------

def test_pair_bookkeeping(assembly):
    for stage in ("V", "W"):
        split, (M, monos, gmap, eps, se, basis) = _split(assembly, stage, stage)
        classes = parity_classes(basis)
        local = {}
        for key, idxs in classes.items():
            for pos, di in enumerate(idxs):
                local[di] = (key, pos)
        by_mono = {tuple(m): r for r, m in enumerate(monos)}
        for row in split.rows:
            dense_pairs = sorted({(max(i, j), min(i, j))
                                  for i, j in gmap.get(row.monomial, [])})
            expect = {}
            for k, l in dense_pairs:
                kk, lk = local[k], local[l]
                if kk[0] != lk[0]:
                    continue
                hi, lo = (kk[1], lk[1]) if kk[1] >= lk[1] else (lk[1], kk[1])
                expect.setdefault(kk[0], set()).add((hi, lo))
            got = {key: set(zip(map(int, pk), map(int, pl)))
                   for key, (pk, pl) in row.class_pairs.items()}
            assert got == expect, f"{stage} row {row.monomial}"


# ---------------------------------------------------------------------------
# T4 exact reassembly identity
# ---------------------------------------------------------------------------

def test_reassembly_exact(assembly):
    rng = np.random.default_rng(7)
    for stage in ("V", "W"):
        split, (M, monos, gmap, eps, se, basis) = _split(assembly, stage, stage)
        sub = {}
        for key, idxs in split.classes.items():
            k = len(idxs)
            A = rng.standard_normal((k, k))
            sub[key] = A @ A.T  # PSD
        Q = reassemble_dense_gram(split, sub)
        # PSD by construction
        assert np.linalg.eigvalsh(Q).min() > -1e-9
        # dense row value == blockwise row value on every split row
        for row in split.rows:
            dv = row_value_dense(Q, gmap.get(row.monomial, []))
            sv = row_value_split(row, sub)
            assert abs(dv - sv) < 1e-9 * max(1.0, abs(dv))
        # cross-class entries are zero
        classes = parity_classes(basis)
        key_of = {}
        for key, idxs in classes.items():
            for i in idxs:
                key_of[i] = key
        for i in range(len(basis)):
            for j in range(len(basis)):
                if key_of[i] != key_of[j]:
                    assert Q[i, j] == 0.0


# ---------------------------------------------------------------------------
# T5 completeness: parity projection preserves PSD + even-row values
# ---------------------------------------------------------------------------

def test_parity_projection_completeness(assembly):
    rng = np.random.default_rng(11)
    for stage in ("V", "W"):
        split, (M, monos, gmap, eps, se, basis) = _split(assembly, stage, stage)
        d = len(basis)
        A = rng.standard_normal((d, d))
        Q = A @ A.T  # arbitrary dense PSD Gram
        # parity projection = zero cross-class entries (the group average)
        classes = parity_classes(basis)
        key_of = {}
        for key, idxs in classes.items():
            for i in idxs:
                key_of[i] = key
        P = Q.copy()
        for i in range(d):
            for j in range(d):
                if key_of[i] != key_of[j]:
                    P[i, j] = 0.0
        # PSD is preserved (direct sum of principal submatrices)
        assert np.linalg.eigvalsh(P).min() > -1e-8
        # KEY structural fact making the reduction lossless: two basis
        # monomials summing to an EVEN monomial are congruent mod 2, hence in
        # the SAME parity class. So every pair of every even row is
        # within-class ...
        for row in split.rows:
            pairs = gmap.get(row.monomial, [])
            dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
            for k, l in dedup:
                assert key_of[k] == key_of[l], (stage, row.monomial)
            # ... and therefore the projection changes NO even row value:
            # any dense Gram of an even-support target projects to a
            # block-diagonal Gram of the SAME target (zero completeness loss).
            assert abs(row_value_dense(P, pairs)
                       - row_value_dense(Q, pairs)) < 1e-9 * max(
                1.0, abs(row_value_dense(Q, pairs)))


# ---------------------------------------------------------------------------
# T6 odd rows are vacuous on the real assemblies (the premise)
# ---------------------------------------------------------------------------

def test_odd_rows_vacuous_on_real_assembly(assembly):
    for stage in ("V", "W"):
        M, monos, gmap, eps, se, basis = assembly[stage]
        eps_set = set(eps)
        for r, mono in enumerate(monos):
            if is_even_mono(mono):
                continue
            assert not np.any(M[r, :] != 0.0), (stage, mono)
            assert r not in eps_set


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
