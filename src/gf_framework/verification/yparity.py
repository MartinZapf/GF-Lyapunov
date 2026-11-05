#!/usr/bin/env python3
"""Exact y-parity block decomposition of per-octant SOS Gram problems.

Promoted into the package by WP P5-04 from
``artifacts/campaign_2026-08/E4_levant4d/structure/prototype/yparity.py``
(campaign C16 successor; STRUCTURE track, 2026-08-03). Nothing here calls a
solver: the module is pure bookkeeping over an already assembled block, and
``tests/test_yparity.py`` runs MOSEK-free.

S1 discipline (``artifacts/campaign_2026-08/S1_validation/README.md``) applies
unchanged: the blocking is a DESIGN-time change of encoding only. Every
certificate designed on blocked Grams is re-gated on the UNREDUCED strict gate
and the UNREDUCED exact check, and the reduction is lossless by the
group-averaging argument below, not a relaxation.

Mathematical content
--------------------
Per octant, the design targets (V stage and each endpoint W stage) are
polynomials in y whose support lies entirely in the EVEN lattice (2Z)^n:
every library term image has y-exponents mu_i * rho_i = 2 r_i rho_i with
rho_i in (1/r_i) Z (the mu-lattice), the epsilon rows are pure even powers,
and the W products/clearing stay on the lattice. Hence each target p is
invariant under the sign-flip group G = (Z/2)^n acting on y.

G acts on the Gram basis monomials diagonally: for g = diag(g_1..g_n),
z_b(gy) = chi_b(g) z_b(y) with character chi_b(g) = prod g_i^{b_i}. Two basis
monomials b, c carry the same character iff b == c (mod 2). If Q is any PSD
Gram of p, the group average

    Q' = (1/|G|) sum_g D_g Q D_g,   D_g = diag(chi_b(g))

is PSD, is again a Gram of p (each D_g Q D_g is, because p(gy) = p(y)), and
Q'[b, c] = Q[b, c] if b == c (mod 2), else 0. So restricting Q to be
block-diagonal over the PARITY CLASSES of the basis is exact: zero
completeness loss, and any block-diagonal solution is trivially a solution
of the dense problem (soundness). The coefficient-match row of an even
monomial couples the within-class pair sums of ALL classes (one row, several
sub-blocks); rows of odd monomials become vacuous 0 == 0 and are dropped.

Refusal discipline (mirrors S1): if any monomial with a nonzero matching row
(alpha columns or rhs) has an odd coordinate exponent, the decomposition
must NOT be applied; callers fall back to the dense block.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Monomial = Tuple[int, ...]


class YParityRefused(RuntimeError):
    """Decomposition not applicable; caller must use the dense block."""


def parity_of(mono: Sequence[int]) -> Monomial:
    return tuple(int(e) % 2 for e in mono)


def is_even_mono(mono: Sequence[int]) -> bool:
    return all(int(e) % 2 == 0 for e in mono)


def parity_classes(basis: Sequence[Monomial]) -> Dict[Monomial, List[int]]:
    """Partition basis indices by parity class (insertion-ordered)."""
    classes: Dict[Monomial, List[int]] = defaultdict(list)
    for idx, b in enumerate(basis):
        classes[parity_of(b)].append(idx)
    return dict(classes)


@dataclass
class SplitRow:
    """One monomial-matching row split across the parity sub-blocks.

    a_cols/a_vals: alpha columns of the matching matrix (unchanged).
    rhs: unchanged.
    class_pairs: {class_key: (pairs_k, pairs_l)} with LOCAL sub-block
    indices, lower triangle (k >= l), deduplicated. Only classes that
    contribute at least one pair appear.
    """
    monomial: Monomial
    a_cols: np.ndarray
    a_vals: np.ndarray
    rhs: float
    class_pairs: Dict[Monomial, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict)


def psd_scalars(dim: int) -> int:
    """Free scalars of a symmetric ``dim x dim`` PSD variable."""
    d = int(dim)
    return d * (d + 1) // 2


@dataclass
class SplitBlock:
    """Parity-decomposed image of one dense Gram block."""
    name: str
    dense_dim: int
    basis: List[Monomial]
    classes: Dict[Monomial, List[int]]          # class -> dense indices
    sub_dims: Dict[Monomial, int]
    rows: List[SplitRow] = field(default_factory=list)
    n_dropped_odd_rows: int = 0
    n_dropped_cross_pairs: int = 0

    @property
    def class_order(self) -> List[Monomial]:
        """Canonical, hash-free emission order of the parity classes."""
        return sorted(self.classes)

    @property
    def dense_psd_scalars(self) -> int:
        return psd_scalars(self.dense_dim)

    @property
    def blocked_psd_scalars(self) -> int:
        return sum(psd_scalars(d) for d in self.sub_dims.values())

    def record(self) -> Dict[str, object]:
        """Diagnostic block-structure record for the design Gram bundle."""
        order = self.class_order
        return {
            "mode": "blocked",
            "n_classes": len(order),
            "classes": [[list(key), int(self.sub_dims[key])] for key in order],
            "dense_dim": int(self.dense_dim),
            "psd_scalars_dense": int(self.dense_psd_scalars),
            "psd_scalars_blocked": int(self.blocked_psd_scalars),
            "n_dropped_odd_rows": int(self.n_dropped_odd_rows),
            "n_dropped_cross_pairs": int(self.n_dropped_cross_pairs),
        }


def dense_record(dim: int, reason: str = "") -> Dict[str, object]:
    """The ``y_parity`` record of a block that was NOT split."""
    return {
        "mode": "dense",
        "n_classes": 1,
        "classes": [],
        "dense_dim": int(dim),
        "psd_scalars_dense": psd_scalars(dim),
        "psd_scalars_blocked": psd_scalars(dim),
        "n_dropped_odd_rows": 0,
        "n_dropped_cross_pairs": 0,
        "reason": reason,
    }


def split_gram_block(
    M: np.ndarray,
    monomials: Sequence[Monomial],
    gram_map: Dict[Monomial, List[Tuple[int, int]]],
    eps_rows: Sequence[int],
    scaled_epsilon: float,
    basis: Sequence[Monomial],
    name: str,
    zero_pairs: Optional[Dict[Monomial, List[Tuple[int, int]]]] = None,
) -> SplitBlock:
    """Parity-split analogue of mosek_direct.blocks_from_assembly.

    Same inputs as blocks_from_assembly plus the explicit Gram `basis`
    (blocks_from_assembly only receives its dimension). Raises
    YParityRefused if any nonzero matching row sits on an odd monomial.

    ``zero_pairs`` is P2-09's completion map: basis products OUTSIDE the target
    support, which the dense encoding pins to zero so no Gram entry is ever a
    free variable. The invariant is preserved here block by block — an
    out-of-support EVEN monomial keeps an explicit ``<A, Q_c> = 0`` row on each
    class that contributes a within-class pair, and an out-of-support ODD
    monomial needs none because every one of its pairs is cross-class and is
    already identically zero under the block-diagonal restriction. So after the
    split, exactly as before it, every entry of every sub-Gram appears in at
    least one row.
    """
    classes = parity_classes(basis)
    local = {}  # dense index -> (class_key, local index)
    for key, idxs in classes.items():
        for pos, di in enumerate(idxs):
            local[di] = (key, pos)

    eps_set = set(eps_rows)
    out = SplitBlock(
        name=name, dense_dim=len(basis), basis=list(basis), classes=classes,
        sub_dims={k: len(v) for k, v in classes.items()})

    for r, mono in enumerate(monomials):
        a_row = M[r, :]
        cols = np.nonzero(a_row)[0]
        pairs = gram_map.get(tuple(mono), [])
        rhs = float(scaled_epsilon) if r in eps_set else 0.0
        if cols.size == 0 and not pairs and rhs == 0.0:
            continue  # vacuous, as in the dense path
        if not is_even_mono(mono):
            if cols.size > 0 or rhs != 0.0:
                raise YParityRefused(
                    f"{name}: nonzero matching row on odd monomial {mono}; "
                    f"the even-support premise fails, dense block required")
            # pairs only: all pairs are necessarily cross-class (their sum is
            # odd), the block-diagonal restriction makes the row vacuous.
            out.n_dropped_odd_rows += 1
            continue
        # even monomial: split the deduplicated lower-triangle pairs by class
        dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
        per_class: Dict[Monomial, Tuple[List[int], List[int]]] = defaultdict(
            lambda: ([], []))
        for k_idx, l_idx in dedup:
            key_k, loc_k = local[k_idx]
            key_l, loc_l = local[l_idx]
            if key_k != key_l:
                # Cross-class entry: zero under the restriction, so dropping it
                # from the row is the reduction (never an error). Structurally
                # unreachable HERE: b + c is odd when b, c are in different
                # classes, so such a pair only ever appears on an odd monomial,
                # whose whole row is dropped above. The counter therefore reads
                # 0 on every real assembly (pinned in
                # tests/test_yparity_design.py) and exists so that a future
                # non-homogeneous support cannot silently drop a coefficient.
                out.n_dropped_cross_pairs += 1
                continue
            hi, lo = (loc_k, loc_l) if loc_k >= loc_l else (loc_l, loc_k)
            per_class[key_k][0].append(hi)
            per_class[key_k][1].append(lo)
        row = SplitRow(
            monomial=tuple(mono),
            a_cols=cols.astype(np.int32),
            a_vals=a_row[cols].astype(float),
            rhs=rhs,
            class_pairs={
                key: (np.asarray(pk, dtype=np.int32),
                      np.asarray(pl, dtype=np.int32))
                for key, (pk, pl) in per_class.items()},
        )
        if row.a_cols.size == 0 and not row.class_pairs and row.rhs == 0.0:
            continue
        out.rows.append(row)

    # P2-09 completion rows, split the same way. An ODD out-of-support monomial
    # contributes nothing (all its pairs are cross-class, already zero); an EVEN
    # one keeps one <A, Q_c> = 0 row per contributing class.
    for mono, pairs in sorted((zero_pairs or {}).items()):
        if not pairs:
            continue
        if not is_even_mono(mono):
            out.n_dropped_odd_rows += 1
            continue
        dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
        per_class = defaultdict(lambda: ([], []))
        for k_idx, l_idx in dedup:
            key_k, loc_k = local[k_idx]
            key_l, loc_l = local[l_idx]
            if key_k != key_l:
                out.n_dropped_cross_pairs += 1
                continue
            hi, lo = (loc_k, loc_l) if loc_k >= loc_l else (loc_l, loc_k)
            per_class[key_k][0].append(hi)
            per_class[key_k][1].append(lo)
        if not per_class:
            continue
        out.rows.append(SplitRow(
            monomial=tuple(mono),
            a_cols=np.empty(0, dtype=np.int32),
            a_vals=np.empty(0, dtype=float),
            rhs=0.0,
            class_pairs={
                key: (np.asarray(pk, dtype=np.int32),
                      np.asarray(pl, dtype=np.int32))
                for key, (pk, pl) in per_class.items()},
        ))
    return out


def reassemble_dense_gram(split: SplitBlock,
                          sub_grams: Dict[Monomial, np.ndarray]) -> np.ndarray:
    """Exact embedding: block-diagonal dense Gram from per-class sub-Grams.

    Placement is by the DENSE basis indices of each class, so the result is a
    Gram for the original basis ordering. PSD by construction when every
    sub-Gram is PSD (permutation-similar to a direct sum).
    """
    d = split.dense_dim
    Q = np.zeros((d, d))
    for key, idxs in split.classes.items():
        sub = np.asarray(sub_grams[key], dtype=float)
        if sub.shape != (len(idxs), len(idxs)):
            raise ValueError(f"sub-Gram shape {sub.shape} != class size "
                             f"{len(idxs)} for {key}")
        for a, da in enumerate(idxs):
            for b, db in enumerate(idxs):
                Q[da, db] = sub[a, b]
    return Q


def reassemble_from_order(split: SplitBlock,
                          sub_grams: Sequence[np.ndarray]) -> np.ndarray:
    """``reassemble_dense_gram`` for sub-Grams given in ``class_order``."""
    order = split.class_order
    if len(sub_grams) != len(order):
        raise ValueError(f"{len(sub_grams)} sub-Grams for {len(order)} classes")
    return reassemble_dense_gram(split, dict(zip(order, sub_grams)))


def row_value_dense(Q: np.ndarray,
                    pairs: List[Tuple[int, int]]) -> float:
    """<A, Q> for a dense row's (deduplicated) lower-triangle pair list."""
    dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
    val = 0.0
    for k, l in dedup:
        val += Q[k, l] if k == l else 2.0 * Q[k, l]
    return val


def row_value_split(row: SplitRow,
                    sub_grams: Dict[Monomial, np.ndarray]) -> float:
    """<A, Q> computed blockwise from the sub-Grams (must equal dense)."""
    val = 0.0
    for key, (pk, pl) in row.class_pairs.items():
        sub = sub_grams[key]
        for k, l in zip(pk, pl):
            val += sub[k, l] if k == l else 2.0 * sub[k, l]
    return float(val)
