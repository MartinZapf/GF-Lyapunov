"""X (exact_gate) states exactly the charge P (verify_certificate) derives.

reports/XVC_RESIDUAL_2026-09-25.md. Before fix (a), X charged the residual
of the parity-PROJECTED Gram and P the residual of the persisted DENSE Gram,
so on every re-solve witness X stated a margin P could not re-derive and the
emitter refused (margin_claim_exceeds_proof; X2 m = 10, X3 delta0 = 269/100,
X5 delta0 = 1 rl1, 2026-09-25). Fix (a) (X charges the dense residual) landed
2026-09-25; the step-1 tests below pin it.

The two step-2 tests pin fix (e), X adopting P's eta ladder rung for rung
(0, 1e-15 ... 1e-9, then the exact Gershgorin bound of the dense Gram).
Adopted by the author on 2026-09-26 (decision D9) and implemented the same
day; until then the two tests were strict xfails. Every stored octant has
eta = 0, the first rung of both ladders, so no stored verdict or margin
changes.
"""
import gzip
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "standalone"))

import verify_certificate as vc  # noqa: E402
from gf_framework.verification.exact_gate import (  # noqa: E402
    deck_group_basis, exact_validate_octant, parity_blocks,
)


def _x(target, d, gram, basis, eps):
    c = exact_validate_octant(target, d, np.asarray(gram, dtype=float), basis, eps)
    return c.ok, (Fraction(c.eps_certified) if c.eps_certified else None), c


def _p(target, d, gram, basis, eps, blocked=True):
    ok, ratio, why, _ = vc.validate_octant(target, d, gram, basis, eps,
                                           blocked=blocked)
    return ok, (eps - ratio * eps if ratio is not None else None), why


def _zqz(Q, basis):
    t = {}
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            m = tuple(a + b for a, b in zip(bi, bj))
            t[m] = t.get(m, Fraction(0)) + Fraction(Q[i][j])
    return {k: v for k, v in t.items() if v != 0}


# --- step 1: the residual (fix (a)) ----------------------------------------

def test_cross_class_entry_is_charged_like_the_portable_verifier():
    """PSD Gram, two parity blocks, one cross-class entry 1/4: the invariant
    target y0^2 + y1^2 has no y0*y1 term, so the dense residual is 1/2."""
    basis = [(1, 0), (0, 1)]
    target = {(2, 0): Fraction(1), (0, 2): Fraction(1)}
    gram = [[1.0, 0.25], [0.25, 1.0]]
    ok_x, m_x, chk = _x(target, 2, gram, basis, Fraction(1))
    assert chk.block_sizes == (1, 1)            # the blocking still runs
    assert Fraction(chk.eta_hat) == 0
    for blocked in (True, False):
        ok_p, m_p, _ = _p(target, 2, gram, basis, Fraction(1), blocked)
        assert (ok_x, m_x) == (ok_p, m_p) == (True, Fraction(1, 2))


def test_cross_class_mass_can_no_longer_pass_x_where_p_fails():
    basis = [(1, 0), (0, 1)]
    target = {(2, 0): Fraction(1), (0, 2): Fraction(1)}
    gram = [[1.0, 0.25], [0.25, 1.0]]
    ok_x, _, chk = _x(target, 2, gram, basis, Fraction(1, 10))
    ok_p, _, why = _p(target, 2, gram, basis, Fraction(1, 10))
    assert not ok_x and not ok_p
    assert chk.reason == why == "budget_exceeded"


def test_block_diagonal_witness_is_unchanged():
    """A design-bundle Gram (y-parity blocks, cross entries exactly 0) gets
    the same charge before and after: the fix touches re-solve witnesses only."""
    basis = [(2, 0), (1, 1), (0, 2)]
    target = {(4, 0): Fraction(1), (2, 2): Fraction(1, 2), (0, 4): Fraction(1)}
    gram = [[1.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 1.0]]
    ok_x, m_x, chk = _x(target, 4, gram, basis, Fraction(1, 100))
    ok_p, m_p, _ = _p(target, 4, gram, basis, Fraction(1, 100))
    assert chk.n_blocks == 2
    assert ok_x and (ok_x, m_x) == (ok_p, m_p)


@pytest.mark.parametrize("seed", range(20))
def test_random_psd_grams_with_cross_mass_agree_exactly(seed):
    """Random PSD Gram on an all-even target: X and P state the same margin."""
    rng = np.random.default_rng(seed)
    basis = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    A = rng.normal(size=(6, 6))
    gram = (A @ A.T + 6 * np.eye(6)).tolist()
    # the part of z^T Q z over pairs with equal exponent parity (an all-even
    # polynomial, so its deck group is all of F2^3 and those parity classes
    # are exactly the gate's blocks), plus a small perturbation of every
    # coefficient; the cross-parity entries of Q stay non-zero
    same = {(i, j) for i, bi in enumerate(basis) for j, bj in enumerate(basis)
            if all((a - b) % 2 == 0 for a, b in zip(bi, bj))}
    inv = {}
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            if (i, j) in same:
                m = tuple(a + b for a, b in zip(bi, bj))
                inv[m] = inv.get(m, Fraction(0)) + Fraction(gram[i][j])
    target = {m: v + Fraction(int(rng.integers(-5, 6)), 10 ** 6)
              for m, v in inv.items()}
    assert all(all(e % 2 == 0 for e in m) for m in target)
    assert len(parity_blocks(basis, deck_group_basis(list(target), 3))) == 4
    eps = Fraction(1000)
    ok_x, m_x, _ = _x(target, 4, gram, basis, eps)
    ok_p, m_p, _ = _p(target, 4, gram, basis, eps)
    assert ok_p
    assert (ok_x, m_x) == (ok_p, m_p)


RESOLVE_RECORD = (ROOT / "artifacts" / "campaign_2026-09" / "TB" / "smoke_sta" / "m6"
                  / "portable" / "sta_tb_clarabel_design_not_D_m6_delta1")


@pytest.mark.skipif(not RESOLVE_RECORD.with_suffix(".record.json.gz").exists(),
                    reason="stored re-solve record not present")
def test_stored_resolve_witness_x_equals_p_on_every_octant():
    """A real re-solve witness (CLARABEL, non-zero cross entries on all 12
    octants): X's per-octant charge must equal P's exactly."""
    rec = json.load(gzip.open(str(RESOLVE_RECORD) + ".record.json.gz"))
    bundle = json.load(gzip.open(str(RESOLVE_RECORD) + ".grams.json.gz"))
    terms = vc.validate_gf_terms(rec["V"]["terms"], len(rec["V"]["weights_r"]),
                                 "V")
    eps = vc.q_exact(rec["epsilon_exact"], "epsilon_exact")
    n_cross = 0
    for cap in bundle["V_grams"]:
        target, d = vc.exact_octant_target(
            terms, tuple(cap["mu"]), tuple(cap["signs"]), eps)
        gram = [[float.fromhex(x) for x in row] for row in cap["gram"]]
        basis = [tuple(b) for b in cap["basis"]]
        blocks = parity_blocks(basis, deck_group_basis(list(target), len(basis[0])))
        cls = {i: k for k, b in enumerate(blocks) for i in b}
        n_cross += any(gram[i][j] != 0.0 for i in range(len(basis))
                       for j in range(len(basis)) if cls[i] != cls[j])
        ok_x, m_x, _ = _x(target, d, gram, basis, eps)
        ok_p, m_p, _ = _p(target, d, gram, basis, eps)
        assert ok_x and (ok_x, m_x) == (ok_p, m_p)
    assert n_cross > 0, "fixture must exercise non-zero cross-class entries"


# --- step 2: the eta ladder (fix (e), author decision D9, 2026-09-26) ------

def _single_block_indefinite(delta):
    basis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    b = 1.0 + delta
    Q = [[1.0, b, b], [b, 1.0, b], [b, b, 1.0]]
    return _zqz(Q, basis), Q, basis


def test_eta_ladder_same_rung_both_pass():
    """lambda_min ~ -3e-13, one parity block, zero residual: before step 2 X
    took eta = |lambda_min| + 1e-15 and P the rung 1e-12, so X stated more."""
    target, Q, basis = _single_block_indefinite(3e-13)
    eps = Fraction(1, 10 ** 6)
    ok_x, m_x, _ = _x(target, 2, Q, basis, eps)
    ok_p, m_p, _ = _p(target, 2, Q, basis, eps)
    assert ok_x and (ok_x, m_x) == (ok_p, m_p)


def test_eta_ladder_x_does_not_pass_what_p_rejects():
    """lambda_min = -2^-20: P's ladder tops out at 1e-9, then Gershgorin ~1,
    so P rejects; X must not report a PASS P cannot reproduce."""
    target, Q, basis = _single_block_indefinite(2.0 ** -20)
    eps = Fraction(1, 1000)
    ok_x, _, _ = _x(target, 2, Q, basis, eps)
    ok_p, _, _ = _p(target, 2, Q, basis, eps)
    assert ok_x == ok_p is False
