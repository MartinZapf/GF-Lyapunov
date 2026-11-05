"""C¹ (restriction R2) basis tests.

These are MOSEK-free and fast. They lock two invariants:

1. The canonical per-term predicate (``term_is_c1`` / ``classify_term_regularity``
   in ``algebra.enumerate``) is the SINGLE SOURCE OF TRUTH: the verifier's
   ``classify_smoothness`` must agree with it on every term, so the discovery basis
   cannot drift from the certification gate.
2. ``enumerate_*`` with ``c1_only=True`` only ever yields continuously-
   differentiable terms (the class the sound Lemma-5 gate can certify).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.algebra.enumerate import (
    enumerate_all_terms,
    enumerate_diagonal_terms,
    enumerate_cross_terms,
    term_is_c1,
    classify_term_regularity,
)
from gf_framework.verification.verifier import classify_smoothness

# A spread of homogeneity settings covering 1D/2D/3D and m≤maxr / m>maxr.
CASES = [
    (2, (1,)), (3, (1,)),
    (2, (2, 1)), (3, (2, 1)), (4, (2, 1)), (5, (2, 1)),
    (5, (3, 2)), (6, (3, 2)),
    (5, (3, 2, 1)), (6, (3, 2, 1)), (7, (3, 2, 1)),
]


def _as_gf(spec, m, r):
    return GeneralizedForm(
        degree_m=m, weights_r=r,
        terms=[GFTerm(coefficient=1.0, exponents=spec.exponents, term_types=spec.term_types)],
    )


def test_per_factor_rule_matches_restriction_R2():
    # signed ⌈x⌋^ρ: C¹ iff ρ≥1; abs |x|^ρ: C¹ iff ρ=0 or ρ>1.
    assert classify_term_regularity(0, "abs") == "C1"           # |x|^0 = 1
    assert classify_term_regularity(0, "signed") == "discontinuous"  # sign(x)
    assert classify_term_regularity(0.5, "abs") == "sub_lipschitz"
    assert classify_term_regularity(0.5, "signed") == "sub_lipschitz"
    assert classify_term_regularity(1, "abs") == "lipschitz_kink"     # |x| kink
    assert classify_term_regularity(1, "signed") == "C1"             # x
    assert classify_term_regularity(2, "abs") == "C1"
    assert classify_term_regularity(1.5, "signed") == "C1"


@pytest.mark.parametrize("m,r", CASES)
def test_c1_only_enumeration_is_actually_c1(m, r):
    diag, cross = enumerate_all_terms(m, r, c1_only=True)
    for spec in diag + cross:
        assert term_is_c1(spec)
        # The verifier's classifier must independently agree (single source of truth).
        assert classify_smoothness(_as_gf(spec, m, r))["regularity"] == "C1"


@pytest.mark.parametrize("m,r", CASES)
def test_c1_filter_is_a_strict_subset(m, r):
    # Compare the UNCAPPED enumerators (enumerate_all_terms applies a max_cross_terms
    # cap after filtering, which would confound a raw subset comparison).
    d0 = enumerate_diagonal_terms(m, r, c1_only=False)
    c0 = enumerate_cross_terms(m, r, max_denominator=4, minimal=False,
                               include_abs_variants=True, c1_only=False)
    d1 = enumerate_diagonal_terms(m, r, c1_only=True)
    c1 = enumerate_cross_terms(m, r, max_denominator=4, minimal=False,
                               include_abs_variants=True, c1_only=True)
    full = {(t.exponents, t.term_types) for t in d0 + c0}
    restricted = {(t.exponents, t.term_types) for t in d1 + c1}
    assert restricted.issubset(full)
    # Every dropped term must be genuinely non-C¹.
    for t in d0 + c0:
        if (t.exponents, t.term_types) not in restricted:
            assert not term_is_c1(t)


def test_sta_m2_diagonal_dropped_forces_escalation():
    # STA r=(2,1): at m=2 the max-weight diagonal |x1|^1 is a (non-C¹) kink, so the
    # C¹ library cannot span x1 — discovery must escalate the degree.
    diag = enumerate_diagonal_terms(2, (2, 1), c1_only=True)
    spanned = {i for t in diag for i, e in enumerate(t.exponents) if float(e) != 0}
    assert spanned == {1}  # only x2 has a C¹ diagonal at m=2
    diag3 = enumerate_diagonal_terms(3, (2, 1), c1_only=True)
    spanned3 = {i for t in diag3 for i, e in enumerate(t.exponents) if float(e) != 0}
    assert spanned3 == {0, 1}  # m=3 > maxr=2: both variables spanned


def test_enumerated_library_is_c1_only():
    # The discovery library (diagonals + cross terms with c1_only=True) must be
    # entirely C¹ — the class the Lemma-5 gate can certify.
    diag = enumerate_diagonal_terms(5, (3, 2), c1_only=True)
    cross = enumerate_cross_terms(5, (3, 2), max_denominator=4, include_signed=True,
                                  minimal=False, include_abs_variants=True, c1_only=True)
    assert diag and cross
    for spec in list(diag) + list(cross):
        assert term_is_c1(spec), f"enumerate produced a non-C¹ term: {spec}"
