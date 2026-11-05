#!/usr/bin/env python3
"""MOSEK-free tests for the exact symbolic even-support guard (repair R2).

Covers:
  G1  levant4d (real spec + real library) passes; report counts sane.
  G2  the REVIEW COUNTEREXAMPLE refuses: r=(1), mu=(2), f containing
      |x|^{1/2} gives lift 2*(1/2)=1, odd (finding 5).
  G3  NUMERIC CANCELLATION does not fool the symbolic guard: two dynamics
      monomials with the same odd-lift exponent and opposite coefficients
      cancel in the float assembly (the float M_W row is zero) but the
      symbolic guard still refuses. This is the exact failure mode of the
      float-side runtime guard named in review finding 6.
  G4  odd mu is refused (the Lie shift -e_ch breaks parity when mu_ch odd).
  G5  clearing factor with odd lift is refused; even clearing passes.
  G6  multiplier support checked; float exponents refused unless integral.
  G7  epsilon-row degree check fires on an odd V lift degree.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from gf_framework.algebra.support_guard import (
    GuardReport,
    OddLiftRefusal,
    check_even_support,
    guard_from_spec,
)

F = Fraction


# ---------------------------------------------------------------------------
# G1: real system passes
# ---------------------------------------------------------------------------

def test_levant4d_real_spec_passes():
    import gf_framework.discovery.universal_discovery as ud
    from gf_framework.algebra.enumerate import enumerate_diagonal_terms
    from gf_framework.discovery.gf_discovery import mu_filtered_cross

    spec = ud.create_levant4d_spec(delta0=1.0)
    weights = tuple(int(r) for r in spec.weights_r)
    diag = enumerate_diagonal_terms(7, weights, c1_only=True)
    mu, _, cross = mu_filtered_cross(7, weights, diag, spec.dynamics_gf)
    rep = guard_from_spec(spec, list(diag) + list(cross), mu)
    assert rep.ok
    assert rep.n_endpoints == 2
    assert rep.n_v_terms == len(diag) + len(cross)
    assert rep.n_w_products > 0


# ---------------------------------------------------------------------------
# G2: the review counterexample refuses
# ---------------------------------------------------------------------------

def test_review_counterexample_refuses():
    # r=(1), mu=(2), V=|x|^2 (library exponent 2), f=|x|^{1/2}:
    # lift(f) = 2 * 1/2 = 1, odd -> refuse.
    with pytest.raises(OddLiftRefusal) as ei:
        check_even_support(
            mu=(2,), weights_r=(1,),
            library_exponents=[(F(2),)],
            dynamics_vertices=[[[((F(1, 2),), "sqrt-term")]]])
    v = ei.value.violations[0]
    assert v.stage == "W" and v.lift == F(1)


# ---------------------------------------------------------------------------
# G3: numeric cancellation cannot hide an odd symbolic support
# ---------------------------------------------------------------------------

def test_cancelling_odd_pair_still_refuses():
    # Two channel monomials with the SAME odd-lift exponent and coefficients
    # +c and -c: the float W assembly for their shared product monomial sums
    # to zero (row looks vacuous numerically), but symbolically each is
    # present. The guard must refuse on presence, not on the sum.
    with pytest.raises(OddLiftRefusal) as ei:
        check_even_support(
            mu=(2, 2), weights_r=(2, 1),
            library_exponents=[(F(1), F(0)), (F(0), F(2))],
            dynamics_vertices=[[
                [((F(1, 2), F(0)), "plus-c"), ((F(1, 2), F(0)), "minus-c")],
                [((F(0), F(1)), "linear")],
            ]])
    stages = {v.stage for v in ei.value.violations}
    assert stages == {"W"}
    # BOTH cancelling partners are reported (presence-based, sum-blind)
    provs = {v.obj for v in ei.value.violations}
    assert any("plus-c" in p for p in provs)
    assert any("minus-c" in p for p in provs)


def test_float_assembly_would_pass_where_symbolic_refuses():
    # Demonstrate the float-side blind spot the symbolic guard closes:
    # the numeric row for the cancelled odd monomial is exactly zero.
    import numpy as np
    coeffs = np.array([+3.7, -3.7])
    row_value = float(coeffs.sum())  # the float assembly's collected entry
    assert row_value == 0.0  # float guard sees nothing here...
    # ...but the symbolic guard refuses (same support as above):
    with pytest.raises(OddLiftRefusal):
        check_even_support(
            mu=(2, 2), weights_r=(2, 1),
            library_exponents=[(F(1), F(0))],
            dynamics_vertices=[[
                [((F(1, 2), F(0)), "p"), ((F(1, 2), F(0)), "m")],
                [],
            ]])


# ---------------------------------------------------------------------------
# G4: odd mu refused
# ---------------------------------------------------------------------------

def test_odd_mu_refused():
    with pytest.raises(OddLiftRefusal) as ei:
        check_even_support(
            mu=(3, 2), weights_r=(3, 2),
            library_exponents=[(F(2), F(0))],
            dynamics_vertices=[[[((F(2), F(0)), "even-term")], []]])
    assert any("mu_ch odd" in v.obj or v.stage == "V"
               for v in ei.value.violations)


# ---------------------------------------------------------------------------
# G5: clearing factor
# ---------------------------------------------------------------------------

def test_clearing_factor_checked():
    # mu=(2,2), r=(3,1): clearing exponent on var 0 is k*3/3=k, on var 1 is
    # k*3/1=3k; lifts 2k and 6k, always even -> passes for k=1.
    rep = check_even_support(
        mu=(2, 2), weights_r=(3, 1),
        library_exponents=[(F(1), F(0))],
        dynamics_vertices=[[[((F(1), F(0)), "t")], []]],
        clearing_power=1)
    assert rep.ok and rep.clearing_checked
    # mu=(1,2) makes the var-0 clearing lift odd (1*k) -> refuse at k=1.
    with pytest.raises(OddLiftRefusal):
        check_even_support(
            mu=(1, 2), weights_r=(3, 1),
            library_exponents=[(F(2), F(0))],
            dynamics_vertices=[[[], []]],
            clearing_power=1)


# ---------------------------------------------------------------------------
# G6: multipliers + exactness discipline
# ---------------------------------------------------------------------------

def test_multiplier_support_and_float_refusal():
    rep = check_even_support(
        mu=(2,), weights_r=(1,),
        library_exponents=[(F(2),)],
        dynamics_vertices=[[[((F(1),), "lin")]]],
        multiplier_supports=[(F(1),)])  # lift 2, even -> ok
    assert rep.ok
    with pytest.raises(TypeError):
        check_even_support(
            mu=(2,), weights_r=(1,),
            library_exponents=[(0.5,)],  # non-integer float exponent
            dynamics_vertices=[[[]]])


# ---------------------------------------------------------------------------
# G7: epsilon rows
# ---------------------------------------------------------------------------

def test_epsilon_row_degree():
    # single library term with odd total lift degree: V stage itself flags
    # the term; the epsilon check flags the degree too.
    try:
        check_even_support(
            mu=(2,), weights_r=(1,),
            library_exponents=[(F(3, 2),)],  # lift 3, odd
            dynamics_vertices=[[[]]])
        assert False, "expected refusal"
    except OddLiftRefusal as e:
        stages = {v.stage for v in e.violations}
        assert "V" in stages and "epsilon" in stages


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
