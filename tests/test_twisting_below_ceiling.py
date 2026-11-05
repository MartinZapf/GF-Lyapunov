"""Twisting below the ceiling (CANONICAL_STATE.md, "Twisting below the ceiling (2026-09-29)").

``scripts/twisting_below_ceiling.py`` shows with Fractions and Sturm sequences that the
printed twisting function eq. (18) has a nonnegative decrease at delta = +-1/2 on every
closed orthant. These tests pin:

  1. every check of the script passes, and the script exits with code 0;
  2. its four distinct off-axis polynomials equal the list of the ledger and the paper;
  3. the same polynomials, root counts and axis formula, re-derived independently with
     sympy from the closed loop and eq. (18) (no code shared with the script);
  4. the Sturm root counter on polynomials with known roots in [0, oo).
"""
import subprocess
import sys
from fractions import Fraction as Fr
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "twisting_below_ceiling.py"
sys.path.insert(0, str(ROOT / "scripts"))

import twisting_below_ceiling as T  # noqa: E402

# The list of the review plan and the ledger, coefficients from u^0 upwards.
EXPECTED = {
    # (s1, s2, delta) and (-s1, -s2, -delta): 45u^4/32 + 5u^3 - 69u^2/16 - 15u/8 + 183/128
    ((1, 1), Fr(1, 2)): [Fr(183, 128), Fr(-15, 8), Fr(-69, 16), Fr(5), Fr(45, 32)],
    # 15u^4/8 + 15u^3/2 - 87u^2/16 - 15u/8 + 57/32
    ((1, 1), Fr(-1, 2)): [Fr(57, 32), Fr(-15, 8), Fr(-87, 16), Fr(15, 2), Fr(15, 8)],
    # 5u^3/2 - 15u^2/16 + 15u/8 + 3/8
    ((1, -1), Fr(1, 2)): [Fr(3, 8), Fr(15, 8), Fr(-15, 16), Fr(5, 2)],
    # 15u^4/32 - 33u^2/16 + 15u/8 + 3/128
    ((1, -1), Fr(-1, 2)): [Fr(3, 128), Fr(15, 8), Fr(-33, 16), Fr(0), Fr(15, 32)],
}


@pytest.fixture(scope="module")
def res():
    return T.run_checks()


def _expected_for(s, d):
    if (s, d) in EXPECTED:
        return EXPECTED[(s, d)]
    return EXPECTED[((-s[0], -s[1]), -d)]


def test_all_checks_pass(res):
    failed = [name for name, ok in res["checks"] if not ok]
    assert not failed, failed
    assert res["ok"]
    assert len(res["checks"]) >= 40


def test_script_exit_code_and_summary():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                         cwd=str(ROOT))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ALL CHECKS PASS" in out.stdout
    assert "[FAIL]" not in out.stdout


def test_four_polynomials_match_the_list(res):
    assert len(res["distinct"]) == 4
    assert sorted(map(tuple, res["distinct"])) == sorted(map(tuple, EXPECTED.values()))
    assert len(res["cases"]) == 8
    for (s, d), p in res["cases"].items():
        assert p == _expected_for(s, d), (s, d)


def test_margin_threshold_and_axis_value(res):
    assert res["margin_factor"] == 4096
    assert res["threshold"] == Fr(9371, 18750)
    assert res["axis_at_delta1"] == Fr(15, 131072)
    for (s1, s2), (const, lin) in res["axis"].items():
        assert const == Fr(15, 32) * (2 + Fr(3, 2) * s1 * s2)
        assert lin == -Fr(15, 32) * s1


# ---------------------------------------------------------------------------
# Independent derivation with sympy
# ---------------------------------------------------------------------------

def _sympy_W(s1, s2, delta, a1, a2):
    """-grad V . (f_s + delta e2) on the closed orthant s at |x1| = a1, |x2| = a2 (sympy)."""
    sp = pytest.importorskip("sympy")
    R = sp.Rational
    x1, x2 = sp.symbols("x1 x2", positive=True)   # absolute values; signs applied below
    X1, X2 = s1 * x1, s2 * x2
    # On the open orthant s: |x_i| = x_i, [x_i]^a = s_i x_i^a; V is smooth there.
    V = (x1 ** R(5, 2) + R(9, 128) * x2 ** 5 + R(15, 32) * s1 * x1 ** 2 * X2
         + R(5, 4) * x1 ** R(3, 2) * x2 ** 2 - R(3, 8) * X1 * s2 * x2 ** 3)
    # d/dX_i = s_i d/dx_i on the open orthant
    dV1 = s1 * sp.diff(V, x1)
    dV2 = s2 * sp.diff(V, x2)
    f1 = X2
    f2 = -2 * s1 - R(3, 2) * s2 + delta
    W = sp.expand(-(dV1 * f1 + dV2 * f2))
    # the one-sided extension is the continuous extension of this expression
    return sp.expand(W.subs({x1: a1, x2: a2}))


@pytest.mark.parametrize("s", T.ORTHANTS)
@pytest.mark.parametrize("d", [Fr(1, 2), Fr(-1, 2)])
def test_polynomials_against_sympy(res, s, d):
    sp = pytest.importorskip("sympy")
    u = sp.symbols("u", nonnegative=True)
    W = _sympy_W(s[0], s[1], sp.Rational(d.numerator, d.denominator), u ** 2, 1)
    P = sp.Poly(W, u)
    coeffs = [Fr(int(c.p), int(c.q)) for c in reversed(P.all_coeffs())]
    assert coeffs == res["cases"][(s, d)]
    assert P.count_roots(0, sp.oo) == 0
    assert P.LC() > 0 and W.subs(u, 0) > 0
    # the script's Sturm count agrees with sympy's real-root isolation
    assert T.roots_in_nonneg_halfline(coeffs) == P.count_roots(0, sp.oo)


@pytest.mark.parametrize("s", T.ORTHANTS)
def test_axis_formula_against_sympy(s):
    sp = pytest.importorskip("sympy")
    delta, t = sp.symbols("delta t", positive=True)
    W = _sympy_W(s[0], s[1], delta, t, 0)
    s1, s2 = s
    want = sp.Rational(15, 32) * t ** 2 * (2 + sp.Rational(3, 2) * s1 * s2 - s1 * delta)
    assert sp.simplify(W - want) == 0


def test_sturm_counter_on_known_polynomials():
    # (u - 1)(u - 2)(u + 3): two roots in [0, oo)
    p = [Fr(6), Fr(-7), Fr(0), Fr(1)]
    assert T.roots_in_nonneg_halfline(p) == 2
    # (u - 1)^2 (u^2 + 1): one distinct root
    p = [Fr(1), Fr(-2), Fr(2), Fr(-2), Fr(1)]
    assert T.roots_in_nonneg_halfline(p) == 1
    # u (u + 1): the root at 0 is counted
    assert T.roots_in_nonneg_halfline([Fr(0), Fr(1), Fr(1)]) == 1
    # u (u - 1): a root at 0 and a positive root
    assert T.roots_in_nonneg_halfline([Fr(0), Fr(-1), Fr(1)]) == 2
    # u^2 + u + 1: none
    assert T.roots_in_nonneg_halfline([Fr(1), Fr(1), Fr(1)]) == 0
    # near a double root at u = 5: (u - 5)^2 + 1/1000 has none, (u - 5)^2 - 1/1000 has two
    assert T.roots_in_nonneg_halfline([Fr(25) + Fr(1, 1000), Fr(-10), Fr(1)]) == 0
    assert T.roots_in_nonneg_halfline([Fr(25) - Fr(1, 1000), Fr(-10), Fr(1)]) == 2


def test_a_perturbed_coefficient_is_detected(monkeypatch):
    """Mutation: raising the relay gain c2 to the ceiling-breaking 2 must fail a check."""
    monkeypatch.setattr(T, "F_TERMS", (T.F_TERMS[0],
                                       [(-T.C1, ((Fr(0), True), T.ONE)),
                                        (-Fr(2), (T.ONE, (Fr(0), True)))]))
    r = T.run_checks()
    assert not r["ok"]


def test_off_axis_check_can_fail(monkeypatch):
    """Mutation that leaves the axis formula unchanged (|x1|^(3/2)|x2|^2 has zero x2-derivative
    on x2 = 0): the off-axis Sturm check, not the axis check, must reject it."""
    terms = list(T.V_TERMS)
    coef, factors = terms[3]
    assert factors == ((Fr(3, 2), False), (Fr(2), False))
    terms[3] = (Fr(-2), factors)
    monkeypatch.setattr(T, "V_TERMS", terms)
    r = T.run_checks()
    failed = [name for name, ok in r["checks"] if not ok]
    assert any("no root in [0, oo)" in name for name in failed), failed
    assert not any("axis" in name for name in failed), failed
