#!/usr/bin/env python3
"""Twisting below the ceiling: the printed function eq. (18) is a strict Lyapunov function
for every delta0 < 1/2 at the gains (c1, c2) = (2, 3/2).

Closed loop (paper eq. (17)): x1' = x2, x2' = -c1 [x1]^0 - c2 [x2]^0 + delta, |delta| <= delta0,
weights r = (2, 1), kappa = -1, ceiling c1 - c2 = 1/2 (paper Remark rem:ceilings). The printed
function (paper eq. (18), `eq:twisting-V`) is

    V = |x1|^(5/2) + 9/128 |x2|^5 + 15/32 [x1]^2 x2 + 5/4 |x1|^(3/2) |x2|^2 - 3/8 x1 [x2]^3,

certified by Theorem 1 at delta0 = delta1 := 2047/4096. On the closed orthant s the one-sided
decrease is W_s(x; delta) = -grad V(x) . (f_s(x) + delta e2), affine in delta. This script
establishes, in exact rational arithmetic (fractions.Fraction, Python standard library only):

  1. axis x2 = 0: W_s(x1, 0; delta) = (15/32) |x1|^2 (2 + (3/2) s1 s2 - s1 delta), derived
     symbolically in delta (coefficients of 1 and of delta), so that
     W_s >= (15/32)(1/2 - |delta|) |x1|^2;
  2. off the axis, normalized by homogeneity to |x2| = 1, x1 = s1 u^2 (u >= 0; u = 0 is the
     one-sided limit at x1 = 0 on orthant s): at delta = +-1/2 the eight cases (s, delta) are
     polynomials in u with exactly four distinct members; each has a positive value at u = 0,
     a positive leading coefficient and no root in [0, oo) (Sturm sequence over Q). Hence
     W_s(.; +-1/2) >= 0 on every closed orthant (> 0 off the axis);
  3. the margin: for delta1 < |delta| <= delta0 < 1/2,
     W_s(x; delta) = theta W_s(x; +-delta1) + (1 - theta) W_s(x; +-1/2) with
     theta = (1/2 - |delta|)/(1/2 - delta1) >= 4096 (1/2 - delta0), since 1/(1/2 - delta1) = 4096;
  4. side fact about eq. (18) only: the axis decrease (15/32)(1/2 - delta0) meets the design
     margin eps = 10^-4 only for delta0 <= 1/2 - 32 eps/15 = 9371/18750; at delta0 = 2047/4096
     the axis value is 15/131072.

V's positivity is checked the same way (redundant with the certificate, cheap).

V is built from its generalized-form terms and differentiated term by term with the signed
and absolute power rules d|x|^a = a [x]^(a-1), d[x]^a = a |x|^(a-1); the four polynomials are
derived, not hard-coded.

Usage (from the repository root):

    python scripts/twisting_below_ceiling.py

Exit code 0 iff every check passes. Ledger: docs/papers/big_draft/CANONICAL_STATE.md,
section "Twisting below the ceiling (2026-09-29)"; log: the internal review
folder of 2026-09-29 (twisting_below_ceiling.log).
"""
from __future__ import annotations

import sys
from fractions import Fraction as Fr
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# The closed loop and the printed function
# ---------------------------------------------------------------------------

C1, C2 = Fr(2), Fr(3, 2)          # gains (c1, c2)
WEIGHTS = (Fr(2), Fr(1))          # r = (r1, r2)
KAPPA = Fr(-1)                    # degree of homogeneity of the field
DEGREE_V = Fr(5)                  # m
DELTA1 = Fr(2047, 4096)           # certified bound of eq. (18) (Theorem 1)
HALF = Fr(1, 2)                   # the ceiling c1 - c2
DESIGN_EPS = Fr(1, 10 ** 4)       # design margin of the twisting record
ORTHANTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# A factor on one variable is (a, signed): |x|^a if not signed, [x]^a = sign(x)|x|^a if
# signed. (0, False) is the constant 1, (0, True) the relay [x]^0.
Factor = Tuple[Fr, bool]
# A generalized-form term: (coefficient, (factor on x1, factor on x2)).
Term = Tuple[Fr, Tuple[Factor, Factor]]

ONE: Factor = (Fr(0), False)

# eq. (18), term by term.
V_TERMS: List[Term] = [
    (Fr(1), ((Fr(5, 2), False), ONE)),                     # |x1|^(5/2)
    (Fr(9, 128), (ONE, (Fr(5), False))),                   # 9/128 |x2|^5
    (Fr(15, 32), ((Fr(2), True), (Fr(1), True))),          # 15/32 [x1]^2 x2
    (Fr(5, 4), ((Fr(3, 2), False), (Fr(2), False))),       # 5/4 |x1|^(3/2) |x2|^2
    (Fr(-3, 8), ((Fr(1), True), (Fr(3), True))),           # -3/8 x1 [x2]^3
]

# Nominal field f = (f1, f2) as generalized forms (the disturbance is added separately).
F_TERMS: Tuple[List[Term], List[Term]] = (
    [(Fr(1), (ONE, (Fr(1), True)))],                       # f1 = x2
    [(-C1, ((Fr(0), True), ONE)),                          # f2 = -c1 [x1]^0
     (-C2, (ONE, (Fr(0), True)))],                         #      -c2 [x2]^0
)

# ---------------------------------------------------------------------------
# Generalized-form algebra (sign-preserving products, term-wise derivatives)
# ---------------------------------------------------------------------------


def mul_factor(p: Factor, q: Factor) -> Factor:
    """|x|^a |x|^b = |x|^(a+b), [x]^a |x|^b = [x]^(a+b), [x]^a [x]^b = |x|^(a+b).

    Exact on every open orthant, and hence for the one-sided extension to its closure.
    """
    return (p[0] + q[0], p[1] != q[1])


def mul_terms(t: Term, u: Term) -> Term:
    return (t[0] * u[0], tuple(mul_factor(p, q) for p, q in zip(t[1], u[1])))


def d_term(t: Term, i: int) -> List[Term]:
    """Partial derivative of one term in x_i: d|x|^a = a [x]^(a-1), d[x]^a = a |x|^(a-1)."""
    coef, facs = t
    a, signed = facs[i]
    if a == 0:
        if signed:
            raise ValueError("a relay factor [x]^0 is not differentiable")
        return []                                  # constant in x_i
    if a < 1:
        raise ValueError("exponent %s < 1: the term is not C^1" % a)
    new = list(facs)
    new[i] = (a - 1, not signed)
    return [(coef * a, tuple(new))]


def gradient(terms: Sequence[Term]) -> Tuple[List[Term], List[Term]]:
    return tuple([d for t in terms for d in d_term(t, i)] for i in range(2))


def weighted_degree(t: Term) -> Fr:
    return sum((r * f[0] for r, f in zip(WEIGHTS, t[1])), Fr(0))


# Evaluation on the closed orthant s: a factor (a, signed) equals s_i^signed |x_i|^a there.
# A GF on orthant s thus becomes a map (delta power, a1, a2) -> rational coefficient.
Orth = Dict[Tuple[int, Fr, Fr], Fr]


def on_orthant(terms: Sequence[Term], s: Tuple[int, int], dpow: int = 0,
               acc: Orth | None = None, scale: Fr = Fr(1)) -> Orth:
    acc = {} if acc is None else acc
    for coef, facs in terms:
        c = coef * scale
        for si, (_, signed) in zip(s, facs):
            if signed:
                c *= si
        key = (dpow, facs[0][0], facs[1][0])
        acc[key] = acc.get(key, Fr(0)) + c
    return {k: v for k, v in acc.items() if v != 0}


def decrease_on_orthant(s: Tuple[int, int]) -> Orth:
    """W_s(x; delta) = -grad V . (f_s + delta e2) on the closed orthant s, affine in delta."""
    g = gradient(V_TERMS)
    nominal = [mul_terms(dv, ft) for i in range(2) for dv in g[i] for ft in F_TERMS[i]]
    w = on_orthant(nominal, s, dpow=0, scale=Fr(-1))
    return on_orthant(g[1], s, dpow=1, acc=w, scale=Fr(-1))    # -dV/dx2 * delta


# ---------------------------------------------------------------------------
# Univariate polynomials over Q (coefficient lists, index = power) and Sturm sequences
# ---------------------------------------------------------------------------

Poly = List[Fr]


def trim(p: Poly) -> Poly:
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def evaluate(p: Poly, x: Fr) -> Fr:
    v = Fr(0)
    for c in reversed(p):
        v = v * x + c
    return v


def derivative(p: Poly) -> Poly:
    return trim([k * p[k] for k in range(1, len(p))])


def remainder(a: Poly, b: Poly) -> Poly:
    a, b = trim(a), trim(b)
    if not b:
        raise ZeroDivisionError("polynomial division by zero")
    while len(a) >= len(b) and a:
        q = a[-1] / b[-1]
        shift = len(a) - len(b)
        for k, c in enumerate(b):
            a[k + shift] -= q * c
        a = trim(a)
    return a


def sturm_sequence(p: Poly) -> List[Poly]:
    seq = [trim(p), derivative(p)]
    while seq[-1]:
        seq.append([-c for c in remainder(seq[-2], seq[-1])])
    return [q for q in seq if q]


def sign_changes(values: Sequence[Fr]) -> int:
    signs = [v > 0 for v in values if v != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def roots_in_nonneg_halfline(p: Poly) -> int:
    """Number of distinct real roots of p in [0, oo), by Sturm's theorem.

    For p(0) != 0, the sign changes of the Sturm sequence at 0 minus those at +oo count the
    distinct roots in (0, oo). If p(0) = 0, the root 0 is counted once and the factor u^k is
    divided out first.
    """
    p = trim(p)
    if not p:
        raise ValueError("the zero polynomial has every point as a root")
    at_origin = 0
    while p[0] == 0:                   # divide out u
        p = p[1:]
        at_origin = 1
    seq = sturm_sequence(p)
    at_zero = sign_changes([evaluate(q, Fr(0)) for q in seq])
    at_inf = sign_changes([q[-1] for q in seq])
    return at_origin + at_zero - at_inf


def to_u_poly(orth: Orth, delta: Fr) -> Poly:
    """Substitute |x2| = 1, |x1| = u^2, delta = value: a polynomial in u (integer powers)."""
    out: Dict[int, Fr] = {}
    for (dpow, a1, _a2), c in orth.items():
        e = 2 * a1
        if e.denominator != 1:
            raise ValueError("non-integer power of u: %s" % e)
        out[int(e)] = out.get(int(e), Fr(0)) + c * delta ** dpow
    deg = max(out) if out else -1
    return trim([out.get(k, Fr(0)) for k in range(deg + 1)])


def fmt_poly(p: Poly, var: str = "u") -> str:
    """Format as 45u^4/32 + 5u^3 - 69u^2/16 - 15u/8 + 183/128 (highest power first)."""
    parts = []
    for k in range(len(p) - 1, -1, -1):
        c = p[k]
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        c = abs(c)
        mono = "" if k == 0 else (var if k == 1 else "%s^%d" % (var, k))
        if k == 0:
            body = str(c)
        else:
            num = "" if c.numerator == 1 else str(c.numerator)
            body = num + mono + ("" if c.denominator == 1 else "/%d" % c.denominator)
        parts.append((sign, body))
    if not parts:
        return "0"
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sign, body in parts[1:]:
        s += " %s %s" % (sign, body)
    return s


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def run_checks() -> dict:
    """Run every check; return the results (used by tests/test_twisting_below_ceiling.py)."""
    checks: List[Tuple[str, bool]] = []
    lines: List[str] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        lines.append("  [%s] %s" % ("PASS" if ok else "FAIL", name))

    def say(text: str = "") -> None:
        lines.append(text)

    say("== Setup ==")
    say("closed loop: x1' = x2, x2' = -%s [x1]^0 - %s [x2]^0 + delta; r = (%s, %s), kappa = %s"
        % (C1, C2, WEIGHTS[0], WEIGHTS[1], KAPPA))
    say("ceiling c1 - c2 = %s; certified bound delta1 = %s" % (C1 - C2, DELTA1))
    check("ceiling c1 - c2 = 1/2", C1 - C2 == HALF)
    check("c2 = 3/2 > 1/2: the classical condition c2 > delta0 holds below the ceiling",
          C2 > HALF)
    check("every term of V has weighted degree m = 5",
          all(weighted_degree(t) == DEGREE_V for t in V_TERMS))
    g = gradient(V_TERMS)           # raises if a term is not C^1
    c1_ok = all(not (f[0] == 0 and f[1]) for comp in g for t in comp for f in t[1])
    check("grad V has no relay factor (V is C^1 term by term)", c1_ok)
    w_terms = [mul_terms(dv, ft) for i in range(2) for dv in g[i] for ft in F_TERMS[i]]
    check("every term of grad V . f has weighted degree m + kappa = 4",
          all(weighted_degree(t) == DEGREE_V + KAPPA for t in w_terms))
    check("matched disturbance: r2 = -kappa, and every term of dV/dx2 has degree m + kappa",
          WEIGHTS[1] == -KAPPA
          and all(weighted_degree(t) == DEGREE_V + KAPPA for t in g[1]))

    W = {s: decrease_on_orthant(s) for s in ORTHANTS}
    check("W_s is affine in delta on every orthant",
          all(k[0] in (0, 1) for s in ORTHANTS for k in W[s]))

    # --- Fact 1: axis x2 = 0, symbolic in delta ------------------------------------------
    say()
    say("== Fact 1: axis x2 = 0 (x1 != 0), symbolic in delta ==")
    axis = {}
    for s in ORTHANTS:
        on_axis = {k: c for k, c in W[s].items() if k[2] == 0}
        a1s = {k[1] for k in on_axis}
        const = sum((c for k, c in on_axis.items() if k[0] == 0), Fr(0))
        lin = sum((c for k, c in on_axis.items() if k[0] == 1), Fr(0))
        axis[s] = (const, lin)
        s1, s2 = s
        want_const = Fr(15, 32) * (2 + Fr(3, 2) * s1 * s2)
        want_lin = -Fr(15, 32) * s1
        say("s=(%+d,%+d): W_s(x1, 0; delta) = (%s + (%s) delta) |x1|^2" % (s1, s2, const, lin))
        check("s=(%+d,%+d): only |x1|^2 survives on the axis" % s, a1s == {Fr(2)})
        check("s=(%+d,%+d): equals (15/32)|x1|^2 (2 + (3/2) s1 s2 - s1 delta)" % s,
              const == want_const and lin == want_lin)
    # min over s and over the sign of delta, as a function of |delta|: const - |lin| |delta|
    worst_const = min(c for c, _ in axis.values())
    worst_lin = max(abs(l) for _, l in axis.values())
    say("minimum over s and sign(delta): (%s - %s |delta|) |x1|^2 = (15/32)(1/2 - |delta|) |x1|^2"
        % (worst_const, worst_lin))
    check("axis minimum is (15/32)(1/2 - |delta|)|x1|^2",
          worst_const == Fr(15, 64) and worst_lin == Fr(15, 32))
    at_eq = axis[(1, -1)][0] + axis[(1, -1)][1] * HALF
    say("s=(+1,-1), delta = 1/2: axis value %s (the equilibrium of Remark ceilings)" % at_eq)
    check("axis value vanishes at s=(+1,-1), delta=1/2 (delta0 = 1/2 excluded)", at_eq == 0)

    # --- Fact 2: off the axis, delta = +-1/2 ----------------------------------------------
    say()
    say("== Fact 2: off the axis, |x2| = 1, x1 = s1 u^2, u >= 0, delta = +-1/2 ==")
    cases = {}
    for s in ORTHANTS:
        for d in (HALF, -HALF):
            p = to_u_poly(W[s], d)
            cases[(s, d)] = p
            n_roots = roots_in_nonneg_halfline(p)
            p0, lc = evaluate(p, Fr(0)), p[-1]
            say("s=(%+d,%+d) delta=%5s: W = %s" % (s[0], s[1], d, fmt_poly(p)))
            say("      value at u=0: %s, leading coeff: %s, roots in [0, oo) (Sturm): %d"
                % (p0, lc, n_roots))
            check("s=(%+d,%+d) delta=%s: p(0) > 0, leading coeff > 0, no root in [0, oo)"
                  % (s[0], s[1], d), p0 > 0 and lc > 0 and n_roots == 0)
    check("symmetry: the case (s, delta) equals the case (-s, -delta)",
          all(cases[(s, d)] == cases[((-s[0], -s[1]), -d)] for (s, d) in cases))
    distinct: List[Poly] = []
    for key in sorted(cases, key=lambda k: (-k[0][0], -k[0][1], -k[1])):
        if cases[key] not in distinct:
            distinct.append(cases[key])
    say("distinct polynomials: %d" % len(distinct))
    for p in distinct:
        members = [k for k in sorted(cases, key=lambda k: (-k[0][0], -k[0][1], -k[1]))
                   if cases[k] == p]
        say("  %s   <- %s" % (fmt_poly(p), ", ".join("(%+d,%+d, %s)" % (k[0][0], k[0][1], k[1])
                                                    for k in members)))
    check("exactly 4 distinct polynomials", len(distinct) == 4)
    say("hence W_s(.; +-1/2) > 0 off the axis and >= 0 on it, on every closed orthant")

    # --- Fact 3: margin ---------------------------------------------------------------------
    say()
    say("== Fact 3: margin below the ceiling ==")
    factor = 1 / (HALF - DELTA1)
    say("1/(1/2 - delta1) = %s" % factor)
    check("margin factor 1/(1/2 - delta1) = 4096", factor == 4096)
    say("theta = (1/2 - |delta|)/(1/2 - delta1) >= %s (1/2 - delta0) for delta1 < |delta| <= delta0"
        % factor)
    say("min{1, 4096 (1/2 - delta0)} = 1 iff delta0 <= %s" % (HALF - 1 / factor))
    check("4096 (1/2 - delta0) >= 1 exactly for delta0 <= delta1", HALF - 1 / factor == DELTA1)
    # The interpolation identity at one interior point, as a sanity check of affinity.
    dm = (DELTA1 + HALF) / 2
    theta = (HALF - dm) / (HALF - DELTA1)
    ok_interp = all(
        to_u_poly(W[s], sg * dm) == trim([theta * a + (1 - theta) * b for a, b in zip(
            _pad(to_u_poly(W[s], sg * DELTA1), 5), _pad(to_u_poly(W[s], sg * HALF), 5))])
        for s in ORTHANTS for sg in (1, -1))
    check("W(delta) = theta W(+-delta1) + (1 - theta) W(+-1/2) at delta = +-%s" % dm, ok_interp)

    # --- Fact 4: design margin on the axis -------------------------------------------------
    say()
    say("== Fact 4: axis decrease of eq. (18) against the design margin eps = 1e-4 ==")
    threshold = HALF - 32 * DESIGN_EPS / 15
    axis_at_d1 = Fr(15, 32) * (HALF - DELTA1)
    say("(15/32)(1/2 - delta0) >= eps  iff  delta0 <= 1/2 - 32 eps/15 = %s (= %.8f)"
        % (threshold, float(threshold)))
    say("axis value at delta0 = %s: (15/32)(1/2 - delta0) = %s (= %.6e)"
        % (DELTA1, axis_at_d1, float(axis_at_d1)))
    check("threshold 1/2 - 32 eps/15 = 9371/18750", threshold == Fr(9371, 18750))
    check("axis value at 2047/4096 is 15/131072 >= eps", axis_at_d1 == Fr(15, 131072)
          and axis_at_d1 >= DESIGN_EPS)
    check("delta1 = 2047/4096 lies below the threshold", DELTA1 <= threshold)

    # --- V positivity -----------------------------------------------------------------------
    say()
    say("== V positivity (redundant with the certificate) ==")
    for s in ORTHANTS:
        v = on_orthant(V_TERMS, s)
        pv = to_u_poly(v, Fr(0))
        n_roots = roots_in_nonneg_halfline(pv)
        say("s=(%+d,%+d): V(s1 u^2, s2) = %s; V(0) = %s, roots in [0, oo): %d"
            % (s[0], s[1], fmt_poly(pv), evaluate(pv, Fr(0)), n_roots))
        check("s=(%+d,%+d): V > 0 on |x2| = 1" % s,
              evaluate(pv, Fr(0)) > 0 and pv[-1] > 0 and n_roots == 0)
        on_axis = {k: c for k, c in v.items() if k[2] == 0}
        check("s=(%+d,%+d): V(x1, 0) = |x1|^(5/2)" % s, on_axis == {(0, Fr(5, 2), Fr(0)): Fr(1)})

    n_fail = sum(1 for _, ok in checks if not ok)
    say()
    say("== Summary ==")
    say("checks: %d, passed: %d, failed: %d" % (len(checks), len(checks) - n_fail, n_fail))
    say("ALL CHECKS PASS" if n_fail == 0 else "SOME CHECKS FAILED")
    return {
        "checks": checks,
        "lines": lines,
        "cases": cases,
        "distinct": distinct,
        "axis": axis,
        "margin_factor": factor,
        "threshold": threshold,
        "axis_at_delta1": axis_at_d1,
        "ok": n_fail == 0,
    }


def _pad(p: Poly, n: int) -> Poly:
    return list(p) + [Fr(0)] * (n + 1 - len(p))


def main() -> int:
    res = run_checks()
    print("scripts/twisting_below_ceiling.py: eq. (18) at (c1, c2) = (2, 3/2), exact rational "
          "arithmetic (fractions.Fraction), Python %s" % sys.version.split()[0])
    print()
    for line in res["lines"]:
        print(line)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
