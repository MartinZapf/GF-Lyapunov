"""Exact-rational a-posteriori verification of SOS Gram certificates.

Implements Proposition 7 of ``docs/papers/theory_notes.md``: re-prove the
margin-budget inequality of the hardened gate (Proposition 6,
``_validate_gram_certificate``) entirely in exact rational arithmetic, so that
no floating-point operation is *trusted* anywhere in the chain — floats only
*propose* the Gram and the PSD shift.

The verified object is the **rational certificate**: the published V with its
stored float64 coefficients read as exact dyadic rationals (``Fraction(float)``
is exact), exact ``Fraction`` exponents (already exact in the artifacts),
exact-dyadic dynamics gains, and the exact-dyadic margin ε that the float
pipeline actually subtracted. The exact octant targets are rebuilt from
scratch here — exact Lie derivative, exact association at the captured μ (a bad μ can only
make the target checks fail, never accept) — mirroring the float pipeline
term-for-term, so that the float pipeline's assembly is never trusted either;
any float-vs-exact assembly discrepancy lands in the exactly-computed
residual E and is charged against the margin.

Chain per octant (Proposition 7):
    1. exact target  P = Φ_s − ε·Σ_i y_i^d  (homogeneous, even d — checked),
    2. Q̄ = sym(Gram) as exact rationals; E = P − zᵀQ̄z exact; ‖E‖₁ exact,
    3. find rational η̂ ≥ 0 with Q̃ + η̂I ⪰ 0, where Q̃ is the block-diagonal
       (parity-class) projection of Q̄ (Q̃ = Q̄ when there is one block),
       verified block by block by exact rational LDLᵀ (largest-diagonal
       pivoting, no square roots); E stays the dense residual of Q̄, which
       bounds the residual of Q̃ (see "Residual accounting, exactly" below),
    4. accept iff charge = η̂·C₁ + ‖E‖₁ < ε in exact arithmetic (STRICT),
       with C₁ = maxᵢ Σ_β 2βᵢ/d over the Gram basis.

What step 4 buys is ``F₀ ≥ (ε − charge) Σᵢ yᵢᵈ``, so any charge strictly under
ε proves a strictly positive margin. ``margin_policy="achieved"`` is the
DEFAULT since WP P2-14 (owner decision D-l, 2026-09-04): accept at
``charge < ε`` and record the margin actually proved. ``margin_policy="half"``
is the historical convention — accept only at ``charge ≤ ε/2``, which makes the
proved margin at least ε/2 — and it is the inequality the frozen Automatica
note prints (``docs/papers/automatica_note/main.tex``, Prop. "exact-rational
gate"). ``half`` ⊂ ``achieved``: the flip proves a smaller number by the same
exact argument, never the same number less rigorously. Every octant records
``eps_certified = ε − charge`` under both policies: that, not the requested ε,
is the margin a certificate claim may quote.

Deck-group (sign-symmetry) blocking — campaign item R1
------------------------------------------------------
Step 3 is 99.9% of the exact phase and costs O(N³) rational eliminations with
bit growth linear in N (measured t ~ N^4.3; N = 120 took ~476 s per octant).
It is almost always avoidable, exactly.

Let S = supp(P) and let the DECK GROUP of the target be

    H = { g ∈ F₂ⁿ : ⟨g, α⟩ ≡ 0 (mod 2)  for every α ∈ S } ,

i.e. the sign flips y_i ↦ −y_i (for i with g_i = 1) that leave P invariant.
Averaging the captured Gram over the induced group action (Reynolds
projection, ρ(g)_ββ = (−1)^⟨g,β⟩),

    Q̃ = |H|⁻¹ Σ_{g∈H} ρ(g) Q̄ ρ(g)ᵀ ,

zeroes exactly the entries whose row/column classes differ, where the class of
a basis exponent β is the character (⟨g,β⟩ mod 2)_{g∈H}. So Q̃ is Q̄ restricted
to its parity-diagonal BLOCKS, and:

  * ``zᵀQ̃z = R_H(zᵀQ̄z)``, and P is H-invariant, so Ẽ = R_H(E): projection
    deletes non-invariant residual coefficients, ‖Ẽ‖₁ ≤ ‖E‖₁. The gate
    nevertheless CHARGES the dense ‖E‖₁ (every index pair of the captured
    Q̄), the portable verifier's definition (review 2026-09-05 finding 12,
    reports/XVC_RESIDUAL_2026-09-25.md): the persisted proof object is Q̄,
    and a margin stated here must be re-derivable from it by P. By the
    inequality this is conservative, never a relaxation;
  * Q̃ is block diagonal, hence PSD iff every block is PSD — so the exact LDLᵀ
    runs per block, at Σ_b N_b³ instead of N³ (measured 1/48 for levant4d V);
  * Q̄ itself need NOT be PSD: the PSD witness is Q̃ + η̂I. Q̃ is a
    deterministic function of the persisted Q̄ and the exact target (the
    target fixes H), so the proof object a reader re-checks is Q̄ itself: P
    rebuilds Q̃ from it and charges the same dense ‖E‖₁. The η̂ ladder is
    P's (0, 10⁻¹⁵ … 10⁻⁹, then the exact Gershgorin bound of the DENSE Q̄);
    pinching gives λ_min(Q̃) ≥ λ_min(Q̄), so the Gershgorin rung, sufficient
    for Q̄, is sufficient for Q̃ too (at worst conservative);
  * C₁ is unchanged (same basis, same identity shift), so the acceptance
    inequality of step 4 is evaluated identically.

Residual accounting, exactly: write C = zᵀ(Q̄ − Q̃)z, the cross-class part.
Every monomial of C is β_i + β_j with i, j in DIFFERENT classes, hence
non-invariant, while every monomial of P − zᵀQ̃z is H-invariant. The supports
are disjoint, so ‖E‖₁ = ‖Ẽ‖₁ + ‖C‖₁. Until 2026-09-25 this gate charged ‖Ẽ‖₁
(sound, since Q̃ + η̂I ⪰ 0 is a certificate on its own), while the portable
verifier charges ‖E‖₁; on a re-solve witness whose cross-class entries carry
solver noise, X then stated a margin larger than P re-derives from the
persisted Gram, by exactly ‖C‖₁, and the record emitter refused it (R-08;
NEGATIVES N25, N29, N31). The gate now charges ‖E‖₁ (fix (a) of
reports/XVC_RESIDUAL_2026-09-25.md). On a design-bundle witness solved with
y-parity blocks the cross-class entries are exactly 0, C = 0, and nothing
changes. Since 2026-09-26 the η̂ proposal is P's fixed ladder as well (fix (e)
of the same report, author decision D9): on identical inputs X and P select
the same rung and state the same charge, one acceptance rule. Before that X
proposed η̂ from the dense float λ_min, and on a Q̄ that is not float-PSD the
two could select different rungs; every stored octant has η̂ = 0, the first
rung of both ladders, so no stored verdict or margin changed.

Hence a blocked octant ACCEPTS whenever the dense one does, with the SAME
charge at the same η̂: the reduction is a pure speedup, never a relaxation.
The implication runs ONE way only (external review 2026-09-05, finding 12):
the blocked PSD test is the weaker one and may accept an octant the dense
path rejects, which is sound rather than a bug — projection cannot lower
λ_min. The residual is the same number in both paths (dense, as above), so
identical verdicts need only the same η̂ rung, which holds in particular
whenever the captured Q̄ is already PSD (both take the base rung).
A verdict difference on an arbitrary indefinite witness is not a regression,
and neither implementation may be "fixed" into an equality.
Measured on the stored levant4d certificate: V octant 476.04 s → 0.45 s
(1059x), W octant 82.24 s → 0.11 s (781x), residual bit-identical.

The guard is the exact support test on the ALREADY-BUILT exact target (never
on a float-collected support), which is what discharges review verdict 6 on
the gate side. When μ is doubled every V target is all-even and H = F₂ⁿ; the
review's verdict 5 correctly notes this is NOT guaranteed for W (the Lie
derivative can lift to odd y-exponents), so H is computed per octant from the
actual support and degrades gracefully: a partial subgroup still blocks, and
H = {0} falls through to the dense path unchanged, byte for byte.

Provenance: PIPELINE.md §3 R1 (binding decision doc), an internal y-parity
review (verdicts 5/6), and one external consult which independently confirmed the
projection/pinching argument and contributed the maximal-subgroup fallback.
Positioning literature: Gatermann & Parrilo, JPAA 192:95-128, 2004; Löfberg,
IEEE TAC 54(5):1007-1011, 2009.

R1 acts only on a GIVEN Gram, so it needs no equivalence theorem. It must not
be confused with the gate-SDP sign-symmetry reduction (STATUS P2 / R8), which
changes the solved problem and carries its own validation decision.

Designed to run OFFLINE on Gram captures from
``verify_gf_sos_full(..., capture_grams=True)`` — see
``review/reverify_exact.py``. This module is numpy-light and torch-free; it
never touches the float design/verify pipeline.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# (coefficient, exponents, term_types) with every numeric entry an exact Fraction.
ExactTerm = Tuple[Fraction, Tuple[Fraction, ...], Tuple[str, ...]]


class ExactGateError(Exception):
    """A certificate object violates a structural precondition of the exact gate."""


# ---------------------------------------------------------------------------
# Margin policy — how much of ε the acceptance threshold demands (WP P2-13)
# ---------------------------------------------------------------------------
#
# Proposition 6/7's proof line is
#
#     F₀ ≥ (ε − η̂·C₁ − ‖Ê‖₁) Σᵢ yᵢᵈ = (ε − charge) Σᵢ yᵢᵈ ,
#
# and the published statement ``F₀ ≥ (ε/2) Σᵢ yᵢᵈ`` is what you get by
# ALSO demanding ``charge ≤ ε/2``. The ε/2 threshold is therefore a
# CONVENTION on how much of the requested margin the certificate must retain,
# not a soundness requirement: every value of ``charge`` strictly below ε
# yields a strictly positive certified margin ``ε_cert = ε − charge`` by the
# same proof, with no step of Proposition 7 changed.
#
#   ``"achieved"``  (DEFAULT since P2-14 / owner decision D-l): accept iff
#                   ``charge < ε`` STRICTLY. The certified margin is
#                   ``ε − charge > 0``, exactly the largest margin this Gram
#                   supports through the Proposition-7 chain. Never weaker
#                   than a proof: it proves a smaller number, it does not
#                   prove it less rigorously.
#   ``"half"``      (historical, and the inequality the frozen Automatica note
#                   prints): accept iff ``charge ≤ ε/2``. The certified margin
#                   is then ``ε − charge ≥ ε/2``. Contained in ``"achieved"``
#                   because ε > 0, so nothing it proved is lost by the flip.
#
# Both policies record ``eps_certified`` (exact, in ℚ), which is the number a
# certificate claim must quote. The REQUESTED ε is an input to the design; the
# certificate's margin is whatever the exact phase proved.
#
# Measured motivation (P5-01 N20, P7-07, and P2-13's 2x2): on CTA m = 5 at
# design ε = 1e-7 the
# design Gram charges 7.36e-08 — over ε/2 = 5e-08, under ε — and on the
# levant4d E8 32-term row at ε = 1e-6 it charges 5.016e-07 against ε/2 =
# 5.00e-07, 0.3 % over. Both are D → X closures thrown away by the ε/2
# convention, not by the arithmetic.
MARGIN_POLICY_HALF: str = "half"
MARGIN_POLICY_ACHIEVED: str = "achieved"
MARGIN_POLICIES: Tuple[str, ...] = (MARGIN_POLICY_HALF, MARGIN_POLICY_ACHIEVED)


def resolve_margin_policy(policy: str) -> str:
    """Validate a margin policy, raising on an unknown one (never silently half)."""
    if policy not in MARGIN_POLICIES:
        raise ExactGateError(
            f"unknown margin_policy {policy!r}; "
            f"expected one of {list(MARGIN_POLICIES)}")
    return policy


def margin_budget(eps: Fraction,
                  margin_policy: str = MARGIN_POLICY_ACHIEVED) -> Fraction:
    """The exact acceptance threshold for ``charge`` under ``margin_policy``."""
    return eps / 2 if resolve_margin_policy(margin_policy) == MARGIN_POLICY_HALF else eps


# ---------------------------------------------------------------------------
# Raw input validation (external scientific review 2026-09-05, findings 2 / 5)
#
# Proposition 7's two bounds — ‖E‖₁ on the residual and η̂·C₁ on the PSD shift
# — are inequalities about POLYNOMIAL monomials: nonnegative integer exponent
# vectors of one common degree. Coercing a capture with ``int(...)`` does not
# establish that; it only makes a violating value look plausible. The review's
# instance: basis {(8, −4), (0, 4)} satisfies ``2·Σβ = d`` and a positive
# definite Gram then "proves" V = −x₁² + 4x₂⁴ ≥ margin, although V(1, 0) = −1
# (the residual −½ y₁¹⁶y₂⁻⁸ is unbounded as y₂ → 0 and no coefficient norm
# bounds it). The same applies to the declared lift μ: ``int(2.9) == 2``
# silently proves a statement about a different substitution.
#
# These helpers therefore run on the RAW capture values, before any coercion,
# and every premise the arithmetic below relies on is established here.
# ---------------------------------------------------------------------------

def require_int(value: Any, what: str) -> int:
    """An exact integer from a RAW capture value, before any coercion.

    Accepts Python/NumPy integers, a float that is exactly integral, and a
    string denoting an exact integer. Refuses booleans, floats with a
    fractional part (2.9 must never become 2), non-finite floats and every
    other type — each of which ``int()`` would have accepted silently.
    """
    if isinstance(value, bool):
        raise ExactGateError(f"{what}: boolean {value!r} where an integer was expected")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise ExactGateError(f"{what}: {value} is not an integer")
        return int(value)
    if isinstance(value, numbers.Real):
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ExactGateError(f"{what}: non-finite {value!r}")
        if not float(v).is_integer():
            raise ExactGateError(f"{what}: {value!r} is not an integer")
        return int(v)
    if isinstance(value, str):
        try:
            q = Fraction(value.strip())
        except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
            raise ExactGateError(f"{what}: {value!r} is not an integer ({exc})")
        if q.denominator != 1:
            raise ExactGateError(f"{what}: {value!r} is not an integer")
        return int(q)
    raise ExactGateError(
        f"{what}: {type(value).__name__} {value!r} where an integer was expected")


def require_int_vector(value: Any, n: int, what: str,
                       minimum: Optional[int] = None) -> Tuple[int, ...]:
    """Exactly ``n`` raw integers, optionally each ``>= minimum``."""
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__len__"):
        raise ExactGateError(
            f"{what}: {type(value).__name__} {value!r} where a vector of {n} "
            f"integers was expected")
    if len(value) != n:
        raise ExactGateError(f"{what}: {len(value)} entries, need {n}")
    out: List[int] = []
    for i, entry in enumerate(value):
        v = require_int(entry, f"{what}[{i}]")
        if minimum is not None and v < minimum:
            raise ExactGateError(f"{what}[{i}] = {v} violates >= {minimum}")
        out.append(v)
    return tuple(out)


def exact_eps(value: Any, what: str = "eps") -> Fraction:
    """The capture's margin as an exact Fraction, from a finite real only.

    ``Fraction(float(...))`` is exact on a float64 and is what the float
    pipeline actually subtracted; a boolean, a NaN/inf or a non-numeric entry
    is a malformed capture and is refused rather than coerced.
    """
    if isinstance(value, bool):
        raise ExactGateError(f"{what}: boolean where a margin was expected")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Real):
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ExactGateError(f"{what}: non-finite {value!r}")
        return Fraction(v)
    raise ExactGateError(
        f"{what}: {type(value).__name__} {value!r} where a margin was expected")


def validate_mu(value: Any, n_vars: int, what: str = "mu") -> Tuple[int, ...]:
    """The declared association lift: ``n_vars`` POSITIVE integers, raw.

    x_i = s_i y_i^{μ_i} is a chart of one octant only for integral μ_i ≥ 1.
    μ_i = 0 collapses a coordinate, a negative one is not a chart, and 2.9 is
    not the substitution the capture declares (review finding 5).
    """
    return require_int_vector(value, n_vars, what, minimum=1)


def validate_signs(value: Any, n_vars: int, what: str = "signs",
                   strict: bool = False) -> Tuple[int, ...]:
    """The octant sign label: ``n_vars`` raw integers, optionally exactly ±1.

    ``strict=False`` inside :func:`exact_verify_captures`, whose coverage rule
    (the sign set must EQUAL {−1,+1}ⁿ) is the stronger statement and owns the
    ``invalid_sign_label`` refusal.
    """
    signs = require_int_vector(value, n_vars, what)
    if strict and any(s not in (-1, 1) for s in signs):
        raise ExactGateError(f"{what}: {signs} is not in {{-1,+1}}^{n_vars}")
    return signs


def validate_basis(basis: Any, d: int, n_vars: int,
                   what: str = "basis") -> List[Tuple[int, ...]]:
    """Gram basis: nonnegative integer exponent vectors of degree d/2.

    The premise of both Proposition-7 bounds (see the section comment above).
    Duplicates are refused as well: a repeated basis vector is a degenerate
    Gram indexing no honest capture produces, and it makes the derived parity
    classes ambiguous. The legacy reason token ``basis_degree_mismatch`` is
    kept in the message so ``failure_typing`` still classifies it.
    """
    if isinstance(basis, (str, bytes, dict)) or not hasattr(basis, "__len__"):
        raise ExactGateError(f"invalid_basis({type(basis).__name__})")
    if len(basis) == 0:
        raise ExactGateError("empty_basis")
    out: List[Tuple[int, ...]] = []
    seen = set()
    for k, beta in enumerate(basis):
        try:
            b = require_int_vector(beta, n_vars, f"{what}[{k}]", minimum=0)
        except ExactGateError as exc:
            raise ExactGateError(f"invalid_basis_exponent({exc})")
        if 2 * sum(b) != d:
            raise ExactGateError(
                f"basis_degree_mismatch({what}[{k}]={b}, need d/2={d}/2)")
        if b in seen:
            raise ExactGateError(f"invalid_basis_duplicate({what}[{k}]={b})")
        seen.add(b)
        out.append(b)
    return out


def validate_exact_target(target: Any, d: int, what: str = "target") -> int:
    """Octant target: nonnegative integer monomials, every one of degree d.

    Returns the number of variables, which then fixes the basis dimension.
    """
    if not isinstance(target, dict) or not target:
        raise ExactGateError(f"invalid_target({what} is not a non-empty dict)")
    if d <= 0 or d % 2 != 0:
        raise ExactGateError(f"invalid_target(degree {d} not positive even)")
    n_vars: Optional[int] = None
    for mono, coeff in target.items():
        if not isinstance(mono, tuple):
            raise ExactGateError(f"invalid_target(monomial {mono!r} is not a tuple)")
        if n_vars is None:
            n_vars = len(mono)
        try:
            mono_i = require_int_vector(mono, n_vars, f"{what} monomial", minimum=0)
        except ExactGateError as exc:
            raise ExactGateError(f"invalid_target({exc})")
        if sum(mono_i) != d:
            raise ExactGateError(
                f"invalid_target(monomial {mono_i} has degree {sum(mono_i)}, need {d})")
        if isinstance(coeff, bool) or not isinstance(coeff, (int, Fraction)):
            raise ExactGateError(
                f"invalid_target(coefficient {coeff!r} of {mono_i} is not exact)")
    return int(n_vars)


# ---------------------------------------------------------------------------
# Exact GF-term algebra (mirror of algebra.lie_derivative, Fraction coefficients)
# ---------------------------------------------------------------------------

def exact_terms_from_gf(gf) -> List[ExactTerm]:
    """Read a GeneralizedForm as exact rational terms.

    A coefficient that is already an exact ``Fraction`` (injected gains, e.g.
    λ₃ = 11/10 or a rational λ₂-bracket endpoint) passes through UNTOUCHED —
    this is what makes exact-rational gain injection sound end-to-end.
    Float coefficients are read as ``Fraction(float)``, the exact value of the
    stored float64 (a dyadic rational) — the published object, bit for bit.
    Exponents are already exact Fractions in the artifacts/GFTerm.
    """
    out: List[ExactTerm] = []
    for t in gf.terms:
        c = t.coefficient
        coeff = c if isinstance(c, Fraction) else Fraction(float(c))
        exps = tuple(e if isinstance(e, Fraction) else Fraction(e) for e in t.exponents)
        out.append((coeff, exps, tuple(t.term_types)))
    return out


def exact_partial(term: ExactTerm, var_idx: int) -> Optional[ExactTerm]:
    """∂/∂xᵢ of a GF term (mirror of ``gf_partial_derivative_term``).

    |x|^ρ → ρ·⌈x⌋^(ρ−1)  (abs → signed);  ⌈x⌋^ρ → ρ·|x|^(ρ−1)  (signed → abs).
    """
    coeff, exps, types = term
    e = exps[var_idx]
    if e == 0:
        return None
    new_exps = list(exps)
    new_exps[var_idx] = e - 1
    new_types = list(types)
    new_types[var_idx] = 'signed' if types[var_idx] == 'abs' else 'abs'
    return (coeff * e, tuple(new_exps), tuple(new_types))


def exact_multiply(t1: ExactTerm, t2: ExactTerm) -> ExactTerm:
    """Product of two GF terms (mirror of ``multiply_gf_terms``).

    Type algebra: abs·abs = abs, signed·signed = abs (sign² = 1), mixed = signed.
    """
    c = t1[0] * t2[0]
    exps = tuple(a + b for a, b in zip(t1[1], t2[1]))
    types = tuple('abs' if a == b else 'signed' for a, b in zip(t1[2], t2[2]))
    return (c, exps, types)


def exact_collect(terms: Sequence[ExactTerm]) -> List[ExactTerm]:
    """Merge terms with identical (exponents, types); drop exact zeros."""
    acc: Dict[Tuple[Tuple[Fraction, ...], Tuple[str, ...]], Fraction] = {}
    for c, e, t in terms:
        key = (e, t)
        acc[key] = acc.get(key, Fraction(0)) + c
    return [(c, e, t) for (e, t), c in acc.items() if c != 0]


def exact_neg_lie_derivative(
    v_terms: Sequence[ExactTerm],
    dynamics_terms: Sequence[Sequence[ExactTerm]],
) -> List[ExactTerm]:
    """W = −dV/dt = −Σᵢ (∂V/∂xᵢ)·fᵢ, exactly (mirror of compute_lie_derivative_gf + negate)."""
    out: List[ExactTerm] = []
    for i, f_i in enumerate(dynamics_terms):
        for vt in v_terms:
            d = exact_partial(vt, i)
            if d is None:
                continue
            for ft in f_i:
                c, e, t = exact_multiply(d, ft)
                out.append((-c, e, t))
    return exact_collect(out)


def exact_octant_target(
    terms: Sequence[ExactTerm],
    mu: Sequence[int],
    signs: Sequence[int],
    eps: Fraction,
) -> Tuple[Dict[Tuple[int, ...], Fraction], int]:
    """Exact per-octant associated form minus the strictness margin.

    Mirror of the target assembly in ``verify_gf_sos_full`` /
    ``term_to_y_monomial``: y-exponent μᵢ·aᵢ (must be a non-negative integer —
    checked), sign factor ∏ sᵢ over 'signed' positions (always, independent of
    the exponent), then ε subtracted from each pure top-degree power.

    Returns ``(target, d)`` with the homogeneity degree d. Raises
    ExactGateError if the lifted form is not a polynomial or not homogeneous —
    Propositions 6/7 require a homogeneous even-degree target.

    ``mu`` and ``signs`` are validated RAW here (positive integers resp.
    exactly ±1, both of length n) instead of being coerced: the declared lift
    is a premise of the whole chain, not a hint (review finding 5).
    """
    if isinstance(mu, (str, bytes, dict)) or not hasattr(mu, "__len__"):
        raise ExactGateError(f"invalid_mu({type(mu).__name__})")
    n = len(mu)
    mu = validate_mu(mu, n)
    signs = validate_signs(signs, n, strict=True)
    target: Dict[Tuple[int, ...], Fraction] = {}
    for c, exps, types in terms:
        if len(exps) != n or len(types) != n:
            raise ExactGateError(
                f"form term has {len(exps)} exponents / {len(types)} types, "
                f"the lift has {n} variables")
        if any(t not in ("abs", "signed") for t in types):
            raise ExactGateError(f"form term factor types {tuple(types)!r} are "
                                 f"not all 'abs'/'signed'")
        y_exp = []
        sign = 1
        for i in range(n):
            ye = mu[i] * exps[i]
            if ye.denominator != 1 or ye < 0:
                raise ExactGateError(
                    f"non-integer/negative y-exponent {ye} (mu={tuple(mu)}, exp={exps[i]})"
                )
            y_exp.append(int(ye))
            if types[i] == 'signed':
                sign *= signs[i]
        key = tuple(y_exp)
        target[key] = target.get(key, Fraction(0)) + sign * c
    target = {k: v for k, v in target.items() if v != 0}
    if not target:
        raise ExactGateError("octant target identically zero")
    degrees = {sum(k) for k in target}
    if len(degrees) != 1:
        raise ExactGateError(f"target not homogeneous (degrees {sorted(degrees)})")
    d = degrees.pop()
    if d <= 0 or d % 2 != 0:
        raise ExactGateError(f"target degree {d} not positive even")
    for i in range(n):
        pure = tuple(d if t == i else 0 for t in range(n))
        target[pure] = target.get(pure, Fraction(0)) - eps
    return target, d


# ---------------------------------------------------------------------------
# Deck group of the target support and the induced Gram parity blocks (R1)
# ---------------------------------------------------------------------------

def _f2_rref(rows: Sequence[int], n_vars: int) -> Tuple[List[int], List[int]]:
    """Reduced row echelon form over F₂ of bitmask-encoded rows.

    Returns ``(reduced_rows, pivot_columns)`` with one pivot per reduced row
    and zeros in every other pivot column, so the null space reads off
    directly. Pure integer bit twiddling: exact, no floats anywhere.
    """
    work = [r for r in rows if r]
    pivots: List[int] = []
    rank = 0
    for col in range(n_vars):
        pivot_row = None
        for i in range(rank, len(work)):
            if (work[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        for i in range(len(work)):
            if i != rank and ((work[i] >> col) & 1):
                work[i] ^= work[rank]
        pivots.append(col)
        rank += 1
        if rank == len(work):
            break
    return work[:rank], pivots


def deck_group_basis(
    support: Sequence[Tuple[int, ...]],
    n_vars: int,
) -> List[Tuple[int, ...]]:
    """Basis of H = {g ∈ F₂ⁿ : ⟨g, α⟩ ≡ 0 (mod 2) ∀ α ∈ support}.

    H is the group of sign flips y_i ↦ −y_i leaving the target invariant; it
    is the null space over F₂ of the matrix whose rows are the support
    exponents reduced mod 2. An all-even support gives only zero rows, hence
    H = F₂ⁿ (dimension n, the full deck group). ``dim H = 0`` means no usable
    symmetry and the caller must take the dense path.

    Exact by construction: the exponents are already integers in the exact
    target, and everything here is integer arithmetic mod 2.
    """
    rows: List[int] = []
    for alpha in support:
        mask = 0
        for i in range(n_vars):
            if int(alpha[i]) & 1:
                mask |= 1 << i
        if mask:
            rows.append(mask)
    reduced, pivot_cols = _f2_rref(rows, n_vars)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n_vars) if c not in pivot_set]
    basis: List[Tuple[int, ...]] = []
    for free in free_cols:
        g = 1 << free
        for k, pcol in enumerate(pivot_cols):
            # row k reads  g_{pcol} + Σ_{free c in row k} g_c = 0
            if (reduced[k] >> free) & 1:
                g |= 1 << pcol
        basis.append(tuple((g >> i) & 1 for i in range(n_vars)))
    return basis


def parity_blocks(
    basis: Sequence[Tuple[int, ...]],
    group_basis: Sequence[Tuple[int, ...]],
) -> List[List[int]]:
    """Partition Gram basis indices into the deck group's character classes.

    Basis exponents β, β' may pair in the symmetrized Gram Q̃ iff
    ⟨g, β + β'⟩ ≡ 0 (mod 2) for every g ∈ H, i.e. iff they carry the same
    character (⟨g, β⟩ mod 2)_{g ∈ H}. Returns one index list per non-empty
    class, ordered deterministically by class key, each list in increasing
    index order (so a single class reproduces the dense matrix exactly).
    """
    if not group_basis:
        return [list(range(len(basis)))]
    classes: Dict[Tuple[int, ...], List[int]] = {}
    for idx, beta in enumerate(basis):
        key = tuple(
            sum(g[i] * int(beta[i]) for i in range(len(g))) & 1
            for g in group_basis
        )
        classes.setdefault(key, []).append(idx)
    return [classes[k] for k in sorted(classes)]


def _blocks_psd(
    Q: Sequence[Sequence[Fraction]],
    blocks: Sequence[Sequence[int]],
    eta: Fraction,
) -> bool:
    """``Q̃ + η·I ⪰ 0`` by exact LDLᵀ on each parity block.

    A block-diagonal matrix is PSD iff every block is. With a single block
    spanning all indices this is literally the dense test on ``Q + ηI``.
    """
    for idxs in blocks:
        sub = [
            [Q[i][j] + eta if i == j else Q[i][j] for j in idxs]
            for i in idxs
        ]
        if not ldl_psd(sub):
            return False
    return True


# ---------------------------------------------------------------------------
# Exact PSD check: rational LDLᵀ with largest-diagonal pivoting
# ---------------------------------------------------------------------------

def ldl_psd(A: List[List[Fraction]]) -> bool:
    """Exact PSD test for a symmetric rational matrix (no square roots).

    Symmetric Gaussian elimination with largest-diagonal pivoting:
    - a negative pivot refutes PSD (diagonal entries of a PSD matrix are ≥ 0);
    - a zero maximal diagonal forces the whole remaining block to be zero
      (PSD with a_ii = 0 implies row/column i vanish, by 2×2 minors);
    - otherwise eliminate: the Schur complement of a positive pivot in a PSD
      matrix is PSD, and conversely all-nonnegative pivots exhibit A = LDLᵀ
      with D ≥ 0. All operations are exact in ℚ.
    """
    n = len(A)
    M = [list(row) for row in A]
    remaining = list(range(n))
    while remaining:
        p = max(remaining, key=lambda i: M[i][i])
        piv = M[p][p]
        if piv < 0:
            return False
        if piv == 0:
            # Max diagonal is 0 ⇒ all remaining diagonals ≤ 0; PSD needs them 0
            # and every remaining off-diagonal 0.
            return all(M[i][j] == 0 for i in remaining for j in remaining)
        remaining.remove(p)
        col = {i: M[i][p] for i in remaining}
        for i in remaining:
            ci = col[i]
            if ci == 0:
                continue
            row_i = M[i]
            for j in remaining:
                cj = col[j]
                if cj != 0:
                    row_i[j] -= ci * cj / piv
    return True


# ---------------------------------------------------------------------------
# Proposition 7: exact validation of one captured octant certificate
# ---------------------------------------------------------------------------

@dataclass
class ExactOctantCheck:
    ok: bool
    signs: Tuple[int, ...] = ()
    reason: str = ""
    degree: int = 0
    n_basis: int = 0
    eta_hat: str = ""          # exact Fraction as string (provenance)
    eta_hat_float: float = 0.0
    resid_l1_float: float = 0.0
    charge_float: float = 0.0
    budget_float: float = 0.0
    min_eig_float: float = 0.0
    # --- margin policy and the margin actually proved (WP P2-13) -----------
    # ``eps_certified`` is ``ε − charge`` in exact ℚ: the largest margin the
    # Proposition-7 chain proves from THIS Gram and THIS η̂, i.e. the number a
    # certificate claim may quote. It is recorded under both policies, and
    # under ``"half"`` it is automatically ≥ ε/2. On a FAILED octant it is
    # recorded too (it is then ≤ 0 or below the policy threshold) so an
    # artifact reader can see how far the Gram was from proving anything.
    margin_policy: str = MARGIN_POLICY_ACHIEVED
    eps_float: float = 0.0
    eps_certified: str = ""          # exact Fraction as string
    eps_certified_float: float = 0.0
    # --- deck-group blocking provenance (R1) -------------------------------
    # target_all_even is the exact support guard of review verdict 6; it is
    # recorded even when the group is computed from a partial symmetry, so an
    # artifact reader can tell the full-deck case from the degraded one.
    target_all_even: bool = False
    sym_group_dim: int = 0     # dim H; 0 = dense path taken
    n_blocks: int = 1
    block_sizes: Tuple[int, ...] = ()

    @property
    def charge_over_eps(self) -> Optional[float]:
        """``charge / ε``: the POLICY-FREE closeness measure (WP P2-14).

        ``charge_float / budget_float`` (the legacy ``charge/budget``) has the
        POLICY's budget as its denominator, so it halves when the policy moves
        from ``"half"`` to ``"achieved"`` on the very same Gram. This one names
        its denominator and never moves: ``charge_over_eps < 1`` is exactly the
        ``"achieved"`` acceptance test, and ``charge_over_eps ≤ 1/2`` is
        exactly the ε/2 inequality the frozen Automatica note prints.
        """
        if not self.eps_float:
            return None
        return self.charge_float / self.eps_float

    @property
    def note_half_margin_holds(self) -> bool:
        """Whether ``charge ≤ ε/2``, the inequality PRINTED in the frozen note.

        Reported, never enforced: acceptance is ``charge < ε``. A shipped
        record for which this is False proves a margin below ε/2 and must not
        be described by the note's sentence.
        """
        ratio = self.charge_over_eps
        return ratio is not None and ratio <= 0.5

    def r1_provenance(self) -> Dict[str, object]:
        """The four R1 blocking fields of this octant, JSON-ready.

        Serializing these is what makes the review-verdict-6 discharge
        auditable from an artifact rather than only from the source: an
        artifact reader sees whether the EXACT target support was all-even
        (the guard), which deck group H the arithmetic actually used
        (``sym_group_dim = 0`` means the dense path was taken), and the parity
        blocks the exact LDLᵀ ran on. ``block_sizes`` is a list, not a tuple,
        so the dict drops straight into ``json.dump``.

        The fields carry their dataclass defaults on an octant that failed
        before the group was computed (empty basis, degree mismatch, Gram
        shape mismatch, or a target that never assembled). Those defaults are
        the honest reading: no symmetry established, dense path.
        """
        return {
            "target_all_even": bool(self.target_all_even),
            "sym_group_dim": int(self.sym_group_dim),
            "n_blocks": int(self.n_blocks),
            "block_sizes": [int(b) for b in self.block_sizes],
        }


def exact_validate_octant(
    exact_target: Dict[Tuple[int, ...], Fraction],
    d: int,
    gram: np.ndarray,
    basis: Sequence[Tuple[int, ...]],
    eps: Fraction,
    signs: Tuple[int, ...] = (),
    margin_policy: str = MARGIN_POLICY_ACHIEVED,
) -> ExactOctantCheck:
    """Run the Proposition-7 chain on one captured (target, Gram, basis) triple.

    Every structural premise is established on the RAW capture first (review
    2026-09-05, finding 2): the target must be a nonnegative integer monomial
    support of one even degree d, the basis nonnegative integer exponents of
    that dimension and of degree d/2, without duplicates, and the Gram a
    square array of finite numbers. Those are the hypotheses of both bounds
    below; ``int()``-coercing them does not establish them.

    The Gram is symmetrized over the target's deck group H before the PSD test
    (module docstring, R1): the PSD test runs on the parity-block-diagonal
    Q̃ and decomposes into blocks, and the residual charged is the DENSE one
    of the captured Q̄, which bounds Q̃'s own residual from above (the
    portable verifier's rule, so P re-derives the same charge from the
    persisted Gram). ``dim H = 0`` reproduces the dense computation exactly.
    ``resid_l1_float`` and ``eps_certified`` therefore follow the dense
    residual (since 2026-09-25; before, the projected one).

    ``margin_policy`` selects the acceptance threshold only: ``"achieved"``
    (DEFAULT since P2-14) demands ``charge < ε`` strictly, ``"half"``
    (historical, the note's printed inequality) demands ``charge ≤ ε/2``.
    Nothing else in the chain changes — the same exact target, the same exact
    residual, the same exact LDLᵀ. What the accepted octant PROVES is
    ``F₀ ≥ (ε − charge) Σᵢ yᵢᵈ`` either way, and that number is recorded
    exactly as ``eps_certified``. ``note_half_margin_holds`` says whether this
    octant also satisfies the note's ε/2 inequality; it is reported, never
    enforced.
    """
    margin_policy = resolve_margin_policy(margin_policy)
    n_basis = len(basis) if hasattr(basis, "__len__") else 0
    fail = lambda why, **kw: ExactOctantCheck(ok=False, signs=tuple(signs), reason=why,
                                              degree=d, n_basis=n_basis,
                                              margin_policy=margin_policy,
                                              eps_float=float(eps), **kw)
    # Every premise of the Proposition-7 bounds, established on the RAW
    # capture before anything is coerced (review finding 2). The target fixes
    # the number of variables; the basis must be nonnegative integer exponents
    # of that dimension and of degree d/2 — a Laurent basis satisfies
    # ``2·Σβ = d`` and breaks both bounds.
    try:
        nv = validate_exact_target(exact_target, d)
        basis = validate_basis(basis, d, nv)
    except ExactGateError as exc:
        return fail(str(exc))
    n_basis = len(basis)

    # Q̄ = sym(Gram), exactly (Fraction(float) is exact).
    try:
        g = np.asarray(gram, dtype=float)
    except (TypeError, ValueError) as exc:
        return fail(f"gram_not_numeric({exc})")
    if g.shape != (n_basis, n_basis):
        return fail("gram_shape_mismatch")
    if not bool(np.all(np.isfinite(g))):
        # Fraction(nan) raises, and an inf entry would poison every exact
        # comparison below; refuse the capture instead.
        return fail("gram_not_finite")
    Q = [
        [(Fraction(float(g[i, j])) + Fraction(float(g[j, i]))) / 2 for j in range(n_basis)]
        for i in range(n_basis)
    ]

    # Deck group H of the EXACT target support (never a float-collected one),
    # and the Gram parity blocks it induces. The all-even test is recorded
    # separately as the verdict-6 guard; H is what the arithmetic uses, so a
    # partially-symmetric W target still blocks instead of falling to dense.
    all_even = all(all(e % 2 == 0 for e in mono) for mono in exact_target)
    group_basis = deck_group_basis(list(exact_target), nv)
    blocks = parity_blocks(basis, group_basis)
    prov = {
        "target_all_even": all_even,
        "sym_group_dim": len(group_basis),
        "n_blocks": len(blocks),
        "block_sizes": tuple(len(b) for b in blocks),
    }
    fail = lambda why, **kw: ExactOctantCheck(  # noqa: E731 - rebind with provenance
        ok=False, signs=tuple(signs), reason=why, degree=d, n_basis=n_basis,
        margin_policy=margin_policy, eps_float=float(eps), **prov, **kw)

    # E = target − zᵀQ̄z over ALL (i, j) pairs, coefficientwise and exact:
    # the DENSE residual of the captured Gram, the portable verifier's
    # definition (verify_certificate.py::validate_octant, review 2026-09-05
    # finding 12). The PSD test below still runs on the blocks of Q̃. Sound:
    # the cross-class entries only hit non-invariant monomials, which the
    # H-invariant target cannot contain, so target − zᵀQ̃z and the cross part
    # have disjoint supports and ‖target − zᵀQ̃z‖₁ ≤ ‖E‖₁. Charging ‖E‖₁ is
    # therefore conservative, and it is the number P re-derives from the
    # persisted Gram, so X never states a margin P cannot reproduce
    # (reports/XVC_RESIDUAL_2026-09-25.md).
    E: Dict[Tuple[int, ...], Fraction] = dict(exact_target)
    for i in range(n_basis):
        row_i = Q[i]
        beta_i = basis[i]
        for j in range(n_basis):
            q = row_i[j]
            if q == 0:
                continue
            mono = tuple(beta_i[k] + basis[j][k] for k in range(nv))
            E[mono] = E.get(mono, Fraction(0)) - q
    resid_l1 = sum(abs(v) for v in E.values())

    # C₁ = maxᵢ Σ_β 2βᵢ/d, exact.
    c1 = max(
        (Fraction(sum(2 * beta[i] for beta in basis), d) for i in range(nv)),
        default=Fraction(0),
    )
    budget = margin_budget(eps, margin_policy)
    # Under "achieved" the threshold is ε itself and the comparison is STRICT,
    # so the certified margin ε − charge stays strictly positive; under "half"
    # ``charge == ε/2`` is accepted, exactly as Proposition 6/7 is stated.
    strict = margin_policy == MARGIN_POLICY_ACHIEVED

    # The η̂ ladder is the portable verifier's, rung for rung
    # (verify_certificate.py::validate_octant): 0, 10⁻¹⁵ … 10⁻⁹ and the exact
    # Gershgorin bound of the DENSE Q̄, ascending. By pinching the Gershgorin
    # rung bounds Q̃ too, and the exact LDLᵀ is the authority on every rung.
    # Same Gram, same residual, same ladder: X and P select the same η̂ and
    # state the same charge (fix (e) of reports/XVC_RESIDUAL_2026-09-25.md,
    # author decision D9, 2026-09-26). The float λ_min of Q̄ is kept for the
    # record only.
    sym = (g + g.T) / 2.0
    lam_min = float(np.linalg.eigvalsh(sym).min())
    eta_g = max((sum(abs(Q[i][j]) for j in range(n_basis) if j != i) - Q[i][i]
                 for i in range(n_basis)), default=Fraction(0))
    ladder = {Fraction(0)}
    ladder.update(Fraction(1, 10 ** k) for k in range(15, 8, -1))
    if eta_g > 0:
        ladder.add(eta_g)
    candidates: List[Fraction] = sorted(ladder)

    last_charge: Optional[Fraction] = None
    for eta in candidates:
        charge = eta * c1 + resid_l1
        last_charge = charge
        over = charge >= budget if strict else charge > budget
        if over:
            # Larger η̂ only increases the charge — every later candidate fails too.
            return fail(
                "budget_exceeded",
                eta_hat=str(eta), eta_hat_float=float(eta),
                resid_l1_float=float(resid_l1), charge_float=float(charge),
                budget_float=float(budget), min_eig_float=lam_min,
                eps_certified=str(eps - charge),
                eps_certified_float=float(eps - charge),
            )
        if _blocks_psd(Q, blocks, eta):
            return ExactOctantCheck(
                ok=True, signs=tuple(signs), degree=d, n_basis=n_basis,
                eta_hat=str(eta), eta_hat_float=float(eta),
                resid_l1_float=float(resid_l1), charge_float=float(charge),
                budget_float=float(budget), min_eig_float=lam_min,
                margin_policy=margin_policy, eps_float=float(eps),
                eps_certified=str(eps - charge),
                eps_certified_float=float(eps - charge),
                **prov,
            )
    last = last_charge if last_charge is not None else Fraction(0)
    return fail(
        "no_psd_shift_found",
        resid_l1_float=float(resid_l1),
        charge_float=float(last),
        budget_float=float(budget), min_eig_float=lam_min,
        eps_certified=str(eps - last), eps_certified_float=float(eps - last),
    )


# ---------------------------------------------------------------------------
# Whole-certificate orchestration over verify_gf_sos_full captures
# ---------------------------------------------------------------------------

@dataclass
class ExactCaptureResult:
    ok: bool
    reason: str = ""
    octants: List[ExactOctantCheck] = field(default_factory=list)
    n_octants_expected: int = 0
    mu: Tuple[int, ...] = ()
    eps: str = ""
    margin_policy: str = MARGIN_POLICY_ACHIEVED

    @property
    def worst_charge_ratio(self) -> Optional[float]:
        """max over octants of charge/budget (closeness to the acceptance cliff).

        The denominator is the POLICY's budget: ε under ``"achieved"`` (the
        default since P2-14), ε/2 under ``"half"``. So the same Gram reported a
        ratio twice as large before the flip — the number measures headroom
        against the threshold in force, not against a fixed quantity. Compare
        ratios only within one policy; ``worst_charge_over_eps`` and
        ``eps_certified_float`` are the policy-free numbers, and a record
        without a ``margin_policy`` key predates P2-13 and is ``"half"``.
        """
        ratios = [
            c.charge_float / c.budget_float
            for c in self.octants if c.budget_float > 0
        ]
        return max(ratios) if ratios else None

    @property
    def worst_charge_over_eps(self) -> Optional[float]:
        """max over octants of ``charge / ε`` — the policy-free ratio (P2-14).

        Comparable ACROSS policies, unlike :meth:`worst_charge_ratio`, because
        its denominator is the requested ε and not the threshold in force.
        Under ``"achieved"`` the two coincide; under ``"half"`` this one is
        half of ``worst_charge_ratio``.
        """
        ratios = [r for r in (c.charge_over_eps for c in self.octants)
                  if r is not None]
        return max(ratios) if ratios else None

    @property
    def note_half_margin_holds(self) -> Optional[bool]:
        """Whether EVERY octant satisfies the note's printed ``charge ≤ ε/2``.

        ``None`` when no octant produced a charge. Reported, never enforced.
        """
        worst = self.worst_charge_over_eps
        return None if worst is None else worst <= 0.5

    @property
    def eps_certified_float(self) -> Optional[float]:
        """min over octants of ``ε − charge``: the margin this form PROVES.

        Policy-free: the same Gram gives the same number under ``"half"`` and
        ``"achieved"``. What the policy decides is only whether that number is
        large enough to accept. ``None`` when no octant produced a charge.
        """
        vals = [c.eps_certified_float for c in self.octants if c.eps_certified]
        return min(vals) if vals else None

    def r1_provenance(self) -> Dict[str, object]:
        """Roll-up of :meth:`ExactOctantCheck.r1_provenance` over one form.

        Summarizes the whole V (or W) proof in five numbers so a reader does
        not have to scan 2ⁿ octant entries to answer the two questions that
        matter: did the verdict-6 all-even guard hold everywhere, and did any
        octant fall back to the dense path. ``sym_group_dim_min = 0`` and
        ``n_octants_dense_fallback > 0`` are the same fact stated twice, kept
        because the first also reports HOW degraded a partial subgroup was.
        ``max_block_size`` is the largest matrix the exact LDLᵀ actually
        factored, i.e. what the O(N³) cost was really paid on.
        """
        octs = self.octants
        return {
            "target_all_even_all": bool(octs) and all(c.target_all_even for c in octs),
            "sym_group_dim_min": min((c.sym_group_dim for c in octs), default=0),
            "sym_group_dim_max": max((c.sym_group_dim for c in octs), default=0),
            "n_octants_dense_fallback": sum(1 for c in octs if c.sym_group_dim == 0),
            "max_block_size": max(
                (max(c.block_sizes, default=0) for c in octs), default=0),
        }


#: Sentinel :attr:`ExactCaptureResult.reason` for an octant whose exact target
#: could not be assembled at all. Kept as a module constant because
#: :func:`validate_one_capture` reports it out of band (a worker process cannot
#: mutate the caller's result object) and :func:`exact_verify_captures`
#: reproduces the historical early return from it.
TARGET_ASSEMBLY_FAILED = "target_assembly_failed"


def validate_one_capture(
    terms: Sequence[ExactTerm],
    capture: dict,
    n_vars: int,
    margin_policy: str = MARGIN_POLICY_ACHIEVED,
) -> Tuple[ExactOctantCheck, bool]:
    """The Proposition-7 chain for ONE captured octant. Pure and picklable.

    Extracted from the body of :func:`exact_verify_captures` so that the same
    unit of work runs in-process (``mapper=None``) or in a worker process
    (WP P3-03). It reads nothing outside its arguments, writes nothing, and
    holds no state, so N copies of it in N processes cannot interfere; the
    exact octant target is rebuilt from ``terms`` inside the call, exactly as
    the serial loop rebuilt it.

    Returns ``(check, target_assembly_failed)``. The second element is True
    when :func:`exact_octant_target` raised, which the caller turns into the
    :data:`TARGET_ASSEMBLY_FAILED` early return; it is NOT a verdict about the
    Gram, so it is typed separately from ``check.ok``.

    ``n_vars`` is the declared arity of the lift and of the sign label, both
    of which are validated RAW here (review finding 5): a capture declaring
    μ = 2.9, μ_i ≤ 0 or the wrong number of entries is refused, never coerced
    into a plausible neighbour. ``margin_policy`` (WP P2-13) travels
    positionally, so it pickles into a worker with the rest of the item and a
    worker never reads a default.
    """
    try:
        for field in ("mu", "signs", "eps"):
            if field not in capture:
                raise ExactGateError(f"capture has no {field!r}")
        mu = validate_mu(capture["mu"], n_vars)
        signs = validate_signs(capture["signs"], n_vars)
        eps = exact_eps(capture["eps"])  # the value the float pipeline subtracted
    except ExactGateError as exc:
        return ExactOctantCheck(ok=False, reason=f"invalid_capture({exc})"), True
    try:
        target, d = exact_octant_target(terms, mu, signs, eps)
    except ExactGateError as exc:
        return ExactOctantCheck(ok=False, signs=signs, reason=str(exc)), True
    return exact_validate_octant(
        target, d, capture["gram"], capture["basis"], eps, signs,
        margin_policy), False


def exact_verify_captures(
    terms: Sequence[ExactTerm],
    captures: Optional[Sequence[dict]],
    n_vars: int,
    mapper=None,
    margin_policy: str = MARGIN_POLICY_ACHIEVED,
) -> ExactCaptureResult:
    """Exact-verify all per-octant Gram captures for one form (V or W).

    ``terms`` is the exact form (V̂ as published, or Ŵ recomputed exactly from
    V̂ and the rationalized dynamics); ``captures`` comes from
    ``verify_gf_sos_full(..., capture_grams=True).gram_certificates``. The
    exact octant targets are rebuilt from ``terms`` with the captured μ and ε,
    then validated against the captured Grams (Proposition 7).

    ``mapper`` (WP P3-03) is the optional parallel evaluator. ``None`` is the
    historical serial loop, byte for byte, INCLUDING its early return at the
    first octant that fails: no later octant is even computed. A mapper is a
    callable ``mapper(fn, items) -> iterable of (index, fn(*item))`` which may
    yield in ANY completion order; this function assembles the results by
    index, so the octant order, the recorded reason and the truncation at the
    first failure are decided here and are independent of how the work was
    scheduled. A mapper that drops a result is an error
    (``octant_results_missing``), never a silently short certificate. See
    ``verification.exact_phase.octant_mapper``.

    This module still imports no ``multiprocessing``: the pool lives in
    ``exact_phase``, and what crosses the boundary is one pure function
    (:func:`validate_one_capture`) plus its arguments.
    """
    margin_policy = resolve_margin_policy(margin_policy)
    expected = 2 ** n_vars
    res = ExactCaptureResult(ok=False, n_octants_expected=expected,
                             margin_policy=margin_policy)
    if not captures:
        res.reason = "no_gram_captures"
        return res
    for i, c in enumerate(captures):
        if not isinstance(c, dict) or "signs" not in c or "eps" not in c:
            res.reason = f"capture_incomplete({i})"
            return res
    full_signs = set(product((-1, 1), repeat=n_vars))
    seen = set()
    for c in captures:
        try:
            # RAW integers: a label of 1.5 is refused, not truncated to 1.
            signs = validate_signs(c["signs"], n_vars)
        except ExactGateError as exc:
            res.reason = f"invalid_sign_label({exc})"
            return res
        if len(signs) != n_vars or signs not in full_signs:
            # M9: a fabricated/corrupted capture could carry a sign label
            # outside {-1,+1}^n (e.g. (0,) or a truncated tuple) — accepting
            # it here could smuggle a non-octant (hence possibly negative)
            # form past the count/distinctness check alone. Mirrors the
            # standalone verifier's `seen != full` membership check.
            res.reason = f"invalid_sign_label({signs})"
            return res
        seen.add(signs)
    if len(captures) != expected or len(seen) != expected:
        res.reason = f"octant_coverage_incomplete({len(seen)}/{expected})"
        return res
    try:
        eps_values = {exact_eps(c["eps"]) for c in captures}
    except ExactGateError as exc:
        res.reason = f"invalid_epsilon({exc})"
        return res
    if len(eps_values) != 1:
        # Defense in depth: every octant of one verification set must share
        # the same margin — a non-uniform ε across captures would let one
        # octant's proof rest on a smaller (weaker) margin than the others
        # while the caller believes a single ε was certified.
        res.reason = f"epsilon_not_uniform_across_captures({sorted(str(e) for e in eps_values)})"
        return res
    eps0 = next(iter(eps_values))
    if eps0 <= 0:
        # ε = 0 makes the budget ε/2 = 0: Proposition 7's charge bound would
        # have to be exactly 0 to "pass", which silently proves only
        # non-strict positivity (or accepts on a razor's edge) instead of the
        # intended strict margin.
        res.reason = f"epsilon_not_positive({eps0})"
        return res
    # One form is proved under ONE lift: every capture must declare the same
    # positive integral μ of the right arity, or the 2ⁿ octants are charts of
    # different substitutions and do not cover ℝⁿ between them (review
    # finding 5). Validated RAW — ``int(2.9)`` would have said 2.
    for i, c in enumerate(captures):
        if "mu" not in c:
            res.reason = f"capture_incomplete({i}:['mu'])"
            return res
    try:
        mu_values = {validate_mu(c["mu"], n_vars) for c in captures}
    except ExactGateError as exc:
        res.reason = f"invalid_mu({exc})"
        return res
    if len(mu_values) != 1:
        res.reason = (f"mu_not_uniform_across_captures("
                      f"{sorted(str(m) for m in mu_values)})")
        return res
    res.mu = next(iter(mu_values))
    res.eps = str(eps0)

    if mapper is None:
        # Serial: unchanged, including the early return that leaves the
        # remaining octants uncomputed.
        outcomes = ((i, validate_one_capture(terms, cap, n_vars, margin_policy))
                    for i, cap in enumerate(captures))
    else:
        slots: List[Optional[Tuple[ExactOctantCheck, bool]]] = [None] * len(captures)
        for index, outcome in mapper(
                validate_one_capture,
                [(terms, cap, n_vars, margin_policy) for cap in captures]):
            slots[int(index)] = outcome
        missing = [i for i, s in enumerate(slots) if s is None]
        if missing:
            # A mapper that lost work must never look like a shorter, passing
            # certificate: the octant coverage rule above counted these.
            raise ExactGateError(
                f"octant_results_missing({missing[:8]}"
                f"{'...' if len(missing) > 8 else ''} of {len(captures)})")
        outcomes = enumerate(slots)

    # Assembly is ALWAYS in capture order and always truncates at the first
    # failing octant, so a parallel run records exactly what the serial run
    # records.
    for i, (check, assembly_failed) in outcomes:
        res.octants.append(check)
        if assembly_failed:
            res.reason = TARGET_ASSEMBLY_FAILED
            return res
        if not check.ok:
            res.reason = f"octant{captures[i]['octant_index']}:{check.reason}"
            return res
    res.ok = True
    return res
