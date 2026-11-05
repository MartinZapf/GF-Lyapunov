#!/usr/bin/env python3
"""Exact SYMBOLIC even-support guard for the y-parity block decomposition.

Promoted into the package by WP P5-04 from
``artifacts/campaign_2026-08/E4_levant4d/structure/prototype/support_guard.py``.
It is the premise check of ``verification/yparity.py``: no block is ever split
before this module has confirmed, symbolically and before any float
arithmetic, that every object that can reach a design target lies on the even
lattice.

Repair R2 for finding 6 of the internal y-parity review: the runtime guard
in yparity.split_gram_block inspects the ASSEMBLED FLOAT matrices, i.e. the
support AFTER float collection and tolerance deletion. Two odd-lift terms
with cancelling float coefficients leave a numerically zero row, so the
float guard can pass while the symbolic support is odd, and the averaging
argument then has no premise. This module inspects the EXACT
rational/symbolic support BEFORE any float arithmetic or cancellation:

  * every V library column: lift mu_i * a_i for each term exponent a;
  * the epsilon rows: pure powers y_i^{d_V} (d_V from the exact lifts);
  * every W column, per inclusion endpoint: the Lie-product support
    { a - e_ch + b : a in library, x^b in channel ch of the endpoint
    dynamics }, term by term, WITHOUT summing coefficients (a cancelling
    pair still counts as present);
  * the clearing factor: exponents k * rho_max / r_i, checked symbolically;
  * optional SOS multiplier supports (future (sum y_i^2)^k gates): each
    multiplier monomial lift must be even.

Gain values (including exact-phase bracket corners) multiply COEFFICIENTS
only and never change any exponent, so corner coverage is inherited from
the per-endpoint dynamics support; this is asserted structurally by taking
exponents from the GF term objects, never from evaluated coefficients.

The check is the coordinatewise condition of review finding 5: for every
object monomial x^b in play, mu_i * b_i must lie in 2Z for every i. All
arithmetic is over fractions.Fraction; a violation raises a typed
OddLiftRefusal carrying the offending (stage, object, variable, exponent,
lift). No solver, no numpy, no float.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

Exponents = Tuple[Fraction, ...]


@dataclass
class OddLift:
    stage: str          # "V" | "W" | "clearing" | "epsilon" | "multiplier"
    obj: str            # human-readable provenance of the term
    variable: int
    exponent: Fraction  # x-space exponent b_i
    lift: Fraction      # mu_i * b_i (not in 2Z)


class OddLiftRefusal(RuntimeError):
    """The even-support premise fails SYMBOLICALLY; y-parity must refuse."""

    def __init__(self, violations: List[OddLift]):
        self.violations = violations
        head = violations[0]
        super().__init__(
            f"odd lift: stage={head.stage} obj={head.obj} var={head.variable} "
            f"exponent={head.exponent} lift={head.lift} "
            f"(+{len(violations) - 1} more)")


@dataclass
class GuardReport:
    ok: bool
    n_v_terms: int
    n_w_products: int
    n_endpoints: int
    clearing_checked: bool
    violations: List[OddLift] = field(default_factory=list)


def _fr(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    # exact only: strings of rationals are fine, floats are refused unless
    # they are exactly representable small dyadics coming from int-valued
    # exponents. Guard callers should hand over Fractions.
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, float):
        if float(int(x)) == x:
            return Fraction(int(x))
        raise TypeError(f"non-integer float exponent {x!r} refused: the "
                        f"symbolic guard requires exact exponents")
    raise TypeError(f"unsupported exponent type {type(x)}")


def _check_monomial(mu: Sequence[int], exps: Iterable, stage: str, obj: str,
                    out: List[OddLift]) -> None:
    for i, e in enumerate(exps):
        b = _fr(e)
        lift = Fraction(mu[i]) * b
        if lift.denominator != 1 or lift.numerator % 2 != 0:
            out.append(OddLift(stage=stage, obj=obj, variable=i,
                               exponent=b, lift=lift))


def check_even_support(
    *,
    mu: Sequence[int],
    weights_r: Sequence[int],
    library_exponents: Sequence[Exponents],
    dynamics_vertices: Sequence[Sequence[Sequence[Tuple[Exponents, str]]]],
    clearing_power: int = 0,
    multiplier_supports: Optional[Sequence[Exponents]] = None,
    raise_on_violation: bool = True,
) -> GuardReport:
    """Symbolic even-support check over every object that can reach a target.

    Args:
      mu: per-variable lift exponents (integers).
      weights_r: homogeneity weights (for the clearing factor).
      library_exponents: exact x-space exponent tuples of every V term.
      dynamics_vertices: per ENDPOINT, per CHANNEL, a list of
        (exponents, provenance-string) for every GF monomial of that
        channel, exactly as in the spec objects (constant disturbance
        terms have all-zero exponents). Coefficients are irrelevant here:
        presence is decided symbolically, cancellation is ignored ON
        PURPOSE.
      clearing_power: k of the clearing monomial C_k (exponent
        k * max(r) / r_i on variable i); 0 = no clearing.
      multiplier_supports: optional extra multiplier monomials (x-space).
    """
    n = len(mu)
    viol: List[OddLift] = []

    # --- V stage: library lifts + epsilon rows -------------------------
    v_lift_degrees = set()
    for t_idx, exps in enumerate(library_exponents):
        _check_monomial(mu, exps, "V", f"library[{t_idx}]", viol)
        d = sum(Fraction(mu[i]) * _fr(e) for i, e in enumerate(exps))
        v_lift_degrees.add(d)
    for d in v_lift_degrees:
        # epsilon rows are pure powers y_i^d at the V lift degree
        if d.denominator != 1 or d.numerator % 2 != 0:
            viol.append(OddLift(stage="epsilon", obj=f"y_i^{d}",
                                variable=-1, exponent=d, lift=d))

    # --- W stage: symbolic Lie-product support, per endpoint -----------
    n_products = 0
    for v_idx, channels in enumerate(dynamics_vertices):
        if len(channels) != n:
            raise ValueError(f"endpoint {v_idx}: {len(channels)} channels, "
                             f"expected {n}")
        for ch, monos in enumerate(channels):
            for exps, prov in monos:
                # the dynamics monomial itself must have even lifts
                # (coordinatewise condition, review finding 5)
                _check_monomial(mu, exps, "W",
                                f"endpoint{v_idx}.f{ch}.{prov}", viol)
                # and every Lie product a - e_ch + b (only the terms with
                # a_ch != 0 exist, but a_ch's lift parity is unchanged by
                # the shift since mu_ch * 1 = mu_ch must be even; check
                # that too, once per channel)
                n_products += len(library_exponents)
        if any(int(mu[ch]) % 2 != 0 for ch in range(n)):
            for ch in range(n):
                if int(mu[ch]) % 2 != 0:
                    viol.append(OddLift(
                        stage="W", obj=f"d/dx_{ch} shift (mu_ch odd)",
                        variable=ch, exponent=Fraction(1),
                        lift=Fraction(mu[ch])))

    # NOTE the decomposition of the product condition: lift(a - e_ch + b)
    # = lift(a) - mu_ch + lift(b) coordinatewise; with lift(a), lift(b)
    # even (checked above) and mu even (checked above) every Lie product
    # is even. This is exactly the provable conditional version of review
    # finding 5; no per-product enumeration is needed once the three
    # ingredient checks hold, but n_products records the covered count.

    # --- clearing factor ----------------------------------------------
    clearing_checked = False
    if clearing_power:
        rho = max(int(r) for r in weights_r)
        for i in range(n):
            e = Fraction(clearing_power * rho, int(weights_r[i]))
            _check_monomial(mu, (e if j == i else Fraction(0)
                                 for j in range(n)),
                            "clearing", f"C_{clearing_power}", viol)
        clearing_checked = True

    # --- optional multipliers ------------------------------------------
    for m_idx, exps in enumerate(multiplier_supports or []):
        _check_monomial(mu, exps, "multiplier", f"multiplier[{m_idx}]", viol)

    report = GuardReport(ok=not viol, n_v_terms=len(library_exponents),
                         n_w_products=n_products,
                         n_endpoints=len(dynamics_vertices),
                         clearing_checked=clearing_checked,
                         violations=viol)
    if viol and raise_on_violation:
        raise OddLiftRefusal(viol)
    return report


class GuardUnreadable(TypeError):
    """A dynamics/library object the symbolic guard cannot read exactly.

    Never a proof of anything: it means the even-support premise could not be
    CHECKED, so the caller must decline the y-parity blocking rather than
    assume it (WP P5-04).
    """


def gf_exponent_support(vertices) -> List[List[List[Tuple[Exponents, str]]]]:
    """Symbolic support of a list of inclusion endpoints, exponents only.

    ``vertices`` is a list (one entry per inclusion endpoint) of channel lists
    of ``GeneralizedForm`` (or ``None``). Coefficients are ignored ON PURPOSE:
    a cancelling pair of odd-lift monomials must still count as present, which
    is exactly the failure mode the float-side runtime guard cannot see.
    Raises :class:`GuardUnreadable` for a term whose exponents are not exact.
    """
    out: List[List[List[Tuple[Exponents, str]]]] = []
    for v_idx, dyn in enumerate(vertices):
        channels: List[List[Tuple[Exponents, str]]] = []
        for ch, gf in enumerate(dyn):
            monos: List[Tuple[Exponents, str]] = []
            if gf is not None:
                for t_idx, term in enumerate(getattr(gf, "terms", ())):
                    exps = getattr(term, "exponents", None)
                    if exps is None:
                        raise GuardUnreadable(
                            f"endpoint{v_idx}.f{ch}.term{t_idx} carries no "
                            f"exponent tuple ({type(term).__name__})")
                    try:
                        monos.append((tuple(_fr(e) for e in exps),
                                      f"term{t_idx}"))
                    except TypeError as exc:
                        raise GuardUnreadable(
                            f"endpoint{v_idx}.f{ch}.term{t_idx}: {exc}") from exc
            channels.append(monos)
        out.append(channels)
    return out


def guard_from_spec(spec, term_specs, mu,
                    raise_on_violation: bool = True) -> GuardReport:
    """Convenience wrapper: run the symbolic guard on a SystemSpec +
    library, walking the actual GF objects (exponents only, never
    coefficients)."""
    vertices = spec.dynamics_gf_vertices or [spec.dynamics_gf]
    dyn_sym = gf_exponent_support(vertices)
    lib = [tuple(_fr(e) for e in t.exponents) for t in term_specs]
    return check_even_support(
        mu=mu, weights_r=tuple(int(r) for r in spec.weights_r),
        library_exponents=lib, dynamics_vertices=dyn_sym,
        clearing_power=0, raise_on_violation=raise_on_violation)
