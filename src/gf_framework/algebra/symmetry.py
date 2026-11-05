"""
Parity and symmetry reduction for the GF-SOS problem (campaign item C1, block S1).

The benchmark dynamics are odd, f(-x) = -f(x). Oddness is a property of the
system, so it is VERIFIED SYMBOLICALLY per system here and never assumed.
Two reductions follow from it.

**Nominal.** If V solves the joint SOS problem then so does V∘(-id), and the
feasible set is convex, so their average solves it too. That average has zero
coefficient on every odd library term. Restricting the library to its
even-parity part is therefore without loss of generality. For an even V both
associated forms of a ± octant pair coincide,

    Φ_{-s}^V = Φ_s^V,      Φ_{-s}^W = Φ_s^W,

so one Gram block per pair suffices: 2^(n-1) blocks instead of 2^n.

**Perturbed (the subtlety).** A FIXED inclusion endpoint is NOT odd. With
f_σ = f + σ·δ₀·e_c (σ = ±1, the endpoints built by
``discovery.universal_discovery.matched_disturbance_vertices``),

    f_σ(-x) = -f_{-σ}(x),

so the global sign reversal SWAPS the endpoints. The induced identity on the
decrease targets is

    W_σ(-x) = W_{-σ}(x)      hence      Φ_{-s}^{W,σ} = Φ_s^{W,-σ},

and the correct reduction pairs (octant s, endpoint +) with
(octant -s, endpoint -). Naive octant halving at fixed endpoint is UNSOUND:
the mismatch W_σ(-x) - W_σ(x) equals 2σδ₀·∂V/∂x_c, which never vanishes
identically for a positive definite V and δ₀ > 0. The API below therefore
requires every caller to state the inclusion structure: the vertex argument of
each constructor is mandatory, and a caller holding endpoints cannot obtain a
nominal-style halving at all.

**Gain corners (conceptual extension, wiring lands later).** In the exact
phase the fields are additionally indexed by a corner c of a gain box
(``scripts/exact_gate_standard_gains.py``). A gain multiplies terms of the odd
base field, so f_c stays odd at every fixed corner and the corner index is
INERT under the reflection. The block involution on the full index set is

    (s, σ, c)  ↦  (-s, -σ, c),

that is: one paired endpoint reduction per corner, corners never mixed. The
proof is in ``docs/papers/theory_notes.md`` §10 (Propositions 10.1, 10.2).

Representations used here are the ones the pipeline already uses, never
parallel ones:

* library terms: :class:`~gf_framework.algebra.enumerate.TermSpec` (and
  :class:`~gf_framework.algebra.core.GFTerm`), whose ``term_types`` entry is
  ``'signed'`` or ``'abs'`` per axis. A ``'signed'`` axis always contributes
  sign(x_i), also at exponent 0 (``joint_sos.term_to_y_monomial``), so the
  parity of a term is (-1)^(number of signed axes) and does not depend on the
  exponents or on the weights r.
* octants: sign tuples from
  :func:`~gf_framework.algebra.associated.get_hyperoctant_signs`, in that
  order; :func:`octant_index` reproduces the index used by the SOS loops.
* endpoints: the INDEX into ``dynamics_gf_vertices``. That list is built as
  ``[f + δ₀·e_c, f - δ₀·e_c]``, so index 0 is the ``+`` endpoint and index 1
  the ``-`` endpoint (:data:`ENDPOINT_PLUS`, :data:`ENDPOINT_MINUS`).

Nothing in this module solves anything; it is symbolic and combinatorial only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp

from .associated import get_hyperoctant_signs
from .core import CompositeGFTerm, GeneralizedForm, GFTerm
from .enumerate import TermSpec

__all__ = [
    "PARITY_EVEN",
    "PARITY_ODD",
    "ENDPOINT_PLUS",
    "ENDPOINT_MINUS",
    "ParityReductionError",
    "SymmetryCheck",
    "ConsistencyReport",
    "NominalParityReduction",
    "PairedParityReduction",
    "ParityReductionPlan",
    "signed_axis_count",
    "term_parity",
    "is_even_term",
    "gf_parity",
    "verify_dynamics_odd",
    "verify_endpoint_swap",
    "split_library_by_parity",
    "filter_even_parity_terms",
    "validate_library_split",
    "flip_octant",
    "flip_endpoint",
    "octant_index",
    "octant_from_index",
    "endpoint_count",
    "nominal_octant_reduction",
    "paired_endpoint_reduction",
    "build_parity_reduction",
    "validate_reduction",
    "plan_parity_reduction",
]


# =============================================================================
# Conventions
# =============================================================================

PARITY_EVEN = 1
PARITY_ODD = -1

#: Index of the ``f + δ₀·e_c`` endpoint in a ``dynamics_gf_vertices`` list.
ENDPOINT_PLUS = 0
#: Index of the ``f - δ₀·e_c`` endpoint in a ``dynamics_gf_vertices`` list.
ENDPOINT_MINUS = 1

ENDPOINT_LABELS: Dict[int, str] = {ENDPOINT_PLUS: "+", ENDPOINT_MINUS: "-"}

Octant = Tuple[int, ...]
#: A verification block: an octant plus an endpoint index (None when nominal).
BlockLabel = Tuple[Octant, Optional[int]]

#: Accepted ways to declare the inclusion structure: no endpoints (None), the
#: vertex list itself, or just the number of endpoints (for pure size counting).
VertexArg = Union[None, int, Sequence[Sequence[Optional[GeneralizedForm]]]]


class ParityReductionError(ValueError):
    """Raised when a parity reduction is requested in an unsound way."""


# =============================================================================
# Reports
# =============================================================================

@dataclass(frozen=True)
class SymmetryCheck:
    """Result of one symbolic symmetry check on a vector field.

    Attributes:
        holds: True if the identity holds exactly for every component.
        identity: The identity that was checked, as text.
        n_vars: Number of state variables.
        residuals: Per-component residual, simplified, as a string. "0" when
            the component satisfies the identity. Kept as text so it can go
            straight into a run JSON.
        offending: Indices of the components whose residual is not zero.
        structural_parity: Per-component parity derived from the term
            structure (+1 even, -1 odd, None undetermined). Corroborating
            information only; the symbolic residual decides.
        message: Human-readable summary.
    """

    holds: bool
    identity: str
    n_vars: int
    residuals: Tuple[str, ...]
    offending: Tuple[int, ...]
    structural_parity: Tuple[Optional[int], ...]
    message: str
    residual_exprs: Tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __bool__(self) -> bool:
        return self.holds


@dataclass
class ConsistencyReport:
    """Outcome of the self-consistency checks of a reduction or a split."""

    ok: bool
    subject: str
    checks: Dict[str, bool] = field(default_factory=dict)
    message: str = ""

    def failures(self) -> List[str]:
        """Names of the checks that failed."""
        return [name for name, passed in self.checks.items() if not passed]

    def __bool__(self) -> bool:
        return self.ok


# =============================================================================
# Term parity
# =============================================================================

def signed_axis_count(term: Union[TermSpec, GFTerm]) -> int:
    """Number of axes on which the term carries a signed power.

    A ``'signed'`` axis contributes the factor sign(x_i) in every octant,
    including at exponent 0 (framework convention, see
    ``joint_sos.term_to_y_monomial``). That factor is the only source of
    sign change under x -> -x.
    """
    term_types = getattr(term, "term_types", None)
    if term_types is None:
        raise TypeError(
            f"expected a TermSpec/GFTerm with 'term_types', got {type(term).__name__}"
        )
    return sum(1 for t in term_types if t == "signed")


def term_parity(term: Union[TermSpec, GFTerm]) -> int:
    """Parity of a GF term under x -> -x.

    Returns ``PARITY_EVEN`` (+1) if t(-x) = t(x) and ``PARITY_ODD`` (-1) if
    t(-x) = -t(x). Every GF term has one of the two: |x_i|^a is invariant and
    ⌈x_i⌋^a flips, so t(-x) = (-1)^k t(x) with k the number of signed axes.
    The exponents and the weights r do not enter.
    """
    return PARITY_EVEN if signed_axis_count(term) % 2 == 0 else PARITY_ODD


def is_even_term(term: Union[TermSpec, GFTerm]) -> bool:
    """True if the term is invariant under x -> -x."""
    return term_parity(term) == PARITY_EVEN


def _composite_term_parity(ct: CompositeGFTerm) -> Optional[int]:
    """Parity of a nested GF term, or None if the inner form has no parity.

    ``|σ|^a`` is even whenever σ has a definite parity, and ``⌈σ⌋^a`` inherits
    the parity of σ.
    """
    inner = gf_parity(ct.inner_gf)
    if inner is None:
        return None
    if ct.outer_type == "abs":
        return PARITY_EVEN
    return inner


def gf_parity(gf: Optional[GeneralizedForm]) -> Optional[int]:
    """Parity of a whole generalized form, or None if it has no definite one.

    None means either "mixed" (terms of both parities are present) or "not
    determined" (a nested term whose inner form is itself mixed). The empty
    form is both even and odd and also returns None; callers that need a
    convention for the zero field must state it themselves.
    """
    if gf is None:
        return None
    parities = set()
    for t in gf.terms:
        if t.coefficient == 0:
            continue
        parities.add(term_parity(t))
    for ct in getattr(gf, "composite_terms", []):
        if ct.coefficient == 0:
            continue
        p = _composite_term_parity(ct)
        if p is None:
            return None
        parities.add(p)
    if len(parities) != 1:
        return None
    return parities.pop()


# =============================================================================
# Symbolic checks
# =============================================================================

def _gf_to_sympy(gf: Optional[GeneralizedForm], symbols: Sequence[sp.Symbol]) -> sp.Expr:
    """Sum the term expressions of a GF without the global simplify pass."""
    if gf is None:
        return sp.Integer(0)
    expr = sp.Integer(0)
    for t in gf.terms:
        expr = expr + t.to_sympy(list(symbols))
    for ct in getattr(gf, "composite_terms", []):
        expr = expr + ct.to_sympy(list(symbols))
    return expr


def _reflect(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> sp.Expr:
    """Substitute x -> -x simultaneously in every variable."""
    return expr.subs({s: -s for s in symbols}, simultaneous=True)


def _state_symbols(n_vars: int) -> List[sp.Symbol]:
    return [sp.Symbol(f"x{i + 1}", real=True) for i in range(n_vars)]


def verify_dynamics_odd(
    dynamics_gf: Sequence[Optional[GeneralizedForm]],
) -> SymmetryCheck:
    """Check f(-x) = -f(x) symbolically, component by component.

    The residual of component i is ``simplify(f_i(-x) + f_i(x))``. The check
    holds only when every residual is exactly zero. A component that does not
    reduce to zero is reported with its residual, so a failure carries the
    proof-relevant object and not just a flag.

    The parity of the terms is reported alongside as ``structural_parity``,
    but it never overrides the symbolic verdict: a disagreement between the
    two is itself reported as a failure, since it means the term
    representation and the symbolic image drifted apart.
    """
    n_vars = len(dynamics_gf)
    if n_vars == 0:
        raise ValueError("dynamics_gf is empty")
    symbols = _state_symbols(n_vars)

    residual_exprs: List[sp.Expr] = []
    residuals: List[str] = []
    offending: List[int] = []
    structural: List[Optional[int]] = []

    for i, comp in enumerate(dynamics_gf):
        expr = _gf_to_sympy(comp, symbols)
        residual = sp.simplify(sp.expand(_reflect(expr, symbols) + expr))
        residual_exprs.append(residual)
        residuals.append(str(residual))
        if residual != 0:
            offending.append(i)
        structural.append(gf_parity(comp))

    struct_odd = all(p == PARITY_ODD for p in structural if p is not None)
    holds = not offending
    if holds and not struct_odd:
        holds = False
        message = (
            "symbolic residuals vanish but the term structure is not odd on "
            "every component; term representation and symbolic image disagree"
        )
    elif holds:
        message = f"f(-x) = -f(x) verified symbolically on {n_vars} components"
    else:
        message = (
            f"dynamics not odd: components {offending} have nonzero residual "
            f"f_i(-x) + f_i(x)"
        )

    return SymmetryCheck(
        holds=holds,
        identity="f(-x) = -f(x)",
        n_vars=n_vars,
        residuals=tuple(residuals),
        offending=tuple(offending),
        structural_parity=tuple(structural),
        message=message,
        residual_exprs=tuple(residual_exprs),
    )


def verify_endpoint_swap(
    dynamics_gf_vertices: Sequence[Sequence[Optional[GeneralizedForm]]],
) -> SymmetryCheck:
    """Check the endpoint-swap identity f_+(-x) = -f_-(x) symbolically.

    This is the hypothesis of the paired reduction. It is the perturbed
    replacement for oddness: neither endpoint is odd on its own once δ₀ > 0,
    but the reflection maps one endpoint onto the other.

    The residual of component i is
    ``simplify(f_{+,i}(-x) + f_{-,i}(x))``. Only this direction is checked:
    substituting x -> -x in it gives the other direction, so the two are
    equivalent.

    Args:
        dynamics_gf_vertices: Exactly two vertex dynamics lists, in the order
            produced by ``matched_disturbance_vertices``: index 0 is the ``+``
            endpoint, index 1 the ``-`` endpoint.
    """
    if len(dynamics_gf_vertices) != 2:
        raise ParityReductionError(
            f"endpoint swap needs exactly 2 inclusion endpoints, got "
            f"{len(dynamics_gf_vertices)}"
        )
    plus, minus = dynamics_gf_vertices[ENDPOINT_PLUS], dynamics_gf_vertices[ENDPOINT_MINUS]
    if len(plus) != len(minus):
        raise ParityReductionError(
            f"endpoint dynamics have different dimensions: {len(plus)} vs {len(minus)}"
        )

    n_vars = len(plus)
    symbols = _state_symbols(n_vars)

    residual_exprs: List[sp.Expr] = []
    residuals: List[str] = []
    offending: List[int] = []

    for i in range(n_vars):
        expr_plus = _gf_to_sympy(plus[i], symbols)
        expr_minus = _gf_to_sympy(minus[i], symbols)
        residual = sp.simplify(sp.expand(_reflect(expr_plus, symbols) + expr_minus))
        residual_exprs.append(residual)
        residuals.append(str(residual))
        if residual != 0:
            offending.append(i)

    holds = not offending
    message = (
        f"f_+(-x) = -f_-(x) verified symbolically on {n_vars} components"
        if holds
        else f"endpoint swap fails on components {offending}"
    )

    return SymmetryCheck(
        holds=holds,
        identity="f_+(-x) = -f_-(x)",
        n_vars=n_vars,
        residuals=tuple(residuals),
        offending=tuple(offending),
        structural_parity=tuple(gf_parity(c) for c in plus),
        message=message,
        residual_exprs=tuple(residual_exprs),
    )


# =============================================================================
# Even-parity sub-library
# =============================================================================

def _check_arity(terms: Sequence[Any], weights_r: Optional[Tuple[int, ...]]) -> None:
    if weights_r is None:
        return
    n = len(weights_r)
    for t in terms:
        if len(t.term_types) != n:
            raise ValueError(
                f"term has {len(t.term_types)} axes but weights_r has {n}"
            )


def split_library_by_parity(
    terms: Sequence[Union[TermSpec, GFTerm]],
    weights_r: Optional[Tuple[int, ...]] = None,
) -> Tuple[List[Union[TermSpec, GFTerm]], List[Union[TermSpec, GFTerm]]]:
    """Split a term library into its even and its odd part, order preserved.

    Args:
        terms: The library, as the pipeline builds it (diagonal + μ-compatible
            cross terms, see ``gf_discovery.mu_filtered_cross``).
        weights_r: Optional, only used to assert that the terms have the right
            number of axes. Parity itself does not depend on r.

    Returns:
        ``(even_terms, odd_terms)``.
    """
    _check_arity(terms, weights_r)
    even: List[Union[TermSpec, GFTerm]] = []
    odd: List[Union[TermSpec, GFTerm]] = []
    for t in terms:
        (even if is_even_term(t) else odd).append(t)
    return even, odd


def filter_even_parity_terms(
    terms: Sequence[Union[TermSpec, GFTerm]],
    weights_r: Optional[Tuple[int, ...]] = None,
) -> List[Union[TermSpec, GFTerm]]:
    """Return the even-parity sub-library, in the input order.

    Restricting the design to this sub-library is without loss of generality
    for odd dynamics (theory_notes.md §10, Proposition 10.1). The diagonal
    terms |x_i|^(m/r_i) are always even, so the positive-definiteness backbone
    of the library always survives the filter.
    """
    even, _ = split_library_by_parity(terms, weights_r)
    return even


def validate_library_split(
    terms: Sequence[Union[TermSpec, GFTerm]],
    even_terms: Sequence[Union[TermSpec, GFTerm]],
    odd_terms: Sequence[Union[TermSpec, GFTerm]],
) -> ConsistencyReport:
    """Check that a parity split kept exactly the even terms and no others."""
    checks = {
        "kept_all_even": all(is_even_term(t) for t in even_terms),
        "dropped_all_odd": all(not is_even_term(t) for t in odd_terms),
        "counts_add_up": len(even_terms) + len(odd_terms) == len(terms),
        "kept_from_library": all(t in list(terms) for t in even_terms),
        "diagonals_survive": all(
            is_even_term(t) for t in terms if getattr(t, "is_diagonal", lambda: False)()
        ),
    }
    ok = all(checks.values())
    return ConsistencyReport(
        ok=ok,
        subject="library parity split",
        checks=checks,
        message=(
            f"{len(even_terms)} even / {len(odd_terms)} odd of {len(terms)} terms"
            if ok
            else "library parity split inconsistent"
        ),
    )


# =============================================================================
# Octant and endpoint combinatorics
# =============================================================================

def flip_octant(signs: Octant) -> Octant:
    """The antipodal octant -s."""
    return tuple(-int(s) for s in signs)


def flip_endpoint(endpoint: Optional[int]) -> Optional[int]:
    """Swap the inclusion endpoints, + <-> -. None (nominal) stays None."""
    if endpoint is None:
        return None
    if endpoint not in (ENDPOINT_PLUS, ENDPOINT_MINUS):
        raise ParityReductionError(f"unknown endpoint index {endpoint}")
    return ENDPOINT_MINUS if endpoint == ENDPOINT_PLUS else ENDPOINT_PLUS


def octant_index(signs: Octant) -> int:
    """Position of an octant in ``get_hyperoctant_signs(len(signs))``.

    That enumeration sets bit i of the index when s_i = -1, so the index is
    the one the SOS loops use for their Gram block names.
    """
    idx = 0
    for i, s in enumerate(signs):
        if int(s) == -1:
            idx |= 1 << i
        elif int(s) != 1:
            raise ParityReductionError(f"octant entries must be +1 or -1, got {signs}")
    return idx


def octant_from_index(index: int, n_vars: int) -> Octant:
    """Inverse of :func:`octant_index`."""
    if not 0 <= index < 2 ** n_vars:
        raise ParityReductionError(
            f"octant index {index} out of range for n_vars={n_vars}"
        )
    return tuple(1 if (index >> i) & 1 == 0 else -1 for i in range(n_vars))


def _representative_octants(n_vars: int) -> Tuple[Octant, ...]:
    """One octant per antipodal pair: those with s_1 = +1.

    The order of ``get_hyperoctant_signs`` is preserved, so a reduced loop
    visits the representatives in the same relative order as the full loop.
    """
    return tuple(s for s in get_hyperoctant_signs(n_vars) if s[0] == 1)


def endpoint_count(dynamics_gf_vertices: VertexArg) -> int:
    """Number of inclusion endpoints declared by a vertex argument.

    Accepts ``None`` (nominal), the vertex list itself, or a plain count (for
    analytic size studies that never build dynamics). Anything other than one
    or two endpoints is rejected: the pairing below is proved for the
    two-endpoint matched-disturbance inclusion only.
    """
    if dynamics_gf_vertices is None:
        return 1
    if isinstance(dynamics_gf_vertices, bool):
        raise ParityReductionError("endpoint count must be an int or a vertex list")
    count = (
        int(dynamics_gf_vertices)
        if isinstance(dynamics_gf_vertices, int)
        else len(dynamics_gf_vertices)
    )
    if count not in (1, 2):
        raise ParityReductionError(
            f"parity reduction is proved for 1 (nominal) or 2 (matched-disturbance "
            f"endpoints) vertices, got {count}"
        )
    return count


# =============================================================================
# Reductions
# =============================================================================

@dataclass(frozen=True)
class NominalParityReduction:
    """Octant reduction for a nominal (single field) problem.

    Valid only when the dynamics are odd and the library is restricted to its
    even part. Then V and W are both even, the associated forms of an
    antipodal octant pair are identical, and one block per pair certifies
    both.
    """

    n_vars: int
    representatives: Tuple[Octant, ...]
    images: Tuple[Octant, ...]

    def __post_init__(self) -> None:
        report = validate_reduction(self)
        if not report.ok:
            raise ParityReductionError(
                f"inconsistent nominal reduction: {report.failures()}"
            )

    # -- structure ---------------------------------------------------------
    @property
    def has_endpoints(self) -> bool:
        return False

    @property
    def n_endpoints(self) -> int:
        return 1

    @property
    def n_octants(self) -> int:
        return 2 ** self.n_vars

    @property
    def n_v_blocks(self) -> int:
        """Gram blocks for V after the reduction (2^(n-1))."""
        return len(self.representatives)

    @property
    def n_w_blocks(self) -> int:
        """Gram blocks for W after the reduction (2^(n-1))."""
        return len(self.representatives)

    # -- maps --------------------------------------------------------------
    def partner(self, signs: Octant) -> Octant:
        """The octant certified for free once ``signs`` is certified."""
        signs = tuple(int(s) for s in signs)
        octant_index(signs)  # validates the entries
        return flip_octant(signs)

    def is_representative(self, signs: Octant) -> bool:
        return tuple(int(s) for s in signs) in set(self.representatives)

    # -- block iteration (contract for the later joint_sos wiring) ---------
    def v_blocks(self) -> List[Octant]:
        return list(self.representatives)

    def w_blocks(self) -> List[BlockLabel]:
        return [(s, None) for s in self.representatives]

    def full_labels(self) -> Tuple[BlockLabel, ...]:
        return tuple((s, None) for s in get_hyperoctant_signs(self.n_vars))

    def representative_labels(self) -> Tuple[BlockLabel, ...]:
        return tuple((s, None) for s in self.representatives)

    def image_labels(self) -> Tuple[BlockLabel, ...]:
        return tuple((s, None) for s in self.images)

    def partner_label(self, label: BlockLabel) -> BlockLabel:
        signs, endpoint = label
        if endpoint is not None:
            raise ParityReductionError(
                "nominal reduction cannot carry an endpoint index"
            )
        return (self.partner(signs), None)

    def describe(self) -> str:
        return (
            f"nominal parity reduction n={self.n_vars}: "
            f"{self.n_v_blocks}/{self.n_octants} V blocks, "
            f"{self.n_w_blocks}/{self.n_octants} W blocks"
        )


@dataclass(frozen=True)
class PairedParityReduction:
    """Paired octant/endpoint reduction for a two-endpoint inclusion.

    The involution is ``(s, σ) -> (-s, -σ)``. It has no fixed point, so the
    2^(n+1) W blocks fall into 2^n orbits of size two and the representative
    set holds both endpoints of every representative octant. V does not see
    the disturbance and still halves to 2^(n-1) blocks.
    """

    n_vars: int
    octant_representatives: Tuple[Octant, ...]
    representatives: Tuple[Tuple[Octant, int], ...]
    images: Tuple[Tuple[Octant, int], ...]

    def __post_init__(self) -> None:
        report = validate_reduction(self)
        if not report.ok:
            raise ParityReductionError(
                f"inconsistent paired reduction: {report.failures()}"
            )

    # -- structure ---------------------------------------------------------
    @property
    def has_endpoints(self) -> bool:
        return True

    @property
    def n_endpoints(self) -> int:
        return 2

    @property
    def n_octants(self) -> int:
        return 2 ** self.n_vars

    @property
    def n_v_blocks(self) -> int:
        """Gram blocks for V after the reduction (2^(n-1))."""
        return len(self.octant_representatives)

    @property
    def n_w_blocks(self) -> int:
        """Gram blocks for W after the reduction (2^n of 2^(n+1))."""
        return len(self.representatives)

    # -- maps --------------------------------------------------------------
    def partner(self, signs: Octant, endpoint: int) -> Tuple[Octant, int]:
        """The (octant, endpoint) block certified for free by ``(signs, endpoint)``.

        Both entries flip. Keeping the endpoint fixed while flipping the
        octant is exactly the unsound naive halving.
        """
        signs = tuple(int(s) for s in signs)
        octant_index(signs)
        flipped = flip_endpoint(endpoint)
        if flipped is None:
            raise ParityReductionError("paired reduction requires an endpoint index")
        return (flip_octant(signs), flipped)

    def is_representative(self, signs: Octant, endpoint: int) -> bool:
        return (tuple(int(s) for s in signs), endpoint) in set(self.representatives)

    # -- block iteration ---------------------------------------------------
    def v_blocks(self) -> List[Octant]:
        return list(self.octant_representatives)

    def w_blocks(self) -> List[BlockLabel]:
        return [(s, e) for s, e in self.representatives]

    def full_labels(self) -> Tuple[BlockLabel, ...]:
        return tuple(
            (s, e)
            for s in get_hyperoctant_signs(self.n_vars)
            for e in (ENDPOINT_PLUS, ENDPOINT_MINUS)
        )

    def representative_labels(self) -> Tuple[BlockLabel, ...]:
        return tuple(self.representatives)

    def image_labels(self) -> Tuple[BlockLabel, ...]:
        return tuple(self.images)

    def partner_label(self, label: BlockLabel) -> BlockLabel:
        signs, endpoint = label
        if endpoint is None:
            raise ParityReductionError(
                "paired reduction needs an endpoint index on every block"
            )
        return self.partner(signs, endpoint)

    def describe(self) -> str:
        return (
            f"paired parity reduction n={self.n_vars}: "
            f"{self.n_v_blocks}/{self.n_octants} V blocks, "
            f"{self.n_w_blocks}/{2 * self.n_octants} W blocks "
            f"(endpoints paired)"
        )


ParityReduction = Union[NominalParityReduction, PairedParityReduction]


def nominal_octant_reduction(
    n_vars: int,
    dynamics_gf_vertices: VertexArg,
) -> NominalParityReduction:
    """Build the nominal 2^(n-1) octant reduction.

    ``dynamics_gf_vertices`` is MANDATORY and must declare a single field
    (``None``, ``1``, or a one-element vertex list). Passing an inclusion with
    two endpoints raises: the antipodal octants of one fixed endpoint carry
    DIFFERENT targets, so nominal halving would drop a constraint that no
    other block covers.
    """
    if n_vars < 1:
        raise ParityReductionError(f"n_vars must be >= 1, got {n_vars}")
    if endpoint_count(dynamics_gf_vertices) != 1:
        raise ParityReductionError(
            "nominal octant halving is unsound on a problem with inclusion "
            "endpoints: a fixed endpoint is not odd, f_sigma(-x) = -f_{-sigma}(x). "
            "Use paired_endpoint_reduction (or build_parity_reduction)."
        )
    reps = _representative_octants(n_vars)
    return NominalParityReduction(
        n_vars=n_vars,
        representatives=reps,
        images=tuple(flip_octant(s) for s in reps),
    )


def paired_endpoint_reduction(
    n_vars: int,
    dynamics_gf_vertices: VertexArg,
) -> PairedParityReduction:
    """Build the paired (octant, endpoint) reduction for a two-endpoint inclusion.

    ``dynamics_gf_vertices`` is MANDATORY and must declare exactly two
    endpoints. The representatives are both endpoints of every representative
    octant; their images are the flipped octant with the OTHER endpoint.
    """
    if n_vars < 1:
        raise ParityReductionError(f"n_vars must be >= 1, got {n_vars}")
    if endpoint_count(dynamics_gf_vertices) != 2:
        raise ParityReductionError(
            "paired endpoint reduction needs exactly 2 inclusion endpoints; "
            "for a nominal problem use nominal_octant_reduction"
        )
    oct_reps = _representative_octants(n_vars)
    reps = tuple(
        (s, e) for s in oct_reps for e in (ENDPOINT_PLUS, ENDPOINT_MINUS)
    )
    return PairedParityReduction(
        n_vars=n_vars,
        octant_representatives=oct_reps,
        representatives=reps,
        images=tuple((flip_octant(s), flip_endpoint(e)) for s, e in reps),
    )


def build_parity_reduction(
    n_vars: int,
    dynamics_gf_vertices: VertexArg,
) -> ParityReduction:
    """Dispatch to the reduction that matches the inclusion structure.

    This is the entry point for callers. The vertex argument is mandatory, so
    the inclusion structure is always stated at the call site and a perturbed
    problem can never fall through to nominal halving.
    """
    if endpoint_count(dynamics_gf_vertices) == 1:
        return nominal_octant_reduction(n_vars, dynamics_gf_vertices)
    return paired_endpoint_reduction(n_vars, dynamics_gf_vertices)


def validate_reduction(reduction: ParityReduction) -> ConsistencyReport:
    """Self-consistency checks of a reduction. Never raises.

    Checked: the partner map is an involution on the full block set, it has no
    fixed point, representatives and images partition the full set, the stored
    images are the partners of the stored representatives in the same order,
    the octant always flips, and (paired case) the endpoint always swaps.
    """
    labels = reduction.full_labels()
    reps = reduction.representative_labels()
    imgs = reduction.image_labels()
    label_set = set(labels)

    checks: Dict[str, bool] = {}

    checks["labels_unique"] = len(label_set) == len(labels)
    checks["involution"] = all(
        reduction.partner_label(reduction.partner_label(lb)) == lb for lb in labels
    )
    checks["partner_in_set"] = all(
        reduction.partner_label(lb) in label_set for lb in labels
    )
    checks["no_fixed_point"] = all(reduction.partner_label(lb) != lb for lb in labels)
    checks["images_match_representatives"] = len(reps) == len(imgs) and all(
        reduction.partner_label(r) == i for r, i in zip(reps, imgs)
    )
    checks["representatives_disjoint_from_images"] = not (set(reps) & set(imgs))
    checks["partition_covers"] = set(reps) | set(imgs) == label_set
    checks["partition_sizes"] = len(reps) + len(imgs) == len(labels)
    checks["octant_flips"] = all(
        reduction.partner_label(lb)[0] == flip_octant(lb[0]) for lb in labels
    )

    expected_reps = 2 ** (reduction.n_vars - 1) * reduction.n_endpoints
    checks["representative_count"] = len(reps) == expected_reps

    if reduction.has_endpoints:
        checks["endpoint_swaps"] = all(
            reduction.partner_label(lb)[1] == flip_endpoint(lb[1]) for lb in labels
        )
        checks["both_endpoints_kept"] = {e for _, e in reps} == {
            ENDPOINT_PLUS,
            ENDPOINT_MINUS,
        }
    else:
        checks["no_endpoint_index"] = all(lb[1] is None for lb in labels)

    ok = all(checks.values())
    return ConsistencyReport(
        ok=ok,
        subject=type(reduction).__name__,
        checks=checks,
        message=(
            f"{len(reps)} representatives cover {len(labels)} blocks"
            if ok
            else "reduction failed its consistency checks"
        ),
    )


# =============================================================================
# One-shot plan (the entry point the CLI wiring will use)
# =============================================================================

@dataclass
class ParityReductionPlan:
    """Everything a solve needs to run reduced, plus the evidence for it.

    ``usable`` is True only when the symbolic prerequisites hold: oddness of
    the base dynamics, and, when endpoints are present, the endpoint-swap
    identity. A plan that is not usable must be discarded and the solve run
    unreduced.
    """

    n_vars: int
    reduction: ParityReduction
    oddness: SymmetryCheck
    endpoint_swap: Optional[SymmetryCheck]
    even_terms: List[Union[TermSpec, GFTerm]]
    odd_terms: List[Union[TermSpec, GFTerm]]
    library_report: Optional[ConsistencyReport]
    reduction_report: ConsistencyReport

    @property
    def usable(self) -> bool:
        if not self.oddness.holds or not self.reduction_report.ok:
            return False
        if self.library_report is not None and not self.library_report.ok:
            return False
        if self.reduction.has_endpoints:
            return self.endpoint_swap is not None and self.endpoint_swap.holds
        return True

    def raise_if_unusable(self) -> None:
        """Raise with the reason if the reduction may not be applied."""
        if self.usable:
            return
        reasons = []
        if not self.oddness.holds:
            reasons.append(self.oddness.message)
        if self.reduction.has_endpoints and self.endpoint_swap is None:
            reasons.append(
                "endpoint-swap identity not verified: pass the vertex dynamics, "
                "not just the endpoint count"
            )
        if self.endpoint_swap is not None and not self.endpoint_swap.holds:
            reasons.append(self.endpoint_swap.message)
        if not self.reduction_report.ok:
            reasons.append(str(self.reduction_report.failures()))
        if self.library_report is not None and not self.library_report.ok:
            reasons.append(str(self.library_report.failures()))
        raise ParityReductionError("; ".join(reasons))


def plan_parity_reduction(
    dynamics_gf: Sequence[Optional[GeneralizedForm]],
    dynamics_gf_vertices: VertexArg,
    terms: Optional[Sequence[Union[TermSpec, GFTerm]]] = None,
    weights_r: Optional[Tuple[int, ...]] = None,
) -> ParityReductionPlan:
    """Verify the prerequisites, filter the library, and build the reduction.

    Args:
        dynamics_gf: The NOMINAL dynamics, one GF per component. Oddness is
            checked on these, never on an endpoint.
        dynamics_gf_vertices: MANDATORY declaration of the inclusion
            structure: ``None`` for nominal, or the two-element vertex list.
            When the list is given, the endpoint-swap identity is verified on
            it symbolically. Passing the bare count ``2`` instead of the
            vertex list yields a plan that is NOT usable, because the identity
            cannot be verified from a count.
        terms: Optional term library to filter. When omitted, the plan only
            carries the block reduction.
        weights_r: Optional arity guard for the library.

    Returns:
        A :class:`ParityReductionPlan`. Call ``raise_if_unusable`` before
        applying it.
    """
    n_vars = len(dynamics_gf)
    oddness = verify_dynamics_odd(dynamics_gf)

    swap: Optional[SymmetryCheck] = None
    if endpoint_count(dynamics_gf_vertices) == 2 and not isinstance(
        dynamics_gf_vertices, int
    ):
        swap = verify_endpoint_swap(dynamics_gf_vertices)  # type: ignore[arg-type]

    reduction = build_parity_reduction(n_vars, dynamics_gf_vertices)

    even: List[Union[TermSpec, GFTerm]] = []
    odd: List[Union[TermSpec, GFTerm]] = []
    library_report: Optional[ConsistencyReport] = None
    if terms is not None:
        even, odd = split_library_by_parity(terms, weights_r)
        library_report = validate_library_split(terms, even, odd)

    return ParityReductionPlan(
        n_vars=n_vars,
        reduction=reduction,
        oddness=oddness,
        endpoint_swap=swap,
        even_terms=even,
        odd_terms=odd,
        library_report=library_report,
        reduction_report=validate_reduction(reduction),
    )
