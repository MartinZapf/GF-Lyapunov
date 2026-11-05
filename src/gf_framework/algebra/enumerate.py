"""
GF Term Enumeration Module.

Enumerates all valid Generalized Form terms satisfying the homogeneity
constraint: Σᵢ rᵢ * ρᵢ = m

For PID-SMC with r = (3, 2, 1) and m = 7, this generates ~30 valid terms.

LIBRARY ORDER (WP P5-13). ``enumerate_cross_terms`` de-duplicates through a
``set`` of :class:`TermSpec`, whose ``__hash__`` mixes in ``term_types`` — a
tuple of ``str``. Python randomizes string hashes per process, so the set's
iteration order, and therefore the library order, the SDP variable order and
the interior point the solver lands on, used to change from run to run. Six of
the twelve nominal E1 systems (twisting, continuous_twisting, levant3d, pidsmc,
fc_cta, levant4d) returned a different designed α on every process (measured
|Δα| up to 5.8e+02, P2-09 contradiction 2 / P2-06 part C).

The fix is a total, hash-free canonical order applied at the single point where
the set is turned back into a list: :data:`LIBRARY_ORDER` ``= "grlex-v1"``,
:func:`canonical_term_key`. **Diagonals are never re-sorted**:
:func:`enumerate_diagonal_terms` emits one pure power per axis in axis order and
that order is load-bearing, because the joint designer pins
``alpha[diag_indices[0]] == coefficient_scale`` (``joint_sos.py``). Sorting the
diagonals moves the pinned monomial and hence the whole design point.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple, Set, Iterator
from itertools import product
from dataclasses import dataclass

from .core import GFTerm, GeneralizedForm, lcm_list

#: Identifier of the canonical library order produced by
#: :func:`canonical_term_key`. Recorded in the design Gram bundle metadata as
#: ``library_order`` so a stored certificate says which ordering built it.
#: Bump the version suffix on any change to the key.
LIBRARY_ORDER: str = "grlex-v1"


@dataclass
class TermSpec:
    """Specification for a GF term (without coefficient)."""
    exponents: Tuple[Fraction, ...]
    term_types: Tuple[str, ...]

    def to_gf_term(self, coefficient: float = 1.0) -> GFTerm:
        """Convert to GFTerm with given coefficient."""
        return GFTerm(
            coefficient=coefficient,
            exponents=self.exponents,
            term_types=self.term_types
        )

    def is_diagonal(self) -> bool:
        """Check if this is a diagonal term (only one nonzero exponent)."""
        nonzero_count = sum(1 for e in self.exponents if e != 0)
        return nonzero_count == 1

    def evaluate(self, x: 'np.ndarray') -> 'np.ndarray':
        """
        Evaluate this term at points x (in physical coordinates).

        Args:
            x: Input points of shape (N, n_vars) or (n_vars,)

        Returns:
            Values of shape (N,) or scalar
        """
        import numpy as np

        if x.ndim == 1:
            x = x.reshape(1, -1)

        result = np.ones(len(x), dtype=np.float64)

        for i, (exp, typ) in enumerate(zip(self.exponents, self.term_types)):
            exp_float = float(exp)

            if exp_float == 0:
                if typ == 'signed':
                    result *= np.sign(x[:, i])
                continue

            if typ == 'abs':
                result *= np.abs(x[:, i]) ** exp_float
            else:  # 'signed'
                result *= np.sign(x[:, i]) * np.abs(x[:, i]) ** exp_float

        return result.squeeze()

    def evaluate_lifted(
        self,
        xi: 'np.ndarray',
        weights_r: Tuple[int, ...]
    ) -> 'np.ndarray':
        """
        Evaluate this term at points in lifted coordinates ξ.

        In lifted coordinates, GF terms become polynomial (with possibly
        signed monomials). This method evaluates the term structure at
        lifted sample points.

        The relationship between x and ξ is:
            xᵢ = ⌈ξᵢ⌋^rᵢ  (unlift)
            ξᵢ = ⌈xᵢ⌋^(1/rᵢ)  (lift)

        For a term with exponent ρᵢ in variable i:
            |xᵢ|^ρᵢ = |ξᵢ|^(rᵢ·ρᵢ)
            ⌈xᵢ⌋^ρᵢ = ⌈ξᵢ⌋^(rᵢ·ρᵢ)

        Args:
            xi: Lifted coordinates, shape (N, n_vars) or (n_vars,)
            weights_r: Homogeneity weights tuple (r₁, ..., rₙ)

        Returns:
            Values of shape (N,) or scalar

        Example:
            >>> spec = TermSpec((Fraction(1), Fraction(1)), ('signed', 'abs'))
            >>> # This is ⌈x₁⌋¹ · |x₂|¹ with STA weights (2, 1)
            >>> xi = np.array([[1.0, 2.0], [-1.0, 1.0]])
            >>> spec.evaluate_lifted(xi, (2, 1))
            >>> # Evaluates ⌈ξ₁⌋² · |ξ₂|¹ = sign(ξ₁)|ξ₁|² · |ξ₂|
        """
        import numpy as np

        if xi.ndim == 1:
            xi = xi.reshape(1, -1)

        n_vars = xi.shape[1]
        if len(self.exponents) != n_vars:
            raise ValueError(f"Term has {len(self.exponents)} vars, xi has {n_vars}")
        if len(weights_r) != n_vars:
            raise ValueError(f"weights_r has {len(weights_r)} elements, need {n_vars}")

        result = np.ones(len(xi), dtype=np.float64)

        for i, (exp, typ, r) in enumerate(zip(self.exponents, self.term_types, weights_r)):
            # In lifted space, exponent becomes r * ρ
            lifted_exp = float(exp) * r

            if lifted_exp == 0:
                if typ == 'signed':
                    result *= np.sign(xi[:, i])
                continue

            if typ == 'abs':
                result *= np.abs(xi[:, i]) ** lifted_exp
            else:  # 'signed'
                result *= np.sign(xi[:, i]) * np.abs(xi[:, i]) ** lifted_exp

        return result.squeeze()

    def __hash__(self):
        return hash((self.exponents, self.term_types))

    def __eq__(self, other):
        if not isinstance(other, TermSpec):
            return False
        return self.exponents == other.exponents and self.term_types == other.term_types


# =============================================================================
# C¹ (restriction R2) REGULARITY — canonical per-term predicate
# =============================================================================
# This is the SINGLE SOURCE OF TRUTH for "is this GF term continuously
# differentiable" (Sanchez-Moreno 2018, Lemma 3.4 / restriction R2). The verifier's
# classify_smoothness delegates to classify_term_regularity, so the discovery basis
# cannot drift from the certification gate. Kept in `algebra` (no dependency on
# `verification`) to avoid an import cycle.

def classify_term_regularity(exp, typ: str) -> str:
    """Regularity contributed by a single (exponent, type) factor of a GF term.

    Per restriction R2: signed ⌈x⌋^ρ needs ρ≥1; abs |x|^ρ needs ρ=0 or ρ>1.
    Returns one of 'C1' | 'lipschitz_kink' | 'sub_lipschitz' | 'discontinuous'.
    """
    # Preserve exact rational comparisons at the C1 boundary rho = 1.
    rho = exp
    if rho == 0:
        return 'discontinuous' if typ == 'signed' else 'C1'  # ⌈x⌋^0=sign(x) vs |x|^0=1
    if rho < 1:
        return 'sub_lipschitz'
    if rho == 1:
        return 'lipschitz_kink' if typ == 'abs' else 'C1'    # |x| kink vs x smooth
    return 'C1'  # rho > 1


def term_is_c1(spec_or_exps, term_types=None) -> bool:
    """True iff a single GF term is C¹ (restriction R2).

    Accepts a TermSpec, or (exponents, term_types). The global condition
    m > maxᵢ rᵢ is captured implicitly: the diagonal term |xᵢ|^{m/rᵢ} is C¹ only
    when m/rᵢ > 1, so a degree m ≤ maxr makes the max-weight diagonal non-C¹.
    """
    if term_types is None:
        exps, types = spec_or_exps.exponents, spec_or_exps.term_types
    else:
        exps, types = spec_or_exps, term_types
    return all(classify_term_regularity(e, t) == 'C1' for e, t in zip(exps, types))


def canonical_term_key(spec: TermSpec) -> Tuple:
    """Total, process-independent sort key for a GF term (``grlex-v1``).

    Graded lexicographic on the exponent tuple, with the term-type pattern as
    the final tie-break:

      1. ``sum(exponents)`` ascending — the *grade*. Cross terms of one weighted
         degree ``m`` do not share an unweighted degree, so this separates most
         of the library before any lexicographic comparison is needed.
      2. the exponent tuple itself, lexicographically ascending, compared as
         exact :class:`~fractions.Fraction` values (never floats, so ``1/3`` and
         ``2/6`` cannot swap).
      3. ``term_types`` lexicographically ascending (``'abs' < 'signed'``), which
         is what makes the key TOTAL: :class:`TermSpec` equality is exactly
         ``(exponents, term_types)``, so no two distinct specs can tie.

    Nothing here reads a ``hash()``, so the order is identical in every process
    regardless of ``PYTHONHASHSEED``. The key is deliberately *independent of
    the weights*: a library is only ever sorted among terms of one ``(m, r)``.

    This key is applied to CROSS terms only. See the module docstring for why
    the diagonals keep their axis order.
    """
    exps = spec.exponents
    if not all(type(e) is Fraction for e in exps):
        exps = tuple(Fraction(e) for e in exps)
    grade = sum(exps, Fraction(0))
    return (grade, exps, tuple(spec.term_types))


def sort_terms_canonical(specs) -> List[TermSpec]:
    """Return ``specs`` in the canonical :data:`LIBRARY_ORDER` order."""
    return sorted(specs, key=canonical_term_key)


def enumerate_diagonal_terms(
    degree_m: int,
    weights_r: Tuple[int, ...],
    c1_only: bool = False,
) -> List[TermSpec]:
    """
    Enumerate diagonal terms: |xᵢ|^(m/rᵢ) for each i.

    These are the "pure" terms that depend on only one variable.

    Args:
        degree_m: Homogeneous degree
        weights_r: Weight tuple
        c1_only: If True, keep only C¹ (restriction R2) diagonal terms. A diagonal
                 is dropped when m/rᵢ ≤ 1 (non-differentiable), which forces the
                 caller to escalate the degree until all variables are spanned.

    Returns:
        List of diagonal TermSpecs
    """
    n = len(weights_r)
    terms = []

    for i in range(n):
        exponents = tuple(
            Fraction(degree_m, weights_r[i]) if j == i else Fraction(0)
            for j in range(n)
        )
        # Diagonal terms use 'abs' type (no sign dependence)
        term_types = tuple('abs' for _ in range(n))
        spec = TermSpec(exponents=exponents, term_types=term_types)
        if c1_only and not term_is_c1(spec):
            continue
        terms.append(spec)

    return terms


def _generate_exponent_candidates(
    degree_m: int,
    weights_r: Tuple[int, ...],
    max_denominator: int = 6
) -> Iterator[Tuple[Fraction, ...]]:
    """
    Generate candidate exponent tuples satisfying Σᵢ rᵢρᵢ = m.

    Uses bounded denominators to keep search space finite.

    Args:
        degree_m: Target weighted degree
        weights_r: Weight tuple
        max_denominator: Maximum denominator for rational exponents

    Yields:
        Valid exponent tuples
    """
    n = len(weights_r)

    # Generate all rational values with bounded denominator
    # that could appear in valid exponents
    def rational_range(max_val: int, max_denom: int) -> List[Fraction]:
        """Generate rationals in [0, max_val] with denominator <= max_denom."""
        rationals = {Fraction(0)}
        for d in range(1, max_denom + 1):
            for n in range(0, max_val * d + 1):
                rationals.add(Fraction(n, d))
        return sorted(rationals)

    # Maximum possible exponent for any variable
    max_exp = degree_m  # Since min weight is at least 1

    candidates = rational_range(max_exp, max_denominator)

    # Recursive search with pruning
    def search(idx: int, remaining: Fraction, current: List[Fraction]):
        if idx == n:
            if remaining == 0:
                yield tuple(current)
            return

        r_i = weights_r[idx]
        # Maximum exponent for this variable given remaining budget
        max_rho = remaining / r_i

        for rho in candidates:
            if rho > max_rho:
                break
            if rho < 0:
                continue

            current.append(rho)
            yield from search(idx + 1, remaining - r_i * rho, current)
            current.pop()

    yield from search(0, Fraction(degree_m), [])


def enumerate_cross_terms(
    degree_m: int,
    weights_r: Tuple[int, ...],
    max_denominator: int = 6,
    include_signed: bool = True,
    min_nonzero_vars: int = 2,
    minimal: bool = True,
    include_abs_variants: bool = True,
    c1_only: bool = False,
) -> List[TermSpec]:
    """
    Enumerate cross-terms involving multiple variables.

    Cross-terms have at least min_nonzero_vars variables with nonzero exponents.

    Args:
        degree_m: Homogeneous degree
        weights_r: Weight tuple
        max_denominator: Maximum denominator for exponents
        include_signed: Whether to include signed power variants
        min_nonzero_vars: Minimum variables with nonzero exponents
        minimal: If True, use minimal enumeration (see include_abs_variants)
        c1_only: If True, keep only C¹ (restriction R2) cross-terms. This drops the
                 non-differentiable structures (e.g. |x|^{1/2} or |x| factors) that
                 the sound Lemma-5 gate would reject.
        include_abs_variants: When minimal=True, also include all-'abs' variant
                              for each cross-term. This gives 2 terms per exponent
                              tuple instead of 2^n, covering both:
                              - all-signed: ⌈x₁⌋^a·⌈x₂⌋^b (can be +/-)
                              - all-abs: |x₁|^a·|x₂|^b (always positive)
                              This is needed for Pérez-Ventura style solutions
                              that use mixed type patterns.

    Returns:
        List of cross-term TermSpecs
    """
    n = len(weights_r)
    terms: Set[TermSpec] = set()

    for exponents in _generate_exponent_candidates(degree_m, weights_r, max_denominator):
        # Count nonzero exponents
        nonzero_count = sum(1 for e in exponents if e != 0)
        if nonzero_count < min_nonzero_vars:
            continue

        # For cross-terms, generate different type combinations
        if minimal:
            # All-'signed' variant: ⌈x₁⌋^a · ⌈x₂⌋^b = sign(x₁)sign(x₂)|x₁|^a|x₂|^b
            term_types_signed = tuple('signed' for _ in range(n))
            terms.add(TermSpec(exponents=exponents, term_types=term_types_signed))

            if include_abs_variants:
                # All-'abs' variant: |x₁|^a · |x₂|^b (always positive)
                term_types_abs = tuple('abs' for _ in range(n))
                terms.add(TermSpec(exponents=exponents, term_types=term_types_abs))

                # Also add "smart" variants: signed only for variables with
                # fractional/odd exponents (sign matters), abs otherwise
                smart_types = []
                for i, exp in enumerate(exponents):
                    exp_float = float(exp)
                    if exp_float == 0:
                        smart_types.append('abs')  # Zero exponent: type doesn't matter
                    elif exp.denominator == 1 and int(exp) % 2 == 0:
                        smart_types.append('abs')  # Even integer: |x|^2k = x^2k
                    else:
                        smart_types.append('signed')  # Fractional or odd: sign matters
                smart_types_tuple = tuple(smart_types)
                if smart_types_tuple not in [term_types_signed, term_types_abs]:
                    terms.add(TermSpec(exponents=exponents, term_types=smart_types_tuple))

                # Add all 2-signed variants (patterns with exactly 2 'signed' types)
                # These are common in PID-SMC Lyapunov functions
                from itertools import combinations
                for signed_indices in combinations(range(n), 2):
                    two_signed_types = list('abs' for _ in range(n))
                    for idx in signed_indices:
                        two_signed_types[idx] = 'signed'
                    two_signed_tuple = tuple(two_signed_types)
                    terms.add(TermSpec(exponents=exponents, term_types=two_signed_tuple))

        elif include_signed:
            # Original behavior: all 2^n combinations
            for type_combo in product(['abs', 'signed'], repeat=n):
                term_types = tuple(type_combo)
                terms.add(TermSpec(exponents=exponents, term_types=term_types))
        else:
            # Only absolute values
            term_types = tuple('abs' for _ in range(n))
            terms.add(TermSpec(exponents=exponents, term_types=term_types))

    # P5-13: `terms` is a set keyed on (exponents, term_types); term_types is a
    # tuple of str, so its iteration order is hash-seed dependent. Sort into the
    # canonical LIBRARY_ORDER before anything downstream indexes the list.
    # The C¹ filter is order-independent, so it runs FIRST: at n=5 with the
    # μ-derived cap the raw set is ~2.8e6 specs and the C¹ slice 339, and
    # sorting the raw set instead of the slice costs about 67 s per call.
    result = [t for t in terms if term_is_c1(t)] if c1_only else terms
    return sort_terms_canonical(result)


def enumerate_all_terms(
    degree_m: int,
    weights_r: Tuple[int, ...],
    max_denominator: int = 6,
    include_signed: bool = True,
    max_cross_terms: int = None,
    minimal: bool = True,
    max_terms: int = None,
    c1_only: bool = False,
) -> Tuple[List[TermSpec], List[TermSpec]]:
    """
    Enumerate all valid GF terms (diagonal + cross).

    Note on term counts: The GF framework (Sanchez & Moreno 2018, Definition 3)
    only requires that the number of terms N be finite (N ∈ ℤ₊). There is NO
    theoretical upper bound on N in the paper.

    The n² pattern (n diagonal + n(n-1) cross-terms) is an empirical observation
    from the D₁ Lyapunov structure (Pérez-Ventura et al.), which represents one
    valid solution - not a theoretical maximum. The framework may discover
    solutions with more or fewer terms.

    Args:
        degree_m: Homogeneous degree
        weights_r: Weight tuple
        max_denominator: Maximum denominator for exponents
        include_signed: Whether to include signed power variants
        max_cross_terms: Maximum cross-terms to return. Default is n(n-1),
                         which matches the D₁ pattern but is not a theoretical bound.
        minimal: If True, use minimal term enumeration (one cross-term per
                 exponent tuple instead of 2^n combinations)
        max_terms: Deprecated alias for max_cross_terms (kept for compatibility)

    Returns:
        (diagonal_terms, cross_terms)
    """
    n = len(weights_r)

    # Backward-compatibility alias used by older NN modules.
    if max_terms is not None:
        if max_cross_terms is not None and max_cross_terms != max_terms:
            raise ValueError(
                f"Conflicting limits: max_cross_terms={max_cross_terms}, max_terms={max_terms}"
            )
        max_cross_terms = max_terms

    # Default: n(n-1) cross-terms based on D₁ Lyapunov pattern (Pérez-Ventura)
    # Note: This is an empirical default, not a theoretical bound from GF theory
    if max_cross_terms is None:
        max_cross_terms = n * (n - 1)

    diag = enumerate_diagonal_terms(degree_m, weights_r, c1_only=c1_only)
    cross = enumerate_cross_terms(
        degree_m, weights_r, max_denominator,
        include_signed=include_signed,
        minimal=minimal,
        c1_only=c1_only,
    )

    # Sort cross terms by complexity (fewer nonzero exponents first)
    def term_complexity(spec: TermSpec) -> Tuple[int, int, float]:
        nonzero = sum(1 for e in spec.exponents if e != 0)
        signed = sum(1 for t in spec.term_types if t == 'signed')
        total_exp = sum(float(e) for e in spec.exponents)
        return (nonzero, signed, total_exp)

    # `cross` arrives in LIBRARY_ORDER and `sorted` is stable, so complexity
    # ties keep the canonical order and the truncation below is reproducible.
    cross = sorted(cross, key=term_complexity)

    if len(cross) > max_cross_terms:
        cross = cross[:max_cross_terms]

    return diag, cross


def involves_variable_pair(spec: TermSpec, i: int, j: int) -> bool:
    """Check if a term involves exactly variables i and j (nonzero exponents)."""
    nonzero_indices = [k for k, e in enumerate(spec.exponents) if e != 0]
    return set(nonzero_indices) == {i, j}


def prioritize_cross_terms(
    cross_terms: List[TermSpec],
    weights_r: Tuple[int, ...],
    terms_per_pair: int = 2,
    max_denom_per_var: Tuple[int, ...] = None,
) -> List[TermSpec]:
    """
    Prioritize cross-terms: select best terms per variable pair.

    Based on GF framework analysis (Sanchez-Moreno 2018), the D₁ Lyapunov
    has exactly 2 cross-terms per variable pair. This function selects
    the most promising cross-terms for each pair.

    Selection criteria per pair (i,j):
    1. Filter terms that would cause large μ values (per-variable denominator constraint)
    2. One term with integer exponents (if available)
    3. One term with fractional exponents (for diversity)

    The per-variable denominator constraint matches the D₁ pattern:
    - Higher-weight variables (like x₁ with r=3) can have larger denominators
    - Lower-weight variables (like x₃ with r=1) should have integer exponents
    Default: max_denom[i] = weights_r[i] (e.g., (3, 2, 1) for PID-SMC)

    Args:
        cross_terms: List of candidate cross-terms
        weights_r: Homogeneity weights tuple
        terms_per_pair: Number of terms to select per variable pair (default: 2)
        max_denom_per_var: Per-variable max denominator. Default: weights_r

    Returns:
        Prioritized list of cross-terms, at most n(n-1) terms for n variables
    """
    from math import lcm
    from functools import reduce

    n = len(weights_r)
    selected = []

    # Default: max denominator per variable matches the weight (D₁ pattern)
    # For (3, 2, 1): x₁ allows denom 3, x₂ allows denom 2, x₃ only integers
    if max_denom_per_var is None:
        max_denom_per_var = weights_r

    for i in range(n):
        for j in range(i + 1, n):
            # Find cross-terms involving exactly variables i and j
            pair_terms = [t for t in cross_terms if involves_variable_pair(t, i, j)]

            if not pair_terms:
                continue

            # Filter out terms with large denominators that cause μ explosion
            # Each variable has its own max denominator (matching D₁ structure)
            def denominator_ok(spec: TermSpec) -> bool:
                """Check if term's denominators respect per-variable limits."""
                for k, exp in enumerate(spec.exponents):
                    if exp != 0 and exp.denominator > max_denom_per_var[k]:
                        return False
                return True

            pair_terms = [t for t in pair_terms if denominator_ok(t)]

            if not pair_terms:
                continue

            # D₁ pattern: one term where variable i dominates + one where j dominates
            # This gives diversity in the Lie derivative structure
            def dominant_var(spec: TermSpec) -> int:
                """Return which variable (i or j) has larger exponent."""
                exp_i = float(spec.exponents[i])
                exp_j = float(spec.exponents[j])
                if exp_i > exp_j:
                    return i
                elif exp_j > exp_i:
                    return j
                else:
                    return -1  # Equal/balanced

            i_dominant = [t for t in pair_terms if dominant_var(t) == i]
            j_dominant = [t for t in pair_terms if dominant_var(t) == j]
            balanced = [t for t in pair_terms if dominant_var(t) == -1]

            # Score by total exponent magnitude (prefer larger for stability)
            def magnitude_score(spec: TermSpec) -> float:
                exp_i = float(spec.exponents[i])
                exp_j = float(spec.exponents[j])
                return -(exp_i + exp_j)  # Negative = prefer larger

            i_dominant.sort(key=magnitude_score)
            j_dominant.sort(key=magnitude_score)
            balanced.sort(key=magnitude_score)

            # Select: one term from each dominance group for diversity (like D₁)
            added_for_pair = 0
            # First: term where variable i dominates
            for t in i_dominant:
                if added_for_pair >= terms_per_pair:
                    break
                if t not in selected:
                    selected.append(t)
                    added_for_pair += 1
                    break
            # Second: term where variable j dominates
            for t in j_dominant:
                if added_for_pair >= terms_per_pair:
                    break
                if t not in selected:
                    selected.append(t)
                    added_for_pair += 1
                    break
            # Fill remaining slots with balanced or remaining terms
            for t in balanced + i_dominant + j_dominant:
                if added_for_pair >= terms_per_pair:
                    break
                if t not in selected:
                    selected.append(t)
                    added_for_pair += 1

    return selected


def enumerate_cross_term_pool(
    degree_m: int,
    weights_r: Tuple[int, ...],
    max_per_pair: int = 4,
    max_denominator: int = 3,
) -> List[TermSpec]:
    """
    Enumerate a diverse pool of cross-terms for SOS-guided exploration.

    Unlike prioritize_cross_terms() which tries to guess the "best" terms,
    this function creates a larger pool that the SOS solver can explore.
    The pool is designed to stay small enough for tractable SOS (~12-20 terms total).

    Pool structure per variable pair (i, j):
    - Up to max_per_pair terms selected for diversity
    - Diversity criteria:
      * Integer vs fractional exponents
      * Variable i dominant vs j dominant
      * Different term_type patterns (abs vs signed)

    Args:
        degree_m: Homogeneous degree
        weights_r: Weight tuple
        max_per_pair: Maximum terms per variable pair (default: 4)
        max_denominator: Maximum denominator for exponents (default: 3)

    Returns:
        List of TermSpecs forming the cross-term pool.
        For n=3 with max_per_pair=4: up to 12 cross-terms.
    """
    n = len(weights_r)
    pool = []

    # Get all cross-terms with the given max_denominator
    # Using minimal=False to include all exponent patterns including fractional
    all_cross = enumerate_cross_terms(
        degree_m=degree_m,
        weights_r=weights_r,
        max_denominator=max_denominator,
        include_signed=True,
        minimal=False,
        include_abs_variants=True,
    )

    for i in range(n):
        for j in range(i + 1, n):
            # Get terms involving exactly this pair
            pair_terms = [t for t in all_cross if involves_variable_pair(t, i, j)]

            if not pair_terms:
                continue

            # Group by exponent tuple (ignoring term_types)
            exponent_groups = {}
            for t in pair_terms:
                exp_key = t.exponents
                if exp_key not in exponent_groups:
                    exponent_groups[exp_key] = []
                exponent_groups[exp_key].append(t)

            # Score exponent patterns by preference
            # For universal discovery: don't penalize fractional exponents (needed for D1)
            # Only penalize denominators that cause μ explosion (exceed weights)
            def exponent_score(exponents: tuple) -> tuple:
                exp_i = exponents[i]
                exp_j = exponents[j]
                # Only penalize denominators that exceed weights (causes μ explosion)
                denom_penalty = 0
                if exp_i != 0 and exp_i.denominator > weights_r[i]:
                    denom_penalty += 1
                if exp_j != 0 and exp_j.denominator > weights_r[j]:
                    denom_penalty += 1
                total_exp = float(exp_i) + float(exp_j)
                # Note: max_denom removed from scoring to include fractional patterns
                return (denom_penalty, -total_exp)  # Prefer larger totals, avoid μ explosion

            # Sort exponent groups by score
            sorted_exponents = sorted(exponent_groups.keys(), key=exponent_score)

            # Select up to max_per_pair terms with diverse exponents and types
            # For universal discovery, we need ALL type variants for top exponents
            # to cover unknown patterns like D1's ('signed', 'signed', 'abs')
            selected_for_pair = []
            seen_exponents = set()

            # Include all 8 type variants for several exponent patterns
            # This ensures we cover patterns like ('signed', 'signed', 'abs')
            # D1 uses exponents at ranks 2 and 4, so we need at least top 5
            # With max_per_pair=20, we can fit 2-3 full patterns + partial
            n_top_exponents = max(5, max_per_pair // 4)  # ~5 patterns with all variants

            for exp_key in sorted_exponents:
                if len(selected_for_pair) >= max_per_pair:
                    break

                terms_with_exp = exponent_groups[exp_key]

                # For top N exponent patterns, include ALL type variants
                # This ensures we don't miss patterns like ('signed', 'signed', 'abs')
                if len(seen_exponents) < n_top_exponents:
                    for t in terms_with_exp:
                        if len(selected_for_pair) >= max_per_pair:
                            break
                        selected_for_pair.append(t)
                    seen_exponents.add(exp_key)
                    continue

                # For remaining patterns, select one abs and one signed variant
                abs_term = None
                signed_term = None
                for t in terms_with_exp:
                    is_all_abs = all(tt == 'abs' for tt in t.term_types)
                    if is_all_abs and abs_term is None:
                        abs_term = t
                    elif not is_all_abs and signed_term is None:
                        signed_term = t

                # Add abs variant first (if not already have this exponent pattern)
                if abs_term and exp_key not in seen_exponents:
                    selected_for_pair.append(abs_term)
                    seen_exponents.add(exp_key)

                # Add signed variant if we have room and it's different
                if (signed_term and len(selected_for_pair) < max_per_pair
                    and signed_term not in selected_for_pair):
                    selected_for_pair.append(signed_term)

            pool.extend(selected_for_pair)

    return pool


def create_gf_from_specs(
    degree_m: int,
    weights_r: Tuple[int, ...],
    term_specs: List[TermSpec],
    coefficients: List[float]
) -> GeneralizedForm:
    """
    Create a GeneralizedForm from term specifications and coefficients.

    Args:
        degree_m: Homogeneous degree
        weights_r: Weight tuple
        term_specs: List of TermSpecs defining the structure
        coefficients: Coefficient for each term

    Returns:
        Complete GeneralizedForm
    """
    assert len(term_specs) == len(coefficients), "Spec/coefficient count mismatch"

    terms = [spec.to_gf_term(coef) for spec, coef in zip(term_specs, coefficients)]
    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


def validate_degree_compatibility(degree_m: int, weights_r: Tuple[int, ...]) -> bool:
    """
    Check that degree_m produces valid integer exponents in associated forms.

    UPDATED FORMULA (per Sanchez & Moreno 2018, Lemma 6):

    The μᵢ exponents are computed as μᵢ = k × rᵢ where k = LCM of all
    exponent denominators in the GF. This ensures ALL y-exponents are integers:

        y-exponent = μᵢ × (ρᵢ) = k × rᵢ × ρᵢ

    For diagonal term |xᵢ|^(m/rᵢ), ρᵢ = m/rᵢ, so:
        y-exponent = k × rᵢ × (m/rᵢ) = k × m  (always integer!)

    This means ALL degrees are compatible with the correct μ formula.
    The function now always returns True but is kept for API compatibility.

    NOTE: The OLD (incorrect) formula was μᵢ = LCM(r)/rᵢ, which produced
    fractional y-exponents like 14/3 for PID-SMC at degree 7.

    Args:
        degree_m: Candidate homogeneous degree
        weights_r: Weight tuple

    Returns:
        True (always, with correct μ formula)
    """
    # With the correct formula μᵢ = k × rᵢ (see algebra.py:214-236),
    # all degrees produce integer y-exponents for diagonal terms.
    #
    # For any diagonal term |xᵢ|^(m/rᵢ):
    #   - exponent ρᵢ = m/rᵢ has denominator rᵢ
    #   - k = LCM(all denominators) >= rᵢ
    #   - μᵢ = k × rᵢ
    #   - y-exponent = μᵢ × ρᵢ = k × rᵢ × (m/rᵢ) = k × m (integer!)
    #
    # Previously this function used the WRONG formula μᵢ = L // rᵢ which
    # gave fractional y-exponents for some degree/weight combinations.

    # Sanity check: degree should be positive
    if degree_m < 1:
        return False

    # All positive degrees are compatible with the correct μ formula
    return True


def suggest_compatible_degrees(
    weights_r: Tuple[int, ...],
    min_degree: int = None,
    max_degree: int = 20
) -> List[int]:
    """
    Find degrees compatible with the given weights.

    Args:
        weights_r: Weight tuple
        min_degree: Minimum degree (default: max(weights) + 1)
        max_degree: Maximum degree to check

    Returns:
        List of compatible degrees
    """
    if min_degree is None:
        min_degree = max(weights_r) + 1

    compatible = []
    for m in range(min_degree, max_degree + 1):
        if validate_degree_compatibility(m, weights_r):
            compatible.append(m)

    return compatible


# Pre-defined configurations for common SMC systems
PID_SMC_CONFIG = {
    'weights': (3, 2, 1),
    'suggested_degrees': [5, 6, 7, 8],
    'default_degree': 7,  # Pérez-Ventura result
}

STA_CONFIG = {
    'weights': (2, 1),
    'suggested_degrees': [2, 3, 4],
    'default_degree': 2,  # Moreno-Osorio quadratic
}

FOSMC_CONFIG = {
    'weights': (1,),
    'suggested_degrees': [1],
    'default_degree': 1,  # V = |s|
}
