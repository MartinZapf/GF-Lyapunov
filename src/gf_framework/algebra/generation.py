"""
Generation Module for the GF Framework.

Main entry point for generating valid GF term bases. This module wraps the
existing enumeration functionality with the standardized interface from
interfaces.py.

The generation module is responsible for:
1. Enumerating all valid GF terms satisfying homogeneity constraint Σrᵢρᵢ = m
2. Computing μ exponents for y-space transformation
3. Validating degree compatibility with weights

Status: TEST-ONLY (reclassified by WP P6-06, 2026-09-02). Outside its
re-export in ``algebra/__init__.py`` the only importer is
``tests/test_integration.py``. The live discovery pipeline builds its library
through ``algebra/enumerate.py`` directly, not through this wrapper.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from fractions import Fraction

from .core import GeneralizedForm, GFTerm, lcm_list
from .enumerate import (
    TermSpec,
    enumerate_all_terms,
    enumerate_diagonal_terms,
    enumerate_cross_terms,
    validate_degree_compatibility,
    suggest_compatible_degrees,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gf_framework.interfaces import SystemSpec, TermBasis


def generate_term_basis(
    spec: 'SystemSpec',
    max_cross_terms: int = 50,
    max_denominator: int = 6,
    include_signed: bool = True,
    mu_policy: Optional[str] = None,
    ensure_even_exponents: Optional[bool] = None,
) -> 'TermBasis':
    """
    Main entry point for Generation Module.

    Generates a complete basis of valid GF terms for the given system
    specification. Terms satisfy the homogeneity constraint Σrᵢρᵢ = m.

    Args:
        spec: System specification (weights, degree, dynamics)
        max_cross_terms: Maximum number of cross-terms to enumerate
        max_denominator: Maximum denominator for fractional exponents
        include_signed: Whether to include signed power variants (⌈x⌋^ρ)
        mu_policy: Optional μ policy override ('per_variable' or 'global_lcm')
        ensure_even_exponents: Optional override for SOS-even μ enforcement

    Returns:
        TermBasis containing all valid diagonal and cross terms

    Raises:
        ValueError: If degree is incompatible with weights

    Example:
        >>> spec = SystemSpec(weights_r=(2, 1), degree_m=2)
        >>> basis = generate_term_basis(spec)
        >>> print(f"Generated {basis.n_total} terms")
    """
    # Validate degree compatibility
    if not validate_degree_compatibility(spec.degree_m, spec.weights_r):
        compatible = suggest_compatible_degrees(spec.weights_r, max_degree=20)
        raise ValueError(
            f"Degree {spec.degree_m} is incompatible with weights {spec.weights_r}. "
            f"Compatible degrees: {compatible[:10]}"
        )

    # Enumerate diagonal terms
    diagonal_terms = enumerate_diagonal_terms(spec.degree_m, spec.weights_r)

    # Enumerate cross terms
    cross_terms = enumerate_cross_terms(
        spec.degree_m,
        spec.weights_r,
        max_denominator=max_denominator,
        include_signed=include_signed,
        min_nonzero_vars=2
    )

    # Sort and limit cross terms by complexity
    def term_complexity(t: TermSpec) -> Tuple[int, int, float]:
        nonzero = sum(1 for e in t.exponents if e != 0)
        signed = sum(1 for typ in t.term_types if typ == 'signed')
        total_exp = sum(float(e) for e in t.exponents)
        return (nonzero, signed, total_exp)

    cross_terms = sorted(cross_terms, key=term_complexity)
    if len(cross_terms) > max_cross_terms:
        cross_terms = cross_terms[:max_cross_terms]

    # Compute μ exponents for y-space transformation
    effective_mu_policy = mu_policy if mu_policy is not None else spec.mu_policy
    effective_even = (
        ensure_even_exponents if ensure_even_exponents is not None
        else spec.ensure_even_exponents
    )
    mu_exponents = _compute_mu_exponents(
        diagonal_terms + cross_terms,
        spec.weights_r,
        mu_policy=effective_mu_policy,
        ensure_even_exponents=effective_even,
    )

    from gf_framework.interfaces import TermBasis
    return TermBasis(
        diagonal_terms=diagonal_terms,
        cross_terms=cross_terms,
        mu_exponents=mu_exponents,
        system_spec=spec,
        mu_policy=effective_mu_policy,
        ensure_even_exponents=effective_even,
    )


def _compute_mu_exponents(
    terms: List[TermSpec],
    weights_r: Tuple[int, ...],
    mu_policy: str = 'per_variable',
    ensure_even_exponents: bool = False,
) -> Tuple[int, ...]:
    """
    Compute μ exponents for y-space transformation.

    The transformation yᵢ = ⌈xᵢ⌋^(1/μᵢ).

    Supported policies:
    - per_variable: μᵢ = LCM of denominators in variable i exponents.
      This keeps y-degrees small and is the default.
    - global_lcm: μᵢ = k * rᵢ with global k = LCM of all denominators.
      This is retained for backward compatibility with older runs.

    Both policies ensure integer y-exponents for polynomial verification.

    Args:
        terms: List of term specifications
        weights_r: Homogeneity weights
        mu_policy: μ exponent policy ('per_variable' or 'global_lcm')
        ensure_even_exponents: If True, doubles all μ values

    Returns:
        Tuple of μ exponents
    """
    if mu_policy not in {'per_variable', 'global_lcm'}:
        raise ValueError(
            f"Unsupported mu_policy '{mu_policy}'. Use 'per_variable' or 'global_lcm'."
        )

    if mu_policy == 'per_variable':
        per_var_denoms = [set() for _ in range(len(weights_r))]
        for term in terms:
            for i, exp in enumerate(term.exponents):
                if isinstance(exp, Fraction) and exp != 0:
                    per_var_denoms[i].add(exp.denominator)
        mu = [lcm_list(list(denoms)) if denoms else 1 for denoms in per_var_denoms]
    else:
        denoms = set()
        for term in terms:
            for exp in term.exponents:
                if isinstance(exp, Fraction) and exp.denominator > 1:
                    denoms.add(exp.denominator)
        k = lcm_list(list(denoms)) if denoms else 1
        mu = [k * r for r in weights_r]

    if ensure_even_exponents:
        mu = [2 * m for m in mu]

    return tuple(mu)


def generate_from_weights(
    weights_r: Tuple[int, ...],
    degree_m: Optional[int] = None,
    max_cross_terms: int = 50,
    max_denominator: int = 6,
    include_signed: bool = True,
    mu_policy: str = 'per_variable',
    ensure_even_exponents: bool = False,
) -> TermBasis:
    """
    Convenience function to generate terms from weights only.

    If degree is not specified, uses the default compatible degree
    for the given weights.

    Args:
        weights_r: Homogeneity weights
        degree_m: Optional Lyapunov degree (auto-selected if None)
        max_cross_terms: Maximum cross-terms
        max_denominator: Maximum denominator for exponents
        include_signed: Include signed power variants
        mu_policy: μ exponent policy ('per_variable' or 'global_lcm')
        ensure_even_exponents: If True, doubles μ values for SOS compatibility

    Returns:
        TermBasis with generated terms
    """
    if degree_m is None:
        # Use default degree based on weights
        compatible = suggest_compatible_degrees(weights_r, max_degree=20)
        if not compatible:
            raise ValueError(f"No compatible degrees found for weights {weights_r}")
        # Use smallest compatible degree >= max(weights) + 1
        min_degree = max(weights_r) + 1
        degree_m = next((d for d in compatible if d >= min_degree), compatible[0])

    spec = SystemSpec(
        weights_r=weights_r,
        degree_m=degree_m,
        mu_policy=mu_policy,
        ensure_even_exponents=ensure_even_exponents,
    )
    return generate_term_basis(
        spec,
        max_cross_terms=max_cross_terms,
        max_denominator=max_denominator,
        include_signed=include_signed,
        mu_policy=mu_policy,
        ensure_even_exponents=ensure_even_exponents,
    )


def create_gf_from_basis(
    basis: TermBasis,
    coefficients: List[float]
) -> GeneralizedForm:
    """
    Create a GeneralizedForm from a term basis and coefficients.

    Args:
        basis: Term basis from generation module
        coefficients: Coefficient for each term (diagonal first, then cross)

    Returns:
        Complete GeneralizedForm

    Raises:
        ValueError: If coefficient count doesn't match term count
    """
    all_terms = basis.all_terms
    if len(coefficients) != len(all_terms):
        raise ValueError(
            f"Expected {len(all_terms)} coefficients, got {len(coefficients)}"
        )

    gf_terms = [
        spec.to_gf_term(coef)
        for spec, coef in zip(all_terms, coefficients)
    ]

    return GeneralizedForm(
        degree_m=basis.system_spec.degree_m,
        weights_r=basis.system_spec.weights_r,
        terms=gf_terms
    )


def get_term_y_exponents(
    term: TermSpec,
    mu: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Get y-space exponents for a term.

    After transformation yᵢ = ⌈xᵢ⌋^(1/μᵢ), the exponent of yᵢ
    in the transformed term is μᵢ × ρᵢ.

    Args:
        term: Term specification
        mu: μ exponents from TermBasis

    Returns:
        Tuple of integer y-exponents
    """
    y_exps = []
    for i, (exp, mu_i) in enumerate(zip(term.exponents, mu)):
        y_exp = mu_i * float(exp)
        # Should be integer if degree is compatible
        if abs(y_exp - round(y_exp)) > 1e-10:
            raise ValueError(
                f"Non-integer y-exponent {y_exp} for term {term}"
            )
        y_exps.append(int(round(y_exp)))
    return tuple(y_exps)


def analyze_term_basis(basis: TermBasis) -> dict:
    """
    Analyze a term basis and return statistics.

    Args:
        basis: Term basis to analyze

    Returns:
        Dictionary with analysis results
    """
    spec = basis.system_spec

    # Analyze diagonal terms
    diag_exps = []
    for term in basis.diagonal_terms:
        for i, e in enumerate(term.exponents):
            if e != 0:
                diag_exps.append((i, float(e)))

    # Analyze cross terms
    cross_stats = {
        'n_total': basis.n_cross,
        'n_signed': sum(
            1 for t in basis.cross_terms
            if any(typ == 'signed' for typ in t.term_types)
        ),
        'n_all_abs': sum(
            1 for t in basis.cross_terms
            if all(typ == 'abs' for typ in t.term_types)
        ),
    }

    # Count terms by number of active variables
    active_var_counts = {}
    for term in basis.cross_terms:
        n_active = sum(1 for e in term.exponents if e != 0)
        active_var_counts[n_active] = active_var_counts.get(n_active, 0) + 1

    # Compute y-degree (max sum of y-exponents)
    y_degree = 0
    for term in basis.all_terms:
        y_exp = get_term_y_exponents(term, basis.mu_exponents)
        y_degree = max(y_degree, sum(y_exp))

    return {
        'n_vars': spec.n_vars,
        'degree_m': spec.degree_m,
        'weights_r': spec.weights_r,
        'mu_exponents': basis.mu_exponents,
        'n_diagonal': basis.n_diagonal,
        'n_cross': basis.n_cross,
        'n_total': basis.n_total,
        'diagonal_exponents': diag_exps,
        'cross_stats': cross_stats,
        'active_var_counts': active_var_counts,
        'y_degree': y_degree,
    }


# =============================================================================
# PRESET SYSTEM GENERATORS
# =============================================================================

def generate_fosmc_basis() -> TermBasis:
    """Generate term basis for FOSMC (First-Order SMC)."""
    spec = SystemSpec(weights_r=(1,), degree_m=1)
    return generate_term_basis(spec, max_cross_terms=0)


def generate_sta_basis(degree_m: int = 2) -> TermBasis:
    """
    Generate term basis for STA (Super-Twisting Algorithm).

    Args:
        degree_m: Lyapunov degree (default 2 for Moreno-Osorio)
    """
    spec = SystemSpec(weights_r=(2, 1), degree_m=degree_m)
    return generate_term_basis(spec, max_cross_terms=20)


def generate_pidsmc_basis(degree_m: int = 6) -> TermBasis:
    """
    Generate term basis for PID-SMC.

    Args:
        degree_m: Lyapunov degree (default 6, which is compatible with r=(3,2,1))
                  Note: degree 7 from Pérez-Ventura requires different μ computation
    """
    spec = SystemSpec(weights_r=(3, 2, 1), degree_m=degree_m)
    return generate_term_basis(spec, max_cross_terms=50)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main entry point
    'generate_term_basis',
    # Convenience functions
    'generate_from_weights',
    'create_gf_from_basis',
    'get_term_y_exponents',
    'analyze_term_basis',
    # Preset generators
    'generate_fosmc_basis',
    'generate_sta_basis',
    'generate_pidsmc_basis',
    # Re-export validation functions
    'validate_degree_compatibility',
    'suggest_compatible_degrees',
]
