"""GF algebra: generalized forms, term enumeration, associated forms, Lie derivatives."""

from __future__ import annotations

from .core import (
    GFTerm,
    CompositeGFTerm,
    GeneralizedForm,
    lcm,
    lcm_list,
    sgnpow_numpy,
    sgnpow_sympy,
    create_diagonal_gf,
    build_weighted_norm_gf,
    build_clearing_monomial_gf,
    partitions_summing_to,
    multinomial_coefficient,
    lift_coordinates,
    unlift_coordinates,
)

from .enumerate import (
    LIBRARY_ORDER,
    TermSpec,
    canonical_term_key,
    sort_terms_canonical,
    enumerate_diagonal_terms,
    enumerate_cross_terms,
    enumerate_all_terms,
    enumerate_cross_term_pool,
    prioritize_cross_terms,
    create_gf_from_specs,
    validate_degree_compatibility,
    suggest_compatible_degrees,
)

from .associated import (
    gf_to_associated_forms,
    transform_to_y_space,
    get_hyperoctant_signs,
    inverse_transform_point,
    forward_transform_point,
    get_y_degree,
    extract_polynomial_coefficients,
    AssociatedFormsCache,
)

# Test-only wrapper over `enumerate.py` (P6-06); no live pipeline importer.
from .generation import (
    generate_term_basis,
)

from .symmetry import (
    PARITY_EVEN,
    PARITY_ODD,
    ENDPOINT_PLUS,
    ENDPOINT_MINUS,
    ParityReductionError,
    SymmetryCheck,
    ConsistencyReport,
    NominalParityReduction,
    PairedParityReduction,
    ParityReductionPlan,
    term_parity,
    is_even_term,
    gf_parity,
    verify_dynamics_odd,
    verify_endpoint_swap,
    split_library_by_parity,
    filter_even_parity_terms,
    validate_library_split,
    flip_octant,
    flip_endpoint,
    octant_index,
    octant_from_index,
    endpoint_count,
    nominal_octant_reduction,
    paired_endpoint_reduction,
    build_parity_reduction,
    validate_reduction,
    plan_parity_reduction,
)

# WP P5-04: the exact symbolic even-support guard, the premise check of the
# y-parity Gram block decomposition in ``verification/yparity.py``.
from .support_guard import (
    GuardReport,
    GuardUnreadable,
    OddLift,
    OddLiftRefusal,
    check_even_support,
    gf_exponent_support,
    guard_from_spec,
)

from .lie_derivative import (
    compute_gradient_symbolic,
    compute_lie_derivative,
    compute_lie_derivative_gf,
    negate_gf,
    multiply_gf_terms,
    multiply_gf_by_gf,
    collect_terms,
    LieDerivativeEvaluator,
    FilippovLieDerivative,
    evaluate_gf_with_composites,
)

__all__ = [
    # Core types
    "GFTerm",
    "CompositeGFTerm",
    "GeneralizedForm",
    # Enumeration
    "LIBRARY_ORDER",
    "TermSpec",
    "canonical_term_key",
    "sort_terms_canonical",
    "enumerate_diagonal_terms",
    "enumerate_cross_terms",
    "enumerate_all_terms",
    "enumerate_cross_term_pool",
    "prioritize_cross_terms",
    "create_gf_from_specs",
    "validate_degree_compatibility",
    "suggest_compatible_degrees",
    # Associated forms
    "gf_to_associated_forms",
    "transform_to_y_space",
    "get_hyperoctant_signs",
    "AssociatedFormsCache",
    # Lie derivative
    "compute_gradient_symbolic",
    "compute_lie_derivative",
    "compute_lie_derivative_gf",
    "negate_gf",
    "multiply_gf_terms",
    "multiply_gf_by_gf",
    "LieDerivativeEvaluator",
    "FilippovLieDerivative",
    # Generation
    "generate_term_basis",
    # Even-support guard (WP P5-04)
    "GuardReport",
    "GuardUnreadable",
    "OddLift",
    "OddLiftRefusal",
    "check_even_support",
    "gf_exponent_support",
    "guard_from_spec",
    # Parity / symmetry reduction (campaign item C1)
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
    # Builders
    "create_diagonal_gf",
    "build_weighted_norm_gf",
    "build_clearing_monomial_gf",
    # Utilities
    "lcm",
    "lcm_list",
    "sgnpow_numpy",
    "sgnpow_sympy",
    "lift_coordinates",
    "unlift_coordinates",
]
