"""
Lie Derivative Module for Generalized Forms.

Computes dV/dt = ∇V · f symbolically within the GF class.

For homogeneous systems, the Lie derivative of a GF is also a GF
(with different degree: m + κ where κ is the system degree).

This module handles:
1. Symbolic computation of ∇V for GF
2. Lie derivative dV/dt = Σᵢ (∂V/∂xᵢ) · fᵢ
3. Special handling for discontinuous systems (Filippov)
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Callable, Dict
from fractions import Fraction
from dataclasses import dataclass
import sympy as sp
import numpy as np

from .core import GeneralizedForm, GFTerm, CompositeGFTerm, sgnpow_sympy


def gf_partial_derivative(
    term: GFTerm,
    var_idx: int,
    symbols: List[sp.Symbol]
) -> sp.Expr:
    """
    Compute partial derivative of a GF term with respect to xᵢ.

    For |xᵢ|^ρ: ∂/∂xᵢ = ρ · |xᵢ|^(ρ-1) · sign(xᵢ) = ρ · ⌈xᵢ⌋^(ρ-1)
    For ⌈xᵢ⌋^ρ: ∂/∂xᵢ = ρ · |xᵢ|^(ρ-1)  (for ρ > 0)

    Args:
        term: GF term to differentiate
        var_idx: Index of variable to differentiate w.r.t.
        symbols: SymPy symbols

    Returns:
        SymPy expression for partial derivative
    """
    n = term.n_vars
    exp_i = term.exponents[var_idx]
    type_i = term.term_types[var_idx]

    if exp_i == 0:
        # Term doesn't depend on xᵢ
        return sp.Integer(0)

    # Build the derivative term
    result = sp.Float(term.coefficient) * float(exp_i)  # ρᵢ · coef

    for j in range(n):
        exp_j = term.exponents[j]
        type_j = term.term_types[j]

        if j == var_idx:
            # Derivative of this factor
            new_exp = float(exp_j - 1)

            if type_j == 'abs':
                # d/dx |x|^ρ = ρ|x|^(ρ-1) sign(x)
                # Already multiplied by ρ above
                if new_exp != 0:
                    result *= sp.Abs(symbols[j]) ** new_exp
                result *= sp.sign(symbols[j])
            else:
                # d/dx ⌈x⌋^ρ = d/dx (sign(x)|x|^ρ) = ρ|x|^(ρ-1)
                # (treating sign as constant except at x=0)
                if new_exp != 0:
                    result *= sp.Abs(symbols[j]) ** new_exp
        else:
            # Other factors unchanged
            if exp_j == 0:
                if type_j == 'signed':
                    result *= sp.sign(symbols[j])
                continue

            if type_j == 'abs':
                result *= sp.Abs(symbols[j]) ** float(exp_j)
            else:
                result *= sgnpow_sympy(symbols[j], float(exp_j))

    return result


def compute_gradient_symbolic(
    gf: GeneralizedForm,
    symbols: Optional[List[sp.Symbol]] = None
) -> List[sp.Expr]:
    """
    Compute symbolic gradient ∇V for a GF.

    Args:
        gf: Generalized Form
        symbols: SymPy symbols (created if not provided)

    Returns:
        List of partial derivatives [∂V/∂x₁, ..., ∂V/∂xₙ]
    """
    if symbols is None:
        symbols = gf.get_symbols()

    gradient = []
    for i in range(gf.n_vars):
        partial = sp.Integer(0)
        for term in gf.terms:
            partial += gf_partial_derivative(term, i, symbols)
        gradient.append(sp.simplify(partial))

    return gradient


def compute_lie_derivative(
    gf: GeneralizedForm,
    dynamics: List[sp.Expr],
    symbols: Optional[List[sp.Symbol]] = None
) -> sp.Expr:
    """
    Compute Lie derivative dV/dt = ∇V · f.

    Args:
        gf: Lyapunov candidate (Generalized Form)
        dynamics: List of expressions [f₁, ..., fₙ] where ẋᵢ = fᵢ
        symbols: SymPy symbols

    Returns:
        SymPy expression for dV/dt
    """
    if symbols is None:
        symbols = gf.get_symbols()

    gradient = compute_gradient_symbolic(gf, symbols)

    lie_deriv = sp.Integer(0)
    for i in range(gf.n_vars):
        lie_deriv += gradient[i] * dynamics[i]

    return sp.simplify(lie_deriv)


def multiply_gf_terms(term1: GFTerm, term2: GFTerm) -> GFTerm:
    """
    Multiply two GF terms to produce a new GF term.

    For term types:
    - abs * abs = abs (|x|^a * |x|^b = |x|^(a+b))
    - signed * signed = abs (⌈x⌋^a * ⌈x⌋^b = |x|^(a+b), since sign(x)^2 = 1)
    - abs * signed = signed (|x|^a * ⌈x⌋^b = ⌈x⌋^(a+b))
    - signed * abs = signed

    Args:
        term1: First GF term
        term2: Second GF term

    Returns:
        Product term
    """
    assert term1.n_vars == term2.n_vars, "Term dimensions must match"

    n = term1.n_vars
    new_coef = term1.coefficient * term2.coefficient
    new_exps = []
    new_types = []

    for i in range(n):
        exp1, exp2 = term1.exponents[i], term2.exponents[i]
        type1, type2 = term1.term_types[i], term2.term_types[i]

        new_exp = exp1 + exp2

        # Determine resulting type
        if type1 == 'abs' and type2 == 'abs':
            new_type = 'abs'
        elif type1 == 'signed' and type2 == 'signed':
            # sign(x)^2 = 1, so product is absolute
            new_type = 'abs'
        else:
            # abs * signed or signed * abs -> signed
            new_type = 'signed'

        new_exps.append(new_exp)
        new_types.append(new_type)

    return GFTerm(
        coefficient=new_coef,
        exponents=tuple(new_exps),
        term_types=tuple(new_types)
    )


def term_signature(term: GFTerm) -> Tuple[Tuple[Fraction, ...], Tuple[str, ...]]:
    """
    Get the signature of a term (exponents + types) for grouping.

    Terms with the same signature can be combined by adding coefficients.
    """
    return (term.exponents, term.term_types)


def collect_terms(terms: List[GFTerm]) -> List[GFTerm]:
    """
    Collect and combine terms with the same signature.

    Terms with identical exponents and types are merged by summing coefficients.
    Terms with zero coefficient are removed.

    Args:
        terms: List of GF terms (possibly with duplicates)

    Returns:
        Simplified list of GF terms
    """
    collected: Dict[Tuple, float] = {}

    for term in terms:
        sig = term_signature(term)
        if sig in collected:
            collected[sig] += term.coefficient
        else:
            collected[sig] = term.coefficient

    result = []
    for (exps, types), coef in collected.items():
        if abs(coef) > 1e-15:  # Skip near-zero terms
            result.append(GFTerm(
                coefficient=coef,
                exponents=exps,
                term_types=types
            ))

    return result


def gf_partial_derivative_term(
    term: GFTerm,
    var_idx: int
) -> Optional[GFTerm]:
    """
    Compute partial derivative of a GF term as another GF term.

    For |xᵢ|^ρ: ∂/∂xᵢ = ρ · |xᵢ|^(ρ-1) · sign(xᵢ) = ρ · ⌈xᵢ⌋^(ρ-1)
    For ⌈xᵢ⌋^ρ: ∂/∂xᵢ = ρ · |xᵢ|^(ρ-1)  (for ρ > 0)

    Args:
        term: GF term to differentiate
        var_idx: Index of variable to differentiate w.r.t.

    Returns:
        GFTerm for the derivative, or None if derivative is zero
    """
    exp_i = term.exponents[var_idx]

    if exp_i == 0:
        # Term doesn't depend on xᵢ (except possibly for sign)
        return None

    n = term.n_vars
    new_coef = term.coefficient * float(exp_i)
    new_exps = list(term.exponents)
    new_types = list(term.term_types)

    # Modify the i-th variable's exponent and type
    type_i = term.term_types[var_idx]

    if type_i == 'abs':
        # d/dx |x|^ρ = ρ|x|^(ρ-1) sign(x) = ρ ⌈x⌋^(ρ-1)
        new_exps[var_idx] = exp_i - 1
        new_types[var_idx] = 'signed'  # sign(x) appears
    else:
        # d/dx ⌈x⌋^ρ = d/dx (sign(x)|x|^ρ) = ρ|x|^(ρ-1)
        # (treating sign as constant except at x=0)
        new_exps[var_idx] = exp_i - 1
        new_types[var_idx] = 'abs'

    return GFTerm(
        coefficient=new_coef,
        exponents=tuple(new_exps),
        term_types=tuple(new_types)
    )


def multiply_gf_by_term(gf: GeneralizedForm, term: GFTerm) -> List[GFTerm]:
    """
    Multiply a GF by a single term, returning list of product terms.

    Args:
        gf: Generalized Form
        term: Term to multiply by

    Returns:
        List of product terms
    """
    result = []
    for gf_term in gf.terms:
        product = multiply_gf_terms(gf_term, term)
        result.append(product)
    return result


@dataclass
class ProductCompositeTerm:
    """
    Product of a GFTerm and a CompositeGFTerm.

    Represents: gf_term * composite_term
    This is used when computing Lie derivatives with composite dynamics.

    For evaluation, we compute: gf_term(x) * composite_term(x)
    """
    gf_term: GFTerm
    composite_term: CompositeGFTerm

    @property
    def n_vars(self) -> int:
        return self.gf_term.n_vars

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the product at points x."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.gf_term.evaluate(x) * self.composite_term.evaluate(x)

    def weighted_degree(self, weights: Tuple[int, ...]) -> Fraction:
        """Weighted degree is sum of component degrees."""
        return self.gf_term.weighted_degree(weights) + self.composite_term.weighted_degree(weights)


def multiply_term_by_composite(
    term: GFTerm,
    composite: CompositeGFTerm
) -> ProductCompositeTerm:
    """
    Multiply a GFTerm by a CompositeGFTerm.

    Args:
        term: Standard GF term
        composite: Composite GF term (nested expression)

    Returns:
        ProductCompositeTerm representing the product
    """
    return ProductCompositeTerm(gf_term=term, composite_term=composite)


def multiply_gf_by_gf(
    gf1: GeneralizedForm,
    gf2: GeneralizedForm,
    validate_degree: bool = False
) -> GeneralizedForm:
    """
    Multiply two GFs term-by-term.

    The product of two r-homogeneous GFs with degrees m₁ and m₂
    is an r-homogeneous GF with degree m₁ + m₂.

    For mixed-degree GFs, the result will also be mixed-degree.

    This is used for weighted-sphere SOS: ||x||_r^k · W where
    ||x||_r^k clears negative exponents in W.

    Args:
        gf1: First Generalized Form (degree m₁, or None for mixed)
        gf2: Second Generalized Form (degree m₂, or None for mixed)
        validate_degree: If True, validate result degree matches m₁ + m₂

    Returns:
        Product GF with degree m₁ + m₂ (or mixed if inputs are mixed)

    Example:
        For STA (r=(2,1)):
        ||x||_r = |x₁| + |x₂|²  (degree 2)
        W = ... (degree 2 + κ = 2 + (-0.5) = 1.5, but contains negative exponents)
        ||x||_r · W has degree 2 + 1.5 = 3.5 and all non-negative exponents
    """
    assert gf1.weights_r == gf2.weights_r, \
        f"Weights must match: {gf1.weights_r} vs {gf2.weights_r}"

    result_terms = []

    # Multiply every term of gf1 with every term of gf2
    for t1 in gf1.terms:
        for t2 in gf2.terms:
            product = multiply_gf_terms(t1, t2)
            result_terms.append(product)

    # Collect and simplify
    collected = collect_terms(result_terms)

    # Compute result degree (handle mixed-degree inputs)
    if gf1.degree_m is None or gf2.degree_m is None:
        # One or both inputs are mixed-degree
        # Check if result is homogeneous
        term_degrees = {term.weighted_degree(gf1.weights_r) for term in collected}
        if len(term_degrees) == 1:
            result_degree = int(list(term_degrees)[0])
            strict = True
        else:
            result_degree = None
            strict = False
    else:
        result_degree = gf1.degree_m + gf2.degree_m
        strict = True

    # Build result GF (bypass validation since we compute degree)
    result = GeneralizedForm.__new__(GeneralizedForm)
    result.degree_m = result_degree
    result.weights_r = gf1.weights_r
    result.terms = collected
    result.composite_terms = []  # No composite terms from multiplication of standard GFs
    result.strict_homogeneous = strict

    if validate_degree and collected and result_degree is not None:
        for term in collected:
            deg = term.weighted_degree(gf1.weights_r)
            if deg != result_degree:
                import warnings
                warnings.warn(
                    f"Term {term} has degree {deg}, expected {result_degree}"
                )

    return result


def compute_lie_derivative_gf(
    V: GeneralizedForm,
    f: List[GeneralizedForm],
    output_degree: Optional[int] = None,
    validate: bool = True
) -> GeneralizedForm:
    """
    Compute Lie derivative as a GF (when f is also GF-structured).

    For r-homogeneous systems with degree κ:
    - V is degree m
    - f is degree κ (each component fᵢ has weighted degree rᵢ + κ)
    - dV/dt has degree m + κ

    For quasi-homogeneous systems (like STA) where dynamics have mixed degrees:
    - f₁ has degree d₁, f₂ has degree d₂, etc.
    - ∂V/∂xᵢ has degree m - rᵢ
    - (∂V/∂xᵢ) * fᵢ has degree m - rᵢ + dᵢ
    - W may have mixed degrees OR may end up homogeneous (e.g., STA)

    The Lie derivative is: dV/dt = Σᵢ (∂V/∂xᵢ) · fᵢ

    NOTE: This function now supports dynamics with CompositeGFTerms.
    When dynamics have composite terms, the result may include
    ProductCompositeTerms which are stored separately for numerical evaluation.

    Args:
        V: Lyapunov candidate GF
        f: Dynamics as list of GFs (one per component fᵢ)
            Each fᵢ should be a GF with appropriate degree (can be different)
            Can include composite_terms for nested GF expressions
        output_degree: Expected degree of dV/dt (default: inferred, None for mixed)
        validate: If True, validate output degree (disable for flexibility)

    Returns:
        GeneralizedForm for dV/dt (NOT negated - caller should negate for W)
        If output has mixed degrees, returns with strict_homogeneous=False
        If dynamics have composite terms, the result will have _product_composite_terms
        attribute for numerical evaluation.
    """
    n = V.n_vars
    weights_r = V.weights_r
    degree_m = V.degree_m

    assert len(f) == n, f"Expected {n} dynamics components, got {len(f)}"

    all_terms = []
    product_composite_terms = []  # For composite term contributions

    # Check if any dynamics have composite terms
    has_composite_dynamics = any(fi.has_composite_terms for fi in f)

    # For each component i, compute (∂V/∂xᵢ) · fᵢ
    for i in range(n):
        # Compute ∂V/∂xᵢ as a sum of GF terms
        for v_term in V.terms:
            deriv_term = gf_partial_derivative_term(v_term, i)

            if deriv_term is None:
                continue  # This term doesn't depend on xᵢ

            # Multiply deriv_term by standard terms in fᵢ
            product_terms = multiply_gf_by_term(f[i], deriv_term)
            all_terms.extend(product_terms)

            # Handle composite terms in fᵢ
            if hasattr(f[i], 'composite_terms') and f[i].composite_terms:
                for ct in f[i].composite_terms:
                    # Product of deriv_term (GFTerm) and ct (CompositeGFTerm)
                    pct = multiply_term_by_composite(deriv_term, ct)
                    product_composite_terms.append(pct)

    # Collect and simplify standard terms
    collected_terms = collect_terms(all_terms)

    if len(collected_terms) == 0 and len(product_composite_terms) == 0:
        # Lie derivative is zero (unusual but possible)
        # Return a GF with a single zero term
        zero_term = GFTerm(
            coefficient=0.0,
            exponents=tuple(Fraction(0) for _ in range(n)),
            term_types=tuple('abs' for _ in range(n))
        )
        # Use degree 0 for zero GF
        return GeneralizedForm(degree_m=0, weights_r=weights_r, terms=[zero_term])

    # Check if all standard terms have the same weighted degree
    if collected_terms:
        term_degrees = {term.weighted_degree(weights_r) for term in collected_terms}
        is_homogeneous_result = len(term_degrees) == 1
    else:
        term_degrees = set()
        is_homogeneous_result = True

    # Include composite term degrees in homogeneity check
    if product_composite_terms:
        composite_degrees = {pct.weighted_degree(weights_r) for pct in product_composite_terms}
        term_degrees = term_degrees.union(composite_degrees)
        is_homogeneous_result = len(term_degrees) == 1

    # Determine output degree and homogeneity
    if is_homogeneous_result and term_degrees:
        # All terms have same degree - create homogeneous GF
        computed_degree = list(term_degrees)[0]
        # Handle fractional degrees
        if computed_degree == int(computed_degree):
            computed_degree = int(computed_degree)
        if output_degree is not None and output_degree != computed_degree:
            if validate:
                import warnings
                warnings.warn(
                    f"Output degree {output_degree} doesn't match computed {computed_degree}. "
                    "Using computed degree."
                )
        output_degree = computed_degree if isinstance(computed_degree, int) else None
        strict = isinstance(computed_degree, int)
    else:
        # Mixed degrees - create non-strict GF
        if validate and term_degrees:
            import warnings
            warnings.warn(
                f"Lie derivative has mixed degrees: {term_degrees}. "
                "Creating mixed-degree GF with strict_homogeneous=False."
            )
        # Use None for degree_m to indicate mixed degrees
        output_degree = None
        strict = False

    # Build the result GF
    result = GeneralizedForm.__new__(GeneralizedForm)
    result.degree_m = output_degree
    result.weights_r = weights_r
    result.terms = collected_terms
    result.composite_terms = []  # Standard composite terms (not product composites)
    result.strict_homogeneous = strict

    # Store product composite terms for numerical evaluation
    # These are NOT standard GF terms but are needed for correct W evaluation
    if product_composite_terms:
        result._product_composite_terms = product_composite_terms
        if validate:
            import warnings
            warnings.warn(
                f"Lie derivative contains {len(product_composite_terms)} product composite terms. "
                "These will be included in numerical evaluation but may not be SOS-verifiable."
            )

    return result


def negate_gf(gf: GeneralizedForm) -> GeneralizedForm:
    """
    Negate a GF (multiply all coefficients by -1).

    Useful for computing W = -dV/dt from dV/dt.

    Args:
        gf: Generalized Form to negate

    Returns:
        New GF with negated coefficients
    """
    negated_terms = [
        GFTerm(
            coefficient=-term.coefficient,
            exponents=term.exponents,
            term_types=term.term_types
        )
        for term in gf.terms
    ]

    # Negate composite terms if present
    negated_composite_terms = []
    if hasattr(gf, 'composite_terms') and gf.composite_terms:
        negated_composite_terms = [
            CompositeGFTerm(
                inner_gf=ct.inner_gf,
                outer_exponent=ct.outer_exponent,
                outer_type=ct.outer_type,
                coefficient=-ct.coefficient
            )
            for ct in gf.composite_terms
        ]

    # Preserve mixed-degree handling
    result = GeneralizedForm.__new__(GeneralizedForm)
    result.degree_m = gf.degree_m
    result.weights_r = gf.weights_r
    result.terms = negated_terms
    result.composite_terms = negated_composite_terms
    result.strict_homogeneous = gf.strict_homogeneous

    # Handle product composite terms from Lie derivative
    if hasattr(gf, '_product_composite_terms') and gf._product_composite_terms:
        negated_pct = [
            ProductCompositeTerm(
                gf_term=GFTerm(
                    coefficient=-pct.gf_term.coefficient,
                    exponents=pct.gf_term.exponents,
                    term_types=pct.gf_term.term_types
                ),
                composite_term=pct.composite_term
            )
            for pct in gf._product_composite_terms
        ]
        result._product_composite_terms = negated_pct

    return result


def evaluate_gf_with_composites(gf: GeneralizedForm, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a GF that may have product composite terms.

    This is a helper for evaluating Lie derivatives when dynamics have
    composite terms. The standard GF.evaluate() doesn't include the
    _product_composite_terms contribution.

    Args:
        gf: GeneralizedForm (possibly from compute_lie_derivative_gf)
        x: Points of shape (N, n) or (n,)

    Returns:
        Values including all contributions (standard + composite + product composite)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Standard evaluation
    result = gf.evaluate(x)

    # Add product composite term contributions if present
    if hasattr(gf, '_product_composite_terms') and gf._product_composite_terms:
        for pct in gf._product_composite_terms:
            result = result + pct.evaluate(x)

    return result.squeeze() if x.shape[0] == 1 else result


class LieDerivativeEvaluator:
    """
    Efficient numerical evaluator for Lie derivatives.

    Caches symbolic computations for fast batch evaluation.
    """

    def __init__(
        self,
        gf: GeneralizedForm,
        dynamics_fn: Callable[[np.ndarray], np.ndarray]
    ):
        """
        Initialize evaluator.

        Args:
            gf: Generalized Form V(x)
            dynamics_fn: Function x → f(x) returning (N, n) array
        """
        self.gf = gf
        self.dynamics_fn = dynamics_fn
        self.n = gf.n_vars

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate dV/dt at points x.

        Args:
            x: Points of shape (N, n) or (n,)

        Returns:
            dV/dt values of shape (N,) or scalar
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Compute gradient numerically
        grad_V = self._compute_gradient_batch(x)  # (N, n)

        # Get dynamics
        f = self.dynamics_fn(x)  # (N, n)

        # Lie derivative: sum of grad_V * f along axis 1
        dVdt = np.sum(grad_V * f, axis=1)

        return dVdt.squeeze()

    def _compute_gradient_batch(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Compute gradient at batch of points using finite differences.

        Args:
            x: Points of shape (N, n)
            eps: Finite difference step

        Returns:
            Gradients of shape (N, n)
        """
        N = len(x)
        grad = np.zeros((N, self.n))

        for i in range(self.n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[:, i] += eps
            x_minus[:, i] -= eps

            V_plus = self.gf.evaluate(x_plus)
            V_minus = self.gf.evaluate(x_minus)

            grad[:, i] = (V_plus - V_minus) / (2 * eps)

        return grad


class FilippovLieDerivative:
    """
    Lie derivative evaluator for discontinuous (Filippov) systems.

    For systems with sign(x) discontinuities, computes worst-case
    dV/dt over the Filippov differential inclusion at switching surfaces.
    """

    def __init__(
        self,
        gf: GeneralizedForm,
        dynamics_plus: Callable[[np.ndarray], np.ndarray],
        dynamics_minus: Callable[[np.ndarray], np.ndarray],
        switching_index: int = 0,
        switching_threshold: float = 1e-6
    ):
        """
        Initialize Filippov evaluator.

        Args:
            gf: Generalized Form V(x)
            dynamics_plus: Dynamics when x[switching_index] > 0
            dynamics_minus: Dynamics when x[switching_index] < 0
            switching_index: Index of switching variable
            switching_threshold: Threshold for "near switching surface"
        """
        self.gf = gf
        self.f_plus = dynamics_plus
        self.f_minus = dynamics_minus
        self.switch_idx = switching_index
        self.threshold = switching_threshold
        self.n = gf.n_vars

        self.eval_plus = LieDerivativeEvaluator(gf, dynamics_plus)
        self.eval_minus = LieDerivativeEvaluator(gf, dynamics_minus)

    def evaluate_worst_case(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate worst-case dV/dt over Filippov inclusion.

        For points near the switching surface, returns max(dV⁺, dV⁻).
        Away from the surface, returns the appropriate single-valued dV/dt.

        Args:
            x: Points of shape (N, n) or (n,)

        Returns:
            Worst-case dV/dt values
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        s = x[:, self.switch_idx]  # Switching variable values

        dV_plus = self.eval_plus.evaluate(x)
        dV_minus = self.eval_minus.evaluate(x)

        # Near switching surface: take worst case
        near_surface = np.abs(s) < self.threshold
        dV = np.where(
            near_surface,
            np.maximum(dV_plus, dV_minus),  # Worst case
            np.where(s > 0, dV_plus, dV_minus)  # Single-valued
        )

        return dV.squeeze()

    def evaluate_both(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate both dV⁺ and dV⁻.

        Useful for detailed analysis and debugging.

        Args:
            x: Points of shape (N, n) or (n,)

        Returns:
            (dV_plus, dV_minus) arrays
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        dV_plus = self.eval_plus.evaluate(x)
        dV_minus = self.eval_minus.evaluate(x)

        return dV_plus.squeeze(), dV_minus.squeeze()
