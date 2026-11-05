"""
Associated Forms Module.

Converts Generalized Forms to 2^n associated polynomial forms for
verification via Pólya's theorem.

Key transformation (Lemma 6, Sanchez-Moreno):
    yᵢ = ⌈xᵢ⌋^(1/μᵢ)  where μᵢ = LCM(denominators in xᵢ exponents)

Note: For SOS verification (Algorithm 7), μ is doubled to ensure even y-exponents.

This homeomorphism transforms GF terms into standard polynomial terms,
enabling verification via classical Pólya's theorem.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Union
from fractions import Fraction
import sympy as sp
import numpy as np

from .core import GeneralizedForm, GFTerm, lcm_list


def safe_to_int(value: Union[int, float, Fraction], context: str = "") -> int:
    """
    Convert to int with integrality assertion.

    Args:
        value: Value to convert (int, float, or Fraction)
        context: Description of context for error messages

    Returns:
        Integer value

    Raises:
        ValueError: If value is not an integer (within tolerance for floats)
    """
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise ValueError(f"Non-integer y-exponent {value} {context}")
        return value.numerator
    if isinstance(value, float):
        int_val = int(round(value))
        if abs(value - int_val) > 1e-9:
            raise ValueError(f"Non-integer y-exponent {value} {context}")
        return int_val
    return int(value)


def get_hyperoctant_signs(n: int) -> List[Tuple[int, ...]]:
    """
    Generate all 2^n hyperoctant sign combinations.

    Args:
        n: Number of dimensions

    Returns:
        List of sign tuples, e.g., [(1,1), (1,-1), (-1,1), (-1,-1)] for n=2
    """
    signs = []
    for gamma in range(2**n):
        sign_tuple = tuple(1 if (gamma >> i) & 1 == 0 else -1 for i in range(n))
        signs.append(sign_tuple)
    return signs


def transform_to_y_space(
    gf: GeneralizedForm,
    signs: Tuple[int, ...],
    y_symbols: List[sp.Symbol]
) -> sp.Expr:
    """
    Transform GF to polynomial in y-space for a specific hyperoctant.

    The transformation is:
        xᵢ = γᵢ · yᵢ^μᵢ  (inverse of yᵢ = ⌈xᵢ⌋^(1/μᵢ))

    where γᵢ ∈ {-1, +1} specifies the hyperoctant.

    For a term |xᵢ|^ρ:
        |xᵢ|^ρ = |γᵢ|^ρ · yᵢ^(μᵢ·ρ) = yᵢ^(μᵢ·ρ)  (since |γᵢ| = 1)

    For a term ⌈xᵢ⌋^ρ = sign(xᵢ)·|xᵢ|^ρ:
        sign(xᵢ) = γᵢ (constant in this octant)
        |xᵢ|^ρ = yᵢ^(μᵢ·ρ)
        So ⌈xᵢ⌋^ρ = γᵢ · yᵢ^(μᵢ·ρ)

    Args:
        gf: Generalized Form to transform
        signs: Hyperoctant signs (γ₁, ..., γₙ)
        y_symbols: SymPy symbols for y variables

    Returns:
        Polynomial expression in y (valid on positive orthant)
    """
    mu = gf.mu_exponents  # μᵢ = LCM(denominators in xᵢ exponents)
    n = gf.n_vars

    poly = sp.Integer(0)

    for term in gf.terms:
        # Start with coefficient
        term_expr = sp.Float(term.coefficient)

        for i in range(n):
            exp = term.exponents[i]
            typ = term.term_types[i]

            if exp == 0:
                if typ == 'signed':
                    # sign(xᵢ)^0 in signed context still contributes sign(xᵢ) = γᵢ
                    term_expr *= signs[i]
                # |xᵢ|^0 = 1, no contribution
                continue

            # Exponent in y-space: μᵢ · ρᵢ must be an integer
            y_exp = mu[i] * exp
            y_exp_val = safe_to_int(y_exp, f"for term with exp={exp}, mu={mu[i]}")

            if typ == 'abs':
                # |xᵢ|^ρ = yᵢ^(μᵢ·ρ)
                term_expr *= y_symbols[i] ** y_exp_val
            else:
                # ⌈xᵢ⌋^ρ = sign(xᵢ) · |xᵢ|^ρ = γᵢ · yᵢ^{μᵢ·ρ}
                # The sign factor γᵢ ALWAYS appears for signed powers,
                # regardless of whether ρ is odd, even, or fractional.
                term_expr *= signs[i]

                term_expr *= y_symbols[i] ** y_exp_val

        poly += term_expr

    return sp.expand(poly)


def gf_to_associated_forms(gf: GeneralizedForm) -> Dict[Tuple[int, ...], sp.Expr]:
    """
    Convert GF to all 2^n associated polynomial forms.

    Each associated form corresponds to one hyperoctant of the state space.
    Verification on all octants implies global positive definiteness.

    Args:
        gf: Generalized Form to convert

    Returns:
        Dictionary mapping sign tuple → polynomial in y
    """
    n = gf.n_vars
    y_symbols = [sp.Symbol(f'y{i+1}', positive=True) for i in range(n)]

    associated = {}
    for signs in get_hyperoctant_signs(n):
        poly = transform_to_y_space(gf, signs, y_symbols)
        associated[signs] = poly

    return associated


def inverse_transform_point(
    y: np.ndarray,
    signs: Tuple[int, ...],
    mu_exponents: Tuple[int, ...]
) -> np.ndarray:
    """
    Transform a point from y-space back to x-space.

    xᵢ = γᵢ · yᵢ^μᵢ

    Args:
        y: Point in y-space (positive orthant)
        signs: Hyperoctant signs
        mu_exponents: μ = (μ₁, ..., μₙ)

    Returns:
        Point in x-space
    """
    n = len(y)
    x = np.zeros(n)
    for i in range(n):
        x[i] = signs[i] * (y[i] ** mu_exponents[i])
    return x


def forward_transform_point(
    x: np.ndarray,
    mu_exponents: Tuple[int, ...]
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Transform a point from x-space to y-space.

    yᵢ = |xᵢ|^(1/μᵢ)
    signs = (sign(x₁), ..., sign(xₙ))

    Args:
        x: Point in x-space
        mu_exponents: μ = (μ₁, ..., μₙ)

    Returns:
        (y, signs) where y is in positive orthant
    """
    n = len(x)
    y = np.zeros(n)
    signs = []

    for i in range(n):
        signs.append(1 if x[i] >= 0 else -1)
        y[i] = np.abs(x[i]) ** (1.0 / mu_exponents[i])

    return y, tuple(signs)


def get_y_degree(gf: GeneralizedForm) -> int:
    """
    Compute the total degree of the associated form in y-space.

    This is useful for Pólya verification complexity estimation.

    Args:
        gf: Generalized Form

    Returns:
        Degree of associated polynomial
    """
    mu = gf.mu_exponents
    max_degree = 0

    for term in gf.terms:
        term_degree = sum(
            safe_to_int(mu[i] * term.exponents[i], f"for term exponents={term.exponents}")
            for i in range(gf.n_vars)
        )
        max_degree = max(max_degree, term_degree)

    return max_degree


def extract_polynomial_coefficients(
    poly: sp.Expr,
    symbols: List[sp.Symbol]
) -> Dict[Tuple[int, ...], float]:
    """
    Extract all monomial coefficients from a polynomial.

    Args:
        poly: SymPy polynomial expression
        symbols: Variables in the polynomial

    Returns:
        Dictionary mapping exponent tuple → coefficient
    """
    poly = sp.expand(poly)

    try:
        poly_obj = sp.Poly(poly, symbols)
        monoms = poly_obj.monoms()
        coeffs = poly_obj.coeffs()
        return {m: float(c) for m, c in zip(monoms, coeffs)}
    except sp.PolynomialError:
        # Handle case where expression isn't a proper polynomial
        # This shouldn't happen for valid GFs but let's be safe
        return {}


def simplify_associated_form(poly: sp.Expr) -> sp.Expr:
    """
    Simplify an associated form polynomial.

    Args:
        poly: Raw polynomial expression

    Returns:
        Simplified polynomial
    """
    return sp.simplify(sp.expand(poly))


class AssociatedFormsCache:
    """
    Cache for associated forms to avoid recomputation.

    Useful when iterating over coefficients with fixed structure.
    """

    def __init__(self, gf: GeneralizedForm):
        """
        Initialize cache with GF structure.

        Args:
            gf: Template GF (coefficients will be updated)
        """
        self.gf = gf
        self.n = gf.n_vars
        self.y_symbols = [sp.Symbol(f'y{i+1}', positive=True) for i in range(self.n)]
        self.mu = gf.mu_exponents
        self._structure_cache = self._build_structure_cache()

    def _build_structure_cache(self) -> Dict[Tuple[int, ...], List[Tuple[int, sp.Expr]]]:
        """
        Build cache mapping signs → [(term_idx, monomial_expr), ...].

        The polynomial for a given set of coefficients c is:
            poly = Σᵢ cᵢ · structure_cache[signs][i]
        """
        cache = {}

        for signs in get_hyperoctant_signs(self.n):
            term_structures = []

            for term_idx, term in enumerate(self.gf.terms):
                # Build monomial expression (without coefficient)
                mono = sp.Integer(1)
                sign_factor = 1

                for i in range(self.n):
                    exp = term.exponents[i]
                    typ = term.term_types[i]

                    if exp == 0:
                        if typ == 'signed':
                            sign_factor *= signs[i]
                        continue

                    y_exp = self.mu[i] * exp
                    y_exp_val = safe_to_int(y_exp, f"for term with exp={exp}, mu={self.mu[i]}")

                    # ⌈xᵢ⌋^ρ = sign(xᵢ) · |xᵢ|^ρ: sign factor ALWAYS appears for signed powers
                    if typ == 'signed':
                        sign_factor *= signs[i]

                    mono *= self.y_symbols[i] ** y_exp_val

                term_structures.append((term_idx, sign_factor * mono))

            cache[signs] = term_structures

        return cache

    def compute_forms(self, coefficients: np.ndarray) -> Dict[Tuple[int, ...], sp.Expr]:
        """
        Quickly compute associated forms for given coefficients.

        Args:
            coefficients: Coefficient vector matching GF term order

        Returns:
            Dictionary of associated polynomial forms
        """
        forms = {}

        for signs, structures in self._structure_cache.items():
            poly = sp.Integer(0)
            for term_idx, mono_expr in structures:
                poly += coefficients[term_idx] * mono_expr
            forms[signs] = sp.expand(poly)

        return forms
