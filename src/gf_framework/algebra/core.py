"""
Generalized Forms Algebra Module.

Implements the algebraic representation of Generalized Forms (GF) as defined
in Sanchez-Moreno (2019): "Design of Lyapunov functions for a class of
homogeneous systems: Generalized forms approach."

A GF of degree m with weights r = (r₁, ..., rₙ) is:
    V(x) = Σᵢ αᵢ|xᵢ|^(m/rᵢ) + Σⱼ ᾱⱼ ∏ᵢ υᵢ,ⱼ(xᵢ, ρᵢ,ⱼ)

where υᵢ,ⱼ is either |xᵢ|^ρ (absolute) or ⌈xᵢ⌋^ρ (signed power).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Union
from math import gcd
from functools import reduce
import numpy as np
import sympy as sp


def lcm(a: int, b: int) -> int:
    """Compute least common multiple of two integers."""
    return abs(a * b) // gcd(a, b)


def lcm_list(nums: List[int]) -> int:
    """Compute LCM of a list of integers."""
    return reduce(lcm, nums)


def sgnpow_numpy(x: np.ndarray, p: float) -> np.ndarray:
    """
    Compute signed power: ⌈x⌋^p = sign(x) * |x|^p

    For p = 0, returns sign(x) (set-valued at x=0, we use 0).
    """
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


def sgnpow_sympy(x: sp.Expr, p: Union[int, float, Fraction]) -> sp.Expr:
    """
    Symbolic signed power: ⌈x⌋^p = sign(x) * |x|^p
    """
    p_val = float(p) if isinstance(p, Fraction) else p
    if p_val == 0:
        return sp.sign(x)
    return sp.sign(x) * sp.Abs(x) ** p_val


@dataclass
class GFTerm:
    """
    A single term in a Generalized Form.

    Represents: coefficient * ∏ᵢ υᵢ(xᵢ, exponents[i])
    where υᵢ is |xᵢ|^exp if types[i] == 'abs', or ⌈xᵢ⌋^exp if types[i] == 'signed'.

    Attributes:
        coefficient: The scalar multiplier
        exponents: Tuple of rational exponents (ρ₁, ..., ρₙ)
        term_types: Tuple of 'abs' or 'signed' for each variable
    """
    coefficient: float
    exponents: Tuple[Fraction, ...]
    term_types: Tuple[str, ...]  # Each element is 'abs' or 'signed'

    def __post_init__(self):
        """Validate term structure."""
        assert len(self.exponents) == len(self.term_types), \
            "Exponents and term_types must have same length"
        for typ in self.term_types:
            assert typ in ('abs', 'signed'), f"Invalid term type: {typ}"

    @property
    def n_vars(self) -> int:
        """Number of variables in this term."""
        return len(self.exponents)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate term at points x.

        Args:
            x: Input points of shape (N, n_vars) or (n_vars,)

        Returns:
            Values of shape (N,) or scalar
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        result = np.full(len(x), self.coefficient, dtype=np.float64)

        for i, (exp, typ) in enumerate(zip(self.exponents, self.term_types)):
            exp_float = float(exp)

            if exp_float == 0:
                # |x|^0 = 1 (no contribution), but ⌈x⌋^0 = sign(x)!
                if typ == 'signed':
                    result *= np.sign(x[:, i])
                # For 'abs' type, |x|^0 = 1, so no multiplication needed
                continue

            if typ == 'abs':
                result *= np.abs(x[:, i]) ** exp_float
            else:  # 'signed'
                result *= sgnpow_numpy(x[:, i], exp_float)

        return result.squeeze()

    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        """
        Convert term to SymPy expression.

        Args:
            symbols: List of SymPy symbols [x₁, x₂, ..., xₙ]

        Returns:
            SymPy expression for this term
        """
        expr = sp.Float(self.coefficient)

        for i, (exp, typ) in enumerate(zip(self.exponents, self.term_types)):
            exp_float = float(exp)
            if exp_float == 0:
                if typ == 'signed':
                    expr *= sp.sign(symbols[i])
                # |x|^0 = 1, no multiplication needed
                continue

            if typ == 'abs':
                expr *= sp.Abs(symbols[i]) ** exp_float
            else:  # 'signed'
                expr *= sgnpow_sympy(symbols[i], exp_float)

        return expr

    def weighted_degree(self, weights: Tuple[int, ...]) -> Fraction:
        """
        Compute weighted degree Σᵢ rᵢ * ρᵢ.

        Args:
            weights: Homogeneity weights (r₁, ..., rₙ)

        Returns:
            Weighted degree as a Fraction
        """
        return sum(Fraction(w) * exp for w, exp in zip(weights, self.exponents))

    def __repr__(self) -> str:
        # float(...) keeps this version-safe: Fraction did not support the
        # general 'g' format spec before Python 3.12, so formatting an
        # exact-rational coefficient (Fraction) directly with :.4g crashes on
        # 3.11 and earlier. float(coefficient) matches 3.12+'s own behavior
        # (which converts to float for 'g' presentation) for float/int/Fraction
        # coefficients alike.
        parts = [f"{float(self.coefficient):.4g}"]
        for i, (exp, typ) in enumerate(zip(self.exponents, self.term_types)):
            if exp == 0:
                if typ == 'signed':
                    parts.append(f"sign(x{i+1})")
            elif typ == 'abs':
                parts.append(f"|x{i+1}|^{exp}")
            else:
                parts.append(f"⌈x{i+1}⌋^{exp}")
        return " * ".join(parts)


@dataclass
class CompositeGFTerm:
    """
    A GF term that wraps a nested GF expression.

    Represents: coefficient * ⌈inner_gf⌋^outer_exponent (signed)
             or: coefficient * |inner_gf|^outer_exponent (abs)

    This is needed for controllers with nested GF structures, such as
    the Continuous Twisting Algorithm (CTA) which has terms like:
        -k₂⌈⌈x₁⌋^(2/3) + x₂⌋^(1/2)

    Here:
        inner_gf = GF with terms [⌈x₁⌋^(2/3), x₂]
        outer_exponent = Fraction(1, 2)
        outer_type = 'signed'
        coefficient = -k₂

    The inner GF represents a "sliding variable" σ = ⌈x₁⌋^(2/3) + x₂,
    and the composite term represents ⌈σ⌋^(1/2).

    Attributes:
        inner_gf: The inner GF expression (can be any GeneralizedForm)
        outer_exponent: Exponent applied to the inner GF result
        outer_type: 'signed' for ⌈·⌋^exp or 'abs' for |·|^exp
        coefficient: Scalar multiplier
    """
    inner_gf: 'GeneralizedForm'  # Forward reference
    outer_exponent: Fraction
    outer_type: str  # 'signed' or 'abs'
    coefficient: float = 1.0

    def __post_init__(self):
        """Validate composite term structure."""
        assert self.outer_type in ('abs', 'signed'), \
            f"Invalid outer_type: {self.outer_type}"

    @property
    def n_vars(self) -> int:
        """Number of variables (from inner GF)."""
        return self.inner_gf.n_vars

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the composite term at points x.

        Args:
            x: Input points of shape (N, n_vars) or (n_vars,)

        Returns:
            Values of shape (N,) or scalar
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Evaluate inner GF: σ(x)
        inner_val = self.inner_gf.evaluate(x)

        # Apply outer power
        exp_float = float(self.outer_exponent)
        if self.outer_type == 'signed':
            result = self.coefficient * sgnpow_numpy(inner_val, exp_float)
        else:  # 'abs'
            if exp_float == 0:
                result = self.coefficient * np.ones_like(inner_val)
            else:
                result = self.coefficient * np.abs(inner_val) ** exp_float

        return result.squeeze()

    def to_sympy(self, symbols: List[sp.Symbol]) -> sp.Expr:
        """
        Convert composite term to SymPy expression.

        Args:
            symbols: List of SymPy symbols [x₁, x₂, ..., xₙ]

        Returns:
            SymPy expression for this composite term
        """
        # Get inner GF as sympy expression
        inner_expr = self.inner_gf.to_sympy(symbols)

        exp_float = float(self.outer_exponent)
        if self.outer_type == 'signed':
            outer_expr = sgnpow_sympy(inner_expr, exp_float)
        else:  # 'abs'
            if exp_float == 0:
                outer_expr = sp.Integer(1)
            else:
                outer_expr = sp.Abs(inner_expr) ** exp_float

        return sp.Float(self.coefficient) * outer_expr

    def weighted_degree(self, weights: Tuple[int, ...]) -> Fraction:
        """
        Compute weighted degree of composite term.

        For a composite term ⌈σ(x)⌋^ρ where σ has weighted degree d_σ,
        the composite has weighted degree: d_σ * ρ.

        This follows from the homogeneity property:
            ⌈σ(Λ_ε^r x)⌋^ρ = ⌈ε^{d_σ} σ(x)⌋^ρ = ε^{d_σ·ρ} ⌈σ(x)⌋^ρ

        Args:
            weights: Homogeneity weights (r₁, ..., rₙ)

        Returns:
            Weighted degree as a Fraction
        """
        # Inner GF should be homogeneous for this to make sense
        if self.inner_gf.degree_m is not None:
            inner_degree = Fraction(self.inner_gf.degree_m)
        else:
            # Use max degree for non-homogeneous inner GF
            inner_degree = self.inner_gf.max_degree

        return inner_degree * self.outer_exponent

    def __repr__(self) -> str:
        inner_str = f"({self.inner_gf.n_terms} terms)"
        # Same Fraction-format guard as GFTerm.__repr__ (crashes on 3.11).
        if self.outer_type == 'abs':
            return f"{float(self.coefficient):.4g} * |σ{inner_str}|^{self.outer_exponent}"
        else:
            return f"{float(self.coefficient):.4g} * ⌈σ{inner_str}⌋^{self.outer_exponent}"


@dataclass
class GeneralizedForm:
    """
    Complete Generalized Form representation.

    Can represent:
    - Homogeneous GF (degree_m set, strict=True): All terms have same degree
    - Mixed-degree GF (degree_m=None or strict=False): Terms can have different degrees

    A GF is r-homogeneous of degree m if all terms satisfy:
        Σᵢ rᵢ * ρᵢ,ⱼ = m  (for each term j)

    Extended to support composite terms (nested GF expressions) for controllers
    like CTA that have terms of the form ⌈σ(x)⌋^ρ where σ is itself a GF.

    Attributes:
        degree_m: Homogeneous degree (None = mixed degrees allowed)
        weights_r: Dilation weights (r₁, ..., rₙ)
        terms: List of GFTerm objects (standard product terms)
        composite_terms: List of CompositeGFTerm objects (nested GF terms)
        strict_homogeneous: If True, validate all terms have same degree (default True)
    """
    degree_m: Optional[int]
    weights_r: Tuple[int, ...]
    terms: List[GFTerm] = field(default_factory=list)
    composite_terms: List[CompositeGFTerm] = field(default_factory=list)
    strict_homogeneous: bool = True

    def __post_init__(self):
        """Validate GF structure."""
        self._validate_terms()

    def _validate_terms(self):
        """Check that all terms satisfy homogeneity constraint (if strict)."""
        for term in self.terms:
            assert term.n_vars == len(self.weights_r), \
                f"Term has {term.n_vars} vars but GF has {len(self.weights_r)} weights"

            if self.strict_homogeneous and self.degree_m is not None:
                weighted_deg = term.weighted_degree(self.weights_r)
                assert weighted_deg == self.degree_m, \
                    f"Term {term} has weighted degree {weighted_deg}, expected {self.degree_m}"

    @property
    def min_degree(self) -> Fraction:
        """Minimum weighted degree among all terms."""
        if not self.terms:
            return Fraction(0)
        return min(t.weighted_degree(self.weights_r) for t in self.terms)

    @property
    def max_degree(self) -> Fraction:
        """Maximum weighted degree among all terms."""
        if not self.terms:
            return Fraction(0)
        return max(t.weighted_degree(self.weights_r) for t in self.terms)

    @property
    def is_homogeneous(self) -> bool:
        """Check if all terms have the same weighted degree."""
        if len(self.terms) <= 1:
            return True
        degrees = {t.weighted_degree(self.weights_r) for t in self.terms}
        return len(degrees) == 1

    @property
    def n_vars(self) -> int:
        """Number of state variables."""
        return len(self.weights_r)

    @property
    def n_terms(self) -> int:
        """Number of terms in the GF (including composite terms)."""
        return len(self.terms) + len(self.composite_terms)

    @property
    def has_composite_terms(self) -> bool:
        """Check if GF contains composite (nested) terms."""
        return len(self.composite_terms) > 0

    @property
    def lcm_weights(self) -> int:
        """LCM of the weights (needed for Lemma 6 transform)."""
        return lcm_list(list(self.weights_r))

    @property
    def mu_exponents(self) -> Tuple[int, ...]:
        """
        Compute per-variable μᵢ = LCM of denominators in xᵢ exponents.

        Implements Lemma 6 from Sanchez & Moreno (2019):
        "There exist cᵢ ∈ ℤ₊ such that rᵢ/(cᵢLᵢ) = β for any i,j"

        The transformation yᵢ = |xᵢ|^(1/μᵢ) converts fractional-exponent GF terms
        to integer-exponent polynomials in y-space, enabling Pólya/SOS verification.

        For term with exponent ρᵢ, y-exponent = μᵢ × ρᵢ must be an integer.

        SOSTOOLS Approach (more efficient):
        Compute μᵢ = LCM of denominators in xᵢ exponents only, not all denominators.
        This minimizes the maximum y-degree and keeps SOS problems tractable.

        Reference: Sanchez, T., & Moreno, J.A. (2019). Int J Robust Nonlinear
        Control, 29(3), 661-681. Lemma 6 and Algorithm 1.

        Example for PID-SMC (r=(3,2,1), m=7):
        - Old (inefficient): μ = (18, 12, 6) → y-degree 42 (2024 SOS monomials)
        - New (efficient): μ = (3, 2, 1) → y-degree 7 (20 SOS monomials!)

        The per-variable approach matches SOSTOOLS and dramatically reduces SDP size.
        """
        n_vars = len(self.weights_r)

        # Collect denominators per variable
        per_var_denoms = [set() for _ in range(n_vars)]
        for term in self.terms:
            for i, exp in enumerate(term.exponents):
                per_var_denoms[i].add(exp.denominator)

        # Compute μᵢ = LCM of denominators for variable i
        mu = []
        for i in range(n_vars):
            denoms = list(per_var_denoms[i])
            mu_i = lcm_list(denoms) if denoms else 1
            mu.append(mu_i)

        return tuple(mu)

    def add_term(self, term: GFTerm):
        """Add a term to the GF, validating homogeneity if strict."""
        assert term.n_vars == self.n_vars, "Term dimension mismatch"
        if self.strict_homogeneous and self.degree_m is not None:
            weighted_deg = term.weighted_degree(self.weights_r)
            assert weighted_deg == self.degree_m, \
                f"Term has weighted degree {weighted_deg}, expected {self.degree_m}"
        self.terms.append(term)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate GF at points x.

        Args:
            x: Input points of shape (N, n_vars) or (n_vars,)

        Returns:
            V(x) values
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        result = np.zeros(len(x), dtype=np.float64)

        # Evaluate standard terms
        for term in self.terms:
            result += term.evaluate(x)

        # Evaluate composite terms (nested GF expressions)
        for ct in self.composite_terms:
            result += ct.evaluate(x)

        return result.squeeze()

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇V(x) numerically.

        Args:
            x: Single point of shape (n_vars,)

        Returns:
            Gradient of shape (n_vars,)
        """
        eps = 1e-8
        grad = np.zeros(self.n_vars)

        for i in range(self.n_vars):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)

        return grad

    def numerical_hessian(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute Hessian matrix H_V(x) via finite differences.

        Args:
            x: Single point of shape (n_vars,).
            eps: Finite-difference step size.

        Returns:
            Hessian matrix of shape (n_vars, n_vars).
        """
        n = self.n_vars
        H = np.zeros((n, n))
        V0 = self.evaluate(x)
        for i in range(n):
            xi_plus = x.copy()
            xi_plus[i] += eps
            xi_minus = x.copy()
            xi_minus[i] -= eps
            H[i, i] = (self.evaluate(xi_plus) + self.evaluate(xi_minus) - 2.0 * V0) / (eps * eps)
            for j in range(i + 1, n):
                xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                H[i, j] = (self.evaluate(xpp) - self.evaluate(xpm) - self.evaluate(xmp) + self.evaluate(xmm)) / (4.0 * eps * eps)
                H[j, i] = H[i, j]
        return H

    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute Hessian-vector product H_V(x) @ v via finite differences.

        More efficient than computing full Hessian when only one direction needed.

        Args:
            x: Single point of shape (n_vars,).
            v: Direction vector of shape (n_vars,).
            eps: Finite-difference step size.

        Returns:
            H @ v of shape (n_vars,).
        """
        g_plus = self.gradient(x + eps * v)
        g_minus = self.gradient(x - eps * v)
        return (g_plus - g_minus) / (2.0 * eps)

    def numerical_hessian_batch(self, points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute Hessian matrices for a batch of points.

        Args:
            points: Array of shape (N, n_vars).
            eps: Finite-difference step size.

        Returns:
            Array of shape (N, n_vars, n_vars).
        """
        return np.array([self.numerical_hessian(p, eps) for p in points])

    def get_symbols(self) -> List[sp.Symbol]:
        """Create SymPy symbols for this GF's variables."""
        return [sp.Symbol(f'x{i+1}', real=True) for i in range(self.n_vars)]

    def to_sympy(self, symbols: Optional[List[sp.Symbol]] = None) -> sp.Expr:
        """
        Convert GF to SymPy expression.

        Args:
            symbols: Optional list of symbols; creates default if None

        Returns:
            SymPy expression V(x)
        """
        if symbols is None:
            symbols = self.get_symbols()

        expr = sp.Integer(0)

        # Add standard terms
        for term in self.terms:
            expr += term.to_sympy(symbols)

        # Add composite terms
        for ct in self.composite_terms:
            expr += ct.to_sympy(symbols)

        return sp.simplify(expr)

    def is_positive_at(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if V(x) > 0 at a point (excluding origin)."""
        if np.allclose(x, 0, atol=tol):
            return True  # V(0) = 0 by homogeneity
        return self.evaluate(x) > tol

    def get_coefficients(self) -> np.ndarray:
        """Extract coefficient vector from all terms."""
        return np.array([t.coefficient for t in self.terms])

    def set_coefficients(self, coeffs: np.ndarray):
        """Set coefficients from a vector (in-place)."""
        assert len(coeffs) == len(self.terms), "Coefficient count mismatch"
        for i, c in enumerate(coeffs):
            self.terms[i].coefficient = float(c)

    def copy(self) -> 'GeneralizedForm':
        """Create a deep copy of this GF."""
        new_terms = [
            GFTerm(
                coefficient=t.coefficient,
                exponents=t.exponents,
                term_types=t.term_types
            )
            for t in self.terms
        ]
        new_composite_terms = [
            CompositeGFTerm(
                inner_gf=ct.inner_gf.copy(),
                outer_exponent=ct.outer_exponent,
                outer_type=ct.outer_type,
                coefficient=ct.coefficient
            )
            for ct in self.composite_terms
        ]
        return GeneralizedForm(
            degree_m=self.degree_m,
            weights_r=self.weights_r,
            terms=new_terms,
            composite_terms=new_composite_terms,
            strict_homogeneous=self.strict_homogeneous
        )

    def __repr__(self) -> str:
        if self.degree_m is not None:
            header = f"GF(m={self.degree_m}, r={self.weights_r}, {self.n_terms} terms)"
        else:
            # Mixed-degree GF
            header = f"GF(m=[{self.min_degree}-{self.max_degree}], r={self.weights_r}, {self.n_terms} terms)"
        if self.n_terms <= 5:
            term_strs = [f"  {t}" for t in self.terms]
            return header + "\n" + "\n".join(term_strs)
        return header


def create_diagonal_gf(
    degree_m: int,
    weights_r: Tuple[int, ...],
    coefficients: Optional[List[float]] = None
) -> GeneralizedForm:
    """
    Create a GF with only diagonal terms: V(x) = Σᵢ αᵢ|xᵢ|^(m/rᵢ).

    Args:
        degree_m: Homogeneous degree
        weights_r: Weight tuple (r₁, ..., rₙ)
        coefficients: Optional coefficients (defaults to 1.0)

    Returns:
        GeneralizedForm with diagonal terms only
    """
    n = len(weights_r)
    if coefficients is None:
        coefficients = [1.0] * n

    terms = []
    for i in range(n):
        exponents = tuple(
            Fraction(degree_m, weights_r[i]) if j == i else Fraction(0)
            for j in range(n)
        )
        term = GFTerm(
            coefficient=coefficients[i],
            exponents=exponents,
            term_types=tuple('abs' for _ in range(n))
        )
        terms.append(term)

    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


def partitions_summing_to(n_vars: int, total: int) -> List[Tuple[int, ...]]:
    """
    Generate all n-tuples of non-negative integers summing to total.

    Args:
        n_vars: Number of variables (tuple length)
        total: Target sum

    Returns:
        List of tuples (k₁, k₂, ..., kₙ) where Σkᵢ = total

    Example:
        partitions_summing_to(2, 2) = [(0, 2), (1, 1), (2, 0)]
    """
    if n_vars == 1:
        return [(total,)]

    result = []
    for first in range(total + 1):
        for rest in partitions_summing_to(n_vars - 1, total - first):
            result.append((first,) + rest)
    return result


def multinomial_coefficient(n: int, partition: Tuple[int, ...]) -> int:
    """
    Compute multinomial coefficient n! / (k₁! k₂! ... kₘ!).

    Args:
        n: Total (must equal sum of partition)
        partition: Tuple (k₁, k₂, ..., kₘ)

    Returns:
        Multinomial coefficient as int

    Example:
        multinomial_coefficient(4, (2, 1, 1)) = 4!/(2!1!1!) = 12
    """
    from math import factorial
    return factorial(n) // reduce(lambda a, b: a * b, [factorial(k) for k in partition], 1)


def build_weighted_norm_gf(
    weights_r: Tuple[int, ...],
    power_k: int
) -> 'GeneralizedForm':
    """
    Construct ||x||_r^k as a GeneralizedForm.

    The weighted r-homogeneous norm is:
        ||x||_r = Σᵢ |xᵢ|^{ρ/rᵢ}  where ρ = max(r)

    This is r-homogeneous of degree ρ:
        ||δᵣλ · x||_r = λ^ρ · ||x||_r

    Raised to power k:
        ||x||_r^k = (Σᵢ |xᵢ|^{ρ/rᵢ})^k

    We expand using the multinomial theorem:
        (a₁ + a₂ + ... + aₙ)^k = Σ (k!/(k₁!k₂!...kₙ!)) · a₁^k₁ · a₂^k₂ · ... · aₙ^kₙ
    where the sum is over all (k₁, ..., kₙ) with Σkᵢ = k.

    Args:
        weights_r: Homogeneity weights (r₁, ..., rₙ)
        power_k: Power to raise the norm to

    Returns:
        GeneralizedForm with degree k*ρ (r-homogeneous)

    Example:
        For STA (r=(2,1), k=2):
        ||x||_r² = (|x₁| + |x₂|²)² = |x₁|² + 2|x₁||x₂|² + |x₂|⁴
        This is r-homogeneous of degree 2*2 = 4.
    """
    n_vars = len(weights_r)
    rho = max(weights_r)

    # Base exponents: |xᵢ|^{ρ/rᵢ}
    # These are the aᵢ in the multinomial expansion
    base_exps = [Fraction(rho, r) for r in weights_r]

    # Multinomial expansion
    terms = []
    for partition in partitions_summing_to(n_vars, power_k):
        # partition = (k₁, k₂, ..., kₙ) with Σkᵢ = k

        # Multinomial coefficient: k! / (k₁!k₂!...kₙ!)
        coef = multinomial_coefficient(power_k, partition)

        # Exponents: kᵢ * (ρ/rᵢ) for each variable
        # This gives the exponent of |xᵢ| in this term
        exponents = tuple(
            Fraction(ki * rho, weights_r[i])
            for i, ki in enumerate(partition)
        )

        # All terms are absolute (no signed powers in norm)
        term_types = ('abs',) * n_vars

        terms.append(GFTerm(
            coefficient=float(coef),
            exponents=exponents,
            term_types=term_types
        ))

    return GeneralizedForm(
        degree_m=power_k * rho,  # ||x||_r^k has r-degree k*ρ
        weights_r=weights_r,
        terms=terms
    )


def build_clearing_monomial_gf(
    weights_r: Tuple[int, ...],
    power_k: int
) -> 'GeneralizedForm':
    """
    Construct a product-based clearing monomial: C_k(x) = ∏ᵢ |xᵢ|^{k·ρ/rᵢ}

    Unlike the sum-based norm ||x||_r = Σᵢ |xᵢ|^{ρ/rᵢ}, this is a SINGLE TERM
    (monomial) that contributes to ALL variable y-exponents simultaneously.

    This is essential for clearing negative y-exponents in W = -dV/dt:
    - Sum norm ||x||_r^k has terms like |x₂|^{2k} that don't help clear |x₁|^{-1/2}
    - Product C_k always adds k·ρ to EVERY variable's y-exponent

    Properties:
    - C_k is r-homogeneous of degree k·n·ρ where n = len(weights_r)
    - C_k > 0 for all x with all xᵢ ≠ 0 (positive on open orthants)
    - C_k = 0 on coordinate hyperplanes {x : xᵢ = 0 for some i}
    - Verification relies on continuity: if C_k·W > 0 on open orthants and
      W is continuous, then W ≥ 0 on hyperplanes by continuity.
    - In y-space: C_k → y₁^{k·μ₁·ρ/r₁} · ... = (const) · ∏yᵢ^{k·ρ}

    Args:
        weights_r: Homogeneity weights (r₁, ..., rₙ)
        power_k: Power multiplier for clearing

    Returns:
        GeneralizedForm with single term C_k(x)

    Example:
        For STA (r=(2,1), k=1):
        C_1 = |x₁|^{2/2} · |x₂|^{2/1} = |x₁| · |x₂|²
        r-degree = 2*1 + 1*2 = 4 = 1 * 2 * 2 = k * n * ρ

        Multiplying W term |x₁|^{-1/2}|x₂| by C_1:
        C_1 · |x₁|^{-1/2}|x₂| = |x₁|^{1/2}|x₂|³
        y-exponents: (4·1/2, 2·3) = (2, 6) ✓ all positive!
    """
    n_vars = len(weights_r)
    rho = max(weights_r)

    # Single term: ∏ᵢ |xᵢ|^{k·ρ/rᵢ}
    exponents = tuple(
        Fraction(power_k * rho, r)
        for r in weights_r
    )

    # All absolute (no signed powers)
    term_types = ('abs',) * n_vars

    term = GFTerm(
        coefficient=1.0,
        exponents=exponents,
        term_types=term_types
    )

    # r-degree = Σᵢ rᵢ · (k·ρ/rᵢ) = k · n · ρ
    degree_m = power_k * n_vars * rho

    return GeneralizedForm(
        degree_m=degree_m,
        weights_r=weights_r,
        terms=[term]
    )


# =============================================================================
# COORDINATE LIFTING FUNCTIONS
# =============================================================================

def lift_coordinates(
    x: np.ndarray,
    weights_r: Tuple[int, ...],
    eps: float = 1e-8
) -> np.ndarray:
    """
    Transform physical coordinates x to lifted coordinates ξ.

    For homogeneous systems with weights r = (r₁, r₂, ..., rₙ), the lifted
    coordinates are:
        ξᵢ = ⌈xᵢ⌋^(1/rᵢ) = sign(xᵢ) · |xᵢ|^(1/rᵢ)

    This transformation exposes polynomial structure: a GF in x becomes
    polynomial in ξ (with integer exponents in the μ-space).

    Args:
        x: Physical coordinates, shape (N, n_vars) or (n_vars,)
        weights_r: Homogeneity weights tuple (r₁, ..., rₙ)
        eps: Small epsilon for numerical stability

    Returns:
        ξ: Lifted coordinates, same shape as x

    Example:
        >>> x = np.array([[4.0, 2.0]])  # STA state
        >>> xi = lift_coordinates(x, (2, 1))
        >>> # xi = [sign(4)*|4|^0.5, sign(2)*|2|^1] = [2.0, 2.0]
    """
    was_vector = np.asarray(x).ndim == 1
    x = np.atleast_2d(x)
    n_vars = x.shape[1]

    if len(weights_r) != n_vars:
        raise ValueError(f"weights_r has {len(weights_r)} elements, x has {n_vars} columns")

    xi = np.zeros_like(x, dtype=np.float64)
    for i, r in enumerate(weights_r):
        # ξᵢ = sign(xᵢ) · |xᵢ|^(1/rᵢ)
        xi[:, i] = np.sign(x[:, i]) * (np.abs(x[:, i]) + eps) ** (1.0 / r)

    return xi.squeeze() if was_vector else xi


def unlift_coordinates(
    xi: np.ndarray,
    weights_r: Tuple[int, ...],
    eps: float = 1e-8
) -> np.ndarray:
    """
    Transform lifted coordinates ξ back to physical coordinates x.

    Inverse of lift_coordinates:
        xᵢ = ⌈ξᵢ⌋^rᵢ = sign(ξᵢ) · |ξᵢ|^rᵢ

    Args:
        xi: Lifted coordinates, shape (N, n_vars) or (n_vars,)
        weights_r: Homogeneity weights tuple (r₁, ..., rₙ)
        eps: Small epsilon for numerical stability

    Returns:
        x: Physical coordinates, same shape as ξ

    Example:
        >>> xi = np.array([[2.0, 2.0]])  # Lifted STA state
        >>> x = unlift_coordinates(xi, (2, 1))
        >>> # x = [sign(2)*|2|^2, sign(2)*|2|^1] = [4.0, 2.0]
    """
    was_vector = np.asarray(xi).ndim == 1
    xi = np.atleast_2d(xi)
    n_vars = xi.shape[1]

    if len(weights_r) != n_vars:
        raise ValueError(f"weights_r has {len(weights_r)} elements, xi has {n_vars} columns")

    x = np.zeros_like(xi, dtype=np.float64)
    for i, r in enumerate(weights_r):
        # xᵢ = sign(ξᵢ) · |ξᵢ|^rᵢ
        x[:, i] = np.sign(xi[:, i]) * (np.abs(xi[:, i]) + eps) ** r

    return x.squeeze() if was_vector else x
