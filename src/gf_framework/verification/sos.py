from __future__ import annotations
"""
Sum-of-Squares (SOS) Verification Fallback.

When Pólya verification fails or is too slow, we can fall back to
SOS-based verification using semidefinite programming (SDP).

A polynomial p(y) is SOS iff there exists a positive semidefinite
matrix Q such that p(y) = m(y)^T Q m(y), where m(y) is a vector of
monomials.

SOS ⟹ p(y) ≥ 0, but not all non-negative polynomials are SOS.
However, for GFs arising from Lyapunov analysis, SOS often works.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import sympy as sp

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from gf_framework.algebra.core import GeneralizedForm
from gf_framework.algebra.associated import gf_to_associated_forms, get_hyperoctant_signs


@dataclass
class SOSResult:
    """Result of SOS verification."""
    verified: bool
    solver_status: str
    min_eigenvalue: float = 0.0
    message: str = ""


def _normalize_solver_name(solver: str) -> str:
    """Normalize CVXPY solver name with SCS fallback default."""
    candidate = str(solver).strip().upper()
    return candidate if candidate else "SCS"


def get_monomial_basis(
    symbols: List[sp.Symbol],
    max_degree: int
) -> List[sp.Expr]:
    """
    Generate monomial basis up to given degree.

    Args:
        symbols: Variables
        max_degree: Maximum total degree

    Returns:
        List of monomials
    """
    from itertools import combinations_with_replacement

    n = len(symbols)
    monomials = [sp.Integer(1)]  # Constant term

    for deg in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n), deg):
            mono = sp.Integer(1)
            for idx in combo:
                mono *= symbols[idx]
            monomials.append(mono)

    return monomials


def polynomial_to_gram_matrix_constraint(
    poly: sp.Expr,
    symbols: List[sp.Symbol],
    half_degree: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert polynomial to Gram matrix representation.

    For p(y) to be SOS, we need p(y) = m(y)^T Q m(y) for PSD matrix Q.

    This creates the linear constraints on Q entries that match
    the polynomial coefficients.

    Args:
        poly: Polynomial to represent
        symbols: Variables
        half_degree: Maximum degree of monomials in basis

    Returns:
        (A, b) such that A @ vec(Q) = b represents coefficient matching
    """
    # Get monomial basis for half degree
    basis = get_monomial_basis(symbols, half_degree)
    n_basis = len(basis)

    # Get polynomial coefficients
    poly_expanded = sp.expand(poly)
    poly_obj = sp.Poly(poly_expanded, symbols)
    poly_dict = poly_obj.as_dict()

    # All possible product monomials
    product_to_coef = {}
    for mono, coef in poly_dict.items():
        product_to_coef[mono] = float(coef)

    # Build constraint matrix
    # Each product m_i * m_j corresponds to a monomial in the polynomial
    # Sum over all (i,j) pairs that produce the same monomial

    # Get all monomials that appear in m_i * m_j products
    product_monomials = set()
    for i, mi in enumerate(basis):
        for j, mj in enumerate(basis):
            prod = sp.expand(mi * mj)
            if isinstance(prod, sp.Integer) or isinstance(prod, sp.Float):
                mono = tuple([0] * len(symbols))
            else:
                try:
                    prod_poly = sp.Poly(prod, symbols)
                    mono = prod_poly.monoms()[0]
                except Exception:
                    mono = tuple([0] * len(symbols))
            product_monomials.add(mono)

    # Create mapping from monomial to equation index
    mono_list = sorted(product_monomials)
    mono_to_idx = {m: i for i, m in enumerate(mono_list)}
    n_constraints = len(mono_list)

    # Build A matrix (n_constraints x n_basis^2)
    A = np.zeros((n_constraints, n_basis * n_basis))

    for i, mi in enumerate(basis):
        for j, mj in enumerate(basis):
            prod = sp.expand(mi * mj)
            if isinstance(prod, sp.Integer) or isinstance(prod, sp.Float):
                mono = tuple([0] * len(symbols))
            else:
                try:
                    prod_poly = sp.Poly(prod, symbols)
                    mono = prod_poly.monoms()[0]
                except Exception:
                    mono = tuple([0] * len(symbols))

            constraint_idx = mono_to_idx[mono]
            var_idx = i * n_basis + j
            A[constraint_idx, var_idx] += 1.0

    # Build b vector (target coefficients)
    b = np.zeros(n_constraints)
    for mono, idx in mono_to_idx.items():
        b[idx] = product_to_coef.get(mono, 0.0)

    return A, b


def verify_sos_cvxpy(
    poly: sp.Expr,
    symbols: List[sp.Symbol],
    strict_pd: bool = True,
    epsilon: float = 1e-6,
    solver: str = 'SCS',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
) -> SOSResult:
    """
    Verify polynomial is SOS using CVXPY.

    For strict positive definiteness, we check if p(y) - ε·||y||^d is SOS,
    where d is the polynomial degree.

    Args:
        poly: Polynomial to verify
        symbols: Variables
        strict_pd: Whether to verify strict positive definiteness
        epsilon: Strictness parameter
        solver: Preferred CVXPY solver name
        strict_solver: If True, fail instead of trying fallback solvers.
        allow_solver_fallback: If False, use preferred solver only.

    Returns:
        SOSResult
    """
    if not CVXPY_AVAILABLE:
        return SOSResult(
            verified=False,
            solver_status="unavailable",
            message="CVXPY not installed"
        )

    n = len(symbols)

    # Get polynomial degree
    poly_expanded = sp.expand(poly)
    try:
        poly_obj = sp.Poly(poly_expanded, symbols)
        degree = poly_obj.total_degree()
    except Exception:
        return SOSResult(
            verified=False,
            solver_status="error",
            message="Could not determine polynomial degree"
        )

    half_degree = (degree + 1) // 2

    # Modify polynomial for strict PD check
    if strict_pd:
        sum_squares = sum(s**degree for s in symbols)
        poly_check = poly_expanded - epsilon * sum_squares
    else:
        poly_check = poly_expanded

    # Build SOS constraints
    try:
        A, b = polynomial_to_gram_matrix_constraint(poly_check, symbols, half_degree)
    except Exception as e:
        return SOSResult(
            verified=False,
            solver_status="error",
            message=f"Constraint construction failed: {e}"
        )

    n_basis = len(get_monomial_basis(symbols, half_degree))

    # Define SDP problem
    Q = cp.Variable((n_basis, n_basis), symmetric=True)

    constraints = [
        Q >> 0,  # PSD constraint
        A @ cp.reshape(Q, (n_basis * n_basis,), order='F') == b  # Coefficient matching
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)

    requested_solver = _normalize_solver_name(solver)
    candidates: List[str]
    if strict_solver or not allow_solver_fallback:
        candidates = [requested_solver]
    else:
        if getattr(cp, requested_solver, None) is None:
            # Preserve legacy behavior: unknown solver names fallback directly to SCS.
            candidates = ["SCS"]
        elif requested_solver == "SCS":
            candidates = ["SCS"]
        else:
            candidates = [requested_solver, "SCS"]

    solver_name = ""
    solve_error = ""
    for candidate in candidates:
        solver_obj = getattr(cp, candidate, None)
        if solver_obj is None:
            if strict_solver:
                return SOSResult(
                    verified=False,
                    solver_status="strict_solver_error",
                    message=f"Strict solver mode: solver '{candidate}' unavailable",
                )
            continue
        try:
            prob.solve(solver=solver_obj, verbose=False)
            solver_name = candidate
            break
        except Exception as exc:
            solve_error = str(exc)
            if strict_solver:
                return SOSResult(
                    verified=False,
                    solver_status="strict_solver_error",
                    message=f"Strict solver mode: solver '{candidate}' failed: {exc}",
                )
            continue

    if not solver_name:
        return SOSResult(
            verified=False,
            solver_status="solver_error",
            message=(
                f"No solver succeeded (requested={requested_solver}, strict={strict_solver}, "
                f"allow_fallback={allow_solver_fallback and not strict_solver}). "
                f"Last error: {solve_error}"
            ),
        )

    if prob.status == cp.OPTIMAL:
        min_eig = np.linalg.eigvalsh(Q.value).min()
        return SOSResult(
            verified=True,
            solver_status=f"{prob.status}:{solver_name}",
            min_eigenvalue=min_eig,
            message=f"SOS verified, min eigenvalue: {min_eig:.6f}"
        )
    else:
        return SOSResult(
            verified=False,
            solver_status=f"{prob.status}:{solver_name}",
            message=f"SDP infeasible or failed: {prob.status}"
        )


def verify_gf_sos(
    gf: GeneralizedForm,
    strict_pd: bool = True,
    epsilon: float = 1e-6,
    solver: str = 'SCS',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
) -> SOSResult:
    """
    Verify GF positive definiteness using SOS.

    Checks all 2^n associated forms.

    Args:
        gf: Generalized Form to verify
        strict_pd: Whether to check strict positive definiteness
        epsilon: Strictness parameter
        solver: Preferred CVXPY solver name
        strict_solver: If True, fail instead of trying fallback solvers.
        allow_solver_fallback: If False, use preferred solver only.

    Returns:
        SOSResult (failure reports first failing octant)
    """
    n = gf.n_vars
    y_symbols = [sp.Symbol(f'y{i+1}', positive=True) for i in range(n)]

    associated = gf_to_associated_forms(gf)
    last_solver_status = "optimal"

    for signs in get_hyperoctant_signs(n):
        poly = associated[signs]
        result = verify_sos_cvxpy(
            poly,
            y_symbols,
            strict_pd,
            epsilon,
            solver=solver,
            strict_solver=strict_solver,
            allow_solver_fallback=allow_solver_fallback,
        )
        last_solver_status = result.solver_status

        if not result.verified:
            result.message = f"Octant {signs}: {result.message}"
            return result

    return SOSResult(
        verified=True,
        solver_status=last_solver_status,
        message="All octants verified via SOS"
    )


def get_symbol_exponent(term: sp.Expr, sym: sp.Symbol) -> int:
    """
    Extract the exponent of a symbol in a term, handling negative exponents.

    Args:
        term: A single term (monomial or product)
        sym: Symbol to find exponent of

    Returns:
        The exponent (can be negative for rational terms)
    """
    # Handle product terms
    if term.is_Mul:
        exp_total = 0
        for factor in term.args:
            if factor.is_Pow:
                base, exp = factor.as_base_exp()
                if base == sym:
                    exp_total += int(exp) if exp.is_integer else float(exp)
            elif factor == sym:
                exp_total += 1
            elif factor == 1/sym:
                exp_total -= 1
        return int(exp_total)
    elif term.is_Pow:
        base, exp = term.as_base_exp()
        if base == sym:
            return int(exp) if exp.is_integer else int(float(exp))
        return 0
    elif term == sym:
        return 1
    elif term == 1/sym:
        return -1
    else:
        return 0


def clear_denominators(
    expr: sp.Expr,
    symbols: List[sp.Symbol]
) -> Tuple[sp.Expr, bool]:
    """
    Clear denominators from a rational expression in y-space.

    When W = -dV/dt has negative y-exponents (e.g., y₁⁻²), the associated
    form becomes a rational function. This function multiplies by the
    appropriate yᵢ powers to make it a polynomial.

    The resulting polynomial p(y) satisfies:
        p(y) > 0 on positive orthant ⟺ original expr > 0 on positive orthant
    because the clearing factor (product of yᵢ powers) is always positive.

    Args:
        expr: SymPy expression (possibly rational)
        symbols: y-variables

    Returns:
        (polynomial, was_rational) where polynomial has no negative exponents
    """
    # Use together() to get a common denominator form
    expr_rational = sp.together(expr)

    # Get numerator and denominator
    numer, denom = sp.fraction(expr_rational)

    # If denominator is 1, no clearing needed
    if denom == 1:
        return sp.expand(numer), False

    # The denominator should be a product of y_i powers
    # We multiply by the denominator to clear it
    cleared = sp.expand(numer)

    return cleared, True


def verify_gf_sos_rational(
    gf: GeneralizedForm,
    strict_pd: bool = True,
    epsilon: float = 1e-6,
    solver: str = 'SCS',
    strict_solver: bool = False,
    allow_solver_fallback: bool = True,
) -> SOSResult:
    """
    Verify GF positive definiteness using SOS, handling rational forms.

    When the associated form has negative y-exponents (from differentiation),
    this function clears denominators to create a verifiable polynomial.

    This is essential for verifying W = -dV/dt which often has negative
    exponents from differentiating terms like |x|^(1/2).

    Args:
        gf: Generalized Form to verify (may have negative exponents producing rationals)
        strict_pd: Whether to check strict positive definiteness
        epsilon: Strictness parameter
        solver: Preferred CVXPY solver name
        strict_solver: If True, fail instead of trying fallback solvers.
        allow_solver_fallback: If False, use preferred solver only.

    Returns:
        SOSResult
    """
    n = gf.n_vars
    y_symbols = [sp.Symbol(f'y{i+1}', positive=True) for i in range(n)]

    associated = gf_to_associated_forms(gf)
    last_solver_status = "optimal"

    for signs in get_hyperoctant_signs(n):
        poly = associated[signs]

        # Clear any negative exponents
        poly_cleared, was_rational = clear_denominators(poly, y_symbols)

        if was_rational:
            # For rational forms, we need special handling
            # Check if the cleared polynomial is a valid polynomial
            try:
                poly_obj = sp.Poly(poly_cleared, y_symbols)
            except sp.PolynomialError as e:
                return SOSResult(
                    verified=False,
                    solver_status="error",
                    message=f"Octant {signs}: Could not convert to polynomial after clearing: {e}"
                )

        # Verify the (possibly cleared) polynomial
        result = verify_sos_cvxpy(
            poly_cleared,
            y_symbols,
            strict_pd,
            epsilon,
            solver=solver,
            strict_solver=strict_solver,
            allow_solver_fallback=allow_solver_fallback,
        )
        last_solver_status = result.solver_status

        if not result.verified:
            prefix = (
                f"Octant {signs} (denominator cleared): "
                if was_rational else f"Octant {signs}: "
            )
            result.message = prefix + result.message
            return result

    return SOSResult(
        verified=True,
        solver_status=last_solver_status,
        message="All octants verified via SOS (with denominator clearing where needed)"
    )


class SOSVerifier:
    """
    SOS verifier with configurable parameters.
    """

    def __init__(
        self,
        strict_pd: bool = True,
        epsilon: float = 1e-6,
        solver: str = 'SCS',
        handle_rationals: bool = True,
        strict_solver: bool = False,
        allow_solver_fallback: bool = True,
    ):
        """
        Initialize SOS verifier.

        Args:
            strict_pd: Verify strict positive definiteness
            epsilon: Strictness parameter
            solver: CVXPY solver to use
            handle_rationals: If True, clear denominators for rational forms
            strict_solver: If True, fail instead of trying fallback solvers.
            allow_solver_fallback: If False, use requested solver only.
        """
        self.strict_pd = strict_pd
        self.epsilon = epsilon
        self.solver = _normalize_solver_name(solver)
        self.handle_rationals = handle_rationals
        self.strict_solver = bool(strict_solver)
        self.allow_solver_fallback = bool(allow_solver_fallback and not self.strict_solver)

    def verify(self, gf: GeneralizedForm) -> SOSResult:
        """Verify GF via SOS."""
        if self.handle_rationals:
            return verify_gf_sos_rational(
                gf,
                self.strict_pd,
                self.epsilon,
                solver=self.solver,
                strict_solver=self.strict_solver,
                allow_solver_fallback=self.allow_solver_fallback,
            )
        else:
            return verify_gf_sos(
                gf,
                self.strict_pd,
                self.epsilon,
                solver=self.solver,
                strict_solver=self.strict_solver,
                allow_solver_fallback=self.allow_solver_fallback,
            )

    def is_available(self) -> bool:
        """Check if SOS verification is available."""
        return CVXPY_AVAILABLE
