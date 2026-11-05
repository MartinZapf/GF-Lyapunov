from __future__ import annotations

"""
Pólya Theorem Verification Module.

Implements Algorithm 5 from Sanchez-Moreno (2019) for verifying
positive definiteness of Generalized Forms.

Pólya's Theorem: A homogeneous polynomial p(y) is strictly positive
on the open positive orthant iff there exists p ≥ 0 such that all
coefficients of (y₁ + ... + yₙ)^p · p(y) are positive.

This is a complete decision procedure for positive definiteness of GFs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from fractions import Fraction
import numpy as np
import sympy as sp
from scipy.optimize import minimize

from gf_framework.algebra.core import GeneralizedForm, lcm_list
from gf_framework.algebra.associated import (
    gf_to_associated_forms,
    get_hyperoctant_signs,
    inverse_transform_point,
    extract_polynomial_coefficients,
)


@dataclass
class PolyaResult:
    """Result of Pólya verification."""
    verified: bool
    polya_p: int  # Exponent used (or max tried if failed)
    counterexample: Optional[np.ndarray] = None  # In x-space
    octant_results: Optional[Dict[Tuple[int, ...], Dict]] = None
    message: str = ""


def expand_polya_polynomial(
    poly: sp.Expr,
    y_symbols: List[sp.Symbol],
    p: int
) -> sp.Expr:
    """
    Expand (y₁ + ... + yₙ)^p · poly.

    Args:
        poly: Original polynomial
        y_symbols: Variables
        p: Pólya exponent

    Returns:
        Expanded polynomial
    """
    sum_y = sum(y_symbols)
    expanded = sp.expand(sum_y**p * poly)
    return expanded


# Typed statuses produced by :func:`analyze_closed_orthant_positivity`.
# They are strings so they survive into records and messages unchanged.
STATUS_OK = "closed_orthant_positive"
STATUS_NOT_POLYNOMIAL = "not_polynomial"
STATUS_ZERO_POLYNOMIAL = "zero_polynomial"
STATUS_NONPOSITIVE_COEFFICIENT = "nonpositive_coefficient"
STATUS_MISSING_PURE_POWER = "missing_pure_power"

# Sentinel monomial kept for backward compatibility with the pre-fix contract of
# ``check_all_coefficients_positive``: a non-polynomial input reports (-1,...,-1)
# with a NaN coefficient.
def _not_polynomial_marker(n: int) -> Tuple[Tuple[int, ...], float]:
    return ((-1,) * n, float('nan'))


def analyze_closed_orthant_positivity(
    poly: sp.Expr,
    symbols: List[sp.Symbol],
    tol: float = 1e-10
) -> Tuple[bool, str, Optional[Tuple[Tuple[int, ...], float]]]:
    """Decide strict positivity on the CLOSED non-negative orthant minus the origin.

    Fixed after the 2026-09-05 external scientific review, finding 1. The old
    test iterated only over the monomials PRESENT in the sparse polynomial, so a
    form that is positive on the OPEN orthant but vanishes on a face (a missing
    pure power) was accepted as positive definite. That accepted an unstable
    system through the public gate.

    Sound rule implemented here. Let ``F`` have all coefficients ``> tol``. Then
    ``F(y) > 0`` for every ``y >= 0`` with ``y != 0`` **iff** for every variable
    ``i`` there is a monomial supported on ``{i}`` alone (i.e. ``y_i^d`` for some
    ``d >= 0``, the constant monomial included) with a coefficient ``> tol``:

      * sufficiency — pick any ``i`` with ``y_i > 0``; that monomial contributes
        ``c*y_i^d > 0`` and every other monomial is ``>= 0``;
      * necessity — without it, ``F`` restricted to the ``y_i`` axis is
        identically zero, so ``F`` vanishes at the axis point, a vertex of the
        simplex.

    The rule is invariant under the Pólya multiplication: the coefficient of
    ``y_i^e`` in ``(y_1+...+y_n)^p F`` is the coefficient of ``y_i^{e-p}`` in
    ``F``, because only the pure ``y_i^p`` monomial of the Pólya factor can
    produce a pure ``y_i`` power. So proving the rule on the expanded polynomial
    proves it for ``F`` itself (``(sum y)^p > 0`` on the closed orthant minus the
    origin).

    Note on homogeneity: this framework picks ``mu_i`` as the per-variable LCM of
    the exponent denominators (``core.mu_exponents``), NOT proportional to the
    weights ``r``, so the associated forms are in general weighted-homogeneous
    but not standard-homogeneous (e.g. CTA at m=8, r=(3,2,1) gives
    ``y1^8 + y2^4 + ...``). Only the *sufficient* direction of Pólya is used
    here, and the rule above is valid for any polynomial with non-negative
    coefficients, homogeneous or not. What IS enforced is that the input is a
    genuine polynomial with non-negative integer exponents and is not identically
    zero.

    Returns:
        ``(ok, status, (failing_monomial, coefficient) or None)``.
    """
    n = len(symbols)
    coeffs = extract_polynomial_coefficients(poly, symbols)

    if len(coeffs) == 0:
        # Not a proper polynomial (fractional or negative y-exponents).
        # A bare constant still round-trips through sympy as {(0,...,0): c},
        # so reaching here really does mean "cannot be handled by Pólya".
        return False, STATUS_NOT_POLYNOMIAL, _not_polynomial_marker(n)

    # Defensive: reject anything whose exponents are not non-negative integers
    # of the right arity (Laurent / fractional monomials must never be treated
    # as polynomial data — cf. review finding 2 on the exact side).
    for monomial in coeffs:
        if len(monomial) != n or any((not isinstance(e, (int, np.integer))) or e < 0
                                     for e in monomial):
            return False, STATUS_NOT_POLYNOMIAL, _not_polynomial_marker(n)

    if all(abs(c) <= tol for c in coeffs.values()):
        # Identically zero (or numerically indistinguishable from it): positive
        # NOWHERE, so it can never certify positive definiteness.
        return False, STATUS_ZERO_POLYNOMIAL, ((0,) * n, 0.0)

    # (a) every coefficient that IS present must be strictly positive.
    for monomial in sorted(coeffs):
        coef = coeffs[monomial]
        if coef <= tol:
            return False, STATUS_NONPOSITIVE_COEFFICIENT, (monomial, coef)

    # (b) every variable must be covered by a strictly positive monomial that is
    #     supported on that variable alone (the constant monomial covers all).
    for i in range(n):
        covered = any(
            coef > tol and all(monomial[j] == 0 for j in range(n) if j != i)
            for monomial, coef in coeffs.items()
        )
        if not covered:
            # Witness: the i-th axis, where the form is identically zero.
            axis = tuple(1 if j == i else 0 for j in range(n))
            return False, STATUS_MISSING_PURE_POWER, (axis, 0.0)

    return True, STATUS_OK, None


def check_all_coefficients_positive(
    poly: sp.Expr,
    symbols: List[sp.Symbol],
    tol: float = 1e-10
) -> Tuple[bool, Optional[Tuple[Tuple[int, ...], float]]]:
    """
    Check that a polynomial is strictly positive on the CLOSED non-negative
    orthant minus the origin.

    Thin wrapper over :func:`analyze_closed_orthant_positivity`, kept so every
    caller of the historical name inherits the closed-orthant rule (regression
    from the 2026-09-05 external scientific review, finding 1). All present
    coefficients must be positive AND every variable must carry a strictly
    positive pure power; "all present coefficients positive" alone is NOT
    sufficient.

    Args:
        poly: Polynomial to check
        symbols: Variables
        tol: Tolerance for positivity

    Returns:
        (all_positive, (failing_monomial, coefficient) if not positive).
        A non-polynomial input reports the ``(-1, ..., -1)`` / NaN marker.
    """
    ok, _status, failure = analyze_closed_orthant_positivity(poly, symbols, tol)
    return ok, failure


def extract_counterexample_from_monomial(
    negative_monomial: Tuple[int, ...],
    signs: Tuple[int, ...],
    mu_exponents: Tuple[int, ...],
    poly: sp.Expr,
    y_symbols: List[sp.Symbol]
) -> np.ndarray:
    """
    Extract a numerical counterexample from a negative coefficient.

    When coefficient c_k of (y₁+...+yₙ)^p·F(y) is negative:
    1. c_k corresponds to monomial y₁^e₁...yₙ^eₙ
    2. Worst-case direction is along ray defined by exponents eᵢ
    3. Map back to x-space via inverse transform

    Args:
        negative_monomial: Exponent tuple (e₁, ..., eₙ) of failing monomial
        signs: Hyperoctant signs (γ₁, ..., γₙ)
        mu_exponents: Transform exponents (μ₁, ..., μₙ)
        poly: Original polynomial (without Pólya multiplication)
        y_symbols: SymPy symbols for y

    Returns:
        Counterexample point in x-space
    """
    n = len(signs)

    # Direction in y-space proportional to monomial exponents
    exps = np.array(negative_monomial, dtype=float)
    exp_sum = exps.sum()

    if exp_sum <= 0:
        # Constant term negative - use center of simplex
        y_direction = np.ones(n) / n
    else:
        # Normalize to unit simplex
        y_direction = exps / exp_sum

    # Find actual violation via line search on this ray
    poly_func = sp.lambdify(y_symbols, poly, 'numpy')

    def eval_on_ray(t):
        """Evaluate polynomial at t * y_direction (with small offset)."""
        y_point = t * y_direction + 0.01  # Small offset from boundary
        return poly_func(*y_point)

    # Search for point with small or negative value
    best_t = 1.0
    best_val = float('inf')

    for t in np.linspace(0.1, 5.0, 50):
        val = eval_on_ray(t)
        if val < best_val:
            best_val = val
            best_t = t

    # Refine with local optimization
    try:
        result = minimize(lambda t: eval_on_ray(t[0]), [best_t],
                         bounds=[(0.01, 10.0)], method='L-BFGS-B')
        if result.fun < best_val:
            best_t = result.x[0]
    except Exception:
        pass  # Use grid search result

    y_ce = best_t * y_direction + 0.01

    # Transform back to x-space: xᵢ = γᵢ · yᵢ^μᵢ
    x_ce = inverse_transform_point(y_ce, signs, mu_exponents)

    return x_ce


def verify_single_octant(
    poly: sp.Expr,
    y_symbols: List[sp.Symbol],
    signs: Tuple[int, ...],
    mu_exponents: Tuple[int, ...],
    max_p: int = 15,
    tol: float = 1e-10
) -> Tuple[bool, int, Optional[np.ndarray], str]:
    """
    Verify positive definiteness in a single hyperoctant using Pólya.

    Args:
        poly: Associated polynomial form
        y_symbols: SymPy symbols
        signs: Hyperoctant signs
        mu_exponents: Transform exponents
        max_p: Maximum Pólya exponent to try
        tol: Coefficient positivity tolerance

    Returns:
        (verified, p_used, counterexample_if_failed, message)

    The acceptance rule is strict positivity on the CLOSED orthant (review
    2026-09-05, finding 1): all coefficients positive AND every variable covered
    by a strictly positive pure power. Structural preconditions (genuine
    polynomial, not identically zero, pure-power coverage) are decided ONCE on
    the base form, because none of them can be repaired by the Pólya factor.
    """
    n = len(y_symbols)

    base_ok, base_status, base_failure = analyze_closed_orthant_positivity(
        poly, y_symbols, tol
    )

    if base_status == STATUS_NOT_POLYNOMIAL:
        return False, 0, None, "not_polynomial"

    if base_status == STATUS_ZERO_POLYNOMIAL:
        # W ≡ 0 (or V ≡ 0) on this octant: never a positivity certificate.
        y_ce = np.ones(n)
        return False, 0, inverse_transform_point(y_ce, signs, mu_exponents), "zero_polynomial"

    if base_status == STATUS_MISSING_PURE_POWER:
        # Multiplying by (y₁+...+yₙ)^p shifts pure powers but cannot create them,
        # so this is decided for every p. The witness is the axis point itself,
        # where the form is identically zero.
        axis, _ = base_failure
        y_ce = np.array([1.0 if e else 0.0 for e in axis])
        return (
            False,
            0,
            inverse_transform_point(y_ce, signs, mu_exponents),
            "missing_pure_power",
        )

    if base_ok:
        return True, 0, None, "verified"

    expanded = poly
    for p in range(0, max_p + 1):
        expanded = expand_polya_polynomial(poly, y_symbols, p)
        is_positive, status, failure_info = analyze_closed_orthant_positivity(
            expanded, y_symbols, tol
        )

        if is_positive:
            return True, p, None, "verified"

        if status == STATUS_NOT_POLYNOMIAL:
            return False, p, None, "not_polynomial"

    # Failed - extract counterexample from last attempt
    _, status, failure_info = analyze_closed_orthant_positivity(expanded, y_symbols, tol)
    if failure_info is None:
        return False, max_p, None, "unknown_failure"

    neg_mono, _ = failure_info
    # Skip counterexample extraction for non-polynomial case
    if any(m == -1 for m in neg_mono):
        return False, max_p, None, "not_polynomial"

    ce = extract_counterexample_from_monomial(
        neg_mono, signs, mu_exponents, poly, y_symbols
    )

    return False, max_p, ce, "negative_coefficient"


def verify_gf_polya(
    gf: GeneralizedForm,
    max_p: int = 15,
    tol: float = 1e-10
) -> PolyaResult:
    """
    Verify positive definiteness of a GF using Pólya's theorem.

    This is the main verification function that checks all 2^n octants.

    Args:
        gf: Generalized Form to verify
        max_p: Maximum Pólya exponent
        tol: Coefficient tolerance

    Returns:
        PolyaResult with verification status and details
    """
    n = gf.n_vars
    y_symbols = [sp.Symbol(f'y{i+1}', positive=True) for i in range(n)]
    mu = gf.mu_exponents

    # A GF with no terms is identically zero; it can never be positive definite.
    # (Regression from the 2026-09-05 external scientific review, finding 1.)
    if not gf.terms:
        return PolyaResult(
            verified=False,
            polya_p=0,
            counterexample=None,
            octant_results={},
            message="Cannot verify: GF has no terms (identically zero)",
        )

    # Get associated forms for all octants
    associated = gf_to_associated_forms(gf)

    octant_results = {}
    max_p_needed = 0
    all_verified = True

    for signs in get_hyperoctant_signs(n):
        poly = associated[signs]

        verified, p_used, ce, msg = verify_single_octant(
            poly, y_symbols, signs, mu, max_p, tol
        )

        octant_results[signs] = {
            'verified': verified,
            'p': p_used,
            'status': msg,
        }

        if verified:
            max_p_needed = max(max_p_needed, p_used)
        else:
            all_verified = False
            # Provide informative message based on failure type
            if msg == "not_polynomial":
                fail_msg = (f"Cannot verify: associated form in octant {signs} has "
                           f"fractional exponents (not a polynomial). "
                           f"Choose a degree m where all y-exponents are integers.")
            elif msg == "zero_polynomial":
                fail_msg = (f"Cannot verify: associated form in octant {signs} is "
                           f"identically zero, so it is positive nowhere.")
            elif msg == "missing_pure_power":
                fail_msg = (f"Not positive definite: associated form in octant "
                           f"{signs} has no strictly positive pure power in every "
                           f"variable, so it vanishes on a face of the closed "
                           f"orthant (axis witness returned).")
            else:
                fail_msg = f"Failed in octant {signs} after p={max_p}"

            return PolyaResult(
                verified=False,
                polya_p=max_p,
                counterexample=ce,
                octant_results=octant_results,
                message=fail_msg
            )

    return PolyaResult(
        verified=True,
        polya_p=max_p_needed,
        counterexample=None,
        octant_results=octant_results,
        message=f"Verified with Pólya p={max_p_needed}"
    )


def verify_lyapunov_decrease_polya(
    V: GeneralizedForm,
    W: sp.Expr,  # W = -dV/dt (should be positive definite)
    y_symbols: List[sp.Symbol],
    max_p: int = 15,
    tol: float = 1e-10
) -> PolyaResult:
    """
    Verify that -dV/dt is positive definite (Lyapunov decrease condition).

    For this, we need W = -dV/dt as a GF. If W is given as a SymPy
    expression, we convert it to associated forms and verify each.

    Note: This is a simplified version. Full implementation would
    construct W as a GF and use verify_gf_polya.

    Args:
        V: Lyapunov candidate (for structure info)
        W: Negative Lie derivative expression
        y_symbols: SymPy symbols
        max_p: Maximum Pólya exponent
        tol: Coefficient tolerance

    Returns:
        PolyaResult for the decrease condition
    """
    # For now, verify W as a polynomial directly
    # Full implementation would construct W as GF

    n = len(y_symbols)
    mu = V.mu_exponents

    octant_results = {}
    max_p_needed = 0

    for signs in get_hyperoctant_signs(n):
        # W should already be in polynomial form for each octant
        # This is a simplified version - full implementation needs
        # proper W construction per octant

        verified, p_used, ce, msg = verify_single_octant(
            W, y_symbols, signs, mu, max_p, tol
        )

        # Same closed-orthant rule as verify_gf_polya: verify_single_octant now
        # refuses forms that vanish on a face (review 2026-09-05, finding 1).
        octant_results[signs] = {'verified': verified, 'p': p_used, 'status': msg}

        if verified:
            max_p_needed = max(max_p_needed, p_used)
        else:
            return PolyaResult(
                verified=False,
                polya_p=max_p,
                counterexample=ce,
                octant_results=octant_results,
                message=f"Decrease condition failed in octant {signs} [{msg}]"
            )

    return PolyaResult(
        verified=True,
        polya_p=max_p_needed,
        counterexample=None,
        octant_results=octant_results,
        message=f"Decrease verified with Pólya p={max_p_needed}"
    )


class PolyaVerifier:
    """
    Main Pólya verifier class with caching and batch operations.
    """

    def __init__(self, max_p: int = 15, tol: float = 1e-10):
        """
        Initialize verifier.

        Args:
            max_p: Maximum Pólya exponent
            tol: Coefficient tolerance
        """
        self.max_p = max_p
        self.tol = tol

    def verify(self, gf: GeneralizedForm) -> PolyaResult:
        """
        Verify positive definiteness of a GF.

        Args:
            gf: Generalized Form to verify

        Returns:
            PolyaResult
        """
        return verify_gf_polya(gf, self.max_p, self.tol)

    def verify_lyapunov_conditions(
        self,
        V: GeneralizedForm,
        W: GeneralizedForm
    ) -> Tuple[PolyaResult, PolyaResult]:
        """
        Verify both Lyapunov conditions: V > 0 and W = -dV/dt > 0.

        This is the COMPLETE verification for a Lyapunov function.
        Both conditions must pass for a valid Lyapunov function.

        Args:
            V: Lyapunov candidate (should be positive definite)
            W: Negative Lie derivative W = -dV/dt (should be positive definite)

        Returns:
            (V_result, W_result) tuple of PolyaResult
        """
        # Verify V > 0
        V_result = self.verify(V)

        if not V_result.verified:
            # No point checking W if V not positive definite
            return V_result, PolyaResult(
                verified=False,
                polya_p=0,
                message="Skipped W verification: V not positive definite"
            )

        # Verify W > 0
        W_result = self.verify(W)

        return V_result, W_result

    def verify_complete(
        self,
        V: GeneralizedForm,
        W: GeneralizedForm
    ) -> bool:
        """
        Check if BOTH Lyapunov conditions are verified.

        Convenience method that returns True only if both V and W
        are verified as positive definite via Pólya.

        Args:
            V: Lyapunov candidate
            W: Negative Lie derivative W = -dV/dt

        Returns:
            True if both V > 0 and W > 0 verified, False otherwise
        """
        V_result, W_result = self.verify_lyapunov_conditions(V, W)
        return V_result.verified and W_result.verified


def sample_verification_points(
    gf: GeneralizedForm,
    n_samples: int = 10000,
    domain_scale: float = 2.0
) -> Tuple[bool, float, Optional[np.ndarray]]:
    """
    Quick sampling-based verification (pre-check before Pólya).

    Args:
        gf: GF to check
        n_samples: Number of random samples
        domain_scale: Scale of sampling domain

    Returns:
        (passes, min_value, worst_point)
    """
    n = gf.n_vars

    # Sample on unit sphere with weighted norm, then scale
    samples = np.random.randn(n_samples, n)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / norms  # Unit sphere

    # Scale by weights
    for i in range(n):
        samples[:, i] *= domain_scale ** (1.0 / gf.weights_r[i])

    # Evaluate
    values = gf.evaluate(samples)
    min_idx = np.argmin(values)
    min_val = values[min_idx]

    if min_val <= 0:
        return False, min_val, samples[min_idx]

    return True, min_val, None


@dataclass
class VerificationLevel:
    """Describes the rigor level achieved in verification."""
    ANALYTICAL_POLYA = "analytical_polya"      # Pólya theorem - full proof
    ANALYTICAL_SOS = "analytical_sos"          # SOS decomposition - full proof
    NUMERICAL_HIGH_CONF = "numerical_high"     # 50k+ samples, bounded domain
    NUMERICAL_LOW_CONF = "numerical_low"       # Fewer samples or issues
    FAILED = "failed"                          # Could not verify


def verify_w_cascade(
    W: GeneralizedForm,
    max_polya_p: int = 15,
    n_numerical_samples: int = 50000,
    domain_scales: List[float] = [0.1, 0.5, 1.0, 2.0],
    numerical_margin: float = 1e-6
) -> Tuple[PolyaResult, str]:
    """
    Comprehensive W verification with fallback cascade.

    Tries verification methods in order of rigor:
    1. Pólya theorem (if W has polynomial associated forms)
    2. SOS with denominator clearing (for rational forms)
    3. Numerical sampling (high-confidence fallback)

    Args:
        W: The GF W = -dV/dt to verify
        max_polya_p: Maximum Pólya exponent
        n_numerical_samples: Samples per scale for numerical check
        domain_scales: Scales for numerical verification
        numerical_margin: Threshold for numerical positivity

    Returns:
        (PolyaResult, verification_level) tuple
    """
    # Try Pólya first (fastest, rigorous)
    polya_result = verify_gf_polya(W, max_p=max_polya_p)

    if polya_result.verified:
        return polya_result, VerificationLevel.ANALYTICAL_POLYA

    # Structural refutation on the closed orthant: the form vanishes on a face
    # (or is identically zero). Random sampling never hits that face, so the
    # numerical fallback below must NOT be allowed to overrule it.
    # (Regression from the 2026-09-05 external scientific review, finding 1.)
    octant_statuses = {
        str(info.get('status', ''))
        for info in (polya_result.octant_results or {}).values()
    }
    if octant_statuses & {"missing_pure_power", "zero_polynomial"}:
        return polya_result, VerificationLevel.FAILED

    # Check if failure was due to non-polynomial (fractional/negative exponents)
    is_rational = ("fractional" in polya_result.message.lower() or
                   "not a polynomial" in polya_result.message.lower())

    if is_rational:
        # Try SOS with denominator clearing
        try:
            from .sos import verify_gf_sos_rational, CVXPY_AVAILABLE
            if CVXPY_AVAILABLE:
                sos_result = verify_gf_sos_rational(W)
                if sos_result.verified:
                    return PolyaResult(
                        verified=True,
                        polya_p=-2,  # -2 indicates SOS verification
                        message=f"SOS verified (with denominator clearing): {sos_result.message}"
                    ), VerificationLevel.ANALYTICAL_SOS
        except ImportError:
            pass

    # Fall back to numerical verification
    numerical_result = verify_w_numerical(
        W, n_numerical_samples, domain_scales, numerical_margin
    )

    if numerical_result.verified:
        return numerical_result, VerificationLevel.NUMERICAL_HIGH_CONF
    else:
        return numerical_result, VerificationLevel.FAILED


def verify_w_numerical(
    W: GeneralizedForm,
    n_samples: int = 50000,
    domain_scales: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    margin: float = 1e-6
) -> PolyaResult:
    """
    Numerical verification of W = -dV/dt > 0 as fallback when Pólya fails.

    This is used when W has fractional y-exponents (not a polynomial in y-space)
    which makes Pólya verification inapplicable.

    WARNING: Numerical verification is NOT a formal proof. It only provides
    high-confidence evidence of positivity. Use only when algebraic verification
    is mathematically impossible.

    Args:
        W: The GF W = -dV/dt to verify
        n_samples: Number of samples per scale
        domain_scales: Different scales to test (homogeneity helps)
        margin: Minimum acceptable W value

    Returns:
        PolyaResult with numerical verification details
    """
    n = W.n_vars

    min_val_overall = float('inf')
    worst_point = None

    for scale in domain_scales:
        # Sample uniformly in scaled box
        samples = np.random.randn(n_samples, n)
        for i in range(n):
            samples[:, i] *= scale ** (1.0 / W.weights_r[i])

        # Evaluate W
        values = W.evaluate(samples)
        min_val = values.min()
        min_idx = np.argmin(values)

        if min_val < min_val_overall:
            min_val_overall = min_val
            worst_point = samples[min_idx]

        if min_val <= -margin:
            return PolyaResult(
                verified=False,
                polya_p=-1,  # Indicates numerical verification
                counterexample=worst_point,
                message=f"Numerical: W < 0 detected (min={min_val:.6f} at scale {scale})"
            )

    # All samples passed
    if min_val_overall > margin:
        return PolyaResult(
            verified=True,
            polya_p=-1,  # Indicates numerical verification
            counterexample=None,
            message=f"Numerical: W > 0 on {n_samples * len(domain_scales)} samples (min={min_val_overall:.6f})"
        )
    else:
        # Close to zero - inconclusive
        return PolyaResult(
            verified=False,
            polya_p=-1,
            counterexample=worst_point,
            message=f"Numerical: W borderline (min={min_val_overall:.6f}), cannot confirm positivity"
        )
