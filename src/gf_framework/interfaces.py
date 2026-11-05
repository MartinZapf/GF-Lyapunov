from __future__ import annotations

"""
Shared dataclass interfaces of the GF-SOS Lyapunov discovery framework.

1. Generation - valid GF term bases (SystemSpec, TermBasis)
2. Candidates - a designed Lyapunov candidate handed to verification
   (SelectionResult)
3. Verification - rigorous analytical certification via SOS/Pólya
   (VerificationConfig, Certificate)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Import existing types for compatibility
from gf_framework.algebra.core import GeneralizedForm
from gf_framework.algebra.enumerate import TermSpec


# =============================================================================
# MODULE 1: GENERATION INTERFACES
# =============================================================================

@dataclass
class SystemSpec:
    """
    Input specification for term generation.

    Defines the r-homogeneous system structure including weights,
    Lyapunov degree, and dynamics in GF form.

    Attributes:
        weights_r: Homogeneity weights, e.g., (2, 1) for STA, (3, 2, 1) for PID-SMC
        degree_m: Lyapunov function degree (must satisfy compatibility constraints)
        dynamics_gf: System dynamics as list of GeneralizedForm [f_1, ..., f_n]
        dynamics_gf_vertices: Optional extremal dynamics for robust certification
            under a bounded matched disturbance. Each entry is a full dynamics
            list [f_1, ..., f_n] (a vertex of the differential inclusion, e.g.
            the disturbance endpoints ±δ₀). W = -V̇ is verified for EVERY vertex;
            since W is affine in the disturbance, endpoint positivity covers the
            whole interval by convexity. None ⇒ nominal only ([dynamics_gf]).
        mu_policy: μ exponent policy ('per_variable' or 'global_lcm')
        ensure_even_exponents: If True, doubles μ values for SOS-even exponents
    """
    weights_r: Tuple[int, ...]
    degree_m: int
    dynamics_gf: Optional[List[GeneralizedForm]] = None
    dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]] = None
    mu_policy: str = 'per_variable'
    ensure_even_exponents: bool = False

    @property
    def n_vars(self) -> int:
        """Number of state variables."""
        return len(self.weights_r)

    def __post_init__(self):
        """Validate specification."""
        if self.degree_m <= 0:
            raise ValueError(f"degree_m must be positive, got {self.degree_m}")
        if not all(r > 0 for r in self.weights_r):
            raise ValueError(f"All weights must be positive, got {self.weights_r}")
        if self.mu_policy not in {'per_variable', 'global_lcm'}:
            raise ValueError(
                f"mu_policy must be 'per_variable' or 'global_lcm', got {self.mu_policy}"
            )


@dataclass
class TermBasis:
    """
    Output of Generation Module: Complete basis of valid GF terms.

    Contains all terms satisfying the homogeneity constraint Σrᵢρᵢ = m,
    separated into diagonal (single-variable) and cross (multi-variable) terms.

    Attributes:
        diagonal_terms: Pure diagonal terms |xᵢ|^(m/rᵢ)
        cross_terms: Mixed terms involving products of multiple variables
        mu_exponents: y-space lifting exponents for Lemma 6 transform
        system_spec: Reference back to input specification
        mu_policy: Policy used to compute mu_exponents
        ensure_even_exponents: Whether even-exponent enforcement was applied
    """
    diagonal_terms: List[TermSpec]
    cross_terms: List[TermSpec]
    mu_exponents: Tuple[int, ...]
    system_spec: SystemSpec
    mu_policy: str = 'per_variable'
    ensure_even_exponents: bool = False

    @property
    def all_terms(self) -> List[TermSpec]:
        """Get all terms (diagonal + cross)."""
        return self.diagonal_terms + self.cross_terms

    @property
    def n_diagonal(self) -> int:
        """Number of diagonal terms."""
        return len(self.diagonal_terms)

    @property
    def n_cross(self) -> int:
        """Number of cross terms."""
        return len(self.cross_terms)

    @property
    def n_total(self) -> int:
        """Total number of terms."""
        return self.n_diagonal + self.n_cross


# =============================================================================
# MODULE 2: CANDIDATE INTERFACES
# =============================================================================

@dataclass
class SelectionResult:
    """
    A designed Lyapunov candidate handed to the Verification Module.

    Attributes:
        active_terms: Terms with |coef| > threshold
        coefficients: Mapping from term index to coefficient value
        V_gf: Explicit Lyapunov candidate as GeneralizedForm
        numerical_valid: True if V > 0 and W > 0 on validation grid
        max_V_violation: max(-V) on grid (should be < 0 for valid V)
        max_W_violation: max(dV/dt) on grid (should be < 0 for valid W)
    """
    active_terms: List[TermSpec]
    coefficients: Dict[int, float]
    V_gf: GeneralizedForm
    numerical_valid: bool
    max_V_violation: float
    max_W_violation: float

    @property
    def n_active(self) -> int:
        """Number of active terms."""
        return len(self.active_terms)

    @property
    def sparsity_ratio(self) -> float:
        """Ratio of active to total terms."""
        total = len(self.coefficients)
        return self.n_active / total if total > 0 else 0.0


# =============================================================================
# MODULE 3: VERIFICATION INTERFACES
# =============================================================================

@dataclass
class VerificationConfig:
    """
    Configuration for SOS/Pólya verification.

    Controls the rigorous analytical verification process.

    Attributes:
        epsilon: Strictness margin for SOS (V > ε||y||^d)
        solver: SDP solver to use ('MOSEK', 'SCS', 'CLARABEL', etc.)
        timeout: Solver timeout in seconds
        max_polya_p: Maximum Pólya exponent to try
        polya_tol: Tolerance for Pólya coefficient positivity
        use_weighted_sphere: Use weighted-sphere SOS for W with negative exponents

    Note: Default is MOSEK (fast, accurate). Falls back to SCS if MOSEK unavailable
    unless strict_solver=True.
    Install MOSEK and set MOSEKLM_LICENSE_FILE for best performance.
    """
    epsilon: float = 1e-6
    solver: str = 'MOSEK'  # Prefer MOSEK, fallback to SCS if unavailable
    strict_solver: bool = False
    allow_solver_fallback: bool = True
    timeout: float = 300.0
    max_polya_p: int = 15
    polya_tol: float = 1e-10
    use_weighted_sphere: bool = True

    def __post_init__(self):
        """Normalize solver policy flags."""
        self.solver = str(self.solver).strip().upper() if str(self.solver).strip() else "MOSEK"
        if self.strict_solver:
            self.allow_solver_fallback = False


@dataclass
class Certificate:
    """
    Output of Verification Module: Rigorous verification certificate.

    Provides formal proof of V > 0 and W = -dV/dt > 0 via SOS or Pólya theorem.

    IMPORTANT: This framework requires FORMAL certificates only. Numerical
    verification is NOT accepted - if SOS/Pólya fail, verification fails.

    Attributes:
        V_verified: True if V > 0 formally proven via SOS/Pólya
        W_verified: True if W > 0 formally proven via SOS/Pólya
        V_method: Verification method used for V ('sos', 'polya', 'sos_weighted_sphere', 'failed')
        W_method: Verification method used for W ('sos', 'polya', 'sos_weighted_sphere', 'sos_failed', 'skipped')
        V_gf: Final certified Lyapunov function
        W_gf: Computed W = -dV/dt (if available)
        solver_status: SDP solver status string
        solver_requested: Requested SDP solver name
        solver_used: Solver name actually used for certification (if any)
        solver_fallback_used: True if certification used a solver fallback path
        solver_attempts: Ordered solver attempt trace
        clearing_power: k used for ∏ᵢ|xᵢ|^{k·ρ/rᵢ} clearing monomial (0 if none)
        polya_p_V: Pólya exponent used for V (if polya method)
        polya_p_W: Pólya exponent used for W (if polya method)
        message: Human-readable status message
    """
    V_verified: bool
    W_verified: bool
    V_method: str
    W_method: str
    V_gf: Optional[GeneralizedForm] = None
    W_gf: Optional[GeneralizedForm] = None
    solver_status: str = ""
    solver_requested: str = ""
    solver_used: str = ""
    solver_fallback_used: bool = False
    solver_attempts: List[str] = field(default_factory=list)
    # A-posteriori SOS Gram diagnostics (worst case across the V and W stages):
    # smallest Gram eigenvalue, largest ℓ1 equality residual, largest Gram
    # condition number λ_max/λ_min (None if a Gram was not strictly PD), and the
    # smallest strictness margin ε actually used.
    # See joint_sos._validate_gram_certificate.
    gram_min_eig: Optional[float] = None
    gram_residual_l1: Optional[float] = None
    gram_cond: Optional[float] = None
    epsilon_used: Optional[str] = None
    # Robust certification: number of differential-inclusion vertices for which
    # W > 0 was verified (1 = nominal only), and the disturbance bound the
    # vertices encode (informational; the vertices themselves are the proof).
    n_vertices: int = 1
    disturbance_bound: Optional[float] = None
    clearing_power: int = 0
    polya_p_V: Optional[int] = None
    polya_p_W: Optional[int] = None
    message: str = ""

    # Convexity / quasiconvexity certificate (independent from stability claim)
    convexity_certificate_level: str = "none"  # none | tangential_sos | hessian_psd_sos
    convexity_certificate_scope: str = "none"  # none | formal_open_orthant | ...
    convexity_certificate_method: str = ""
    convexity_certificate_message: str = ""

    # Clarke condition verification (coordinate hyperplane check)
    clarke_verified: bool = False

    # Regularity of V and the resulting switching-surface scope (honest reporting),
    # via the Sanchez-Moreno continuous-time bridge — Lemma 5 (Filippov), NOT Clarke.
    # regularity: C1 | lipschitz_kink | sub_lipschitz | discontinuous
    # surface_scope:
    #   lemma5_filippov     — V ∈ C¹ (R2) and V>0, W>0 certified ⇒ asymptotic
    #                         stability by Lemma 5; surfaces covered (no Clarke).
    #   non_c1_out_of_scope — V not C¹; outside the continuous-time method (would
    #                         need non-smooth/discretization analysis, out of scope).
    #   open_orthant_only   — V or W not certified; nothing claimed at surfaces.
    regularity: str = ""
    surface_certified: bool = False
    surface_scope: str = "open_orthant_only"

    # Typed failure classes (WP P3-02). Before P3-02 every rejection collapsed
    # to the literal V_method/W_method "failed", so solver infeasibility, a
    # solver "unknown", a raised exception and an a-posteriori Gram rejection
    # were indistinguishable in the record. The vocabulary lives in
    # verification/failure_typing.py (FailureClass); "none" means the stage
    # succeeded and "" means the field was never populated (an old record).
    # Claim hygiene (D6): solver_infeasible is typed FLOAT evidence, never proof.
    V_failure_class: str = ""
    W_failure_class: str = ""
    failure_class: str = ""
    # One entry per verification stage (V, then W per inclusion vertex), each a
    # JSON-native dict from failure_typing.stage_report: method, failure class,
    # solver status/attempts, the stage's worst-case Gram diagnostics, any
    # exception type/message, and PER-OCTANT gram_min_eig / gram_residual_l1
    # parsed back out of the attempt trace. None on records written before P3-02.
    stage_reports: Optional[List[dict]] = None

    @property
    def fully_verified(self) -> bool:
        """True if both V and W are verified."""
        return self.V_verified and self.W_verified

    @property
    def method_summary(self) -> str:
        """Summary of verification methods used."""
        return f"V: {self.V_method}, W: {self.W_method}"
