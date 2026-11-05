"""Formal verification: the honest gate, joint SOS design, exact re-proof, Pólya.

The certificate path is ``verifier.certify_discovered_gf`` → ``joint_sos``
(CVXPY or the ``mosek_direct`` Task-API branch) → ``_validate_gram_certificate``,
with ``sphere_check.min_V_on_sphere`` as the necessary PD pre-check and
``exact_gate`` / ``exact_phase`` as the OFFLINE exact-rational re-proof
(Prop. 7). Everything re-exported below is live.

Not exported here: the modules under ``gf_framework._archive``, including
``clarke.py`` (out of scope, see ``AGENTS.md``; archived 2026-09-23).
"""

from __future__ import annotations

from .polya import (
    PolyaResult,
    PolyaVerifier,
    verify_gf_polya,
)

from .joint_sos import (
    JointSOSResult,
    JointSOSOptimizer,
    verify_gf_sos_full,
    verify_sos_polynomial,
    compute_mu_exponents,
    analyze_w_exponents,
)

from .sos import (
    SOSResult,
    SOSVerifier,
    verify_gf_sos,
)

from .verifier import (
    certify_discovered_gf,
    verify_lyapunov,
    verify_v_only,
    quick_verify,
)

from .sphere_check import (
    SphereCheck,
    min_V_on_sphere,
)

from .mosek_direct import (
    DirectSolveOutcome,
    GramBlock,
    GramRow,
    solve_joint_sos_direct,
)

from .exact_gate import (
    ExactCaptureResult,
    ExactGateError,
    ExactOctantCheck,
    exact_neg_lie_derivative,
    exact_terms_from_gf,
    exact_validate_octant,
    exact_verify_captures,
)

from .exact_phase import (
    ExactPhaseError,
    ExactPhaseResult,
    run_exact_phase,
)

from .mu_guard import (
    MU_GUARD_MODES,
    MuCheck,
    check_mu_preservation,
    is_proportional_to_r,
    v_only_mu,
)

from .sparse_sos import (
    BACKWARD_OUTCOMES,
    BackwardEliminationResult,
    BackwardStep,
    ChargeProbe,
    SparseDesignConfig,
    SparseDesignResult,
    atom_scales,
    backward_eliminate,
    design_gram_charge_probe,
    design_sparse_lyapunov,
)

# Test-only. `admissibility.py` has no importer outside `tests/`; it validates
# representational assumptions and never produces a certificate. Kept exported
# for backwards compatibility, not part of the certificate path.
from .admissibility import (
    validate_dynamics_gf_admissibility,
    validate_system_spec_admissibility,
)

__all__ = [
    # Pólya
    "PolyaResult",
    "PolyaVerifier",
    "verify_gf_polya",
    # Joint SOS
    "JointSOSResult",
    "JointSOSOptimizer",
    "verify_gf_sos_full",
    "verify_sos_polynomial",
    "compute_mu_exponents",
    "analyze_w_exponents",
    # SOS
    "SOSResult",
    "SOSVerifier",
    "verify_gf_sos",
    # Verifier orchestrator — THE HONEST GATE
    "certify_discovered_gf",
    "verify_lyapunov",
    "verify_v_only",
    "quick_verify",
    # Sphere pre-check
    "SphereCheck",
    "min_V_on_sphere",
    # MOSEK Task-API direct design path
    "DirectSolveOutcome",
    "GramBlock",
    "GramRow",
    "solve_joint_sos_direct",
    # Exact-rational offline re-proof (Prop. 7)
    "ExactCaptureResult",
    "ExactGateError",
    "ExactOctantCheck",
    "exact_neg_lie_derivative",
    "exact_terms_from_gf",
    "exact_validate_octant",
    "exact_verify_captures",
    "ExactPhaseError",
    "ExactPhaseResult",
    "run_exact_phase",
    # μ-preservation guard (sparse supports vs the V-only μ path)
    "MU_GUARD_MODES",
    "MuCheck",
    "check_mu_preservation",
    "is_proportional_to_r",
    "v_only_mu",
    # Sparse SOS (reweighted ℓ₁) + the optional backward-elimination pass
    "BACKWARD_OUTCOMES",
    "BackwardEliminationResult",
    "BackwardStep",
    "ChargeProbe",
    "SparseDesignConfig",
    "SparseDesignResult",
    "atom_scales",
    "backward_eliminate",
    "design_gram_charge_probe",
    "design_sparse_lyapunov",
    # Admissibility (test-only, not on the certificate path)
    "validate_dynamics_gf_admissibility",
    "validate_system_spec_admissibility",
]
