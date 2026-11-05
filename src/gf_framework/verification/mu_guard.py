"""μ-preservation guard for sparsified GF supports.

Why this exists. The designer computes μ from the V library **and the
dynamics** (``compute_mu_exponents_with_dynamics``), which forces ``μ = k·r``
and therefore homogeneous associated forms. Every downstream V-only path
recomputes μ from the V terms alone (``compute_mu_exponents``, used by
``verify_gf_sos_full`` and by the exact gate's target assembly), where μᵢ is
just twice the LCM of the denominators present on axis i. Those two numbers
agree on a dense library and can disagree after sparsification: dropping the
terms that pin one axis's denominator lowers that μᵢ.

Measured instance (``docs/FINDINGS_2026-09-02.md`` section 5): pidsmc at
r = (3,2,1), m = 6. Dense 28-term support has V-only μ = (6,4,2), equal to the
design μ. The reweighted-ℓ₁ 9-term support drops every denominator-2 exponent
on the x₂ axis, so V-only μ becomes **(6,2,2)**, which is not proportional to
r. The associated form is then not homogeneous, and ``exact_octant_target``
refuses with ``target not homogeneous`` (``exact_gate.py:215-217``), typed as
``target_assembly_failed``. The float path tolerates it because it can fall
back to a non-homogeneous monomial basis; Proposition 7 cannot.

The guard is cheap and exact: integer arithmetic on exponent denominators, no
solve. It converts a silent exact-phase failure into a design-time refusal with
a typed reason.

Two rules, both checked:

1. **Preservation.** The candidate's V-only μ must equal the reference (the
   V-only μ of the dense library the sparsification started from).
2. **Homogeneity.** The candidate's V-only μ must be proportional to the
   homogeneity weights r. This is the condition the exact gate actually needs,
   and it is checked independently of rule 1 so a reference that is itself
   defective cannot launder a candidate through.

The guard makes no stability claim. It only refuses to hand the gate an object
whose exact re-proof is structurally impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence, Tuple

# How a caller reacts to a μ-breaking support. "reject" is the default: the
# support is skipped exactly like a gate rejection, and the ladder walks on.
MU_GUARD_MODES = ("reject", "warn", "off")

# Typed reasons. Empty string means the check passed.
MU_OK = ""
MU_NOT_PRESERVED = "mu_not_preserved"
MU_NOT_PROPORTIONAL = "mu_not_proportional_to_r"
MU_NOT_PRESERVED_NONHOMOGENEOUS = "mu_not_preserved_nonhomogeneous"


@dataclass(frozen=True)
class MuCheck:
    """Typed outcome of the μ-preservation check on one candidate support."""
    ok: bool
    reason: str
    mu_reference: Tuple[int, ...]
    mu_candidate: Tuple[int, ...]
    proportional_to_r: bool
    reference_proportional_to_r: bool
    detail: str = ""

    def as_record(self) -> dict:
        """JSON-serializable form for an artifact record."""
        return {
            "ok": bool(self.ok),
            "reason": self.reason,
            "mu_reference": list(self.mu_reference),
            "mu_candidate": list(self.mu_candidate),
            "proportional_to_r": bool(self.proportional_to_r),
            "reference_proportional_to_r": bool(self.reference_proportional_to_r),
            "detail": self.detail,
        }


def resolve_mu_guard(mode: str) -> str:
    """Validate a guard mode, raising on an unknown one (never silently off)."""
    if mode not in MU_GUARD_MODES:
        raise ValueError(
            f"unknown mu_guard {mode!r}; expected one of {sorted(MU_GUARD_MODES)}")
    return mode


def specs_from_gf(V_gf) -> list:
    """TermSpecs of a materialized GF, exactly as ``verify_gf_sos_full`` builds them."""
    from ..algebra.enumerate import TermSpec
    return [TermSpec(exponents=t.exponents, term_types=t.term_types)
            for t in V_gf.terms]


def v_only_mu(term_specs: Sequence, weights_r, ensure_even_exponents: bool = True
              ) -> Tuple[int, ...]:
    """μ as every V-only path recomputes it: per-axis LCM of denominators.

    This is ``compute_mu_exponents``, the function ``verify_gf_sos_full`` uses
    at its μ step and the one whose value is stamped into the Gram captures the
    exact gate consumes. Imported lazily so this module stays importable from
    ``joint_sos`` itself without a cycle.
    """
    from .joint_sos import compute_mu_exponents
    _, mu = compute_mu_exponents(
        list(term_specs), tuple(weights_r), ensure_even_exponents)
    return tuple(int(v) for v in mu)


def design_mu(term_specs: Sequence, weights_r, dynamics_gf,
              ensure_even_exponents: bool = True) -> Tuple[int, ...]:
    """μ as the designer computes it: V terms AND dynamics, forced to μ = k·r."""
    from .joint_sos import compute_mu_exponents_with_dynamics
    _, mu = compute_mu_exponents_with_dynamics(
        term_specs=list(term_specs), weights_r=tuple(weights_r),
        dynamics_gf=dynamics_gf, ensure_even_exponents=ensure_even_exponents)
    return tuple(int(v) for v in mu)


def is_proportional_to_r(mu: Sequence[int], weights_r: Sequence[int]) -> bool:
    """True iff μᵢ = c·rᵢ for one positive rational c on every axis.

    Proportionality to r is what makes the associated forms homogeneous, which
    is the hypothesis ``exact_octant_target`` enforces by checking that every
    monomial of the octant target has the same total y-degree.
    """
    if len(mu) != len(weights_r):
        return False
    ratios = set()
    for m_i, r_i in zip(mu, weights_r):
        if int(r_i) == 0:
            return False
        ratios.add(Fraction(int(m_i), int(r_i)))
    return len(ratios) == 1 and next(iter(ratios)) > 0


def check_mu_preservation(
    term_specs: Sequence,
    weights_r,
    mu_reference: Sequence[int],
    ensure_even_exponents: bool = True,
    mu_candidate: Optional[Sequence[int]] = None,
) -> MuCheck:
    """Check one candidate support against the dense design μ.

    ``mu_reference`` is the V-only μ of the dense library the sparsification
    started from. ``mu_candidate`` may be supplied when the caller already
    computed it; otherwise it is recomputed from ``term_specs``.
    """
    ref = tuple(int(v) for v in mu_reference)
    cand = (tuple(int(v) for v in mu_candidate) if mu_candidate is not None
            else v_only_mu(term_specs, weights_r, ensure_even_exponents))
    prop = is_proportional_to_r(cand, weights_r)
    ref_prop = is_proportional_to_r(ref, weights_r)
    preserved = cand == ref

    if preserved and prop:
        return MuCheck(True, MU_OK, ref, cand, prop, ref_prop)
    if not preserved and not prop:
        reason = MU_NOT_PRESERVED_NONHOMOGENEOUS
        detail = (f"V-only mu {cand} differs from the dense design mu {ref} and "
                  f"is not proportional to r={tuple(int(v) for v in weights_r)}; "
                  f"the associated forms are not homogeneous and the exact gate "
                  f"refuses target assembly")
    elif not preserved:
        reason = MU_NOT_PRESERVED
        detail = (f"V-only mu {cand} differs from the dense design mu {ref}; "
                  f"the y-degree and hence the Gram basis change with the support")
    else:
        reason = MU_NOT_PROPORTIONAL
        detail = (f"V-only mu {cand} is not proportional to "
                  f"r={tuple(int(v) for v in weights_r)} (the reference carries the "
                  f"same defect); the associated forms are not homogeneous")
    return MuCheck(False, reason, ref, cand, prop, ref_prop, detail)
