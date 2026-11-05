"""Soundness regression tests — lock the verifier defects closed.

These pin the behaviour established by the review (`review/REVIEW.md`): the
published artifacts that the *old* verifier wrongly certified must now be
rejected. MOSEK-free: everything runs under CLARABEL.

Reference (independent re-verification, commit 1323284 / results_sparse set):
  * CTA, Levant3D : V is NOT positive definite (min V < 0 on the unit sphere)
  * PID-SMC       : V is PD but V̇ > 0 on ~4.2% of the domain — the decrease
                    stage (W SOS) and the strict gate must both reject it.
  * STA, Twisting : degenerate weak LFs (artifact-only checks below).

The invalid certificates are EMBEDDED verbatim so these regressions can never
silently skip; the untracked artifact directory is only needed for the
genuinely-PD over-rejection checks (sta/twisting).

Also unit-tests the a-posteriori Gram validation
(`joint_sos._validate_gram_certificate`): solver status alone must never accept
a Gram whose equality residual or eigenvalue shift exceeds the ε-margin budget
— the exact mechanism behind the old false positives.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.interfaces import VerificationConfig
from gf_framework.verification.joint_sos import (
    _validate_gram_certificate,
    verify_w_given_numeric,
)
from gf_framework.verification.sphere_check import min_V_on_sphere
from gf_framework.verification.verifier import certify_discovered_gf, verify_v_only

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts/_archive/publication/results_sparse_20260421T093238Z"

# ---------------------------------------------------------------------------
# Embedded invalid published certificates (commit 1323284), verbatim.
# (coefficient, exponents, term_types) per term.
# ---------------------------------------------------------------------------
INVALID_PUBLISHED = {
    "continuous_twisting": {
        "degree_m": 5,
        "weights_r": (3, 2, 1),
        "terms": [
            (1.0, ("5/3", "0", "0"), ("abs", "abs", "abs")),
            (0.03575816344408119, ("0", "5/2", "0"), ("abs", "abs", "abs")),
            (6.79308750165037e-05, ("0", "0", "5"), ("abs", "abs", "abs")),
            (0.2675848629562511, ("1", "1", "0"), ("signed", "signed", "abs")),
            (0.01438567794410601, ("0", "2", "1"), ("signed", "signed", "abs")),
            (-0.17937232692179997, ("4/3", "0", "1"), ("signed", "signed", "abs")),
            (-0.13527833637853162, ("1", "0", "2"), ("signed", "signed", "abs")),
        ],
    },
    "levant3d": {
        "degree_m": 6,
        "weights_r": (3, 2, 1),
        "terms": [
            (1.0, ("2", "0", "0"), ("abs", "abs", "abs")),
            (0.01931678047620025, ("0", "3", "0"), ("abs", "abs", "abs")),
            (0.030906880870852006, ("0", "0", "6"), ("abs", "abs", "abs")),
            (-60.78764863884497, ("1", "1", "1"), ("abs", "signed", "signed")),
            (61.44116704875443, ("1", "1", "1"), ("abs", "abs", "abs")),
            (2446.4681758705083, ("1", "0", "3"), ("abs", "abs", "abs")),
            (-1331.0076898876064, ("0", "2", "2"), ("abs", "signed", "signed")),
            (-1103.3194116169655, ("0", "1", "4"), ("abs", "signed", "signed")),
            (1103.298556769209, ("0", "1", "4"), ("abs", "abs", "abs")),
            (2443.952625945513, ("1", "0", "3"), ("signed", "abs", "signed")),
            (-2443.718763692153, ("1", "0", "3"), ("abs", "signed", "signed")),
            (1331.367866241832, ("0", "2", "2"), ("abs", "abs", "abs")),
            (61.43524199416021, ("1", "1", "1"), ("signed", "abs", "signed")),
            (1103.2984654361826, ("0", "1", "4"), ("signed", "abs", "signed")),
            (-61.6327834918257, ("1", "1", "1"), ("signed", "signed", "abs")),
            (1331.0732074545783, ("0", "2", "2"), ("signed", "abs", "signed")),
            (-1331.0201001552944, ("0", "2", "2"), ("signed", "signed", "abs")),
            (-1103.0976315444939, ("0", "1", "4"), ("signed", "signed", "abs")),
            (-2446.6200030267455, ("1", "0", "3"), ("signed", "signed", "abs")),
            (3.401468079190644e-11, ("0", "2", "2"), ("abs", "signed", "abs")),
        ],
    },
    "pidsmc": {
        "degree_m": 7,
        "weights_r": (3, 2, 1),
        "terms": [
            (1.0, ("7/3", "0", "0"), ("abs", "abs", "abs")),
            (11.429967400684816, ("0", "7/2", "0"), ("abs", "abs", "abs")),
            (7.231695389708729e-05, ("0", "0", "7"), ("abs", "abs", "abs")),
            (6.822046949722947, ("5/3", "1", "0"), ("abs", "abs", "abs")),
            (25.09493574002725, ("4/3", "3/2", "0"), ("abs", "abs", "abs")),
            (-24.66984369514575, ("1", "2", "0"), ("abs", "abs", "abs")),
            (14.413337750535208, ("5/3", "1", "0"), ("signed", "signed", "abs")),
        ],
    },
}


def build_embedded_V(name: str) -> GeneralizedForm:
    data = INVALID_PUBLISHED[name]
    terms = [
        GFTerm(
            coefficient=float(c),
            exponents=tuple(Fraction(e) for e in exps),
            term_types=tuple(types),
        )
        for c, exps, types in data["terms"]
    ]
    return GeneralizedForm(
        degree_m=data["degree_m"], weights_r=data["weights_r"], terms=terms
    )


def load_artifact_V(name: str) -> GeneralizedForm:
    r = json.load(open(ARTIFACTS / f"{name}_seed42.json"))["result"]
    coeffs = list(r["coefficients"].values())
    terms = [
        GFTerm(
            coefficient=float(c),
            exponents=tuple(Fraction(e) for e in t["exponents"]),
            term_types=tuple(t["term_types"]),
        )
        for c, t in zip(coeffs, r["term_specs"])
    ]
    return GeneralizedForm(degree_m=r["degree_m"], weights_r=tuple(r["weights_r"]), terms=terms)


def load_V(name: str) -> GeneralizedForm:
    if ARTIFACTS.exists():
        return load_artifact_V(name)
    return build_embedded_V(name)


def _config():
    cfg = VerificationConfig()
    cfg.solver = "CLARABEL"  # no MOSEK needed
    return cfg


# ---------------------------------------------------------------------------
# Published false positives must be rejected (embedded — never skip)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["continuous_twisting", "levant3d"])
def test_non_pd_V_refuted_on_sphere(name):
    """The sphere necessary-check must expose min V < 0 (no SOS can exist)."""
    sc = min_V_on_sphere(build_embedded_V(name))
    assert sc.refutes_pd
    assert sc.min_V < -1e-4


@pytest.mark.parametrize("name", ["continuous_twisting", "levant3d"])
def test_non_pd_V_rejected_by_verifier(name):
    """verify_v_only must return False for the non-PD certified V's."""
    verified, method, _ = verify_v_only(build_embedded_V(name), _config(), verbose=False)
    assert verified is False, f"{name}: V-positivity wrongly certified via {method}"


def test_pidsmc_decrease_failure_rejected():
    """The published PID-SMC V is PD but V̇ > 0 on part of the domain: the
    decrease stage (W SOS on open orthants) must be infeasible for it."""
    import gf_framework.discovery.universal_discovery as ud

    spec = ud.create_pidsmc_spec()
    V = build_embedded_V("pidsmc")
    res = verify_w_given_numeric(
        V, spec.dynamics_gf, V.weights_r, V.degree_m, solver="CLARABEL"
    )
    assert res.feasible is False, "non-decreasing published PID-SMC V passed W SOS"


@pytest.mark.parametrize("name", ["continuous_twisting", "levant3d", "pidsmc"])
def test_invalid_published_certs_rejected_by_strict_gate(name):
    """End-to-end: certify_discovered_gf (the only source of emitted claims)
    must refuse all three invalid published certificates, whichever guard fires
    first (sphere refutation, W infeasibility, or non-C¹ scope)."""
    import gf_framework.discovery.universal_discovery as ud

    factory = {
        "continuous_twisting": ud.create_cta_spec,
        "levant3d": ud.create_levant3d_spec,
        "pidsmc": ud.create_pidsmc_spec,
    }[name]
    spec = factory()
    V = build_embedded_V(name)
    cert = certify_discovered_gf(
        V, V.weights_r, V.degree_m, spec.dynamics_gf, solver="CLARABEL"
    )
    assert cert.surface_certified is False, (
        f"{name}: invalid published certificate re-certified ({cert.message})"
    )


# ---------------------------------------------------------------------------
# Genuinely PD V's keep certifying (over-rejection guard; artifact-only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACTS.exists(), reason="published artifacts not present")
@pytest.mark.parametrize("name", ["sta", "twisting", "pidsmc"])
def test_genuinely_pd_V_still_certifies(name):
    """V's that really are PD keep certifying (the regression must not over-reject)."""
    sc = min_V_on_sphere(load_V(name))
    assert sc.is_possibly_pd, f"{name}: min V on sphere = {sc.min_V}"
    verified, _method, _ = verify_v_only(load_V(name), _config(), verbose=False)
    assert verified is True


# ---------------------------------------------------------------------------
# A-posteriori Gram validation (the mechanism behind the old false positives)
# ---------------------------------------------------------------------------

_BASIS_2D = [(1, 0), (0, 1)]  # degree-1 basis for a degree-2 homogeneous target


def test_gram_validator_accepts_exact_certificate():
    """Exact Gram (Q = I for x² + y²) passes with room to spare."""
    target = {(2, 0): 1.0, (0, 2): 1.0}
    ok, min_eig, resid, reason = _validate_gram_certificate(
        target, np.eye(2), _BASIS_2D, 2, eps_budget=1e-6, psd_tol=1e-7
    )
    assert ok, reason
    assert min_eig == pytest.approx(1.0)
    assert resid == pytest.approx(0.0, abs=1e-15)


def test_gram_validator_rejects_residual_beyond_margin():
    """A PSD Gram whose polynomial does NOT match the target (equality residual
    larger than the ε-margin) must be rejected even though λ_min > 0 — this is
    the OPTIMAL_INACCURATE false-positive class."""
    target = {(2, 0): 1.0, (0, 2): 1.0 + 1e-3}  # Gram reproduces only 1.0
    ok, _min_eig, resid, reason = _validate_gram_certificate(
        target, np.eye(2), _BASIS_2D, 2, eps_budget=1e-6, psd_tol=1e-7
    )
    assert not ok
    assert resid == pytest.approx(1e-3)
    assert "gram_margin_exceeded" in reason


def test_gram_validator_charges_negative_eigenvalue_to_margin():
    """A slightly indefinite Gram is only acceptable if the eigenvalue shift
    fits inside the ε budget; 1e-3 against a 1e-6 budget must fail."""
    gram = np.diag([1.0, -1e-3])
    target = {(2, 0): 1.0, (0, 2): -1e-3}  # matches the Gram exactly: resid = 0
    ok, min_eig, _resid, reason = _validate_gram_certificate(
        target, gram, _BASIS_2D, 2, eps_budget=1e-6, psd_tol=1e-7
    )
    assert not ok
    assert min_eig == pytest.approx(-1e-3)
    assert "gram_margin_exceeded" in reason


def test_gram_validator_accepts_solver_noise_within_margin():
    """Realistic solver noise (1e-10 residual, 1e-12 eigenvalue dip) is absorbed
    by a 1e-6 margin: certification must still succeed."""
    gram = np.diag([1.0, 1.0]) + np.array([[0.0, 1e-12], [1e-12, -1e-12]])
    target = {(2, 0): 1.0, (0, 2): 1.0 - 1e-10, (1, 1): 2e-12}
    ok, _min_eig, _resid, reason = _validate_gram_certificate(
        target, gram, _BASIS_2D, 2, eps_budget=1e-6, psd_tol=1e-7
    )
    assert ok, reason


def test_gram_validator_legacy_policy_without_budget():
    """Without a margin budget (legacy callers), the conservative absolute
    policy applies: tiny residuals pass, anything beyond 1e-7 fails."""
    clean = {(2, 0): 1.0, (0, 2): 1.0}
    ok, _e, _r, _m = _validate_gram_certificate(
        clean, np.eye(2), _BASIS_2D, 2, eps_budget=0.0, psd_tol=1e-7
    )
    assert ok

    drifted = {(2, 0): 1.0, (0, 2): 1.0 + 1e-3}
    ok, _e, resid, reason = _validate_gram_certificate(
        drifted, np.eye(2), _BASIS_2D, 2, eps_budget=0.0, psd_tol=1e-7
    )
    assert not ok
    assert "gram_residual" in reason
