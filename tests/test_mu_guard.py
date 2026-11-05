"""Tests for the μ-preservation guard (verification.mu_guard) and its wiring
into the reweighted-ℓ₁ sparsifier.

The hazard under test, from `docs/FINDINGS_2026-09-02.md` section 5: the
designer computes μ with the dynamics (μ = k·r, always homogeneous), while
every V-only path recomputes μ from the V terms alone. On pidsmc (r = (3,2,1),
m = 6) the diagonals are |x₁|², |x₂|³, |x₃|⁶ — all integer exponents — so
nothing but the CROSS terms pins the denominator-2 exponents on the x₂ axis.
Drop them and V-only μ falls from (6,4,2) to (6,2,2), which is not proportional
to r; the associated forms stop being homogeneous and `exact_octant_target`
refuses. The float gate does not notice.

The library is rebuilt live here (no artifact dependency); the audit over the
published records lives in
`artifacts/analysis_2026-09-02/sparse_reaudit/`.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import (
    enumerate_cross_terms, enumerate_diagonal_terms,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.verification import sparse_sos
from gf_framework.verification.mu_guard import (
    MU_GUARD_MODES,
    MU_NOT_PRESERVED,
    MU_NOT_PRESERVED_NONHOMOGENEOUS,
    MU_NOT_PROPORTIONAL,
    MuCheck,
    check_mu_preservation,
    design_mu,
    is_proportional_to_r,
    resolve_mu_guard,
    specs_from_gf,
    v_only_mu,
)
from gf_framework.verification.sparse_sos import (
    SparseDesignConfig, design_sparse_lyapunov,
)


def _library(spec, m, max_denominator=4):
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r,
                                       max_denominator=max_denominator,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return list(diag) + list(cross)


def _pidsmc_library(m=6):
    return ud.create_pidsmc_spec(), m, _library(ud.create_pidsmc_spec(), m)


def _twisting_library(m=5):
    spec = ud.create_twisting_spec()
    return spec, m, _library(spec, m)


# --------------------------------------------------------------------------
# The mechanism, on the real pidsmc library
# --------------------------------------------------------------------------

def test_pidsmc_dense_library_mu_is_642_and_proportional_to_r():
    spec, m, terms = _pidsmc_library()
    assert v_only_mu(terms, spec.weights_r) == (6, 4, 2)
    assert design_mu(terms, spec.weights_r, spec.dynamics_gf) == (6, 4, 2)
    assert is_proportional_to_r((6, 4, 2), spec.weights_r)


def test_dropping_the_x2_half_exponents_breaks_mu_exactly_as_published():
    """The published 28 -> 9 mechanism, reproduced from the library alone."""
    spec, m, terms = _pidsmc_library()
    dense_mu = v_only_mu(terms, spec.weights_r)

    def pins_mu2(t):
        e = t.exponents[1]
        return isinstance(e, Fraction) and e != 0 and e.denominator == 2

    broken = [t for t in terms if not pins_mu2(t)]
    assert len(broken) < len(terms), "the library must contain x2 half-exponents"

    broken_mu = v_only_mu(broken, spec.weights_r)
    assert broken_mu == (6, 2, 2)
    assert not is_proportional_to_r(broken_mu, spec.weights_r)

    chk = check_mu_preservation(broken, spec.weights_r, dense_mu)
    assert not chk.ok
    assert chk.reason == MU_NOT_PRESERVED_NONHOMOGENEOUS
    assert chk.mu_candidate == (6, 2, 2) and chk.mu_reference == (6, 4, 2)
    assert "not homogeneous" in chk.detail

    # A support that keeps every denominator-carrying term is clean, and the
    # dense library trivially is.
    def carries_a_denominator(t):
        return any(isinstance(e, Fraction) and e != 0 and e.denominator > 1
                   for e in t.exponents)

    kept = [t for t in terms if t.is_diagonal() or carries_a_denominator(t)]
    assert v_only_mu(kept, spec.weights_r) == (6, 4, 2)
    assert check_mu_preservation(kept, spec.weights_r, dense_mu).ok
    assert check_mu_preservation(terms, spec.weights_r, dense_mu).ok


def test_is_proportional_to_r_accepts_only_one_common_ratio():
    assert is_proportional_to_r((6, 4, 2), (3, 2, 1))
    assert is_proportional_to_r((3, 2, 1), (3, 2, 1))
    assert not is_proportional_to_r((6, 2, 2), (3, 2, 1))
    assert not is_proportional_to_r((6, 4), (3, 2, 1))       # wrong length
    assert not is_proportional_to_r((0, 0, 0), (3, 2, 1))    # not positive


def test_typed_reasons_are_distinct_and_populated():
    r = (3, 2, 1)
    # differs from the reference but still homogeneous
    chk = check_mu_preservation([], r, (6, 4, 2), mu_candidate=(3, 2, 1))
    assert not chk.ok and chk.reason == MU_NOT_PRESERVED
    assert chk.proportional_to_r and chk.detail

    # equals a reference that is itself defective: still refused
    chk = check_mu_preservation([], r, (6, 2, 2), mu_candidate=(6, 2, 2))
    assert not chk.ok and chk.reason == MU_NOT_PROPORTIONAL
    assert not chk.proportional_to_r and not chk.reference_proportional_to_r

    ok = check_mu_preservation([], r, (6, 4, 2), mu_candidate=(6, 4, 2))
    assert ok.ok and ok.reason == "" and isinstance(ok, MuCheck)
    assert ok.as_record()["mu_candidate"] == [6, 4, 2]


def test_resolve_mu_guard_rejects_unknown_mode():
    for mode in MU_GUARD_MODES:
        assert resolve_mu_guard(mode) == mode
    with pytest.raises(ValueError):
        resolve_mu_guard("ignore")
    with pytest.raises(ValueError):
        SparseDesignConfig(mu_guard="ignore").resolved_mu_guard()
    assert SparseDesignConfig().resolved_mu_guard() == "reject"


# --------------------------------------------------------------------------
# Wiring into design_sparse_lyapunov
# --------------------------------------------------------------------------

def test_design_records_mu_and_passes_a_clean_system():
    """Twisting is μ-clean (its x₁ diagonal |x₁|^{5/2} pins μ₁), so the guard
    is silent and the reference numbers are on the record."""
    spec, m, terms = _twisting_library()
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL",
                                 config=SparseDesignConfig(max_reweight_iters=4))
    assert res.feasible and res.V_gf is not None
    assert res.mu_guard == "reject"
    assert res.mu_design == [4, 2]
    assert res.mu_dense_v_only == [4, 2]
    assert res.mu_sparse_v_only == [4, 2]
    assert res.mu_checks and all(c["ok"] for c in res.mu_checks)
    assert v_only_mu(specs_from_gf(res.V_gf), spec.weights_r) == (4, 2)


def _reject_supports_smaller_than(limit):
    """A guard stub that fails exactly the supports we want failed.

    The MECHANISM is covered by the library tests above and by the artifact
    re-audit; what this stub isolates is the LOOP WIRING: is a failing check
    typed into ``rejected``, does the ladder walk on, is it recorded.
    """
    real = check_mu_preservation

    def stub(term_specs, weights_r, mu_reference, *a, **kw):
        chk = real(term_specs, weights_r, mu_reference, *a, **kw)
        if len(list(term_specs)) < limit:
            return MuCheck(False, MU_NOT_PRESERVED_NONHOMOGENEOUS,
                           chk.mu_reference, (6, 2, 2), False, True,
                           "forced mu break for the wiring test")
        return chk
    return stub


def test_mu_breaking_support_is_rejected_with_a_typed_reason(monkeypatch):
    spec, m, terms = _twisting_library()
    monkeypatch.setattr(sparse_sos, "check_mu_preservation",
                        _reject_supports_smaller_than(len(terms)))

    def gate(V):
        return True  # the gate would accept; only the μ guard may refuse

    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL", gate_fn=gate,
                                 config=SparseDesignConfig(max_reweight_iters=4))
    # Every support the ladder offered was smaller than the library, so all
    # were refused and nothing gated.
    assert not res.gated
    assert MU_NOT_PRESERVED_NONHOMOGENEOUS in res.message
    assert "mu guard WARNING" not in res.message  # that wording is the warn mode
    assert res.mu_checks and not any(c["ok"] for c in res.mu_checks)
    assert all(c["enforced"] for c in res.mu_checks)


def test_mu_guard_off_restores_the_previous_behaviour(monkeypatch):
    spec, m, terms = _twisting_library()
    monkeypatch.setattr(sparse_sos, "check_mu_preservation",
                        _reject_supports_smaller_than(len(terms)))
    res = design_sparse_lyapunov(
        spec.weights_r, m, spec.dynamics_gf, terms, solver="CLARABEL",
        gate_fn=lambda V: True,
        config=SparseDesignConfig(max_reweight_iters=4, mu_guard="off"))
    assert res.gated and res.feasible
    assert res.mu_checks == []          # not even evaluated
    assert res.mu_guard == "off"


def test_mu_guard_warn_keeps_the_support_but_types_the_reason(monkeypatch):
    spec, m, terms = _twisting_library()
    monkeypatch.setattr(sparse_sos, "check_mu_preservation",
                        _reject_supports_smaller_than(len(terms)))
    res = design_sparse_lyapunov(
        spec.weights_r, m, spec.dynamics_gf, terms, solver="CLARABEL",
        gate_fn=lambda V: True,
        config=SparseDesignConfig(max_reweight_iters=4, mu_guard="warn"))
    assert res.gated and res.feasible
    assert "mu guard WARNING" in res.message
    assert MU_NOT_PRESERVED_NONHOMOGENEOUS in res.message
    assert res.mu_checks and not res.mu_checks[0]["enforced"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
