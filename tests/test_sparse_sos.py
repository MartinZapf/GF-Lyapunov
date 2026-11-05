"""Tests for the reweighted-ℓ₁ sparse SOS design (verification.sparse_sos).

MOSEK-free (CLARABEL). The system under test is Twisting (r=(2,1), m=5): its
μ-filtered C¹ library has 10 terms, of which the dense designer uses all — a
real sparsification target, unlike the already-minimal 3-term 2D systems.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import enumerate_diagonal_terms, enumerate_cross_terms
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.verification import sparse_sos
from gf_framework.verification.sparse_sos import (
    SparseDesignConfig,
    atom_scales,
    design_sparse_lyapunov,
)
from gf_framework.verification.verifier import certify_discovered_gf


def _twisting_library(m=5):
    spec = ud.create_twisting_spec()
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return spec, m, list(diag) + list(cross)


def test_atom_scales_positive_and_gain_sensitive():
    spec, m, terms = _twisting_library()
    scales = atom_scales(terms, spec.weights_r, spec.dynamics_gf)
    assert scales.shape == (len(terms),)
    assert np.all(np.isfinite(scales)) and np.all(scales > 0)

    # Rescaling the dynamics gains must change the W-column contribution and
    # therefore the relative atom ranking input (the point of normalization).
    spec_hot = ud.create_twisting_spec(r1=20.0, r2=15.0)
    scales_hot = atom_scales(terms, spec_hot.weights_r, spec_hot.dynamics_gf)
    assert not np.allclose(scales / scales.max(), scales_hot / scales_hot.max())


def test_sparse_design_feasible_and_passes_strict_gate():
    spec, m, terms = _twisting_library()
    cfg = SparseDesignConfig(max_reweight_iters=4)
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL", config=cfg)
    assert res.feasible
    assert res.V_gf is not None
    assert res.n_terms_dense == len(terms)
    assert 0 < res.n_terms_sparse <= res.n_terms_dense
    # Diagnostics recorded per iteration.
    assert res.iterations and all("solve_time" in it for it in res.iterations)

    # The sparse candidate must clear the SAME strict gate as the dense path.
    cert = certify_discovered_gf(res.V_gf, spec.weights_r, m, spec.dynamics_gf,
                                 solver="CLARABEL", strict_solver=True)
    assert cert.surface_certified


def test_support_stability_stops_early():
    spec, m, terms = _twisting_library()
    cfg = SparseDesignConfig(max_reweight_iters=8, stable_window=2)
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL", config=cfg)
    assert res.feasible
    # Either the support stabilized (early stop) or all iterations ran.
    assert res.support_stable or len(res.iterations) == cfg.max_reweight_iters


def test_remargin_failure_returns_pruned_l1_candidate(monkeypatch):
    spec, m, terms = _twisting_library()

    real_find = sparse_sos.JointSOSOptimizer.find_lyapunov_with_dynamics

    def failing_remargin(self, *args, **kwargs):
        # Stage 2 runs without l1_weights — fail exactly that solve.
        if self.l1_weights is None:
            from gf_framework.verification.joint_sos import JointSOSResult
            return JointSOSResult(feasible=False, solver_status="forced_fail",
                                  message="forced remargin failure")
        return real_find(self, *args, **kwargs)

    monkeypatch.setattr(sparse_sos.JointSOSOptimizer,
                        "find_lyapunov_with_dynamics", failing_remargin)
    cfg = SparseDesignConfig(max_reweight_iters=2)
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL", config=cfg)
    assert res.feasible  # the ℓ₁ candidate is still returned …
    assert "remargin" in res.message  # … with the failure recorded
    assert res.V_gf is not None
    assert len(res.V_gf.terms) == res.n_terms_sparse  # pruned to the support


def test_discover_gf_sparsify_replaces_dense_certificate():
    import gf_framework.discovery.gf_discovery as gfd

    spec = ud.create_twisting_spec()
    r = gfd.discover_gf(spec, solver="CLARABEL", max_extra_degree=3,
                        sparsify="reweighted_l1")
    assert r.certified
    assert r.sparsifier == "reweighted_l1"
    assert r.method == "gf_sos_reweighted_l1"
    assert 0 < r.n_terms < r.n_terms_dense
    assert r.reweight_iters > 0


def test_discover_gf_sparsify_gate_failure_keeps_dense(monkeypatch):
    import gf_framework.discovery.gf_discovery as gfd

    spec = ud.create_twisting_spec()
    real_gate = gfd.certify_discovered_gf
    calls = {"n": 0}

    def gate_rejecting_sparse(V_gf, *args, **kwargs):
        calls["n"] += 1
        cert = real_gate(V_gf, *args, **kwargs)
        if calls["n"] > 1:  # first call gates the dense candidate; later = sparse
            cert.surface_certified = False
        return cert

    monkeypatch.setattr(gfd, "certify_discovered_gf", gate_rejecting_sparse)
    r = gfd.discover_gf(spec, solver="CLARABEL", max_extra_degree=3,
                        sparsify="reweighted_l1")
    assert r.certified  # dense certificate kept
    assert r.sparsifier == "none"
    assert r.method == "gf_sos_full_library"
    assert r.n_terms == r.n_terms_dense
    assert "gate" in r.sparsify_failed_reason


def test_uniform_weighting_arm_runs_and_is_recorded():
    """The uniform arm is a real second arm: it runs and names itself."""
    spec, m, terms = _twisting_library()
    cfg = SparseDesignConfig(max_reweight_iters=4, weighting="uniform")
    assert cfg.resolved_weighting() == "uniform"
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL", config=cfg)
    assert res.feasible
    assert res.V_gf is not None
    assert 0 < res.n_terms_sparse <= res.n_terms_dense
    # The record names the arm that produced it.
    assert res.weighting == "uniform"
    assert "uniform weights" in res.message

    # Default is unchanged: the shipped atom normalization.
    default = SparseDesignConfig(max_reweight_iters=4)
    assert default.resolved_weighting() == "normalized"
    res_n = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                   solver="CLARABEL", config=default)
    assert res_n.weighting == "normalized"
    assert res_n.feasible


def test_weighting_aliases_legacy_normalize_atoms_flag():
    # Legacy callers pass normalize_atoms; it stays an alias of weighting.
    assert SparseDesignConfig(normalize_atoms=False).resolved_weighting() == "uniform"
    assert SparseDesignConfig(normalize_atoms=True).resolved_weighting() == "normalized"
    with pytest.raises(ValueError):
        SparseDesignConfig(weighting="sup-norm").resolved_weighting()


def test_no_cross_terms_is_noop():
    spec, m, terms = _twisting_library()
    diag_only = [t for t in terms if t.is_diagonal()]
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, diag_only,
                                 solver="CLARABEL")
    assert not res.feasible
    assert "no cross terms" in res.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _record_design_epsilons(monkeypatch):
    """Record (stage, epsilon) of every design solve inside the rl1 loop."""
    seen = []
    real_find = sparse_sos.JointSOSOptimizer.find_lyapunov_with_dynamics

    def recording(self, *args, **kwargs):
        seen.append(("rl1" if self.l1_weights is not None else "remargin",
                     float(self.epsilon)))
        return real_find(self, *args, **kwargs)

    monkeypatch.setattr(sparse_sos.JointSOSOptimizer,
                        "find_lyapunov_with_dynamics", recording)
    return seen


def test_design_epsilon_default_is_unchanged(monkeypatch):
    """SparseDesignConfig.design_epsilon (X5, 2026-09-25) defaults to None,
    and then every ladder and remargin solve runs at the designer's own
    default epsilon, as before the knob existed."""
    assert SparseDesignConfig().design_epsilon is None
    default_eps = sparse_sos.JointSOSOptimizer(weights_r=(2, 1)).epsilon
    spec, m, terms = _twisting_library()
    seen = _record_design_epsilons(monkeypatch)
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL",
                                 config=SparseDesignConfig(max_reweight_iters=2))
    assert res.feasible
    assert {s for s, _ in seen} == {"rl1", "remargin"}
    assert all(eps == default_eps for _, eps in seen)


def test_design_epsilon_reaches_ladder_and_remargin(monkeypatch):
    """A set design_epsilon reaches BOTH the reweighted-l1 solves and the
    stage-2 remargin re-solve, so the support is selected on D's own SDP."""
    spec, m, terms = _twisting_library()
    seen = _record_design_epsilons(monkeypatch)
    cfg = SparseDesignConfig(max_reweight_iters=2, design_epsilon=1e-6)
    res = design_sparse_lyapunov(spec.weights_r, m, spec.dynamics_gf, terms,
                                 solver="CLARABEL", config=cfg)
    assert res.feasible
    assert {s for s, _ in seen} == {"rl1", "remargin"}
    assert all(eps == 1e-6 for _, eps in seen)
