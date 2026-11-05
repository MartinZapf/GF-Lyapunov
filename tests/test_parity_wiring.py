"""Wiring of the parity/symmetry reduction into the SOS DESIGN path (block S1).

``tests/test_parity_reduction.py`` pins the reduction itself (symbolic and
combinatorial, solver-free). This file pins the WIRING: that
``JointSOSOptimizer.find_lyapunov_with_dynamics`` actually poses the smaller
problem, that it poses the RIGHT smaller problem, that it refuses when it may
not, and — the campaign's S1 gate — that a certificate designed from a reduced
formulation still passes the UNREDUCED strict gate.

Solver: CLARABEL throughout (no MOSEK on CI or on the travel machine). The
MOSEK_DIRECT backend takes the same reduction through the same code path
(``_find_lyapunov_direct``); it is exercised on the compute box, not here.

What is deliberately NOT asserted (theory_notes.md §10.6): equality of the
design-solver VERDICT between reduced and unreduced runs. The full formulation
may return unknown from size or conditioning while the equivalent reduced one
succeeds; that is a legitimate benefit, not a discrepancy. What IS
disqualifying is a reduced certificate that fails an unreduced gate.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.algebra.enumerate import enumerate_diagonal_terms
from gf_framework.algebra.symmetry import (
    ENDPOINT_MINUS,
    ENDPOINT_PLUS,
    flip_octant,
    is_even_term,
    octant_from_index,
    octant_index,
)
from gf_framework.discovery.gf_discovery import mu_filtered_cross
from gf_framework.verification.joint_sos import (
    PARITY_REFUSAL_CODES,
    JointSOSOptimizer,
    get_parity_reduction_default,
    set_parity_reduction_default,
)
from gf_framework.verification.verifier import certify_discovered_gf

SOLVER = "CLARABEL"


# =============================================================================
# Helpers
# =============================================================================

def _library(spec, m):
    """The μ-filtered complete lattice slice at degree m — what the drivers use."""
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    _mu, _full, cross = mu_filtered_cross(m, spec.weights_r, diag, spec.dynamics_gf)
    return list(diag) + list(cross)


def _design(spec, m, *, parity_reduction, terms=None, **kwargs):
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver=SOLVER, verbose=False,
                            parity_reduction=parity_reduction, **kwargs)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m,
        dynamics_gf=spec.dynamics_gf,
        active_terms=list(terms if terms is not None else _library(spec, m)),
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
    )


def _coeff_vector(result):
    return np.array([result.coefficients[f"alpha{i}"]
                     for i in range(len(result.term_specs))])


# The four local validation cases (the full 7-case MOSEK gate runs on the
# compute box). (name, spec factory, degree m, delta0 or None).
NOMINAL_CASES = [
    ("sta", lambda: ud.create_sta_spec(), 3),
    ("twisting", lambda: ud.create_twisting_spec(), 5),
    ("levant3d", lambda: ud.create_levant3d_spec(), 5),
]


# =============================================================================
# Default OFF: a run without the flag is the pre-existing run
# =============================================================================

def test_default_is_off_and_the_switch_round_trips():
    assert get_parity_reduction_default() is False
    previous = set_parity_reduction_default(True)
    try:
        assert previous is False
        assert get_parity_reduction_default() is True
        # An explicit argument always beats the process default, both ways.
        assert JointSOSOptimizer(weights_r=(2, 1), solver=SOLVER).parity_reduction
        assert not JointSOSOptimizer(weights_r=(2, 1), solver=SOLVER,
                                     parity_reduction=False).parity_reduction
    finally:
        set_parity_reduction_default(previous)
    assert get_parity_reduction_default() is False
    assert not JointSOSOptimizer(weights_r=(2, 1), solver=SOLVER).parity_reduction


@pytest.mark.parametrize("delta0", [None, 0.5])
def test_flag_off_carries_no_parity_metadata(delta0):
    """Without the flag the result is indistinguishable from before the module."""
    spec = ud.create_sta_spec() if delta0 is None else ud.create_sta_spec(delta0=delta0)
    res = _design(spec, 3, parity_reduction=False)
    assert res.feasible
    assert res.parity_reduction is None


def test_flag_off_still_certifies_and_the_refusal_path_is_bit_identical():
    """Flag off ⇒ same verdict as always; refusal ⇒ literally the same solve.

    ``coefficient_bounds={0: (None, None)}`` is an INERT trigger: it makes the
    (A3) guard refuse, but the bounds loop appends no constraint, so the SDP
    that gets solved is exactly the unreduced one. Bit-identical coefficients
    are therefore the sharpest available check that the refusal falls back to
    the untouched formulation rather than to a lookalike.
    """
    spec = ud.create_sta_spec()
    off = _design(spec, 3, parity_reduction=False)
    refused = _design(spec, 3, parity_reduction=True,
                      coefficient_bounds={0: (None, None)})

    assert off.feasible and refused.feasible
    assert off.parity_reduction is None
    assert refused.parity_reduction["mode"] == "refused"
    assert refused.parity_reduction["reason_code"] == "asymmetric_constraints"
    assert np.array_equal(_coeff_vector(off), _coeff_vector(refused))

    cert = certify_discovered_gf(off.V_gf, spec.weights_r, 3, spec.dynamics_gf,
                                 solver=SOLVER, strict_solver=True)
    assert cert.surface_certified


# =============================================================================
# The reduced formulation is the RIGHT smaller one
# =============================================================================

@pytest.mark.parametrize("name,factory,m", NOMINAL_CASES)
def test_nominal_block_counts_are_halved(name, factory, m):
    spec = factory()
    n = len(spec.weights_r)
    res = _design(spec, m, parity_reduction=True)
    meta = res.parity_reduction

    assert meta["mode"] == "nominal", f"{name}: {meta}"
    assert meta["n_endpoints"] == 1
    assert meta["n_v_blocks"] == 2 ** (n - 1)
    assert meta["n_w_blocks"] == 2 ** (n - 1)
    assert meta["n_v_blocks_full"] == 2 ** n
    assert meta["n_w_blocks_full"] == 2 ** n
    # Representatives are the s_1 = +1 octants, addressed by FULL octant index.
    assert len(set(meta["v_block_octants"])) == 2 ** (n - 1)
    for idx in meta["v_block_octants"]:
        assert octant_from_index(idx, n)[0] == 1
    # No endpoint index exists nominally, and the partner is the antipode.
    for block in meta["w_blocks"]:
        assert block["endpoint"] is None
        assert block["partner_endpoint"] is None
        signs = tuple(block["signs"])
        assert block["partner_octant_index"] == octant_index(flip_octant(signs))


def test_paired_reduction_keeps_both_endpoints_and_pairs_them():
    """The perturbed case: 2^n W blocks, and every partner FLIPS the endpoint.

    Naive octant halving at a fixed endpoint would give 2^(n-1) W blocks and a
    partner with the SAME endpoint. Corollary 10.4 says that drops constraints
    no other block covers, so both are asserted absent here.
    """
    spec = ud.create_sta_spec(k1=1.5, k2=1.1, delta0=0.5)
    n = len(spec.weights_r)
    assert len(spec.dynamics_gf_vertices) == 2

    res = _design(spec, 3, parity_reduction=True)
    meta = res.parity_reduction

    assert meta["mode"] == "paired"
    assert meta["n_endpoints"] == 2
    assert meta["n_v_blocks"] == 2 ** (n - 1)          # V does not see δ
    assert meta["n_w_blocks"] == 2 ** n                # NOT 2^(n-1)
    assert meta["n_w_blocks_full"] == 2 ** (n + 1)
    assert meta["naive_octant_halving"] is False
    assert meta["checks"] == {"dynamics_odd": True, "endpoint_swap": True}

    # Both endpoints of every representative octant are present, exactly once.
    labels = sorted(b["label"] for b in meta["w_blocks"])
    assert labels == ["oct0+", "oct0-", "oct2+", "oct2-"]
    reps = {(b["octant_index"], b["endpoint"]) for b in meta["w_blocks"]}
    assert len(reps) == 2 ** n
    assert {e for _, e in reps} == {ENDPOINT_PLUS, ENDPOINT_MINUS}

    for block in meta["w_blocks"]:
        signs = tuple(block["signs"])
        assert signs[0] == 1, "representatives are the s_1 = +1 octants"
        # (s, σ) ↦ (−s, −σ): BOTH entries flip.
        assert block["partner_octant_index"] == octant_index(flip_octant(signs))
        assert block["partner_endpoint"] != block["endpoint"]
        assert block["partner_endpoint"] == (
            ENDPOINT_MINUS if block["endpoint"] == ENDPOINT_PLUS else ENDPOINT_PLUS)
        # The naive identification is never claimed.
        assert (block["partner_octant_index"], block["partner_endpoint"]) != (
            block["partner_octant_index"], block["endpoint"])


def test_reduced_library_is_the_even_sub_library_with_the_diagonals_intact():
    """Finding 6: the filter shifts indices, so the backbone is re-mapped.

    The scale normalization pins ``alpha[diag_indices[0]]``. If the filter
    reordered or dropped a diagonal, the reduced run would silently pin a
    DIFFERENT term than the unreduced run does, and the two certificates would
    not be comparable.
    """
    spec = ud.create_twisting_spec()
    m = 5
    full_library = _library(spec, m)
    res = _design(spec, m, parity_reduction=True, terms=full_library)
    meta = res.parity_reduction

    assert meta["mode"] == "nominal"
    assert meta["library"]["n_terms_full"] == len(full_library)
    assert meta["library"]["n_terms_even"] == len(res.term_specs)
    assert meta["library"]["n_terms_dropped_odd"] > 0, "twisting m=5 has odd terms"
    assert meta["library"]["n_terms_even"] + meta["library"]["n_terms_dropped_odd"] \
        == len(full_library)

    # Only even terms survive, in the original relative order.
    assert all(is_even_term(t) for t in res.term_specs)
    kept = [t for t in full_library if is_even_term(t)]
    assert list(res.term_specs) == kept

    # The diagonal backbone survived unchanged, and the FIRST diagonal — the
    # one the scale normalization pins — is still the same term, still pinned.
    diag_full = [t for t in full_library if t.is_diagonal()]
    diag_reduced = [t for t in res.term_specs if t.is_diagonal()]
    assert diag_full == diag_reduced
    first_diag_pos = next(i for i, t in enumerate(res.term_specs) if t.is_diagonal())
    assert res.coefficients[f"alpha{first_diag_pos}"] == pytest.approx(1.0)


def test_reduced_run_emits_a_complete_ordinary_certificate():
    """Finding 4: the emitted candidate must be a normal certificate.

    The reduction is a formulation change, not a certificate format. What comes
    out is a full V with a coefficient per (even) library term; the parity
    record is metadata beside it, never a caveat inside it.
    """
    spec = ud.create_levant3d_spec()
    res = _design(spec, 5, parity_reduction=True)

    assert res.feasible
    assert res.V_gf is not None
    assert len(res.V_gf.terms) == len(res.term_specs)
    assert set(res.coefficients) == {f"alpha{i}" for i in range(len(res.term_specs))}
    assert all(np.isfinite(c) for c in res.coefficients.values())
    assert res.V_gf.weights_r == tuple(spec.weights_r)
    assert res.V_gf.degree_m == 5
    for term, spec_ in zip(res.V_gf.terms, res.term_specs):
        assert term.exponents == spec_.exponents
        assert term.term_types == spec_.term_types
    assert "parity reduction: nominal" in res.message


# =============================================================================
# THE S1 GATE: reduced design ⇒ UNREDUCED strict verification must pass
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("name,factory,m", NOMINAL_CASES)
def test_reduced_candidate_passes_the_unreduced_strict_gate_nominal(name, factory, m):
    spec = factory()
    n = len(spec.weights_r)
    res = _design(spec, m, parity_reduction=True)
    assert res.feasible, f"{name}: reduced design infeasible ({res.solver_status})"
    assert res.parity_reduction["n_v_blocks"] == 2 ** (n - 1)
    assert res.parity_reduction["n_w_blocks"] == 2 ** (n - 1)

    # certify_discovered_gf → verify_gf_sos_full, which loops ALL 2^n octants.
    cert = certify_discovered_gf(
        res.V_gf, spec.weights_r, m, spec.dynamics_gf,
        solver=SOLVER, strict_solver=True)
    assert cert.surface_certified, f"{name}: unreduced gate rejected: {cert.message}"
    assert f"all {2 ** n} octants" in str(cert.message) or cert.surface_certified


@pytest.mark.slow
def test_reduced_candidate_passes_the_unreduced_strict_gate_perturbed():
    """STA at δ₀ = 0.5: reduced PAIRED design, then the full vertex-aware gate.

    The gate re-verifies W at BOTH endpoints over all 2^n octants — the 2^(n+1)
    blocks the design collapsed to 2^n.
    """
    delta0 = 0.5
    spec = ud.create_sta_spec(k1=1.5, k2=1.1, delta0=delta0)
    n = len(spec.weights_r)
    res = _design(spec, 3, parity_reduction=True)

    assert res.feasible, res.solver_status
    assert res.parity_reduction["mode"] == "paired"
    assert res.parity_reduction["n_w_blocks"] == 2 ** n

    cert = certify_discovered_gf(
        res.V_gf, spec.weights_r, 3, spec.dynamics_gf,
        solver=SOLVER, strict_solver=True,
        dynamics_gf_vertices=spec.dynamics_gf_vertices,
        disturbance_bound=delta0)
    assert cert.surface_certified, f"unreduced vertex-aware gate rejected: {cert.message}"
    # The gate really did see both endpoints, not a halved set.
    assert "2 inclusion vertices" in str(cert.message)


# =============================================================================
# Refusals: loud, typed, recorded — and always followed by the FULL formulation
# =============================================================================

def _assert_refused(result, code):
    meta = result.parity_reduction
    assert meta is not None, "a requested reduction must always leave a record"
    assert meta["mode"] == "refused"
    assert meta["requested"] is True
    assert meta["reason_code"] == code
    assert code in PARITY_REFUSAL_CODES
    assert meta["reason"], "a refusal must carry its reason, not just a code"
    # Refusing is a fallback, not a failure: the design still ran, unreduced.
    assert "n_v_blocks" not in meta


def test_refuses_on_more_than_two_endpoints(capsys):
    """Finding 2: a third vertex has no proved pairing anywhere in §10."""
    spec = ud.create_sta_spec(delta0=0.5)
    terms = _library(spec, 3)
    three = list(spec.dynamics_gf_vertices) + [spec.dynamics_gf]
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver=SOLVER,
                            parity_reduction=True)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=list(terms),
        dynamics_gf_vertices=three)

    _assert_refused(res, "endpoint_count_unsupported")
    assert res.feasible, "the unreduced fallback must still design normally"
    assert "REFUSED" in capsys.readouterr().err


def test_refuses_when_the_endpoint_swap_identity_fails(capsys):
    """Finding 2: the wiring never trusts index ORDER, only the verified identity.

    Two copies of the SAME endpoint look like a well-formed 2-vertex inclusion
    positionally. They are not: f_+(-x) = -f_+(x) is false for δ₀ > 0, so the
    pairing (s, +) ↔ (−s, −) has nothing to stand on and must be refused.
    """
    spec = ud.create_sta_spec(delta0=0.5)
    terms = _library(spec, 3)
    duplicated = [spec.dynamics_gf_vertices[ENDPOINT_PLUS],
                  spec.dynamics_gf_vertices[ENDPOINT_PLUS]]
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver=SOLVER,
                            parity_reduction=True)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf, active_terms=list(terms),
        dynamics_gf_vertices=duplicated)

    _assert_refused(res, "endpoint_swap_failed")
    assert "REFUSED" in capsys.readouterr().err


def test_refuses_on_non_odd_dynamics(capsys):
    """(A1) is verified, never assumed: an even component kills the reduction."""
    spec = ud.create_sta_spec()
    terms = _library(spec, 3)
    even_component = GeneralizedForm(
        degree_m=2, weights_r=(2, 1),
        terms=[GFTerm(1.0, (Fraction(1), Fraction(0)), ("abs", "abs"))])
    tampered = [even_component, spec.dynamics_gf[1]]

    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver=SOLVER,
                            parity_reduction=True)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=tampered, active_terms=list(terms))

    _assert_refused(res, "dynamics_not_odd")
    assert "REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize("option", [
    {"coefficient_bounds": {0: (0.0, None)}},
    {"domination_pairs": [(0, 1)]},
    {"reference_coefficients": [1.0, 1.0, 1.0]},
    {"linear_coefficient_constraints": [([1.0, 1.0, 1.0], 0.0, None)]},
    {"convexity_cuts": np.ones((1, 3)), "convexity_cut_margin": 1e-3},
])
def test_refuses_when_an_asymmetric_per_term_constraint_is_active(option):
    """(A3), theory_notes.md §10.0: the averaging step needs an R-invariant F.

    Each of these attaches something to individual coefficient indices that is
    not invariant under the coefficient reflection Rα = (π(t_j)·α_j)_j (or that
    indexes the unfiltered library with no sound re-map). Soundness is not at
    stake — the WLOG step is — so the reduction backs off instead of guessing.
    """
    spec = ud.create_sta_spec()
    terms = _library(spec, 3)
    assert len(terms) == 3, "the option fixtures above are sized for this library"
    res = _design(spec, 3, parity_reduction=True, terms=terms, **option)
    _assert_refused(res, "asymmetric_constraints")


def test_weighted_l1_is_kept_and_remapped_onto_the_even_sub_library(capsys):
    """Σ wⱼ|αⱼ| is R-invariant, so it does NOT refuse — it gets re-mapped.

    Finding 6: the weights index the UNFILTERED library, so they must be
    gathered at the kept positions. Sending them through unchanged would raise
    a length mismatch; truncating them would silently penalize the wrong terms.
    """
    spec = ud.create_twisting_spec()
    m = 5
    full_library = _library(spec, m)
    n_full = len(full_library)
    even_positions = [i for i, t in enumerate(full_library) if is_even_term(t)]
    assert len(even_positions) < n_full, "this library must have odd terms to drop"

    # A distinctive weight per term so a wrong re-map would change the optimum.
    weights = np.linspace(1.0, 2.0, n_full)
    res = _design(spec, m, parity_reduction=True, terms=full_library,
                  l1_weights=weights)

    assert res.parity_reduction["mode"] == "nominal", "ℓ₁ must not trigger a refusal"
    assert res.feasible
    assert len(res.term_specs) == len(even_positions)
    # The weights are re-mapped, but a caller that maps RETURNED term positions
    # back onto its own unfiltered library (sparse_sos does) would alias. That
    # combination has to be visible, not silent.
    assert "MUST re-map" in capsys.readouterr().err


def test_a_wrong_length_l1_weights_still_raises_under_the_reduction():
    """The reduction must not mask a caller bug behind its re-map."""
    spec = ud.create_twisting_spec()
    terms = _library(spec, 5)
    with pytest.raises(ValueError, match="l1_weights"):
        _design(spec, 5, parity_reduction=True, terms=terms,
                l1_weights=np.ones(len(terms) + 3))


# =============================================================================
# Gram naming stays on the FULL octant index (finding 3)
# =============================================================================

@pytest.mark.parametrize("y_parity_blocks", [False, True])
def test_gram_block_names_use_full_octant_indices(y_parity_blocks):
    """A reduced run must name its blocks as the unreduced run names the same
    octants, so ``Q_V_*`` / ``Q_W_*`` records stay comparable across the two.

    Checked on the constraint objects themselves: the reduced STA run at n = 2
    must emit Q_V_0 and Q_V_2 (the s_1 = +1 octants), never Q_V_0 and Q_V_1
    (the loop-position naming a naive split would produce).

    WP P5-04: with the y-parity block decomposition on, ONE Gram block becomes
    one PSD variable per parity class of its basis, named
    ``<block>#p<class key>``. The block IDENTITY — which is what finding 3 is
    about and what a ``Q_V_*`` record is keyed by — is the part before the
    ``#``, and it must be the full octant index under either encoding. Run both
    ways so the invariant cannot be lost to the default.
    """
    import cvxpy as cp

    spec = ud.create_sta_spec()
    captured = {}
    real_problem = cp.Problem

    class _Spy(cp.Problem):
        def __init__(self, objective, constraints=None, **kw):
            names = set()
            for c in (constraints or []):
                for v in c.variables():
                    if v.name().startswith(("Q_V_", "Q_W_")):
                        names.add(v.name())
            captured.setdefault("names", names)
            super().__init__(objective, constraints, **kw)

    cp.Problem = _Spy
    try:
        res = _design(spec, 3, parity_reduction=True,
                      y_parity_blocks=y_parity_blocks)
    finally:
        cp.Problem = real_problem

    assert res.feasible
    names = captured["names"]
    assert all(("#p" in n) == y_parity_blocks for n in names), sorted(names)
    blocks = {n.split("#", 1)[0] for n in names}
    v_names = sorted(n for n in blocks if n.startswith("Q_V_"))
    w_names = sorted(n for n in blocks if n.startswith("Q_W_"))
    assert v_names == ["Q_V_0", "Q_V_2"]
    assert w_names == ["Q_W_0_0", "Q_W_2_0"]
