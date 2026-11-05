"""Design-stage Gram bundle (WP P2-01/P2-02): schema, scale, hash, ε rows.

The MOSEK_DIRECT design solve already recovers one Gram per emitted PSD block.
``JointSOSResult.gram_certificates`` now persists them so the exact-rational
phase can consume the DESIGN Grams instead of re-solving (toolchain D -> X).

What is pinned here:

* the bundle covers every emitted block (V and W) with the block identity;
* the entries are stated in the TRUE coefficient scale, so ``zᵀQz`` reproduces
  the target rebuilt from ``V_gf`` and ``epsilon`` (this is the property that
  lets ``exact_verify_captures`` accept them with no re-solve);
* the content hash covers metadata AND Gram bytes, and moves when either does;
* the capture is switchable and switching it off changes no verdict;
* the W blocks carry the same pure-power ε rows as V whenever the weighted-
  sphere clearing is inactive, so ``exact_verify_captures`` accepts them too
  (P2-02) — and carry a TYPED reason, not a bare zero, when they cannot.
"""
import numpy as np
import pytest

from gf_framework.algebra.enumerate import (
    enumerate_cross_terms,
    enumerate_diagonal_terms,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.discovery.universal_discovery import (
    create_cta_spec,
    create_levant3d_spec,
    create_pidsmc_spec,
    create_power_rate_spec,
    create_sta_spec,
)
from gf_framework.verification.exact_gate import (
    exact_neg_lie_derivative,
    exact_octant_target,
    exact_terms_from_gf,
    exact_verify_captures,
)
from gf_framework.verification.joint_sos import (
    GRAM_BUNDLE_HASHED_KEYS,
    GRAM_BUNDLE_SCHEMA,
    JointSOSOptimizer,
    gram_bundle_entry_hash,
    gram_bundle_fingerprint,
    term_to_y_monomial,
)
from gf_framework.verification.mosek_direct import MOSEK_AVAILABLE

needs_mosek = pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not installed")

#: Keys ``exact_gate.exact_verify_captures`` reads off a capture.
CONSUMED_KEYS = ("signs", "octant_index", "eps", "mu", "y_degree", "target",
                 "basis", "gram")


def _library(spec, m):
    """The μ-filtered C¹ library discover_gf solves over."""
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return list(diag) + list(cross)


def _design(spec, m, **opt_kwargs):
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                            allow_solver_fallback=False, **opt_kwargs)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf,
        active_terms=_library(spec, m),
        dynamics_gf_vertices=spec.dynamics_gf_vertices)


def _gram_poly(entry, n_vars):
    """z(y)ᵀ Q z(y) as a coefficient dict, from the stored Gram and basis."""
    g = (entry["gram"] + entry["gram"].T) / 2.0
    basis = entry["basis"]
    poly = {}
    for i in range(len(basis)):
        for j in range(len(basis)):
            mono = tuple(basis[i][k] + basis[j][k] for k in range(n_vars))
            poly[mono] = poly.get(mono, 0.0) + float(g[i, j])
    return poly


def _residual_l1(entry, n_vars):
    poly = _gram_poly(entry, n_vars)
    target = entry["target"]
    return sum(abs(target.get(m, 0.0) - poly.get(m, 0.0))
               for m in set(poly) | set(target))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@needs_mosek
def test_pidsmc_design_bundle_shape_and_identity():
    """8 V blocks (dim 28) + 8 W blocks (dim 21), one per emitted PSD block."""
    spec = create_pidsmc_spec()
    res = _design(spec, 6)
    assert res.feasible, res.message
    bundle = res.gram_certificates
    assert bundle is not None and len(bundle) == 16

    v = [e for e in bundle if e["block"] == "V"]
    w = [e for e in bundle if e["block"] == "W"]
    assert len(v) == 8 and len(w) == 8
    assert {e["dim"] for e in v} == {28}
    assert {e["dim"] for e in w} == {21}
    # Block identity: full octant coverage on both stages, names carry it.
    assert {e["octant_index"] for e in v} == set(range(8))
    assert {tuple(e["signs"]) for e in v} == {tuple(e["signs"]) for e in w}
    assert {e["name"] for e in v} == {f"V_oct{i}" for i in range(8)}
    assert {e["name"] for e in w} == {f"W_oct{i}_v0" for i in range(8)}
    assert {e["endpoint"] for e in v} == {None}
    assert {e["endpoint"] for e in w} == {0}
    # No parity reduction was requested, so no parity metadata is emitted.
    assert {e["parity_class"] for e in bundle} == {None}

    for e in bundle:
        assert e["schema"] == GRAM_BUNDLE_SCHEMA
        for key in CONSUMED_KEYS:
            assert key in e, key
        for key in GRAM_BUNDLE_HASHED_KEYS:
            assert key in e, key
        assert e["dtype"] == "float64"
        assert e["gram"].dtype == np.float64
        assert e["gram"].shape == (e["dim"], e["dim"])
        assert len(e["basis"]) == e["dim"]
        assert all(len(b) == 3 for b in e["basis"])
        assert e["mu"] == bundle[0]["mu"]
    # Both stages carry the ε margin: pidsmc needs no weighted-sphere
    # clearing, so the W block gets the same pure-power ε rows as V (P2-02).
    assert {e["eps"] for e in v} == {1e-4}
    assert {e["eps"] for e in w} == {1e-4}
    assert {e["eps_reason"] for e in bundle} == {""}


@needs_mosek
def test_design_bundle_is_a_valid_gram_certificate():
    """zᵀQz reproduces the stored target: the persisted triple is coherent."""
    spec = create_pidsmc_spec()
    res = _design(spec, 6)
    assert res.feasible, res.message
    for e in res.gram_certificates:
        assert _residual_l1(e, 3) < 1e-8, e["name"]
        assert float(np.linalg.eigvalsh(
            (e["gram"] + e["gram"].T) / 2.0).min()) > -1e-8, e["name"]


@needs_mosek
def test_exact_gate_accepts_design_v_grams_without_resolve():
    """The acceptance of P2-01: X consumes D's Gram directly (D1/D2/R3)."""
    spec = create_pidsmc_spec()
    res = _design(spec, 6)
    assert res.feasible, res.message
    v_caps = [e for e in res.gram_certificates if e["block"] == "V"]
    exact = exact_verify_captures(exact_terms_from_gf(res.V_gf), v_caps, 3)
    assert exact.ok, exact.reason
    assert len(exact.octants) == 8
    assert 0.0 < exact.worst_charge_ratio <= 1.0


# ---------------------------------------------------------------------------
# Scale: the bundle is stated in TRUE coefficients, not the SDP's scaled ones
# ---------------------------------------------------------------------------

@needs_mosek
@pytest.mark.parametrize("coefficient_scale", [1.0, 50.0])
def test_v_target_matches_true_scale_v_gf(coefficient_scale):
    """target == Σ_j sign_j·coef_j·y^{exp_j} − ε on the pure top powers.

    The SDP solves on ``coefficient_scale·alpha_true`` with ``ε`` scaled to
    match; if the bundle forgot to divide either the Gram or the target back,
    this comparison against ``V_gf`` (true scale) fails for scale ≠ 1.
    """
    spec = create_sta_spec()
    res = _design(spec, 3, coefficient_scale=coefficient_scale)
    assert res.feasible, res.message
    specs = res.term_specs
    coeffs = [float(t.coefficient) for t in res.V_gf.terms]
    for e in (x for x in res.gram_certificates if x["block"] == "V"):
        mu, signs, d = tuple(e["mu"]), tuple(e["signs"]), e["y_degree"]
        rebuilt = {}
        for j, sp in enumerate(specs):
            y_exp, sign = term_to_y_monomial(sp, mu, signs)
            rebuilt[y_exp] = rebuilt.get(y_exp, 0.0) + float(sign) * coeffs[j]
        for i in range(2):
            pure = tuple(d if t == i else 0 for t in range(2))
            rebuilt[pure] = rebuilt.get(pure, 0.0) - e["eps"]
        for mono in set(rebuilt) | set(e["target"]):
            assert e["target"].get(mono, 0.0) == pytest.approx(
                rebuilt.get(mono, 0.0), abs=1e-9), (e["name"], mono)
        assert e["coefficient_scale"] == coefficient_scale


# ---------------------------------------------------------------------------
# Content hash (what P2-03's input-hash refusal rests on)
# ---------------------------------------------------------------------------

def _entry():
    return {
        "schema": GRAM_BUNDLE_SCHEMA, "block": "V", "name": "V_oct0",
        "signs": (1, 1), "octant_index": 0, "endpoint": None,
        "parity_class": None, "eps": 1e-4, "mu": (2, 2), "y_degree": 4,
        "clearing_power": 0, "coefficient_scale": 1.0, "dim": 2,
        "dtype": "float64",
        "basis": [(2, 0), (0, 2)],
        "target": {(4, 0): 1.0, (0, 4): 2.0},
        "gram": np.array([[1.0, 0.0], [0.0, 2.0]]),
    }


def test_entry_hash_is_deterministic_and_order_independent():
    a = _entry()
    b = {k: a[k] for k in reversed(list(a))}
    assert gram_bundle_entry_hash(a) == gram_bundle_entry_hash(b)
    # Insertion order of the target dict must not matter either.
    c = _entry()
    c["target"] = {(0, 4): 2.0, (4, 0): 1.0}
    assert gram_bundle_entry_hash(c) == gram_bundle_entry_hash(a)


@pytest.mark.parametrize("mutate", [
    lambda e: e.__setitem__("gram", e["gram"] + np.array([[1e-12, 0.0],
                                                          [0.0, 0.0]])),
    lambda e: e.__setitem__("signs", (1, -1)),
    lambda e: e.__setitem__("eps", 1e-5),
    lambda e: e.__setitem__("target", {(4, 0): 1.0, (0, 4): 2.5}),
    lambda e: e.__setitem__("basis", [(0, 2), (2, 0)]),
    lambda e: e.__setitem__("octant_index", 1),
    lambda e: e.__setitem__("mu", (4, 4)),
], ids=["gram", "signs", "eps", "target", "basis_order", "octant", "mu"])
def test_entry_hash_moves_on_any_mutation(mutate):
    base = gram_bundle_entry_hash(_entry())
    bad = _entry()
    mutate(bad)
    assert gram_bundle_entry_hash(bad) != base


def test_fingerprint_covers_order_and_membership():
    a, b = _entry(), _entry()
    b["name"] = "V_oct1"
    b["octant_index"] = 1
    b["signs"] = (1, -1)
    assert gram_bundle_fingerprint([a, b]) != gram_bundle_fingerprint([b, a])
    assert gram_bundle_fingerprint([a]) != gram_bundle_fingerprint([a, b])
    assert gram_bundle_fingerprint(None) == gram_bundle_fingerprint([])


@needs_mosek
def test_bundle_entries_carry_their_own_hash():
    spec = create_sta_spec()
    res = _design(spec, 3)
    assert res.feasible, res.message
    for e in res.gram_certificates:
        assert e["hash"] == gram_bundle_entry_hash(e), e["name"]


# ---------------------------------------------------------------------------
# Opt-out and non-interference
# ---------------------------------------------------------------------------

@needs_mosek
def test_capture_can_be_switched_off_without_changing_the_verdict():
    spec = create_pidsmc_spec()
    on = _design(spec, 6)
    off = _design(spec, 6, capture_design_grams=False)
    assert on.feasible and off.feasible
    assert off.gram_certificates is None
    assert on.solver_status == off.solver_status
    assert on.solver_used == off.solver_used == "MOSEK_DIRECT"
    # The scalar diagnostics are computed from the same Grams either way.
    assert off.gram_residual_l1 == pytest.approx(on.gram_residual_l1, rel=1e-9)
    assert off.gram_min_eig == pytest.approx(on.gram_min_eig, rel=1e-9)


@needs_mosek
@pytest.mark.parametrize("delta0,mode,n_blocks", [(None, "nominal", 4),
                                                  (0.5, "paired", 6)],
                         ids=["nominal", "paired"])
def test_parity_reduced_bundle_records_its_partner_labels(delta0, mode, n_blocks):
    """A reduced design emits REPRESENTATIVE blocks only; parity_class says so.

    The bundle then deliberately does not cover all 2ⁿ octants, which is what
    a consumer must be able to see before running a coverage check on it.
    Under the paired reduction the partner is always the OTHER endpoint
    (Corollary 10.4), never the same one.
    """
    spec = (create_sta_spec() if delta0 is None
            else create_sta_spec(k1=1.5, k2=1.1, delta0=delta0))
    res = _design(spec, 3, parity_reduction=True)
    assert res.feasible, res.message
    assert res.parity_reduction["mode"] == mode
    bundle = res.gram_certificates
    assert bundle is not None and len(bundle) == n_blocks
    v = [e for e in bundle if e["block"] == "V"]
    assert len(v) == 2 and {e["octant_index"] for e in v} == {0, 2}
    for e in bundle:
        pc = e["parity_class"]
        assert pc is not None and pc["mode"] == mode
        assert pc["partner_octant_index"] != e["octant_index"]
        if e["block"] == "W" and mode == "paired":
            assert pc["partner_endpoint"] == 1 - e["endpoint"]
        else:
            assert pc["partner_endpoint"] is None
        assert e["hash"] == gram_bundle_entry_hash(e)


def test_cvxpy_design_path_carries_no_bundle():
    """The CVXPY encoding recovers no Grams, so it persists none."""
    spec = create_sta_spec()
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="CLARABEL",
                            allow_solver_fallback=False)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=3, dynamics_gf=spec.dynamics_gf,
        active_terms=_library(spec, 3))
    assert res.feasible, res.message
    assert res.gram_certificates is None


# ---------------------------------------------------------------------------
# W ε rows (WP P2-02): the design W Gram is exact-gateable in the
# clearing-inactive case, and carries a TYPED reason when it is not
# ---------------------------------------------------------------------------

def _w_terms(res, dynamics):
    """Ŵ = −dV̂/dt, exactly, from the designed V and the rationalized dynamics."""
    return exact_neg_lie_derivative(
        exact_terms_from_gf(res.V_gf),
        [exact_terms_from_gf(f) for f in dynamics])


@needs_mosek
@pytest.mark.parametrize("factory,m,n_vars", [
    (create_pidsmc_spec, 6, 3),
    (create_levant3d_spec, 5, 3),
    (create_cta_spec, 9, 3),
], ids=["pidsmc_m6", "levant3d_m5", "cta_m9"])
def test_exact_gate_accepts_design_w_grams_without_resolve(factory, m, n_vars):
    """The acceptance of P2-02: X consumes D's W Gram directly, no re-solve.

    Before the ε rows the same call returned ``epsilon_not_positive(0)``: a
    zero margin makes Proposition 7's budget ε/2 = 0, so the design W Gram
    could never be re-proved exactly without re-solving it.
    """
    spec = factory()
    res = _design(spec, m)
    assert res.feasible, res.message
    w_caps = [e for e in res.gram_certificates if e["block"] == "W"]
    assert len(w_caps) == 2 ** n_vars
    assert {e["eps"] for e in w_caps} == {1e-4}
    assert {e["clearing_power"] for e in w_caps} == {0}

    exact = exact_verify_captures(_w_terms(res, spec.dynamics_gf), w_caps, n_vars)
    assert exact.ok, exact.reason
    assert len(exact.octants) == 2 ** n_vars
    assert 0.0 < exact.worst_charge_ratio <= 1.0
    # ... and V is still accepted at the same time, on the same bundle.
    v_caps = [e for e in res.gram_certificates if e["block"] == "V"]
    assert exact_verify_captures(
        exact_terms_from_gf(res.V_gf), v_caps, n_vars).ok


@needs_mosek
def test_w_epsilon_rows_shift_exactly_the_n_pure_top_powers():
    """The stored W target is the associated form minus ε on each yᵢ^d.

    ``exact_octant_target`` subtracts ε from ALL n pure powers
    unconditionally, so the design must emit all n rows or none — a partial
    set would silently disagree with the exact target it is checked against.
    """
    from fractions import Fraction

    spec = create_sta_spec()
    res = _design(spec, 3)
    assert res.feasible, res.message
    w_terms = _w_terms(res, spec.dynamics_gf)
    for e in (x for x in res.gram_certificates if x["block"] == "W"):
        unshifted, d = exact_octant_target(
            w_terms, tuple(e["mu"]), tuple(e["signs"]), Fraction(0))
        assert d == e["y_degree"]
        pures = {tuple(d if t == i else 0 for t in range(2)) for i in range(2)}
        assert pures <= set(unshifted), e["name"]
        for mono in set(unshifted) | set(e["target"]):
            shift = e["eps"] if mono in pures else 0.0
            assert e["target"].get(mono, 0.0) == pytest.approx(
                float(unshifted.get(mono, 0)) - shift, abs=1e-9), (e["name"], mono)


@needs_mosek
def test_w_epsilon_rows_can_be_switched_off_with_a_typed_reason():
    """The opt-out restores the historical zero-margin W block."""
    spec = create_pidsmc_spec()
    res = _design(spec, 6, w_epsilon_rows=False)
    assert res.feasible, res.message
    w_caps = [e for e in res.gram_certificates if e["block"] == "W"]
    assert {e["eps"] for e in w_caps} == {0.0}
    assert {e["eps_reason"] for e in w_caps} == {"w_epsilon_rows_disabled"}
    # V is unaffected by the W margin either way.
    v_caps = [e for e in res.gram_certificates if e["block"] == "V"]
    assert {e["eps"] for e in v_caps} == {1e-4}
    # And this is exactly the state the exact gate refuses.
    exact = exact_verify_captures(_w_terms(res, spec.dynamics_gf), w_caps, 3)
    assert not exact.ok
    assert exact.reason.startswith("epsilon_not_positive")


@needs_mosek
def test_robust_design_carries_the_margin_on_both_inclusion_endpoints():
    """Two-endpoint inclusion: each endpoint is its own exact-gateable set."""
    spec = create_sta_spec(k1=1.5, k2=1.1, delta0=0.5)
    res = _design(spec, 3)
    assert res.feasible, res.message
    w_caps = [e for e in res.gram_certificates if e["block"] == "W"]
    assert sorted({e["endpoint"] for e in w_caps}) == [0, 1]
    assert {e["eps"] for e in w_caps} == {1e-4}
    for endpoint, dyn in enumerate(spec.dynamics_gf_vertices):
        caps = [e for e in w_caps if e["endpoint"] == endpoint]
        assert len(caps) == 4
        exact = exact_verify_captures(_w_terms(res, dyn), caps, 2)
        assert exact.ok, (endpoint, exact.reason)


@needs_mosek
def test_parity_reduced_robust_design_keeps_the_w_margin():
    """The reduction drops blocks, not the margin (P2-03 still needs R1)."""
    spec = create_sta_spec(k1=1.5, k2=1.1, delta0=0.5)
    res = _design(spec, 3, parity_reduction=True)
    assert res.feasible, res.message
    assert res.parity_reduction["mode"] == "paired"
    w_caps = [e for e in res.gram_certificates if e["block"] == "W"]
    assert len(w_caps) == 4
    assert {e["eps"] for e in w_caps} == {1e-4}
    assert {e["eps_reason"] for e in w_caps} == {""}
    # Coverage is deliberately partial, so the gate still refuses on octants.
    exact = exact_verify_captures(
        _w_terms(res, spec.dynamics_gf_vertices[0]),
        [e for e in w_caps if e["endpoint"] == 0], 2)
    assert not exact.ok
    assert exact.reason.startswith("octant_coverage_incomplete")


@needs_mosek
def test_odd_degree_w_block_is_lifted_to_an_even_mu():
    """power_rate's W is odd at the design μ, so the W blocks get 2μ (P2-09).

    Historically (P2-02/P2-04) these two blocks came out at y-degree 3 with
    ``eps = 0.0`` and ``eps_reason = "w_degree_not_positive_even"``, on the
    non-homogeneous basis ``[1, y, y²]`` whose product ``y⁴`` no constraint row
    covered — ``Q[2,2] = 6.16e+04``. Doubling μ makes the block homogeneous of
    even degree 6 over the basis ``[y³]``, which is exactly the object
    ``verify_gf_sos_full`` builds when it re-solves W on its own.
    """
    spec = create_power_rate_spec()
    res = _design(spec, 2)
    assert res.feasible, res.message
    v_caps = [e for e in res.gram_certificates if e["block"] == "V"]
    w_caps = [e for e in res.gram_certificates if e["block"] == "W"]
    assert {e["mu"] for e in v_caps} == {(2,)}
    assert {e["mu"] for e in w_caps} == {(4,)}
    assert {e["mu_policy"] for e in w_caps} == {"w_mu_doubled_odd_y_degree"}
    assert {e["mu_policy"] for e in v_caps} == {""}
    assert {e["y_degree"] for e in w_caps} == {6}
    assert {e["dim"] for e in w_caps} == {1}
    assert {e["eps"] for e in w_caps} == {1e-4}
    assert {e["eps_reason"] for e in w_caps} == {""}
    # The block now IS a certificate of its target: zᵀQz reproduces it.
    for e in w_caps:
        assert _residual_l1(e, 1) < 1e-8


@needs_mosek
def test_even_degree_w_blocks_keep_the_design_mu():
    """The μ policy fires ONLY on an odd W y-degree; sta is the control."""
    spec = create_sta_spec()
    res = _design(spec, 3)
    assert res.feasible, res.message
    bundle = res.gram_certificates
    v_mu = {e["mu"] for e in bundle if e["block"] == "V"}
    w_mu = {e["mu"] for e in bundle if e["block"] == "W"}
    assert v_mu == w_mu
    assert {e["mu_policy"] for e in bundle} == {""}
    assert all(e["y_degree"] % 2 == 0 for e in bundle)


@pytest.mark.parametrize("clearing_power,flag,reason", [
    (2, True, "weighted_sphere_clearing_active"),
    (1, True, "weighted_sphere_clearing_active"),
    (0, False, "w_epsilon_rows_disabled"),
], ids=["clearing_k2", "clearing_k1", "opt_out"])
def test_w_epsilon_rows_withheld_reasons(clearing_power, flag, reason):
    """No margin is ever emitted onto a block that cannot carry one.

    With the weighted-sphere clearing active the block is
    ``∏ᵢ|xᵢ|^{k·ρ/rᵢ}·W``, which vanishes on every axis: an ε shift there
    asks for a strictly negative pure-power coefficient, which no SOS form
    can have, so the SDP would be spuriously infeasible.
    """
    opt = JointSOSOptimizer(weights_r=(2, 1), w_epsilon_rows=flag)
    monomials = [[(4, 0), (2, 2), (0, 4), (3, 1), (1, 3)]]
    rows, monos, got = opt._w_epsilon_rows(monomials, clearing_power)
    assert got == reason
    assert rows == [[]] and monos == [()]


def test_w_epsilon_rows_are_the_pure_top_powers():
    opt = JointSOSOptimizer(weights_r=(2, 1))
    monomials = [[(0, 0), (4, 0), (2, 2), (0, 4), (3, 1)]]
    rows, monos, reason = opt._w_epsilon_rows(monomials, 0)
    assert reason == ""
    assert rows == [[1, 3]]
    assert monos == [((4, 0), (0, 4))]


def test_w_epsilon_rows_refuse_an_incomplete_pure_power_support():
    """All n pure powers or none: a partial set would disagree with the gate."""
    opt = JointSOSOptimizer(weights_r=(2, 1))
    rows, monos, reason = opt._w_epsilon_rows([[(4, 0), (2, 2), (3, 1)]], 0)
    assert reason == "pure_power_rows_incomplete"
    assert rows == [[]] and monos == [()]
