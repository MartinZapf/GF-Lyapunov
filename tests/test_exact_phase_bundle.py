"""``run_exact_phase`` consuming the design Gram bundle (WP P2-03, R3).

The exact phase used to PROPOSE its own witness with an independent float
re-solve through ``verify_gf_sos_full`` (the CVXPY encoding). Under decisions
D1/D2 the DESIGN Gram is the witness instead, which makes the citability
chain D -> X -> P and removes the last certifying float solve from the exact
route.

What is pinned here:

* with a bundle supplied, ``verify_gf_sos_full`` is NOT called — the
  certifying path runs no float solve at all;
* the exact targets are still rebuilt from scratch in Q, so the bundle
  contributes Q and its basis and nothing else;
* a modified entry, a modified Gram, a missing hash and a wrong fingerprint
  are all refused as ``gram_bundle_hash_mismatch``;
* a bundle from a different instance, a different margin, a different octant
  arity or a different inclusion structure is refused as
  ``gram_bundle_input_mismatch``;
* a parity-REDUCED bundle (2^(n-1) blocks) is expanded from ``parity_class``
  to all 2^n octants, the record says which octant came from a partner, and
  the partner target is re-derived rather than assumed;
* the bundle survives a ``.json`` / ``.json.gz`` / ``.npz`` round trip with
  the same fingerprint;
* with no bundle the historical re-solve path is unchanged.
"""
import json

import numpy as np
import pytest

from gf_framework.algebra.enumerate import (
    enumerate_cross_terms,
    enumerate_diagonal_terms,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.discovery.universal_discovery import (
    create_dsta_spec,
    create_sta_spec,
)
from gf_framework.verification import exact_phase as ep
from gf_framework.verification.exact_phase import (
    WITNESS_DESIGN_BUNDLE,
    WITNESS_RESOLVE,
    GramBundleError,
    gram_bundle_from_jsonable,
    gram_bundle_to_jsonable,
    load_gram_bundle,
    prepare_gram_bundle,
    run_exact_phase,
    save_gram_bundle,
)
from gf_framework.verification.joint_sos import (
    GRAM_BUNDLE_SCHEMA,
    JointSOSOptimizer,
    gram_bundle_entry_hash,
    gram_bundle_fingerprint,
)
from gf_framework.verification.mosek_direct import MOSEK_AVAILABLE

needs_mosek = pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not installed")

#: The design margin. The bundle can only witness the margin the design
#: actually subtracted, so the exact phase is run at the same value.
EPS = 1e-4


def _library(spec, m):
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return list(diag) + list(cross)


def _design(spec, m, **opt_kwargs):
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                            allow_solver_fallback=False, epsilon=EPS,
                            **opt_kwargs)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf,
        active_terms=_library(spec, m),
        dynamics_gf_vertices=spec.dynamics_gf_vertices)


@pytest.fixture(scope="module")
def sta_design():
    """sta at m = 3 (n = 2): the smallest design that carries V and W blocks."""
    spec = create_sta_spec()
    res = _design(spec, 3)
    assert res.feasible and res.gram_certificates
    return spec, res


@pytest.fixture(scope="module")
def dsta_design():
    """dsta at m = 3: a DIFFERENT instance with the same n_vars as sta."""
    spec = create_dsta_spec()
    res = _design(spec, 3)
    assert res.feasible and res.gram_certificates
    return spec, res


def _phase(spec, res, **kwargs):
    return run_exact_phase(res.V_gf, [spec.dynamics_gf], len(spec.weights_r),
                           epsilon=EPS, **kwargs)


def _copy(entries):
    """A deep-enough copy: entry dicts and their Grams, targets left shared."""
    out = []
    for entry in entries:
        clone = dict(entry)
        clone["gram"] = np.array(entry["gram"], dtype=np.float64)
        out.append(clone)
    return out


# ---------------------------------------------------------------------------
# 1. The bundle proves, and it proves without solving
# ---------------------------------------------------------------------------

@needs_mosek
def test_design_bundle_proves_the_phase(sta_design):
    spec, res = sta_design
    phase = _phase(spec, res, gram_bundle=res)
    assert phase.exact_pass
    assert phase.witness_source == WITNESS_DESIGN_BUNDLE
    assert phase.gram_bundle_fingerprint == gram_bundle_fingerprint(
        res.gram_certificates)
    assert 0.0 < phase.worst_charge_ratio < 1.0


@needs_mosek
def test_no_float_solve_runs_on_the_bundle_path(sta_design, monkeypatch):
    """The soundness point of R3: no certifying float solve, at all.

    ``verify_gf_sos_full`` is the CVXPY re-solve. If the bundle path touched
    it the exact phase would still be running a float SDP on the certifying
    route, which decision D1 forbids.
    """
    spec, res = sta_design

    def _boom(*args, **kwargs):  # pragma: no cover - the point is that it never runs
        raise AssertionError("verify_gf_sos_full was called on the bundle path")

    monkeypatch.setattr(ep, "verify_gf_sos_full", _boom)
    phase = _phase(spec, res, gram_bundle=res)
    assert phase.exact_pass


@needs_mosek
def test_targets_are_rebuilt_exactly_not_read_from_the_bundle(sta_design,
                                                             monkeypatch):
    """The bundle contributes Q and its basis; the target comes from Q(x).

    ``exact_octant_target`` is the from-scratch rational rebuild. It must run
    once per octant per form even when a bundle carries a stored target.
    """
    spec, res = sta_design
    calls = []
    real = ep.exact_octant_target

    def _counting(*args, **kwargs):
        calls.append(args[1:])
        return real(*args, **kwargs)

    monkeypatch.setattr(ep, "exact_octant_target", _counting)
    phase = _phase(spec, res, gram_bundle=res)
    assert phase.exact_pass
    # 2^n octants for V and 2^n for W, in the input check alone.
    assert len(calls) >= 2 * 2 ** len(spec.weights_r)


@needs_mosek
def test_witness_record_names_every_octant(sta_design):
    spec, res = sta_design
    phase = _phase(spec, res, gram_bundle=res)
    record = phase.witness_record()
    assert record["source"] == WITNESS_DESIGN_BUNDLE
    assert record["gram_bundle_fingerprint"] == phase.gram_bundle_fingerprint
    assert len(record["V"]) == 2 ** len(spec.weights_r)
    assert len(record["endpoints"]) == 1
    assert len(record["endpoints"][0]) == 2 ** len(spec.weights_r)
    for row in record["V"]:
        assert row["entry_hash"]
        assert row["from_parity_partner"] is False
    # JSON-ready: an artifact writer must not have to convert anything.
    json.dumps(record)


# ---------------------------------------------------------------------------
# 2. Hash refusals
# ---------------------------------------------------------------------------

@needs_mosek
def test_modified_gram_is_refused(sta_design):
    spec, res = sta_design
    entries = _copy(res.gram_certificates)
    entries[0]["gram"][0, 0] += 1e-9
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=entries)
    assert exc.value.reason == "gram_bundle_hash_mismatch"


@needs_mosek
def test_modified_metadata_is_refused(sta_design):
    spec, res = sta_design
    entries = _copy(res.gram_certificates)
    entries[0]["eps"] = float(entries[0]["eps"]) * 2
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=entries)
    assert exc.value.reason == "gram_bundle_hash_mismatch"


@needs_mosek
def test_entry_without_a_hash_is_refused(sta_design):
    spec, res = sta_design
    entries = _copy(res.gram_certificates)
    entries[0].pop("hash")
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=entries)
    assert exc.value.reason == "gram_bundle_hash_mismatch"


@needs_mosek
def test_wrong_expected_fingerprint_is_refused(sta_design):
    spec, res = sta_design
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=res,
               expect_bundle_fingerprint="0" * 64)
    assert exc.value.reason == "gram_bundle_hash_mismatch"


@needs_mosek
def test_document_fingerprint_is_checked(sta_design):
    spec, res = sta_design
    document = gram_bundle_to_jsonable(res.gram_certificates)
    document["fingerprint"] = "1" * 64
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=document)
    assert exc.value.reason == "gram_bundle_hash_mismatch"


@needs_mosek
def test_unknown_schema_is_refused(sta_design):
    spec, res = sta_design
    entries = _copy(res.gram_certificates)
    entries[0]["schema"] = "gf.design_grams/99"
    entries[0]["hash"] = gram_bundle_entry_hash(entries[0])
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=entries)
    assert exc.value.reason == "gram_bundle_schema_unknown"


# ---------------------------------------------------------------------------
# 3. Input refusals (right shape, wrong instance)
# ---------------------------------------------------------------------------

@needs_mosek
def test_bundle_from_a_different_system_is_refused(sta_design, dsta_design):
    """Same n_vars, same degree, same margin, different dynamics and V.

    Every hash in the dsta bundle is valid, so only the target comparison can
    catch this. It is the substantive half of the refusal.
    """
    spec, _res = sta_design
    _dspec, dres = dsta_design
    with pytest.raises(GramBundleError) as exc:
        run_exact_phase(_res.V_gf, [spec.dynamics_gf], 2, epsilon=EPS,
                        gram_bundle=dres)
    assert exc.value.reason == "gram_bundle_input_mismatch"


@needs_mosek
def test_bundle_at_a_different_margin_is_refused(sta_design):
    spec, res = sta_design
    with pytest.raises(GramBundleError) as exc:
        run_exact_phase(res.V_gf, [spec.dynamics_gf], 2, epsilon=1e-6,
                        gram_bundle=res)
    assert exc.value.reason == "gram_bundle_input_mismatch"


@needs_mosek
def test_bundle_with_the_wrong_octant_arity_is_refused(sta_design):
    spec, res = sta_design
    with pytest.raises(GramBundleError) as exc:
        run_exact_phase(res.V_gf, [spec.dynamics_gf], 3, epsilon=EPS,
                        gram_bundle=res)
    assert exc.value.reason == "gram_bundle_input_mismatch"


@needs_mosek
def test_bundle_with_the_wrong_inclusion_structure_is_refused(sta_design):
    """A nominal bundle carries one W vertex; two dynamics need two."""
    spec, res = sta_design
    with pytest.raises(GramBundleError) as exc:
        run_exact_phase(res.V_gf, [spec.dynamics_gf, spec.dynamics_gf], 2,
                        epsilon=EPS, gram_bundle=res)
    assert exc.value.reason == "gram_bundle_input_mismatch"


@needs_mosek
def test_incomplete_octant_coverage_is_refused(sta_design):
    spec, res = sta_design
    entries = [e for e in _copy(res.gram_certificates)
               if not (e["block"] == "V" and e["octant_index"] == 1)]
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=entries)
    assert exc.value.reason == "gram_bundle_coverage_incomplete"


@needs_mosek
def test_block_without_epsilon_rows_falls_through_with_its_design_reason(sta_design):
    """P2-02's typed ``eps_reason`` is what a consumer must report.

    Updated by P2-05: a block with no ε rows has no bundle witness, which is a
    DESIGN-side limit, not a wrong bundle. It no longer aborts the phase — it
    is recorded as a failed witness attempt carrying the typed reason and the
    design's own ``eps_reason``, and the phase falls through to the re-solve.
    """
    spec, res = sta_design
    entries = _copy(res.gram_certificates)
    for entry in entries:
        if entry["block"] == "W":
            entry["eps"] = 0.0
            entry["eps_reason"] = "w_degree_not_positive_even"
            entry["hash"] = gram_bundle_entry_hash(entry)
    phase = _phase(spec, res, gram_bundle=entries)
    first = phase.witness_attempts[0]
    assert first["source"] == WITNESS_DESIGN_BUNDLE
    assert first["ok"] is False
    assert first["shortfall"] == "gram_bundle_margin_missing"
    assert "w_degree_not_positive_even" in first["detail"]
    assert phase.witness_source == WITNESS_RESOLVE
    assert phase.exact_pass


@needs_mosek
def test_clearing_active_block_falls_through_to_the_resolve(sta_design):
    spec, res = sta_design
    entries = _copy(res.gram_certificates)
    for entry in entries:
        if entry["block"] == "W":
            entry["clearing_power"] = 2
            entry["hash"] = gram_bundle_entry_hash(entry)
    phase = _phase(spec, res, gram_bundle=entries)
    assert phase.witness_attempts[0]["shortfall"] == "gram_bundle_clearing_active"
    assert phase.witness_source == WITNESS_RESOLVE
    assert phase.exact_pass


def test_empty_bundle_is_refused():
    with pytest.raises(GramBundleError) as exc:
        prepare_gram_bundle([], 2)
    assert exc.value.reason == "gram_bundle_empty"


# ---------------------------------------------------------------------------
# 4. Parity-reduced bundles
# ---------------------------------------------------------------------------

@needs_mosek
def test_parity_reduced_bundle_is_expanded_and_proves(sta_design):
    """A reduced design emits 2^(n-1) V blocks; all 2^n octants must be proved."""
    spec = create_sta_spec()
    res = _design(spec, 3, parity_reduction=True)
    assert res.feasible and res.parity_reduction
    assert res.parity_reduction["mode"] == "nominal"
    n = len(spec.weights_r)
    v_entries = [e for e in res.gram_certificates if e["block"] == "V"]
    assert len(v_entries) == 2 ** (n - 1)

    phase = _phase(spec, res, gram_bundle=res)
    assert phase.exact_pass
    assert phase.gram_bundle_parity_expanded is True
    assert len(phase.v_exact.octants) == 2 ** n
    witnesses = phase.witness_record()["V"]
    assert sum(1 for w in witnesses if w["from_parity_partner"]) == 2 ** (n - 1)
    for row in witnesses:
        if row["from_parity_partner"]:
            assert row["parity_source"]["entry_hash"]
            assert row["parity_source"]["octant_index"] != row["octant_index"]


@needs_mosek
def test_parity_expansion_covers_every_octant_exactly_once(sta_design):
    spec = create_sta_spec()
    res = _design(spec, 3, parity_reduction=True)
    prepared = prepare_gram_bundle(res, len(spec.weights_r))
    for group in [prepared.v] + list(prepared.w.values()):
        indices = [e["octant_index"] for e in group]
        assert sorted(indices) == list(range(2 ** len(spec.weights_r)))


def test_paired_expansion_flips_octant_and_endpoint():
    """The paired involution is (s, σ) -> (−s, −σ), not octant halving alone.

    Naive halving at a FIXED endpoint is unsound (``algebra.symmetry``: a
    fixed inclusion endpoint is not odd). This pins the endpoint flip on the
    relabelling itself, without a solve: no perturbed design is reachable
    through the nominal specs, so the paired branch has no system-level run.
    """
    entry = {
        "block": "W", "name": "W_oct2_v0", "signs": (1, -1),
        "octant_index": 2, "endpoint": 0,
        "parity_class": {"mode": "paired", "role": "representative",
                         "partner_octant_index": 1, "partner_endpoint": 1},
    }
    expanded = ep._expand_parity_partners([entry], 2)
    assert len(expanded) == 2
    partner = expanded[1]
    assert partner["signs"] == (-1, 1)
    assert partner["octant_index"] == 1
    assert partner["endpoint"] == 1
    assert partner["hash"] is None
    assert partner["_parity_source"]["octant_index"] == 2
    assert partner["_parity_source"]["endpoint"] == 0


def test_paired_expansion_refuses_a_wrong_endpoint_partner():
    entry = {
        "block": "W", "name": "W_oct2_v0", "signs": (1, -1),
        "octant_index": 2, "endpoint": 0,
        "parity_class": {"mode": "paired", "role": "representative",
                         "partner_octant_index": 1, "partner_endpoint": 0},
    }
    with pytest.raises(GramBundleError) as exc:
        ep._expand_parity_partners([entry], 2)
    assert exc.value.reason == "gram_bundle_parity_inconsistent"


@needs_mosek
def test_inconsistent_parity_class_is_refused(sta_design):
    spec = create_sta_spec()
    res = _design(spec, 3, parity_reduction=True)
    entries = _copy(res.gram_certificates)
    for entry in entries:
        if entry.get("parity_class"):
            entry["parity_class"] = dict(entry["parity_class"])
            entry["parity_class"]["partner_octant_index"] = 0
            entry["hash"] = gram_bundle_entry_hash(entry)
            break
    with pytest.raises(GramBundleError) as exc:
        _phase(spec, res, gram_bundle=entries)
    assert exc.value.reason == "gram_bundle_parity_inconsistent"


# ---------------------------------------------------------------------------
# 5. Serialization round trip
# ---------------------------------------------------------------------------

@needs_mosek
@pytest.mark.parametrize("suffix", [".json", ".json.gz", ".npz"])
def test_bundle_round_trips_through_a_file(sta_design, tmp_path, suffix):
    spec, res = sta_design
    path = save_gram_bundle(tmp_path / f"bundle{suffix}", res.gram_certificates,
                            system="sta")
    reloaded, meta = load_gram_bundle(path)
    assert meta["system"] == "sta"
    assert gram_bundle_fingerprint(reloaded) == gram_bundle_fingerprint(
        res.gram_certificates)
    phase = _phase(spec, res, gram_bundle=path)
    assert phase.exact_pass
    assert phase.witness_source == WITNESS_DESIGN_BUNDLE


@needs_mosek
def test_jsonable_round_trip_preserves_every_hash(sta_design):
    _spec, res = sta_design
    entries, _meta = gram_bundle_from_jsonable(
        gram_bundle_to_jsonable(res.gram_certificates))
    assert len(entries) == len(res.gram_certificates)
    for before, after in zip(res.gram_certificates, entries):
        assert after["schema"] == GRAM_BUNDLE_SCHEMA
        assert gram_bundle_entry_hash(after) == before["hash"]


# ---------------------------------------------------------------------------
# 6. The old path is untouched
# ---------------------------------------------------------------------------

@needs_mosek
def test_no_bundle_keeps_the_resolve_path(sta_design):
    spec, res = sta_design
    phase = _phase(spec, res)
    assert phase.exact_pass
    assert phase.witness_source == WITNESS_RESOLVE
    assert phase.gram_bundle_fingerprint == ""
    assert phase.v_witnesses == []
    assert phase.endpoints[0].witness_source == WITNESS_RESOLVE
    assert phase.witness_record()["source"] == WITNESS_RESOLVE


@needs_mosek
def test_both_witnesses_agree_on_the_verdict(sta_design):
    """Same V, same ε, two witnesses: the verdict must not depend on which."""
    spec, res = sta_design
    bundled = _phase(spec, res, gram_bundle=res)
    resolved = _phase(spec, res)
    assert bundled.exact_pass == resolved.exact_pass is True
    assert bundled.v_exact.mu == resolved.v_exact.mu
    # Different Gram, so the charge may differ; both stay under budget, which
    # is what ``worst_charge_ratio < 1`` states.
    assert bundled.worst_charge_ratio < 1.0
    assert resolved.worst_charge_ratio < 1.0
