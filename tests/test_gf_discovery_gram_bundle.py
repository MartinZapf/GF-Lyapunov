"""P2-11: ``discover_gf`` keeps the design Gram bundle it solved.

Before P2-11 ``_design_full_library`` returned ``res.V_gf`` and dropped the
``JointSOSResult``, so the design Grams — the witness the exact phase re-proves
on (P2-03) — never reached the caller and the publication driver had to
re-solve the same SDP in a sibling script to put them on disk.

These tests pin: the bundle reaches ``GFDiscoveryResult``; it round trips
through ``save_gram_bundle``/``load_gram_bundle`` with every per-entry hash and
the bundle fingerprint intact; the design margin travels with it; and the
bundle is REFUSED (with a typed reason, not silently attached) when the
certified V is not the design proposal.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gf_framework.discovery.gf_discovery as gfd
import gf_framework.discovery.universal_discovery as ud
from gf_framework.verification.exact_phase import (
    load_gram_bundle, save_gram_bundle)
from gf_framework.verification.joint_sos import (
    GRAM_BUNDLE_SCHEMA, gram_bundle_entry_hash, gram_bundle_fingerprint)

try:
    import mosek  # noqa: F401
    HAVE_MOSEK = True
except Exception:                                                # noqa: BLE001
    HAVE_MOSEK = False

requires_mosek = pytest.mark.skipif(
    not HAVE_MOSEK, reason="the design Gram bundle exists only on the "
                           "MOSEK_DIRECT Task-API design path")


@pytest.fixture(scope="module")
def sta_discovery():
    """One real MOSEK_DIRECT design + strict gate on the smallest 2D row."""
    spec = ud.create_sta_spec()
    return gfd.discover_gf(spec, solver="MOSEK_DIRECT", gate_solver="MOSEK",
                           design_fallback=False, gate_fallback=False,
                           max_extra_degree=2)


# --------------------------------------------------------------- population
@requires_mosek
def test_certified_result_carries_the_design_gram_bundle(sta_discovery):
    r = sta_discovery
    assert r.certified is True
    assert r.gram_bundle_reason == ""
    assert r.gram_certificates, "GFDiscoveryResult.gram_certificates is empty"
    # 2^n V blocks plus one W block per octant for a nominal (single-vertex)
    # system: n = 2 -> 8.
    assert len(r.gram_certificates) == 8
    assert {e["block"] for e in r.gram_certificates} == {"V", "W"}


@requires_mosek
def test_every_entry_carries_the_schema_and_the_keys_the_gate_reads(
        sta_discovery):
    for entry in sta_discovery.gram_certificates:
        assert entry["schema"] == GRAM_BUNDLE_SCHEMA
        for key in ("signs", "octant_index", "eps", "mu", "y_degree",
                    "target", "basis", "gram"):
            assert key in entry, key
        assert entry["gram"].shape == (entry["dim"], entry["dim"])
        assert len(entry["basis"]) == entry["dim"]


@requires_mosek
def test_fingerprint_is_the_fingerprint_of_the_entries(sta_discovery):
    assert sta_discovery.gram_bundle_fingerprint == gram_bundle_fingerprint(
        sta_discovery.gram_certificates)


@requires_mosek
def test_design_epsilon_is_the_margin_the_design_subtracted(sta_discovery):
    # The only margin these Grams can witness (P2-03); stage X must be given
    # exactly this value or every row refuses gram_bundle_input_mismatch.
    assert sta_discovery.design_epsilon == pytest.approx(1e-4)
    v_eps = sorted({e["eps"] for e in sta_discovery.gram_certificates
                    if e["block"] == "V"})
    assert v_eps == [pytest.approx(1e-4)]


@requires_mosek
def test_library_order_is_recorded_on_every_entry(sta_discovery):
    # P5-13: a bundle says which library order produced it.
    assert {e.get("library_order")
            for e in sta_discovery.gram_certificates} == {"grlex-v1"}


# ------------------------------------------------------------- round trips
@requires_mosek
@pytest.mark.parametrize("suffix", [".json", ".json.gz", ".npz"])
def test_bundle_round_trips_with_the_hashes_intact(sta_discovery, tmp_path,
                                                   suffix):
    entries = sta_discovery.gram_certificates
    path = save_gram_bundle(tmp_path / f"sta{suffix}", entries,
                            system="sta", epsilon=sta_discovery.design_epsilon,
                            versions={"mosek": "test"})
    loaded, meta = load_gram_bundle(path)

    assert len(loaded) == len(entries)
    for before, after in zip(entries, loaded):
        assert after["hash"] == before["hash"]
        assert gram_bundle_entry_hash(after) == after["hash"]
    assert gram_bundle_fingerprint(loaded) == \
        sta_discovery.gram_bundle_fingerprint
    assert meta["system"] == "sta"
    assert meta["epsilon"] == pytest.approx(1e-4)
    # Version metadata is DOCUMENT-level and therefore not hashed: recording
    # the solver build cannot change the object the exact phase re-proves.
    assert meta["versions"] == {"mosek": "test"}
    assert meta["fingerprint"] == sta_discovery.gram_bundle_fingerprint


@requires_mosek
def test_a_tampered_gram_breaks_its_entry_hash(sta_discovery, tmp_path):
    entries = [dict(e) for e in sta_discovery.gram_certificates]
    path = save_gram_bundle(tmp_path / "sta.json", entries, system="sta")
    doc = json.loads(path.read_text())
    doc["entries"][0]["gram"][0][0] += 1e-12
    path.write_text(json.dumps(doc))

    loaded, _ = load_gram_bundle(path)
    assert gram_bundle_entry_hash(loaded[0]) != loaded[0]["hash"]


# --------------------------------------------------------------- refusals
def test_bundle_is_refused_when_the_certified_V_is_not_the_design_V():
    class _Res:
        gram_certificates = [{"block": "V"}]

    v_design, v_gate = object(), object()
    entries, fp, reason = gfd._adopt_design_bundle(_Res(), v_gate, v_design)
    assert entries is None and fp is None
    assert reason == "certified_V_is_not_the_design_proposal"


def test_bundle_is_refused_when_the_design_captured_no_grams():
    class _Res:
        gram_certificates = None

    v = object()
    entries, fp, reason = gfd._adopt_design_bundle(_Res(), v, v)
    assert entries is None and fp is None
    assert "no_design_grams" in reason


def test_no_design_result_is_a_typed_reason_not_a_crash():
    v = object()
    entries, fp, reason = gfd._adopt_design_bundle(None, v, v)
    assert (entries, fp, reason) == (None, None, "design_result_unavailable")


def test_design_out_is_optional(monkeypatch):
    """The pre-P2-11 contract still holds: the return value is the V.

    Two scripts and the orchestration tests bind ``_design_full_library``'s
    return value directly, so the JointSOSResult travels through an optional
    out-parameter rather than by changing what the function returns.
    """
    import inspect
    sig = inspect.signature(gfd._design_full_library)
    assert sig.parameters["design_out"].default is None
