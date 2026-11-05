"""The V proof persisted across corner PROCESSES (P3-05) — MOSEK-free.

The in-process :class:`VProofCache` (R2) never crosses an exact closure that
runs one corner per process. :class:`DiskVProofCache` persists it. These tests
pin the two things that make that safe:

  * a hit is re-decided, not trusted. The stored object is the V Gram witness;
    on load the exact rational gate runs again on it, and a Gram that no
    longer proves the target is refused with a typed reason;
  * every way of not being the right proof is a typed refusal that falls back
    to the full proof: no entry, a foreign key, a changed verifier, a damaged
    payload, an unknown schema, a Gram that does not reproduce its
    fingerprint, a proof whose output fingerprint does not reproduce.

No solver is needed: the V Grams come from a stored campaign record and the
one end-to-end test stubs ``verify_gf_sos_full``.
"""
import hashlib
import json
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification import exact_phase as ep
from gf_framework.verification.exact_gate import (
    ExactCaptureResult,
    exact_terms_from_gf,
    exact_verify_captures,
)
from gf_framework.verification.exact_phase import (
    VPROOF_CACHE_REASONS,
    VPROOF_CACHE_SCHEMA,
    DiskVProofCache,
    ExactPhaseResult,
    as_disk_vproof_cache,
    resolve_vproof_cache_root,
    run_exact_phase,
    vproof_cache_for_margin_policy,
    v_input_hash,
    v_proof_fingerprint,
    vproof_cache_key,
)

REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "artifacts/ablation/levant_2003_perturbed"
BUNDLE = BLOCK / "levant3d_2003print_exactgains_grams_20260712T062256Z.json"
RECORD = BLOCK / "levant3d_2003print_exactgains_20260712T062256Z.json"

EPS = 1e-6
N_VARS = 3


# ---------------------------------------------------------------------------
# Fixtures: a REAL V and its REAL Gram captures, no solver
# ---------------------------------------------------------------------------

def _fixture():
    if not BUNDLE.exists() or not RECORD.exists():
        pytest.skip("stored levant3d_2003print Gram bundle not present")
    record = json.loads(RECORD.read_text())
    bundle = json.loads(BUNDLE.read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]],
    )
    grams = [
        {
            "signs": tuple(int(s) for s in c["signs"]),
            "octant_index": int(c["octant_index"]),
            "eps": float(c["eps"]),
            "mu": tuple(int(v) for v in c["mu"]),
            "y_degree": int(c["y_degree"]),
            "basis": [tuple(int(e) for e in b) for b in c["basis"]],
            "gram": np.asarray(c["gram"], dtype=np.float64),
        }
        for c in bundle["V_grams"]
    ]
    return V, grams


@pytest.fixture(scope="module")
def proved():
    """(V, grams, ExactPhaseResult) with a V half that really was proved."""
    V, grams = _fixture()
    v_terms = exact_terms_from_gf(V)
    exact = exact_verify_captures(v_terms, grams, N_VARS)
    assert exact.ok, f"fixture is not a proof: {exact.reason}"
    result = ExactPhaseResult(
        v_resolve_feasible=True, v_exact=exact, v_grams=grams,
        v_input_hash=v_input_hash(V, EPS),
        v_fingerprint=v_proof_fingerprint(exact, grams),
        witness_source=ep.WITNESS_RESOLVE, witness_solver="CLARABEL",
    )
    return V, grams, result


def _store(cache, proved):
    V, _grams, result = proved
    return cache.store(V, EPS, N_VARS, result)


def _load(cache, proved):
    V, _grams, _result = proved
    return cache.load(V, EPS, N_VARS, v_terms=exact_terms_from_gf(V))


def _rewrite(path: Path, **changes) -> None:
    """Edit an entry and repair its content digest, so the OTHER guard fires."""
    doc = json.loads(path.read_text())
    doc.update(changes)
    doc.pop("payload_sha256", None)
    doc["payload_sha256"] = hashlib.sha256(
        ep._canonical_json(doc).encode()).hexdigest()
    path.write_text(ep._canonical_json(doc))


# ---------------------------------------------------------------------------
# 1. Miss, store, hit
# ---------------------------------------------------------------------------

def test_empty_cache_is_a_typed_miss(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="levant3d_2003print")
    hit, event = _load(cache, proved)
    assert hit is None
    assert event.enabled and not event.hit
    assert event.reason == "vproof_cache_miss"
    assert event.path.endswith(".json") and event.key


def test_store_then_hit_revalidates_and_reproduces_the_proof(tmp_path, proved):
    V, grams, result = proved
    cache = DiskVProofCache(tmp_path, library="levant3d_2003print")
    store = _store(cache, proved)
    assert store.stored and store.store_reason == "vproof_cache_stored"
    assert store.entry_bytes > 0

    hit, event = _load(cache, proved)
    assert hit is not None
    assert event.hit and event.reason == "vproof_cache_hit"
    # The default lookup RE-DECIDES: the exact gate ran on the stored Grams.
    assert event.revalidated is True
    assert hit.input_hash == v_input_hash(V, EPS)
    assert hit.fingerprint == result.v_fingerprint
    assert hit.exact.ok and hit.resolve_feasible
    assert len(hit.grams) == len(grams)
    # and the reloaded Grams are bit-identical to the ones written
    for a, b in zip(hit.grams, grams):
        assert np.array_equal(np.asarray(a["gram"]), np.asarray(b["gram"]))
        assert a["signs"] == b["signs"] and a["basis"] == b["basis"]


def test_the_entry_file_is_named_by_the_key_and_nothing_else_is_left(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="levant3d_2003print")
    key = _store(cache, proved).key
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f"v_{key}.json"], "a temporary file survived the write"


def test_revalidate_false_adopts_the_stored_verdict(tmp_path, proved):
    """The in-process semantics, opt-in and recorded (never the default)."""
    _store(DiskVProofCache(tmp_path, library="L"), proved)
    cache = DiskVProofCache(tmp_path, library="L", revalidate=False)
    hit, event = _load(cache, proved)
    assert hit is not None and event.hit
    assert event.revalidated is False
    assert hit.exact.ok
    assert len(hit.exact.octants) == 2 ** N_VARS
    assert hit.exact.mu == proved[2].v_exact.mu


def test_read_only_cache_never_writes(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L", read_only=True)
    event = _store(cache, proved)
    assert not event.stored and event.store_reason == "vproof_cache_read_only"
    assert list(tmp_path.iterdir()) == []


def test_a_failed_v_half_is_never_stored(tmp_path, proved):
    V, grams, _result = proved
    bad = ExactPhaseResult(
        v_resolve_feasible=True,
        v_exact=ExactCaptureResult(ok=False, reason="octant0:not_psd"),
        v_grams=grams)
    event = DiskVProofCache(tmp_path).store(V, EPS, N_VARS, bad)
    assert not event.stored and event.store_reason == "vproof_cache_not_proved"
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 2. Key refusals: a foreign proof can never be adopted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"library": "other_design_point"},
    {"extra": {"parity": "reduced"}},
])
def test_a_different_key_component_is_a_different_entry(tmp_path, proved, kwargs):
    _store(DiskVProofCache(tmp_path, library="L"), proved)
    hit, event = _load(DiskVProofCache(tmp_path, **kwargs), proved)
    assert hit is None and event.reason == "vproof_cache_miss"


def test_a_different_epsilon_is_a_different_entry(tmp_path, proved):
    V, _grams, _result = proved
    _store(DiskVProofCache(tmp_path, library="L"), proved)
    cache = DiskVProofCache(tmp_path, library="L")
    hit, event = cache.load(V, 1e-9, N_VARS, v_terms=exact_terms_from_gf(V))
    assert hit is None and event.reason == "vproof_cache_miss"


def test_the_key_covers_the_v_coefficients_as_exact_rationals(proved):
    V, _grams, _result = proved
    bumped = GeneralizedForm(
        degree_m=V.degree_m, weights_r=V.weights_r,
        terms=[GFTerm(coefficient=t.coefficient + (1e-12 if i == 0 else 0.0),
                      exponents=t.exponents, term_types=t.term_types)
               for i, t in enumerate(V.terms)])
    assert vproof_cache_key(V, EPS, N_VARS) != vproof_cache_key(bumped, EPS, N_VARS)


def test_the_key_covers_the_verifier_code_version(monkeypatch, proved):
    V, _grams, _result = proved
    before = vproof_cache_key(V, EPS, N_VARS)
    monkeypatch.setattr(ep, "verifier_code_version", lambda: "a-different-verifier")
    assert vproof_cache_key(V, EPS, N_VARS) != before


def test_an_entry_that_lands_at_a_foreign_key_is_refused(tmp_path, proved):
    """Path collision or a hand-copied file: the entry repeats every key
    component, so it is refused on READ and not only by its file name."""
    cache = DiskVProofCache(tmp_path, library="L")
    key = _store(cache, proved).key
    other = DiskVProofCache(tmp_path, library="OTHER")
    other_key = other.key_for(proved[0], EPS, N_VARS)
    cache.path_for(key).rename(other.path_for(other_key))
    hit, event = _load(other, proved)
    assert hit is None and event.reason == "vproof_cache_key_mismatch"
    assert "library" in event.detail or "key" in event.detail


def test_a_tampered_v_input_hash_is_refused(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    _rewrite(path, v_input_hash="0" * 64)
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_key_mismatch"


def test_a_stale_verifier_version_is_refused(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    _rewrite(path, code_version="written-by-an-older-exact-gate")
    hit, event = _load(cache, proved)
    assert hit is None
    assert event.reason == "vproof_cache_code_version_mismatch"


def test_an_unknown_schema_is_refused(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    _rewrite(path, schema="gf.vproof_cache/99")
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_schema_unknown"
    assert VPROOF_CACHE_SCHEMA in event.detail


# ---------------------------------------------------------------------------
# 3. Corrupt entries
# ---------------------------------------------------------------------------

def test_a_truncated_entry_is_refused(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_unreadable"


def test_a_non_json_entry_is_refused(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    path.write_bytes(b"\x00\x01\x02 not json at all")
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_unreadable"


def test_an_edited_entry_without_a_repaired_digest_is_refused(tmp_path, proved):
    """Any silent edit trips the entry's own content digest first."""
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    doc = json.loads(path.read_text())
    doc["fingerprint"] = "0" * 64
    path.write_text(json.dumps(doc))
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_payload_corrupt"


def test_a_damaged_gram_no_longer_reproduces_the_bundle_fingerprint(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    doc = json.loads(path.read_text())
    doc["grams"]["entries"][0]["gram"][0][0] += 1.0
    _rewrite(path, grams=doc["grams"])
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_gram_mismatch"


def test_a_damaged_gram_with_a_repaired_fingerprint_is_still_rejected(tmp_path, proved):
    """The soundness case: every hash repaired, and the EXACT GATE refuses.

    This is why a hit is revalidated. A Gram that does not prove the target
    cannot be smuggled in by fixing the bookkeeping around it.
    """
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    doc = json.loads(path.read_text())
    doc["grams"]["entries"][0]["gram"][0][0] -= 1.0
    entries, _meta = ep.gram_bundle_from_jsonable(doc["grams"])
    _rewrite(path, grams=doc["grams"],
             gram_fingerprint=ep.gram_bundle_fingerprint(entries))
    hit, event = _load(cache, proved)
    assert hit is None
    assert event.reason == "vproof_cache_revalidation_failed"
    assert event.revalidated is True and event.revalidate_s > 0.0


def test_a_stored_fingerprint_that_the_revalidation_does_not_reproduce(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    _rewrite(path, fingerprint="f" * 64)
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_fingerprint_mismatch"


def test_an_entry_recording_a_failure_is_refused_on_read(tmp_path, proved):
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    doc = json.loads(path.read_text())
    doc["exact"]["ok"] = False
    _rewrite(path, exact=doc["exact"], resolve_feasible=False)
    hit, event = _load(cache, proved)
    assert hit is None and event.reason == "vproof_cache_not_proved"


def test_every_reason_a_lookup_can_return_is_typed(tmp_path, proved):
    """No refusal escapes the vocabulary a record is allowed to carry."""
    cache = DiskVProofCache(tmp_path, library="L")
    path = cache.path_for(_store(cache, proved).key)
    seen = {_load(cache, proved)[1].reason}
    for mutate in (
        lambda: path.write_bytes(b"junk"),
        lambda: _rewrite(path, schema="x"),
    ):
        _store(cache, proved)
        mutate()
        seen.add(_load(cache, proved)[1].reason)
    assert seen <= VPROOF_CACHE_REASONS


# ---------------------------------------------------------------------------
# 4. Concurrency: two corner processes, one cache
# ---------------------------------------------------------------------------

_WRITER = textwrap.dedent(
    """
    import json, sys, time
    from fractions import Fraction
    from pathlib import Path
    sys.path.insert(0, {src!r})
    import numpy as np
    from gf_framework.algebra.core import GeneralizedForm, GFTerm
    from gf_framework.verification.exact_gate import (
        exact_terms_from_gf, exact_verify_captures)
    from gf_framework.verification.exact_phase import (
        DiskVProofCache, ExactPhaseResult, v_input_hash, v_proof_fingerprint)

    record = json.loads(Path({record!r}).read_text())
    bundle = json.loads(Path({bundle!r}).read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]])
    grams = [{{"signs": tuple(int(s) for s in c["signs"]),
              "octant_index": int(c["octant_index"]), "eps": float(c["eps"]),
              "mu": tuple(int(v) for v in c["mu"]),
              "y_degree": int(c["y_degree"]),
              "basis": [tuple(int(e) for e in b) for b in c["basis"]],
              "gram": np.asarray(c["gram"], dtype=np.float64)}}
             for c in bundle["V_grams"]]
    terms = exact_terms_from_gf(V)
    exact = exact_verify_captures(terms, grams, 3)
    res = ExactPhaseResult(v_resolve_feasible=True, v_exact=exact, v_grams=grams,
                           v_input_hash=v_input_hash(V, 1e-6),
                           v_fingerprint=v_proof_fingerprint(exact, grams))
    cache = DiskVProofCache({root!r}, library="L")
    for _ in range({n!r}):
        # skip_if_valid=False (added by WP P3-03): store() otherwise refuses to
        # overwrite an entry that is already valid, which would make this test
        # write once and then stop generating the concurrent write traffic it
        # exists to race against. The atomicity under test is unchanged.
        ev = cache.store(V, 1e-6, 3, res, skip_if_valid=False)
        assert ev.stored, (ev.store_reason, ev.detail)
    print("OK")
    """
)


def test_two_processes_writing_the_same_key_never_produce_a_partial_read(
        tmp_path, proved):
    """Atomic rename: a concurrent reader sees a complete entry or none."""
    src = str(REPO / "src")
    code = _WRITER.format(src=src, record=str(RECORD), bundle=str(BUNDLE),
                          root=str(tmp_path), n=6)
    procs = [subprocess.Popen([sys.executable, "-c", code],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
             for _ in range(2)]
    cache = DiskVProofCache(tmp_path, library="L")
    reads = {"missing": 0, "hit": 0}
    try:
        while any(p.poll() is None for p in procs):
            hit, event = _load(cache, proved)
            if hit is None:
                # A transient Windows sharing conflict is retried by the
                # cache; corruption and persistent denial must still fail.
                assert event.reason == "vproof_cache_miss", (event.reason, event.detail)
                reads["missing"] += 1
            else:
                reads["hit"] += 1
        outs = [p.communicate() for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.communicate()
    for (out, err), p in zip(outs, procs):
        assert p.returncode == 0, err
        assert "OK" in out
    assert reads["hit"] + reads["missing"] > 0
    # nothing half-written survives, and the final entry still proves
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftovers == [], leftovers
    hit, event = _load(cache, proved)
    assert hit is not None and event.hit and event.revalidated


# ---------------------------------------------------------------------------
# 5. run_exact_phase integration and the recorded provenance
# ---------------------------------------------------------------------------

class _StubSolve:
    """Stands in for ``verify_gf_sos_full``: returns the fixture's V Grams.

    ``solver_used`` is what P2-10 added to ``JointSOSResult`` and what P3-05
    carries across the phase boundary: the ENCODING that ran, as opposed to
    the solver that was requested.
    """

    def __init__(self, grams, solver_used="MOSEK_DIRECT"):
        self.grams = grams
        self.solver_used = solver_used
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1

        class _Res:
            feasible = True
            gram_certificates = self.grams
            solver_used = self.solver_used
        return _Res()


def test_a_second_process_skips_the_v_resolve_and_records_the_hit(
        tmp_path, proved, monkeypatch):
    V, grams, _result = proved
    stub = _StubSolve(grams)
    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    cache = DiskVProofCache(tmp_path, library="L")

    first = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                            fallback_solver=None, v_proof_cache=cache)
    assert stub.calls == 1
    assert first.v_proof_cache["hit"] is False
    assert first.v_proof_cache["reason"] == "vproof_cache_miss"
    assert first.v_proof_cache["store_reason"] == "vproof_cache_stored"
    assert first.v_reused is False

    second = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                             fallback_solver=None, v_proof_cache=cache)
    assert stub.calls == 1, "the V re-solve ran again on a cache hit"
    assert second.v_reused is True
    assert second.v_proof_cache["hit"] is True
    assert second.v_proof_cache["reason"] == "vproof_cache_hit"
    assert second.v_proof_cache["revalidated"] is True
    assert second.v_fingerprint == first.v_fingerprint
    assert second.v_exact.ok


def test_a_refused_entry_falls_back_to_the_full_proof(tmp_path, proved, monkeypatch):
    V, grams, _result = proved
    stub = _StubSolve(grams)
    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    cache = DiskVProofCache(tmp_path, library="L")
    run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                    fallback_solver=None, v_proof_cache=cache)
    # The phase keys the entry under the margin policy it ran with (P2-14: the
    # default is "achieved"), so damage the entry the phase will actually read.
    keyed = vproof_cache_for_margin_policy(cache, ep.MARGIN_POLICY_ACHIEVED)
    keyed.path_for(keyed.key_for(V, EPS, N_VARS)).write_bytes(b"garbage")

    again = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                            fallback_solver=None, v_proof_cache=cache)
    assert stub.calls == 2, "a refusal must fall back to the full proof"
    assert again.v_proof_cache["reason"] == "vproof_cache_unreadable"
    assert again.v_proof_cache["store_reason"] == "vproof_cache_stored"
    assert again.v_reused is False
    assert again.v_exact.ok


def test_the_cache_state_is_in_the_witness_record(tmp_path, proved, monkeypatch):
    V, grams, _result = proved
    monkeypatch.setattr(ep, "verify_gf_sos_full", _StubSolve(grams))
    cache = DiskVProofCache(tmp_path, library="L")
    phase = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                            fallback_solver=None, v_proof_cache=cache)
    block = phase.witness_record()["v_proof_cache"]
    assert block["enabled"] is True
    assert block["path"].startswith(str(tmp_path))
    keyed = vproof_cache_for_margin_policy(cache, ep.MARGIN_POLICY_ACHIEVED)
    assert block["key"] == keyed.key_for(V, EPS, N_VARS)
    assert block["code_version"] == ep.verifier_code_version()
    assert json.dumps(phase.witness_record())  # JSON-ready


def test_no_cache_is_recorded_as_disabled_and_changes_nothing(
        tmp_path, proved, monkeypatch):
    V, grams, _result = proved
    monkeypatch.setattr(ep, "verify_gf_sos_full", _StubSolve(grams))
    phase = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                            fallback_solver=None)
    assert phase.v_proof_cache["enabled"] is False
    assert phase.v_proof_cache["reason"] == "vproof_cache_disabled"
    assert phase.v_proof_cache["path"] == ""
    assert list(tmp_path.iterdir()) == []


def test_an_in_process_cache_takes_priority_and_is_recorded(
        tmp_path, proved, monkeypatch):
    V, grams, _result = proved
    stub = _StubSolve(grams)
    monkeypatch.setattr(ep, "verify_gf_sos_full", stub)
    in_process = ep.build_v_cache(V, EPS, _result)
    cache = DiskVProofCache(tmp_path, library="L")
    phase = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                            fallback_solver=None, v_cache=in_process,
                            v_proof_cache=cache)
    assert stub.calls == 0
    assert phase.v_fingerprint == in_process.fingerprint
    assert phase.v_proof_cache["reason"] == "vproof_cache_in_process_hit"
    assert phase.v_proof_cache["stored"] is False
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 5b. The ENCODING that ran survives the phase boundary (P2-10 follow-up)
# ---------------------------------------------------------------------------

def test_the_actual_encoding_is_recorded_beside_the_requested_solver(
        tmp_path, proved, monkeypatch):
    """``solver`` is the REQUEST, ``solver_used`` is what actually ran.

    P2-10 made a request for MOSEK reachable by two encodings, the CVXPY
    interface ("MOSEK") and the Task API ("MOSEK_DIRECT"). Only
    ``JointSOSResult.solver_used`` separates them, and before this it was
    dropped at the phase boundary, so an E1 record could not tell them apart.
    """
    V, grams, _result = proved
    monkeypatch.setattr(ep, "verify_gf_sos_full",
                        _StubSolve(grams, solver_used="MOSEK_DIRECT"))
    phase = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="MOSEK",
                            fallback_solver=None)
    assert phase.witness_solver == "MOSEK", "the REQUEST must stay as it was"
    assert phase.witness_solver_used == "MOSEK_DIRECT"
    attempt = phase.witness_attempts[0]
    assert attempt["solver"] == "MOSEK"
    assert attempt["solver_used"] == "MOSEK_DIRECT"
    record = phase.witness_record()
    assert record["witness_solver"] == "MOSEK"
    assert record["witness_solver_used"] == "MOSEK_DIRECT"


def test_disagreeing_encodings_within_one_attempt_are_both_named(proved, monkeypatch):
    """V through the Task API and W through CVXPY must not be reported as one."""
    V, grams, _result = proved
    names = iter(["MOSEK_DIRECT", "MOSEK"])

    def _solve(*args, **kwargs):
        used = next(names)

        class _Res:
            feasible = True
            gram_certificates = grams
            solver_used = used
        return _Res()

    monkeypatch.setattr(ep, "verify_gf_sos_full", _solve)
    dyn = [GeneralizedForm(degree_m=1, weights_r=V.weights_r, terms=[])
           for _ in range(N_VARS)]
    phase = run_exact_phase(V, [dyn], N_VARS, epsilon=EPS, solver="MOSEK",
                            fallback_solver=None)
    assert phase.witness_solver_used == "MOSEK_DIRECT+MOSEK"
    assert phase.endpoints[0].witness_solver_used == "MOSEK"


def test_a_solver_that_reports_no_encoding_leaves_the_field_empty(proved, monkeypatch):
    """No guess: a producer without ``solver_used`` records nothing."""
    V, grams, _result = proved
    monkeypatch.setattr(ep, "verify_gf_sos_full", _StubSolve(grams, solver_used=""))
    phase = run_exact_phase(V, [], N_VARS, epsilon=EPS, solver="CLARABEL",
                            fallback_solver=None)
    assert phase.witness_solver == "CLARABEL"
    assert phase.witness_solver_used == ""


# ---------------------------------------------------------------------------
# 6. Configuration
# ---------------------------------------------------------------------------

def test_as_disk_vproof_cache_normalizes_every_accepted_form(tmp_path, monkeypatch):
    assert as_disk_vproof_cache(None) is None
    assert as_disk_vproof_cache(False) is None
    assert as_disk_vproof_cache(str(tmp_path)).root == tmp_path
    assert as_disk_vproof_cache(tmp_path).root == tmp_path
    made = DiskVProofCache(tmp_path)
    assert as_disk_vproof_cache(made) is made
    monkeypatch.setenv(ep.VPROOF_CACHE_ENV, str(tmp_path / "env"))
    assert as_disk_vproof_cache(True).root == tmp_path / "env"
    with pytest.raises(TypeError):
        as_disk_vproof_cache(17)


def test_the_default_root_is_under_the_run_output_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(ep.VPROOF_CACHE_ENV, raising=False)
    assert resolve_vproof_cache_root(tmp_path) == tmp_path / ep.VPROOF_CACHE_DIRNAME
    monkeypatch.setenv(ep.VPROOF_CACHE_ENV, str(tmp_path / "shared"))
    assert resolve_vproof_cache_root(tmp_path) == tmp_path / "shared"
