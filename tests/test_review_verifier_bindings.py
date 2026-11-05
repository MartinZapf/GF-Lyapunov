"""Small adversarial regressions for the September 9 input-boundary review.

All witnesses are scalar exact proofs. No solver or campaign artifact needed.
"""
import copy
import errno
import json
import sys
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification import exact_phase as ep
from gf_framework.verification.exact_gate import exact_terms_from_gf, exact_verify_captures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "standalone"))
sys.path.insert(0, str(ROOT / "scripts"))
import verify_certificate as portable
import emit_certificate_record as emitter


def tiny_proof(charge_ratio=F(1, 4), policy="achieved"):
    V = GeneralizedForm(2, (1,), [GFTerm(1.0, (F(2),), ("abs",))])
    caps = [dict(signs=(s,), octant_index=i, eps=.25, mu=(1,),
                 y_degree=2, basis=[(1,)],
                 gram=np.array([[float(F(3, 4) - charge_ratio / 4)]]))
            for i, s in enumerate((-1, 1))]
    exact = exact_verify_captures(exact_terms_from_gf(V), caps, 1, margin_policy=policy)
    result = ep.ExactPhaseResult(v_resolve_feasible=True, v_exact=exact, v_grams=caps,
        v_input_hash=ep.v_input_hash(V, .25), v_fingerprint=ep.v_proof_fingerprint(exact, caps))
    return V, result


@pytest.mark.parametrize("change", ["V", "epsilon", "missing_source_hash"])
def test_cache_factory_cannot_relabel_a_proof(change):
    V, result = tiny_proof()
    epsilon = .25
    if change == "V":
        V = GeneralizedForm(2, (1,), [GFTerm(-1.0, (F(2),), ("abs",))])
        assert not exact_verify_captures(exact_terms_from_gf(V), result.v_grams, 1).ok
    elif change == "epsilon":
        epsilon = .125
    else:
        result.v_input_hash = ""
    with pytest.raises(ep.ExactPhaseError, match="source input hash"):
        ep.build_v_cache(V, epsilon, result)


@pytest.mark.parametrize("change", ["proof_octant", "gram_octant", "failed_octant", "exact_margin"])
def test_matching_hash_does_not_excuse_incomplete_cached_proof(change):
    V, result = tiny_proof()
    cache = ep.build_v_cache(V, .25, result)
    if change == "proof_octant":
        cache.exact.octants.pop()
    elif change == "gram_octant":
        cache.grams.pop()
    elif change == "failed_octant":
        cache.exact.octants[0].ok = False
    else:
        cache.exact.octants[0].eps_certified = ""
    with pytest.raises(ep.ExactPhaseError, match="cached V proof"):
        ep.run_exact_phase(V, [], 1, epsilon=.25, v_cache=cache)


def test_achieved_cache_cannot_pass_half_with_charge_between_thresholds(tmp_path):
    V, result = tiny_proof(F(3, 4))
    cache = ep.build_v_cache(V, .25, result)
    with pytest.raises(ep.ExactPhaseError, match="half policy"):
        ep.run_exact_phase(V, [], 1, epsilon=.25, v_cache=cache, margin_policy="half")
    disk = ep.vproof_cache_for_margin_policy(ep.DiskVProofCache(tmp_path), "half")
    event = disk.store(V, .25, 1, result)
    assert not event.stored and event.store_reason == "vproof_cache_not_proved"
    assert "half policy" in event.detail


def test_half_cache_reuse_under_achieved_reports_the_requested_denominator():
    V, result = tiny_proof(policy="half")
    cache = ep.build_v_cache(V, .25, result)
    reused = ep.run_exact_phase(V, [], 1, epsilon=.25, v_cache=cache, margin_policy="achieved")
    assert reused.v_reused and reused.v_exact.ok
    assert reused.v_exact.margin_policy == "achieved"
    assert reused.v_exact.worst_charge_ratio == .25
    assert cache.exact.margin_policy == "half"
    assert cache.exact.worst_charge_ratio == .5


def test_disk_half_revalidation_uses_half_and_serialization_keeps_exact_margins(tmp_path, monkeypatch):
    V, result = tiny_proof()
    disk = ep.DiskVProofCache(tmp_path)
    assert disk.store(V, .25, 1, result).stored
    policies = []
    original = ep.exact_verify_captures
    def trace(*args, **kwargs):
        policies.append(kwargs.get("margin_policy"))
        return original(*args, **kwargs)
    monkeypatch.setattr(ep, "exact_verify_captures", trace)
    cached, event = disk.load(V, .25, 1, v_terms=exact_terms_from_gf(V))
    assert event.hit and policies == ["half"]
    assert cached.exact.margin_policy == "half"
    trusted, event = ep.DiskVProofCache(tmp_path, revalidate=False).load(V, .25, 1)
    assert event.hit and not event.revalidated
    assert trusted.exact.octants[0].eps_certified == cached.exact.octants[0].eps_certified
    assert trusted.exact.worst_charge_over_eps == .25


def test_switching_disk_key_back_to_half_removes_achieved_policy(tmp_path):
    original = ep.DiskVProofCache(tmp_path, extra={"probe": "test"})
    achieved = ep.vproof_cache_for_margin_policy(original, "achieved")
    half = ep.vproof_cache_for_margin_policy(achieved, "half")
    assert half.extra == {"probe": "test"} and half.margin_policy == "half"
    assert achieved.margin_policy == "achieved"


def test_malformed_trusted_verdict_is_a_typed_cache_refusal(tmp_path, monkeypatch):
    V, result = tiny_proof()
    disk = ep.DiskVProofCache(tmp_path, revalidate=False)
    assert disk.store(V, .25, 1, result).stored
    def invalid_policy(document):
        ep.resolve_margin_policy("unknown policy")
    monkeypatch.setattr(ep, "_vproof_exact_from_jsonable", invalid_policy)
    cached, event = disk.load(V, .25, 1)
    assert cached is None and event.reason == "vproof_cache_unreadable"
    assert "unknown margin_policy" in event.detail


def test_cache_access_retry_is_bounded_and_does_not_accept_a_persistent_denial(monkeypatch):
    monkeypatch.setattr(ep, "_transient_cache_access_error", lambda exc: isinstance(exc, PermissionError))
    sleeps = []
    monkeypatch.setattr(ep.time, "sleep", sleeps.append)
    attempts = []
    def denied():
        attempts.append(1)
        raise PermissionError(errno.EACCES, "test sharing conflict")
    with pytest.raises(PermissionError):
        ep._cache_io(denied)
    assert len(attempts) == 1 + len(ep._CACHE_IO_RETRY_DELAYS)
    assert sleeps == list(ep._CACHE_IO_RETRY_DELAYS)


def test_transient_cache_access_can_recover_but_missing_files_are_not_retried(monkeypatch):
    monkeypatch.setattr(ep, "_transient_cache_access_error", lambda exc: isinstance(exc, PermissionError))
    monkeypatch.setattr(ep.time, "sleep", lambda seconds: None)
    attempts = []
    def transient():
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError(errno.EACCES, "test contention")
        return b"complete payload"
    assert ep._cache_io(transient) == b"complete payload"
    def missing():
        raise FileNotFoundError("missing")
    with pytest.raises(FileNotFoundError):
        ep._cache_io(missing)


def portable_record(tmp_path):
    def caps(q, degree):
        return [dict(signs=[s], octant_index=i, eps=.25, mu=[2],
                     y_degree=degree, basis=[[degree // 2]], gram=[[q]])
                for i, s in enumerate((-1, 1))]
    record = dict(schema_version="gf-certificate-record/2", system="fosmc", nominal=True,
        rational_gains={"k": "1"}, delta0_exact="0", epsilon_exact="1/4", brackets={},
        grams_file="grams.json", corners=[{"assignment": {}}],
        V=dict(degree_m=2, weights_r=[1], terms=[dict(coefficient=1., exponents=["2"], term_types=["abs"])]))
    bundle = dict(V_grams=caps(.75, 4), corners=[dict(assignment={}, corner_label="rational",
                                                  W_grams=caps(1.75, 2))])
    (tmp_path / "grams.json").write_text(json.dumps(bundle), encoding="utf-8")
    return record


def invoke_portable(tmp_path, monkeypatch, record):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(portable.__file__), str(path)])
    return portable.main()


def test_tiny_portable_control_is_verified(tmp_path, monkeypatch):
    assert invoke_portable(tmp_path, monkeypatch, portable_record(tmp_path)) == 0


def test_portable_rejects_overlapping_gain_namespaces(tmp_path, monkeypatch):
    record = portable_record(tmp_path)
    record["rational_gains"]["k"] = "-100"
    record["brackets"] = {"k": dict(lo="1", hi="2", power_k=2, power_target="2")}
    with pytest.raises(portable.VerifyError, match="both rational and bracketed"):
        invoke_portable(tmp_path, monkeypatch, record)


@pytest.mark.parametrize("lo,hi", [("1", "-2"), ("2", "1"), ("1", "1"), ("0", "2")])
def test_portable_requires_ordered_positive_root_brackets(tmp_path, monkeypatch, lo, hi):
    record = portable_record(tmp_path)
    record["brackets"] = {"k": dict(lo=lo, hi=hi, power_k=2, power_target="2")}
    with pytest.raises(portable.VerifyError, match="bracket k INVALID"):
        invoke_portable(tmp_path, monkeypatch, record)


@pytest.mark.parametrize("exact,approx,transcription,error", [
    ("999", .25, False, "margin declarations disagree"),
    ("1/2", .5, False, "MARGIN CLAIM EXCEEDS PROOF"),
    ("1/2", None, False, "MARGIN CLAIM EXCEEDS PROOF"),
])
def test_portable_refuses_false_exact_margin_claims(tmp_path, monkeypatch, exact, approx, transcription, error):
    record = portable_record(tmp_path)
    record.update(eps_certified_exact=exact, eps_certified=approx,
                  eps_certified_exact_is_float_transcription=transcription)
    with pytest.raises(portable.VerifyError, match=error):
        invoke_portable(tmp_path, monkeypatch, record)


def test_exact_bound_has_no_float_rounding_allowance_but_labelled_transcription_does():
    derived = F(1, 3)
    rounded = float(derived)
    portable.check_declared_margin(dict(eps_certified_exact=str(F(rounded)), eps_certified=rounded,
        eps_certified_exact_is_float_transcription=True), derived)
    larger = F(1, 4) + F(1, 10**30)
    declaration = dict(eps_certified_exact=str(larger), eps_certified=.25,
                       eps_certified_exact_is_float_transcription=False)
    with pytest.raises(portable.VerifyError, match="MARGIN CLAIM EXCEEDS PROOF"):
        portable.check_declared_margin(declaration, F(1, 4))
    assert emitter.margin_claim_exceeds(declaration, F(1, 4))


def test_emitter_refuses_contradictory_exact_margin_before_emission_or_attachment():
    with pytest.raises(emitter.Refusal) as caught:
        emitter.margin_claim_exceeds(dict(eps_certified=.25, eps_certified_exact="999"), F(1, 4))
    assert caught.value.reason == "margin_declaration_inconsistent"


def test_emitter_preserves_an_existing_float_transcription_label():
    witness = dict(margin_policy="achieved", eps_certified=.1,
                   eps_certified_exact=str(F(.1)),
                   eps_certified_exact_is_float_transcription=True)
    resolved = emitter.resolve_margin(witness, .25, SimpleNamespace(margin_policy=None), "tiny")
    assert resolved["eps_certified_exact_is_float_transcription"] is True
    assert not emitter.margin_claim_exceeds(resolved, F(1, 10))
