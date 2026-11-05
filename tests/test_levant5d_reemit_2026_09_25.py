"""levant5d m = 11: the 2026-09-25 re-emission that corrects the margin declaration.

Review 2026-09-25, blocker B1: the canonical levant5d record of
2026-09-05 (``portable_2026-09-05/``) declares ``eps_certified_exact =
7377626901370455/2^66``, the exact value of the float64 the exact phase
stored, rounded UP relative to the rational P derives, and it carries no
``eps_certified_exact_is_float_transcription`` key. It was emitted about three
hours before WP R-08 added that key to the emitter, and four days before
``check_declared_margin`` entered the portable verifier (``4f9a39c``). The
current P therefore rejects it on its declaration alone, after the proof.

``artifacts/campaign_2026-09/L5D_reemit_2026-09-25/portable/`` is the same
certificate re-emitted by today's canonical emitter from the same exact
inputs. This file pins three things, none of which needs the ten-hour proof:

  * the re-emission differs from the 2026-09-05 record only on an explicit
    allow-list (the flag, the source label, provenance, a timestamp, and the
    registry path/hash that the ``gain_point_digest`` binds instead);
  * the Gram sidecar differs only in ``timestamp_utc``: everything from
    ``"V_grams":[`` on is byte-identical (pinned by the sha256 of that body);
  * ``check_declared_margin`` with the margin P derived on 2026-09-05 accepts
    the new record and rejects the old one.

The P verdict on the new record is NOT pinned here; it is the quiet-window run
named in the directory's README.
"""
import gzip
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = (ROOT / "artifacts" / "campaign_2026-08" / "E5_levant5d"
           / "exact_2026-09-04" / "portable_2026-09-05")
NEW_DIR = (ROOT / "artifacts" / "campaign_2026-09" / "L5D_reemit_2026-09-25"
           / "portable")
STEM = "levant5d_m11_nominal"
OLD_RECORD = OLD_DIR / f"{STEM}.record.json.gz"
NEW_RECORD = NEW_DIR / f"{STEM}.record.json.gz"
OLD_GRAMS = OLD_DIR / f"{STEM}.grams.json.gz"
NEW_GRAMS = NEW_DIR / f"{STEM}.grams.json.gz"

# P's own derivation, field eps_certified_exact of
# portable_2026-09-05/logs/verify_systempython.json (VERIFIED, 2026-09-05).
DERIVED = Fraction(181312558728080294056082717963797409,
                   1813388729421943762059264000000000000000)
# sha256 of the decompressed sidecar from '"V_grams":[' to the end, identical
# for the 2026-09-05 and the 2026-09-25 sidecar.
GRAMS_BODY_SHA256 = \
    "6f83b8b69014923cf1297856d2d9e23922691930f46a9cd23ab5a36cceb52078"

# Record paths allowed to differ between the two emissions; nothing else may.
ALLOWED = {
    "$.eps_certified_exact_is_float_transcription",
    "$.eps_certified_source",
    "$.provenance.source_reported_eps_certified",
    "$.timestamp_utc",
    "$.source_artifacts[13].path",
    "$.source_artifacts[13].sha256",
}

needs_records = pytest.mark.skipif(
    not (OLD_RECORD.exists() and NEW_RECORD.exists()),
    reason="the two levant5d m=11 records are not both present")
needs_new_grams = pytest.mark.skipif(
    not NEW_GRAMS.exists(), reason="the 2026-09-25 sidecar is not present")


def read_gz_json(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def differing_paths(a, b, path="$"):
    if type(a) is not type(b):
        yield path
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield f"{path}.{key}"
            else:
                yield from differing_paths(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}[len]"
        for index, (x, y) in enumerate(zip(a, b)):
            yield from differing_paths(x, y, f"{path}[{index}]")
    elif a != b:
        yield path


def portable_verifier():
    sys.path.insert(0, str(ROOT / "verification" / "standalone"))
    import verify_certificate  # noqa: PLC0415
    return verify_certificate


@pytest.fixture(scope="module")
def records():
    return read_gz_json(OLD_RECORD), read_gz_json(NEW_RECORD)


@needs_records
def test_only_the_margin_declaration_and_provenance_differ(records):
    old, new = records
    diff = set(differing_paths(old, new))
    assert diff <= ALLOWED, sorted(diff - ALLOWED)
    assert "$.eps_certified_exact_is_float_transcription" in diff
    assert old["V"] == new["V"] and len(new["V"]["terms"]) == 507
    # The registry file moved on since 2026-09-05; the levant5d point did not.
    assert old["source_artifacts"][13]["role"] == "gain_point_registry"
    assert old["source_artifacts"][13]["gain_point_digest"] == \
        new["source_artifacts"][13]["gain_point_digest"]


@needs_records
def test_new_record_declares_a_labelled_float_transcription(records):
    _, new = records
    assert new["eps_certified_exact_is_float_transcription"] is True
    assert new["eps_certified_source"] == \
        "exact_phase_unverified_by_portable_stage"
    assert Fraction(new["eps_certified_exact"]) == \
        Fraction(new["eps_certified"])


@needs_records
def test_current_p_margin_check_accepts_new_and_rejects_old(records):
    """B1, reproduced and fixed, with the margin P derived from these Grams."""
    vc = portable_verifier()
    old, new = records
    stated = Fraction(old["eps_certified_exact"])
    assert stated > DERIVED and float(stated) == float(DERIVED)
    assert "eps_certified_exact_is_float_transcription" not in old
    with pytest.raises(vc.VerifyError, match="MARGIN CLAIM EXCEEDS PROOF"):
        vc.check_declared_margin(old, DERIVED)
    vc.check_declared_margin(new, None)
    vc.check_declared_margin(new, DERIVED)


@needs_new_grams
def test_new_sidecar_differs_only_in_its_timestamp():
    marker = b'"V_grams":['
    with gzip.open(NEW_GRAMS, "rb") as fh:
        head = fh.read(1 << 16)
        cut = head.find(marker)
        assert cut > 0
        header = json.loads(head[:cut].rstrip(b",").decode() + "}")
        digest = hashlib.sha256(head[cut:])
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    assert set(header) == {"schema_version", "timestamp_utc", "system",
                           "nominal"}
    assert header["system"] == "levant5d" and header["nominal"] is True
    assert digest.hexdigest() == GRAMS_BODY_SHA256
