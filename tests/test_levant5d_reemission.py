"""The five-state degree-11 flagship, re-emitted through the canonical path.

The 2026-09-05 scientific review (finding 4) found that the portable record of
the ``levant5d`` ``m = 11`` flagship had been written by an artifact-local
one-off transcriber rather than by ``scripts/emit_certificate_record.py``: it
carried no ``margin_policy``, no ``eps_certified`` and no ``source_artifacts``,
and it carried free-text ``claim`` and ``provenance`` fields the verifier
ignores. WP R-05 re-emits it through the canonical emitter, into
``portable_2026-09-05/``, leaving the 2026-09-04 emission untouched.

What this file checks is that the re-emission is a TRANSCRIPTION and nothing
else. It compares the two records field by field, it re-derives the margin
from the exact phase's own corner records, and it compares the two Gram
sidecars capture by capture, streaming, without ever parsing 1.09 GB of JSON:

  * same V (507 terms, same encoding), same brackets, corners and gains;
  * same epsilon, same coverage counts, same shared-W witness shape;
  * the new record additionally states the margin policy the exact phase ran
    under, the margin it proved, and every source artifact by content hash;
  * every Gram capture of the new sidecar is byte-identical to the old one.

It does NOT re-prove anything. The exact PSD proof of this record is the
ten-hour portable run, which is the orchestrator's to start:

    /usr/bin/python3 verification/standalone/verify_certificate.py \
        artifacts/campaign_2026-08/E5_levant5d/exact_2026-09-04/\
portable_2026-09-05/levant5d_m11_nominal.record.json.gz \
        --grams .../levant5d_m11_nominal.grams.json.gz
"""
import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCK = (ROOT / "artifacts" / "campaign_2026-08" / "E5_levant5d"
         / "exact_2026-09-04")
CELL = BLOCK / "l5d_m11_eps1e-4"
OLD = BLOCK / "portable_2026-09-04"
NEW = BLOCK / "portable_2026-09-05"

OLD_RECORD = OLD / "levant5d_m11_nominal.record.json.gz"
NEW_RECORD = NEW / "levant5d_m11_nominal.record.json.gz"
OLD_GRAMS = OLD / "levant5d_m11_nominal.grams.json.gz"
NEW_GRAMS = NEW / "levant5d_m11_nominal.grams.json.gz"

needs_records = pytest.mark.skipif(
    not (OLD_RECORD.exists() and NEW_RECORD.exists()),
    reason="the levant5d m=11 portable records are not both present")
needs_grams = pytest.mark.skipif(
    not (OLD_GRAMS.exists() and NEW_GRAMS.exists()),
    reason="the levant5d m=11 Gram sidecars are not both present")


def read_gz_json(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def _levant5d_gain_point_digest():
    """The emitter's digest of the levant5d flagship point in today's registry."""
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "scripts"))
    import emit_certificate_record as em  # noqa: PLC0415
    from gf_framework.systems.printed_gains import gain_point  # noqa: PLC0415
    point = gain_point("levant5d", flagship=True)
    return hashlib.sha256(json.dumps(em.gain_point_record(point), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def records():
    return read_gz_json(OLD_RECORD), read_gz_json(NEW_RECORD)


@pytest.fixture(scope="module")
def cell():
    row = json.loads((CELL / "ROW.json").read_text())
    corners = [json.loads((CELL / f"corner_{k}.json").read_text())
               for k in range(8)]
    return row, corners


# ---------------------------------------------------------------------------
# The proved statement: identical, field for field
# ---------------------------------------------------------------------------

@needs_records
def test_the_two_records_state_the_same_certificate(records):
    old, new = records
    for field in ("schema_version", "system", "system_family", "nominal",
                  "channel", "weights_r", "kappa", "delta0_exact", "epsilon",
                  "epsilon_exact", "rational_gains", "brackets", "corners",
                  "gain_point", "git_sha"):
        assert old[field] == new[field], field
    assert old["V"] == new["V"]
    assert len(new["V"]["terms"]) == 507
    assert new["epsilon_exact"] == str(Fraction(1e-4))
    assert {c["corner_label"] for c in new["corners"]} == \
        {c["corner_label"] for c in old["corners"]}
    assert len({tuple(sorted(c["assignment"].items()))
                for c in new["corners"]}) == 8


@needs_records
def test_coverage_counts_are_unchanged(records):
    old, new = records
    for field in ("octants_per_form", "bracket_corners",
                  "inclusion_endpoints"):
        assert old["coverage"][field] == new["coverage"][field], field
    assert new["coverage"]["octants_per_form"] == 32
    assert new["coverage"]["bracket_corners"] == 8
    assert new["coverage"]["inclusion_endpoints"] == 1


@needs_records
def test_the_shared_w_witness_shape_is_kept(records):
    old, new = records
    assert new["shared_W_witness"] is True
    assert "shared_W" in new["encoding"] and new["encoding"]["shared_W"]
    assert new["witness"]["source"] == "design_bundle"
    assert new["witness"]["gram_bundle_fingerprint"] == \
        old["witness"]["gram_bundle_fingerprint"]


# ---------------------------------------------------------------------------
# What the review asked for and the old record did not have
# ---------------------------------------------------------------------------

@needs_records
def test_the_old_record_is_the_one_the_review_described(records):
    """Pins the defect, so this comparison cannot silently become vacuous."""
    old, _ = records
    assert old.get("margin_policy") is None
    assert old.get("eps_certified") is None
    assert old.get("source_artifacts") is None
    assert isinstance(old.get("claim"), str)


@needs_records
def test_margin_policy_is_the_one_the_exact_phase_ran_under(records, cell):
    """P4-03 ran under the note's half-margin rule, and says so.

    The exact phase was driven with ``--margin-policy half``, and its header
    and all eight corner records record that. The record therefore states
    ``half``, which is the STRICTER of the two rules: the achieved charges
    are far below eps/2, so the note's printed inequality holds as well.
    """
    _, new = records
    _, corners = cell
    header = json.loads((CELL / "header.json").read_text())
    assert header["margin_policy"] == "half"
    assert {c["margin_policy"] for c in corners} == {"half"}
    assert new["margin_policy"] == "half"
    assert new["margin_policy_source"] == "exact_phase"
    assert new["note_half_margin_holds"] is True


@needs_records
def test_eps_certified_is_the_exact_phase_minimum_over_the_corners(records,
                                                                  cell):
    """The number, and what kind of number it is.

    WP R-08 (follow-up review 2026-09-05 B): this record is emitted with
    ``--no-self-verify``, because its portable proof takes about ten hours.
    So the margin it states is the exact phase's OWN report, and this test
    asserts the labelled SOURCE of that number rather than pinning it as an
    independently derived exact bound. The authoritative exact margin is the
    one the ten-hour portable run derives; it arrives beside the record as a
    ``levant5d_m11_nominal.verified_margin.json`` sidecar
    (``--attach-verify-summary``), never by rewriting the verified bytes.
    """
    _, new = records
    row, corners = cell
    worst = min(float(c["eps_certified_float"]) for c in corners)
    assert new["eps_certified"] == worst
    assert new["eps_certified"] == float(row["eps_certified_over_corners"])
    # The 2026-09-05 emission predates R-08 and says "exact_phase"; a
    # re-emission today says so more explicitly. Either way it is the source
    # stage's report, and neither is a portable-verified margin.
    assert new["eps_certified_source"] in (
        "exact_phase", "exact_phase_unverified_by_portable_stage")
    assert new["eps_certified_source"] != "portable_verifier"
    assert 0 < worst < new["epsilon"]
    assert worst >= 0.5 * new["epsilon"]


@needs_records
def test_the_exact_margin_string_is_a_float_transcription(records, cell):
    """P4-03 stored floats, so the record's "exact" string is one too.

    The X records carry ``eps_certified_float`` and ``worst_charge_ratio``
    and no rational anywhere, so ``eps_certified_exact`` here is the exact
    value of a rounded float64, not the exact proof margin. This test pins
    that reading: it is what the record may be quoted as, and it is why the
    number must not be presented as an independently derived bound.
    """
    _, new = records
    row, corners = cell
    worst = min(float(c["eps_certified_float"]) for c in corners)
    for corner in corners:
        assert "eps_certified_exact" not in corner
        assert "eps_certified_exact" not in (corner.get("witness") or {})
    assert "eps_certified_exact" not in row
    assert Fraction(new["eps_certified_exact"]) == Fraction(worst)
    assert float(Fraction(new["eps_certified_exact"])) == worst
    # A record re-emitted since R-08 says so in a field of its own; the
    # shipped one predates the field and is covered by the same evidence.
    assert new.get("eps_certified_exact_is_float_transcription") in (
        True, None)


@needs_records
def test_source_artifacts_name_every_input_by_content_hash(records):
    _, new = records
    sources = new["source_artifacts"]
    roles = {entry["role"] for entry in sources}
    assert {"design_record", "design_gram_bundle", "exact_cell_header",
            "exact_cell_row", "gain_point_registry"} <= roles
    assert {f"exact_corner_{k}" for k in range(8)} <= roles
    for entry in sources:
        # Historical Linux paths identify the same checkout-relative input.
        # Git's Windows checkout converts text to CRLF; the recorded source
        # hash is of the original LF bytes, not of that Windows byte stream.
        stated = entry["path"].removeprefix("/home/martin/Repos/latex/GF Framework/")
        path = Path(stated)
        if not path.is_absolute():
            path = ROOT / path
        assert path.exists(), entry["path"]
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        original_lf = None
        if digest.hexdigest() != entry["sha256"] and path.suffix in {".json", ".py"}:
            original_lf = path.read_bytes().decode("utf-8").replace("\r\n", "\n").encode("utf-8")
        if (entry["role"] == "gain_point_registry"
                and digest.hexdigest() != entry["sha256"]
                and hashlib.sha256(original_lf).hexdigest() != entry["sha256"]):
            # The registry module may move on (a note, a new gain point); the
            # emitter records gain_point_digest for exactly this case. The
            # module then binds by the digest of the levant5d gain point.
            assert entry["gain_point_digest"] == _levant5d_gain_point_digest()
        elif original_lf is not None:
            assert hashlib.sha256(original_lf).hexdigest() == entry["sha256"], entry["role"]
        else:
            assert digest.hexdigest() == entry["sha256"], entry["role"]


@needs_records
def test_free_text_lives_under_unverified_provenance(records):
    _, new = records
    assert "claim" not in new
    provenance = new["provenance"]
    assert "unverified provenance" in provenance["note"]
    assert provenance["source_claim"]
    assert provenance["exact_stage"]["gram_bundle_sha256"]
    # Every number the record states outside provenance comes from a stage,
    # and names which one; nothing is hand typed.
    assert new["eps_certified_source"] in (
        "exact_phase", "exact_phase_unverified_by_portable_stage",
        "portable_verifier")


# ---------------------------------------------------------------------------
# The Grams: byte-identical, compared streaming
# ---------------------------------------------------------------------------

def iter_nested_objects(path, chunk_bytes=1 << 22):
    """Yield ``(prefix, sha256, n_bytes)`` for each object below the root.

    The sidecar is 1.09 GB of JSON, so it is never parsed: the scan walks the
    gzip stream for braces only. A capture object contains no nested object
    of its own, and the corner entries nest exactly one (their assignment),
    so tracking brace depth is enough to cut the document into the objects
    the two emissions must agree on. ``prefix`` is the first bytes of the
    object, which is what tells a capture from a corner entry.
    """
    depth, digest, size, prefix = 0, None, 0, b""
    with gzip.open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            pos = 0
            while pos < len(chunk):
                opening = chunk.find(b"{", pos)
                closing = chunk.find(b"}", pos)
                found = [i for i in (opening, closing) if i != -1]
                end = min(found) if found else len(chunk)
                if digest is not None and end > pos:
                    body = chunk[pos:end]
                    digest.update(body)
                    size += len(body)
                    if len(prefix) < 24:
                        prefix += body[:24 - len(prefix)]
                if not found:
                    break
                mark = min(found)
                char = chunk[mark:mark + 1]
                if char == b"{":
                    depth += 1
                    if depth == 2:
                        digest, size, prefix = hashlib.sha256(), 0, b""
                    if digest is not None:
                        digest.update(char)
                        size += 1
                        if len(prefix) < 24:
                            prefix += char
                else:
                    if digest is not None:
                        digest.update(char)
                        size += 1
                    depth -= 1
                    if depth == 1 and digest is not None:
                        yield prefix.decode("utf-8", "replace"), \
                            digest.hexdigest(), size
                        digest = None
                pos = mark + 1
    assert depth == 0, f"{path}: unbalanced braces"


def split_objects(path):
    captures, corners = [], []
    for prefix, digest, size in iter_nested_objects(path):
        (captures if prefix.startswith('{"signs"') else corners).append(
            (digest, size))
    return captures, corners


@needs_grams
def test_every_gram_capture_is_byte_identical_to_the_2026_09_04_sidecar():
    """The re-emission changed the record, not one Gram entry.

    Both sidecars are streamed and hashed capture by capture. 32 V captures
    plus one shared 32-capture W witness set, in the same order, with the
    same bytes: the matrices this record ships are the matrices the exact
    phase proved on, in both emissions.
    """
    old_caps, old_corners = split_objects(OLD_GRAMS)
    new_caps, new_corners = split_objects(NEW_GRAMS)
    assert len(old_caps) == 64, len(old_caps)
    assert len(new_caps) == len(old_caps)
    assert len(new_corners) == len(old_corners) == 8
    for index, (old_cap, new_cap) in enumerate(zip(old_caps, new_caps)):
        assert new_cap == old_cap, f"capture {index}"
    assert new_corners == old_corners


@needs_grams
def test_the_new_sidecar_declares_the_shared_witness_shape():
    """The header of the sidecar, read without the captures behind it."""
    head = gzip.open(NEW_GRAMS, "rb").read(400).decode("utf-8", "replace")
    assert head.startswith('{"schema_version":"gf-certificate-record/2"')
    assert '"system":"levant5d"' in head
    assert '"nominal":true' in head
    assert '"V_grams":[' in head
    tail_marker = '"shared_W_grams":['
    with gzip.open(NEW_GRAMS, "rb") as fh:
        found = False
        window = b""
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            window = (window + chunk)[-(1 << 22):]
            if tail_marker.encode() in window:
                found = True
                break
    assert found, "the sidecar carries no shared_W_grams capture set"
