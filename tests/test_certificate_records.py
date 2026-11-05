"""Portable-record emitter (scripts/emit_certificate_record.py, item C9).

Binding rule 8 of the pre-writing campaign makes a portable record part of
what "certified" means, so the emitter has exactly two acceptable outcomes:
a record the dependency-free verifier accepts, or a TYPED REFUSAL naming what
a re-run has to produce. A partial record would be the failure mode that
matters, because it would look citable while covering less than it claims.

These tests pin both outcomes:

  * the refusal path on the artifact the campaign actually has today
    (``reverify_exact_printed_*.json`` records per-octant verdicts, not Gram
    certificates, so no benchmark row can be recorded from it yet);
  * the emit-then-verify roundtrip end to end, nominal (fosmc, the smallest
    row) and perturbed (the stored flagship sta record);
  * the completeness rules of the E3 card: 2^n octants, 2^k bracket corners,
    two inclusion endpoints for a perturbed row, one dynamics for a nominal
    one, and one margin throughout.

The nominal roundtrip reads V, the margin, the gain binding and the
provenance from the REAL campaign artifacts, and supplies only the Gram
captures, which it derives here from fosmc's own single-monomial octant
targets. That is a test fixture standing in for the re-run listed in the
refusal, not a claim that the campaign row ships portably today.
"""
import gzip
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "standalone"))

from verify_certificate import exact_octant_target  # noqa: E402

EMITTER = ROOT / "scripts" / "emit_certificate_record.py"
VERIFIER = ROOT / "verification" / "standalone" / "verify_certificate.py"

CAMPAIGN = ROOT / "artifacts" / "campaign_2026-08" / "E1_matrix"
FOSMC_ROW = CAMPAIGN / "systems" / "fosmc.json"
EXACT_PRINTED = CAMPAIGN / "reverify_exact_printed_20260802T181035Z.json"
CAMPAIGN_ENV = CAMPAIGN / "env.json"

FLAGSHIP = ROOT / "artifacts" / "ablation" / "levant_2003_perturbed"
STA_ROW = FLAGSHIP / "sta_delta1_sparse_row.json"
STA_EXACT = FLAGSHIP / "sta_exactgains_20260712T062215Z.json"

EPSILON = 1e-6

needs_campaign = pytest.mark.skipif(
    not (FOSMC_ROW.exists() and EXACT_PRINTED.exists()),
    reason="campaign E1_matrix artifacts not present")
needs_flagship = pytest.mark.skipif(
    not (STA_ROW.exists() and STA_EXACT.exists()),
    reason="flagship exactgains artifacts not present")


# ---------------------------------------------------------------------------
# Fixture: the Gram captures a capture-enabled re-run would persist
# ---------------------------------------------------------------------------

def _row_terms(row):
    return [(Fraction(str(t["coefficient"])),
             tuple(Fraction(e) for e in t["exponents"]),
             tuple(t["term_types"]))
            for t in row["V_terms"]]


def _diagonal_capture(terms, mu, signs, index):
    """A Gram capture for an octant target that is a single monomial.

    fosmc's V = |x1|^2 and its W = 2|x1| both associate to one y-monomial per
    octant, so the 1x1 Gram is the target coefficient itself and no solver is
    involved. The verifier still does the whole job: it recomputes the target
    from V and its own hand-written dynamics, forms the residual, and runs the
    exact LDL^T. A wrong entry here fails there.
    """
    target, degree = exact_octant_target(terms, mu, signs, Fraction(EPSILON))
    assert len(target) == 1, "fixture only covers single-monomial targets"
    (monomial, coefficient), = target.items()
    assert all(e % 2 == 0 for e in monomial)
    return {"signs": list(signs), "octant_index": index, "eps": EPSILON,
            "mu": list(mu), "y_degree": degree,
            "basis": [[e // 2 for e in monomial]],
            "gram": [[float(coefficient)]]}


def fosmc_grams_bundle():
    """V and W captures for the stored campaign fosmc row, both octants."""
    row = json.loads(FOSMC_ROW.read_text())
    v_terms = _row_terms(row)
    # W = -dV/dt for x1' = -k sign(x1) with the registered k = 1.
    w_terms = [(Fraction(2), (Fraction(1),), ("abs",))]
    mu = (2,)
    octants = [(-1,), (1,)]
    return {
        "timestamp_utc": "fixture", "system": "fosmc",
        "V_grams": [_diagonal_capture(v_terms, mu, s, i)
                    for i, s in enumerate(octants)],
        "corners": [{
            "corner_label": "rational gains only", "assignment": {},
            "W_grams": [_diagonal_capture(w_terms, mu, s, i)
                        for i, s in enumerate(octants)],
        }],
    }


def run_emitter(tmp_path, *extra, row=None, exact=None, grams=None,
                margin_policy="half"):
    """Run the emitter on a stored artifact.

    ``--margin-policy half`` is passed by default because every artifact
    these tests read predates the field: the June/August exact phases ran
    under the note's half-margin rule and recorded no policy. Since WP R-05
    the emitter refuses to guess that, so the operator states it and the
    record carries it as an override. The refusal itself is pinned in
    ``test_refuses_when_no_margin_policy_can_be_determined``.
    """
    cmd = [sys.executable, str(EMITTER),
           "--row", str(row or FOSMC_ROW),
           "--exact", str(exact or EXACT_PRINTED),
           "--out-dir", str(tmp_path)]
    if grams is not None:
        cmd += ["--grams", str(grams)]
    if margin_policy is not None:
        cmd += ["--margin-policy", margin_policy]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True)


def write_grams(tmp_path, bundle, name="grams.json"):
    path = tmp_path / name
    path.write_text(json.dumps(bundle))
    return path


# ---------------------------------------------------------------------------
# The refusal path: what the campaign can and cannot record today
# ---------------------------------------------------------------------------

@needs_campaign
def test_refuses_campaign_row_without_gram_captures(tmp_path):
    """The E1 benchmark exact artifact records verdicts, not certificates.

    This is the state of every nominal row of the 2026-08-02 matrix: the
    emitter must say so in a typed, indexable way instead of emitting a record
    that the standalone verifier could never replay.
    """
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix")
    assert res.returncode == 3, res.stdout
    payload = json.loads((tmp_path / "fosmc.refusal.json").read_text())
    assert payload["refused"] is True
    assert payload["reason"] == "missing_gram_captures"
    assert payload["required_rerun"], "a refusal must name the re-run"
    assert not list(tmp_path.glob("*.record.json.gz"))


@needs_campaign
def test_refuses_incomplete_octant_coverage(tmp_path):
    bundle = fosmc_grams_bundle()
    bundle["V_grams"] = bundle["V_grams"][:1]
    res = run_emitter(tmp_path, grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "incomplete_octant_coverage"


@needs_campaign
def test_refuses_duplicated_octant(tmp_path):
    """2^n capture entries that are not 2^n DISTINCT octants leave a true
    octant unproved while the count looks right."""
    bundle = fosmc_grams_bundle()
    bundle["V_grams"][1] = json.loads(json.dumps(bundle["V_grams"][0]))
    res = run_emitter(tmp_path, grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "incomplete_octant_coverage"


@needs_campaign
def test_refuses_endpoints_on_a_nominal_row(tmp_path):
    bundle = fosmc_grams_bundle()
    corner = bundle["corners"][0]
    corner["endpoints"] = [corner.pop("W_grams")]
    res = run_emitter(tmp_path, grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "nominal_perturbed_mismatch"


@needs_campaign
def test_refuses_missing_w_captures(tmp_path):
    bundle = fosmc_grams_bundle()
    bundle["corners"][0].pop("W_grams")
    res = run_emitter(tmp_path, grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "missing_gram_captures"


@needs_campaign
def test_refuses_epsilon_ladder_mismatch(tmp_path):
    """A capture that certified at a lower eps rung must not be recorded under
    the margin the record advertises."""
    bundle = fosmc_grams_bundle()
    bundle["V_grams"][0]["eps"] = EPSILON * 1e-3
    res = run_emitter(tmp_path, grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "epsilon_ladder_mismatch"


@needs_campaign
def test_refuses_a_gram_that_does_not_prove_its_target(tmp_path):
    """Self-verification is part of emission: a record the dependency-free
    verifier rejects is never written out as a record."""
    bundle = fosmc_grams_bundle()
    bundle["corners"][0]["W_grams"][0]["gram"][0][0] += 1.0
    res = run_emitter(tmp_path, grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "self_verification_failed"
    # A rejected record must not survive on disk under a record's name.
    assert not list(tmp_path.glob("*.record.json.gz"))
    assert not list(tmp_path.glob("*.grams.json.gz"))


# ---------------------------------------------------------------------------
# The emit-then-verify roundtrip
# ---------------------------------------------------------------------------

@needs_campaign
def test_emit_then_verify_nominal_fosmc(tmp_path):
    """End to end on the smallest nominal row: emit a gzipped schema-v2
    record from the stored campaign artifacts, then have the standalone
    verifier re-prove it in a subprocess with no repo imports."""
    grams = write_grams(tmp_path, fosmc_grams_bundle())
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      "--env", str(CAMPAIGN_ENV), grams=grams)
    assert res.returncode == 0, res.stdout + res.stderr
    emitted = json.loads(res.stdout)
    assert emitted["emitted"] is True and emitted["self_verified"] is True

    record_path = Path(emitted["record"])
    grams_path = Path(emitted["grams"])
    assert record_path.read_bytes()[:2] == b"\x1f\x8b"
    assert grams_path.read_bytes()[:2] == b"\x1f\x8b"

    check = subprocess.run([sys.executable, str(VERIFIER), str(record_path)],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    assert "VERIFIED" in check.stdout


@needs_campaign
def test_sidecar_grams_file_linked_from_matrix_row(tmp_path):
    """A matrix artifact whose ROW names its Gram sidecar emits without an
    explicit --grams. This is the shape ``review/reverify_exact.py
    --dump-grams`` writes (per-system ``grams_file`` link, gzipped bundle
    next to the artifact)."""
    exact = json.loads(EXACT_PRINTED.read_text())
    gname = "reverify_exact_printed_fixture_grams_fosmc.json.gz"
    for entry in exact["systems"]:
        if entry["system"] == "fosmc":
            entry["grams_file"] = gname
    exact_copy = tmp_path / EXACT_PRINTED.name
    exact_copy.write_text(json.dumps(exact))
    (tmp_path / gname).write_bytes(
        gzip.compress(json.dumps(fosmc_grams_bundle()).encode()))

    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      "--env", str(CAMPAIGN_ENV), exact=exact_copy)
    assert res.returncode == 0, res.stdout + res.stderr
    emitted = json.loads(res.stdout)
    assert emitted["emitted"] is True and emitted["self_verified"] is True

    check = subprocess.run([sys.executable, str(VERIFIER),
                            str(Path(emitted["record"]))],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    assert "VERIFIED" in check.stdout


@needs_campaign
def test_emitted_record_schema_and_encoding(tmp_path):
    """The schema-v2 fields of the E3 card, and the compact encoding."""
    grams = write_grams(tmp_path, fosmc_grams_bundle())
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      "--env", str(CAMPAIGN_ENV), grams=grams)
    assert res.returncode == 0, res.stdout + res.stderr
    record = json.loads(gzip.decompress(
        Path(json.loads(res.stdout)["record"]).read_bytes()))

    assert record["schema_version"] == "gf-certificate-record/2"
    for field in ("system_family", "channel", "nominal", "weights_r", "kappa",
                  "env", "campaign_block", "delta0_exact", "epsilon_exact",
                  "rational_gains", "brackets", "corners", "coverage",
                  "source_artifacts", "encoding", "grams_file"):
        assert field in record, field
    assert record["system_family"] == "levant_differentiator"
    assert record["nominal"] is True
    assert record["channel"] is None
    assert record["delta0_exact"] == "0"
    assert record["kappa"] == "-1"
    assert record["campaign_block"] == "E1_matrix"
    assert record["env"]["python_version"] == "3.12.3"
    assert {a["role"] for a in record["source_artifacts"]} == {
        "row", "exact_phase", "grams", "gain_point_registry"}
    assert all(a["sha256"] for a in record["source_artifacts"])

    # WP R-05: the margin is stated, its source is named, and free text and
    # the source stage's own numbers sit under the unverified provenance
    # object rather than beside the proved ones.
    assert record["margin_policy"] == "half"
    assert record["margin_policy_source"] == "operator_override"
    assert record["eps_certified"] is not None
    assert record["eps_certified_source"] == "portable_verifier"
    assert record["provenance"]["margin_policy_override"] == "half"
    assert "unverified provenance" in record["provenance"]["note"]
    assert record["verifier"]["eps_certified_exact"]
    # WP R-08: a self-verified record's margin is the verifier's own exact
    # rational, so it is not the exact value of some stage's rounded float.
    assert record["eps_certified_exact_is_float_transcription"] is False
    assert record["worst_charge_over_eps"] == \
        record["verifier"]["worst_charge_over_eps"]
    reported = record["provenance"]["source_reported_eps_certified"]
    assert "UNVERIFIED" in reported["note"]
    assert reported["value"] is None   # this artifact reports none at all

    # Published values exact, float64 in hex, and no em dash in any text field.
    assert record["rational_gains"] == {"k": "1"}
    assert record["V"]["terms"][0]["coefficient"].startswith("0x")
    assert_claim_hygiene(record)


def assert_claim_hygiene(record):
    """No em or en dashes in any text a record carries into the draft."""
    for value in _strings(record):
        assert "—" not in value and "–" not in value, value


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


@needs_campaign
def test_emission_is_content_reproducible(tmp_path):
    """Re-emitting an unchanged certificate must produce the same payload.

    Only the emission timestamp may differ (it is provenance, per binding rule
    4). Everything else, including key order, the hex encoding of every Gram
    entry, and the gzip container itself, has to be stable, or the determinism
    audit of E18 would report emitter noise as a certificate difference.
    """
    grams = write_grams(tmp_path, fosmc_grams_bundle())
    payloads = []
    for run in ("a", "b"):
        res = run_emitter(tmp_path / run, "--campaign-block", "E1_matrix",
                          grams=grams)
        assert res.returncode == 0, res.stdout + res.stderr
        emitted = json.loads(res.stdout)
        blob = Path(emitted["grams"]).read_bytes()
        # mtime 0 in the gzip header: the container adds no nondeterminism.
        assert blob[4:8] == b"\x00\x00\x00\x00"
        payloads.append([_without_timestamps(gzip.decompress(blob)),
                         _without_timestamps(
                             gzip.decompress(Path(emitted["record"]).read_bytes()))])
    assert payloads[0] == payloads[1]


def _without_timestamps(blob):
    payload = json.loads(blob)
    payload.pop("timestamp_utc", None)
    return json.dumps(payload, sort_keys=True)


@needs_flagship
def test_emit_then_verify_perturbed_sta(tmp_path):
    """The perturbed shape, from a stored artifact that really does carry Gram
    captures: 2 inclusion endpoints x 2^n octants, one bracket corner."""
    res = run_emitter(tmp_path, "--gain-point", "flagship",
                      "--campaign-block", "E3_backlog",
                      row=STA_ROW, exact=STA_EXACT)
    assert res.returncode == 0, res.stdout + res.stderr
    emitted = json.loads(res.stdout)
    assert emitted["nominal"] is False and emitted["self_verified"] is True

    record = json.loads(gzip.decompress(
        Path(emitted["record"]).read_bytes()))
    assert record["delta0_exact"] == "1"
    assert record["channel"] == 1
    assert record["rational_gains"] == {"k1": "3/2", "k2": "11/10"}
    assert record["coverage"]["inclusion_endpoints"] == 2
    assert_claim_hygiene(record)

    grams = json.loads(gzip.decompress(Path(emitted["grams"]).read_bytes()))
    assert len(grams["V_grams"]) == 4
    assert len(grams["corners"]) == 1
    assert len(grams["corners"][0]["endpoints"]) == 2
    assert all(len(caps) == 4 for caps in grams["corners"][0]["endpoints"])
    assert emitted["bytes_record_gz"] + emitted["bytes_grams_gz"] < \
        emitted["bytes_uncompressed"]


@needs_flagship
def test_refuses_perturbed_row_with_a_dropped_endpoint(tmp_path):
    """Half a perturbed record leaves half the disturbance interval unproved."""
    bundle = json.loads((FLAGSHIP / json.loads(
        STA_EXACT.read_text())["grams_file"]).read_text())
    for corner in bundle["corners"]:
        corner["endpoints"] = corner["endpoints"][:1]
    res = run_emitter(tmp_path, "--gain-point", "flagship",
                      row=STA_ROW, exact=STA_EXACT,
                      grams=write_grams(tmp_path, bundle))
    assert res.returncode == 3
    assert json.loads(res.stdout)["reason"] == "missing_endpoint_captures"


# ---------------------------------------------------------------------------
# WP R-05: the shared nominal W witness, and the refusals around it
# ---------------------------------------------------------------------------

SYSTEM_PYTHON = Path("/usr/bin/python3")

needs_system_python = pytest.mark.skipif(
    not SYSTEM_PYTHON.exists(),
    reason="/usr/bin/python3 (the dependency-free interpreter) not present")


def verify_with(interpreter, record_path, summary_path=None):
    cmd = [str(interpreter), str(VERIFIER), str(record_path)]
    if summary_path is not None:
        cmd += ["--json", str(summary_path)]
    return subprocess.run(cmd, capture_output=True, text=True)


@needs_campaign
@needs_system_python
def test_shared_w_roundtrip_matches_the_unshared_emission(tmp_path):
    """--share-w-grams must change the BYTES and nothing else.

    The same nominal row is emitted twice, once with the W captures written
    per corner and once with the single shared set the portable verifier
    resolves through ``nominal_w_captures``. Both are re-proved by the
    dependency-free verifier under /usr/bin/python3, and both must reach the
    same verdict with the same exact charges: sharing is storage aliasing on
    the decode side, so a difference in any charge would mean a corner was
    proved against a Gram it does not own.
    """
    grams = write_grams(tmp_path, fosmc_grams_bundle())
    verdicts = {}
    for mode, extra in (("plain", ()), ("shared", ("--share-w-grams",))):
        out = tmp_path / mode
        res = run_emitter(out, "--campaign-block", "E1_matrix", *extra,
                          grams=grams)
        assert res.returncode == 0, res.stdout + res.stderr
        emitted = json.loads(res.stdout)
        record = json.loads(gzip.decompress(
            Path(emitted["record"]).read_bytes()))
        bundle = json.loads(gzip.decompress(
            Path(emitted["grams"]).read_bytes()))
        shared = mode == "shared"
        assert record["shared_W_witness"] is shared
        assert ("shared_W_grams" in bundle) is shared
        for corner in bundle["corners"]:
            assert ("W_grams" in corner) is not shared
            assert corner.get("W_grams_shared", False) is shared
        if shared:
            # The shared set is exactly what the per-corner emission wrote.
            plain = json.loads(gzip.decompress(
                (tmp_path / "plain" / Path(emitted["grams"]).name).read_bytes()))
            assert bundle["shared_W_grams"] == plain["corners"][0]["W_grams"]

        summary_path = out / "verdict.json"
        check = verify_with(SYSTEM_PYTHON, Path(emitted["record"]),
                            summary_path)
        assert check.returncode == 0, check.stdout + check.stderr
        assert "VERIFIED" in check.stdout
        verdicts[mode] = json.loads(summary_path.read_text())

    for key in ("verdict", "worst_charge_over_eps", "eps_certified_exact",
                "note_half_margin_holds", "epsilon_exact"):
        assert verdicts["plain"][key] == verdicts["shared"][key], key

    # WP R-08: what each record STATES about its margin is the verifier's own
    # derivation, named as such. Nothing here pins the exact value of a
    # rounded float as if it were the proof margin.
    for mode in ("plain", "shared"):
        record = json.loads(gzip.decompress(
            next((tmp_path / mode).glob("*.record.json.gz")).read_bytes()))
        assert record["eps_certified_source"] == "portable_verifier"
        assert record["eps_certified_exact_is_float_transcription"] is False
        assert record["eps_certified_exact"] == \
            verdicts[mode]["eps_certified_exact"]
        assert record["eps_certified"] == float(
            Fraction(verdicts[mode]["eps_certified_exact"]))


@needs_flagship
def test_refuses_shared_w_on_a_perturbed_row(tmp_path):
    """A perturbed W depends on the inclusion endpoint as well, so there is no
    single set to share. The verifier refuses the shape there; so does this."""
    res = run_emitter(tmp_path, "--gain-point", "flagship", "--share-w-grams",
                      row=STA_ROW, exact=STA_EXACT)
    assert res.returncode == 3, res.stdout
    assert json.loads(res.stdout)["reason"] == "nominal_perturbed_mismatch"


def _multi_corner_fixture(tmp_path, second_corner_differs):
    """A two-corner NOMINAL fixture for levant3d's flagship bracket box.

    The Grams are placeholders: every refusal these fixtures pin fires while
    the corner box is assembled, long before any Gram proves anything, and
    the emitted record never leaves ``tmp_path``. What the fixture does carry
    faithfully is the SHAPE: 2^3 distinct octants per form, one margin, and
    one capture set per bracket corner.
    """
    def capture(index, signs, value):
        return {"signs": list(signs), "octant_index": index, "eps": EPSILON,
                "mu": [2, 2, 2], "y_degree": 6, "basis": [[1, 1, 1]],
                "gram": [[value]]}

    octants = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    corners = []
    for corner_index in range(2):
        offset = 1.0 if (second_corner_differs and corner_index == 1) else 0.0
        corners.append({
            "corner_label": f"corner {corner_index}",
            "W_grams": [capture(i, s, 2.0 + offset)
                        for i, s in enumerate(octants)],
        })
    bundle = {"system": "levant3d", "timestamp_utc": "fixture",
              "V_grams": [capture(i, s, 1.0) for i, s in enumerate(octants)],
              "corners": corners}
    row = {"system": "levant3d", "weights_r": [3, 2, 1], "degree_used": 4,
           "certified": True,
           "V_terms": [{"coefficient": 1.0,
                        "exponents": ["4/3", "0", "0"],
                        "term_types": ["abs", "abs", "abs"]}]}
    exact = {"system": "levant3d", "epsilon": EPSILON, "exact_pass": True,
             "gain_binding": "printed_rationals",
             "witness": {"source": "design_bundle", "margin_policy": "half",
                         "eps_certified": EPSILON / 2}}
    row_path = tmp_path / "levant3d_row.json"
    exact_path = tmp_path / "levant3d_exact.json"
    row_path.write_text(json.dumps(row))
    exact_path.write_text(json.dumps(exact))
    return row_path, exact_path, write_grams(tmp_path, bundle)


def test_refuses_shared_w_when_corners_name_different_grams(tmp_path):
    """Sharing is a claim that every corner has the same witness.

    When one corner's captures differ, choosing which of the two Gram sets is
    "the" witness is not a decision an emitter may make, so it refuses. This
    is the check that stands in for the entry-hash comparison when a source
    carries no exact-phase hashes: the encoded captures themselves are
    hashed, which is strictly stronger.
    """
    row, exact, grams = _multi_corner_fixture(tmp_path,
                                              second_corner_differs=True)
    res = run_emitter(tmp_path, "--gain-point", "flagship", "--share-w-grams",
                      row=row, exact=exact, grams=grams)
    assert res.returncode == 3, res.stdout
    payload = json.loads(res.stdout)
    assert payload["reason"] == "shared_w_mismatch"
    assert "capture" in payload["detail"]
    assert not list(tmp_path.glob("*.record.json.gz"))


def test_shares_w_across_corners_that_name_the_same_grams(tmp_path):
    """The positive half of the same fixture: 2^k corners, one capture set.

    Self-verification is skipped here because the fixture's Grams are
    placeholders; what is pinned is the collapse itself, on more than one
    bracket corner, which the single-corner roundtrip above cannot show.
    """
    row, exact, grams = _multi_corner_fixture(tmp_path,
                                              second_corner_differs=False)
    res = run_emitter(tmp_path, "--gain-point", "flagship", "--share-w-grams",
                      "--no-self-verify", row=row, exact=exact, grams=grams)
    assert res.returncode == 0, res.stdout + res.stderr
    emitted = json.loads(res.stdout)
    bundle = json.loads(gzip.decompress(Path(emitted["grams"]).read_bytes()))
    assert len(bundle["shared_W_grams"]) == 8
    assert len(bundle["corners"]) == 2
    assert all(c["W_grams_shared"] is True and "W_grams" not in c
               for c in bundle["corners"])

    # WP R-08: no portable stage ran here, so the margin is the source's own
    # report and the record says exactly that. Its "exact" string is the
    # exact value of the float64 the source stored, and it is flagged.
    record = json.loads(gzip.decompress(Path(emitted["record"]).read_bytes()))
    assert record["eps_certified_source"] == \
        "exact_phase_unverified_by_portable_stage"
    assert record["eps_certified_exact_is_float_transcription"] is True
    assert record["eps_certified"] == EPSILON / 2
    assert Fraction(record["eps_certified_exact"]) == Fraction(EPSILON / 2)
    assert record["provenance"]["source_reported_eps_certified"]["value"] == \
        EPSILON / 2


@needs_campaign
def test_refuses_when_no_margin_policy_can_be_determined(tmp_path):
    """The 2026-09-05 review's finding 4, as a test.

    The stored artifact records no margin policy. Before WP R-05 the emitter
    wrote ``"half"`` anyway, so the record advertised an acceptance rule that
    nothing in its sources states. Now it refuses, and names the override
    that an operator who knows the rule can supply.
    """
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      margin_policy=None,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 3, res.stdout
    payload = json.loads(res.stdout)
    assert payload["reason"] == "missing_margin_policy"
    assert any("--margin-policy" in hint for hint in payload["required_rerun"])
    assert not list(tmp_path.glob("*.record.json.gz"))


@needs_campaign
def test_refuses_a_margin_policy_that_contradicts_the_exact_phase(tmp_path):
    """An override may fill a hole; it may not overrule a proof stage."""
    exact = json.loads(EXACT_PRINTED.read_text())
    for entry in exact["systems"]:
        if entry["system"] == "fosmc":
            entry["witness"] = {"source": "resolve", "margin_policy": "half",
                                "eps_certified": EPSILON / 2}
    exact_path = tmp_path / EXACT_PRINTED.name
    exact_path.write_text(json.dumps(exact))
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      margin_policy="achieved", exact=exact_path,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 3, res.stdout
    assert json.loads(res.stdout)["reason"] == "missing_margin_policy"


# ---------------------------------------------------------------------------
# WP R-08: the margin a self-verified record states is the one that was proved
# (follow-up review 2026-09-05, section B)
# ---------------------------------------------------------------------------

def _exact_with_reported_margin(tmp_path, value, name="exact_reported.json"):
    """The stored fosmc exact artifact, with a reported achieved margin."""
    exact = json.loads(EXACT_PRINTED.read_text())
    for entry in exact["systems"]:
        if entry["system"] == "fosmc":
            entry["witness"] = {"source": "resolve", "margin_policy": "half",
                                "eps_certified": value}
    path = tmp_path / name
    path.write_text(json.dumps(exact))
    return path


@needs_campaign
def test_refuses_a_source_margin_larger_than_the_re_proof(tmp_path):
    """The follow-up review's probe, as a test.

    The reviewer changed the source's reported ``witness.eps_certified`` to
    999 and got exit 0, ``self_verified: true`` and a record that said
    ``eps_certified: 999.0`` beside a summary deriving about 1e-6. The proof
    was sound and the advertised number was unchecked. A producer that reports
    more than its own captures prove is broken, so the emission refuses
    instead of quietly publishing the smaller proved number.
    """
    exact = _exact_with_reported_margin(tmp_path, 999.0)
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix", exact=exact,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 3, res.stdout + res.stderr
    payload = json.loads(res.stdout)
    assert payload["reason"] == "margin_claim_exceeds_proof"
    assert "999" in payload["detail"]
    # Nothing survives on disk that a cut-line audit could mistake for a
    # record, and the refusal names the number the proof actually supports.
    assert not list(tmp_path.glob("*.record.json.gz"))
    assert not list(tmp_path.glob("*.grams.json.gz"))
    assert not list(tmp_path.glob("*.verify_summary.json"))


@needs_campaign
def test_verifier_margin_wins_over_a_smaller_source_report(tmp_path):
    """A conservative source report is kept, but it is not what ships.

    1e-9 is below the margin the captures prove, so it is not an overclaim.
    The record still states the verifier's own exact rational, because that
    is the number the shipped bytes were accepted for; the source's report
    goes to unverified provenance, where nothing may quote it.
    """
    exact = _exact_with_reported_margin(tmp_path, 1e-9)
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix", exact=exact,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 0, res.stdout + res.stderr
    emitted = json.loads(res.stdout)
    record = json.loads(gzip.decompress(Path(emitted["record"]).read_bytes()))
    summary = json.loads(
        (tmp_path / "fosmc_nominal.verify_summary.json").read_text())

    assert summary["verdict"] == "VERIFIED"
    assert record["eps_certified_source"] == "portable_verifier"
    assert record["eps_certified_exact"] == summary["eps_certified_exact"]
    assert record["eps_certified"] == float(
        Fraction(summary["eps_certified_exact"]))
    assert record["eps_certified"] > 1e-9
    assert record["eps_certified_exact_is_float_transcription"] is False
    assert record["worst_charge_over_eps"] == summary["worst_charge_over_eps"]
    assert record["note_half_margin_holds"] == \
        summary["note_half_margin_holds"]["overall"]

    reported = record["provenance"]["source_reported_eps_certified"]
    assert reported["value"] == 1e-9
    assert Fraction(reported["exact"]) == Fraction(1e-9)
    assert "UNVERIFIED" in reported["note"]


@needs_campaign
@needs_system_python
def test_attach_verify_summary_writes_a_sidecar_and_keeps_the_bytes(tmp_path):
    """The levant5d shape: emit without self-verification, attach later.

    A ten-hour portable proof cannot run inside an emission, so that record
    ships with the exact phase's own reported number, labelled unverified.
    When the proof finishes, its summary is attached BESIDE the record: the
    verified bytes are never rewritten, or the shipped record would no longer
    be the record that was proved.
    """
    exact = _exact_with_reported_margin(tmp_path, 1e-9)
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      "--no-self-verify", exact=exact,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 0, res.stdout + res.stderr
    record_path = Path(json.loads(res.stdout)["record"])
    before = hashlib.sha256(record_path.read_bytes()).hexdigest()
    record = json.loads(gzip.decompress(record_path.read_bytes()))
    assert record["eps_certified_source"] == \
        "exact_phase_unverified_by_portable_stage"
    assert record["eps_certified_exact_is_float_transcription"] is True

    summary_path = tmp_path / "late_run.json"
    check = verify_with(SYSTEM_PYTHON, record_path, summary_path)
    assert check.returncode == 0, check.stdout + check.stderr

    attach = subprocess.run(
        [sys.executable, str(EMITTER), "--record", str(record_path),
         "--attach-verify-summary", str(summary_path)],
        capture_output=True, text=True)
    assert attach.returncode == 0, attach.stdout + attach.stderr
    sidecar = json.loads(
        (tmp_path / "fosmc_nominal.verified_margin.json").read_text())
    summary = json.loads(summary_path.read_text())

    assert sidecar["verifier"]["eps_certified_exact"] == \
        summary["eps_certified_exact"]
    assert sidecar["authoritative_eps_certified_exact"] == \
        summary["eps_certified_exact"]
    assert sidecar["record_reported"]["eps_certified"] == 1e-9
    assert sidecar["record_reported"]["eps_certified_source"] == \
        "exact_phase_unverified_by_portable_stage"
    assert sidecar["margin_claim_exceeds_proof"] is False
    assert sidecar["record_sha256"] == before
    # The record itself is untouched: same bytes, same unverified label.
    assert hashlib.sha256(record_path.read_bytes()).hexdigest() == before


@needs_campaign
@needs_system_python
def test_attach_verify_summary_refuses_a_summary_from_another_record(tmp_path):
    """A sidecar is only evidence if it is evidence about THIS record."""
    exact = _exact_with_reported_margin(tmp_path, 1e-9)
    res = run_emitter(tmp_path, "--campaign-block", "E1_matrix",
                      "--no-self-verify", exact=exact,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 0, res.stdout + res.stderr
    record_path = Path(json.loads(res.stdout)["record"])
    summary_path = tmp_path / "late_run.json"
    assert verify_with(SYSTEM_PYTHON, record_path,
                       summary_path).returncode == 0

    summary = json.loads(summary_path.read_text())
    summary["record"] = "somewhere/else/other_nominal.record.json.gz"
    other = tmp_path / "foreign_summary.json"
    other.write_text(json.dumps(summary))
    attach = subprocess.run(
        [sys.executable, str(EMITTER), "--record", str(record_path),
         "--attach-verify-summary", str(other)],
        capture_output=True, text=True)
    assert attach.returncode == 3, attach.stdout + attach.stderr
    assert json.loads(attach.stdout)["reason"] == "source_pin_mismatch"
    assert not list(tmp_path.glob("*.verified_margin.json"))


@needs_campaign
@needs_system_python
@pytest.mark.parametrize("change", ["record", "grams", "legacy", "verifier"])
def test_attach_summary_requires_the_inputs_actually_verified(tmp_path, change):
    """An old successful verdict must not authenticate changed V or Gram bytes."""
    exact = _exact_with_reported_margin(tmp_path, 1e-9)
    res = run_emitter(tmp_path, "--no-self-verify", exact=exact,
                      grams=write_grams(tmp_path, fosmc_grams_bundle()))
    assert res.returncode == 0, res.stdout + res.stderr
    record_path = Path(json.loads(res.stdout)["record"])
    summary_path = tmp_path / "proof.json"
    check = verify_with(SYSTEM_PYTHON, record_path, summary_path)
    assert check.returncode == 0, check.stdout + check.stderr
    summary = json.loads(summary_path.read_text())
    record = json.loads(gzip.decompress(record_path.read_bytes()))
    binding = summary["proof_inputs"]
    assert binding["record"]["sha256"] == hashlib.sha256(record_path.read_bytes()).hexdigest()
    assert binding["grams"]["sha256"] == hashlib.sha256(
        Path(binding["grams"]["path"]).read_bytes()).hexdigest()
    if change == "record":
        record["V"]["terms"][0]["coefficient"] = "-0x1.0000000000000p+0"
        record_path.write_bytes(gzip.compress(json.dumps(record).encode()))
    elif change == "grams":
        path = Path(binding["grams"]["path"])
        data = json.loads(gzip.decompress(path.read_bytes()))
        data["test_mutation"] = True
        path.write_bytes(gzip.compress(json.dumps(data).encode()))
    elif change == "legacy":
        del summary["proof_inputs"]
        summary_path.write_text(json.dumps(summary))
    else:
        binding["verifier"]["sha256"] = "0" * 64
        summary_path.write_text(json.dumps(summary))
    attach = subprocess.run(
        [sys.executable, str(EMITTER), "--record", str(record_path),
         "--attach-verify-summary", str(summary_path)], capture_output=True, text=True)
    assert attach.returncode == 3, attach.stdout + attach.stderr
    assert json.loads(attach.stdout)["reason"] == (
        "proof_input_binding_missing" if change == "legacy" else "proof_input_mismatch")
    assert not list(tmp_path.glob("*.verified_margin.json"))
