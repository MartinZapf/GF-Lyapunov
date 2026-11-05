"""Outcome classes of the standalone verifier (2026-11-05).

Moreno's evaluation of 2026-11-04, Section 6, last paragraph: the verifier
should distinguish an accepted certificate, a rejected one, and missing or
invalid input, "so that a reproducibility problem is never mistaken for a
mathematical counterexample". ``verification/standalone/verify_certificate.py``
now ends every run with one ``RESULT:`` line and the exit codes

    0 VERIFIED, 1 REJECTED, 2 INVALID INPUT, 3 ERROR (no verdict).

These tests pin the classification on small published records (the worked
super-twisting example and the nine-term continuous-twisting function) and
on tampered copies. They never touch the acceptance arithmetic: every
tampered input below was refused before the change as well; only its class,
message and exit code are asserted here.
"""
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "verification" / "standalone" / "verify_certificate.py"
sys.path.insert(0, str(VERIFIER.parent))

import verify_certificate as vc  # noqa: E402

STA_DIR = ROOT / "artifacts/campaign_2026-10/WORKED_STA/tierB/printed"
STA_REC = STA_DIR / "sta_exactgains_20261012T072209Z.printed_b07.record.json"
STA_GRAMS = STA_DIR / "sta_exactgains_grams_20261012T072209Z.json"
CTA_DIR = ROOT / "artifacts/campaign_2026-10/SHORT_COEFFS/cta_g1_b05/run/m5/portable"
CTA_REC = CTA_DIR / "continuous_twisting_cta_b05_e9_delta1.record.json.gz"
CTA_GRAMS = CTA_DIR / "continuous_twisting_cta_b05_e9_delta1.grams.json.gz"

needs_sta = pytest.mark.skipif(not (STA_REC.exists() and STA_GRAMS.exists()),
                               reason="worked super-twisting record not present")
needs_cta = pytest.mark.skipif(not (CTA_REC.exists() and CTA_GRAMS.exists()),
                               reason="short continuous-twisting record not present")

RESULT_RE = re.compile(r"^RESULT: (VERIFIED|REJECTED|INVALID INPUT|ERROR) "
                       r"\(exit code (\d)\)")


def run(*args):
    return subprocess.run([sys.executable, str(VERIFIER), *map(str, args)],
                          capture_output=True, text=True, timeout=300)


def outcome(res):
    """(class, exit code) from the last RESULT line, which must agree."""
    lines = [l for l in (res.stdout + res.stderr).splitlines()
             if l.startswith("RESULT:")]
    assert len(lines) == 1, res.stdout + res.stderr
    m = RESULT_RE.match(lines[0])
    assert m, lines[0]
    assert int(m.group(2)) == res.returncode, (lines[0], res.returncode)
    return m.group(1), res.returncode


def load(path):
    return vc.read_json(path)


def write_pair(tmp_path, rec, grams, gz=False):
    """Write a (record, bundle) pair next to each other, the record naming
    the bundle, and return the record path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    g = tmp_path / ("grams.json.gz" if gz else "grams.json")
    r = tmp_path / "rec.json"
    rec = dict(rec, grams_file=g.name)
    raw = json.dumps(grams).encode("utf-8")
    g.write_bytes(gzip.compress(raw, mtime=0) if gz else raw)
    r.write_text(json.dumps(rec))
    return r


# ---------------------------------------------------------------------------
# The class hierarchy: in-process callers that catch VerifyError are unchanged
# ---------------------------------------------------------------------------

def test_both_refusal_classes_are_verify_errors():
    assert issubclass(vc.RejectedError, vc.VerifyError)
    assert issubclass(vc.InvalidInputError, vc.VerifyError)
    assert (vc.EXIT_VERIFIED, vc.EXIT_REJECTED, vc.EXIT_INVALID_INPUT,
            vc.EXIT_ERROR) == (0, 1, 2, 3)


def test_every_refusal_in_the_verifier_is_classified():
    """No bare ``raise VerifyError(``: each refusal names its class."""
    src = VERIFIER.read_text(encoding="utf-8")
    assert "raise VerifyError(" not in src
    assert src.count("raise RejectedError(") > 20
    assert src.count("raise InvalidInputError(") > 50


# ---------------------------------------------------------------------------
# VERIFIED
# ---------------------------------------------------------------------------

@needs_sta
def test_verified_ends_with_the_result_line():
    res = run(STA_REC)
    assert outcome(res) == ("VERIFIED", 0)
    assert res.stdout.rstrip().splitlines()[-1].startswith("RESULT: VERIFIED")
    assert res.stderr == ""


# ---------------------------------------------------------------------------
# INVALID INPUT: reproducibility problems, never a mathematical rejection
# ---------------------------------------------------------------------------

def test_missing_record_file_is_invalid_input(tmp_path):
    res = run(tmp_path / "nope.record.json")
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "INVALID INPUT: cannot read the record file" in res.stderr
    assert "VERIFICATION FAILED" not in res.stderr


def test_usage_error_is_invalid_input():
    res = run()
    assert outcome(res) == ("INVALID INPUT", 2)


def test_help_exits_zero_without_a_result_line():
    res = run("--help")
    assert res.returncode == 0
    assert "RESULT:" not in res.stdout + res.stderr


def test_malformed_json_is_invalid_input(tmp_path):
    bad = tmp_path / "rec.json"
    bad.write_text('{"system": "sta", ')
    res = run(bad)
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "not valid UTF-8 JSON" in res.stderr


def test_json_that_is_not_an_object_is_invalid_input(tmp_path):
    bad = tmp_path / "rec.json"
    bad.write_text("[1, 2, 3]")
    res = run(bad)
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "record must be a JSON object" in res.stderr


@needs_cta
def test_truncated_gzip_download_is_invalid_input(tmp_path):
    raw = CTA_REC.read_bytes()
    cut = tmp_path / "rec.json.gz"
    cut.write_bytes(raw[: len(raw) // 2])
    res = run(cut)
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "not a readable gzip stream" in res.stderr


@needs_cta
def test_missing_gram_bundle_is_invalid_input(tmp_path):
    rec = load(CTA_REC)
    rec["grams_file"] = "absent.grams.json.gz"
    r = tmp_path / "rec.json"
    r.write_text(json.dumps(rec))
    res = run(r)
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "not found next to the record" in res.stderr
    res = run(r, "--grams", tmp_path / "absent.json")
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "cannot read the grams file" in res.stderr


@needs_cta
@pytest.mark.parametrize("mutate, needle", [
    (lambda r, g: r.pop("epsilon_exact"), "record has no 'epsilon_exact'"),
    (lambda r, g: r.__setitem__("delta0_exact", 1.0), "delta0_exact must be"),
    (lambda r, g: r.__setitem__("system", "levant99d"), "no hand-written dynamics"),
    (lambda r, g: r.__setitem__("schema_version", "gf-certificate-record/9"),
     "unknown record schema"),
    (lambda r, g: r["rational_gains"].__setitem__("k3", "2.3.1"),
     "not a rational number"),
    (lambda r, g: r["rational_gains"].__setitem__("bogus", "1"),
     "not a parameter of the published"),
    (lambda r, g: g["V_grams"][0].pop("gram"), "has no 'gram'"),
    (lambda r, g: g["V_grams"][0].__setitem__("gram", [[1]]), "gram shape mismatch"),
    # A missing key that no explicit check names: still INVALID INPUT.
    (lambda r, g: r["corners"][0].clear() or r["corners"].__setitem__(0, []),
     "malformed record or Gram bundle"),
])
def test_schema_and_type_violations_are_invalid_input(tmp_path, mutate, needle):
    rec, grams = load(CTA_REC), load(CTA_GRAMS)
    mutate(rec, grams)
    res = run(write_pair(tmp_path, rec, grams, gz=True))
    assert outcome(res) == ("INVALID INPUT", 2), res.stderr
    assert needle in res.stderr, res.stderr
    assert "VERIFICATION FAILED" not in res.stderr


@needs_sta
def test_unwritable_json_summary_is_invalid_input_after_the_proof(tmp_path):
    res = run(STA_REC, "--json", tmp_path / "no" / "such" / "dir" / "s.json")
    assert outcome(res) == ("INVALID INPUT", 2)
    assert "the certificate verified, but the --json summary" in res.stderr


# ---------------------------------------------------------------------------
# REJECTED: a mathematical check failed on well-formed input
# ---------------------------------------------------------------------------

@needs_sta
def test_corrupted_gram_entry_is_rejected(tmp_path):
    rec, grams = load(STA_REC), load(STA_GRAMS)
    g = grams["V_grams"][0]["gram"]
    g[0][0] = float(vc.q_from(g[0][0]) + 1)
    res = run(write_pair(tmp_path, rec, grams))
    assert outcome(res) == ("REJECTED", 1)
    assert "VERIFICATION FAILED:" in res.stderr and "FAILED (" in res.stderr
    assert "INVALID INPUT" not in res.stderr


@needs_sta
def test_dropped_octant_is_rejected_as_incomplete_coverage(tmp_path):
    rec, grams = load(STA_REC), load(STA_GRAMS)
    grams["V_grams"] = grams["V_grams"][:-1]
    res = run(write_pair(tmp_path, rec, grams))
    assert outcome(res) == ("REJECTED", 1)
    assert "octant coverage" in res.stderr


@needs_sta
def test_dropped_inclusion_endpoint_is_rejected(tmp_path):
    rec, grams = load(STA_REC), load(STA_GRAMS)
    grams["corners"][0]["endpoints"] = grams["corners"][0]["endpoints"][:1]
    res = run(write_pair(tmp_path, rec, grams))
    assert outcome(res) == ("REJECTED", 1)
    assert "need 2 (W+ and W-)" in res.stderr


@needs_sta
def test_non_c1_function_is_rejected(tmp_path):
    rec, grams = load(STA_REC), load(STA_GRAMS)
    for t in rec["V"]["terms"]:
        if [str(e) for e in t["exponents"]][0] in ("1", "1/1") \
                and t["term_types"][0] == "signed":
            t["term_types"][0] = "abs"      # x1 -> |x1|: not C^1 at x1 = 0
            break
    else:
        pytest.skip("no signed exponent-1 factor in this record")
    res = run(write_pair(tmp_path, rec, grams))
    assert outcome(res) == ("REJECTED", 1)
    assert "V not C^1" in res.stderr


@needs_cta
def test_margin_claim_above_the_proof_is_rejected(tmp_path):
    rec, grams = load(CTA_REC), load(CTA_GRAMS)
    rec["eps_certified_exact"] = rec["epsilon_exact"]   # claims charge = 0
    rec.pop("eps_certified", None)
    rec.pop("eps_certified_exact_is_float_transcription", None)
    res = run(write_pair(tmp_path, rec, grams, gz=True))
    assert outcome(res) == ("REJECTED", 1)
    assert "MARGIN CLAIM EXCEEDS PROOF" in res.stderr


def test_invalid_bracket_is_rejected(tmp_path):
    """A bracket that misses its gain, on the worked-example shape: the gain
    k1 = 3/2 is bracketed by [1, 5/4], whose squares miss 9/4."""
    if not (STA_REC.exists() and STA_GRAMS.exists()):
        pytest.skip("worked super-twisting record not present")
    rec, grams = load(STA_REC), load(STA_GRAMS)
    rec["rational_gains"].pop("k1")
    rec["brackets"] = {"k1": {"lo": "1", "hi": "5/4", "power_k": 2,
                              "power_target": "9/4"}}
    res = run(write_pair(tmp_path, rec, grams))
    assert outcome(res) == ("REJECTED", 1)
    assert "bracket k1 INVALID" in res.stderr


# ---------------------------------------------------------------------------
# ERROR: no verdict, never confused with REJECTED (Python's own exit code 1)
# ---------------------------------------------------------------------------

def test_unexpected_condition_is_error_not_rejected(monkeypatch, capsys):
    def boom():
        raise MemoryError("simulated")
    monkeypatch.setattr(vc, "main", boom)
    assert vc.cli() == vc.EXIT_ERROR
    err = capsys.readouterr().err
    assert "RESULT: ERROR (exit code 3)" in err
    assert "VERIFICATION FAILED" not in err


def test_in_process_main_still_raises_verify_error(monkeypatch, tmp_path):
    """scripts/round_certificate.py calls main() in-process and catches
    VerifyError; the classification must not change that contract."""
    bad = tmp_path / "rec.json"
    bad.write_text("not json")
    monkeypatch.setattr(sys, "argv", ["verify_certificate.py", str(bad)])
    with pytest.raises(vc.VerifyError) as exc:
        vc.main()
    assert isinstance(exc.value, vc.InvalidInputError)
