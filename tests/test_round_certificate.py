"""scripts/round_certificate.py (X9 tier A): dyadic coefficient rounding.

Covers:
  1. round_to_dyadic_sig is genuinely dyadic (power-of-2 denominator, or 0),
     matches the input to the requested number of significant decimal
     digits, and handles sign and zero correctly;
  2. an end-to-end run against a REAL, checked-in flagship record (sta, the
     smallest of the three ABL records) and its real Gram sidecar: this
     drives the actual verify_certificate.py as a subprocess exactly as the
     tool does in production, and checks (a) low N genuinely FAILS, (b) the
     smallest verified N matches the one this work package's own run found
     (8 significant digits, recorded in
     artifacts/campaign_2026-09/X9/tierA_demo/sta.round_summary.json), and
     (c) the acceptance criteria: every other field (gains, brackets,
     delta0, corners, exponents, term_types, degree_m, weights_r) is
     unchanged, and the winning record's coefficients are string-equal to
     what round_to_dyadic_sig computed;
  3. refusals: no N in range verifies, and a record missing 'grams_file'.
"""
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "round_certificate.py"
VC_PATH = ROOT / "verification" / "standalone" / "verify_certificate.py"
STA_RECORD = ROOT / "artifacts" / "ablation" / "levant_2003_perturbed" / \
    "sta_exactgains_20260712T062215Z.json"
STA_GRAMS = ROOT / "artifacts" / "ablation" / "levant_2003_perturbed" / \
    "sta_exactgains_grams_20260712T062215Z.json"

sys.path.insert(0, str(ROOT / "scripts"))
import round_certificate as rc  # noqa: E402


# ---------------------------------------------------------------------------
# 1. round_to_dyadic_sig.
# ---------------------------------------------------------------------------

def _is_dyadic(x: Fraction) -> bool:
    d = x.denominator
    while d % 2 == 0:
        d //= 2
    return d == 1


@pytest.mark.parametrize("value", [
    "0.04166419414335491", "-0.5293979093949359", "1.7236798638858641",
    "-1.8024244095738209", "1", "-1", "3.14159265358979", "1/3", "-7/11",
    "1000000.5", "0.0000001234567",
])
@pytest.mark.parametrize("sig", [1, 2, 3, 5, 7, 10])
def test_round_to_dyadic_sig_is_dyadic_and_within_half_ulp(value, sig):
    """round_to_dyadic_sig's actual, unconditional guarantee: the result is
    dyadic and within half a decimal-ULP-at-`sig`-digits of the input. This
    is what the implementation's bisection loop establishes directly (see
    round_to_dyadic_sig's docstring/comments); it is NOT the same as "always
    re-rounds to the identical N-digit decimal", which can legitimately
    differ when the true value sits within that same half-ULP of its own
    decimal tie boundary (e.g. 1/3 at 5 digits: true value 0.333333... is
    only 1.67e-6 below the 0.333335 tie, well inside the 5e-6 half-ULP
    budget, so a maximally-adversarial approximation on the correct side of
    the error bound can still cross the tie -- that is expected, not a bug)."""
    c = Fraction(value)
    r = rc.round_to_dyadic_sig(c, sig)
    assert _is_dyadic(r)
    e = rc.decimal_exponent(abs(c))
    tol = Fraction(1, 2) * Fraction(10) ** (e - sig + 1)
    assert abs(r - c) <= tol


@pytest.mark.parametrize("value", [
    "0.04166419414335491", "-1.8024244095738209", "3.14159265358979", "1/3",
])
def test_round_to_dyadic_sig_error_shrinks_with_more_digits(value):
    c = Fraction(value)
    errors = [abs(rc.round_to_dyadic_sig(c, sig) - c) for sig in range(1, 12)]
    for a, b in zip(errors, errors[1:]):
        assert b <= a


def test_round_to_dyadic_sig_zero():
    assert rc.round_to_dyadic_sig(Fraction(0), 5) == Fraction(0)


def test_round_to_dyadic_sig_rejects_bad_sig():
    with pytest.raises(rc.RoundCertificateError):
        rc.round_to_dyadic_sig(Fraction(1), 0)


def test_round_record_v_only_changes_coefficients():
    record = {
        "system": "toy",
        "rational_gains": {"k": "1"},
        "V": {"degree_m": 4, "weights_r": [1],
              "terms": [{"coefficient": "0.333333333333", "exponents": ["4"],
                         "term_types": ["abs"]}]},
    }
    new_record, report = rc.round_record_v(record, 3)
    assert new_record["system"] == "toy"
    assert new_record["rational_gains"] == {"k": "1"}
    assert new_record["V"]["terms"][0]["exponents"] == ["4"]
    assert new_record["V"]["terms"][0]["term_types"] == ["abs"]
    rounded = Fraction(new_record["V"]["terms"][0]["coefficient"])
    assert _is_dyadic(rounded)
    assert report[0]["rounded"] == new_record["V"]["terms"][0]["coefficient"]


# ---------------------------------------------------------------------------
# 2. End-to-end against the real sta flagship record.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not STA_RECORD.is_file() or not STA_GRAMS.is_file(),
                     reason="flagship sta record not present in this checkout")
def test_end_to_end_sta_smallest_verified_n(tmp_path):
    record_copy = tmp_path / STA_RECORD.name
    grams_copy = tmp_path / STA_GRAMS.name
    shutil.copyfile(STA_RECORD, record_copy)
    shutil.copyfile(STA_GRAMS, grams_copy)

    out_dir = tmp_path / "out"
    result = rc.find_smallest_verified_sig(
        record_copy, out_dir, sys.executable, VC_PATH,
        sig_min=1, sig_max=9, dense=False)

    # Low N must genuinely fail (this is not a tool that always "succeeds").
    verdicts = {entry["sig"]: entry["verdict"] for entry in result["scan"]}
    assert verdicts[1] == "FAILED"
    assert verdicts[7] == "FAILED"
    assert result["smallest_verified_sig"] == 8
    assert verdicts[8] == "VERIFIED"

    winner_record = json.loads(Path(result["winner"]["record"]).read_text())
    original = json.loads(STA_RECORD.read_text())
    for field in ("system", "rational_gains", "delta0_exact", "epsilon_exact",
                  "brackets", "corners"):
        assert winner_record[field] == original[field], field
    assert winner_record["V"]["degree_m"] == original["V"]["degree_m"]
    assert winner_record["V"]["weights_r"] == original["V"]["weights_r"]
    for a, b in zip(original["V"]["terms"], winner_record["V"]["terms"]):
        assert a["exponents"] == b["exponents"]
        assert a["term_types"] == b["term_types"]
        assert _is_dyadic(Fraction(b["coefficient"]))
        # acceptance: printed coefficient string-equal to what the tool
        # itself computed for that term.
    term_report = result["winner"]["term_report"]
    for t_report, term in zip(term_report, winner_record["V"]["terms"]):
        assert t_report["rounded"] == term["coefficient"]


# ---------------------------------------------------------------------------
# 3. Refusals.
# ---------------------------------------------------------------------------

def test_refuses_missing_grams_file(tmp_path):
    record = {"system": "toy", "V": {"degree_m": 2, "weights_r": [1],
              "terms": [{"coefficient": "1", "exponents": ["2"], "term_types": ["abs"]}]}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(record))
    with pytest.raises(rc.RoundCertificateError, match="grams_file"):
        rc.find_smallest_verified_sig(path, tmp_path / "out", sys.executable,
                                       VC_PATH, 1, 5, False)


@pytest.mark.skipif(not STA_RECORD.is_file() or not STA_GRAMS.is_file(),
                     reason="flagship sta record not present in this checkout")
def test_refuses_when_no_n_in_range_verifies(tmp_path):
    record_copy = tmp_path / STA_RECORD.name
    grams_copy = tmp_path / STA_GRAMS.name
    shutil.copyfile(STA_RECORD, record_copy)
    shutil.copyfile(STA_GRAMS, grams_copy)
    with pytest.raises(rc.RoundCertificateError, match="no N in"):
        rc.find_smallest_verified_sig(record_copy, tmp_path / "out", sys.executable,
                                       VC_PATH, sig_min=1, sig_max=2, dense=False)


# ---------------------------------------------------------------------------
# 4. 2026-09-25 extension: binary digits, the residual screen, one function
#    over several records, final records with VC's own margin, tier-B helpers.
# ---------------------------------------------------------------------------

X4_RECORD = ROOT / "artifacts" / "campaign_2026-09" / "X4" / "delta2047_4096_rl1" / \
    "m5" / "portable" / "twisting_tb_rl1_m5_delta2047_4096.record.json.gz"
X4_GRAMS = X4_RECORD.with_name("twisting_tb_rl1_m5_delta2047_4096.grams.json.gz")
STA_ROW = ROOT / "artifacts" / "ablation" / "levant_2003_perturbed" / \
    "sta_delta1_sparse_row.json"


@pytest.mark.parametrize("value", [
    "0.04166419414335491", "-0.5293979093949359", "1.7236798638858641", "1",
    "-1", "1/3", "-7/11", "1000000.5", "0.0000001234567", "3", "0.75", "5/8",
])
@pytest.mark.parametrize("bits", [1, 2, 3, 7, 17, 24, 53])
def test_round_to_dyadic_bits_contract(value, bits):
    c = Fraction(value)
    r = rc.round_to_dyadic_bits(c, bits)
    assert _is_dyadic(r)
    assert (r > 0) == (c > 0)
    e = rc.binary_exponent(abs(c))
    assert abs(r - c) <= Fraction(2) ** (e - bits)          # half a grid step
    assert rc.significant_bits(r) <= bits
    assert Fraction(float(r)) == r                          # an exact float64


def test_round_to_dyadic_bits_ties_to_even_and_exact_values():
    # 5/8 = 0.101b: at 2 bits the grid is 1/4, 5/8 is a tie between 1/2 and 3/4
    assert rc.round_to_dyadic_bits(Fraction(5, 8), 2) == Fraction(1, 2)
    assert rc.round_to_dyadic_bits(Fraction(7, 8), 2) == Fraction(1)
    assert rc.round_to_dyadic_bits(Fraction(-7, 8), 2) == Fraction(-1)
    assert rc.round_to_dyadic_bits(Fraction(3, 4), 2) == Fraction(3, 4)
    assert rc.round_to_dyadic_bits(Fraction(0), 5) == 0
    with pytest.raises(rc.RoundCertificateError):
        rc.round_to_dyadic_bits(Fraction(1), 0)


@pytest.mark.parametrize("x", ["1", "2", "3", "1/2", "3/4", "1/3", "1023/1024", "1025"])
def test_binary_exponent(x):
    x = Fraction(x)
    e = rc.binary_exponent(x)
    assert Fraction(2) ** e <= x < Fraction(2) ** (e + 1)


def test_parse_coefficient_forms():
    assert rc.parse_coefficient("0x1.8000000000000p-1") == Fraction(3, 4)
    assert rc.parse_coefficient("-0x1.0000000000000p+0") == -1
    assert rc.parse_coefficient(0.75) == Fraction(3, 4)
    assert rc.parse_coefficient("3/4") == Fraction(3, 4)
    assert rc.parse_coefficient(1) == 1


def test_round_record_v_bits_unit():
    record = {"V": {"degree_m": 4, "weights_r": [1], "terms": [
        {"coefficient": "0x1.5555555555555p-2", "exponents": ["4"], "term_types": ["abs"]}]}}
    new, report = rc.round_record_v(record, 4, unit="bits")
    assert new["V"]["terms"][0]["coefficient"] == "11/32"   # 0.34375 = 1011b * 2^-5
    assert report[0]["rounded"] == "11/32"


def test_printed_record_moves_margin_declarations_to_provenance():
    record = {"system": "toy", "eps_certified": 1e-6, "eps_certified_exact": "1/1000000",
              "worst_charge_over_eps": {"overall": 0.1}, "provenance": {"note": "x"},
              "V": {"degree_m": 2, "weights_r": [1], "terms": [
                  {"coefficient": 1.0, "exponents": ["2"], "term_types": ["abs"]}]}}
    out, _ = rc.printed_record(record, 8, "bits", {"record": "r", "record_sha256": "a",
                                                   "grams_sha256": "b"})
    for k in rc.MARGIN_DECLARATION_FIELDS:
        assert k not in out
    moved = out["provenance"]["printed_coefficients"]["source_margin_declarations"]
    assert moved["eps_certified_exact"] == "1/1000000"
    assert out["provenance"]["note"] == "x"


def test_v_signature_is_exact():
    a = {"V": {"degree_m": 2, "weights_r": [1], "terms": [
        {"coefficient": "0x1.8000000000000p-1", "exponents": ["2"], "term_types": ["abs"]}]}}
    b = json.loads(json.dumps(a))
    b["V"]["terms"][0]["coefficient"] = "3/4"
    c = json.loads(json.dumps(a))
    c["V"]["terms"][0]["coefficient"] = 0.7500000000000001
    assert rc.v_signature(a) == rc.v_signature(b)
    assert rc.v_signature(a) != rc.v_signature(c)


def test_round_row_and_canonicalize_roundtrip():
    row = {"system": "toy", "degree_used": 2, "weights_r": [1], "V_terms": [
        {"coefficient": 1.0, "exponents": ["2"], "term_types": ["abs"]},
        {"coefficient": -0.3333333333333333, "exponents": ["2"], "term_types": ["abs"]}]}
    out, strings = rc.round_row(row, 5, "bits")
    assert strings == ["1/1", "-21/64"]
    assert [t["coefficient"] for t in out["V_terms"]] == [1.0, -21 / 64]
    emitted = {"V": {"terms": [{"coefficient": float(1).hex()},
                               {"coefficient": (-21 / 64).hex()}]}}
    printed = rc.canonicalize_record(emitted, strings)
    assert [t["coefficient"] for t in printed["V"]["terms"]] == strings
    emitted["V"]["terms"][1]["coefficient"] = (-22 / 64).hex()
    with pytest.raises(rc.RoundCertificateError, match="!= printed"):
        rc.canonicalize_record(emitted, strings)


def test_round_row_refuses_non_float_value():
    row = {"V_terms": [{"coefficient": "1/3", "exponents": ["2"], "term_types": ["abs"]}]}
    with pytest.raises(rc.RoundCertificateError, match="float64"):
        rc.round_row(row, 60, "bits")


@pytest.mark.skipif(not STA_RECORD.is_file() or not STA_GRAMS.is_file(),
                    reason="flagship sta record not present in this checkout")
def test_screen_matches_vc_and_proves_failures(tmp_path):
    """The residual screen equals VC's worst charge/eps on a VERIFIED record
    (the stored Grams are PSD, eta = 0), and a screen ratio >= 1 coincides
    with a VC failure."""
    items = rc.load_function([STA_RECORD], tmp_path)
    for n, expect_fail in ((20, True), (24, False)):
        paths, _ = rc.write_candidates(items, tmp_path, n, "bits")
        screen = rc.predict_residual(paths[0], items[0]["grams"], VC_PATH)
        ok, summary, _ = rc.run_vc(sys.executable, VC_PATH, paths[0], items[0]["grams"],
                                   tmp_path / f"v{n}.json", dense=False)
        assert screen["proves_fail"] is expect_fail
        assert ok is (not expect_fail)
        if ok:
            assert float(screen["overall"]) == summary["worst_charge_over_eps"]["overall"]


@pytest.mark.skipif(not STA_RECORD.is_file() or not STA_GRAMS.is_file(),
                    reason="flagship sta record not present in this checkout")
@pytest.mark.parametrize("strategy", ["scan", "predict"])
def test_tier_a_function_sta_bits(tmp_path, strategy):
    res = rc.tier_a_function("sta", [STA_RECORD], tmp_path, sys.executable, VC_PATH,
                             20, 30, "bits", strategy, False, True, None,
                             screen_all=(strategy == "scan"), log=lambda s: None)
    assert res["smallest_verified_n"] == 24
    assert res["final"]["all_verified"] is True
    assert res["acceptance"]["ok"] is True
    assert res["acceptance"]["max_significant_bits"] <= 24
    if strategy == "predict":
        assert res["screen_start_n"] == 24
        assert all(e["proves_fail"] for e in res["screen"] if e["n"] < 24)
    # the source had no margin declaration, so the final record has none
    final = json.loads(Path(res["final"]["records"][0]["record"]).read_text())
    assert "eps_certified_exact" not in final
    # the Gram sidecar is the source's, byte for byte
    assert (tmp_path / STA_GRAMS.name).read_bytes() == STA_GRAMS.read_bytes()


@pytest.mark.skipif(not X4_RECORD.is_file() or not X4_GRAMS.is_file(),
                    reason="X4 portable record not present in this checkout")
@pytest.mark.parametrize("single_p", [False, True])
def test_tier_a_function_declares_vc_margin(tmp_path, single_p):
    """A source with margin declarations: the candidate carries none (VC would
    refuse the source's larger claim), the final record declares VC's own
    margin, and the independent final VC run accepts it."""
    res = rc.tier_a_function("x4", [X4_RECORD], tmp_path, sys.executable, VC_PATH,
                             1, 30, "bits", "predict", False, True, None,
                             single_p=single_p, log=lambda s: None)
    assert res["smallest_verified_n"] == 17
    fin = res["final"]["records"][0]
    assert fin["verdict"] == "VERIFIED"
    final = json.loads(Path(fin["record"]).read_text())
    assert final["eps_certified_exact"] == fin["eps_certified_exact"]
    assert final["eps_certified_exact_is_float_transcription"] is False
    src = rc.read_json_maybe_gz(X4_RECORD)
    assert Fraction(final["eps_certified_exact"]) < Fraction(src["eps_certified_exact"])


@pytest.mark.skipif(not STA_RECORD.is_file() or not X4_RECORD.is_file(),
                    reason="fixtures not present in this checkout")
def test_function_refuses_records_with_different_v(tmp_path):
    with pytest.raises(rc.RoundCertificateError, match="one function means one V"):
        rc.load_function([X4_RECORD, STA_RECORD], tmp_path)


@pytest.mark.skipif(not STA_RECORD.is_file() or not STA_GRAMS.is_file(),
                    reason="flagship sta record not present in this checkout")
def test_source_declared_margin_is_refused_on_a_rounded_v(tmp_path):
    """Why the declarations must go: keeping the source's claimed margin on a
    rounded V makes VC refuse (margin claim exceeds proof)."""
    items = rc.load_function([X4_RECORD], tmp_path) if X4_RECORD.is_file() else None
    if items is None:
        pytest.skip("X4 record absent")
    rec, _ = rc.round_record_v(items[0]["record"], 17, "bits")
    rec["grams_file"] = items[0]["grams"].name
    path = tmp_path / "kept_decl.record.json"
    path.write_text(json.dumps(rec))
    ok, _, log = rc.run_vc(sys.executable, VC_PATH, path, items[0]["grams"],
                           tmp_path / "kept.json", dense=False)
    assert not ok
    assert "MARGIN CLAIM EXCEEDS PROOF" in log


EXAMPLE1_RECORD = ROOT / "artifacts" / "campaign_2026-08" / "E3_portable" / \
    "nominal_records" / "example1_nominal.record.json.gz"


@pytest.mark.parametrize("record", [STA_RECORD, X4_RECORD, EXAMPLE1_RECORD],
                         ids=["sta_perturbed", "twisting_portable", "example1_nominal"])
def test_batch_screen_equals_per_n_screen(tmp_path, record):
    """The one-pass screen (linearity in V's coefficients) returns exactly the
    per-n screen's rationals, for perturbed, portable and nominal records."""
    if not record.is_file():
        pytest.skip("fixture not present in this checkout")
    items = rc.load_function([record], tmp_path)
    ns = [2, 9, 17, 24, 40]
    batch = rc.predict_residual_all_n(record, items[0]["grams"], VC_PATH, ns, "bits")
    for n in ns:
        paths, _ = rc.write_candidates(items, tmp_path, n, "bits")
        one = rc.predict_residual(paths[0], items[0]["grams"], VC_PATH)
        for k in ("V", "W", "overall"):
            assert batch["per_n"][n][k] == one[k], (n, k)


@pytest.mark.skipif(not X4_RECORD.is_file(), reason="X4 record absent")
def test_tier_a_batch_screen_same_answer(tmp_path):
    res = rc.tier_a_function("x4", [X4_RECORD], tmp_path, sys.executable, VC_PATH,
                             1, 30, "bits", "predict", False, True, None,
                             single_p=True, batch_screen=True, log=lambda s: None)
    assert res["smallest_verified_n"] == 17
    assert res["final"]["all_verified"] is True
