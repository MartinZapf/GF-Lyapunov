"""EM's ``--gain-point exact-record`` mode (WP TB emitter change, option a).

Before this mode, ``scripts/emit_certificate_record.py`` (EM) bound every
record's gains to the printed-gain registry, so the TB driver
``scripts/certify_gain_point.py`` stopped at exit 5 for any gain point that
is not registered: the X2 gain box and the X3 scaled continuous-twisting
point. The mode reads the gain point from the driver's exact-gains record and
re-checks it in Q against what the portable verifier (VC) checks; VC stays
unchanged and is the authority.

CLARABEL only, and every design here is labelled NOT D: nothing in this file
certifies anything for the paper. What it pins:

  1. end to end through the driver, on two non-registry points: a perturbed
     STA tuning and a perturbed twisting gain box (4 corners x 2 endpoints),
     both reaching VC VERIFIED under the stdlib interpreter when present;
  2. the X2 shape on that box: two corner-subset records, each with its own
     named bracket target, verified by VC, carrying one identical V;
  3. the gain-point class against the driver's corner order;
  4. refusals: anything VC could not check (names, fixed parameters,
     brackets and box edges in Q, positivity, non-affine varying gains, no
     disturbance channel, a subset or target VC would read differently), a
     non-TB source, float images, tampered sources, and usage errors;
  5. VC as the authority: a box and bundle edited consistently pass every
     emitter pin and are rejected by VC, and the emitter deletes the files.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "verification" / "standalone"))

import emit_certificate_record as em  # noqa: E402
from verify_certificate import SYSTEMS as VC_SYSTEMS  # noqa: E402

EMITTER = ROOT / "scripts" / "emit_certificate_record.py"
VERIFIER = ROOT / "verification" / "standalone" / "verify_certificate.py"
DRIVER = ROOT / "scripts" / "certify_gain_point.py"

#: The stdlib interpreter P uses on this machine class, when it exists; the
#: current interpreter otherwise (VC imports nothing outside the stdlib).
_STDLIB_CANDIDATES = [
    Path(os.environ.get("GF_STDLIB_PYTHON", "")),
    Path("/usr/bin/python3"),
    Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python311"
    / "python.exe",
]
P_PYTHON = next((str(p) for p in _STDLIB_CANDIDATES
                 if str(p) not in ("", ".") and p.is_file()), sys.executable)


def _load_driver():
    spec = importlib.util.spec_from_file_location("certify_gain_point", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_driver()


def _read_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def _env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _em(*argv):
    return subprocess.run([sys.executable, str(EMITTER), *map(str, argv)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT), env=_env())


def _vc(record):
    return subprocess.run([P_PYTHON, str(VERIFIER), str(record)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT), env=_env())


def _drive(out, *argv):
    code = tb.main(list(argv) + ["--design-solver", "CLARABEL",
                                 "--p-python", P_PYTHON,
                                 "--out-dir", str(out)])
    return code, json.loads((out / "run.json").read_text())


# ---------------------------------------------------------------------------
# 1. End to end through the driver: non-registry points reach VC VERIFIED
# ---------------------------------------------------------------------------

BOX_ARGS = ("--system", "twisting", "--box", "r1:2:21/10",
            "--box", "r2:3/2:8/5", "--delta0", "1/4", "--degrees", "5")


@pytest.fixture(scope="module")
def box_run(tmp_path_factory):
    """One CLARABEL run on a perturbed twisting box: 4 corners x 2 endpoints."""
    out = tmp_path_factory.mktemp("twisting_box")
    code, run = _drive(out, *BOX_ARGS)
    assert code == tb.EXIT_OK, run
    return out, run


def test_twisting_box_reaches_vc_verified(box_run):
    out, run = box_run
    assert run["registry_match"]["table"] is None
    assert run["emission"]["gain_point"] == "exact-record"
    assert run["emission"]["self_verify"]["verdict"] == "VERIFIED"
    assert run["P"]["verdict"] == "VERIFIED" and run["P"]["returncode"] == 0
    assert Fraction(run["P"]["eps_certified_exact"]) > 0
    rec = _read_gz(out / "m5" / "portable" / run["emission"]["record"])
    # a box edge is a VC bracket with power_k = 1 and the midpoint target
    assert rec["brackets"] == {
        "r1": {"lo": "2", "hi": "21/10", "power_k": 1, "power_target": "41/20"},
        "r2": {"lo": "3/2", "hi": "8/5", "power_k": 1, "power_target": "31/20"}}
    assert rec["rational_gains"] == {}
    assert [c["assignment"] for c in rec["corners"]] == [
        {"r1": "2", "r2": "3/2"}, {"r1": "2", "r2": "8/5"},
        {"r1": "21/10", "r2": "3/2"}, {"r1": "21/10", "r2": "8/5"}]
    assert rec["delta0_exact"] == "1/4" and rec["coverage"]["inclusion_endpoints"] == 2
    # provenance: the exact record, never the registry
    src = rec["provenance"]["gain_point_source"]
    assert src["kind"] == "exact_record"
    assert "NOT from the printed-gain registry" in src["note"]
    assert src["exact_record_sha256"] == hashlib.sha256(
        (out / "m5" / "exact.json").read_bytes()).hexdigest()
    assert src["design_stage"] == {"label": "clarabel_design_not_D",
                                   "citable_chain": False}
    assert rec["gain_point"]["source"] == "exact_record"
    assert rec["provenance"]["emitter_options"]["gain_point"] == "exact-record"
    assert not any(s["role"] == "gain_point_registry"
                   for s in rec["source_artifacts"])
    # the record carries no key VC does not check as part of the statement:
    # the box lives only in metadata (gain_point, provenance)
    assert "box" not in rec


def test_sta_non_registry_tuning_reaches_vc_verified(tmp_path):
    code, run = _drive(tmp_path, "--system", "sta", "--gains", "k1=2,k2=3/2",
                       "--delta0", "1/2", "--degrees", "3")
    assert code == tb.EXIT_OK, run
    assert run["registry_match"]["table"] is None
    assert run["emission"]["gain_point"] == "exact-record"
    assert run["P"]["verdict"] == "VERIFIED"
    rec = _read_gz(tmp_path / "m3" / "portable" / run["emission"]["record"])
    assert rec["rational_gains"] == {"k1": "2", "k2": "3/2"}
    assert rec["brackets"] == {} and len(rec["corners"]) == 1
    assert rec["kappa"] == "-1"
    assert "NOT on the D -> X chain" in rec["provenance"]["source_claim"]


# ---------------------------------------------------------------------------
# 2. The X2 shape: two corner-subset records, one V
# ---------------------------------------------------------------------------

def test_corner_subset_records_share_one_v_and_verify(box_run, tmp_path):
    out, _ = box_run
    m5 = out / "m5"
    records = []
    for value, target in (("2", "12/5"), ("21/10", "5/2")):
        proc = _em("--row", m5 / "row.json", "--exact", m5 / "exact.json",
                   "--gain-point", "exact-record", "--corner-subset",
                   f"r1={value}", "--bracket-target", f"r2:2:{target}",
                   "--stem-tag", "x2shape", "--out-dir", tmp_path)
        assert proc.returncode == 0, proc.stderr + proc.stdout
        path = Path(json.loads(proc.stdout)["record"])
        assert "sub_r1_" in path.name
        rec = _read_gz(path)
        assert rec["rational_gains"] == {"r1": value}
        assert rec["brackets"] == {"r2": {"lo": "3/2", "hi": "8/5",
                                          "power_k": 2,
                                          "power_target": target}}
        assert [c["assignment"] for c in rec["corners"]] == [
            {"r2": "3/2"}, {"r2": "8/5"}]
        grams = _read_gz(tmp_path / rec["grams_file"])
        assert len(grams["corners"]) == 2
        assert all(len(c["endpoints"]) == 2 for c in grams["corners"])
        src = rec["provenance"]["gain_point_source"]
        assert src["corner_subset"] == {"r1": value}
        assert src["source_n_corners"] == 4
        assert "not only this subset" in src["exact_stage_scope"]
        vc = _vc(path)
        assert vc.returncode == 0, vc.stdout + vc.stderr
        assert "certificate VERIFIED" in vc.stdout
        assert f"lo^2 < {target} < hi^2 verified in Q" in vc.stdout
        records.append(rec)
    digests = {hashlib.sha256(json.dumps(r["V"], sort_keys=True).encode())
               .hexdigest() for r in records}
    assert len(digests) == 1
    assert records[0]["provenance"]["gain_point_source"]["exact_record_sha256"] \
        == records[1]["provenance"]["gain_point_source"]["exact_record_sha256"]


# ---------------------------------------------------------------------------
# 3. The gain-point class (no solver)
# ---------------------------------------------------------------------------

def _doc(system="twisting", rational=None, brackets=None, box=None,
         delta0="1/4", **extra):
    doc = {"system": system,
           "gain_source": em.EXACT_RECORD_GAIN_SOURCE,
           "gain_binding": "printed_rationals",
           "rational_gains": rational or {},
           "brackets": brackets or {},
           "box": box or {},
           "delta0_exact": delta0}
    doc.update(extra)
    return doc


def _point(doc, subset=None, targets=None):
    return em.ExactRecordGainPoint(doc, doc["system"], VC_SYSTEMS[doc["system"]],
                                   corner_subset=subset, bracket_targets=targets)


def test_corner_order_is_the_drivers():
    spec = tb.build_gain_spec("levant3d", ["lambda3=11/10"],
                              boxes=["lambda1:2:3",
                                     "lambda2:2121/1000:2599/1000"],
                              delta0="1")
    doc = _doc("levant3d", rational={"lambda3": "11/10"},
               box={"lambda1": {"lo": "2", "hi": "3"},
                    "lambda2": {"lo": "2121/1000", "hi": "2599/1000"}},
               delta0="1")
    point = _point(doc)
    assert point.corner_assignments() == spec.corners()
    assert point.bracket_power("lambda1") == (1, Fraction(5, 2))
    # the X2 per-tuning point A: lambda1 = 2, lambda2 bracketed by 9/2
    a = _point(doc, {"lambda1": Fraction(2)}, {"lambda2": (2, Fraction(9, 2))})
    assert a.rational_gains() == {"lambda3": Fraction(11, 10),
                                  "lambda1": Fraction(2)}
    assert a.bracket_power("lambda2") == (2, Fraction(9, 2))
    assert a.corner_assignments() == [{"lambda2": Fraction(2121, 1000)},
                                      {"lambda2": Fraction(2599, 1000)}]
    b = _point(doc, {"lambda1": Fraction(3)}, {"lambda2": (2, Fraction(27, 4))})
    assert b.bracket_power("lambda2") == (2, Fraction(27, 4))


def test_scaled_cta_point_is_a_single_rational_corner():
    s = Fraction(17, 20)
    gains = {"k1": s ** 4 * Fraction(274, 5), "k2": s ** 3 * 27,
             "k3": s ** 6 * Fraction(57, 10), "k4": s ** 6 * 3}
    doc = _doc("continuous_twisting",
               rational={n: str(v) for n, v in gains.items()},
               delta0=str(s ** 6))
    point = _point(doc)
    assert point.rational_gains() == gains
    assert point.corner_assignments() == [{}]
    assert point.delta0 == s ** 6
    assert point.kappa({"weights_r": [3, 2, 1]}) == "-1"


def test_pidsmc_fixed_delta_is_accepted_at_one():
    doc = _doc("pidsmc", rational={"lambda1": "27/10", "lambda2": "1069/200",
                                   "lambda3": "11/10", "delta": "1"},
               delta0="51/64")
    assert _point(doc).rational_gains()["delta"] == 1


def test_select_bundle_keeps_the_subset_and_strips_the_fixed_name():
    doc = _doc(box={"r1": {"lo": "2", "hi": "21/10"},
                    "r2": {"lo": "3/2", "hi": "8/5"}})
    point = _point(doc, {"r1": Fraction(21, 10)})
    corners = [{"corner_label": str(i), "assignment": {"r1": a, "r2": b},
                "endpoints": [[], []]}
               for i, (a, b) in enumerate([("2", "3/2"), ("2", "8/5"),
                                           ("21/10", "3/2"), ("21/10", "8/5")])]
    kept = point.select_bundle({"V_grams": [], "corners": corners})
    assert [c["assignment"] for c in kept["corners"]] == [{"r2": "3/2"},
                                                          {"r2": "8/5"}]
    with pytest.raises(em.Refusal) as exc:
        point.select_bundle({"V_grams": [], "corners": corners[:3]})
    assert exc.value.reason == "incomplete_corner_box"


# ---------------------------------------------------------------------------
# 4. Refusals: anything VC could not check
# ---------------------------------------------------------------------------

TWIST_BOX = {"r1": {"lo": "2", "hi": "21/10"}, "r2": {"lo": "3/2", "hi": "8/5"}}


@pytest.mark.parametrize("doc, subset, targets, reason, needle", [
    (_doc(rational={"r1": "2", "r2": "3/2"}, gain_source="exact_gate_standard_gains"),
     None, None, "unverifiable_gain_point", "gain_source"),
    (_doc(rational={"r1": "2", "r2": "3/2"}, gain_binding="float64"),
     None, None, "float_image_gains", "gain_binding"),
    (_doc(rational={"r1": 2.0, "r2": "3/2"}),
     None, None, "unverifiable_gain_point", "not an exact rational"),
    (_doc(rational={"r1": "2"}),
     None, None, "unverifiable_gain_point", "are not in the exact record"),
    (_doc(rational={"r1": "2", "r2": "3/2", "r3": "1"}),
     None, None, "unverifiable_gain_point", "not parameters"),
    (_doc(rational={"r1": "2", "r2": "3/2"}, box={"r1": {"lo": "2", "hi": "3"}}),
     None, None, "unverifiable_gain_point", "both"),
    (_doc(rational={"r2": "3/2"}, box={"r1": {"lo": "3", "hi": "2"}}),
     None, None, "unverifiable_gain_point", "0 < lo < hi"),
    (_doc(rational={"r1": "-2", "r2": "3/2"}),
     None, None, "unverifiable_gain_point", "positive"),
    (_doc("levant3d", rational={"lambda1": "2", "lambda3": "11/10"},
          brackets={"lambda2": {"lo": "2122/1000", "hi": "2123/1000",
                                "power_k": 2, "power_target": "9/2"}}),
     None, None, "unverifiable_gain_point", "does not bracket"),
    (_doc("levant3d", rational={"lambda1": "2", "lambda3": "11/10"},
          brackets={"lambda2": {"lo": "2", "hi": "3", "power_k": 0,
                                "power_target": "5/2"}}),
     None, None, "unverifiable_gain_point", "power_k"),
    (_doc(box=TWIST_BOX), {"r1": Fraction(3)}, None,
     "unverifiable_gain_point", "not an edge"),
    (_doc(rational={"r1": "2"}, box={"r2": TWIST_BOX["r2"]}),
     {"r1": Fraction(2)}, None, "unverifiable_gain_point", "not a box gain"),
    (_doc("levant3d", rational={"lambda1": "2", "lambda3": "11/10"},
          brackets={"lambda2": {"lo": "2121/1000", "hi": "2122/1000",
                                "power_k": 2, "power_target": "9/2"}}),
     {"lambda2": Fraction(2121, 1000)}, None, "unverifiable_gain_point",
     "irrational gain"),
    (_doc(box=TWIST_BOX), None, {"r2": (2, Fraction(3))},
     "unverifiable_gain_point", "does not bracket"),
    (_doc(box=TWIST_BOX), {"r2": Fraction(3, 2)}, {"r2": (2, Fraction(12, 5))},
     "unverifiable_gain_point", "fixed by --corner-subset"),
    (_doc(rational={"r1": "2", "r2": "3/2"}), None, {"r1": (1, Fraction(2))},
     "unverifiable_gain_point", "only a box gain"),
    (_doc("pidsmc", rational={"lambda1": "27/10", "lambda2": "1069/200",
                              "lambda3": "11/10", "delta": "2"}),
     None, None, "unverifiable_gain_point", "describe delta = 1"),
    (_doc("pidsmc", rational={"lambda1": "27/10", "lambda2": "1069/200",
                              "lambda3": "11/10"},
          box={"delta": {"lo": "1", "hi": "2"}}),
     None, None, "unverifiable_gain_point", "fixed factory parameter"),
    (_doc("power_rate", rational={"k": "1"},
          box={"alpha": {"lo": "1/2", "hi": "3/4"}}, delta0="0"),
     None, None, "unverifiable_gain_point", "affine"),
    (_doc("example1", rational={"k1": "1", "k2": "1"}, delta0="1/2"),
     None, None, "unverifiable_gain_point", "disturbance channel"),
])
def test_refusals_of_what_vc_could_not_check(doc, subset, targets, reason,
                                             needle):
    with pytest.raises(em.Refusal) as exc:
        _point(doc, subset, targets)
    assert exc.value.reason == reason
    assert needle in exc.value.detail


def test_system_of_the_exact_record_must_be_the_rows():
    doc = _doc(rational={"r1": "2", "r2": "3/2"})
    with pytest.raises(em.Refusal) as exc:
        em.ExactRecordGainPoint(doc, "sta", VC_SYSTEMS["sta"])
    assert exc.value.reason == "unverifiable_gain_point"


def _tampered(box_run, tmp_path, edit_exact=None, edit_grams=None,
              edit_row=None):
    """A copy of the box run's m5 sources with one edit applied."""
    out, _ = box_run
    src = tmp_path / "src"
    shutil.copytree(out / "m5", src, ignore=shutil.ignore_patterns("portable"))
    for name, edit in (("exact.json", edit_exact), ("row.json", edit_row)):
        if edit:
            doc = json.loads((src / name).read_text())
            edit(doc)
            (src / name).write_text(json.dumps(doc))
    if edit_grams:
        doc = _read_gz(src / "exact_grams.json.gz")
        edit_grams(doc)
        (src / "exact_grams.json.gz").write_bytes(
            gzip.compress(json.dumps(doc).encode()))
    return src


def _emit_from(src, out, *extra):
    return _em("--row", src / "row.json", "--exact", src / "exact.json",
               "--gain-point", "exact-record", "--out-dir", out, *extra)


def _refusal(proc):
    assert proc.returncode == 3, proc.stdout + proc.stderr
    return json.loads(proc.stdout[proc.stdout.index("{"):])["reason"]


def test_cli_refuses_a_non_tb_exact_record(box_run, tmp_path):
    src = _tampered(box_run, tmp_path,
                    edit_exact=lambda d: d.pop("gain_source"))
    out = tmp_path / "out"
    assert _refusal(_emit_from(src, out)) == "unverifiable_gain_point"
    assert not list(out.glob("*.record.json.gz"))
    assert (out / "twisting.refusal.json").exists()


def test_cli_refuses_a_bundle_corner_of_another_box(box_run, tmp_path):
    def edit(doc):
        doc["corners"][3]["assignment"]["r1"] = "22/10"
    src = _tampered(box_run, tmp_path, edit_grams=edit)
    assert _refusal(_emit_from(src, tmp_path / "out")) == "gain_mismatch"


def test_cli_refuses_a_missing_corner(box_run, tmp_path):
    src = _tampered(box_run, tmp_path,
                    edit_grams=lambda d: d["corners"].pop())
    assert _refusal(_emit_from(src, tmp_path / "out")) == "incomplete_corner_box"


def test_cli_refuses_a_row_at_another_delta0(box_run, tmp_path):
    def edit(doc):
        doc["delta0"] = "1/2"
    src = _tampered(box_run, tmp_path, edit_row=edit)
    assert _refusal(_emit_from(src, tmp_path / "out")) == "source_pin_mismatch"


def test_cli_refuses_a_row_whose_v_is_not_the_proved_v(box_run, tmp_path):
    def edit(doc):
        doc["V_terms"][0]["coefficient"] *= 2
    src = _tampered(box_run, tmp_path, edit_row=edit)
    assert _refusal(_emit_from(src, tmp_path / "out")) == "source_pin_mismatch"


def test_cli_refuses_a_subset_that_leaves_an_invalid_bracket(box_run, tmp_path):
    out, _ = box_run
    proc = _em("--row", out / "m5" / "row.json", "--exact",
               out / "m5" / "exact.json", "--gain-point", "exact-record",
               "--corner-subset", "r1=2", "--bracket-target", "r2:2:3",
               "--out-dir", tmp_path)
    assert _refusal(proc) == "unverifiable_gain_point"


@pytest.mark.parametrize("argv, needle", [
    (["--row", "r.json", "--exact", "e.json", "--corner-subset", "r1=2"],
     "need --gain-point exact-record"),
    (["--row", "r.json", "--exact", "e.json", "--gain-point", "flagship",
      "--bracket-target", "r2:2:3"], "need --gain-point exact-record"),
    (["--exact-cell", "c", "--design-record", "d", "--design-bundle-npz", "b",
      "--gain-point", "exact-record"], "does not take --exact-cell"),
    (["--row", "r.json", "--exact", "e.json", "--gain-point", "exact-record",
      "--representative-p", "1/2"], "--representative-p"),
])
def test_usage_errors(tmp_path, argv, needle):
    proc = _em(*argv, "--out-dir", tmp_path)
    assert proc.returncode == 2
    assert needle in proc.stderr


# ---------------------------------------------------------------------------
# 5. VC is the authority
# ---------------------------------------------------------------------------

def test_consistent_tamper_passes_the_pins_and_vc_rejects_it(box_run, tmp_path):
    """Widen the r1 box to 22/10 in the exact record AND the bundle: every
    emitter check passes (the point is valid in Q, affine, complete), the
    captures no longer prove the targets at the new corner, VC rejects, and
    the emitter leaves no record behind."""
    def exact(doc):
        doc["box"]["r1"]["hi"] = "11/5"

    def grams(doc):
        for corner in doc["corners"]:
            if corner["assignment"]["r1"] == "21/10":
                corner["assignment"]["r1"] = "11/5"
    src = _tampered(box_run, tmp_path, edit_exact=exact, edit_grams=grams)
    out = tmp_path / "out"
    assert _refusal(_emit_from(src, out)) == "self_verification_failed"
    assert not list(out.glob("*.record.json.gz"))
    assert not list(out.glob("*.grams.json.gz"))
