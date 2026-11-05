"""WP X10: the third-order differentiator at the higher-gain standard tuning.

``levant4d_2003print`` is Levant's third-order differentiator (four states,
weights r = (4, 3, 2, 1)) at the recursive parameter set
(lambdahat_0, ..., lambdahat_3) = (1.1, 1.5, 3, 5), L = 1, i.e. the
non-recursive gains (5, 3*5^(2/3), (3/2)*lambda2^(1/2), 11/10). It has the
SAME equations as ``levant4d``; only the gains differ. It is registered in
``FLAGSHIP_GAIN_POINTS`` and in the portable verifier's ``SYSTEMS`` (the key
names the gain point a record was emitted against, following
``levant3d_2003print``).

What this file pins:

  1. the non-recursive conversion and both bracket power targets
     (lambda2^3 = 675, lambda3^6 = 492075/64), derived here in exact
     arithmetic from the recursive set, against the registry entry;
  2. the registry brackets are valid in Q, one decimal ulp wide, and contain
     the float seeds; the point differs from the levant4d point;
  3. the verifier's hand-written equations for the new key equal the
     framework's exact dynamics at all four bracket corners, nominal and at
     both inclusion endpoints, and its ``SYSTEMS`` entry is levant4d's;
  4. the scaled bracket at a representative p (lambda_i -> p^i lambda_i) stays
     valid in Q, and the X10 exact cells state exactly that scaled bracket;
  5. the X10 portable records, when present: P verdict, exact margin and the
     proof-input binding of the committed verification summaries.
"""
import gzip
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "standalone"))

from verify_certificate import (  # noqa: E402
    SYSTEMS as STANDALONE_SYSTEMS,
    build_system_dynamics,
    exact_collect,
)
from gf_framework.discovery.universal_discovery import (  # noqa: E402
    matched_disturbance_vertices,
)
from gf_framework.systems.printed_gains import (  # noqa: E402
    BENCHMARK_GAIN_POINTS,
    FLAGSHIP_GAIN_POINTS,
    brackets,
    corner_assignments,
    exact_dynamics,
    exact_gain_values,
    gain_point,
)
from gf_framework.verification.exact_gate import exact_terms_from_gf  # noqa: E402

KEY = "levant4d_2003print"
#: The recursive set, lambdahat_0..lambdahat_3, as printed (L = 1).
RECURSIVE = (Fraction(11, 10), Fraction(3, 2), Fraction(3), Fraction(5))
GAIN_INDEX = {"lambda1": 1, "lambda2": 2, "lambda3": 3, "lambda4": 4}

X10 = ROOT / "artifacts" / "campaign_2026-09" / "X10"


def _normalize(channels):
    return [sorted(exact_collect(list(terms)), key=lambda t: (t[1], t[2]))
            for terms in channels]


def _assert_same_dynamics(ours, theirs, label):
    assert len(ours) == len(theirs), f"{label}: channel count"
    for channel, (a, b) in enumerate(zip(_normalize(ours), _normalize(theirs))):
        assert a == b, f"{label}: channel {channel}\n  ours   {a}\n  theirs {b}"


# ---------------------------------------------------------------------------
# 1. The conversion, in exact arithmetic
# ---------------------------------------------------------------------------

def test_power_targets_follow_from_the_recursive_set_in_Q():
    """Substituting each recursive stage into the next gives, exactly,
    lambda1 = lh3, lambda2 = lh2 lambda1^(2/3), lambda3 = lh1 lambda2^(1/2),
    lambda4 = lh0. Raised to integer powers, everything is rational:
    lambda2^3 = lh2^3 lambda1^2 and lambda3^6 = lh1^6 lambda2^3."""
    lh0, lh1, lh2, lh3 = RECURSIVE
    lambda1 = lh3
    lambda2_cubed = lh2 ** 3 * lambda1 ** 2
    lambda3_sixth = lh1 ** 6 * lambda2_cubed
    assert lambda2_cubed == Fraction(675)
    assert lambda3_sixth == Fraction(492075, 64)

    point = FLAGSHIP_GAIN_POINTS[KEY]
    assert point.system == KEY
    assert point.factory == "create_levant4d_spec"
    assert point.channel == 3
    assert set(point.gains) == set(GAIN_INDEX)
    assert point.gains["lambda1"].value == lambda1
    assert point.gains["lambda4"].value == lh0
    g2, g3 = point.gains["lambda2"], point.gains["lambda3"]
    assert (g2.power_k, g2.power_target) == (3, lambda2_cubed)
    assert (g3.power_k, g3.power_target) == (6, lambda3_sixth)
    # the float seeds are the conversion formulas in float64
    assert abs(g2.approx ** 3 - 675) < 1e-9
    assert abs(g3.approx - 1.5 * g2.approx ** 0.5) < 1e-14


def test_registered_in_the_flagship_table_only():
    assert KEY in FLAGSHIP_GAIN_POINTS
    assert KEY not in BENCHMARK_GAIN_POINTS
    assert gain_point(KEY, flagship=True) is FLAGSHIP_GAIN_POINTS[KEY]


# ---------------------------------------------------------------------------
# 2. Brackets
# ---------------------------------------------------------------------------

def test_brackets_are_valid_in_Q_and_contain_the_float_seed():
    point = FLAGSHIP_GAIN_POINTS[KEY]
    bs = brackets(point)
    assert sorted(bs) == ["lambda2", "lambda3"]
    for name, (lo, hi) in bs.items():
        g = point.gains[name]
        assert 0 < lo < hi
        assert lo ** g.power_k < g.power_target < hi ** g.power_k
        assert hi - lo == Fraction(1, 10 ** 15)
        assert float(lo) <= g.approx <= float(hi)
    assert bs["lambda2"] == (Fraction(4386026607319299, 500000000000000),
                             Fraction(8772053214638599, 1000000000000000))
    assert bs["lambda3"] == (Fraction(1110661957261773, 250000000000000),
                             Fraction(4442647829047093, 1000000000000000))


def test_it_is_a_different_gain_point_from_levant4d():
    new = FLAGSHIP_GAIN_POINTS[KEY]
    old = FLAGSHIP_GAIN_POINTS["levant4d"]
    assert new.gains["lambda1"].value == Fraction(5)
    assert old.gains["lambda1"].value == Fraction(3)
    assert new.gains["lambda2"].power_target != old.gains["lambda2"].power_target
    assert new.gains["lambda3"].power_target != old.gains["lambda3"].power_target
    assert new.gains["lambda4"].value == old.gains["lambda4"].value


# ---------------------------------------------------------------------------
# 3. The verifier's equations for the new key
# ---------------------------------------------------------------------------

def test_standalone_entry_is_the_levant4d_entry():
    assert STANDALONE_SYSTEMS[KEY] == STANDALONE_SYSTEMS["levant4d"]


@pytest.mark.parametrize("corner_index", range(4))
def test_builder_matches_framework_at_every_corner_nominal(corner_index):
    point = FLAGSHIP_GAIN_POINTS[KEY]
    assignment = corner_assignments(point)[corner_index]
    gains = exact_gain_values(point, assignment)
    ours = build_system_dynamics(KEY, gains, Fraction(0), None)
    theirs = [exact_terms_from_gf(f_i)
              for f_i in exact_dynamics(point, assignment).dynamics]
    _assert_same_dynamics(ours, theirs, f"{KEY} corner {corner_index}")


@pytest.mark.parametrize("corner_index", range(4))
@pytest.mark.parametrize("endpoint_idx,endpoint_sign", [(0, +1), (1, -1)])
def test_builder_matches_framework_at_every_corner_perturbed(
        corner_index, endpoint_idx, endpoint_sign):
    point = FLAGSHIP_GAIN_POINTS[KEY]
    assignment = corner_assignments(point)[corner_index]
    gains = exact_gain_values(point, assignment)
    delta0 = Fraction(1, 16)
    vertices = matched_disturbance_vertices(
        exact_dynamics(point, assignment).dynamics, point.channel, delta0)
    ours = build_system_dynamics(KEY, gains, delta0, endpoint_sign)
    theirs = [exact_terms_from_gf(f_i) for f_i in vertices[endpoint_idx]]
    _assert_same_dynamics(ours, theirs,
                          f"{KEY} corner {corner_index} endpoint {endpoint_sign}")


# ---------------------------------------------------------------------------
# 4. The covariant representative
# ---------------------------------------------------------------------------

def _scaled_brackets(p):
    point = FLAGSHIP_GAIN_POINTS[KEY]
    out = {}
    for name, (lo, hi) in brackets(point).items():
        g = point.gains[name]
        i = GAIN_INDEX[name]
        out[name] = (lo * p ** i, hi * p ** i, g.power_k,
                     g.power_target * p ** (i * g.power_k))
    return out


@pytest.mark.parametrize("p", [Fraction(1, 2), Fraction(3, 4), Fraction(1)])
def test_scaled_bracket_stays_valid_in_Q(p):
    for name, (lo, hi, k, target) in _scaled_brackets(p).items():
        assert lo ** k < target < hi ** k, name
    s = _scaled_brackets(Fraction(1, 2))
    assert s["lambda2"][3] == Fraction(675, 64)
    assert s["lambda3"][3] == Fraction(492075, 16777216)


CELLS = [("cell_p1_2_m9_eps1e-4", Fraction(1, 2)),
         ("cell_p3_4_m9_eps1e-4", Fraction(3, 4))]


@pytest.mark.parametrize("cell,p", CELLS)
def test_x10_cell_states_the_scaled_bracket(cell, p):
    header_path = X10 / cell / "exact" / "header.json"
    if not header_path.exists():
        pytest.skip("X10 exact cell not present")
    header = json.loads(header_path.read_text())
    assert header["system"] == KEY
    assert Fraction(header["qscale_p"]) == p
    assert Fraction(header["delta0_exact"]) == p ** 4
    want = _scaled_brackets(p)
    for name, check in header["bracket_checks_in_Q"].items():
        lo, hi, k, target = want[name]
        assert check["valid_in_Q"] is True
        assert (Fraction(check["lo_scaled"]), Fraction(check["hi_scaled"]),
                int(check["power_k"]),
                Fraction(check["power_target_scaled"])) == (lo, hi, k, target)
    row = json.loads((X10 / cell / "exact" / "ROW.json").read_text())
    assert row["exact_pass_all"] is True
    assert row["witness_sources"] == ["design_bundle"]


# ---------------------------------------------------------------------------
# 5. The X10 portable records
# ---------------------------------------------------------------------------

RECORDS = [("levant4d_2003print_p1_2_m9_delta1_16", "cell_p1_2_m9_eps1e-4",
            Fraction(1, 2)),
           ("levant4d_2003print_p3_4_m9_delta81_256", "cell_p3_4_m9_eps1e-4",
            Fraction(3, 4))]


@pytest.mark.parametrize("stem,cell,p", RECORDS)
def test_x10_record_is_verified_and_bound_to_its_bytes(stem, cell, p):
    portable = X10 / "portable"
    record_path = portable / f"{stem}.record.json.gz"
    summary_path = portable / f"{stem}.verification.json"
    if not (record_path.exists() and summary_path.exists()):
        pytest.skip("X10 portable record not present")
    with gzip.open(record_path, "rt", encoding="utf-8") as fh:
        record = json.load(fh)
    summary = json.loads(summary_path.read_text())
    assert summary["verdict"] == "VERIFIED"
    assert summary["system"] == KEY == record["system"]
    assert summary["eps_certified_exact"] == record["eps_certified_exact"]
    assert Fraction(summary["eps_certified_exact"]) > 0
    assert Fraction(record["delta0_exact"]) == p ** 4
    # the record's gains are the registry point scaled by p^i, in Q
    point = FLAGSHIP_GAIN_POINTS[KEY]
    for name in ("lambda1", "lambda4"):
        assert Fraction(record["rational_gains"][name]) == \
            point.gains[name].value * p ** GAIN_INDEX[name]
    for name, (lo, hi, k, target) in _scaled_brackets(p).items():
        b = record["brackets"][name]
        assert (Fraction(b["lo"]), Fraction(b["hi"]), int(b["power_k"]),
                Fraction(b["power_target"])) == (lo, hi, k, target)
    assert record["provenance"]["representative_p"]["p"] == str(p)
    assert record["provenance"]["representative_p"]["delta0_over_p_power"] == "1"
    binding = summary["proof_inputs"]
    assert binding["schema"] == "gf-proof-inputs/1"
    assert binding["record"]["sha256"] == \
        hashlib.sha256(record_path.read_bytes()).hexdigest()
    grams_path = portable / record["grams_file"]
    assert binding["grams"]["sha256"] == \
        hashlib.sha256(grams_path.read_bytes()).hexdigest()
    row = json.loads((X10 / cell / "exact" / "ROW.json").read_text())
    assert abs(float(Fraction(summary["eps_certified_exact"]))
               - row["eps_certified_over_corners"]) < 1e-12
