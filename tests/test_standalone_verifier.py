"""Standalone verifier (verification/standalone/verify_certificate.py).

Four guarantees:
  1. The verifier's hand-rebuilt dynamics are IDENTICAL (coefficients,
     exponents, types, term-for-term after collection) to the gf_framework
     GF builders for every system it accepts — the one place a standalone
     re-implementation could silently diverge. Since campaign item C10 that
     is all twelve benchmark families, nominal and at both inclusion
     endpoints, not just the four flagship rows.
  2. The verifier stays STDLIB ONLY. Enforced by scanning its imports, not by
     hoping: it ships to readers who run it with a bare python3.
  3. End-to-end: the verifier accepts the shipped exactgains artifacts (when
     present) under a subprocess with the SAME semantics as the in-framework
     gate, and rejects a corrupted Gram. Records may be gzipped.
  4. The nominal and perturbed record shapes cannot be confused for each
     other: neither a dropped inclusion endpoint nor a flipped flag turns a
     half-covered perturbed record into an accepted nominal one.
"""
import ast
import gzip
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "standalone"))
sys.path.insert(0, str(ROOT / "scripts"))

VERIFIER = ROOT / "verification" / "standalone" / "verify_certificate.py"

from tamper_lib import run_tampered as _run_tampered_impl  # noqa: E402

from verify_certificate import (  # noqa: E402
    SYSTEMS as STANDALONE_SYSTEMS,
    VerifyError,
    build_dynamics,
    build_system_dynamics,
    deck_group_basis,
    exact_collect,
    exact_neg_lie_derivative,
    exact_octant_target,
    parity_blocks,
    q_exact,
    q_from,
    read_json,
    validate_octant,
    verify_captures,
)

from gf_framework.discovery.universal_discovery import (  # noqa: E402
    create_levant3d_spec,
    create_levant4d_spec,
    create_sta_spec,
    matched_disturbance_vertices,
)
from gf_framework.systems.pidsmc4d import (  # noqa: E402
    get_pidsmc4d_dynamics_gf,
)
from gf_framework.systems.levant5d import (  # noqa: E402
    get_levant5d_dynamics_gf,
)
from gf_framework.systems.printed_gains import (  # noqa: E402
    BENCHMARK_GAIN_POINTS,
    FLAGSHIP_GAIN_POINTS,
    corner_assignments,
    exact_dynamics,
    exact_gain_values,
    gain_point,
)
from gf_framework.verification.exact_gate import exact_terms_from_gf  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "ablation" / "levant_2003_perturbed"

#: Systems with no weighted-degree-0 channel: no constant matched disturbance
#: is admissible, so they exist only as nominal records.
NOMINAL_ONLY = tuple(sorted(s for s, p in BENCHMARK_GAIN_POINTS.items()
                            if p.channel is None))
PERTURBABLE = tuple(sorted(s for s, p in BENCHMARK_GAIN_POINTS.items()
                           if p.channel is not None))


def _framework_vertex_terms(spec, endpoint_idx):
    return [sorted(exact_terms_from_gf(f_i), key=lambda t: (t[1], t[2]))
            for f_i in spec.dynamics_gf_vertices[endpoint_idx]]


def _normalize(channels):
    """Collect and sort each channel, so the comparison is term for term."""
    return [sorted(exact_collect(list(terms)), key=lambda t: (t[1], t[2]))
            for terms in channels]


def _assert_same_dynamics(ours, theirs, label):
    assert len(ours) == len(theirs), f"{label}: channel count"
    for channel, (a, b) in enumerate(zip(_normalize(ours), _normalize(theirs))):
        assert a == b, f"{label}: channel {channel}\n  ours   {a}\n  theirs {b}"


# ---------------------------------------------------------------------------
# C10: one hand-written builder per family, pinned to the framework's GF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", sorted(BENCHMARK_GAIN_POINTS))
def test_builder_matches_framework_nominal(system):
    """Every one of the twelve benchmark families, nominal, at the registered
    printed-gain point. This is the guard that lets the verifier refuse to
    read dynamics from a record: its own equations are the framework's."""
    point = gain_point(system)
    assignment = corner_assignments(point)[0]
    gains = exact_gain_values(point, assignment)
    ours = build_system_dynamics(system, gains, Fraction(0), None)
    theirs = [exact_terms_from_gf(f_i)
              for f_i in exact_dynamics(point, assignment).dynamics]
    _assert_same_dynamics(ours, theirs, f"{system} nominal")


@pytest.mark.parametrize("system", PERTURBABLE)
@pytest.mark.parametrize("endpoint_idx,endpoint_sign", [(0, +1), (1, -1)])
def test_builder_matches_framework_perturbed(system, endpoint_idx, endpoint_sign):
    """Both inclusion endpoints of every disturbance-admissible family."""
    delta0 = Fraction(1)
    point = gain_point(system)
    assignment = corner_assignments(point)[0]
    gains = exact_gain_values(point, assignment)
    vertices = matched_disturbance_vertices(
        exact_dynamics(point, assignment).dynamics, point.channel, delta0)
    ours = build_system_dynamics(system, gains, delta0, endpoint_sign)
    theirs = [exact_terms_from_gf(f_i) for f_i in vertices[endpoint_idx]]
    _assert_same_dynamics(ours, theirs, f"{system} endpoint {endpoint_sign}")


def test_every_standalone_system_is_registered():
    """No benchmark system without a builder, and every builder that has no
    gain point named here on purpose.

    ``levant3d_2003print`` is the 2003-print variant of a benchmark row, and
    ``levant4d_2003print`` (WP X10) the same for the third-order
    differentiator: identical equations to ``levant4d``, the higher-gain
    standard tuning (1.1, 1.5, 3, 5), registered in the flagship table.
    ``pidsmc4d`` (P4-10) has NO printed-gain registry entry by design: the
    only gain point it has ever been run at is a quarantined best-guess
    candidate, not a printed tuning, so a record for it carries its gains
    inline and ``scripts/emit_certificate_record.py`` (which resolves a
    registered gain point) is not its emission path.
    """
    assert set(BENCHMARK_GAIN_POINTS) <= set(STANDALONE_SYSTEMS)
    extra = set(STANDALONE_SYSTEMS) - set(BENCHMARK_GAIN_POINTS)
    assert extra == {"levant3d_2003print", "levant4d_2003print", "pidsmc4d",
                     "levant5d"}, extra
    assert "levant4d_2003print" in FLAGSHIP_GAIN_POINTS
    assert "levant4d_2003print" not in BENCHMARK_GAIN_POINTS
    # levant5d (P4-04) is the one of the three that DOES have a registry entry:
    # the flagship table, because it is the n = 5 scale-boundary probe and not
    # part of the twelve-row benchmark matrix.
    assert "levant5d" in FLAGSHIP_GAIN_POINTS
    assert "levant5d" not in BENCHMARK_GAIN_POINTS


#: Rational gain points for the levant5d builder pin. Deliberately arbitrary
#: rationals and NOT the printed Levant-2003 conversion: the pin is about the
#: equations, and a builder that agreed only at one tuning would not be pinned.
LEVANT5D_GAIN_POINTS = [
    {"lambda1": Fraction(5), "lambda2": Fraction(7, 3), "lambda3": Fraction(9, 4),
     "lambda4": Fraction(11, 7), "lambda5": Fraction(11, 10)},
    {"lambda1": Fraction(1), "lambda2": Fraction(1), "lambda3": Fraction(1),
     "lambda4": Fraction(1), "lambda5": Fraction(1)},
    {"lambda1": Fraction(3), "lambda2": Fraction(1003110457464633, 100000000000000),
     "lambda3": Fraction(4651208827241701, 500000000000000),
     "lambda4": Fraction(914995950211533, 200000000000000),
     "lambda5": Fraction(4, 7)},
]


@pytest.mark.parametrize("gains", LEVANT5D_GAIN_POINTS)
def test_levant5d_builder_matches_framework_nominal(gains):
    """P4-04: the n = 5 member of the differentiator recursion this file
    writes out is ``gf_framework.systems.levant5d``, term for term, exactly.

    levant5d is not in ``BENCHMARK_GAIN_POINTS`` (it is the scale-boundary
    probe, registered in the flagship table only), so the parametrized pin
    above never reaches it; this is its equivalent, driven straight off
    ``get_levant5d_dynamics_gf``.
    """
    ours = build_system_dynamics("levant5d", dict(gains), Fraction(0), None)
    theirs = [exact_terms_from_gf(f_i)
              for f_i in get_levant5d_dynamics_gf(**{k: v for k, v in gains.items()})]
    _assert_same_dynamics(ours, theirs, "levant5d nominal")


@pytest.mark.parametrize("gains", LEVANT5D_GAIN_POINTS)
@pytest.mark.parametrize("endpoint_idx,endpoint_sign", [(0, +1), (1, -1)])
def test_levant5d_builder_matches_framework_perturbed(gains, endpoint_idx,
                                                      endpoint_sign):
    """Both inclusion endpoints on channel 4, the unique weighted-degree-0
    component of the fourth-order differentiator (kappa + r_5 = -1 + 1 = 0).

    The P4-04 record is NOMINAL, so this half is not exercised by it. It is
    pinned anyway: the moment a perturbed levant5d record exists, the
    equations it is checked against must already be the framework's.
    """
    delta0 = Fraction(3, 10)
    vertices = matched_disturbance_vertices(
        get_levant5d_dynamics_gf(**gains), 4, delta0)
    ours = build_system_dynamics("levant5d", dict(gains), delta0, endpoint_sign)
    theirs = [exact_terms_from_gf(f_i) for f_i in vertices[endpoint_idx]]
    _assert_same_dynamics(ours, theirs, f"levant5d endpoint {endpoint_sign}")


@pytest.mark.parametrize("corner_index", range(8))
def test_levant5d_builder_matches_the_registered_flagship_corners(corner_index):
    """All eight bracket corners of the registered flagship point (P4-01).

    These are the exact rationals the P4-03 exact phase ran at, so this is the
    join between the printed-gain registry and the equations this verifier
    rebuilds: if either side drifts, the portable record stops verifying
    instead of agreeing with the drift.
    """
    point = gain_point("levant5d", flagship=True)
    assignments = corner_assignments(point)
    assert len(assignments) == 8
    assignment = assignments[corner_index]
    gains = exact_gain_values(point, assignment)
    ours = build_system_dynamics("levant5d", gains, Fraction(0), None)
    theirs = [exact_terms_from_gf(f_i)
              for f_i in exact_dynamics(point, assignment).dynamics]
    _assert_same_dynamics(ours, theirs, f"levant5d corner {corner_index}")


def test_levant5d_structure_is_the_registered_one():
    """Weights, channel and gain order are what a record is checked against."""
    cfg = STANDALONE_SYSTEMS["levant5d"]
    assert cfg["weights_r"] == (5, 4, 3, 2, 1)
    assert cfg["n"] == 5 and cfg["channel"] == 4
    assert cfg["gain_names"] == ["lambda1", "lambda2", "lambda3", "lambda4",
                                 "lambda5"]
    assert cfg["family"] == "levant_differentiator"
    assert "fixed" not in cfg
    with pytest.raises(VerifyError, match="missing gains"):
        build_system_dynamics("levant5d", {"lambda1": Fraction(5)},
                              Fraction(0), None)
    gains = dict(LEVANT5D_GAIN_POINTS[0])
    gains["delta"] = Fraction(1)
    with pytest.raises(VerifyError, match="not a parameter"):
        build_system_dynamics("levant5d", gains, Fraction(0), None)
    with pytest.raises(VerifyError, match="nominal dynamics with delta0"):
        build_system_dynamics("levant5d", dict(LEVANT5D_GAIN_POINTS[0]),
                              Fraction(1, 10), None)


#: Rational gain points for the pidsmc4d builder pin. Deliberately arbitrary
#: rationals, NOT the quarantined candidate values: the pin is about the
#: equations, and a builder that agreed only at one tuning would not be pinned.
PIDSMC4D_GAIN_POINTS = [
    {"k1": Fraction(7, 3), "k2": Fraction(19, 5), "k3": Fraction(11, 7),
     "k4": Fraction(11, 10)},
    {"k1": Fraction(1), "k2": Fraction(1), "k3": Fraction(1), "k4": Fraction(1)},
    {"k1": Fraction(3), "k2": Fraction(2080083823051904, 500000000000000),
     "k3": Fraction(30594733539832584, 10000000000000000),
     "k4": Fraction(4, 7)},
]


@pytest.mark.parametrize("gains", PIDSMC4D_GAIN_POINTS)
def test_pidsmc4d_builder_matches_framework_nominal(gains):
    """P4-10: the hand-written fourth-order PID-like equations are the
    framework's, term for term, in exact arithmetic.

    pidsmc4d is not in ``BENCHMARK_GAIN_POINTS``, so the parametrized pin
    above never reaches it; this is its equivalent, driven straight off
    ``gf_framework.systems.pidsmc4d.get_pidsmc4d_dynamics_gf``.
    """
    ours = build_system_dynamics("pidsmc4d", dict(gains), Fraction(0), None)
    theirs = [exact_terms_from_gf(f_i)
              for f_i in get_pidsmc4d_dynamics_gf(**gains)]
    _assert_same_dynamics(ours, theirs, "pidsmc4d nominal")


@pytest.mark.parametrize("gains", PIDSMC4D_GAIN_POINTS)
@pytest.mark.parametrize("endpoint_idx,endpoint_sign", [(0, +1), (1, -1)])
def test_pidsmc4d_builder_matches_framework_perturbed(gains, endpoint_idx,
                                                      endpoint_sign):
    """Both inclusion endpoints on channel 3, the unique weighted-degree-0
    component of this family."""
    delta0 = Fraction(3, 10)
    vertices = matched_disturbance_vertices(
        get_pidsmc4d_dynamics_gf(**gains), 3, delta0)
    ours = build_system_dynamics("pidsmc4d", dict(gains), delta0, endpoint_sign)
    theirs = [exact_terms_from_gf(f_i) for f_i in vertices[endpoint_idx]]
    _assert_same_dynamics(ours, theirs, f"pidsmc4d endpoint {endpoint_sign}")


def test_pidsmc4d_has_no_gain_scaling_parameter():
    """The third-order pidsmc factory scales its gains by delta**(2/3) and
    delta**(1/2), which is why ``SYSTEMS['pidsmc']`` pins ``delta = 1``. The
    fourth-order factory takes no delta at all, so a record carrying one is an
    unregistered extra gain and must be refused, not pinned to a case."""
    assert "fixed" not in STANDALONE_SYSTEMS["pidsmc4d"]
    gains = dict(PIDSMC4D_GAIN_POINTS[0])
    gains["delta"] = Fraction(1)
    with pytest.raises(VerifyError, match="not a parameter"):
        build_system_dynamics("pidsmc4d", gains, Fraction(0), None)


def test_pidsmc4d_structure_is_the_registered_one():
    """Weights, channel and gain order are what a record is checked against."""
    cfg = STANDALONE_SYSTEMS["pidsmc4d"]
    assert cfg["weights_r"] == (4, 3, 2, 1)
    assert cfg["n"] == 4 and cfg["channel"] == 3
    assert cfg["gain_names"] == ["k1", "k2", "k3", "k4"]
    assert cfg["family"] == "pidsmc4d"
    with pytest.raises(VerifyError, match="missing gains"):
        build_system_dynamics("pidsmc4d", {"k1": Fraction(1)}, Fraction(0), None)


@pytest.mark.parametrize("system", NOMINAL_ONLY)
def test_builder_refuses_perturbed_without_a_channel(system):
    """kappa + r_ch != 0 for these: a perturbed record for them is a category
    error, and must be refused rather than built on an arbitrary channel."""
    point = gain_point(system)
    gains = exact_gain_values(point, corner_assignments(point)[0])
    with pytest.raises(VerifyError, match="weighted-degree-0 channel"):
        build_system_dynamics(system, gains, Fraction(1), +1)


def test_builder_refuses_unknown_and_missing_gains():
    """Every builder input is accounted for: nothing silently ignored."""
    with pytest.raises(VerifyError, match="missing gains"):
        build_system_dynamics("sta", {"k1": Fraction(3, 2)}, Fraction(0), None)
    with pytest.raises(VerifyError, match="not a parameter"):
        build_system_dynamics("sta", {"k1": Fraction(3, 2), "k2": Fraction(11, 10),
                                      "k9": Fraction(1)}, Fraction(0), None)


def test_builder_refuses_pidsmc_outside_the_recorded_scaling():
    """pidsmc's factory scales its gains by delta**(2/3) and delta**(1/2); the
    hand-written equations are the delta = 1 case, so any other delta must be
    refused instead of quietly verified against the wrong dynamics."""
    point = gain_point("pidsmc")
    gains = dict(exact_gain_values(point, {}))
    gains["delta"] = Fraction(2)
    with pytest.raises(VerifyError, match="outside the case"):
        build_system_dynamics("pidsmc", gains, Fraction(0), None)


# ---------------------------------------------------------------------------
# C10: the verifier must remain runnable by a bare python3
# ---------------------------------------------------------------------------

def test_verifier_imports_only_stdlib():
    """Scan the verifier's imports. The file ships as supplementary material
    and its whole value is that a reader can run it with nothing installed;
    one convenience import of numpy or gf_framework would destroy that."""
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.level, "relative import in a standalone file"
            imported.add((node.module or "").split(".")[0])
    forbidden = imported - set(sys.stdlib_module_names)
    assert not forbidden, f"non-stdlib imports in the standalone verifier: {sorted(forbidden)}"
    assert "gf_framework" not in imported and "numpy" not in imported


@pytest.mark.parametrize("system,factory,gains", [
    ("sta", create_sta_spec,
     {"k1": Fraction(3, 2), "k2": Fraction(11, 10)}),
    ("levant3d", create_levant3d_spec,
     {"lambda1": Fraction(2), "lambda2": Fraction(21213, 10000),
      "lambda3": Fraction(11, 10)}),
    ("levant4d", create_levant4d_spec,
     {"lambda1": Fraction(3), "lambda2": Fraction(416, 100),
      "lambda3": Fraction(306, 100), "lambda4": Fraction(11, 10)}),
])
def test_standalone_dynamics_match_framework(system, factory, gains):
    delta0 = Fraction(1)
    spec = factory(**gains, delta0=delta0)
    cfg = STANDALONE_SYSTEMS[system]
    gain_list = [gains[g] for g in cfg["gain_names"]]
    for endpoint_idx, endpoint_sign in ((0, +1), (1, -1)):
        ours = build_dynamics(cfg["n"], gain_list, delta0, endpoint_sign)
        theirs = _framework_vertex_terms(spec, endpoint_idx)
        assert len(ours) == len(theirs)
        for ch, (a, b) in enumerate(zip(ours, theirs)):
            a = sorted(exact_collect(a), key=lambda t: (t[1], t[2]))
            assert a == b, f"{system} endpoint {endpoint_sign} channel {ch}"


needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "sta_exactgains_20260711T144752Z.json").exists(),
    reason="exactgains artifacts not present (untracked by policy)",
)


@needs_artifacts
def test_standalone_verifier_accepts_sta():
    res = subprocess.run(
        [sys.executable,
         str(ROOT / "verification/standalone/verify_certificate.py"),
         str(ARTIFACTS / "sta_exactgains_20260711T144752Z.json"),
         "--row", str(ARTIFACTS / "sta_delta1_row.json")],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "VERIFIED" in res.stdout


LEVANT3D_REC = "levant3d_exactgains_20260712T062217Z.json"

needs_bracketed_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / LEVANT3D_REC).exists(),
    reason="bracketed exactgains artifacts not present (untracked by policy)",
)


def _run_tampered(tmp_path, rec, bundle, flags=()):
    """Write the tampered (rec, bundle) pair to tmp_path and run the
    verifier on it. Thin wrapper around ``tamper_lib.run_tampered`` (shared
    with scripts/tamper_battery.py, WP X6) so this file and the battery do
    not each keep their own copy of the write-then-subprocess boilerplate."""
    return _run_tampered_impl(VERIFIER, rec, bundle, tmp_path, extra_args=flags)


def _load_levant3d():
    rec = json.loads((ARTIFACTS / LEVANT3D_REC).read_text())
    bundle = json.loads((ARTIFACTS / rec["grams_file"]).read_text())
    return rec, bundle


@needs_bracketed_artifacts
def test_rejects_missing_bracket_corner(tmp_path):
    """G1: dropping the hi corner must be rejected — the interior irrational
    gain would otherwise be uncovered by the affinity argument."""
    rec, bundle = _load_levant3d()
    rec["corners"] = rec["corners"][:1]
    bundle["corners"] = bundle["corners"][:1]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1
    assert "corner box incomplete" in res.stderr


@needs_bracketed_artifacts
def test_rejects_duplicated_corner(tmp_path):
    """G1: two copies of the lo corner (count right, set wrong) must be
    rejected — pigeonhole completeness needs DISTINCT assignments."""
    rec, bundle = _load_levant3d()
    rec["corners"][1] = json.loads(json.dumps(rec["corners"][0]))
    bundle["corners"][1] = json.loads(json.dumps(bundle["corners"][0]))
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1
    assert "distinct corner" in res.stderr


@needs_bracketed_artifacts
def test_rejects_missing_w_endpoint(tmp_path):
    """G1: dropping every W- capture must be rejected — half the disturbance
    interval would otherwise be uncovered."""
    rec, bundle = _load_levant3d()
    for bc in bundle["corners"]:
        bc["endpoints"] = bc["endpoints"][:1]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1
    assert "need 2 (W+ and W-)" in res.stderr


@needs_bracketed_artifacts
def test_rejects_bogus_octant_signs(tmp_path):
    """Readiness review 2026-07-15: 2^n DISTINCT sign patterns that are not
    the {-1,+1}^n hypercube must be rejected — an invented pattern such as
    (0, 1, 1) could otherwise satisfy the distinct-count check while leaving
    a true octant unproved."""
    rec, bundle = _load_levant3d()
    bundle["V_grams"][0]["signs"] = [0, 1, 1]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1
    assert "octant coverage" in res.stderr


@needs_bracketed_artifacts
def test_rejects_corner_assignment_mismatch(tmp_path):
    """Readiness review 2026-07-15: a grams bundle whose corner assignment
    differs from the record's must raise VerifyError (was a bare assert,
    which python -O removes), or the W captures would be validated against
    the wrong gains."""
    rec, bundle = _load_levant3d()
    bundle["corners"][0]["assignment"] = dict(bundle["corners"][1]["assignment"])
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: record/bundle do not pair
    assert "corner order mismatch" in res.stderr


@needs_artifacts
def test_rejects_eps_mismatch(tmp_path):
    """G15: a capture whose margin differs from the record's epsilon_exact
    must be rejected (the banner cites epsilon_exact)."""
    rec = json.loads(
        (ARTIFACTS / "sta_exactgains_20260711T144752Z.json").read_text())
    bundle = json.loads((ARTIFACTS / rec["grams_file"]).read_text())
    row = json.loads((ARTIFACTS / "sta_delta1_row.json").read_text())
    rec["V"] = {"degree_m": row["degree_used"], "weights_r": row["weights_r"],
                "terms": row["V_terms"]}
    bundle["V_grams"][0]["eps"] = bundle["V_grams"][0]["eps"] / 2
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1
    assert "epsilon_exact" in res.stderr


@needs_artifacts
def test_rejects_zero_epsilon(tmp_path):
    """A record whose epsilon_exact is zero must be rejected cleanly (a
    VerifyError with an explicit message), not crash with an unrelated
    ZeroDivisionError deep inside the gate-budget arithmetic (mirrors the
    companion repo's test_verifier.test_rejects_zero_epsilon)."""
    rec = json.loads(
        (ARTIFACTS / "sta_exactgains_20260711T144752Z.json").read_text())
    bundle = json.loads((ARTIFACTS / rec["grams_file"]).read_text())
    row = json.loads((ARTIFACTS / "sta_delta1_row.json").read_text())
    rec["V"] = {"degree_m": row["degree_used"], "weights_r": row["weights_r"],
                "terms": row["V_terms"]}
    rec["epsilon_exact"] = "0"
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: eps must be positive
    assert "margin epsilon must be positive" in res.stderr


# ---------------------------------------------------------------------------
# C9/C10: gzip container, compact encoding, nominal vs perturbed record shapes
# ---------------------------------------------------------------------------

def _sta_record_with_V():
    rec = json.loads(
        (ARTIFACTS / "sta_exactgains_20260711T144752Z.json").read_text())
    bundle = json.loads((ARTIFACTS / rec["grams_file"]).read_text())
    row = json.loads((ARTIFACTS / "sta_delta1_row.json").read_text())
    rec["V"] = {"degree_m": row["degree_used"], "weights_r": row["weights_r"],
                "terms": row["V_terms"]}
    return rec, bundle


def _write_gz(path, payload):
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(path, "wb"),
                       mtime=0) as fh:
        fh.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


@needs_artifacts
def test_verifier_reads_gzipped_records(tmp_path):
    """Record-size policy: everything stays in git, gzipped. The verifier must
    read the .json.gz container with stdlib gzip, and reach the same verdict."""
    rec, bundle = _sta_record_with_V()
    gr_p = tmp_path / "grams.json.gz"
    rec_p = tmp_path / "rec.json.gz"
    rec["grams_file"] = gr_p.name
    _write_gz(gr_p, bundle)
    _write_gz(rec_p, rec)
    assert rec_p.read_bytes()[:2] == b"\x1f\x8b"
    res = subprocess.run([sys.executable, str(VERIFIER), str(rec_p)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "VERIFIED" in res.stdout


@needs_artifacts
def test_verifier_reads_hex_encoded_grams(tmp_path):
    """Compact encoding: a Gram entry may be its exact float.hex() string.
    Same certificate, same verdict, no precision traded away."""
    rec, bundle = _sta_record_with_V()
    for caps in [bundle["V_grams"]] + [c for bc in bundle["corners"]
                                       for c in bc["endpoints"]]:
        for cap in caps:
            cap["eps"] = float(cap["eps"]).hex()
            cap["gram"] = [[float(x).hex() for x in row] for row in cap["gram"]]
    rec["V"]["terms"] = [dict(t, coefficient=float(t["coefficient"]).hex())
                         for t in rec["V"]["terms"]]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 0, res.stderr
    assert "VERIFIED" in res.stdout


@needs_artifacts
def test_verifier_rejects_hex_encoded_published_gains(tmp_path):
    """A published gain is the printed rational, never a float64. Encoding it
    as hex would silently reinstate the dyadic-image caveat that the printed
    rational injection removed, so it is refused."""
    rec, bundle = _sta_record_with_V()
    rec["rational_gains"]["k2"] = float(1.1).hex()
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: encoding
    assert "hex float64 encoded" in res.stderr


@needs_artifacts
def test_verifier_rejects_nominal_flag_on_a_perturbed_record(tmp_path):
    """Dropping W- and flipping ``nominal`` must not buy acceptance: the
    nominal shape stores its captures under a DIFFERENT key, so half a
    perturbed record cannot masquerade as a whole nominal one."""
    rec, bundle = _sta_record_with_V()
    rec["nominal"] = True
    rec["channel"] = None
    for bc in bundle["corners"]:
        bc["endpoints"] = bc["endpoints"][:1]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: record shape
    assert "nominal" in res.stderr


@needs_artifacts
def test_verifier_rejects_contradicted_family(tmp_path):
    """A v2 record names its family and channel; both are re-derived from the
    system name here and must agree."""
    rec, bundle = _sta_record_with_V()
    rec["schema_version"] = "gf-certificate-record/2"
    rec["system_family"] = "twisting"
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: declarations
    assert "contradicts the published system" in res.stderr


@needs_artifacts
def test_verifier_rejects_unknown_schema(tmp_path):
    rec, bundle = _sta_record_with_V()
    rec["schema_version"] = "gf-certificate-record/99"
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: schema
    assert "unknown record schema" in res.stderr


def test_number_decoders_agree_on_v1_and_v2_encodings():
    """q_from accepts the float encodings, q_exact refuses them; both give the
    same exact value for the rational form."""
    assert q_from(1.1) == q_from(float(1.1).hex()) == Fraction(1.1)
    assert q_exact("11/10", "gain") == Fraction(11, 10) != Fraction(1.1)
    with pytest.raises(VerifyError):
        q_exact(1.1, "gain")
    with pytest.raises(VerifyError):
        q_exact(float(1.1).hex(), "gain")


# ---------------------------------------------------------------------------
# P3-01: deck-group parity blocking of the exact LDL^T
#
# The blocked path is the default; --dense is the reference. Three things must
# hold: the block structure is DERIVED here (never read from the record), the
# two paths agree bit for bit on the stored records, and a Gram entry between
# two parity classes is charged (not silently projected away), so corrupting
# one is rejected by both paths.
# ---------------------------------------------------------------------------

PORTABLE = ROOT / "artifacts" / "campaign_2026-08" / "E3_portable" / "nominal_records"

needs_portable = pytest.mark.skipif(
    not (PORTABLE / "pidsmc_nominal.record.json.gz").exists(),
    reason="E3 portable records not present (untracked by policy)",
)


def test_f2_null_space_of_an_all_even_support_is_everything():
    """An all-even support constrains nothing mod 2, so H = F_2^n and the
    basis has n independent generators."""
    support = [(4, 0, 0), (2, 2, 0), (0, 0, 4)]
    gb = deck_group_basis(support, 3)
    assert len(gb) == 3
    for g in gb:
        for alpha in support:
            assert sum(gi * ai for gi, ai in zip(g, alpha)) % 2 == 0


def test_f2_null_space_shrinks_with_an_odd_exponent():
    """One odd exponent in x_0 kills one dimension of H."""
    assert len(deck_group_basis([(3, 0, 0), (1, 2, 0)], 3)) == 2
    # Two independent odd patterns kill two.
    assert len(deck_group_basis([(1, 0, 0), (0, 1, 0)], 3)) == 1


def test_empty_deck_group_gives_exactly_the_dense_block():
    """dim H = 0 must reproduce the dense matrix, index for index: that is what
    makes the blocked path a strict generalisation of the reference path."""
    basis = [(2, 0), (1, 1), (0, 2)]
    assert parity_blocks(basis, []) == [[0, 1, 2]]


def test_parity_blocks_partition_every_index_and_respect_the_character():
    basis = [(2, 0, 0), (1, 1, 0), (0, 1, 1), (0, 0, 2), (1, 0, 1)]
    gb = deck_group_basis([(4, 0, 0), (0, 4, 0), (0, 0, 4), (2, 2, 0)], 3)
    blocks = parity_blocks(basis, gb)
    assert sorted(i for b in blocks for i in b) == list(range(len(basis)))
    for block in blocks:
        for i in block:
            for j in block:
                mono = tuple(a + b for a, b in zip(basis[i], basis[j]))
                for g in gb:
                    assert sum(gi * m for gi, m in zip(g, mono)) % 2 == 0


def _load_portable(system):
    rec = read_json(PORTABLE / f"{system}_nominal.record.json.gz")
    bundle = read_json(PORTABLE / rec["grams_file"])
    return rec, bundle


def _v_terms(rec):
    return [(q_from(t["coefficient"]),
             tuple(Fraction(e) for e in t["exponents"]),
             tuple(t["term_types"]))
            for t in rec["V"]["terms"]]


def _octant_charges(rec, bundle, blocked):
    """Every capture's exact charge/eps, in bundle order (P2-14)."""
    n = len(rec["weights_r"])
    eps_rec = q_exact(rec["epsilon_exact"], "epsilon_exact")
    v_terms = _v_terms(rec)
    gains = {k: q_exact(v, k) for k, v in rec["rational_gains"].items()}
    out = []
    work = [(v_terms, bundle["V_grams"])]
    for bc in bundle["corners"]:
        g = dict(gains)
        g.update({k: q_exact(v, k) for k, v in bc["assignment"].items()})
        dyn = build_system_dynamics(rec["system"], g, Fraction(0), None)
        work.append((exact_neg_lie_derivative(v_terms, dyn), bc["W_grams"]))
    for terms, caps in work:
        for cap in caps:
            target, d = exact_octant_target(
                terms, tuple(int(v) for v in cap["mu"]),
                tuple(int(s) for s in cap["signs"]), eps_rec)
            ok, ratio, why, shape = validate_octant(
                target, d, cap["gram"], cap["basis"], eps_rec, blocked=blocked)
            out.append((ok, ratio, why, shape[1]))
    assert len(out) == (1 + len(bundle["corners"])) * 2 ** n
    return out


@needs_portable
@pytest.mark.parametrize("system", ["fosmc", "sta", "twisting", "dsta",
                                    "example1", "example3", "levant3d",
                                    "pidsmc"])
def test_blocked_and_dense_agree_exactly_on_the_stored_records(system):
    """Bit-identical verdict AND exact rational charge/eps, per octant, not
    just on the printed worst.

    PRECONDITION (made explicit 2026-09-05, external review finding 12): this
    is a statement about THESE captures, not a law about the two paths. Both
    paths charge the same dense residual, so they agree exactly when they
    select the same eta rung — which they do here because every stored Gram is
    already PSD, so both stop at eta = 0 (pinned by
    ``test_the_stored_grams_are_psd_which_is_why_both_paths_agree``). In
    general only one implication holds: a dense PASS implies the blocked PASS,
    never the converse — see
    ``test_review_finding12_blocked_may_accept_where_dense_rejects``."""
    rec, bundle = _load_portable(system)
    blocked = _octant_charges(rec, bundle, blocked=True)
    dense = _octant_charges(rec, bundle, blocked=False)
    assert [(b[0], b[1]) for b in blocked] == [(d[0], d[1]) for d in dense]
    assert all(b[0] for b in blocked), system


@needs_portable
def test_blocking_actually_blocks_on_a_three_state_record():
    """A guard against a silent fallback to one block: pidsmc's V octants must
    genuinely split, or this WP bought nothing."""
    rec, bundle = _load_portable("pidsmc")
    n_blocks = {r[3] for r in _octant_charges(rec, bundle, blocked=True)}
    assert min(n_blocks) > 1, n_blocks


def _cross_class_pair(rec, bundle):
    """Index pair (i, j) lying in DIFFERENT parity classes of V_grams[0]."""
    cap = bundle["V_grams"][0]
    eps_rec = q_exact(rec["epsilon_exact"], "epsilon_exact")
    target, _ = exact_octant_target(
        _v_terms(rec), tuple(int(v) for v in cap["mu"]),
        tuple(int(s) for s in cap["signs"]), eps_rec)
    basis = [tuple(int(b) for b in beta) for beta in cap["basis"]]
    blocks = parity_blocks(basis, deck_group_basis(list(target), len(basis[0])))
    assert len(blocks) > 1
    return blocks[0][0], blocks[1][0]


@needs_portable
def test_cross_class_gram_corruption_is_rejected_by_both_paths(tmp_path):
    """One off-diagonal entry BETWEEN parity classes, set well beyond the
    budget. The blocked certificate drops that entry, so the naive projection
    would accept the tampered Gram while --dense rejected it. The residual is
    charged over every index pair precisely so that cannot happen: both paths
    must reject, with the same reason."""
    rec, bundle = _load_portable("pidsmc")
    i, j = _cross_class_pair(rec, bundle)
    gram = bundle["V_grams"][0]["gram"]
    gram[i][j] = q_from(gram[i][j]) + Fraction(1, 1000)
    gram[j][i] = q_from(gram[j][i]) + Fraction(1, 1000)
    gram[i][j] = float(gram[i][j])
    gram[j][i] = float(gram[j][i])
    for flags in ([], ["--dense"]):
        res = _run_tampered(tmp_path, json.loads(json.dumps(rec)),
                            json.loads(json.dumps(bundle)), flags)
        assert res.returncode == 1, (flags, res.stdout)
        assert "budget_exceeded" in res.stderr, (flags, res.stderr)


@needs_portable
def test_intra_class_gram_corruption_is_rejected_by_both_paths(tmp_path):
    """The ordinary case: a corrupted entry inside one block."""
    rec, bundle = _load_portable("pidsmc")
    bundle["V_grams"][0]["gram"][0][0] = \
        float(q_from(bundle["V_grams"][0]["gram"][0][0]) + Fraction(1, 1000))
    for flags in ([], ["--dense"]):
        res = _run_tampered(tmp_path, json.loads(json.dumps(rec)),
                            json.loads(json.dumps(bundle)), flags)
        assert res.returncode == 1, (flags, res.stdout)
        assert "FAILED" in res.stderr


@needs_portable
def test_a_declared_block_structure_is_re_derived_and_checked(tmp_path):
    """A record MAY state its block structure; it is never trusted. Widening a
    block would hide a cross-class entry from the residual, so a declaration
    that disagrees with the derivation is refused — in both paths."""
    rec, bundle = _load_portable("pidsmc")
    n_basis = len(bundle["V_grams"][0]["basis"])
    bundle["V_grams"][0]["blocks"] = [list(range(n_basis))]
    for flags in ([], ["--dense"]):
        res = _run_tampered(tmp_path, json.loads(json.dumps(rec)),
                            json.loads(json.dumps(bundle)), flags)
        assert res.returncode == 1, (flags, res.stdout)
        assert "contradicts the parity blocks derived here" in res.stderr


@needs_portable
def test_dense_flag_is_accepted_and_reaches_the_same_verdict(tmp_path):
    """End to end through the CLI, both paths, same VERIFIED banner.

    Same precondition as the octant-level test above: this record's captured
    Grams are PSD, so both paths take eta = 0 and charge the identical
    residual. Verdict equality is a property of healthy captures, not a law
    the two paths obey on an arbitrary Gram (review finding 12)."""
    rec = PORTABLE / "levant3d_nominal.record.json.gz"
    outs = []
    for flags in ([], ["--dense"]):
        res = subprocess.run([sys.executable, str(VERIFIER), str(rec)] + flags,
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "VERIFIED" in res.stdout
        outs.append([ln for ln in res.stdout.splitlines()
                     if "charge/eps" in ln])
    # Same printed worst charge/eps on every stage line, different path tag.
    assert [ln.split(";")[0] for ln in outs[0]] == \
           [ln.split(";")[0] for ln in outs[1]]


@needs_artifacts
def test_standalone_verifier_rejects_corrupted_gram(tmp_path):
    rec = json.loads(
        (ARTIFACTS / "sta_exactgains_20260711T144752Z.json").read_text())
    bundle = json.loads((ARTIFACTS / rec["grams_file"]).read_text())
    # Corrupt one Gram entry beyond the budget
    bundle["corners"][0]["endpoints"][0][0]["gram"][0][0] += 1e-3
    rec_p = tmp_path / "rec.json"
    gr_p = tmp_path / "grams.json"
    rec["grams_file"] = gr_p.name
    rec_p.write_text(json.dumps(rec))
    gr_p.write_text(json.dumps(bundle))
    res = subprocess.run(
        [sys.executable,
         str(ROOT / "verification/standalone/verify_certificate.py"),
         str(rec_p), "--row", str(ARTIFACTS / "sta_delta1_row.json")],
        capture_output=True, text=True)
    assert res.returncode == 1
    assert "FAILED" in res.stderr


# ---------------------------------------------------------------------------
# P4-04: the SHARED nominal W witness (decode-side aliasing of one Gram set
# across the bracket corners). Every test here asks the same question in a
# different way: does sharing the BYTES change anything about what is PROVED?
# ---------------------------------------------------------------------------

NOMINAL_RECORDS = ROOT / "artifacts" / "campaign_2026-08" / "E3_portable" / \
    "nominal_records"

needs_nominal_records = pytest.mark.skipif(
    not (NOMINAL_RECORDS / "levant3d_nominal.record.json.gz").exists(),
    reason="E3 portable nominal records not present",
)


def _load_nominal(name="levant3d_nominal"):
    rec = read_json(NOMINAL_RECORDS / f"{name}.record.json.gz")
    bundle = read_json(NOMINAL_RECORDS / f"{name}.grams.json.gz")
    return rec, bundle


def _share_w(bundle):
    """Rewrite a nominal bundle into the shared-witness form, unchanged bytes."""
    bundle["shared_W_grams"] = bundle["corners"][0]["W_grams"]
    for bc in bundle["corners"]:
        bc.pop("W_grams", None)
        bc["W_grams_shared"] = True
    return bundle


@needs_nominal_records
def test_shared_w_witness_reaches_the_identical_verdict(tmp_path):
    """The decisive one: duplicated bytes and shared bytes must produce the
    SAME stdout, charge ratio included. If they ever differ, the alias is not
    an alias."""
    rec, bundle = _load_nominal()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = _run_tampered(tmp_path / "a", rec, bundle)
    rec2, bundle2 = _load_nominal()
    shared = _run_tampered(tmp_path / "b", rec2, _share_w(bundle2))
    assert plain.returncode == 0, plain.stderr
    assert shared.returncode == 0, shared.stderr
    plain_lines = [l for l in plain.stdout.splitlines() if not l.startswith("[4/5]")]
    shared_lines = [l for l in shared.stdout.splitlines() if not l.startswith("[4/5]")]
    assert plain_lines == shared_lines
    # [4/5] differs only by the added witness annotation, never by the number.
    p4 = [l for l in plain.stdout.splitlines() if l.startswith("[4/5]")][0]
    s4 = [l for l in shared.stdout.splitlines() if l.startswith("[4/5]")][0]
    assert p4.split("charge/eps ")[1].split(";")[0] == \
        s4.split("charge/eps ")[1].split(";")[0]
    assert "shared witness Gram set" in s4 and "shared witness" not in p4


@needs_nominal_records
def test_shared_w_is_still_proved_at_every_corner(tmp_path):
    """A two-corner box whose second corner carries DIFFERENT gains: the
    shared Gram proves corner 0's W and must fail on corner 1's, or the alias
    would be silently skipping corners instead of re-proving them.

    The bracket is synthetic (lo = the certified gain, hi = twice it, a
    rational power target strictly between), which is exactly the shape the
    corner box has on a real bracketed row.
    """
    rec, bundle = _load_nominal()
    lo = Fraction(rec["rational_gains"].pop("lambda2"))
    hi = 2 * lo
    rec["brackets"] = {"lambda2": {"lo": str(lo), "hi": str(hi),
                                   "power_k": 1, "power_target": str(3 * lo / 2)}}
    for value, corners in ((lo, 0), (hi, 1)):
        entry = {"assignment": {"lambda2": str(value)},
                 "corner_label": f"lambda2={'lo' if corners == 0 else 'hi'}"}
        if corners == 0:
            rec["corners"] = [entry]
            bundle["corners"] = [dict(bundle["corners"][0], **entry)]
        else:
            rec["corners"].append(entry)
            bundle["corners"].append(dict(bundle["corners"][0], **entry))
    res = _run_tampered(tmp_path, rec, _share_w(bundle))
    assert res.returncode == 1
    assert "lambda2=hi" in res.stderr and "FAILED" in res.stderr


@needs_nominal_records
def test_shared_w_captures_still_face_the_coverage_check(tmp_path):
    """Dropping one octant from the SHARED set must be rejected exactly as
    dropping it from a per-corner set is: the completeness rule runs on the
    resolved list, not on whichever list happens to be inline."""
    rec, bundle = _load_nominal()
    bundle = _share_w(bundle)
    bundle["shared_W_grams"] = bundle["shared_W_grams"][:-1]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1
    assert "octant coverage" in res.stderr


@needs_nominal_records
def test_shared_w_refuses_both_and_neither(tmp_path):
    """Exactly one of W_grams / W_grams_shared per corner."""
    rec, bundle = _load_nominal()
    both = _share_w(json.loads(json.dumps(bundle)))
    both["corners"][0]["W_grams"] = both["shared_W_grams"]
    (tmp_path / "both").mkdir()
    (tmp_path / "neither").mkdir()
    res = _run_tampered(tmp_path / "both", rec, both)
    assert res.returncode == 2 and "both its own W_grams" in res.stderr

    rec2, bundle2 = _load_nominal()
    for bc in bundle2["corners"]:
        bc.pop("W_grams", None)
    res = _run_tampered(tmp_path / "neither", rec2, bundle2)
    assert res.returncode == 1 and "no W_grams captures" in res.stderr


@needs_nominal_records
def test_shared_w_flag_without_a_shared_set_is_refused(tmp_path):
    """A dangling reference is a refusal, never an empty capture list that
    would pass a coverage check vacuously."""
    rec, bundle = _load_nominal()
    bundle = _share_w(bundle)
    del bundle["shared_W_grams"]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1 and "no shared_W_grams" in res.stderr


@needs_bracketed_artifacts
def test_shared_w_is_refused_on_a_perturbed_record(tmp_path):
    """The alias is nominal-only. A perturbed row's W depends on the inclusion
    endpoint too, and a dropped endpoint is precisely what the two-endpoint
    rule exists to catch, so the shared shape is refused there outright."""
    rec, bundle = _load_levant3d()
    bundle["shared_W_grams"] = bundle["corners"][0]["endpoints"][0]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2  # INVALID INPUT: record shape
    assert "nominal W_grams captures" in res.stderr


# ---------------------------------------------------------------------------
# WP P2-14 (owner decision D-l): the portable verifier accepts at
# ``charge < eps`` and REPORTS the note's printed ``charge <= eps/2``.
#
# The instance is the smallest one that exhibits the whole mechanism, and the
# same one tests/test_margin_policy.py uses for the framework-side gate: one
# variable, d = 2, basis [(1,)], target {y^2: 1}, Gram [[q]]. Then
# z^T Q z = q y^2, E = (1 - q) y^2, ||E||_1 = |1 - q|, C1 = 1 and eta = 0, so
# charge = |1 - q| exactly. Every q below is a dyadic rational, so reading it
# through float64 is exact and the arithmetic is decidable by hand.
# ---------------------------------------------------------------------------

P214_EPS = Fraction(1, 1024)


def _p214_octant(q):
    """(target, d, gram_rows, basis, eps) for the one-variable instance."""
    return ({(2,): Fraction(1)}, 2, [[float(q)]], [(1,)], P214_EPS)


def test_p214_a_charge_between_half_eps_and_eps_is_now_PROVED():
    """The band the CTA m = 5 (eps 1e-7) and levant4d E8 32-term (eps 1e-6)
    rows fell into: refused by the old eps/2 threshold, proved by this one."""
    q = 1 - Fraction(3, 4) * P214_EPS          # charge = 3/4 eps
    ok, ratio, why, _shape = validate_octant(*_p214_octant(q))
    assert ok and why == ""
    assert ratio == Fraction(3, 4)


def test_p214_the_ratio_denominator_is_eps_not_half_eps():
    """The returned ratio is charge/eps. Before P2-14 the identical Gram gave
    twice this number (charge/(eps/2)), which is why every emitted field has
    to name its denominator."""
    q = 1 - Fraction(1, 4) * P214_EPS          # charge = eps/4
    ok, ratio, _why, _shape = validate_octant(*_p214_octant(q))
    assert ok and ratio == Fraction(1, 4)
    assert 2 * ratio == Fraction(1, 2)         # the legacy charge/budget


def test_p214_acceptance_is_STRICT_at_charge_equal_to_eps():
    """charge == eps proves the margin 0, i.e. nothing. Refused."""
    q = 1 - P214_EPS                           # charge = eps exactly
    ok, ratio, why, _shape = validate_octant(*_p214_octant(q))
    assert not ok and why == "budget_exceeded" and ratio == 1


def test_p214_the_note_half_inequality_is_reported_not_enforced():
    """A charge in (eps/2, eps) is ACCEPTED and its note flag is False; the
    verifier's job is to say so, not to refuse it."""
    from verify_certificate import NOTE_HALF_RATIO
    q = 1 - Fraction(3, 4) * P214_EPS
    ok, ratio, _why, _shape = validate_octant(*_p214_octant(q))
    assert ok and not (ratio <= NOTE_HALF_RATIO)
    q_ok = 1 - Fraction(1, 4) * P214_EPS
    ok2, ratio2, _w, _s = validate_octant(*_p214_octant(q_ok))
    assert ok2 and ratio2 <= NOTE_HALF_RATIO


def test_accepted_psd_shift_is_reported_zero_for_a_psd_gram():
    """The accepted shift is reported as a fourth shape entry: 0 when the Gram
    itself is PSD, the case of the paper's shift-free Gram lemma."""
    q = 1 - Fraction(1, 4) * P214_EPS
    ok, _ratio, _why, shape = validate_octant(*_p214_octant(q))
    assert ok and len(shape) == 4 and shape[3] == 0


def test_accepted_psd_shift_is_reported_when_the_ladder_needs_one():
    """A Gram with eigenvalue -1e-13 is accepted on the 1e-13 rung, and that
    rung is what the octant reports; the verdict is unchanged by reporting."""
    s = Fraction(1, 10 ** 13)
    target = {(2, 0): Fraction(1), (1, 1): Fraction(-2), (0, 2): Fraction(1)}
    gram = [[1, f"{-1 - s}"], [f"{-1 - s}", 1]]   # eigenvalues -s and 2 + s
    ok, ratio, why, shape = validate_octant(target, 2, gram, [(1, 0), (0, 1)],
                                            Fraction(1, 10 ** 6))
    assert ok and why == ""
    assert shape[3] == s
    assert ratio == (s * 1 + 2 * s) / Fraction(1, 10 ** 6)  # eta*C1 + ||E||_1


@needs_nominal_records
def test_p214_the_json_summary_records_the_policy_margin_and_note_check(tmp_path):
    """The verification JSON must carry, for a shipped record: the acceptance
    rule in force, the exact eps_certified, both ratios with their
    denominators named, and the note's eps/2 verdict per form and rolled up."""
    out = tmp_path / "summary.json"
    rec = NOMINAL_RECORDS / "sta_nominal.record.json.gz"
    res = subprocess.run([sys.executable, str(VERIFIER), str(rec),
                          "--json", str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "VERIFIED" and doc["margin_policy"] == "achieved"
    assert "charge < eps" in doc["acceptance_rule"]
    # eps_certified is exact and equals eps*(1 - worst charge/eps).
    eps = Fraction(doc["epsilon_exact"])
    cert = Fraction(doc["eps_certified_exact"])
    assert 0 < cert < eps
    # Both ratios, and the legacy one is exactly twice the new one.
    for form in ("V", "W", "overall"):
        assert doc["worst_charge_over_half_eps_legacy"][form] == \
            pytest.approx(2 * doc["worst_charge_over_eps"][form], rel=1e-12)
    # This record is far inside the note's inequality, on every octant.
    assert doc["note_half_margin_holds"] == {"V": True, "W": True,
                                             "overall": True}
    assert doc["note_half_margin_octants"]["ok"] == \
        doc["note_half_margin_octants"]["total"] == 8


@needs_nominal_records
def test_p214_the_legacy_ratio_reproduces_the_committed_P1_11_number():
    """Like-for-like against the pre-flip record: the WITNESS did not move, so
    the legacy charge/(eps/2) printed in
    artifacts/.../E3_portable/verification_2026-09-03/sta_nominal.blocked.verification.json
    must come back digit for digit."""
    stored = ROOT / "artifacts/campaign_2026-08/E3_portable" / \
        "verification_2026-09-03/sta_nominal.blocked.verification.json"
    if not stored.exists():
        pytest.skip("P1-11 verification record not present")
    ref = [Fraction(o["charge_over_budget_exact"])
           for o in json.loads(stored.read_text())["octants"]]
    rec, bundle = _load_nominal("sta_nominal")
    charges = [r[1] for r in _octant_charges(rec, bundle, blocked=True)]
    # _octant_charges returns charge/eps; the stored ones are charge/(eps/2).
    # Exact Fractions on both sides: the flip may not move a single digit.
    assert [2 * c for c in charges] == ref


# ---------------------------------------------------------------------------
# Regressions from the 2026-09-05 external scientific review
#
# Findings 2 (Laurent basis), 5 (coerced mu), 12 (blocked vs dense) and 13
# (unchecked kappa). Each test pins the reviewer's own instance, so a later
# "simplification" of the acceptance boundary has to break a named test.
# ---------------------------------------------------------------------------

REVIEW_V_TERMS = [(Fraction(-1), (Fraction(2), Fraction(0)), ("abs", "abs")),
                  (Fraction(4), (Fraction(0), Fraction(4)), ("abs", "abs"))]
REVIEW_MU = (4, 2)
REVIEW_GRAM = [["1/2", "-1"], ["-1", "3"]]


def _review_target():
    """The reviewer's octant target: V = -x1^2 + 4 x2^4, r = (2,1), m = 4."""
    return exact_octant_target(REVIEW_V_TERMS, REVIEW_MU, (1, 1), Fraction(1))


def test_review_finding2_laurent_basis_is_refused_by_validate_octant():
    """Regression from the 2026-09-05 external scientific review, finding 2.

    V = -x1^2 + 4 x2^4 is an admissible C^1 form and V(1, 0) = -1, so no
    certificate of positivity for it can be sound. With the LAURENT basis
    {(8, -4), (0, 4)} and the positive definite Gram [[1/2, -1], [-1, 3]],
    both paths used to accept a certified margin of 1/2: the residual
    -1/2 y1^16 y2^-8 is unbounded as y2 -> 0, and the coefficient 1-norm the
    gate charges bounds nothing there. Every exponent must be a nonnegative
    integer BEFORE it is read as one."""
    target, degree = _review_target()
    for blocked in (True, False):
        with pytest.raises(VerifyError) as exc:
            validate_octant(target, degree, REVIEW_GRAM, [(8, -4), (0, 4)],
                            Fraction(1), blocked=blocked)
        assert "violates >= 0" in str(exc.value)


def test_review_finding2_laurent_basis_is_refused_by_the_octant_sweep():
    """Regression from the 2026-09-05 external scientific review, finding 2.

    The same hole in the complete four-octant V check, which is what the
    reviewer's reproducer drives. It must refuse, not report a proof."""
    from itertools import product as iproduct
    captures = [{"mu": list(REVIEW_MU), "signs": list(s), "eps": "1",
                 "gram": REVIEW_GRAM, "basis": [(8, -4), (0, 4)]}
                for s in iproduct((-1, 1), repeat=2)]
    with pytest.raises(VerifyError):
        verify_captures(REVIEW_V_TERMS, captures, 2, "V", Fraction(1))


def test_review_finding2_basis_shape_dimension_and_duplicates_are_refused():
    """Regression from the 2026-09-05 external scientific review, finding 2.

    The other ways a basis can fail to be what the bounds assume: a wrong
    dimension, a non-integer exponent, a repeated vector, an empty list."""
    target, degree = _review_target()
    for basis, needle in (
            ([(4,), (0,)], "entries, need 2"),          # wrong dimension
            ([(3.5, 0.5), (0, 4)], "not an integer"),   # fractional exponent
            ([(0, 4), (0, 4)], "duplicate"),            # repeated vector
            ([], "empty basis"),                        # nothing at all
            ([(8, 0), (0, 8)], "need d/2"),             # wrong degree
    ):
        with pytest.raises(VerifyError) as exc:
            validate_octant(target, degree, [[1, 0], [0, 1]][:len(basis) or 1],
                            basis, Fraction(1))
        assert needle in str(exc.value), (basis, str(exc.value))


def test_review_finding2_non_finite_gram_entries_are_refused():
    """Regression from the 2026-09-05 external scientific review, finding 2.

    A NaN or infinity in a Gram would poison every exact comparison; it is a
    malformed capture, not a number to coerce."""
    target, degree = _review_target()
    for bad in (float("nan"), float("inf"), "nan", None, [1]):
        with pytest.raises(VerifyError):
            validate_octant(target, degree, [[bad, 0], [0, 1]],
                            [(8, 0), (0, 8)], Fraction(1))


def test_review_finding5_mu_must_be_positive_integers_of_the_right_arity():
    """Regression from the 2026-09-05 external scientific review, finding 5.

    The lift was coerced with int(), so a declared mu of 2.9 was read as 2 and
    the arithmetic proved a statement about a substitution the record does not
    declare. Fractional, zero, negative and wrong-arity lifts must all be
    refused on the RAW value."""
    for mu, needle in (((4, 2.9), "not an integer"),
                       ((4, 0), "violates >= 1"),
                       ((4, -2), "violates >= 1"),
                       ((4,), "entries, need 1"),
                       ((4, 2, 1), "entries, need 3"),
                       ((4, True), "boolean")):
        with pytest.raises(VerifyError) as exc:
            exact_octant_target(REVIEW_V_TERMS, mu, (1,) * len(mu), Fraction(1))
        assert needle in str(exc.value), (mu, str(exc.value))


def test_review_finding5_every_capture_of_one_form_declares_the_same_mu():
    """Regression from the 2026-09-05 external scientific review, finding 5.

    One form is proved under ONE association. If the 2^n captures declared
    different lifts they would be charts of different substitutions and would
    not cover R^n between them, however complete their sign patterns look."""
    from itertools import product as iproduct
    captures = [{"mu": [4, 2] if i else [8, 4], "signs": list(s), "eps": "1",
                 "gram": REVIEW_GRAM, "basis": [(4, 0), (0, 4)]}
                for i, s in enumerate(iproduct((-1, 1), repeat=2))]
    with pytest.raises(VerifyError) as exc:
        verify_captures(REVIEW_V_TERMS, captures, 2, "V", Fraction(1))
    assert "different lifts mu" in str(exc.value)


def test_review_finding12_blocked_may_accept_where_dense_rejects():
    """Regression from the 2026-09-05 external scientific review, finding 12.

    The reviewer's instance, pinned so nobody "fixes" a sound difference into
    a false equality. F = y1^2 + y2^2 with eps = 1 gives target 0; on basis
    {(1,0),(0,1)} with Q = [[0, 2/5], [2/5, 0]] the blocked path accepts at
    charge 4/5 (the projected Gram is zero, PSD at eta = 0) while the dense
    path needs the Gershgorin rung eta = 2/5 and rejects at charge 6/5.

    Both verdicts are sound: projection cannot lower the smallest eigenvalue
    (so the blocked test is the weaker one), and the residual is charged
    densely in both paths (so the dense arm over-charges what the projection
    already dropped). The true implication is one-directional: a dense PASS
    implies the blocked PASS at the same charge, never the converse."""
    target = {(2, 0): Fraction(0), (0, 2): Fraction(0)}
    basis = [(1, 0), (0, 1)]
    gram = [[0, "2/5"], ["2/5", 0]]
    ok_b, ratio_b, why_b, _ = validate_octant(target, 2, gram, basis,
                                              Fraction(1), blocked=True)
    ok_d, ratio_d, why_d, _ = validate_octant(target, 2, gram, basis,
                                              Fraction(1), blocked=False)
    assert (ok_b, ratio_b, why_b) == (True, Fraction(4, 5), "")
    assert (ok_d, ratio_d, why_d) == (False, Fraction(6, 5), "budget_exceeded")


@needs_nominal_records
def test_review_finding13_a_tampered_kappa_is_refused(tmp_path):
    """Regression from the 2026-09-05 external scientific review, finding 13.

    The FOSMC record with kappa changed from -1 to 999 used to verify with
    exit code 0: the field was carried, never checked. kappa is now DERIVED
    from the dynamics this file rebuilds (every term of f_i has weighted
    degree r_i + kappa) and a declaration that disagrees is a refusal."""
    rec, bundle = _load_nominal("fosmc_nominal")
    rec["kappa"] = "999"
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 1, res.stdout
    assert "contradicts the weighted homogeneity degree -1" in res.stderr


@needs_nominal_records
def test_review_finding13_the_derived_kappa_is_printed_and_recorded(tmp_path):
    """Regression from the 2026-09-05 external scientific review, finding 13.

    The verified quantity must be visible: the summary line states the kappa
    derived from the rebuilt dynamics, and --json records it."""
    out = tmp_path / "summary.json"
    rec = NOMINAL_RECORDS / "fosmc_nominal.record.json.gz"
    res = subprocess.run([sys.executable, str(VERIFIER), str(rec),
                          "--json", str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "kappa = -1 (derived from the rebuilt dynamics)" in res.stdout
    assert json.loads(out.read_text())["kappa_derived"] == "-1/1"


@needs_nominal_records
def test_review_finding13_record_flags_must_be_json_typed(tmp_path):
    """Regression from the 2026-09-05 external scientific review, finding 13.

    ``nominal`` decides which record SHAPE is proved, so a truthy string must
    not become True via bool(); the channel must be an integer or null."""
    rec, bundle = _load_nominal("fosmc_nominal")
    rec["nominal"] = "yes"
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2 and "must be a JSON boolean" in res.stderr

    rec2, bundle2 = _load_nominal("fosmc_nominal")
    rec2["channel"] = "0"
    res2 = _run_tampered(tmp_path, rec2, bundle2)
    assert res2.returncode == 2 and "channel" in res2.stderr


@needs_nominal_records
def test_review_finding13_v_terms_of_the_wrong_shape_are_refused(tmp_path):
    """Regression from the 2026-09-05 external scientific review, finding 13.

    Strict schema on the fields the proof reads: a V term with the wrong
    number of exponents, or a factor type that is neither 'abs' nor 'signed',
    is a refusal rather than something the term algebra guesses at."""
    rec, bundle = _load_nominal()
    rec["V"]["terms"][0]["exponents"] = rec["V"]["terms"][0]["exponents"][:-1]
    res = _run_tampered(tmp_path, rec, bundle)
    assert res.returncode == 2 and "exponents" in res.stderr

    rec2, bundle2 = _load_nominal()
    rec2["V"]["terms"][0]["term_types"] = \
        ["weird"] * len(rec2["V"]["terms"][0]["term_types"])
    res2 = _run_tampered(tmp_path, rec2, bundle2)
    assert res2.returncode == 2 and "neither 'abs' nor 'signed'" in res2.stderr


@needs_portable
@pytest.mark.parametrize("system", ["sta", "pidsmc"])
def test_the_stored_grams_are_psd_which_is_why_both_paths_agree(system):
    """Regression from the 2026-09-05 external scientific review, finding 12.

    Pins the actual precondition of the "same verdict on real captures" tests,
    because the file used to state a different (and false) one. Measured on
    these bundles: the cross-parity entries of the captured Gram are NOT zero,
    so Q_H differs from Q; what makes the two paths agree is that Q itself is
    PSD, so the eta ladder stops at its first rung in both and the charge is
    the (densely charged, hence identical) residual alone."""
    from verify_certificate import blocks_psd
    rec, bundle = _load_portable(system)
    eps = q_exact(rec["epsilon_exact"], "epsilon_exact")
    v_terms = _v_terms(rec)
    for cap in bundle["V_grams"]:
        target, _d = exact_octant_target(
            v_terms, tuple(cap["mu"]), tuple(cap["signs"]), eps)
        basis = [tuple(int(b) for b in beta) for beta in cap["basis"]]
        n = len(basis)
        Q = [[(q_from(cap["gram"][i][j]) + q_from(cap["gram"][j][i])) / 2
              for j in range(n)] for i in range(n)]
        assert blocks_psd(Q, [list(range(n))], Fraction(0)), \
            "the captured Gram is PSD; that is why eta = 0 passes both paths"
