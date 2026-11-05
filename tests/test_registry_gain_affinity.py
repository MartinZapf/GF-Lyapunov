"""Item C2 (2026-09-28): the exact affinity probe for registry gain points.

The portable verifier (VC) covers a bracketed gain's interval by its two
ends; that rests on the dynamics being affine in the gain, which VC states
but does not check in code. The record writer
(``scripts/emit_certificate_record.py``, EM) already checked this for a gain
point read from an exact record (``--gain-point exact-record``): the gain must
be a declared affine coordinate (guard R6, ``scripts/gain_region_fixedV.py``)
and VC's own builder must be affine in it on the interval (same support at
lo, midpoint and hi; midpoint coefficients the average, exactly). C2 runs
the same check for a gain point taken from the printed-gain registry
(``--gain-point flagship|benchmark``, also scaled to a representative p)
and refuses to write a record when it fails.

What this file pins, MOSEK-free and without any solve:

  1. every registered point passes, with one report per bracketed gain, and
     a point without brackets needs nothing;
  2. a representative-p point is probed at its scaled brackets;
  3. non-affine cases are refused: the power-rate exponent alpha (not a
     declared coordinate; and VC's builder changes support in it), and a
     synthetic builder whose coefficient is quadratic in the gain;
  4. EM's command line runs the check on the registry route and refuses
     before the exact record is read or any record file is written.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import emit_certificate_record as em  # noqa: E402
from gf_framework.systems import printed_gains as pg  # noqa: E402

REGISTRY = [(table_name, system, point)
            for table_name, table in (("flagship", pg.FLAGSHIP_GAIN_POINTS),
                                      ("benchmark", pg.BENCHMARK_GAIN_POINTS))
            for system, point in sorted(table.items())]


# ---------------------------------------------------------------------------
# 1-2. Registered points pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table_name, system, point", REGISTRY,
                         ids=[f"{t}-{s}" for t, s, _ in REGISTRY])
def test_every_registry_point_passes(table_name, system, point):
    reports = em.check_registry_point_affinity(point, system)
    bracketed = sorted(pg.brackets(point))
    assert [r["gain"] for r in reports] == bracketed
    for report in reports:
        lo, hi = pg.brackets(point)[report["gain"]]
        assert report["interval"] == [str(lo), str(hi)]
        assert report["declared_coordinate"] == report["gain"]


def test_the_bracketed_registry_points_are_probed():
    """The check is not vacuous: the flagship differentiators have brackets."""
    probed = {(table_name, system):
              len(em.check_registry_point_affinity(point, system))
              for table_name, system, point in REGISTRY}
    assert probed[("flagship", "levant3d")] == 1
    assert probed[("flagship", "levant5d")] == 3
    assert sum(probed.values()) >= 10


def test_representative_p_point_is_probed_at_its_scaled_brackets():
    base = pg.FLAGSHIP_GAIN_POINTS["levant4d"]
    point = em.ScaledGainPoint(base, Fraction(1, 2))
    reports = em.check_registry_point_affinity(point, "levant4d")
    assert [r["gain"] for r in reports] == ["lambda2", "lambda3"]
    for report in reports:
        lo, hi = point.brackets()[report["gain"]]
        assert report["interval"] == [str(lo), str(hi)]
        assert (lo, hi) != pg.brackets(base)[report["gain"]]


# ---------------------------------------------------------------------------
# 3. Non-affine cases are refused
# ---------------------------------------------------------------------------

def _power_rate_with_alpha_bracket():
    """A synthetic registry-shaped point: power-rate with alpha bracketed."""
    template = pg.BENCHMARK_GAIN_POINTS["power_rate"]
    gains = dict(template.gains)
    gains["alpha"] = pg.BracketedGain(
        power_k=2, power_target=Fraction(1, 3), approx=3 ** -0.5,
        provenance="synthetic test point (C2): alpha = 3^(-1/2), an exponent",
        role=template.gains["alpha"].role)
    return pg.GainPoint(system="power_rate", factory=template.factory,
                        module=template.module, gains=gains,
                        channel=template.channel, note="synthetic (C2 test)")


def test_power_rate_alpha_bracket_is_refused():
    point = _power_rate_with_alpha_bracket()
    assert point.bracketed_names == ("alpha",)
    with pytest.raises(em.Refusal) as exc:
        em.check_registry_point_affinity(point, "power_rate")
    assert exc.value.reason == "unverifiable_gain_point"
    assert "not a declared affine gain coordinate" in exc.value.detail


def test_the_exact_probe_alone_refuses_the_power_rate_exponent():
    """VC's own builder: alpha is an exponent, so the support changes."""
    point = _power_rate_with_alpha_bracket()
    with pytest.raises(em.Refusal) as exc:
        em.probe_builder_affinity("power_rate", em.point_rational_gains(point),
                                  em.point_brackets(point), "alpha")
    assert exc.value.reason == "unverifiable_gain_point"
    assert "not affine in alpha" in exc.value.detail


def _levant3d_builder(square_lambda2):
    """VC's levant3d builder, optionally with lambda2 squared (synthetic)."""
    real = em._vc_build_system_dynamics()

    def build(system, gains, delta0, endpoint_sign):
        gains = dict(gains)
        if square_lambda2:
            gains["lambda2"] = gains["lambda2"] ** 2
        return real(system, gains, delta0, endpoint_sign)

    return build


def test_synthetic_affine_builder_passes():
    point = pg.FLAGSHIP_GAIN_POINTS["levant3d"]
    reports = em.check_registry_point_affinity(
        point, "levant3d", build=_levant3d_builder(square_lambda2=False))
    assert [r["gain"] for r in reports] == ["lambda2"]


def test_synthetic_quadratic_builder_is_refused():
    point = pg.FLAGSHIP_GAIN_POINTS["levant3d"]
    with pytest.raises(em.Refusal) as exc:
        em.check_registry_point_affinity(
            point, "levant3d", build=_levant3d_builder(square_lambda2=True))
    assert exc.value.reason == "unverifiable_gain_point"
    assert "not affine in lambda2" in exc.value.detail


def test_a_builder_that_changes_shape_is_refused():
    def build(system, gains, delta0, endpoint_sign):
        # one term at the lower end, two elsewhere
        extra = [] if gains["lambda2"] == Fraction(2) else [
            (Fraction(1), (Fraction(0),) * 3, ("signed",) * 3)]
        return [[(Fraction(1), (Fraction(1),) * 3, ("signed",) * 3)] + extra]

    with pytest.raises(em.Refusal) as exc:
        em.probe_builder_affinity("levant3d", {}, {"lambda2": (2, 3)},
                                  "lambda2", build=build)
    assert "change shape" in exc.value.detail


def test_a_builder_error_is_a_typed_refusal():
    def build(system, gains, delta0, endpoint_sign):
        raise ValueError("builder rejects the gains")

    with pytest.raises(em.Refusal) as exc:
        em.probe_builder_affinity("levant3d", {}, {"lambda2": (2, 3)},
                                  "lambda2", build=build)
    assert exc.value.reason == "unverifiable_gain_point"
    assert "builder refused" in exc.value.detail


def test_exact_record_route_uses_the_same_check():
    """The shared helper keeps the exact-record refusal text unchanged."""
    with pytest.raises(em.Refusal) as exc:
        em.check_varying_gain_affinity(
            "power_rate", {"k": Fraction(1)},
            {"alpha": (Fraction(1, 2), Fraction(3, 4))}, "alpha")
    assert "affine" in exc.value.detail
    report = em.check_varying_gain_affinity(
        "twisting", {"r2": Fraction(3, 2)},
        {"r1": (Fraction(2), Fraction(21, 10))}, "r1")
    assert report["gain"] == "r1"


# ---------------------------------------------------------------------------
# 4. EM's command line: the registry route refuses before writing anything
# ---------------------------------------------------------------------------

def test_cli_registry_route_refuses_a_non_affine_builder(tmp_path, monkeypatch,
                                                          capsys):
    row = {"system": "levant3d", "weights_r": [3, 2, 1], "V_terms": [],
           "delta0": 1}
    row_path = tmp_path / "row.json"
    row_path.write_text(json.dumps(row))
    missing_exact = tmp_path / "never_read_exact.json"
    out = tmp_path / "out"
    quadratic = _levant3d_builder(square_lambda2=True)
    monkeypatch.setattr(em, "_vc_build_system_dynamics", lambda: quadratic)
    monkeypatch.setattr(sys, "argv", [
        "emit_certificate_record.py", "--row", str(row_path),
        "--exact", str(missing_exact), "--gain-point", "flagship",
        "--out-dir", str(out)])
    assert em.main() == 3
    refusal = json.loads((out / "levant3d.refusal.json").read_text())
    assert refusal["reason"] == "unverifiable_gain_point"
    assert "not affine in lambda2" in refusal["detail"]
    assert not list(out.glob("*.record.json.gz"))
    assert not list(out.glob("*.grams.json.gz"))
    assert not missing_exact.exists()
    capsys.readouterr()
