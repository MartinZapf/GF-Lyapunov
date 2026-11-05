"""Guard R6 declaration for ``levant4d_2003print`` (X9 addendum 2026-09-26).

X10 registered the key ``levant4d_2003print`` (Levant's third-order
differentiator at the recursive tuning (1.1, 1.5, 3, 5); the SAME equations
as ``levant4d``) in the printed-gain registry, VC ``SYSTEMS`` and EM
``COVARIANT_SCALING``, but not in ``scripts/gain_region_fixedV.py::SYSTEMS``.
The TB driver and EM's exact-record route therefore refused every bracketed
gain of it (``non_affine_parameter``), and the X9 addendum ran tier B through
a run-time shim (``artifacts/campaign_2026-09/X9/printed_2026-09-26/
grf_shim.py``). The permanent declaration replaces the shim. Pinned here:

  1. the entry is levant4d's (same coordinates, builder, weights, channel)
     with the key of the flagship registry point, the shape of
     ``levant3d_2003print``;
  2. the symbolic affinity assertion passes for it, nominal and perturbed;
  3. the TB driver accepts the addendum's bracketed gain point without the
     shim (``assert_affinity`` and ``--check-only``);
  4. EM's exact-record route accepts the addendum's tier-B exact records;
  5. the shim, when present, reports itself inactive and changes nothing.

CLARABEL/MOSEK-free: no solve runs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "verification" / "standalone"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gain_region_fixedV as grf  # noqa: E402

KEY = "levant4d_2003print"
X9A = ROOT / "artifacts" / "campaign_2026-09" / "X9" / "printed_2026-09-26"
SHIM = X9A / "grf_shim.py"
EXACT_DOCS = {
    "p1_2": X9A / "tierB" / "c9.3_x10_levant4d_2003print_p1_2" / "b15" / "run"
    / "m9" / "exact.json",
    "p3_4": X9A / "tierB" / "c9.3_x10_levant4d_2003print_p3_4" / "b14" / "run"
    / "m9" / "exact.json",
}

#: The addendum's tier-B gain point at p = 1/2 (bNN/run.cmd.txt), verbatim.
GAINS = ["lambda1=5/2,lambda4=11/160"]
BRACKETS = [
    "lambda2:4386026607319299/2000000000000000:8772053214638599/4000000000000000:3:675/64",
    "lambda3:1110661957261773/2000000000000000:4442647829047093/8000000000000000"
    ":6:492075/16777216",
]
DELTA0 = "1/16"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load("certify_gain_point", ROOT / "scripts" / "certify_gain_point.py")


def test_declaration_is_levant4d_with_the_flagship_key():
    decl, base = grf.SYSTEMS[KEY], grf.SYSTEMS["levant4d"]
    assert decl.system == decl.registry_key == KEY
    assert decl.registry_flagship is True
    assert decl.coords == base.coords == ("lambda1", "lambda2", "lambda3", "lambda4")
    assert decl.build is base.build
    assert decl.weights_r == base.weights_r and decl.channel == base.channel
    assert decl.center_from_registry is None
    # same shape as the existing 2003-print entry of the order-2 differentiator
    ref = grf.SYSTEMS["levant3d_2003print"]
    assert ref.build is grf.SYSTEMS["levant3d"].build and ref.registry_flagship
    # the centre is read from the flagship registry point (1.1, 1.5, 3, 5)
    center, prov = grf.registry_center(decl)
    assert prov["registry_table"] == "flagship" and prov["registry_key"] == KEY
    assert center["lambda1"] == pytest.approx(5.0)
    assert center["lambda4"] == pytest.approx(1.1)
    assert center["lambda2"] == pytest.approx(3 * 5 ** (2 / 3))
    assert set(prov["brackets"]) == {"lambda2", "lambda3"}


@pytest.mark.parametrize("delta0", [0.0, 1 / 16])
def test_symbolic_affinity_passes(delta0):
    decl = grf.SYSTEMS[KEY]
    report = grf.assert_gain_affinity(decl, delta0=delta0)
    assert report["W_linear_in_each_declared_gain"] is True
    assert report["no_gain_gain_products_in_W"] is True
    assert report["each_gain_multiplies_exactly_one_term"] is True
    assert report["coordinates"] == list(decl.coords)
    assert set(report["per_gain"]) == set(decl.coords)
    # identical to levant4d's report: the same builder sees the same equations
    base = grf.assert_gain_affinity(grf.SYSTEMS["levant4d"], delta0=delta0)
    assert report["per_gain"] == base["per_gain"]


def test_driver_accepts_the_bracketed_point_without_the_shim(tmp_path):
    spec = tb.build_gain_spec(KEY, GAINS, brackets=BRACKETS, delta0=DELTA0)
    tb.check_gain_names(spec)
    report = tb.assert_affinity(spec)
    assert report["required"] is True
    assert sorted(report["varying"]) == ["lambda2", "lambda3"]
    assert report["source"] == "scripts/gain_region_fixedV.py::assert_gain_affinity"
    argv = ["--system", KEY, "--gains", GAINS[0], "--delta0", DELTA0,
            "--degrees", "9", "--check-only", "--out-dir", str(tmp_path)]
    for b in BRACKETS:
        argv += ["--bracket", b]
    assert tb.main(argv) == tb.EXIT_OK
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["guard_r6"]["required"] is True
    assert plan["dynamics"]["n"] == 8          # 4 bracket corners x 2 endpoints


@pytest.mark.parametrize("which", sorted(EXACT_DOCS))
def test_em_exact_record_route_accepts_the_addendum_records(which):
    path = EXACT_DOCS[which]
    if not path.exists():
        pytest.skip(f"{path} absent")
    import emit_certificate_record as em  # noqa: PLC0415
    from verify_certificate import SYSTEMS as VC_SYSTEMS  # noqa: PLC0415

    doc = json.loads(path.read_text())
    assert doc["system"] == KEY
    point = em.ExactRecordGainPoint(doc, KEY, VC_SYSTEMS[KEY])
    assert point.bracketed_names == ("lambda2", "lambda3")
    assert all(isinstance(v, Fraction) for v in point.rational_gains().values())


def test_shim_is_inactive_now(capsys):
    if not SHIM.exists():
        pytest.skip("X9 addendum shim absent")
    shim = _load("x9_addendum_grf_shim", SHIM)
    before = grf.SYSTEMS[KEY]
    shim.install()
    assert "shim inactive" in capsys.readouterr().err
    assert grf.SYSTEMS[KEY] is before
