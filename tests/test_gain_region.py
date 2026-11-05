"""Tests for scripts/gain_region_fixedV.py (C3, campaign block E9).

Everything here runs under CLARABEL on the smallest published certificate
(the June matrix STA row: m = 3, three terms), so the whole file stays in the
seconds range and needs no MOSEK.

What is pinned:
  * the symbolic affinity assertion (guard R6) ACCEPTS the declared
    k-coordinates of sta / levant3d / pidsmc, and REFUSES, with a typed
    refusal, three deliberately wrong declarations: PID-SMC in
    (lambda, Delta) coordinates (the R6 trap itself), a gain product, and one
    gain multiplying two dynamics terms;
  * a small nominal box around the STA tuning certifies at all four vertices;
  * a box with one deliberately underpowered vertex (k2 far too small) yields
    a typed non-certified cell and a not-certified box verdict;
  * --max-delta0 bisects a value in (0, 2) and checks BOTH inclusion endpoints
    at every positive trial (asserted from the record's own cell counts);
  * the emitted artifact carries the campaign provenance block;
  * the CLI parses, refuses MOSEK_DIRECT, and records the refusal.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gain_region_fixedV as gr  # noqa: E402  (path insert above is required)

MATRIX = REPO_ROOT / "artifacts/publication/results_gf_sos_clean/gf_matrix_20260610_230219.json"

# Windows console cp1252 cannot encode every character in the repo's help
# texts; the subprocess smokes force UTF-8 exactly as test_c19_drivers.py does.
_ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def _cvxpy_available() -> bool:
    try:
        import cvxpy  # noqa: F401
        return True
    except ImportError:
        return False


needs_matrix = pytest.mark.skipif(not MATRIX.exists(),
                                  reason="June matrix artifact absent")
needs_solver = pytest.mark.skipif(not _cvxpy_available(),
                                  reason="cvxpy/CLARABEL not installed")


# ---------------------------------------------------------------------------
# Guard R6: the symbolic affinity assertion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", ["sta", "levant3d", "pidsmc"])
def test_affinity_accepts_declared_k_coordinates(system):
    decl = gr.SYSTEMS[system]
    report = gr.assert_gain_affinity(decl, delta0=0.0)
    assert report["W_linear_in_each_declared_gain"] is True
    assert report["no_gain_gain_products_in_W"] is True
    assert report["each_gain_multiplies_exactly_one_term"] is True
    assert report["coordinates"] == list(decl.coords)
    # every declared gain was located in exactly one dynamics term
    assert set(report["per_gain"]) == set(decl.coords)
    for name, entry in report["per_gain"].items():
        assert entry["d_coefficient_d_gain"] != 0.0


def test_affinity_checks_both_endpoints_when_perturbed():
    report = gr.assert_gain_affinity(gr.SYSTEMS["sta"], delta0=1.0)
    assert report["checked_dynamics"] == ["endpoint_plus", "endpoint_minus"]


def test_affinity_refuses_pidsmc_lambda_delta_declaration():
    """The R6 trap verbatim: k1 = lambda1.Delta^(2/3) is not affine in
    (lambda, Delta) jointly, so declaring those as the gain coordinates must
    abort the system instead of certifying a box in the wrong coordinates."""

    def build(v):
        from gf_framework.systems.pidsmc import get_pidsmc_dynamics_gf
        return get_pidsmc_dynamics_gf(v["lambda1"], v["lambda2"], v["lambda3"],
                                      delta=v["delta"])

    wrong = gr.GainCoordinates(
        system="pidsmc_lambda_delta",
        coords=("lambda1", "lambda2", "lambda3", "delta"),
        build=build, weights_r=(3, 2, 1), channel=2, registry_key="pidsmc",
        note="deliberately wrong declaration (R6 negative control)",
    )
    with pytest.raises(gr.GainRegionRefusal) as exc:
        gr.assert_gain_affinity(wrong, delta0=0.0)
    assert exc.value.kind == "affinity_refused"
    assert "not polynomial" in str(exc.value) or "total degree" in str(exc.value)


def test_affinity_refuses_gain_product():
    def build(v):
        from gf_framework.systems.sta import get_sta_dynamics_gf
        return get_sta_dynamics_gf(v["k1"] * v["k2"], v["k2"])

    wrong = gr.GainCoordinates(
        system="sta_product", coords=("k1", "k2"), build=build,
        weights_r=(2, 1), channel=1, registry_key="sta",
        note="deliberately wrong declaration: k1.k2 in one coefficient",
    )
    with pytest.raises(gr.GainRegionRefusal) as exc:
        gr.assert_gain_affinity(wrong, delta0=0.0)
    assert exc.value.kind == "affinity_refused"
    assert "total degree" in str(exc.value)


def test_affinity_refuses_gain_in_two_terms():
    def build(v):
        from gf_framework.systems.dsta import get_dsta_dynamics_gf
        return get_dsta_dynamics_gf(v["k1"], v["k"], v["k"])

    wrong = gr.GainCoordinates(
        system="dsta_shared", coords=("k1", "k"), build=build,
        weights_r=(2, 1), channel=1, registry_key="dsta",
        note="deliberately wrong declaration: one gain on two terms",
    )
    with pytest.raises(gr.GainRegionRefusal) as exc:
        gr.assert_gain_affinity(wrong, delta0=0.0)
    assert exc.value.kind == "affinity_refused"
    assert "exactly one" in str(exc.value)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_every_registered_system_declares_affine_coordinates():
    for name, decl in gr.SYSTEMS.items():
        report = gr.assert_gain_affinity(decl, delta0=0.0)
        assert report["coordinates"] == list(decl.coords), name
        center, prov = gr.registry_center(decl)
        assert set(center) == set(decl.coords), name
        assert prov["registry_key"] == decl.registry_key


def test_pidsmc_center_is_stated_in_k_coordinates():
    center, prov = gr.registry_center(gr.SYSTEMS["pidsmc"])
    assert set(center) == {"k1", "k2", "k3"}
    assert prov["converted_to_k_coordinates"] is True
    # Delta = 1 in the registry, so k equals lambda numerically there.
    assert center["k1"] == pytest.approx(2.7)
    assert center["k3"] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Boxes (CLARABEL, June matrix STA V)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sta_V():
    V, meta = gr.load_V("matrix", MATRIX, "sta")
    return V, meta


def _box_ctx(V, box):
    decl = gr.SYSTEMS["sta"]
    center = {"k1": 1.5, "k2": 1.1}
    return dict(V=V, decl=decl, box=box, center=center, delta0=0.0,
                solver="CLARABEL", epsilon=gr.DEFAULT_EPSILON, cell_timeout=None,
                center_dyn=decl.build(center), grow_max=8.0, grow_tol=0.05,
                delta0_hi=2.0, delta0_tol=0.005)


@needs_matrix
@needs_solver
def test_tiny_nominal_box_certifies_at_all_four_vertices(sta_V):
    V, meta = sta_V
    assert meta["n_terms"] == 3 and V.degree_m == 3
    out = gr.run_box(_box_ctx(V, {"k1": (1.45, 1.55), "k2": (1.0, 1.1)}))
    res = out["result"]
    assert res["n_vertices"] == 4
    assert res["n_cells"] == 4  # nominal: one endpoint per vertex
    assert res["certified"] is True
    assert [c["status"] for c in res["cells"]] == ["certified"] * 4
    assert res["status_counts"]["certified"] == 4


@needs_matrix
@needs_solver
def test_box_with_underpowered_vertex_is_not_certified(sta_V):
    """k2 far too small: those vertices must come back with a typed
    non-certified status, and the box verdict must be not certified."""
    V, _ = sta_V
    out = gr.run_box(_box_ctx(V, {"k1": (1.45, 1.55), "k2": (0.05, 1.1)}))
    res = out["result"]
    assert res["certified"] is False
    bad = [c for c in res["cells"] if c["gains"]["k2"] == 0.05]
    assert len(bad) == 2
    for cell in bad:
        assert cell["status"] in gr.TYPED_STATUSES
        assert cell["status"] != "certified"
    good = [c for c in res["cells"] if c["gains"]["k2"] == 1.1]
    assert all(c["status"] == "certified" for c in good)


@needs_matrix
@needs_solver
def test_positive_template_reports_the_tuning_in_the_interior(sta_V):
    """The block's pre-registered positive sentence, filled from the data."""
    V, _ = sta_V
    out = gr.run_box(_box_ctx(V, {"k1": (1.4, 1.6), "k2": (1.0, 1.15)}))
    assert out["result"]["certified"] is True
    assert out["result"]["center_position"] == "interior"
    payload = {
        "V_source": {"V_id": "V[test]"}, "delta0": 0.0, "mode": "box",
        "affinity": {"coordinates": ["k1", "k2"], "coordinate_note": "n/a"},
        "result": out["result"], "center": {"k1": 1.5, "k2": 1.1},
    }
    text = " ".join(gr.statements(payload))
    assert "is certified at delta0 = 0" in text
    assert "lies in its interior" in text


def test_inconclusive_cells_never_become_a_claim_about_the_gains():
    """A timeout / oom / unexpected_error cell is a statement about the run,
    so the emitted sentence must not turn it into a limit of the region."""
    payload = {
        "V_source": {"V_id": "V[test]"}, "delta0": 1.0, "mode": "box",
        "affinity": {"coordinates": ["k1", "k2"], "coordinate_note": "n/a"},
        "center": {"k1": 1.5, "k2": 1.1},
        "result": {
            "certified": False, "n_cells": 4,
            "box": {"k1": [1.4, 1.6], "k2": [1.0, 1.2]},
            "rel_halfwidths": {"k1": 0.0667, "k2": 0.0909},
            "status_counts": {"certified": 2, "solver_infeasible": 0,
                              "solver_unknown": 0, "timeout": 2, "oom": 0,
                              "unexpected_error": 0},
        },
    }
    text = " ".join(gr.statements(payload))
    assert "did not reach a conclusion" in text
    assert "not about the gains" in text
    assert "Transfer of this fixed V across gains is limited" not in text


@needs_matrix
@needs_solver
def test_all_cell_statuses_are_in_the_typed_vocabulary(sta_V):
    V, _ = sta_V
    out = gr.run_box(_box_ctx(V, {"k1": (1.45, 1.55), "k2": (0.05, 1.1)}))
    for cell in out["result"]["cells"]:
        assert cell["status"] in gr.TYPED_STATUSES
        assert cell["message"]
        assert "vertex_index" in cell and "endpoint" in cell


# ---------------------------------------------------------------------------
# --max-delta0 (E9.e)
# ---------------------------------------------------------------------------

@needs_matrix
@needs_solver
def test_max_delta0_bisects_and_checks_both_endpoints(sta_V):
    V, _ = sta_V
    ctx = _box_ctx(V, None)
    out = gr.run_max_delta0(ctx)
    res = out["result"]
    assert res["delta0_max"] is not None
    assert 0.0 < res["delta0_max"] < 2.0
    assert res["delta0_first_failed"] > res["delta0_max"]
    assert res["delta0_first_failed"] - res["delta0_max"] <= res["tol"] + 1e-12
    # Both inclusion endpoints at EVERY positive trial: one gain vertex, two
    # cells. The nominal delta0 = 0 trial has a single dynamics, by definition.
    assert res["n_positive_delta0_trials"] >= 1
    assert res["cells_per_positive_delta0_trial"] == [2]
    assert res["n_endpoint_cells"] == 1 + 2 * res["n_positive_delta0_trials"]
    for trial in res["trials"]:
        if trial["delta0"] > 0:
            assert {c["endpoint"] for c in trial["cells"]} == {"plus", "minus"}
            assert sorted(c["endpoint_index"] for c in trial["cells"]) == [0, 1]


# ---------------------------------------------------------------------------
# End to end: artifact, provenance, statements
# ---------------------------------------------------------------------------

@needs_matrix
@needs_solver
def test_end_to_end_box_run_writes_provenance(tmp_path):
    rc = gr.main([
        "--system", "sta", "--v-from", "matrix", str(MATRIX),
        "--output-dir", str(tmp_path), "--solver", "CLARABEL",
        "--box", "k1:1.45:1.55,k2:1.0:1.1",
    ])
    assert rc == 0
    files = sorted(tmp_path.glob("E9_gain_region_box_sta_*.json"))
    assert len(files) == 1
    assert sorted(tmp_path.glob("E9_gain_region_box_sta_*.md"))
    payload = json.loads(files[0].read_text())

    prov = payload["provenance"]
    assert prov["git_sha"] and prov["command_line"]
    assert prov["solver_name"] == "CLARABEL" and prov["solver_version"]
    assert prov["env"]["python_version"]
    assert payload["block"] == "E9"
    assert payload["coordinate_system"].startswith("k-coordinates")
    assert payload["result"]["certified"] is True
    assert payload["v_stage"]["status"] == "certified"
    assert payload["verification_entry_point"].startswith(
        "gf_framework.verification.joint_sos.verify_w_given_numeric")
    # Claim hygiene: no em dashes anywhere in the emitted text.
    assert all("—" not in s for s in payload["statements"])
    assert any("k-coordinates" in s for s in payload["statements"])


@needs_matrix
def test_dry_run_asserts_affinity_without_solving(tmp_path):
    rc = gr.main([
        "--system", "levant3d", "--v-from", "matrix", str(MATRIX),
        "--output-dir", str(tmp_path), "--rel-halfwidth", "0.05", "--dry-run",
    ])
    assert rc == 0
    assert not list(tmp_path.glob("*.json"))  # a dry run writes nothing


# ---------------------------------------------------------------------------
# CLI smokes
# ---------------------------------------------------------------------------

def test_help_smoke_lists_the_flags():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "gain_region_fixedV.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_ENV, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ["--v-from", "--output-dir", "--box", "--grow", "--max-delta0",
                 "--delta0", "--solver", "--cell-timeout"]:
        assert flag in proc.stdout, f"--help does not mention {flag}"
    assert "MOSEK_DIRECT" in proc.stdout


def test_output_dir_is_required():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "gain_region_fixedV.py"),
         "--system", "sta", "--v-from", "matrix", str(MATRIX)],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_ENV, timeout=120,
    )
    assert proc.returncode == 2
    assert "--output-dir" in proc.stderr


def test_unknown_system_is_rejected_by_argparse():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "gain_region_fixedV.py"),
         "--system", "not_a_system", "--v-from", "matrix", str(MATRIX),
         "--output-dir", "."],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_ENV, timeout=120,
    )
    assert proc.returncode == 2


@needs_matrix
def test_mosek_direct_is_refused_and_recorded(tmp_path):
    rc = gr.main([
        "--system", "sta", "--v-from", "matrix", str(MATRIX),
        "--output-dir", str(tmp_path), "--solver", "MOSEK_DIRECT",
        "--box", "k1:1.45:1.55,k2:1.0:1.1",
    ])
    assert rc == gr.REFUSAL_EXIT_CODE
    refusals = sorted(tmp_path.glob("E9_gain_region_REFUSED_sta_*.json"))
    assert len(refusals) == 1
    payload = json.loads(refusals[0].read_text())
    assert payload["refused"] is True
    assert payload["refusal_kind"] == "unsupported_solver_encoding"
    assert payload["provenance"]["git_sha"]


@needs_matrix
def test_box_naming_a_foreign_coordinate_is_refused(tmp_path):
    rc = gr.main([
        "--system", "sta", "--v-from", "matrix", str(MATRIX),
        "--output-dir", str(tmp_path), "--solver", "CLARABEL",
        "--box", "lambda1:1.0:2.0",
    ])
    assert rc == gr.REFUSAL_EXIT_CODE
    payload = json.loads(
        sorted(tmp_path.glob("E9_gain_region_REFUSED_*.json"))[0].read_text())
    assert payload["refusal_kind"] == "box_invalid"


@needs_matrix
def test_center_mismatch_between_V_record_and_registry_is_refused(tmp_path):
    """The 25-term order-2 record was designed at the FLAGSHIP tuning; centring
    its box on the benchmark gain point would certify a box around gains this V
    was never designed for, so it is refused rather than warned about."""
    record = (REPO_ROOT / "artifacts/ablation/levant_2003_perturbed"
              / "levant3d_delta1_sparse_row.json")
    if not record.exists():
        pytest.skip("flagship record absent")
    rc = gr.main([
        "--system", "levant3d", "--v-from", "record", str(record),
        "--output-dir", str(tmp_path), "--solver", "CLARABEL",
        "--rel-halfwidth", "0.02", "--delta0", "1", "--dry-run",
    ])
    assert rc == gr.REFUSAL_EXIT_CODE
