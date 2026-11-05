"""WP TB: the generic gain-point driver ``scripts/certify_gain_point.py``.

CLARABEL only (the CI suite is MOSEK-free by design). Two groups:

* solver-free checks of the input contract: exact parsing, bracket validity
  in Q, the verifier's gain table, the pidsmc ``delta`` pin, guard R6, the
  registry match EM depends on, the corner x endpoint order, the W-group
  count invariant, and the refusal to run D without MOSEK;
* end to end, CLARABEL design (labelled NOT D) -> exact phase -> exactgains
  record -> EM -> portable verifier, on STA and twisting, plus the fixed-V
  mode on the published STA sparse row and a twisting gain box, which EM
  emits through its exact-record gain-point mode (a box is not a registered
  gain point). The exact-record mode itself is pinned in
  ``tests/test_exact_record_emission.py``.

Nothing here certifies anything for the paper: a CLARABEL design is not D.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "certify_gain_point.py"
STA_SPARSE_ROW = (ROOT / "artifacts" / "ablation" / "levant_2003_perturbed"
                  / "sta_delta1_sparse_row.json")


def _load():
    import sys
    spec = importlib.util.spec_from_file_location("certify_gain_point", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load()


def _mosek_available() -> bool:
    try:
        import mosek  # noqa: F401
        return True
    except ImportError:
        return False


def _read_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


# ---------------------------------------------------------------------------
# Input contract (no solver)
# ---------------------------------------------------------------------------

def test_gains_are_exact_rationals():
    spec = tb.build_gain_spec("sta", ["k1=3/2", "k2=1.1"], delta0="1/2")
    assert spec.rational == {"k1": Fraction(3, 2), "k2": Fraction(11, 10)}
    assert spec.delta0 == Fraction(1, 2) and spec.perturbed
    assert spec.corners() == [{}]


def test_bracket_is_checked_in_q():
    ok = tb.build_gain_spec(
        "levant3d", ["lambda1=2,lambda3=11/10"],
        brackets=["lambda2:2121/1000:2122/1000:2:9/2"])
    assert ok.brackets["lambda2"].power_target == Fraction(9, 2)
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_gain_spec("levant3d", ["lambda1=2,lambda3=11/10"],
                           brackets=["lambda2:2122/1000:2123/1000:2:9/2"])
    assert exc.value.kind == "bad_bracket"


def test_duplicate_and_malformed_gains_refused():
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_gain_spec("twisting", ["r1=2,r2=3/2"], boxes=["r1:2:3"])
    assert exc.value.kind == "duplicate_gain"
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_gain_spec("twisting", ["r1=2,r2=abc"])
    assert exc.value.kind == "bad_rational"
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_gain_spec("twisting", ["r1=2"], boxes=["r2:3:2"])
    assert exc.value.kind == "bad_box"


def test_verifier_gain_table_is_the_authority():
    with pytest.raises(tb.TBRefusal) as exc:
        tb.check_gain_names(tb.build_gain_spec("twisting", ["r1=2"]))
    assert exc.value.kind == "missing_gain"
    with pytest.raises(tb.TBRefusal) as exc:
        tb.check_gain_names(tb.build_gain_spec("twisting", ["r1=2,r2=3/2,r3=1"]))
    assert exc.value.kind == "unregistered_gain"
    with pytest.raises(tb.TBRefusal) as exc:
        tb.check_gain_names(tb.build_gain_spec("example1", ["k1=1,k2=1"],
                                               delta0="1/2"))
    assert exc.value.kind == "no_disturbance_channel"


def test_pidsmc_delta_is_pinned_not_a_gain():
    spec = tb.build_gain_spec(
        "pidsmc", ["lambda1=27/10,lambda2=1069/200,lambda3=11/10"],
        delta0="51/64")
    tb.check_gain_names(spec)
    assert spec.rational["delta"] == 1
    bad = tb.build_gain_spec(
        "pidsmc", ["lambda1=27/10,lambda2=1069/200,lambda3=11/10,delta=2"])
    with pytest.raises(tb.TBRefusal) as exc:
        tb.check_gain_names(bad)
    assert exc.value.kind == "non_affine_parameter"
    boxed = tb.build_gain_spec(
        "pidsmc", ["lambda1=27/10,lambda2=1069/200,lambda3=11/10"],
        boxes=["delta:1:2"])
    with pytest.raises(tb.TBRefusal) as exc:
        tb.check_gain_names(boxed)
    assert exc.value.kind == "non_affine_parameter"


def test_guard_r6_reused_for_varying_gains():
    spec = tb.build_gain_spec("twisting", ["r2=3/2"], boxes=["r1:2:3"],
                              delta0="1/4")
    tb.check_gain_names(spec)
    report = tb.assert_affinity(spec)
    assert report["required"] and report["varying"] == ["r1"]
    assert report["W_linear_in_each_declared_gain"]
    # pidsmc: the verifier's lambda names alias R6's k coordinates at delta = 1
    pid = tb.build_gain_spec("pidsmc", ["lambda2=1069/200,lambda3=11/10"],
                             boxes=["lambda1:13/5:14/5"])
    tb.check_gain_names(pid)
    assert tb.assert_affinity(pid)["aliases"] == {"lambda1": "k1"}
    # a system with no declared affine coordinates cannot carry a box
    ex1 = tb.build_gain_spec("example1", ["k2=1"], boxes=["k1:1:2"])
    tb.check_gain_names(ex1)
    with pytest.raises(tb.TBRefusal) as exc:
        tb.assert_affinity(ex1)
    assert exc.value.kind == "non_affine_parameter"
    # a single gain point never needs the vertex argument
    single = tb.build_gain_spec("sta", ["k1=3/2,k2=11/10"])
    assert tb.assert_affinity(single)["required"] is False


def test_registry_match_decides_emission():
    assert tb.registry_match(tb.build_gain_spec("sta", ["k1=3/2,k2=11/10"]))[0] \
        == "flagship"
    assert tb.registry_match(tb.build_gain_spec("twisting", ["r1=2,r2=3/2"]))[0] \
        == "benchmark"
    l3d = tb.build_gain_spec("levant3d", preset="flagship", delta0="1")
    assert tb.registry_match(l3d)[0] == "flagship"
    # the benchmark levant3d point is a different gain point
    assert tb.registry_match(tb.build_gain_spec(
        "levant3d", preset="benchmark"))[0] == "benchmark"
    # a scaled CTA point and any box are not registered
    s = Fraction(17, 20)
    cta = tb.build_gain_spec("continuous_twisting", [
        f"k1={s**4 * Fraction(274, 5)},k2={s**3 * 27},"
        f"k3={s**6 * Fraction(57, 10)},k4={s**6 * 3}"], delta0=str(s**6))
    assert tb.registry_match(cta)[0] is None
    box = tb.build_gain_spec("twisting", ["r2=3/2"], boxes=["r1:2:3"])
    assert tb.registry_match(box)[0] is None


def test_vertex_order_is_corner_major_and_exact():
    spec = tb.build_gain_spec("twisting", ["r2=3/2"], boxes=["r1:2:21/10"],
                              delta0="1/4")
    cfg = tb.check_gain_names(spec)
    verts = tb.build_vertices(spec, cfg)
    assert len(verts.exact) == len(verts.floats) == 4
    assert verts.corner_index == [0, 0, 1, 1]
    assert verts.corners == [{"r1": Fraction(2)}, {"r1": Fraction(21, 10)}]
    assert "endpoint +delta0" in verts.labels[0]
    assert "endpoint -delta0" in verts.labels[1]
    # every exact coefficient is a Fraction; the corner gain enters exactly
    for dyn in verts.exact:
        for gf in dyn:
            assert all(isinstance(t.coefficient, Fraction) for t in gf.terms)
    coeffs = {t.coefficient for t in verts.exact[2][1].terms}
    assert Fraction(-21, 10) in coeffs and Fraction(1, 4) in coeffs


def test_bracket_corners_match_the_registry_order():
    from gf_framework.systems import printed_gains as pg
    spec = tb.build_gain_spec("levant3d", preset="flagship", delta0="1")
    assert spec.corners() == pg.corner_assignments(
        pg.FLAGSHIP_GAIN_POINTS["levant3d"])


def test_w_group_count_must_equal_dynamics(tmp_path):
    path = tmp_path / "b.npz"
    path.write_bytes(b"x")
    prepared = SimpleNamespace(w={0: [], 1: []}, fingerprint="f",
                               n_entries=12, parity_expanded=False)
    with pytest.raises(tb.TBRefusal) as exc:
        tb.check_w_groups(prepared, 4, path)
    assert exc.value.kind == "w_group_count_mismatch"
    rec = tb.check_w_groups(prepared, 2, path)
    assert rec["n_w_groups"] == rec["n_dynamics"] == 2


def test_check_only_writes_a_plan(tmp_path):
    code = tb.main(["--system", "levant3d", "--box", "lambda1:2:3",
                    "--box", "lambda2:2121/1000:2599/1000",
                    "--gains", "lambda3=11/10", "--delta0", "1",
                    "--degrees", "8", "9", "--check-only",
                    "--out-dir", str(tmp_path)])
    assert code == tb.EXIT_OK
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["dynamics"]["n"] == 8          # 4 corners x 2 endpoints
    assert plan["registry_match"]["table"] is None
    assert plan["registry_match"]["emission_route"] == "exact-record"
    assert plan["guard_r6"]["required"] is True


@pytest.mark.skipif(_mosek_available(), reason="MOSEK is installed here")
def test_mosek_direct_never_falls_back(tmp_path):
    code = tb.main(["--system", "sta", "--gains", "k1=3/2,k2=11/10",
                    "--delta0", "1", "--degrees", "3",
                    "--out-dir", str(tmp_path)])
    assert code == tb.EXIT_REFUSED
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["refusal"]["kind"] == "design_solver_unavailable"
    assert not list(tmp_path.glob("m*"))       # nothing was designed


def test_frozen_root_refused(tmp_path):
    with pytest.raises(SystemExit):
        tb.main(["--system", "sta", "--gains", "k1=3/2,k2=11/10",
                 "--degrees", "3", "--check-only",
                 "--out-dir", str(ROOT / "artifacts" / "ablation" / "tb_probe")])
    assert not (ROOT / "artifacts" / "ablation" / "tb_probe").exists()


# ---------------------------------------------------------------------------
# End to end, CLARABEL (NOT D)
# ---------------------------------------------------------------------------

def _run(tmp_path, *argv):
    code = tb.main(list(argv) + ["--design-solver", "CLARABEL",
                                 "--out-dir", str(tmp_path)])
    return code, json.loads((tmp_path / "run.json").read_text())


def _check_emitted(tmp_path, run, m):
    em = run["emission"]
    assert em["returncode"] == 0
    assert em["self_verify"]["verdict"] == "VERIFIED"
    assert Fraction(em["self_verify"]["eps_certified_exact"]) > 0
    rec = _read_gz(tmp_path / f"m{m}" / "portable" / em["record"])
    assert "clarabel_design_not_D" in em["record"]
    assert "NOT on the D -> X chain" in rec["provenance"]["source_claim"]
    assert rec["eps_certified_source"] == "portable_verifier"
    return rec


def test_sta_perturbed_end_to_end(tmp_path):
    code, run = _run(tmp_path, "--system", "sta", "--gains", "k1=3/2,k2=11/10",
                     "--delta0", "1/2", "--degrees", "3")
    assert code == tb.EXIT_OK, run
    assert run["stage_label"] == "clarabel_design_not_D"
    assert run["citable_chain"] is False
    exact = json.loads((tmp_path / "m3" / "exact.json").read_text())
    assert exact["design_stage"]["citable_chain"] is False
    assert exact["design_stage"]["bundle"] is None
    assert exact["n_dynamics"] == 2 and exact["exact_pass_all"]
    rec = _check_emitted(tmp_path, run, 3)
    assert rec["delta0_exact"] == "1/2"
    assert rec["rational_gains"] == {"k1": "3/2", "k2": "11/10"}


def test_twisting_perturbed_degree_ladder(tmp_path):
    code, run = _run(tmp_path, "--system", "twisting", "--gains",
                     "r1=2,r2=3/2", "--delta0", "1/4", "--degrees", "4", "5")
    assert code == tb.EXIT_OK, run
    # the ladder records a degree that did not design and stops at the first
    # degree whose exact phase passes
    assert [d["m"] for d in run["degrees"]][-1] == run["certified_degree"]
    rec = _check_emitted(tmp_path, run, run["certified_degree"])
    assert rec["gain_point"]["system"] == "twisting"
    assert rec["coverage"]["inclusion_endpoints"] == 2


def test_twisting_box_is_emitted_through_the_exact_record(tmp_path):
    """Before the emitter change this run stopped at exit 5 (emission
    blocked); a box is now emitted with EM's --gain-point exact-record."""
    code, run = _run(tmp_path, "--system", "twisting", "--box", "r1:2:21/10",
                     "--gains", "r2=3/2", "--degrees", "5")
    assert code == tb.EXIT_OK, run
    exact = json.loads((tmp_path / "m5" / "exact.json").read_text())
    assert exact["exact_pass_all"] and len(exact["corners"]) == 2
    assert exact["box"] == {"r1": {"lo": "2", "hi": "21/10"}}
    grams = _read_gz(tmp_path / "m5" / "exact_grams.json.gz")
    assert [c["assignment"] for c in grams["corners"]] == [
        {"r1": "2"}, {"r1": "21/10"}]
    assert all(len(c["W_grams"]) == 4 for c in grams["corners"])
    assert run["emission"]["gain_point"] == "exact-record"
    assert run["emission"]["registry_match"]["table"] is None
    rec = _check_emitted(tmp_path, run, 5)
    assert rec["brackets"] == {"r1": {"lo": "2", "hi": "21/10", "power_k": 1,
                                      "power_target": "41/20"}}
    assert rec["provenance"]["gain_point_source"]["kind"] == "exact_record"
    assert not any(s["role"] == "gain_point_registry"
                   for s in rec["source_artifacts"])


def _synthetic_bundle(V, float_vertices, epsilon):
    """A bundle in the design schema, assembled from CLARABEL captures.

    NOT a design bundle and never evidence: it only lets the MOSEK-free suite
    drive the D-path plumbing (persist, W-group pin, X on the bundle) that a
    MOSEK_DIRECT design would feed. Entry hashes follow the schema recipe.
    """
    from gf_framework.algebra.lie_derivative import (
        compute_lie_derivative_gf, negate_gf)
    from gf_framework.verification.joint_sos import (
        GRAM_BUNDLE_SCHEMA, gram_bundle_entry_hash, verify_gf_sos_full)

    def entries(gf, block, endpoint):
        res = verify_gf_sos_full(gf, epsilon=epsilon, solver="CLARABEL",
                                 strict_solver=True, allow_solver_fallback=False,
                                 capture_grams=True)
        assert res.feasible
        out = []
        for cap in res.gram_certificates:
            e = dict(cap)
            e.update(schema=GRAM_BUNDLE_SCHEMA, block=block, endpoint=endpoint,
                     name=(f"V_oct{cap['octant_index']}" if block == "V" else
                           f"W_oct{cap['octant_index']}_v{endpoint}"),
                     parity_class=None, clearing_power=0,
                     coefficient_scale=1.0, dim=len(cap["basis"]),
                     dtype="float64")
            e["hash"] = gram_bundle_entry_hash(e)
            out.append(e)
        return out

    bundle = entries(V, "V", None)
    for k, dyn in enumerate(float_vertices):
        W = negate_gf(compute_lie_derivative_gf(V, dyn, validate=False))
        bundle.extend(entries(W, "W", k))
    return bundle


@pytest.mark.skipif(not STA_SPARSE_ROW.exists(),
                    reason="the published STA sparse row is not present")
def test_bundle_path_plumbing_with_a_synthetic_bundle(tmp_path):
    row = json.loads(STA_SPARSE_ROW.read_text())
    V = tb.reconstruct_v(row)
    spec = tb.build_gain_spec("sta", ["k1=3/2,k2=11/10"], delta0="1")
    verts = tb.build_vertices(spec, tb.check_gain_names(spec))
    eps = 1e-6
    fake = SimpleNamespace(gram_certificates=_synthetic_bundle(
        V, verts.floats, eps), solver_used="MOSEK_DIRECT")
    args = SimpleNamespace(design_epsilon=eps, witness_solver="CLARABEL",
                           fallback_solver="", jobs=1,
                           margin_policy="achieved")
    bundle = tb.persist_bundle(fake, tmp_path, 6, spec, verts, args)
    assert bundle["n_w_groups"] == bundle["n_dynamics"] == 2
    assert (tmp_path / bundle["path"]).exists()
    phase = tb.run_x(V, verts, args, bundle, tmp_path, eps)
    assert phase.exact_pass
    assert phase.witness_source == "design_bundle"
    assert phase.gram_bundle_fingerprint == bundle["fingerprint"]
    record, grams = tb.exact_records(
        phase, spec, verts, V, 6, eps, args,
        {"stage_label": tb.STAGE_D, "design_solver": "MOSEK_DIRECT",
         "design_epsilon": eps}, bundle, "test")
    assert record["design_stage"]["citable_chain"] is True
    assert record["witness"]["witness_source"] == "design_bundle"
    assert all(len(c["endpoints"]) == 2 for c in grams["corners"])

    # the same 2-group bundle can never witness a 4-dynamics box run
    box = tb.build_gain_spec("sta", ["k2=11/10"], boxes=["k1:3/2:8/5"],
                             delta0="1")
    box_verts = tb.build_vertices(box, tb.check_gain_names(box))
    assert len(box_verts.exact) == 4
    (tmp_path / "box").mkdir()
    with pytest.raises(tb.TBRefusal) as exc:
        tb.persist_bundle(fake, tmp_path / "box", 6, box, box_verts, args)
    assert exc.value.kind == "w_group_count_mismatch"
    # and run_exact_phase enforces the same invariant on its own
    from gf_framework.verification.exact_phase import (
        GramBundleError, run_exact_phase)
    with pytest.raises(GramBundleError) as gexc:
        run_exact_phase(V, box_verts.exact, 2, epsilon=eps,
                        gram_bundle=str(tmp_path / bundle["path"]),
                        solver="CLARABEL", fallback_solver=None)
    assert gexc.value.reason == "gram_bundle_input_mismatch"


@pytest.mark.mosek
@pytest.mark.skipif(not _mosek_available(), reason="the D path needs MOSEK")
@pytest.mark.parametrize("system, argv, m, n_terms", [
    # the sta_delta1_sparse scope: rl1 support selection, then D on it
    ("sta", ["--gains", "k1=3/2,k2=11/10", "--delta0", "1", "--degrees", "6",
             "--sparsify", "rl1", "--design-epsilon", "1e-6"], 6, 5),
    # X4's top rung, dense library, degree ladder starting at the anchor
    ("twisting", ["--gains", "r1=2,r2=3/2", "--delta0", "127/256",
                  "--degrees", "5", "6"], 5, 10),
])
def test_mosek_direct_d_path_end_to_end(tmp_path, system, argv, m, n_terms):
    """The real D path (first run 2026-09-25 on the MOSEK machine): a strict
    MOSEK_DIRECT design persists its Gram bundle, X consumes that bundle with
    no re-solve, EM emits through the registry route, and the stdlib verifier
    prints VERIFIED. Each case takes a few seconds."""
    import shutil
    import sys
    p_python = shutil.which("python3") or sys.executable
    code = tb.main(["--system", system, *argv, "--p-python", p_python,
                    "--out-dir", str(tmp_path)])
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == tb.EXIT_OK, run
    assert run["stage_label"] == tb.STAGE_D and run["citable_chain"] is True
    assert run["certified_degree"] == m
    design = json.loads((tmp_path / f"m{m}" / "design.json").read_text())
    assert design["solver_used"] == "MOSEK_DIRECT"
    assert design["solver_fallback_used"] is False
    assert design["n_terms"] == n_terms
    bundle = design["bundle"]
    assert (tmp_path / f"m{m}" / bundle["path"]).exists()
    assert bundle["n_w_groups"] == bundle["n_dynamics"] == 2
    exact = json.loads((tmp_path / f"m{m}" / "exact.json").read_text())
    assert exact["design_stage"]["label"] == tb.STAGE_D
    assert exact["design_stage"]["citable_chain"] is True
    w = exact["witness"]
    # X ran on D's own bundle: one attempt, no fallback re-solve
    assert w["witness_source"] == "design_bundle"
    assert w["witness_solver"] == "MOSEK_DIRECT"
    assert w["n_witness_attempts"] == 1
    assert w["gram_bundle_fingerprint"] == bundle["fingerprint"]
    assert all(ep["witness_source"] == "design_bundle"
               for c in exact["corners"] for key, ep in c.items()
               if key in ("W+", "W-"))
    em = run["emission"]
    assert em["returncode"] == 0 and em["record"].startswith(f"{system}_tb_m{m}_")
    rec = _read_gz(tmp_path / f"m{m}" / "portable" / em["record"])
    assert "D -> X on D's persisted MOSEK_DIRECT Gram bundle" in \
        rec["provenance"]["source_claim"]
    assert run["P"]["verdict"] == "VERIFIED"
    assert Fraction(run["P"]["eps_certified_exact"]) > 0


@pytest.mark.skipif(not STA_SPARSE_ROW.exists(),
                    reason="the published STA sparse row is not present")
def test_fixed_v_row_mode(tmp_path):
    code = tb.main(["--system", "sta", "--gains", "k1=3/2,k2=11/10",
                    "--delta0", "1", "--v-row", str(STA_SPARSE_ROW),
                    "--witness-solver", "CLARABEL", "--fallback-solver", "",
                    "--design-epsilon", "1e-6", "--out-dir", str(tmp_path)])
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == tb.EXIT_OK, run
    assert run["stage_label"] == "fixed_v_row_not_D"
    em = run["emission"]
    assert em["self_verify"]["verdict"] == "VERIFIED"
    rec = _read_gz(tmp_path / "m6" / "portable" / em["record"])
    assert len(rec["V"]["terms"]) == 5 and rec["epsilon_exact"] == str(
        Fraction(1e-6))


def test_sparsify_runs_at_the_design_epsilon(tmp_path, monkeypatch):
    """--sparsify rl1 hands --design-epsilon to the support selection (X5,
    2026-09-25): before, the selection ran at the designer default 1e-4
    whatever D's margin was."""
    from gf_framework.verification import sparse_sos
    seen = []
    real = sparse_sos.design_sparse_lyapunov

    def recording(*args, **kwargs):
        seen.append(kwargs.get("config"))
        return real(*args, **kwargs)

    monkeypatch.setattr(sparse_sos, "design_sparse_lyapunov", recording)
    code, run = _run(tmp_path, "--system", "sta", "--gains", "k1=3/2,k2=11/10",
                     "--delta0", "1/2", "--degrees", "3", "--sparsify", "rl1",
                     "--design-epsilon", "1e-6", "--no-emit")
    assert code == tb.EXIT_OK, run
    assert len(seen) == 1 and seen[0].design_epsilon == 1e-6
    design = json.loads((tmp_path / "m3" / "design.json").read_text())
    assert design["sparsify"]["design_epsilon"] == 1e-6
    assert design["design_epsilon"] == 1e-6


# ---------------------------------------------------------------------------
# Claim text follows the ADOPTED witness (2026-09-25, reports/X2.md issue 2)
# ---------------------------------------------------------------------------

def test_claim_names_the_adopted_witness():
    bundle = {"path": "b.npz"}
    on_bundle = {"witness_source": "design_bundle",
                 "witness_solver_used": "MOSEK_DIRECT", "n_witness_attempts": 1}
    claim, note = tb.claim_texts(tb.STAGE_D, bundle, on_bundle)
    assert "D -> X on D's persisted MOSEK_DIRECT Gram bundle" in claim
    assert note == "X consumed D's persisted Gram bundle"
    # D ran, but X adopted a re-solve: the text must not claim the bundle
    fallback = {"witness_source": "resolve", "witness_solver": "MOSEK",
                "witness_solver_used": "MOSEK_DIRECT", "n_witness_attempts": 2}
    claim, note = tb.claim_texts(tb.STAGE_D, bundle, fallback)
    assert "D -> X on D's persisted" not in claim
    assert "did NOT adopt" in claim and "resolve (MOSEK_DIRECT)" in claim
    assert "after 2 attempt(s)" in claim
    assert "fallback witness resolve (MOSEK_DIRECT)" in note
    # no bundle at all (CLARABEL design): the old wording, plus the witness
    claim, note = tb.claim_texts(tb.STAGE_CLARABEL, None,
                                 {"witness_source": "resolve",
                                  "witness_solver": "CLARABEL"})
    assert "NOT on the D -> X chain" in claim and "resolve (CLARABEL)" in claim
    assert "NOT on the D -> X chain" in note


# ---------------------------------------------------------------------------
# Two-point (union) mode: one V over several gain points (2026-09-25)
# ---------------------------------------------------------------------------

def test_parse_point_items():
    g, b, x = tb.parse_point("lambda1=2; bracket=lambda2:2121/1000:2122/1000:2:9/2")
    assert g == ["lambda1=2"] and b == ["lambda2:2121/1000:2122/1000:2:9/2"]
    assert x == []
    g, b, x = tb.parse_point("r1=2,r2=3/2;box=r2:3/2:8/5")
    assert g == ["r1=2,r2=3/2"] and x == ["r2:3/2:8/5"]
    with pytest.raises(tb.TBRefusal) as exc:
        tb.parse_point(" ; ")
    assert exc.value.kind == "bad_point"


def _two_point_args(*points, extra=()):
    return tb.build_parser().parse_args(
        ["--system", "levant3d", "--gains", "lambda3=11/10", "--delta0", "1",
         "--degrees", "8", "--out-dir", "unused", *extra]
        + [a for p in points for a in ("--point", p)])


A_POINT = "lambda1=2;bracket=lambda2:2121/1000:2122/1000:2:9/2"
B_POINT = "lambda1=3;bracket=lambda2:2598/1000:2599/1000:2:27/4"


def test_two_point_members_are_checked_and_ordered():
    args = _two_point_args(A_POINT, B_POINT)
    members = tb.build_members(args)
    verts = tb.combine_vertices(members)
    assert [m.dyn_index for m in members] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert len(verts.exact) == 8 and len(verts.corners) == 4
    assert verts.corner_index == [0, 0, 1, 1, 2, 2, 3, 3]
    # each point keeps its own gains, bracket (checked in Q) and R6 report
    assert members[0].spec.rational["lambda1"] == 2
    assert members[1].spec.brackets["lambda2"].power_target == Fraction(27, 4)
    assert all(m.affinity["required"] for m in members)
    # neither point is a registry point (the registry brackets are ~1e-15 wide)
    assert [m.table for m in members] == [None, None]
    # point 1's dynamics carry its exact gains: lambda1 = 3 in f0, and the
    # lambda2 bracket's lo endpoint 2598/1000 in f1 at its corner 0
    assert any(t.coefficient == -3 for t in verts.exact[4][0].terms)
    assert any(t.coefficient == Fraction(-2598, 1000)
               for t in verts.exact[4][1].terms)
    assert any(t.coefficient == -2 for t in verts.exact[0][0].terms)
    # a repeated point, a shared-and-per-point name, a bad bracket: refused
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_members(_two_point_args(A_POINT, A_POINT))
    assert exc.value.kind == "duplicate_point"
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_members(_two_point_args(A_POINT + ";lambda3=1"))
    assert exc.value.kind == "duplicate_gain"
    with pytest.raises(tb.TBRefusal) as exc:
        tb.build_members(_two_point_args(
            "lambda1=2;bracket=lambda2:2122/1000:2123/1000:2:9/2"))
    assert exc.value.kind == "bad_bracket"


def test_two_point_check_only_plan(tmp_path):
    code = tb.main(["--system", "levant3d", "--gains", "lambda3=11/10",
                    "--point", A_POINT, "--point", B_POINT, "--delta0", "1",
                    "--degrees", "8", "--check-only", "--out-dir", str(tmp_path)])
    assert code == tb.EXIT_OK
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["dynamics"]["n"] == 8
    assert [p["emission_route"] for p in plan["registry_match"]["points"]] \
        == ["exact-record", "exact-record"]
    assert len(plan["gain_point"]["points"]) == 2
    code = tb.main(["--system", "levant3d", "--point", A_POINT,
                    "--preset", "flagship", "--degrees", "8",
                    "--out-dir", str(tmp_path / "x")])
    assert code == tb.EXIT_REFUSED
    run = json.loads((tmp_path / "x" / "run.json").read_text())
    assert run["refusal"]["kind"] == "bad_usage"


def _canon_v(rec):
    return json.dumps(rec["V"], sort_keys=True)


def test_two_point_twisting_end_to_end(tmp_path):
    """CLARABEL (NOT D): one V over two twisting gain points, one record per
    point, each re-proved by the portable verifier, identical V."""
    import sys
    code = tb.main(["--system", "twisting", "--gains", "r2=3/2",
                    "--point", "r1=2", "--point", "r1=21/10",
                    "--delta0", "1/4", "--degrees", "5",
                    "--design-solver", "CLARABEL", "--p-python", sys.executable,
                    "--out-dir", str(tmp_path)])
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == tb.EXIT_OK, run
    assert run["certified_degree"] == 5
    pts = run["points_result"]
    assert [p["point"] for p in pts] == [0, 1]
    recs = []
    # (2, 3/2) IS the benchmark registry point; (21/10, 3/2) is not
    assert [p["emission"]["gain_point"] for p in pts] == ["benchmark",
                                                          "exact-record"]
    for p in pts:
        assert p["exact_pass"] and p["P"]["verdict"] == "VERIFIED"
        exact = json.loads((tmp_path / "m5" / p["dir"] / "exact.json").read_text())
        assert exact["n_dynamics"] == 2 and len(exact["corners"]) == 1
        joint = exact["design_stage"]["joint_design"]
        assert joint["n_points"] == 2 and joint["n_dynamics_total"] == 4
        assert "jointly over 2 gain points" in exact["claim"]
        grams = _read_gz(tmp_path / "m5" / p["dir"] / "exact_grams.json.gz")
        assert len(grams["corners"]) == 1
        assert len(grams["corners"][0]["endpoints"]) == 2
        rec = _read_gz(tmp_path / "m5" / p["dir"] / "portable"
                       / p["emission"]["record"])
        assert f"_pt{p['point']}_" in p["emission"]["record"]
        recs.append(rec)
    assert recs[0]["rational_gains"]["r1"] == "2"
    assert recs[1]["rational_gains"]["r1"] == "21/10"
    assert _canon_v(recs[0]) == _canon_v(recs[1])


@pytest.mark.mosek
@pytest.mark.skipif(not _mosek_available(), reason="the D path needs MOSEK")
def test_two_point_d_path(tmp_path):
    """Two-point mode on the real D path: ONE bundle with a W group per
    dynamics of every point, X once on it, one VERIFIED record per point."""
    import shutil
    import sys
    p_python = shutil.which("python3") or sys.executable
    code = tb.main(["--system", "twisting", "--gains", "r2=3/2",
                    "--point", "r1=2", "--point", "r1=21/10",
                    "--delta0", "1/4", "--degrees", "5", "6",
                    "--p-python", p_python, "--out-dir", str(tmp_path)])
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == tb.EXIT_OK, run
    m = run["certified_degree"]
    design = json.loads((tmp_path / f"m{m}" / "design.json").read_text())
    assert design["bundle"]["n_w_groups"] == design["bundle"]["n_dynamics"] == 4
    recs = []
    for p in run["points_result"]:
        exact = json.loads((tmp_path / f"m{m}" / p["dir"] / "exact.json")
                           .read_text())
        src = exact["witness"]["witness_source"]
        assert exact["design_stage"]["witness_source"] == src
        # review m8: the joint bundle resolves from the point's directory
        bpath = tmp_path / f"m{m}" / p["dir"] / exact["design_stage"]["bundle"]["path"]
        assert tb.sha256_of(bpath) == design["bundle"]["sha256"]
        # review m10: the point's own worst ratio, not the joint one
        assert exact["witness"]["witness_charge_ratio"] == exact["worst_charge_ratio"]
        if src == "design_bundle":
            assert "D -> X on D's persisted" in exact["claim"]
        else:
            assert "did NOT adopt" in exact["claim"]
        assert p["P"]["verdict"] == "VERIFIED"
        recs.append(_read_gz(tmp_path / f"m{m}" / p["dir"] / "portable"
                             / p["emission"]["record"]))
    assert _canon_v(recs[0]) == _canon_v(recs[1])


# ---------------------------------------------------------------------------
# Review 2026-09-25 fixes: m8 (per-point bundle path), m10 (per-point witness
# charge ratio), m6 (--provenance-json)
# ---------------------------------------------------------------------------

def _joint_sta_run(tmp_path):
    """Two STA gain points, CLARABEL design, a SYNTHETIC bundle (test
    scaffolding, never evidence) persisted like D's, X on it, then the
    per-point records without emission."""
    args = tb.build_parser().parse_args(
        ["--system", "sta", "--gains", "k2=11/10", "--point", "k1=3/2",
         "--point", "k1=8/5", "--delta0", "1/2", "--degrees", "3",
         "--design-solver", "CLARABEL", "--witness-solver", "CLARABEL",
         "--fallback-solver", "", "--no-emit", "--out-dir", str(tmp_path)])
    args.supplied_provenance = None
    members = tb.build_members(args)
    verts = tb.combine_vertices(members)
    V, design_rec, _ = tb.run_design(members[0].spec, verts, 3, args,
                                     "CLARABEL")
    assert V is not None
    out = tmp_path / "m3"
    out.mkdir()
    fake = SimpleNamespace(gram_certificates=_synthetic_bundle(
        V, verts.floats, args.design_epsilon), solver_used="MOSEK_DIRECT")
    bundle = tb.persist_bundle(fake, out, 3, members[0].spec, verts, args)
    design_rec["stage_label"] = tb.STAGE_D   # plumbing only, see docstring
    phase = tb.run_x(V, verts, args, bundle, out, args.design_epsilon)
    assert phase.exact_pass and phase.witness_source == "design_bundle"
    run, entry = {"degrees": []}, {}
    status, code = tb.finish_points(run, entry, out, members, phase, V, 3,
                                    args.design_epsilon, args, design_rec,
                                    bundle, "test", tmp_path / "env.json")
    assert code == tb.EXIT_OK, status
    return out, members, phase, bundle


def test_point_exact_json_bundle_path_resolves(tmp_path):
    """m8: the per-point exact.json names the joint bundle by a path that
    resolves from its own directory (it lives one level up)."""
    out, members, phase, bundle = _joint_sta_run(tmp_path)
    for mem in members:
        pdir = out / f"point{mem.index}"
        stage = json.loads((pdir / "exact.json").read_text())["design_stage"]
        path = pdir / stage["bundle"]["path"]
        assert path.resolve() == (out / bundle["path"]).resolve()
        assert tb.sha256_of(path) == stage["bundle"]["sha256"]


def test_point_records_state_their_own_charge_ratio(tmp_path):
    """m10: witness_charge_ratio of a per-point exact.json is that point's
    own worst (V and its endpoints); the joint value is under joint_design;
    the attempt list says it is the joint phase's."""
    out, members, phase, _ = _joint_sta_run(tmp_path)
    own = []
    for mem in members:
        ex = json.loads((out / f"point{mem.index}" / "exact.json").read_text())
        w = ex["witness"]
        assert w["witness_charge_ratio"] == ex["worst_charge_ratio"]
        assert w["witness_charge_ratio"] == \
            tb.point_phase(phase, mem).worst_charge_ratio
        joint = ex["design_stage"]["joint_design"]
        assert joint["joint_witness_charge_ratio"] == phase.witness_charge_ratio
        assert all("joint phase" in a["scope"] for a in w["witness_attempts"])
        own.append(w["witness_charge_ratio"])
    assert max(own) == phase.worst_charge_ratio


def test_provenance_json_is_copied_verbatim(tmp_path):
    """m6: --provenance-json reaches run.json and exact.json verbatim and is
    named in the claim, which EM copies into the record's provenance."""
    import sys
    prov = {"scaling": {"s": "17/20", "q": "24137569/64000000",
                        "base_gains": {"k1": "3/2", "k2": "11/10"}}}
    pfile = tmp_path / "prov.json"
    pfile.write_text(json.dumps(prov))
    code, run = _run(tmp_path / "r", "--system", "sta", "--gains",
                     "k1=3/2,k2=11/10", "--delta0", "1/2", "--degrees", "3",
                     "--provenance-json", str(pfile),
                     "--p-python", sys.executable)
    assert code == tb.EXIT_OK, run
    assert run["provenance"]["supplied"]["supplied"] == prov
    ex = json.loads((tmp_path / "r" / "m3" / "exact.json").read_text())
    assert ex["provenance"]["supplied"] == prov
    assert ex["provenance"]["source_sha256"] == tb.sha256_of(pfile)
    assert "provenance.supplied" in ex["claim"] and "scaling" in ex["claim"]
    rec = _read_gz(tmp_path / "r" / "m3" / "portable"
                   / run["emission"]["record"])
    assert "provenance.supplied" in rec["provenance"]["source_claim"]
    # the record binds the exact.json that carries the block, by hash
    src = [s for s in rec["source_artifacts"] if s["role"] == "exact_phase"]
    assert src and src[0]["sha256"] == tb.sha256_of(
        tmp_path / "r" / "m3" / "exact.json")
    # a non-object or malformed file is refused before anything runs
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    code = tb.main(["--system", "sta", "--gains", "k1=3/2,k2=11/10",
                    "--degrees", "3", "--check-only", "--provenance-json",
                    str(bad), "--out-dir", str(tmp_path / "b")])
    assert code == tb.EXIT_REFUSED
    run = json.loads((tmp_path / "b" / "run.json").read_text())
    assert run["refusal"]["kind"] == "bad_provenance_json"
