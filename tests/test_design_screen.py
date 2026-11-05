"""Fail-fast design screen (WP P2-04): the two cheap filters of decision D1.

Between the MOSEK design solve (D) and the exact rational re-proof (X) there
used to be nothing but the solver's own status. P2-04 wires the two filters D1
names onto every packaged design result:

* ``sphere_check.min_V_on_sphere`` on V, typed ``sphere_precheck_failed``;
* ``joint_sos._validate_gram_certificate`` on EVERY bundle entry, V and W, the
  residual charged against that entry's own ε budget, typed
  ``gram_validation_failed`` with the worst block named.

What is pinned here:

* both filters run by DEFAULT on the MOSEK_DIRECT path, and the 12 nominal
  systems pass them;
* a design whose packaged Gram is corrupted is REFUSED with the typed reason;
* acceptance never reads a solver status — in particular the design backend's
  own row-wise screen (``mosek_direct``: ``solsta.unknown`` at eig_tol 1e-7 /
  res_tol 1e-5) does not authorize anything here, and Gram mass on monomials
  OUTSIDE the target support (which that row-wise screen cannot see) is
  charged in full;
* the screen only ever rejects: on a passing design, "off", "warn" and
  "reject" give the same design point;
* blocks with no margin (``eps == 0.0``, typed ``eps_reason`` from P2-02) are
  reported with their numbers but cannot refuse — the exact gate refuses them
  before it reads their Gram, so they can never enter X.

Most of the file is MOSEK-free (synthetic bundles); the end-to-end wiring
tests are marked ``needs_mosek``.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from gf_framework.algebra.enumerate import (
    enumerate_cross_terms,
    enumerate_diagonal_terms,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible
from gf_framework.discovery import universal_discovery as ud
from gf_framework.verification.design_screen import (
    GRAM_VALIDATION_FAILED,
    SCREEN_MODES,
    SCREEN_OK,
    SPHERE_PRECHECK_FAILED,
    DesignScreen,
    resolve_screen_mode,
    screen_design_candidate,
    screen_design_result,
    screen_gram_bundle,
)
from gf_framework.verification.joint_sos import JointSOSOptimizer, JointSOSResult
from gf_framework.verification.mosek_direct import MOSEK_AVAILABLE

from test_verifier_soundness import build_embedded_V

needs_mosek = pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not installed")

# The canonical design degrees of the 12 nominal rows. `continuous_twisting`
# is 8, not 9: owner decision D-j (2026-09-04, WP P5-14) pins the framework's
# CTA design degree at the degree its own escalation reaches (69 terms), and
# the m = 9 / 101-term row is the historical August record. See
# `artifacts/campaign_2026-08/E1_matrix/validation_gate_2026-09-04_cta8/`.
DEGREES = {"fosmc": 2, "power_rate": 2, "sta": 3, "twisting": 5, "example1": 4,
           "example3": 5, "continuous_twisting": 8, "levant3d": 5, "pidsmc": 6,
           "dsta": 3, "fc_cta": 7, "levant4d": 7}
SPEC_FACTORY_NAMES = {
    "fosmc": "create_fosmc_spec", "power_rate": "create_power_rate_spec",
    "sta": "create_sta_spec", "twisting": "create_twisting_spec",
    "example1": "create_example1_spec", "example3": "create_example3_spec",
    "continuous_twisting": "create_cta_spec", "levant3d": "create_levant3d_spec",
    "pidsmc": "create_pidsmc_spec", "dsta": "create_dsta_spec",
    "fc_cta": "create_fc_cta_spec", "levant4d": "create_levant4d_spec",
}


def _library(spec, m):
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    cross = [t for t in cross_full if mu_compatible(t, mu)]
    return list(diag) + list(cross)


def _design(name, **kwargs):
    spec = getattr(ud, SPEC_FACTORY_NAMES[name])()
    m = DEGREES[name]
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                            allow_solver_fallback=False, **kwargs)
    return opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf,
        active_terms=_library(spec, m),
        dynamics_gf_vertices=spec.dynamics_gf_vertices)


# ---------------------------------------------------------------------------
# Synthetic bundle entries (no solver)
# ---------------------------------------------------------------------------
EPS = 1e-4


def _entry(gram, *, name="V_oct0", block="V", eps=EPS, eps_reason="",
           target=None, basis=None, signs=(1, 1)):
    """One bundle entry for the 2-variable quartic y1^4 + y2^4 (+ cross)."""
    basis = basis if basis is not None else [(2, 0), (1, 1), (0, 2)]
    if target is None:
        target = {(4, 0): 1.0 - eps, (2, 2): 0.0, (0, 4): 1.0 - eps}
    return {
        "schema": "gf.design_grams/1", "block": block, "name": name,
        "signs": signs, "octant_index": 0, "endpoint": None,
        "parity_class": None, "eps": eps, "eps_reason": eps_reason,
        "mu": (2, 2), "y_degree": 4, "clearing_power": 0,
        "coefficient_scale": 1.0, "dim": len(basis), "dtype": "float64",
        "basis": basis, "target": dict(target),
        "gram": np.asarray(gram, dtype=np.float64),
    }


def _good_gram():
    # diag(1, 0, 1) reproduces y1^4 + y2^4 exactly; the ε already sits in the
    # target, so the charge is 0 and the budget ε/2 is untouched.
    return np.diag([1.0 - EPS, 0.0, 1.0 - EPS])


def test_clean_synthetic_bundle_passes():
    blocks = screen_gram_bundle([_entry(_good_gram())])
    assert len(blocks) == 1
    assert blocks[0].ok and blocks[0].reason == ""
    assert blocks[0].gateable is True
    assert blocks[0].residual_l1 == pytest.approx(0.0, abs=1e-12)


def test_corrupted_gram_entry_is_refused_with_typed_reason():
    g = _good_gram()
    g[0, 0] -= 1.0  # a full unit of coefficient error, and a negative eigenvalue
    screen = screen_design_candidate(None, [_entry(g)], run_sphere=False)
    assert screen.ok is False
    assert screen.reason == GRAM_VALIDATION_FAILED
    assert screen.worst_block == "V_oct0"
    assert "gram_validation_failed(block=V_oct0" in screen.detail
    assert screen.n_blocks_failed == 1


def test_worst_block_is_named_among_several():
    entries = [
        _entry(_good_gram(), name="V_oct0"),
        _entry(_good_gram() - np.diag([0.5, 0.0, 0.0]), name="V_oct1"),
        _entry(_good_gram() - np.diag([2.0, 0.0, 0.0]), name="V_oct2"),
    ]
    screen = screen_design_candidate(None, entries, run_sphere=False)
    assert screen.ok is False
    assert screen.worst_block == "V_oct2"       # lowest min_eig of the failures
    assert screen.n_blocks == 3 and screen.n_blocks_failed == 2
    assert screen.gram_min_eig == pytest.approx((1.0 - EPS) - 2.0, abs=1e-9)


def test_gram_mass_outside_the_target_support_is_charged():
    """The design backend's row-wise screen cannot see this; the filter can.

    Every row constraint is satisfied (the target monomials match exactly), but
    the Gram carries mass on a monomial the target does not contain, so zᵀQz is
    NOT the target and the SOS conclusion does not follow. This is the
    ``power_rate`` mechanism, in miniature.
    """
    g = _good_gram()
    g[1, 1] = 3.0                                # adds 3·y1²y2², absent from the target
    ok, min_eig, resid, reason = _screen_one(g)
    assert min_eig >= 0.0                        # PSD: an eigenvalue screen sees nothing
    assert resid == pytest.approx(3.0, abs=1e-12)
    assert ok is False and reason.startswith("gram_margin_exceeded")


def _screen_one(gram):
    b = screen_gram_bundle([_entry(gram)])[0]
    return b.ok, b.min_eig, b.residual_l1, b.reason


def test_solver_noise_inside_the_budget_is_accepted():
    g = _good_gram()
    g[0, 0] += 1e-8                              # far inside ε/2 = 5e-5
    ok, _e, resid, reason = _screen_one(g)
    assert ok is True and reason == "" and resid < 1e-7


def test_zero_margin_blocks_are_reported_but_cannot_refuse():
    """``eps == 0`` blocks (P2-02 typed reasons) never trigger the refusal.

    The exact gate refuses them on ``eps_reason`` before it reads their Gram,
    so they cannot enter X; there is also no margin to charge against.
    """
    bad_w = _entry(np.diag([1e6, 0.0, 0.0]), name="W_oct0_v0", block="W",
                   eps=0.0, eps_reason="w_degree_not_positive_even")
    screen = screen_design_candidate(None, [_entry(_good_gram()), bad_w],
                                     run_sphere=False)
    assert screen.ok is True and screen.reason == SCREEN_OK
    assert screen.n_blocks == 2 and screen.n_blocks_gateable == 1
    assert screen.n_blocks_failed == 1           # reported, not fatal
    w = [b for b in screen.blocks if b.block == "W"][0]
    assert w.gateable is False
    assert w.eps_reason == "w_degree_not_positive_even"
    assert w.residual_l1 > 1e5


def test_record_is_json_serializable_and_carries_per_block_numbers():
    import json
    screen = screen_design_candidate(None, [_entry(_good_gram())],
                                     run_sphere=False)
    rec = screen.as_record()
    json.dumps(rec)                              # must not raise
    assert rec["gram"]["n_blocks"] == 1
    blk = rec["gram"]["blocks"][0]
    assert set(blk) >= {"name", "gram_min_eig", "gram_residual_l1", "eps",
                        "eps_reason", "gateable", "ok", "reason"}


# ---------------------------------------------------------------------------
# Sphere pre-check (MOSEK-free; the embedded invalid published certificates)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,floor", [("continuous_twisting", -3.0e-3),
                                        ("levant3d", -5.0e-3)])
def test_invalid_published_certificates_refused_on_the_sphere(name, floor):
    screen = screen_design_candidate(build_embedded_V(name), None)
    assert screen.ok is False
    assert screen.reason == SPHERE_PRECHECK_FAILED
    assert screen.sphere_min_V < floor
    assert f"{SPHERE_PRECHECK_FAILED}(min_V=" in screen.detail
    # The sphere filter fires FIRST: no Gram work is attempted after it.
    assert screen.gram_ran is False


def test_pidsmc_published_certificate_is_pd_on_the_sphere():
    """The third invalid certificate is PD: filter 1 does not fire on it.

    Its rejection is at the W stage (E4, and ``test_verifier_soundness``),
    which is why the screen has two filters and not one.
    """
    screen = screen_design_candidate(build_embedded_V("pidsmc"), None)
    assert screen.ok is True
    assert screen.sphere_min_V > 0.0
    assert screen.gram_skipped == "no_gram_bundle"


# ---------------------------------------------------------------------------
# Mode plumbing
# ---------------------------------------------------------------------------
def test_screen_modes_are_validated():
    assert SCREEN_MODES == ("reject", "warn", "off")
    for m in SCREEN_MODES:
        assert resolve_screen_mode(m) == m
    with pytest.raises(ValueError):
        resolve_screen_mode("silent")
    with pytest.raises(ValueError):
        JointSOSOptimizer(weights_r=(1, 1), design_screen="silent")


def test_default_mode_is_reject():
    assert JointSOSOptimizer(weights_r=(1, 1)).design_screen == "reject"


def test_infeasible_result_is_not_screened():
    screen = screen_design_result(JointSOSResult(feasible=False))
    assert screen.ok is True
    assert screen.gram_skipped == "not_feasible"
    assert screen.sphere_ran is False


def test_missing_bundle_cannot_refuse():
    screen = screen_design_candidate(build_embedded_V("pidsmc"), None)
    assert screen.ok is True and screen.gram_ran is False
    assert screen.gram_skipped == "no_gram_bundle"
    screen = screen_design_candidate(build_embedded_V("pidsmc"), [])
    assert screen.ok is True and screen.gram_skipped == "empty_gram_bundle"


def test_empty_screen_defaults():
    s = DesignScreen(ok=True, reason=SCREEN_OK)
    assert s.n_blocks == 0 and s.n_blocks_failed == 0
    assert s.seconds == 0.0


# ---------------------------------------------------------------------------
# End-to-end wiring on the MOSEK_DIRECT design path
# ---------------------------------------------------------------------------
@needs_mosek
def test_design_result_carries_the_screen_record_by_default():
    res = _design("sta")
    assert res.feasible is True
    rec = res.design_screen
    assert rec is not None and rec["ok"] is True and rec["reason"] == ""
    assert rec["sphere"]["ran"] is True and rec["sphere"]["min_V"] > 0.0
    assert rec["gram"]["ran"] is True
    assert rec["gram"]["n_blocks"] == len(res.gram_certificates)
    assert rec["gram"]["n_blocks_failed"] == 0
    assert rec["seconds"] < 5.0


@needs_mosek
@pytest.mark.parametrize("name", sorted(DEGREES))
def test_twelve_nominal_systems_pass_both_filters(name):
    res = _design(name)
    assert res.feasible is True, res.message
    rec = res.design_screen
    assert rec["ok"] is True, rec["detail"]
    assert rec["sphere"]["min_V"] > 0.0
    # Every block of every nominal system is X-eligible and validates. Until
    # P2-09 power_rate was the exception (2 of its 4 blocks non-gateable and
    # failing on 6.16e+04 of unconstrained Gram mass); the odd-y-degree W μ
    # policy and the completion rows removed it.
    gateable_failed = [b for b in rec["gram"]["blocks"]
                       if b["gateable"] and not b["ok"]]
    assert gateable_failed == []
    assert rec["gram"]["n_blocks_gateable"] == rec["gram"]["n_blocks"]
    assert rec["gram"]["n_blocks_failed"] == 0


@needs_mosek
def test_screen_mode_does_not_move_a_passing_design_point():
    out = {}
    for mode in SCREEN_MODES:
        r = _design("sta", design_screen=mode)
        out[mode] = (r.feasible, r.solver_status,
                     tuple(round(v, 12) for v in r.coefficients.values()))
    assert out["off"] == out["warn"] == out["reject"]
    assert _design("sta", design_screen="off").design_screen is None


@needs_mosek
def test_corrupted_design_gram_is_refused_end_to_end():
    """A real design bundle, one Gram entry corrupted, screened again."""
    res = _design("pidsmc", design_screen="off")
    assert res.feasible and res.design_screen is None
    bundle = [dict(e) for e in res.gram_certificates]
    victim = next(e for e in bundle if e["block"] == "W")
    g = np.array(victim["gram"], copy=True)
    g[0, 0] -= 1.0
    victim["gram"] = g
    screen = screen_design_candidate(res.V_gf, bundle)
    assert screen.ok is False
    assert screen.reason == GRAM_VALIDATION_FAILED
    assert screen.worst_block == victim["name"]


@needs_mosek
def test_capture_off_leaves_the_sphere_filter_running():
    res = _design("sta", capture_design_grams=False)
    assert res.feasible is True and res.gram_certificates is None
    rec = res.design_screen
    assert rec["sphere"]["ran"] is True
    assert rec["gram"]["ran"] is False
    assert rec["gram"]["skipped"] == "no_gram_bundle"


# ---------------------------------------------------------------------------
# Review finding 7 (2026-09-05): the module docstring must state the ACTIVE
# default threshold. It said the screen accepts at ``eta*C1 + ||E||1 <= eps/2``
# while every entry point defaults to ``budget_policy="achieved"``, i.e.
# ``charge < eps``. An operator reading it would credit accepted design
# witnesses with the frozen note's half margin, which they need not satisfy.
#
# These tests pin the documentation to the code, and both to the exact gate, so
# the two cannot drift apart again.
# ---------------------------------------------------------------------------

def test_screen_default_budget_policy_is_the_exact_gate_default():
    """One default, spelled the same way in both modules."""
    import inspect

    from gf_framework.verification import design_screen as ds
    from gf_framework.verification import exact_gate as eg

    assert eg.MARGIN_POLICY_ACHIEVED == "achieved"
    assert eg.resolve_margin_policy(eg.MARGIN_POLICY_ACHIEVED) == "achieved"
    # the gate's own resolved default, read off its public entry points
    for fn in (eg.exact_validate_octant, eg.validate_one_capture,
               eg.exact_verify_captures):
        default = inspect.signature(fn).parameters["margin_policy"].default
        assert eg.resolve_margin_policy(default) == "achieved", fn.__name__
    # every screen entry point defaults to the same policy
    for fn in (ds.screen_gram_bundle, ds.screen_design_candidate,
               ds.screen_design_result):
        default = inspect.signature(fn).parameters["budget_policy"].default
        assert ds.resolve_budget_policy(default) == "achieved", fn.__name__
    for cls in (ds.BlockScreen, ds.DesignScreen):
        fields = {f.name: f for f in dataclasses.fields(cls)}
        assert fields["budget_policy"].default == "achieved", cls.__name__
    # and the vocabularies are the same object, verbatim
    assert tuple(ds.BUDGET_POLICIES) == tuple(eg.MARGIN_POLICIES)


def test_screen_docstring_states_the_active_threshold_not_the_historical_one():
    """The docstring names ``charge < eps`` as the default and marks the
    ``eps/2`` half-margin as historical."""
    from gf_framework.verification import design_screen as ds

    doc = ds.__doc__ or ""
    assert "charge < eps" in doc
    # the historical policy is still named, but only as historical
    assert "eps/2" in doc
    half_line = next(line for line in doc.splitlines()
                     if "eps/2" in line and "charge" in line)
    assert "historical" in doc.lower()
    assert "budget_policy" in doc
    # the screen advertises that it mirrors the gate's vocabulary
    assert "MARGIN_POLICIES" in doc
    # and it no longer claims eps/2 as what acceptance NEEDS
    assert "acceptance needs" not in doc
    assert half_line  # the half-margin sentence exists and is the qualified one


def test_screen_and_gate_thresholds_agree_numerically():
    """Not just the names: the same charge decides the same way on both
    sides, for both policies."""
    from fractions import Fraction

    from gf_framework.verification import exact_gate as eg
    from gf_framework.verification.joint_sos import _validate_gram_certificate

    eps = 1e-6
    basis = [(1,)]
    target = {(2,): 1.0}
    for policy, charge_frac in (("achieved", 0.75), ("half", 0.75),
                                ("achieved", 0.25), ("half", 0.25),
                                ("achieved", 1.5), ("half", 1.5)):
        charge = charge_frac * eps
        gram = np.array([[1.0 - charge]])  # ||E||_1 = charge, eta = 0
        ok, _min_eig, _resid, _why = _validate_gram_certificate(
            target, gram, basis, 1, eps, 1e-7, budget_policy=policy)
        budget = eg.margin_budget(Fraction(eps), margin_policy=policy)
        gate_ok = (Fraction(charge) <= budget if policy == "half"
                   else Fraction(charge) < budget)
        assert ok == gate_ok, (policy, charge_frac)
