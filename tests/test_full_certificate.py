"""End-to-end soundness of full (V AND W) certification via the standard-SOS path.

Locks both directions:
  * genuinely stable systems CERTIFY (stable STA), with V and W both SOS;
  * the published certificates that fail the decrease test are REJECTED at W
    (the stored PID-SMC / STA coefficients whose V̇ is not negative-definite).

MOSEK-free (CLARABEL oracle). No exact-rational path: the SDP solver is the
accepted oracle for standard SOS (Sanchez-Moreno 2018 §5.3).
"""
import json
from fractions import Fraction
from pathlib import Path

import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.interfaces import SystemSpec, SelectionResult, VerificationConfig
from gf_framework.systems.sta import get_sta_known_lyapunov, get_sta_dynamics_gf
from gf_framework.verification.verifier import verify_lyapunov
from gf_framework.verification.joint_sos import verify_w_given_numeric
from gf_framework.discovery.universal_discovery import create_sta_spec, create_pidsmc_spec

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts/_archive/publication/results_sparse_20260421T093238Z"


def _cfg():
    cfg = VerificationConfig()
    cfg.solver = "CLARABEL"
    return cfg


def _sel(V):
    return SelectionResult(active_terms=[], coefficients={}, V_gf=V,
                           numerical_valid=True, max_V_violation=0.0, max_W_violation=0.0)


def load_V(name):
    r = json.load(open(ARTIFACTS / f"{name}_seed42.json"))["result"]
    coeffs = list(r["coefficients"].values())
    terms = [GFTerm(coefficient=float(c),
                    exponents=tuple(Fraction(e) for e in t["exponents"]),
                    term_types=tuple(t["term_types"]))
             for c, t in zip(coeffs, r["term_specs"])]
    return GeneralizedForm(degree_m=r["degree_m"], weights_r=tuple(r["weights_r"]), terms=terms)


# ---- genuinely stable systems certify --------------------------------------

@pytest.mark.parametrize("k1,k2", [(3.0, 1.0), (2.0, 1.0)])
def test_stable_sta_fully_certifies(k1, k2):
    V = get_sta_known_lyapunov(k1, k2)
    spec = SystemSpec(weights_r=(2, 1), degree_m=2, dynamics_gf=get_sta_dynamics_gf(k1, k2))
    cert = verify_lyapunov(_sel(V), spec, _cfg(), verbose=False)
    assert cert.V_verified and cert.W_verified


# ---- the published coefficients themselves are NOT valid certificates ------
# verify_w_given_numeric checks the STORED coefficients directly (no
# re-optimization), so a V whose V̇ is not negative-definite is rejected at W.

@pytest.mark.skipif(not ARTIFACTS.exists(), reason="artifacts absent")
@pytest.mark.parametrize("name,spec_fn", [("sta", create_sta_spec), ("pidsmc", create_pidsmc_spec)])
def test_published_coefficients_fail_W(name, spec_fn):
    V = load_V(name)
    spec = spec_fn()
    res = verify_w_given_numeric(V, spec.dynamics_gf, V.weights_r, V.degree_m, solver="CLARABEL")
    assert res.feasible is False, (
        f"{name}: stored coefficients wrongly pass the standard-SOS W check"
    )
