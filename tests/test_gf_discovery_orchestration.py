"""Single-tier orchestration tests for the GF-SOS discovery pipeline.

These monkeypatch the SOS designer so the tests run fast and deterministically,
while still exercising the REAL strict Lemma-5 gate (``certify_discovered_gf``,
standard SOS). MOSEK-free (CLARABEL).

The flow under test: build the C¹ library at degree m; solve the SOS problem
over the full μ-filtered library (``_design_full_library``); re-gate strictly;
escalate m; and report an honest negative if nothing certifies.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gf_framework.discovery.universal_discovery as ud
import gf_framework.discovery.gf_discovery as gfd
from gf_framework.algebra.core import GeneralizedForm, GFTerm


def _example3_known_c1_V() -> GeneralizedForm:
    # Known C¹ GF Lyapunov function for Example 3 (double integrator with fractional
    # feedback), r=(3,2), m=5:  V = 7|x1|^{5/3} + x1 x2 + 5|x2|^{5/2}.
    return GeneralizedForm(
        degree_m=5, weights_r=(3, 2),
        terms=[
            GFTerm(coefficient=7.0, exponents=(Fraction(5, 3), Fraction(0)), term_types=("abs", "abs")),
            GFTerm(coefficient=1.0, exponents=(Fraction(1), Fraction(1)), term_types=("signed", "signed")),
            GFTerm(coefficient=5.0, exponents=(Fraction(0), Fraction(5, 2)), term_types=("abs", "abs")),
        ],
    )


def _fake_design_at(V_gf, at_m):
    """Stub for _design_full_library: return V only at the matching degree."""
    def _inner(weights_r, m, terms, dynamics_gf, solver, verbose, **kwargs):
        return V_gf if m == at_m else None
    return _inner


def test_full_library_path_certifies_globally(monkeypatch):
    # The SOS problem is solved over the full μ-filtered library; the designed V
    # passes the strict Lemma-5 gate → global asymptotic-stability claim.
    spec = ud.create_example3_spec()
    monkeypatch.setattr(gfd, "_design_full_library", _fake_design_at(_example3_known_c1_V(), 5))
    r = gfd.discover_gf(spec, solver="CLARABEL", max_extra_degree=1, verbose=False)
    assert r.certified is True
    assert r.path == "gf_sos"
    assert r.method == "gf_sos_full_library"
    assert r.claim_level == "lemma5_global_asymptotic_stability"
    assert r.regularity == "C1"
    assert r.degree_used == 5
    assert r.n_terms == 3


def test_no_certificate_is_honest_negative(monkeypatch):
    # The full-library SOS yields no certifiable candidate at any reachable degree
    # → honest negative (no regional/grid substitute).
    spec = ud.create_example3_spec()
    monkeypatch.setattr(gfd, "_design_full_library", lambda *a, **k: None)
    r = gfd.discover_gf(spec, solver="CLARABEL", max_extra_degree=1, verbose=False)
    assert r.certified is False
    assert r.path == "none"
    assert r.claim_level == "candidate"
