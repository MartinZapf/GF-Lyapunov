"""Typing of the CTA reference Lyapunov function (WP P5-01).

``systems/continuous_twisting.py::get_cta_known_lyapunov`` transcribes the
hand Lyapunov function of Torres-Gonzalez, Fridman and Moreno 2017, eq. (5):

    V = a1 |x1|^(5/3) + a2 x1 x2 + a3 |x2|^(5/2)
        + a4 x1 ceil(x3)^2 - a5 x2 x3^3 + a6 |x3|^5

The fourth term used to be typed ``('signed', 'abs', 'abs')``, with a comment
reading "ceil(x3)^2 = |x3|^2 (signed squared = absolute)". That reads ceil as
an ordinary power. In the GF convention ceil(x)^a = |x|^a sign(x) for EVERY
exponent a, so ceil(x3)^2 = |x3|^2 sign(x3) and the typing is
``('signed', 'abs', 'signed')``.

What is pinned here:

* the shipped typing of the alpha4 term, term by term, and the fact that BOTH
  typings are members of the mu-filtered C1 library at m = 5, so this was a
  transcription error and never a representability question;
* with the CORRECTED typing the published coefficient vector certifies under
  the strict CLARABEL gate and under the strict MOSEK gate, and the exact
  Proposition-7 phase passes;
* with the OLD typing both gates refuse it and the exact phase does not pass.

The gate/exact tests are the slow half and are marked; the MOSEK arm is
skipped where MOSEK is not installed.
"""
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.algebra.enumerate import enumerate_diagonal_terms
from gf_framework.discovery import universal_discovery as ud
from gf_framework.discovery.gf_discovery import mu_filtered_cross
from gf_framework.systems.continuous_twisting import (
    CTA_DEGREE,
    CTA_WEIGHTS,
    get_cta_known_lyapunov,
)
from gf_framework.verification import exact_phase as ep
from gf_framework.verification.mosek_direct import MOSEK_AVAILABLE
from gf_framework.verification.verifier import certify_discovered_gf

needs_mosek = pytest.mark.skipif(not MOSEK_AVAILABLE,
                                 reason="MOSEK not installed")

#: The alpha4 term, in both typings. ``ALPHA4_CORRECT`` is what the module
#: ships since P5-01; ``ALPHA4_OLD`` is the pre-P5-01 mis-typing.
ALPHA4_EXPONENTS = (Fraction(1), Fraction(0), Fraction(2))
ALPHA4_CORRECT = ('signed', 'abs', 'signed')
ALPHA4_OLD = ('signed', 'abs', 'abs')

#: Benchmark gains, printed rationals (TGFM 2017 Lemma 1 for mu = 1).
GAINS = (Fraction(274, 5), Fraction(27), Fraction(57, 10), Fraction(3))


def _spec():
    return ud.create_cta_spec(*GAINS)


def _library(spec, m):
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    _, _, cross = mu_filtered_cross(m, spec.weights_r, diag, spec.dynamics_gf)
    return list(diag) + list(cross)


def _retyped_old(V):
    """The same V with the alpha4 term restored to the pre-P5-01 typing."""
    terms = []
    n_changed = 0
    for t in V.terms:
        if (t.exponents == ALPHA4_EXPONENTS
                and t.term_types == ALPHA4_CORRECT):
            t = GFTerm(coefficient=t.coefficient, exponents=t.exponents,
                       term_types=ALPHA4_OLD)
            n_changed += 1
        terms.append(t)
    assert n_changed == 1, f"expected exactly one alpha4 term, found {n_changed}"
    return GeneralizedForm(degree_m=V.degree_m, weights_r=V.weights_r,
                           terms=terms)


def _key(t):
    return (tuple(str(e) for e in t.exponents), tuple(t.term_types))


# ---------------------------------------------------------------------------
# solver-free
# ---------------------------------------------------------------------------
def test_alpha4_term_is_signed_in_the_third_variable():
    """ceil(x3)^2 = |x3|^2 sign(x3): the third slot is 'signed', not 'abs'."""
    V = get_cta_known_lyapunov()
    hits = [t for t in V.terms if t.exponents == ALPHA4_EXPONENTS]
    assert len(hits) == 1
    assert hits[0].term_types == ALPHA4_CORRECT
    assert hits[0].term_types != ALPHA4_OLD


def test_reference_V_shape_is_the_published_one():
    V = get_cta_known_lyapunov()
    assert V.degree_m == CTA_DEGREE == 5
    assert tuple(V.weights_r) == tuple(CTA_WEIGHTS) == (3, 2, 1)
    assert len(V.terms) == 6
    # r-homogeneity of weighted degree 5, term by term.
    for t in V.terms:
        deg = sum(Fraction(e) * r for e, r in zip(t.exponents, V.weights_r))
        assert deg == Fraction(5), (t.exponents, deg)


def test_both_typings_sit_in_the_mu_filtered_library_at_m5():
    """A typing error, not a representability question: the old typing is a
    library member too, so the m = 5 library was never the obstruction."""
    spec = _spec()
    lib_keys = {_key(t) for t in _library(spec, 5)}
    V = get_cta_known_lyapunov()
    assert all(_key(t) in lib_keys for t in V.terms)
    assert all(_key(t) in lib_keys for t in _retyped_old(V).terms)


def test_get_cta_known_lyapunov_has_no_callers_in_the_shipped_paths():
    """The mis-typing contaminated nothing that ships: the reference V is a
    hand-comparison object with no caller under src/ or scripts/."""
    root = Path(__file__).resolve().parents[1]
    hits = []
    for sub in ("src", "scripts"):
        for path in (root / sub).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "get_cta_known_lyapunov" not in text:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if "get_cta_known_lyapunov" not in line:
                    continue
                if line.lstrip().startswith("def get_cta_known_lyapunov"):
                    continue
                hits.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert hits == [], "unexpected caller(s):\n" + "\n".join(hits)


# ---------------------------------------------------------------------------
# gates and the exact phase
# ---------------------------------------------------------------------------
def _gate(V, spec, solver):
    return certify_discovered_gf(V, spec.weights_r, CTA_DEGREE,
                                 spec.dynamics_gf, solver=solver,
                                 strict_solver=True,
                                 dynamics_gf_vertices=None)


@pytest.mark.slow
def test_corrected_typing_certifies_under_clarabel():
    spec = _spec()
    cert = _gate(get_cta_known_lyapunov(), spec, "CLARABEL")
    assert cert.surface_certified is True


@pytest.mark.slow
def test_old_typing_is_refused_by_clarabel():
    spec = _spec()
    cert = _gate(_retyped_old(get_cta_known_lyapunov()), spec, "CLARABEL")
    assert cert.surface_certified is False


@needs_mosek
@pytest.mark.slow
def test_corrected_typing_certifies_under_mosek():
    spec = _spec()
    cert = _gate(get_cta_known_lyapunov(), spec, "MOSEK")
    assert cert.surface_certified is True


@needs_mosek
@pytest.mark.slow
def test_old_typing_is_refused_by_mosek():
    spec = _spec()
    cert = _gate(_retyped_old(get_cta_known_lyapunov()), spec, "MOSEK")
    assert cert.surface_certified is False


@pytest.mark.slow
def test_exact_phase_passes_with_the_corrected_typing_and_fails_with_the_old():
    """The Proposition-7 exact phase separates the two typings as well.

    The exact charge ratio is witness-dependent, so only the VERDICT is
    pinned; the recorded numbers live in
    ``artifacts/campaign_2026-08/E9_gain_regions/cta_m5_2026-09-03/``.
    """
    spec = _spec()
    n = len(spec.weights_r)
    good = ep.run_exact_phase(get_cta_known_lyapunov(), [spec.dynamics_gf], n,
                              epsilon=1e-6, solver="CLARABEL")
    assert good.exact_pass is True
    assert good.worst_charge_ratio < 1.0
    bad = ep.run_exact_phase(_retyped_old(get_cta_known_lyapunov()),
                             [spec.dynamics_gf], n,
                             epsilon=1e-6, solver="CLARABEL")
    assert bad.exact_pass is False
