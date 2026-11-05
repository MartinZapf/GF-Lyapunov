"""The publication claim must come from the exact-rational authoritative gate
(``certify_discovered_gf``), never from an internal discovery flag.

These tests pin the two decisive behaviours:
  * a C¹ candidate (restriction R2) that is PD with V̇<0 off the discontinuity
    set earns a COMPLETE Lemma-5 (Filippov) asymptotic-stability certificate;
  * a non-C¹ candidate is honestly downgraded — it must NOT be reported as
    surface-certified even when V>0 ∧ V̇<0 hold on the open orthants.

MOSEK-free: CLARABEL oracle + exact rational certificates.
"""
from fractions import Fraction

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification.verifier import certify_discovered_gf


def _example3_dynamics(k1=1.0, k2=1.0):
    # r=(3,2), kappa=-1: x1dot=x2 ; x2dot=-k1 ceil(x1)^{1/3} - k2 ceil(x2)^{1/2}
    f1 = GeneralizedForm(2, (3, 2), [GFTerm(1.0, (Fraction(0), Fraction(1)), ("abs", "signed"))])
    f2 = GeneralizedForm(1, (3, 2), [
        GFTerm(-k1, (Fraction(1, 3), Fraction(0)), ("signed", "abs")),
        GFTerm(-k2, (Fraction(0), Fraction(1, 2)), ("abs", "signed")),
    ])
    return [f1, f2]


def _example3_C1_LF():
    # S-M Example 7 alpha=[7,1,5]; m=5 > max r = 3 ⇒ C¹ (restriction R2).
    return GeneralizedForm(5, (3, 2), [
        GFTerm(7.0, (Fraction(5, 3), Fraction(0)), ("abs", "abs")),
        GFTerm(1.0, (Fraction(1), Fraction(1)), ("signed", "signed")),
        GFTerm(5.0, (Fraction(0), Fraction(5, 2)), ("abs", "abs")),
    ])


def test_c1_candidate_earns_full_lemma5_certificate():
    cert = certify_discovered_gf(
        _example3_C1_LF(), (3, 2), 5, _example3_dynamics(), solver="CLARABEL"
    )
    assert cert.V_verified and cert.W_verified
    assert cert.regularity == "C1"
    assert cert.surface_certified is True
    assert cert.surface_scope == "lemma5_filippov"


def test_non_c1_candidate_is_honestly_downgraded():
    # |x1|^1 kink (r=(2,1), m=2): even if V>0 ∧ V̇<0 on open orthants, V ∉ C¹ ⇒
    # the Lemma-5 surface bridge must NOT be claimed.
    f = GeneralizedForm(1, (2, 1), [GFTerm(1.0, (Fraction(0), Fraction(1)), ("abs", "signed"))])
    Vkink = GeneralizedForm(2, (2, 1), [
        GFTerm(1.0, (Fraction(1), Fraction(0)), ("abs", "abs")),
        GFTerm(0.5, (Fraction(0), Fraction(2)), ("abs", "abs")),
    ])
    cert = certify_discovered_gf(Vkink, (2, 1), 2, [f, f], solver="CLARABEL")
    assert cert.regularity != "C1"
    assert cert.surface_certified is False
    assert cert.surface_scope != "lemma5_filippov"
