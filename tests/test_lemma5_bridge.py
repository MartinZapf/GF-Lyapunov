"""The continuous-time stability bridge is Sanchez-Moreno Lemma 5 (Filippov), NOT
Clarke (Clarke belongs to the out-of-scope discretization setting).

Lemma 5 requires V ∈ C¹ (restriction R2). For a C¹ V:
  * ∇V is continuous across the switching surfaces, so V̇<0 off the discontinuity
    set S of f (the open hyperoctants) already implies asymptotic stability — no
    surface analysis needed;
  * W = −∇V·f has only non-negative exponents, so no clearing monomial is needed.

These tests certify the S-M Example 3 system (finite-time double integrator) end to
end as a full asymptotic-stability proof, and check the honest scope for non-C¹ V.
MOSEK-free (CLARABEL oracle + exact rational certificates).
"""
from fractions import Fraction

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.interfaces import SystemSpec, SelectionResult, VerificationConfig
from gf_framework.verification.verifier import verify_lyapunov, classify_smoothness


def _example3_spec(k1=1.0, k2=1.0):
    # r=(3,2), kappa=-1: x1dot=x2 ; x2dot=-k1 ceil(x1)^{1/3} - k2 ceil(x2)^{1/2}
    f1 = GeneralizedForm(2, (3, 2), [GFTerm(1.0, (Fraction(0), Fraction(1)), ("abs", "signed"))])
    f2 = GeneralizedForm(1, (3, 2), [
        GFTerm(-k1, (Fraction(1, 3), Fraction(0)), ("signed", "abs")),
        GFTerm(-k2, (Fraction(0), Fraction(1, 2)), ("abs", "signed")),
    ])
    return SystemSpec(weights_r=(3, 2), degree_m=5, dynamics_gf=[f1, f2])


def _example3_C1_LF():
    # S-M eq.(8) with alpha=[7,1,5] (their Example 7); m=5 > max r = 3 ⇒ C¹.
    return GeneralizedForm(5, (3, 2), [
        GFTerm(7.0, (Fraction(5, 3), Fraction(0)), ("abs", "abs")),
        GFTerm(1.0, (Fraction(1), Fraction(1)), ("signed", "signed")),
        GFTerm(5.0, (Fraction(0), Fraction(5, 2)), ("abs", "abs")),
    ])


def _cfg():
    cfg = VerificationConfig()
    cfg.solver = "CLARABEL"
    return cfg


def _sel(V):
    return SelectionResult(active_terms=[], coefficients={}, V_gf=V,
                           numerical_valid=True, max_V_violation=0.0, max_W_violation=0.0)


def test_example3_C1_LF_is_C1():
    assert classify_smoothness(_example3_C1_LF())["regularity"] == "C1"


def test_example3_full_asymptotic_stability_via_lemma5():
    cert = verify_lyapunov(_sel(_example3_C1_LF()), _example3_spec(), _cfg(), verbose=False)
    assert cert.V_verified and cert.W_verified
    assert cert.regularity == "C1"
    assert cert.surface_certified is True
    assert cert.surface_scope == "lemma5_filippov"
    # C¹ V ⇒ W has no negative exponents ⇒ plain SOS, no weighted-sphere clearing.
    assert "weighted_sphere" not in cert.W_method


def test_non_c1_V_is_flagged_out_of_scope():
    # A kinked V (|x1|^1) is non-C¹ ⇒ outside the continuous-time Lemma-5 method.
    Vkink = GeneralizedForm(2, (2, 1), [
        GFTerm(1.0, (Fraction(1), Fraction(0)), ("abs", "abs")),
        GFTerm(0.45, (Fraction(0), Fraction(2)), ("abs", "abs")),
    ])
    assert classify_smoothness(Vkink)["regularity"] != "C1"
