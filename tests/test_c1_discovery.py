"""Deterministic C¹-restricted discovery (Sanchez-Moreno, Lemma 5).

Searching the classical (often non-C¹) homogeneous structures yields candidates
the sound verifier must reject. Searching the C¹ (R2) class directly —
escalating the LF degree m with fractional exponents — recovers smooth GF Lyapunov
functions that certify under Lemma 5. These tests pin two decisive cases:

  * a system whose published LF is non-C¹ (STA at m=2) gains a *smooth* certified
    LF at a higher degree (m=3);
  * a C¹-feasible system (Example 3) certifies at its published degree.

Each certificate is the authoritative exact-rational gate's verdict
(surface_certified ⇒ complete continuous-time asymptotic stability). MOSEK-free.
"""
import gf_framework.discovery.universal_discovery as ud
from gf_framework.discovery.c1_discovery import discover_c1_lyapunov, is_c1_term
from gf_framework.algebra.enumerate import enumerate_diagonal_terms


def _spec(name):
    return getattr(ud, {
        "sta": "create_sta_spec",
        "example3": "create_example3_spec",
    }[name])()


def test_sta_gains_smooth_c1_lf_at_higher_degree():
    s = _spec("sta")  # r=(2,1), published m=2 (non-C¹)
    r = discover_c1_lyapunov("sta", tuple(s.weights_r), s.degree_m, s.dynamics_gf,
                             solver="CLARABEL", verbose=False)
    assert r.certified is True
    assert r.regularity == "C1"
    assert r.surface_scope == "lemma5_filippov"
    assert r.degree_used > max(s.weights_r)          # escalated past maxr=2
    assert r.claim_level == "lemma5_asymptotic_stability"


def test_example3_certifies_at_or_below_published_degree():
    s = _spec("example3")  # r=(3,2), m=5 > maxr ⇒ C¹-feasible
    r = discover_c1_lyapunov("example3", tuple(s.weights_r), s.degree_m, s.dynamics_gf,
                             solver="CLARABEL", verbose=False)
    assert r.certified is True
    assert r.regularity == "C1"
    # The search starts at the cheapest C¹ degree (maxr+1), so it certifies at or
    # below the published degree — here at m=4, not the published m=5. Cheaper is a
    # feature; require C¹ (escalated past maxr) and no worse than published.
    assert max(s.weights_r) < r.degree_used <= s.degree_m


def test_is_c1_term_rejects_kink_and_sublipschitz():
    # |x1|^1 (kink) at m=2,r=(2,1) is the max-weight diagonal term → non-C¹.
    diag = enumerate_diagonal_terms(2, (2, 1))
    flags = [is_c1_term(t, (2, 1), 2) for t in diag]
    # |x1|^{2/2}=|x1|^1 non-C¹; |x2|^{2/1}=|x2|^2 C¹
    assert flags == [False, True]
