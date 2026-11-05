"""Tests for the fourth-order PID-like SMC system (campaign block E6, item C11).

Solver-free by design: every assertion here is symbolic (sympy, via
``algebra.symmetry``), exact-rational (``fractions.Fraction``), or purely
combinatorial (library enumeration counts). No SDP is built and no solver is
called, so this file runs on a machine without a MOSEK license.

Covered (see the C11 implementation report for the full write-up):

* r-homogeneity of every dynamics component, term by term, in exact Fractions
  (kappa + r_i, checked against ``GeneralizedForm.degree_m`` independently of
  the constructor's own internal assertion);
* oddness f(-x) = -f(x), via the S1 parity module, so E6 can use the paired
  (or nominal) reduction;
* the matched-disturbance factory produces exactly 2 inclusion vertices and
  they satisfy the endpoint-swap identity f_+(-x) = -f_-(x);
* the factory registers in ``scripts/discover_gf.py``'s SPEC_FACTORIES and
  resolves to a valid SystemSpec, but stays OUT of the ALL list (opt-in, like
  levant4d/levant5d) because the formulation is claim-hygiene quarantined;
* C1-library sanity at the cheapest admissible degree: the diagonal terms do
  not span all four variables at m=4 (R2: m > max r_i is violated on the
  weight-4 axis) but do at m=5, which is therefore the cheapest admissible
  degree; the mu-compatible cross-term library sizes at m=5 and m=6 are
  pinned to the values measured together with this factory, and are shown to
  coincide with levant4d's (same weights_r drives the same forced mu).
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import enumerate_diagonal_terms
from gf_framework.algebra.symmetry import (
    PARITY_ODD,
    verify_dynamics_odd,
    verify_endpoint_swap,
    filter_even_parity_terms,
)
from gf_framework.discovery.c1_discovery import _spans_all_vars
from gf_framework.discovery.gf_discovery import mu_filtered_cross
from gf_framework.systems.pidsmc4d import (
    PIDSMC4DSystem,
    PIDSMC4D_WEIGHTS,
    PIDSMC4D_KAPPA,
    PIDSMC4D_DEGREE,
    create_pidsmc4d_dynamics,
    get_pidsmc4d_dynamics_gf,
)

import discover_gf  # noqa: E402  (path insert above is required)


# =============================================================================
# Module constants
# =============================================================================

def test_module_constants():
    assert PIDSMC4D_WEIGHTS == (4, 3, 2, 1)
    assert PIDSMC4D_KAPPA == -1.0
    assert PIDSMC4D_DEGREE == 5


# =============================================================================
# r-homogeneity, term by term, exact Fractions
# =============================================================================

class TestHomogeneity:
    """Every component f_i must be r-homogeneous of degree kappa + r_i."""

    def test_component_degree_m_matches_kappa_plus_r(self):
        dyn = get_pidsmc4d_dynamics_gf()
        assert len(dyn) == 4
        kappa = Fraction(int(PIDSMC4D_KAPPA))
        for i, f in enumerate(dyn):
            expected = kappa + PIDSMC4D_WEIGHTS[i]
            assert Fraction(f.degree_m) == expected, (
                f"f{i + 1}: degree_m={f.degree_m}, expected kappa+r{i + 1}={expected}"
            )

    def test_every_term_matches_its_component_degree(self):
        """Independent re-check of the homogeneity constraint sum(r_i*rho_i)=m,
        term by term, in exact Fractions -- not just trusting the constructor's
        own internal assertion."""
        dyn = get_pidsmc4d_dynamics_gf()
        for i, f in enumerate(dyn):
            assert len(f.terms) >= 1
            for term in f.terms:
                weighted_deg = sum(
                    Fraction(PIDSMC4D_WEIGHTS[j]) * term.exponents[j]
                    for j in range(len(PIDSMC4D_WEIGHTS))
                )
                assert weighted_deg == Fraction(f.degree_m), (
                    f"f{i + 1} term {term}: weighted degree {weighted_deg} != "
                    f"{f.degree_m}"
                )

    def test_expected_degrees_are_3210(self):
        """f1..f4 weighted degrees are 3, 2, 1, 0 (kappa=-1, r=(4,3,2,1))."""
        dyn = get_pidsmc4d_dynamics_gf()
        assert [f.degree_m for f in dyn] == [3, 2, 1, 0]

    def test_f3_reaching_terms_exponents(self):
        """The three reaching-law terms in f3 carry exponents 1/4, 1/3, 1/2 on
        x1, x2, x3 respectively (the harmonic ladder extending pidsmc's own
        1/3, 1/2 by one step), each weighted degree 1."""
        f3 = get_pidsmc4d_dynamics_gf()[2]
        by_var = {}
        for term in f3.terms:
            for j, e in enumerate(term.exponents):
                if e != 0:
                    by_var.setdefault(j, []).append(e)
        assert by_var[0] == [Fraction(1, 4)]
        assert by_var[1] == [Fraction(1, 3)]
        assert by_var[2] == [Fraction(1, 2)]
        assert by_var[3] == [Fraction(1)]  # the linear +x4 term

    def test_f4_is_pure_sign_x1(self):
        f4 = get_pidsmc4d_dynamics_gf()[3]
        assert len(f4.terms) == 1
        term = f4.terms[0]
        assert term.exponents == (Fraction(0), Fraction(0), Fraction(0), Fraction(0))
        assert term.term_types[0] == "signed"


# =============================================================================
# Oddness (S1 module) -- required for the paired/nominal parity reduction
# =============================================================================

class TestOddness:
    def test_dynamics_are_odd(self):
        dyn = get_pidsmc4d_dynamics_gf()
        check = verify_dynamics_odd(dyn)
        assert check.holds, f"{check.message} residuals={check.residuals}"
        assert check.identity == "f(-x) = -f(x)"
        assert check.offending == ()
        assert all(r == "0" for r in check.residuals)
        assert all(p == PARITY_ODD for p in check.structural_parity)

    def test_oddness_holds_at_nondefault_gains(self):
        """Oddness is a structural property of the term_types, independent of
        the (quarantined, provisional) gain values."""
        dyn = get_pidsmc4d_dynamics_gf(k1=3.3, k2=0.7, k3=9.9, k4=2.2)
        check = verify_dynamics_odd(dyn)
        assert check.holds


# =============================================================================
# Matched-disturbance inclusion vertices (channel 3, weighted degree 0)
# =============================================================================

class TestDisturbanceVertices:
    def test_nominal_spec_has_no_vertices(self):
        spec = ud.create_pidsmc4d_spec()
        assert spec.dynamics_gf_vertices is None
        assert spec.disturbance_bound is None

    def test_perturbed_spec_has_exactly_two_vertices(self):
        spec = ud.create_pidsmc4d_spec(delta0=0.3)
        assert spec.dynamics_gf_vertices is not None
        assert len(spec.dynamics_gf_vertices) == 2
        assert spec.disturbance_bound == 0.3

    def test_endpoint_swap_identity_holds(self):
        spec = ud.create_pidsmc4d_spec(delta0=0.3)
        check = verify_endpoint_swap(spec.dynamics_gf_vertices)
        assert check.holds, check.message
        assert check.offending == ()

    def test_disturbance_enters_channel_3_x4_equation(self):
        """The card places the disturbance on the weighted-degree-0 channel,
        which is x4' (index 3): the const term must appear appended to f4,
        not f3, and f1..f3 must be byte-identical to the nominal build."""
        nominal = get_pidsmc4d_dynamics_gf()
        spec = ud.create_pidsmc4d_spec(delta0=0.3)
        plus, minus = spec.dynamics_gf_vertices
        for i in range(3):
            assert plus[i] is nominal[i] or plus[i].terms == nominal[i].terms
            assert minus[i] is nominal[i] or minus[i].terms == nominal[i].terms
        # f4 (index 3) gained one extra constant term of weighted degree 0.
        assert len(plus[3].terms) == len(nominal[3].terms) + 1
        assert len(minus[3].terms) == len(nominal[3].terms) + 1
        assert plus[3].degree_m == 0
        assert minus[3].degree_m == 0


# =============================================================================
# Factory registration in scripts/discover_gf.py
# =============================================================================

class TestFactoryRegistration:
    def test_registered_in_spec_factories(self):
        assert "pidsmc4d" in discover_gf.SPEC_FACTORIES
        assert discover_gf.SPEC_FACTORIES["pidsmc4d"] == "create_pidsmc4d_spec"

    def test_resolves_to_a_valid_spec(self):
        factory_name = discover_gf.SPEC_FACTORIES["pidsmc4d"]
        spec = getattr(ud, factory_name)()
        assert spec.name == "PID-SMC-4D"
        assert spec.weights_r == (4, 3, 2, 1)
        assert spec.degree_m == 5
        assert spec.kappa == -1.0
        assert spec.n_vars == 4

    def test_absent_from_all_list(self):
        """pidsmc4d is claim-hygiene quarantined (best-guess, unconfirmed
        formulation): opt-in only, exactly like levant4d/levant5d."""
        assert "pidsmc4d" not in discover_gf.ALL
        assert "levant4d" not in discover_gf.ALL
        assert "levant5d" not in discover_gf.ALL

    def test_all_list_unaffected_in_size(self):
        """Adding the opt-in entry must not change the 11-system benchmark matrix."""
        assert len(discover_gf.ALL) == 11


# =============================================================================
# C1-library sanity at the cheapest admissible degree
# =============================================================================

class TestC1LibrarySanity:
    """R2 (C1) needs m > max(r_i) = 4 for the diagonal terms to span every
    variable; m=5 is therefore the cheapest admissible degree (same as
    levant4d, which shares the same weights_r)."""

    def test_m4_diagonals_do_not_span_all_vars(self):
        diag = enumerate_diagonal_terms(4, PIDSMC4D_WEIGHTS, c1_only=True)
        # x1 (r=4) needs m/4 > 1, i.e. m > 4: dropped at m=4.
        assert not _spans_all_vars(diag, len(PIDSMC4D_WEIGHTS))
        assert len(diag) == 3

    def test_m5_is_the_cheapest_degree_spanning_all_vars(self):
        diag = enumerate_diagonal_terms(5, PIDSMC4D_WEIGHTS, c1_only=True)
        assert _spans_all_vars(diag, len(PIDSMC4D_WEIGHTS))
        assert len(diag) == 4
        assert PIDSMC4D_DEGREE == 5

    @pytest.mark.parametrize(
        "m,expected_diag,expected_cross_full,expected_cross_mu",
        [(5, 4, 118, 14), (6, 4, 234, 30)],
    )
    def test_library_sizes_pinned(
        self, m, expected_diag, expected_cross_full, expected_cross_mu
    ):
        """Measured together with this factory (2026-08-02); mu=(8,6,4,2) at
        both degrees, identical to levant4d's (same weights_r, same forced-mu
        rule -- see gf_discovery.forced_mu)."""
        dyn = get_pidsmc4d_dynamics_gf()
        diag = enumerate_diagonal_terms(m, PIDSMC4D_WEIGHTS, c1_only=True)
        assert len(diag) == expected_diag
        mu, cross_full, cross = mu_filtered_cross(m, PIDSMC4D_WEIGHTS, diag, dyn)
        assert mu == (8, 6, 4, 2)
        assert len(cross_full) == expected_cross_full
        assert len(cross) == expected_cross_mu

    def test_library_sizes_match_levant4d_same_weights(self):
        """Sanity cross-check: the mu-compatible library depends only on
        weights_r (via the C1 diagonal denominators), not on which dynamics
        populate it, so pidsmc4d and levant4d (same r=(4,3,2,1)) must agree."""
        from gf_framework.systems.levant4d import get_levant4d_dynamics_gf

        for m in (5, 6):
            diag_a = enumerate_diagonal_terms(m, PIDSMC4D_WEIGHTS, c1_only=True)
            _, _, cross_a = mu_filtered_cross(
                m, PIDSMC4D_WEIGHTS, diag_a, get_pidsmc4d_dynamics_gf()
            )
            diag_b = enumerate_diagonal_terms(m, PIDSMC4D_WEIGHTS, c1_only=True)
            _, _, cross_b = mu_filtered_cross(
                m, PIDSMC4D_WEIGHTS, diag_b, get_levant4d_dynamics_gf()
            )
            assert len(cross_a) == len(cross_b)

    @pytest.mark.parametrize(
        "m,expected_even_diag,expected_even_cross",
        [(5, 4, 8), (6, 4, 15)],
    )
    def test_even_parity_split_sizes(self, m, expected_even_diag, expected_even_cross):
        """The S1 even-parity filter, applied to this library: diagonals are
        always even (positive-definiteness backbone survives), cross terms
        split roughly in half."""
        dyn = get_pidsmc4d_dynamics_gf()
        diag = enumerate_diagonal_terms(m, PIDSMC4D_WEIGHTS, c1_only=True)
        _, _, cross = mu_filtered_cross(m, PIDSMC4D_WEIGHTS, diag, dyn)
        even_diag = filter_even_parity_terms(diag, PIDSMC4D_WEIGHTS)
        even_cross = filter_even_parity_terms(cross, PIDSMC4D_WEIGHTS)
        assert len(even_diag) == expected_even_diag
        assert len(even_diag) == len(diag)  # all diagonals are even
        assert len(even_cross) == expected_even_cross


# =============================================================================
# Numerical smoke tests (dynamics callables)
# =============================================================================

class TestDynamicsSmoke:
    def test_system_instantiates_with_provisional_defaults(self):
        system = PIDSMC4DSystem()
        assert system.weights == (4, 3, 2, 1)
        assert system.kappa == -1.0
        assert system.k1 == pytest.approx(1.1)
        assert system.k2 == pytest.approx(2.7)
        assert system.k3 == pytest.approx(5.345)
        assert system.k4 == pytest.approx(1.1)

    def test_dynamics_nominal_shape(self):
        system = PIDSMC4DSystem()
        x = np.random.randn(10, 4)
        dx = system.dynamics_nominal(x)
        assert dx.shape == (10, 4)

    def test_dynamics_plus_minus_shape(self):
        system = PIDSMC4DSystem()
        x = np.random.randn(5, 4)
        dp = system.dynamics_plus(x)
        dm = system.dynamics_minus(x)
        assert dp.shape == (5, 4)
        assert dm.shape == (5, 4)

    def test_top_channel_sign_behaviour(self):
        """x4' = -k4*sign(x1): negative for x1>0, positive for x1<0."""
        system = PIDSMC4DSystem(k4=2.0)
        x_pos = np.array([[1.0, 0.0, 0.0, 0.0]])
        x_neg = np.array([[-1.0, 0.0, 0.0, 0.0]])
        dx_pos = system.dynamics_nominal(x_pos)
        dx_neg = system.dynamics_nominal(x_neg)
        assert dx_pos[3] < 0
        assert dx_neg[3] > 0

    def test_create_pidsmc4d_dynamics_pair(self):
        dynamics_plus, dynamics_minus = create_pidsmc4d_dynamics()
        x = np.array([[0.5, -0.3, 0.2, -0.1]])
        dp = dynamics_plus(x)
        dm = dynamics_minus(x)
        assert dp.shape == (4,) or dp.shape == (1, 4)
        assert dm.shape == (4,) or dm.shape == (1, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# --gains CLI (WP P1-04): the gain-override path E6 needed
# =============================================================================

class TestGainOverrideCLI:
    """``scripts/discover_gf.py`` used to call the spec factory with zero
    arguments, so the E6 candidate gain sets could only be run from an
    out-of-repo copy of ``run_one`` (rescued to
    ``scripts/campaign/e6/discover_e6_variant.py``). These tests pin the
    in-repo replacement. Solver-free: ``run_one`` is exercised with
    ``discover_gf.discover_gf`` stubbed out, so no SDP is built.
    """

    C_GAINS = "3,4.160167646103808,3.0594733539832584,1.1"

    def test_gain_parameters_exclude_delta0(self):
        assert discover_gf.gain_parameters("pidsmc4d") == ["k1", "k2", "k3", "k4"]
        assert "delta0" not in discover_gf.gain_parameters("levant4d")

    def test_parse_gains_maps_candidate_c_in_order(self):
        """Candidate C of campaign block E6, verbatim from
        ``E6_pidsmc4d/round_bc/C/systems/pidsmc4d_C.json``."""
        assert discover_gf.parse_gains("pidsmc4d", self.C_GAINS) == {
            "k1": 3.0,
            "k2": 4.160167646103808,
            "k3": 3.0594733539832584,
            "k4": 1.1,
        }

    @pytest.mark.parametrize("bad", ["3,4,3", "3,4,3,1.1,9", "3,4,x,1.1"])
    def test_parse_gains_refuses_bad_input(self, bad):
        """A wrong count must never be zipped silently: that would run a
        different system under the caller's label."""
        with pytest.raises(SystemExit):
            discover_gf.parse_gains("pidsmc4d", bad)

    def test_gains_tag_is_deterministic_and_discriminating(self):
        a = discover_gf.gains_tag(discover_gf.parse_gains("pidsmc4d", self.C_GAINS))
        b = discover_gf.gains_tag(discover_gf.parse_gains("pidsmc4d", self.C_GAINS))
        c = discover_gf.gains_tag(
            discover_gf.parse_gains("pidsmc4d", "1.1,2.7,5.345,1.1"))
        assert a == b and a != c
        assert len(a) == 8

    def test_run_one_forwards_gains_to_the_factory(self, monkeypatch):
        """The gains must reach ``create_pidsmc4d_spec`` and be recorded."""
        seen = {}

        def fake_discover_gf(spec, **kwargs):
            seen["dynamics"] = spec.dynamics_gf
            return types.SimpleNamespace(
                degree_used=0, degrees_tried=[5, 6, 7, 8, 9], certified=False,
                path="none", method="none", claim_level="candidate",
                regularity="", surface_scope="", n_terms=0, mu=[],
                c1_library_diag=0, c1_library_cross=0, c1_library_cross_full=0,
                sdp_solves=0, total_time=1.0, message="stub",
                degree_attempts=[], gram_min_eig=None, gram_residual_l1=None,
                epsilon_used=None, V_gf=None, coefficients=None)

        monkeypatch.setattr(discover_gf, "discover_gf", fake_discover_gf)
        gains = discover_gf.parse_gains("pidsmc4d", self.C_GAINS)
        d = discover_gf.run_one("pidsmc4d", "MOSEK", verbose=False,
                                time_budget=1.0, max_extra_degree=4,
                                gains=gains)
        assert d["gains"] == gains
        # The stubbed factory call is the real one, so the candidate-C gains
        # are visible in the dynamics the spec carries.
        expected = ud.create_pidsmc4d_spec(**gains)
        assert str(seen["dynamics"]) == str(expected.dynamics_gf)
        assert str(seen["dynamics"]) != str(ud.create_pidsmc4d_spec().dynamics_gf)

    def test_run_one_without_gains_records_no_gains_field(self, monkeypatch):
        """Records from published-gain runs stay byte-for-byte unchanged."""
        monkeypatch.setattr(
            discover_gf, "discover_gf",
            lambda spec, **kw: types.SimpleNamespace(
                degree_used=0, degrees_tried=[], certified=False, path="none",
                method="none", claim_level="candidate", regularity="",
                surface_scope="", n_terms=0, mu=[], c1_library_diag=0,
                c1_library_cross=0, c1_library_cross_full=0, sdp_solves=0,
                total_time=0.0, message="stub", degree_attempts=[],
                gram_min_eig=None, gram_residual_l1=None, epsilon_used=None,
                V_gf=None, coefficients=None))
        d = discover_gf.run_one("pidsmc4d", "MOSEK", verbose=False,
                                time_budget=1.0, max_extra_degree=4)
        assert "gains" not in d

    def test_gains_flag_is_documented_in_help(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "discover_gf.py"), "--help"],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert proc.returncode == 0, proc.stderr
        assert "--gains" in proc.stdout

    def test_gains_requires_exactly_one_controller(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "discover_gf.py"),
             "--controllers", "pidsmc4d", "levant4d", "--gains", self.C_GAINS],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        assert proc.returncode != 0
        assert "exactly one controller" in proc.stderr
