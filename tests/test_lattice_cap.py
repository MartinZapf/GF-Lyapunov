"""The μ-compatible library must be the COMPLETE lattice slice.

Regression for the 2026-06-12 audit finding: a hard-coded max_denominator=4
silently truncated the Levant-5D slices (m=7: 59 of 61 cross terms; m=8:
103 of 111) because μ=(10,8,6,4,2) admits denominator-5/10 exponents on the
x₁ axis. ``complete_lattice_cap`` derives the cap from μ (every μ-compatible
reduced denominator divides some μ_i), which is provably exhaustive.

All pinned counts below were measured 2026-06-12 with saturation checks up to
cap 10 (5D) / cap 8 (4D) / cap 12 (3D). The n=5 enumerations at the derived
cap are minutes-heavy, hence the ``slow`` markers (run with `-m slow`).
"""
import pytest

from gf_framework.algebra.enumerate import enumerate_diagonal_terms, enumerate_cross_terms
from gf_framework.discovery.gf_discovery import (
    complete_lattice_cap,
    forced_mu,
    mu_compatible,
    mu_filtered_cross,
)
from gf_framework.systems.levant3d import get_levant3d_dynamics_gf, LEVANT3D_WEIGHTS
from gf_framework.systems.levant4d import get_levant4d_dynamics_gf, LEVANT4D_WEIGHTS
from gf_framework.systems.levant5d import get_levant5d_dynamics_gf, LEVANT5D_WEIGHTS


def _slice_size(m, weights_r, dynamics_gf):
    diag = enumerate_diagonal_terms(m, weights_r, c1_only=True)
    _, _, cross = mu_filtered_cross(m, weights_r, diag, dynamics_gf)
    return len(cross)


def test_levant5d_m6_complete_slice_count():
    """m=6 was complete even under the old cap-4 (31 terms) — must stay 31."""
    assert _slice_size(6, LEVANT5D_WEIGHTS, get_levant5d_dynamics_gf()) == 31


@pytest.mark.slow
@pytest.mark.parametrize("m,expected", [(7, 61), (8, 111)])
def test_levant5d_truncated_slices_now_complete(m, expected):
    """The slices the cap-4 bug truncated (m=7: 59 found, m=8: 103 found)."""
    assert _slice_size(m, LEVANT5D_WEIGHTS, get_levant5d_dynamics_gf()) == expected


@pytest.mark.parametrize("m,expected", [(5, 14), (6, 30), (7, 56)])
def test_levant4d_slice_counts_unchanged(m, expected):
    """cap-4 was already complete at n=4 (measured) — the fix is a no-op here."""
    assert _slice_size(m, LEVANT4D_WEIGHTS, get_levant4d_dynamics_gf()) == expected


@pytest.mark.parametrize("m,expected", [(4, 5), (5, 13), (6, 25), (7, 42), (8, 66)])
def test_levant3d_slice_counts_unchanged(m, expected):
    """cap-4 was already complete for r=(3,2,1) (measured) — no-op here too."""
    assert _slice_size(m, LEVANT3D_WEIGHTS, get_levant3d_dynamics_gf()) == expected


def test_cap_derived_from_mu():
    diag = enumerate_diagonal_terms(6, LEVANT5D_WEIGHTS, c1_only=True)
    mu = forced_mu(diag, LEVANT5D_WEIGHTS, get_levant5d_dynamics_gf())
    assert mu == (10, 8, 6, 4, 2)
    assert complete_lattice_cap(mu) == 10


def test_saturation_beyond_derived_cap_3d():
    """Enumerating past max(μ) adds no μ-compatible term (the divisibility
    proof, checked empirically on the 3D weights where it runs in seconds)."""
    weights_r = LEVANT3D_WEIGHTS
    dyn = get_levant3d_dynamics_gf()
    for m in (5, 8):
        diag = enumerate_diagonal_terms(m, weights_r, c1_only=True)
        mu, _, cross = mu_filtered_cross(m, weights_r, diag, dyn)
        bigger = enumerate_cross_terms(m, weights_r,
                                       max_denominator=2 * complete_lattice_cap(mu),
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
        bigger_mu = [t for t in bigger if mu_compatible(t, mu)]
        assert len(bigger_mu) == len(cross)
