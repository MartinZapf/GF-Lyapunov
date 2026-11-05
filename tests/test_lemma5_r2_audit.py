"""Lemma-5 / R2 enforcement audit (Sánchez–Moreno 2018).

Lemma 5 (Filippov) requires V ∈ C¹, V PD, and V̇ < 0 off the discontinuity set
S of f. Restriction R2 makes V ∈ C¹: m > maxᵢ rᵢ, signed exponents ≥ 1, abs
exponents ∈ {0} ∪ (1, ∞). These tests pin three things a referee will probe:

1. The per-factor regularity predicate (`classify_term_regularity`) at its
   boundary exponents — the certification gate and the discovery basis both
   build on it, so a drift here silently breaks the Lemma-5 claim.

2. Every published certified row satisfies the global R2 condition
   (degree_used > maxᵢ rᵢ) and is classified C¹ end-to-end.

3. The octant-boundary argument: the per-octant SOS certificate proves
   W(y;s) ≥ ε·Σⱼ yⱼᵈ on each CLOSED orthant (by continuity of the associated
   form). On a coordinate hyperplane yᵢ = 0 off the origin, Σ_{j≠i} yⱼᵈ > 0
   still, so W > 0 there — coordinate hyperplanes inside the discontinuity set
   of f are exempted by Lemma 5, all others are genuinely covered. The sampling
   test below checks W > 0 on the non-exempt hyperplanes of the published
   3D certificates (x₂ = 0 and x₃ = 0; the switching plane x₁ = 0 is exempt).
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.algebra.enumerate import classify_term_regularity, term_is_c1
from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf
from gf_framework.verification.verifier import classify_smoothness

REPO = Path(__file__).resolve().parents[1]
# Newest-first: the canonical clean pipeline matrix, then the historical
# NN-era runs (retained for the paper's ablation; the audit predicates apply
# to any published row regardless of which pipeline emitted it).
_MATRIX_DIRS = [
    REPO / "artifacts/publication/results_gf_sos_clean/systems",
    REPO / "artifacts/publication/results_neural_gf_hardened/systems",
    REPO / "artifacts/publication/results_neural_gf/systems",
]


def _published_rows():
    for d in _MATRIX_DIRS:
        if d.exists():
            return [json.load(open(p)) for p in sorted(d.glob("*.json"))]
    return []


def _build_V(row) -> GeneralizedForm:
    terms = [
        GFTerm(coefficient=float(t["coefficient"]),
               exponents=tuple(Fraction(e) for e in t["exponents"]),
               term_types=tuple(t["term_types"]))
        for t in row["V_terms"]
    ]
    return GeneralizedForm(degree_m=row["degree_used"],
                           weights_r=tuple(row["weights_r"]), terms=terms)


# ---------------------------------------------------------------------------
# 1. Per-factor regularity predicate at the boundary exponents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp,typ,expected", [
    # |x|^0 = 1 is a constant factor; ⌈x⌋^0 = sign(x) is discontinuous.
    (0, "abs", "C1"),
    (0, "signed", "discontinuous"),
    # 0 < ρ < 1: not even locally Lipschitz (STA's |x|^{1/2} class).
    (Fraction(1, 2), "abs", "sub_lipschitz"),
    (Fraction(1, 2), "signed", "sub_lipschitz"),
    (Fraction(2, 3), "signed", "sub_lipschitz"),
    # ρ = 1: |x| has a kink; ⌈x⌋^1 = x is linear (C∞).
    (1, "abs", "lipschitz_kink"),
    (1, "signed", "C1"),
    # ρ > 1: C¹ for both types (R2's "abs needs ρ = 0 or ρ > 1").
    (Fraction(4, 3), "abs", "C1"),
    (Fraction(4, 3), "signed", "C1"),
    (Fraction(101, 100), "abs", "C1"),
    (2, "abs", "C1"),
])
def test_classify_term_regularity_boundaries(exp, typ, expected):
    assert classify_term_regularity(exp, typ) == expected


def test_term_is_c1_requires_every_factor():
    """One sub-Lipschitz factor poisons the whole term."""
    assert term_is_c1((Fraction(4, 3), 1), ("abs", "signed"))
    assert not term_is_c1((Fraction(4, 3), Fraction(1, 2)), ("abs", "abs"))
    assert not term_is_c1((1, 1), ("abs", "signed"))  # |x1| kink


# ---------------------------------------------------------------------------
# 2. Published rows satisfy R2 globally and classify C¹ end-to-end
# ---------------------------------------------------------------------------

_ROWS = _published_rows()
_CERTIFIED = [r for r in _ROWS
              if r.get("certified") and r.get("claim_level", "").startswith("lemma5")]


@pytest.mark.skipif(not _CERTIFIED, reason="no published certified rows on disk")
def test_published_rows_satisfy_r2_degree_condition():
    bad = [(r["system"], r["degree_used"], max(r["weights_r"]))
           for r in _CERTIFIED if not r["degree_used"] > max(r["weights_r"])]
    assert not bad, f"rows violating m > max r_i: {bad}"


@pytest.mark.skipif(not _CERTIFIED, reason="no published certified rows on disk")
def test_published_rows_classify_c1():
    for r in _CERTIFIED:
        V = _build_V(r)
        smooth = classify_smoothness(V)
        assert smooth["regularity"] == "C1", (
            f"{r['system']}: published Lemma-5 row is {smooth['regularity']}, not C¹"
        )


# ---------------------------------------------------------------------------
# 3. Octant-boundary coverage: W > 0 on non-exempt coordinate hyperplanes
# ---------------------------------------------------------------------------

def _dynamics_for(system: str):
    import gf_framework.discovery.universal_discovery as ud
    factory = {
        "levant3d": ud.create_levant3d_spec,
        "pidsmc": ud.create_pidsmc_spec,
    }.get(system)
    return factory().dynamics_gf if factory else None


@pytest.mark.parametrize("system", ["levant3d", "pidsmc"])
def test_w_positive_on_non_exempt_hyperplanes(system):
    """W = −V̇ of the published certificate must be strictly positive on the
    hyperplanes x₂=0 and x₃=0 (off the origin and off the switching plane
    x₁=0, which Lemma 5 exempts). This is the octant-boundary argument made
    concrete: the closed-orthant SOS bound W ≥ ε·Σ yⱼᵈ does not vanish there."""
    row = next((r for r in _CERTIFIED if r["system"] == system), None)
    if row is None:
        pytest.skip(f"no certified {system} row on disk")
    V = _build_V(row)
    dynamics = _dynamics_for(system)
    W = compute_lie_derivative_gf(V, dynamics)

    rng = np.random.default_rng(0)
    mags = np.logspace(-2, 1, 8)
    worst = np.inf
    for axis in (1, 2):  # x2 = 0 and x3 = 0 hyperplanes (x1 = 0 is exempt)
        for _ in range(40):
            x = rng.standard_normal(3)
            x[axis] = 0.0
            if abs(x[0]) < 1e-6:        # stay off the switching plane
                x[0] = 0.5 * np.sign(x[0] or 1.0)
            for s in mags:
                pt = x * s
                w_val = -float(W.evaluate(np.asarray(pt)))
                worst = min(worst, w_val)
    assert worst > 0.0, (
        f"{system}: W ≤ 0 on a non-exempt coordinate hyperplane (min={worst:.3e})"
    )
