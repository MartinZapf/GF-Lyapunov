"""Deck-group blocking (R1) and V-proof dedup (R2) — MOSEK-free unit tests.

R1 pins that symmetrizing the captured Gram over the target's deck group is a
pure speedup and never a relaxation:

  * the group is computed EXACTLY from the exact target support (review
    verdict 6 guard), all-even support giving the full deck group F₂ⁿ;
  * an odd-support target degrades to the maximal subgroup, and a target with
    no symmetry falls through to the dense path byte for byte;
  * blocked and dense agree on real captured Grams: same verdict, same η̂,
    and a residual that is never larger;
  * a corrupted Gram is still rejected (the reduction must not create
    false positives).

R2 pins that the V proof is reused only for a provably identical V: matching
inputs hash equal, any change to coefficients / exponents / types / weights /
epsilon changes the hash, and a mismatched cache raises instead of reusing.

No solver is needed: the Grams come from the stored campaign records, and the
dedup tests build tiny GFs directly.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification.exact_gate import (
    deck_group_basis,
    exact_octant_target,
    exact_terms_from_gf,
    exact_validate_octant,
    ldl_psd,
    parity_blocks,
)
from gf_framework.verification.exact_phase import (
    ExactPhaseError,
    VProofCache,
    v_input_hash,
)

REPO = Path(__file__).resolve().parents[1]
GRAMS = REPO / "artifacts/ablation/levant_2003_perturbed"
BUNDLE = GRAMS / "levant3d_2003print_exactgains_grams_20260712T062256Z.json"
RECORD = GRAMS / "levant3d_2003print_exactgains_20260712T062256Z.json"


# ---------------------------------------------------------------------------
# 1. Deck group from the exact target support
# ---------------------------------------------------------------------------

def test_deck_group_all_even_is_full():
    """All-even support ⇒ every sign flip is a symmetry ⇒ dim H = n."""
    support = [(14, 0, 0), (0, 14, 0), (10, 4, 0), (6, 0, 8)]
    basis = deck_group_basis(support, 3)
    assert len(basis) == 3
    # the basis spans F₂³, so the induced classes are exactly β mod 2
    monos = [(7, 0, 0), (5, 2, 0), (1, 1, 5), (0, 3, 4)]
    blocks = parity_blocks(monos, basis)
    keys = {tuple(e % 2 for e in m) for m in monos}
    assert len(blocks) == len(keys)


def test_deck_group_partial_subgroup_on_odd_support():
    """One odd exponent vector cuts the group down but does not kill it.

    supp = {(1,1,0)} forces g₁ + g₂ ≡ 0, leaving a 2-dimensional subgroup
    (g₁ = g₂ free together, g₃ free) instead of the full F₂³.
    """
    basis = deck_group_basis([(1, 1, 0), (2, 0, 2)], 3)
    assert len(basis) == 2
    for g in basis:
        assert (g[0] * 1 + g[1] * 1) % 2 == 0     # ⟨g, (1,1,0)⟩ = 0
        assert (g[0] * 2 + g[2] * 2) % 2 == 0     # ⟨g, (2,0,2)⟩ = 0


def test_deck_group_trivial_falls_through_to_dense():
    """Support that pins every coordinate leaves H = {0}: one block, dense."""
    basis = deck_group_basis([(1, 0, 0), (0, 1, 0), (0, 0, 1)], 3)
    assert basis == []
    blocks = parity_blocks([(1, 0, 0), (0, 1, 0)], basis)
    assert blocks == [[0, 1]], "trivial group must yield the single dense block"


def test_parity_blocks_partition_every_index_exactly_once():
    support = [(12, 0, 0), (0, 12, 0), (0, 0, 12), (6, 6, 0)]
    grp = deck_group_basis(support, 3)
    monos = [(a, b, 6 - a - b) for a in range(7) for b in range(7 - a)]
    blocks = parity_blocks(monos, grp)
    flat = sorted(i for blk in blocks for i in blk)
    assert flat == list(range(len(monos)))
    assert all(blocks[k] == sorted(blocks[k]) for k in range(len(blocks)))


# ---------------------------------------------------------------------------
# 2. Blocked vs dense on REAL captured Grams
# ---------------------------------------------------------------------------

def _load_v_capture():
    if not BUNDLE.exists() or not RECORD.exists():
        pytest.skip("stored levant3d_2003print Gram bundle not present")
    bundle = json.load(open(BUNDLE))
    record = json.load(open(RECORD))
    terms = [
        (Fraction(float(t["coefficient"])),
         tuple(Fraction(e) for e in t["exponents"]),
         tuple(t["term_types"]))
        for t in record["V"]["terms"]
    ]
    return bundle["V_grams"][0], terms


def _dense_reference(target, d, gram, basis, eps):
    """The pre-R1 computation: full residual and one dense LDLᵀ."""
    basis = [tuple(int(b) for b in beta) for beta in basis]
    n = len(basis)
    nv = len(basis[0])
    g = np.asarray(gram, dtype=float)
    Q = [[(Fraction(float(g[i, j])) + Fraction(float(g[j, i]))) / 2
          for j in range(n)] for i in range(n)]
    E = dict(target)
    for i in range(n):
        for j in range(n):
            q = Q[i][j]
            if q == 0:
                continue
            mono = tuple(basis[i][k] + basis[j][k] for k in range(nv))
            E[mono] = E.get(mono, Fraction(0)) - q
    resid = sum(abs(v) for v in E.values())
    lam = float(np.linalg.eigvalsh((g + g.T) / 2.0).min())
    eta = Fraction(0) if lam >= 0 else Fraction(float(-lam)) + Fraction(1, 10 ** 15)
    ok = ldl_psd([[Q[i][j] + eta if i == j else Q[i][j] for j in range(n)]
                  for i in range(n)])
    return ok, resid


def test_blocked_matches_dense_on_real_gram():
    """Same verdict, same η̂, and since fix (a) of 2026-09-25 the same residual:
    X charges the DENSE residual of the persisted Gram (as the portable verifier
    does), while its PSD test still runs on the parity blocks."""
    cap, terms = _load_v_capture()
    eps = Fraction(float(cap["eps"]))
    target, d = exact_octant_target(terms, tuple(cap["mu"]),
                                    tuple(cap["signs"]), eps)
    chk = exact_validate_octant(target, d, np.asarray(cap["gram"], dtype=float),
                                cap["basis"], eps, tuple(cap["signs"]))
    dense_ok, dense_resid = _dense_reference(
        target, d, cap["gram"], cap["basis"], eps)

    assert chk.ok and dense_ok
    assert chk.target_all_even, "doubled mu must give an all-even V target"
    assert chk.sym_group_dim == len(cap["signs"])   # full deck group
    assert chk.n_blocks > 1 and sum(chk.block_sizes) == chk.n_basis
    assert Fraction(chk.resid_l1_float) <= Fraction(dense_resid) * 2, \
        "symmetrized residual must not blow up"
    assert Fraction(chk.resid_l1_float) == Fraction(float(dense_resid)), \
        "X charges the dense residual of the persisted Gram (fix (a), 2026-09-25)"
    assert chk.charge_float <= chk.budget_float


def test_blocked_still_rejects_corrupted_gram():
    """The reduction must not manufacture acceptances."""
    cap, terms = _load_v_capture()
    eps = Fraction(float(cap["eps"]))
    target, d = exact_octant_target(terms, tuple(cap["mu"]),
                                    tuple(cap["signs"]), eps)
    g = np.asarray(cap["gram"], dtype=float).copy()
    g[0, 0] -= 1e-3
    chk = exact_validate_octant(target, d, g, cap["basis"], eps,
                                tuple(cap["signs"]))
    assert not chk.ok


def test_dense_path_preserved_when_group_is_trivial():
    """A target with no sign symmetry must reproduce the dense result exactly.

    V = |x|² + ⌈x⌋³ in one variable with μ = 2 lifts to y⁴ + y⁶ (support
    {(4), (6)}) — all even, so instead force a trivial group by hand and check
    the block machinery collapses to the single dense block.
    """
    basis = [(2, 0), (1, 1), (0, 2)]
    grp = deck_group_basis([(1, 0), (0, 1)], 2)
    assert grp == []
    assert parity_blocks(basis, grp) == [[0, 1, 2]]


def test_guard_runs_on_the_exact_target_not_a_float_support():
    """The guard must read the EXACT target, never a float-thresholded one.

    Regression for a real mistake made while validating R1: deriving the W
    support from the captured Gram expansion with a 1e-14 cutoff reported a
    spurious loss of symmetry (cross-parity Gram noise creates odd monomials
    that the true target does not contain). Review verdict 6 demands the exact
    support, and here W is built through the exact Lie mirror.

    Pins that a real W target lifts to all-even support and blocks, so the R1
    speedup covers the W half and not only V.
    """
    from gf_framework.discovery.universal_discovery import create_sta_spec
    from gf_framework.verification.exact_gate import exact_neg_lie_derivative

    spec = create_sta_spec(delta0=Fraction(1, 2))
    V = GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(1), Fraction(0)),
               term_types=("abs", "abs")),
        GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(2)),
               term_types=("abs", "abs")),
    ])
    v_terms = exact_terms_from_gf(V)
    for dyn in spec.dynamics_gf_vertices:
        w = exact_neg_lie_derivative(v_terms, [exact_terms_from_gf(f) for f in dyn])
        # mu doubled per variable, exactly as verify_gf_sos_full computes it
        mu = []
        for i in range(2):
            den = 1
            for _, exps, _ in w:
                if exps[i] != 0:
                    den = den * exps[i].denominator // np.gcd(den, exps[i].denominator)
            mu.append(2 * den)
        target, d = exact_octant_target(w, tuple(mu), (1, 1), Fraction(1, 10 ** 6))
        assert all(all(e % 2 == 0 for e in m) for m in target), \
            "doubled mu must give an all-even W target"
        assert len(deck_group_basis(list(target), 2)) == 2, \
            "all-even support must yield the FULL deck group"


# ---------------------------------------------------------------------------
# 3. R2: V input hash and dedup refusal
# ---------------------------------------------------------------------------

def _V(coeff=1.0, exp=Fraction(2), typ="abs", weights=(1,)):
    # strict_homogeneous=False: these probes mutate ONE field at a time to
    # isolate what the hash is sensitive to, which would otherwise trip the
    # weighted-degree assertion (the hash itself never inspects degree_m).
    return GeneralizedForm(degree_m=2, weights_r=weights, strict_homogeneous=False,
                           terms=[
        GFTerm(coefficient=coeff, exponents=(exp,), term_types=(typ,)),
    ])


def test_v_input_hash_stable_and_order_independent():
    a = GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=[
        GFTerm(coefficient=1.0, exponents=(Fraction(1), Fraction(0)),
               term_types=("abs", "abs")),
        GFTerm(coefficient=2.0, exponents=(Fraction(0), Fraction(2)),
               term_types=("abs", "abs")),
    ])
    b = GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=list(reversed(a.terms)))
    assert v_input_hash(a, 1e-6) == v_input_hash(b, 1e-6)


@pytest.mark.parametrize("mutate", [
    lambda: _V(coeff=1.0 + 1e-12),
    lambda: _V(exp=Fraction(3)),
    lambda: _V(typ="signed"),
    lambda: _V(weights=(2,)),
])
def test_v_input_hash_detects_every_relevant_change(mutate):
    assert v_input_hash(_V(), 1e-6) != v_input_hash(mutate(), 1e-6)


def test_v_input_hash_detects_epsilon_change():
    assert v_input_hash(_V(), 1e-6) != v_input_hash(_V(), 1e-9)


def test_v_input_hash_exact_rational_vs_float_epsilon():
    """ε is hashed as the exact rational the pipeline uses, not a repr."""
    assert v_input_hash(_V(), 1e-6) == v_input_hash(_V(), Fraction(float(1e-6)))


def test_dedup_refuses_mismatched_cache():
    """Handing a cache built for a different V must RAISE, not reuse."""
    from gf_framework.verification.exact_gate import ExactCaptureResult
    from gf_framework.verification.exact_phase import run_exact_phase

    stale = VProofCache(
        input_hash=v_input_hash(_V(coeff=1.0), 1e-6),
        fingerprint="stale",
        resolve_feasible=True,
        exact=ExactCaptureResult(ok=True),
        grams=[],
    )
    with pytest.raises(ExactPhaseError, match="does not belong to this certificate"):
        run_exact_phase(_V(coeff=2.0), [], 1, epsilon=1e-6, v_cache=stale)


def test_dedup_accepts_matching_cache_without_solving():
    """A matching cache short-circuits the V half (no solver is available or
    needed here: reaching the solver would fail the test by raising)."""
    from gf_framework.verification.exact_gate import exact_verify_captures
    from gf_framework.verification.exact_phase import (
        run_exact_phase, build_v_cache, ExactPhaseResult)

    V = _V(coeff=3.5)
    caps = [{"signs": (s,), "octant_index": i, "eps": 1e-6,
             "mu": (1,), "y_degree": 2, "basis": [(1,)],
             "gram": np.array([[3.5 - 1e-6]])}
            for i, s in enumerate((-1, 1))]
    exact = exact_verify_captures(exact_terms_from_gf(V), caps, 1)
    cache = build_v_cache(V, 1e-6, ExactPhaseResult(
        v_input_hash=v_input_hash(V, 1e-6), v_resolve_feasible=True,
        v_exact=exact, v_grams=caps))
    res = run_exact_phase(V, [], 1, epsilon=1e-6, v_cache=cache)
    assert res.v_reused is True
    assert res.v_resolve_feasible and res.v_exact.ok
    assert res.v_fingerprint == cache.fingerprint
    assert res.v_input_hash == cache.input_hash
    assert res.endpoints == []


def test_v_terms_roundtrip_is_exact():
    """exact_terms_from_gf feeds the hash; Fraction gains must stay exact."""
    gf = GeneralizedForm(degree_m=0, weights_r=(1,), terms=[
        GFTerm(coefficient=Fraction(11, 10), exponents=(Fraction(0),),
               term_types=("abs",)),
    ])
    (c, _, _), = exact_terms_from_gf(gf)
    assert c == Fraction(11, 10)
    assert v_input_hash(gf, 1e-6) != v_input_hash(_V(coeff=1.1), 1e-6)


# ---------------------------------------------------------------------------
# 4. P3-04: the R1 provenance reaches the exact record
# ---------------------------------------------------------------------------
# Without these four fields the deck-group blocking is invisible to anyone
# reading an artifact, so the review-verdict-6 discharge ("the guard runs on
# the EXACT target support") could only be checked by reading the source.
# `grep -rl sym_group_dim artifacts/` used to return nothing.

R1_OCTANT_FIELDS = {"target_all_even", "sym_group_dim", "n_blocks", "block_sizes"}
R1_ROLLUP_FIELDS = {"target_all_even_all", "sym_group_dim_min", "sym_group_dim_max",
                    "n_octants_dense_fallback", "max_block_size"}

E1_MATRIX = REPO / "artifacts/campaign_2026-08/E1_matrix"
E1_MATRIX_JSON = E1_MATRIX / "gf_matrix_20260802_143921.json"
# Written 2026-08-02, before P3-04: no r1_provenance anywhere in it.
PRE_P304_RECORD = E1_MATRIX / "reverify_exact_printed_20260802T181035Z.json"


def _reverify_module():
    import sys
    sys.path.insert(0, str(REPO / "review"))
    import reverify_exact  # noqa: E402
    return reverify_exact


def _row(matrix_json: Path, system: str) -> dict:
    for row in json.loads(matrix_json.read_text())["results"]:
        if row["system"] == system:
            return row
    raise AssertionError(f"{system} not in {matrix_json.name}")


@pytest.mark.skipif(not E1_MATRIX_JSON.exists(),
                    reason="campaign E1_matrix artifacts not present")
def test_fresh_exact_record_carries_r1_provenance():
    """A fresh re-proof of the smallest system writes all four fields.

    fosmc: n = 1, m = 2, one term, one bracket corner — the cheapest end-to-end
    exact phase in the tree, so this pins the SERIALIZATION rather than
    re-testing the arithmetic. Both halves (V and W) and every octant must
    carry the block, and the whole summary must survive ``json.dumps``, which
    is what the artifact writer does with it.
    """
    rv = _reverify_module()
    V = rv._reconstruct_V(_row(E1_MATRIX_JSON, "fosmc"))
    (_, phase), = rv.printed_rational_phases(V, "fosmc", epsilon=1e-6,
                                             solver="CLARABEL")
    assert phase.exact_pass

    for half in (phase.v_exact, phase.endpoints[0].exact):
        summary = rv._capture_summary(half)
        assert R1_ROLLUP_FIELDS <= set(summary["r1_provenance"])
        assert summary["r1_provenance"]["target_all_even_all"] is True
        assert summary["r1_provenance"]["n_octants_dense_fallback"] == 0
        assert len(summary["octants"]) == 2
        for oct_summary in summary["octants"]:
            prov = oct_summary["r1_provenance"]
            assert R1_OCTANT_FIELDS == set(prov)
            assert prov["target_all_even"] is True
            assert prov["sym_group_dim"] == 1          # full deck group, n = 1
            assert sum(prov["block_sizes"]) == oct_summary["n_basis"]
            assert len(prov["block_sizes"]) == prov["n_blocks"]
        # The writer json.dumps the whole artifact; tuples would survive that
        # but Fractions would not, so serialize here rather than at 3 a.m.
        assert json.loads(json.dumps(summary)) == summary


@pytest.mark.skipif(not PRE_P304_RECORD.exists(),
                    reason="pre-P3-04 campaign record not present")
def test_stored_record_without_r1_provenance_still_loads():
    """The addition is backward compatible: old records still load and verify.

    The stored 2026-08-02 benchmark re-proof predates P3-04, so none of its
    octants carries the block. It must still pass through the record loader
    (``scripts/emit_certificate_record.py``) with the SAME verdicts it had
    before: the fosmc entry is found, its gain binding is accepted, and the
    only refusal it raises is the one it always raised — no Gram captures.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import emit_certificate_record as emit  # noqa: E402

    doc = json.loads(PRE_P304_RECORD.read_text())
    entry = emit.exact_entry(doc, "fosmc")
    emit.check_gain_binding(doc, entry, "fosmc")        # must not raise

    assert "r1_provenance" not in entry["V_exact"]
    for oct_summary in entry["V_exact"]["octants"] + entry["W_exact"]["octants"]:
        assert "r1_provenance" not in oct_summary
        # A reader that asks for the new block on an old record gets nothing,
        # not an exception.
        assert oct_summary.get("r1_provenance") is None

    with pytest.raises(emit.Refusal) as exc:
        emit.find_grams(doc, PRE_P304_RECORD, "fosmc", None)
    assert exc.value.reason == "missing_gram_captures"


@pytest.mark.skipif(not (E1_MATRIX_JSON.exists() and PRE_P304_RECORD.exists()),
                    reason="campaign E1_matrix artifacts not present")
def test_r1_provenance_is_purely_additive():
    """No pre-P3-04 key was renamed, removed or retyped by the addition."""
    rv = _reverify_module()
    V = rv._reconstruct_V(_row(E1_MATRIX_JSON, "fosmc"))
    (_, phase), = rv.printed_rational_phases(V, "fosmc", epsilon=1e-6,
                                             solver="CLARABEL")
    fresh = rv._capture_summary(phase.v_exact)
    old = json.loads(PRE_P304_RECORD.read_text())
    old_v = [r for r in old["systems"] if r["system"] == "fosmc"][0]["V_exact"]

    assert set(old_v) | {"r1_provenance"} == set(fresh)
    assert set(old_v["octants"][0]) | {"r1_provenance"} == set(fresh["octants"][0])
