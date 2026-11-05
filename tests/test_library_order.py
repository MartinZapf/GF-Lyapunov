"""The GF library assembly is canonical and process-independent (WP P5-13).

Why this file exists. ``enumerate_cross_terms`` de-duplicates through a ``set``
of :class:`TermSpec`, whose ``__hash__`` mixes in ``term_types`` — a tuple of
``str``, and Python randomizes string hashes per process. Until P5-13 the set
was turned back into a list unsorted, so the library order, the SDP variable
order and the interior point MOSEK landed on changed from process to process:
six of the twelve nominal E1 systems (twisting, continuous_twisting, levant3d,
pidsmc, fc_cta, levant4d) returned a different designed alpha on every run,
measured |dalpha| up to 5.8e+02 (P2-09 contradiction 2, P2-06 part C 6/12).

The falsifying test is :func:`test_library_order_is_identical_under_three_hash_seeds`:
it enumerates the SAME library in three child processes at three different
``PYTHONHASHSEED`` values and compares the ordered digest. It is solver-free.
Against the pre-P5-13 sources it fails on all six large-library systems; the
seed-0 shas it fails with are recorded in
``artifacts/analysis_2026-09-02/library_order/README.md``.

Escape hatch for that falsification: the child reads ``GF_TEST_SRC`` and puts it
in front of ``sys.path``, so the test can be pointed at an older source tree
without editing it. Unset in normal runs.

Everything except the last test is solver-free.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.enumerate import (
    LIBRARY_ORDER,
    TermSpec,
    canonical_term_key,
    enumerate_all_terms,
    enumerate_cross_terms,
    enumerate_cross_term_pool,
    enumerate_diagonal_terms,
    sort_terms_canonical,
)
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible

from gf_framework.verification.mosek_direct import MOSEK_AVAILABLE

SRC = str(Path(__file__).resolve().parent.parent / "src")

#: (factory, degree) for two systems whose cross library is large enough that
#: set-iteration order was demonstrably unstable, plus one that was stable.
UNSTABLE = [("create_levant4d_spec", 7), ("create_pidsmc_spec", 6),
            ("create_twisting_spec", 5)]

#: The child program of the cross-process test. Prints one digest per system.
CHILD = r'''
import hashlib, json, os, sys
sys.path.insert(0, os.environ.get("GF_TEST_SRC") or sys.argv[1])
from fractions import Fraction
from gf_framework.algebra.enumerate import (enumerate_cross_terms,
                                            enumerate_diagonal_terms)
from gf_framework.discovery import universal_discovery as ud
from gf_framework.discovery.gf_discovery import forced_mu, mu_compatible

out = {"hash_randomization": sys.flags.hash_randomization,
       "seed": os.environ.get("PYTHONHASHSEED"), "rows": {}}
for factory, m in json.loads(sys.argv[2]):
    spec = getattr(ud, factory)()
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    terms = list(diag) + [t for t in cross_full if mu_compatible(t, mu)]
    payload = json.dumps([[[str(Fraction(e)) for e in t.exponents],
                           list(t.term_types)] for t in terms])
    out["rows"][factory] = {
        "n": len(terms),
        "n_diag": len(diag),
        "order_sha": hashlib.sha256(payload.encode()).hexdigest(),
        "diag0": [[str(Fraction(e)) for e in diag[0].exponents],
                  list(diag[0].term_types)] if diag else None,
    }
print(json.dumps(out))
'''


def _child(seed: str | None):
    env = dict(os.environ)
    if seed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = seed
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, SRC, json.dumps(UNSTABLE)],
        capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# the falsifying test
# ---------------------------------------------------------------------------
def test_library_order_is_identical_under_three_hash_seeds():
    """The load-bearing regression. Solver-free; fails pre-P5-13."""
    docs = [_child("0"), _child("12345"), _child(None)]
    assert [d["seed"] for d in docs] == ["0", "12345", None]
    assert docs[2]["hash_randomization"] == 1, \
        "the unseeded child must have hash randomization ON"
    for factory, _m in UNSTABLE:
        shas = {d["rows"][factory]["order_sha"] for d in docs}
        assert len(shas) == 1, (
            f"{factory}: library order depends on PYTHONHASHSEED; "
            f"{len(shas)} distinct orders across 3 processes")
        counts = {d["rows"][factory]["n"] for d in docs}
        assert len(counts) == 1, f"{factory}: library SIZE moved, not just order"


def test_the_pinned_diagonal_is_the_same_monomial_in_every_process():
    """``joint_sos`` pins ``alpha[diag_indices[0]] == coefficient_scale``.

    Diagonals must therefore keep their axis order; canonicalizing the CROSS
    terms must not touch them.
    """
    docs = [_child("0"), _child("12345"), _child(None)]
    for factory, _m in UNSTABLE:
        diag0 = {json.dumps(d["rows"][factory]["diag0"]) for d in docs}
        assert len(diag0) == 1, f"{factory}: the pinned diagonal moved"
        n_diag = {d["rows"][factory]["n_diag"] for d in docs}
        assert len(n_diag) == 1


# ---------------------------------------------------------------------------
# properties of the canonical key itself
# ---------------------------------------------------------------------------
def test_canonical_key_is_total_on_every_benchmark_library():
    """No two distinct specs may tie: a tie would reopen the ordering to the
    input (i.e. set) order through ``sorted``'s stability."""
    for factory, m in [("create_levant4d_spec", 7), ("create_pidsmc_spec", 6),
                       ("create_cta_spec", 9), ("create_fc_cta_spec", 7),
                       ("create_levant3d_spec", 5)]:
        spec = getattr(ud, factory)()
        cross = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                      include_signed=True, minimal=False,
                                      include_abs_variants=True, c1_only=True)
        keys = [canonical_term_key(t) for t in cross]
        assert len(set(map(repr, keys))) == len(cross), \
            f"{factory}: canonical_term_key ties on {m}"


def test_sorting_is_invariant_to_the_input_permutation():
    spec = ud.create_pidsmc_spec()
    cross = enumerate_cross_terms(6, spec.weights_r, max_denominator=4,
                                  include_signed=True, minimal=False,
                                  include_abs_variants=True, c1_only=True)
    ref = [canonical_term_key(t) for t in cross]
    rng = random.Random(20260903)
    for _ in range(20):
        shuffled = list(cross)
        rng.shuffle(shuffled)
        assert [canonical_term_key(t) for t in sort_terms_canonical(shuffled)] == ref


def test_enumerate_cross_terms_returns_canonical_order():
    spec = ud.create_levant3d_spec()
    cross = enumerate_cross_terms(5, spec.weights_r, max_denominator=4,
                                  include_signed=True, minimal=False,
                                  include_abs_variants=True, c1_only=True)
    assert cross == sort_terms_canonical(cross)


def test_key_is_graded_then_lexicographic_then_typed():
    """The stated ordering, on hand-built specs."""
    def spec(exps, types):
        return TermSpec(exponents=tuple(Fraction(e) for e in exps),
                        term_types=tuple(types))
    low = spec(("1", "1"), ("abs", "abs"))          # grade 2
    high = spec(("1/2", "3"), ("abs", "abs"))       # grade 3.5
    assert canonical_term_key(low) < canonical_term_key(high)

    a = spec(("1", "2"), ("abs", "abs"))            # grade 3
    b = spec(("2", "1"), ("abs", "abs"))            # grade 3, lex greater
    assert canonical_term_key(a) < canonical_term_key(b)

    absv = spec(("1", "2"), ("abs", "abs"))
    sgnv = spec(("1", "2"), ("abs", "signed"))
    assert canonical_term_key(absv) < canonical_term_key(sgnv)


def test_key_compares_exponents_exactly_not_as_floats():
    def spec(exps):
        return TermSpec(exponents=tuple(exps), term_types=("abs", "abs"))
    a = spec((Fraction(1, 3), Fraction(2, 3)))
    b = spec((Fraction(2, 6), Fraction(4, 6)))      # same value, reduced
    assert canonical_term_key(a) == canonical_term_key(b)
    assert a == b


def test_diagonal_enumeration_is_untouched_axis_order():
    """The order the designer's pin depends on."""
    for weights, m in [((3, 2, 1), 6), ((4, 3, 2, 1), 7), ((2, 1), 5)]:
        diag = enumerate_diagonal_terms(m, weights, c1_only=True)
        seen_axes = []
        for t in diag:
            nz = [i for i, e in enumerate(t.exponents) if e != 0]
            assert len(nz) == 1
            seen_axes.append(nz[0])
        assert seen_axes == sorted(seen_axes), \
            "diagonals must stay in axis order (alpha[diag_indices[0]] is pinned)"
        assert all(tt == "abs" for t in diag for tt in t.term_types)


def test_downstream_enumerators_inherit_the_canonical_order():
    """``enumerate_all_terms`` sorts by complexity and ``enumerate_cross_term_pool``
    groups by exponent; both are stable, so both are now reproducible."""
    weights, m = (3, 2, 1), 6
    diag_a, cross_a = enumerate_all_terms(m, weights, max_denominator=4,
                                          minimal=False, c1_only=True,
                                          max_cross_terms=8)
    diag_b, cross_b = enumerate_all_terms(m, weights, max_denominator=4,
                                          minimal=False, c1_only=True,
                                          max_cross_terms=8)
    assert diag_a == diag_b and cross_a == cross_b
    pool_a = enumerate_cross_term_pool(m, weights, max_per_pair=6,
                                       max_denominator=3)
    pool_b = enumerate_cross_term_pool(m, weights, max_per_pair=6,
                                       max_denominator=3)
    assert pool_a == pool_b


def test_library_order_identifier_is_declared():
    assert LIBRARY_ORDER == "grlex-v1"


# ---------------------------------------------------------------------------
# the identifier reaches the proof object
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not MOSEK_AVAILABLE, reason="MOSEK not available")
def test_the_design_gram_bundle_records_the_library_order():
    from gf_framework.verification.joint_sos import (
        GRAM_BUNDLE_HASHED_KEYS, JointSOSOptimizer)

    spec = ud.create_fosmc_spec()
    m = 2
    diag = enumerate_diagonal_terms(m, spec.weights_r, c1_only=True)
    cross_full = enumerate_cross_terms(m, spec.weights_r, max_denominator=4,
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    mu = forced_mu(diag, spec.weights_r, spec.dynamics_gf)
    terms = list(diag) + [t for t in cross_full if mu_compatible(t, mu)]
    opt = JointSOSOptimizer(weights_r=spec.weights_r, solver="MOSEK_DIRECT",
                            allow_solver_fallback=False, epsilon=1e-4)
    res = opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=spec.dynamics_gf, active_terms=terms,
        dynamics_gf_vertices=spec.dynamics_gf_vertices)
    assert res.feasible
    bundle = res.gram_certificates
    assert bundle
    assert {e["library_order"] for e in bundle} == {LIBRARY_ORDER}
    # diagnostic, exactly like mu_policy: never part of the content hash
    assert "library_order" not in GRAM_BUNDLE_HASHED_KEYS
