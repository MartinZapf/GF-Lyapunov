"""Process parallelism over octants (WP P3-03) — MOSEK-free.

``--jobs``/``jobs=`` may change how long a re-proof takes and NOTHING else.
These tests pin the four properties that make that true:

  * **ordering determinism.** Octant results are assembled by index and
    truncated at the first failing octant, so a mapper that yields in any
    completion order (here: shuffled, reversed, and one that yields the LAST
    octant first) reproduces the serial ``ExactCaptureResult`` field for
    field, on a passing certificate and on a failing one;
  * **typed worker failures.** A worker that raises, a worker that dies, and a
    pool that cannot start are :class:`OctantWorkerError` with a reason from
    :data:`OCTANT_WORKER_REASONS`. A mapper that silently DROPS a result is an
    ``ExactGateError``, never a shorter certificate that passes;
  * **cache double-miss safety.** Two processes that miss the same
    :class:`DiskVProofCache` key at once — the P3-03 corner fan-out's worst
    case — leave exactly one valid entry, no temporary file, and the second
    refuses to overwrite the first;
  * **jobs=1 identity.** ``jobs=1`` creates no pool, takes the pre-P3-03 code
    path, and produces a bit-identical per-octant digest.

The Grams come from a stored campaign record; no solver is called. Two tests
start REAL spawn worker processes (marked in their names), the rest use
in-process mappers.
"""
import json
import os
import random
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification import exact_phase as ep
from gf_framework.verification.exact_gate import (
    ExactGateError,
    exact_terms_from_gf,
    exact_verify_captures,
    validate_one_capture,
)
from gf_framework.verification.exact_phase import (
    DEFAULT_JOBS,
    OCTANT_START_METHOD,
    OCTANT_WORKER_REASONS,
    ExactPhaseError,
    OctantPool,
    OctantWorkerError,
    octant_mapper,
    parallel_record,
    resolve_jobs,
    shutdown_octant_pool,
)

REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "artifacts/ablation/levant_2003_perturbed"
BUNDLE = BLOCK / "levant3d_2003print_exactgains_grams_20260712T062256Z.json"
RECORD = BLOCK / "levant3d_2003print_exactgains_20260712T062256Z.json"
N_VARS = 3


# ---------------------------------------------------------------------------
# Fixture: a real V and its real Gram captures, no solver
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def captures():
    """``(v_terms, grams)`` from a stored record. 2^3 = 8 octants."""
    if not BUNDLE.exists() or not RECORD.exists():
        pytest.skip("stored levant3d_2003print Gram bundle not present")
    record = json.loads(RECORD.read_text())
    bundle = json.loads(BUNDLE.read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]],
    )
    grams = [
        {"signs": tuple(int(s) for s in c["signs"]),
         "octant_index": int(c["octant_index"]),
         "eps": float(c["eps"]),
         "mu": tuple(int(v) for v in c["mu"]),
         "y_degree": int(c["y_degree"]),
         "basis": [tuple(int(e) for e in b) for b in c["basis"]],
         "gram": np.asarray(c["gram"], dtype=np.float64)}
        for c in bundle["V_grams"]
    ]
    return exact_terms_from_gf(V), grams


def digest(res):
    """Everything an artifact records about one form, exactly."""
    return {
        "ok": res.ok, "reason": res.reason, "mu": tuple(res.mu),
        "eps": res.eps, "n_expected": res.n_octants_expected,
        "r1": res.r1_provenance(),
        "octants": [
            (c.ok, tuple(c.signs), c.reason, c.degree, c.n_basis, c.eta_hat,
             float(c.eta_hat_float).hex(), float(c.resid_l1_float).hex(),
             float(c.charge_float).hex(), float(c.budget_float).hex(),
             float(c.min_eig_float).hex(), c.target_all_even, c.sym_group_dim,
             c.n_blocks, tuple(c.block_sizes))
            for c in res.octants],
    }


# --- in-process mappers that abuse the ordering contract -------------------

def _mapper(order):
    """A mapper that computes everything, then yields in ``order(indices)``."""
    def run(fn, items):
        out = [(i, fn(*item)) for i, item in enumerate(items)]
        return [out[i] for i in order(list(range(len(out))))]
    return run


def _reversed(idx):
    return list(reversed(idx))


def _shuffled(idx):
    rng = random.Random(20260903)
    out = list(idx)
    rng.shuffle(out)
    return out


def _last_first(idx):
    return ([idx[-1]] + idx[:-1]) if idx else idx


def _drop_one(fn, items):
    """A broken mapper: it loses the middle result."""
    out = [(i, fn(*item)) for i, item in enumerate(items)]
    return [p for p in out if p[0] != len(out) // 2]


# ---------------------------------------------------------------------------
# 1. Ordering determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [_reversed, _shuffled, _last_first],
                         ids=["reversed", "shuffled", "last_first"])
def test_completion_order_does_not_change_a_passing_result(captures, order):
    terms, grams = captures
    serial = exact_verify_captures(terms, grams, N_VARS)
    assert serial.ok, serial.reason
    parallel = exact_verify_captures(terms, grams, N_VARS, _mapper(order))
    assert digest(parallel) == digest(serial)


@pytest.mark.parametrize("order", [_reversed, _shuffled, _last_first],
                         ids=["reversed", "shuffled", "last_first"])
def test_completion_order_does_not_change_a_FAILING_result(captures, order):
    """The truncation at the first bad octant is an ordering property too.

    The serial loop stops at the first failure, so it records k+1 octants and
    names octant k in ``reason``. A mapper computes all 8; the assembly must
    still record exactly k+1 and name the same octant.
    """
    terms, grams = captures
    damaged = [dict(c) for c in grams]
    bad = 2
    g = np.array(damaged[bad]["gram"], dtype=float)
    g[0, 0] -= 1.0                      # a Gram that cannot prove its target
    damaged[bad]["gram"] = g
    serial = exact_verify_captures(terms, damaged, N_VARS)
    assert not serial.ok
    assert len(serial.octants) == bad + 1, "fixture must fail at octant 2"
    parallel = exact_verify_captures(terms, damaged, N_VARS, _mapper(order))
    assert digest(parallel) == digest(serial)
    assert parallel.reason == serial.reason
    assert len(parallel.octants) == bad + 1


def test_a_mapper_that_drops_a_result_is_refused_not_shortened(captures):
    terms, grams = captures
    with pytest.raises(ExactGateError) as exc:
        exact_verify_captures(terms, grams, N_VARS, _drop_one)
    assert "octant_results_missing" in str(exc.value)


def test_the_serial_path_still_short_circuits_at_the_first_failure(captures):
    """``mapper=None`` must not start computing octants after a failure.

    This is what keeps ``jobs=1`` the pre-P3-03 path in COST as well as in
    result: a parallel mapper computes all 2^n octants, the serial loop does
    not.
    """
    terms, grams = captures
    damaged = [dict(c) for c in grams]
    g = np.array(damaged[0]["gram"], dtype=float)
    g[0, 0] -= 1.0
    damaged[0]["gram"] = g
    res = exact_verify_captures(terms, damaged, N_VARS)
    assert not res.ok and len(res.octants) == 1


# ---------------------------------------------------------------------------
# 2. jobs = 1 identity
# ---------------------------------------------------------------------------

def test_jobs_1_creates_no_pool_at_all():
    shutdown_octant_pool()
    assert DEFAULT_JOBS == 1
    assert octant_mapper(1) == (None, None)
    assert octant_mapper(0) == (None, None)
    assert octant_mapper(None) == (None, None)
    assert ep._OCTANT_POOL is None


def test_jobs_1_takes_the_pre_change_code_path(captures):
    """``mapper=None`` is the historical call, and it is what jobs=1 produces."""
    terms, grams = captures
    mapper, pool = octant_mapper(1)
    assert mapper is None and pool is None
    assert digest(exact_verify_captures(terms, grams, N_VARS, mapper)) == \
        digest(exact_verify_captures(terms, grams, N_VARS))


def test_resolve_jobs_normalizes_and_refuses_nonsense():
    assert resolve_jobs(None) == 1
    assert resolve_jobs(0) == 1
    assert resolve_jobs(-7) == 1
    assert resolve_jobs(8) == 8
    assert resolve_jobs("8") == 8
    for bad in (True, False, "eight", 1.5j, object()):
        with pytest.raises(ExactPhaseError):
            resolve_jobs(bad)


# ---------------------------------------------------------------------------
# 3. The real spawn pool
# ---------------------------------------------------------------------------

def test_the_start_method_is_spawn_and_the_worker_fn_is_the_pure_one():
    assert OCTANT_START_METHOD == "spawn"
    assert ep.OCTANT_WORKER_FN is validate_one_capture
    assert validate_one_capture.__module__ == \
        "gf_framework.verification.exact_gate"


@pytest.mark.slow
def test_real_spawn_workers_reproduce_the_serial_proof_exactly(captures):
    terms, grams = captures
    serial = exact_verify_captures(terms, grams, N_VARS)
    pool = OctantPool(4, OCTANT_START_METHOD)
    try:
        parallel = exact_verify_captures(terms, grams, N_VARS, pool.map)
    finally:
        pool.shutdown()
    assert digest(parallel) == digest(serial)
    assert pool.n_batches == 1 and pool.n_tasks == len(grams)


@pytest.mark.slow
def test_real_spawn_workers_reproduce_a_FAILING_proof_exactly(captures):
    terms, grams = captures
    damaged = [dict(c) for c in grams]
    g = np.array(damaged[5]["gram"], dtype=float)
    g[1, 1] -= 1.0
    damaged[5]["gram"] = g
    serial = exact_verify_captures(terms, damaged, N_VARS)
    assert not serial.ok
    pool = OctantPool(4, OCTANT_START_METHOD)
    try:
        parallel = exact_verify_captures(terms, damaged, N_VARS, pool.map)
    finally:
        pool.shutdown()
    assert digest(parallel) == digest(serial)


# ---------------------------------------------------------------------------
# 4. Typed worker failures
# ---------------------------------------------------------------------------

def _boom(*_args):
    raise ValueError("the worker blew up")


def _die(*_args):
    os._exit(9)


@pytest.mark.slow
def test_a_worker_that_raises_is_typed_and_names_the_octant():
    pool = OctantPool(2, OCTANT_START_METHOD)
    try:
        with pytest.raises(OctantWorkerError) as exc:
            list(pool.map(_boom, [(1,), (2,), (3,)]))
    finally:
        pool.shutdown()
    assert exc.value.reason == "octant_worker_exception"
    assert exc.value.reason in OCTANT_WORKER_REASONS
    assert "ValueError" in exc.value.detail
    assert exc.value.start_method == "spawn"


@pytest.mark.slow
def test_a_worker_that_DIES_is_typed_and_does_not_hang():
    """The requirement is 'fail the row with a typed reason, not hang'."""
    pool = OctantPool(2, OCTANT_START_METHOD)
    try:
        with pytest.raises(OctantWorkerError) as exc:
            list(pool.map(_die, [(1,), (2,)]))
    finally:
        pool.shutdown()
    assert exc.value.reason == "octant_worker_died"
    assert pool.broken is True


def test_a_pool_that_cannot_start_is_typed():
    pool = OctantPool(2, "no-such-start-method")
    with pytest.raises(OctantWorkerError) as exc:
        list(pool.map(_boom, [(1,)]))
    assert exc.value.reason == "octant_pool_unavailable"
    assert pool.broken is True


def test_a_worker_failure_is_a_TYPED_witness_attempt_then_propagates(
        captures, monkeypatch):
    """A worker failure fails the WITNESS ATTEMPT with its OWN reason.

    While another witness is left it is a recorded attempt typed
    ``octant_worker_died`` — not the generic ``unexpected_error`` a P3-02
    ``classify_exception`` would give it — and when nothing is left it
    propagates. It is never swallowed and never a short certificate.
    """
    record = json.loads(RECORD.read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]])
    seen = []

    def exploding_once(*_a, **kw):
        seen.append(kw.get("solver"))
        raise OctantWorkerError("octant_worker_died", "synthetic",
                                jobs=4, start_method="spawn")

    monkeypatch.setattr(ep, "_exact_phase_once", exploding_once)
    with pytest.raises(OctantWorkerError) as exc:
        ep.run_exact_phase(V, [None], N_VARS, epsilon=1e-6, solver="STUBA",
                           fallback_solver="STUBB", jobs=4)
    assert exc.value.reason == "octant_worker_died"
    assert exc.value.reason in OCTANT_WORKER_REASONS
    # both rungs of the ladder were tried; the first is a recorded attempt and
    # the last one propagates rather than returning a verdict
    assert seen == ["STUBA", "STUBB"]


def test_a_worker_failure_short_of_the_last_rung_is_recorded_not_raised(
        captures, monkeypatch):
    record = json.loads(RECORD.read_text())
    terms, grams = captures
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]])
    good = ep.ExactPhaseResult(
        v_resolve_feasible=True,
        v_exact=exact_verify_captures(terms, grams, N_VARS),
        v_grams=grams, witness_solver="STUBB")
    good.endpoints = [ep.EndpointExactResult(
        resolve_feasible=True, exact=good.v_exact, lie_crosscheck={},
        grams=grams)]
    calls = []

    def first_dies_then_works(*_a, **kw):
        calls.append(kw.get("solver"))
        if len(calls) == 1:
            raise OctantWorkerError("octant_worker_exception", "synthetic",
                                    jobs=4, start_method="spawn", index=3)
        return good

    monkeypatch.setattr(ep, "_exact_phase_once", first_dies_then_works)
    res = ep.run_exact_phase(V, [None], N_VARS, epsilon=1e-6, solver="STUBA",
                             fallback_solver="STUBB", jobs=4)
    assert res.exact_pass
    assert [a["shortfall"] for a in res.witness_attempts][0] == \
        "octant_worker_exception"
    assert res.witness_attempts[0]["ok"] is False
    assert res.witness_attempts[1]["adopted"] is True
    assert res.parallel["jobs"] == 4 and res.parallel["start_method"] == "spawn"


# ---------------------------------------------------------------------------
# 5. Provenance
# ---------------------------------------------------------------------------

def test_parallel_provenance_is_recorded_even_when_serial():
    rec = parallel_record(1, OCTANT_START_METHOD, None)
    assert rec["jobs"] == 1 and rec["enabled"] is False
    assert rec["start_method"] == "" and rec["max_workers"] == 1
    assert rec["scope"] == "octant"
    pool = OctantPool(6)
    rec = parallel_record(6, OCTANT_START_METHOD, pool,
                          n_batches=2, n_tasks=32)
    assert rec == {"scope": "octant", "jobs": 6, "enabled": True,
                   "start_method": "spawn", "max_workers": 6,
                   "n_octant_batches": 2, "n_octants": 32,
                   "cpu_count": os.cpu_count()}


def test_the_witness_record_carries_the_parallel_block():
    from gf_framework.verification.exact_gate import ExactCaptureResult
    res = ep.ExactPhaseResult(v_resolve_feasible=True,
                              v_exact=ExactCaptureResult(ok=True))
    assert res.parallel == {}
    assert res.witness_record()["parallel"] == {}
    res.parallel = parallel_record(4, OCTANT_START_METHOD, OctantPool(4))
    assert res.witness_record()["parallel"]["jobs"] == 4
    assert res.witness_record()["parallel"]["start_method"] == "spawn"


# ---------------------------------------------------------------------------
# 6. Disk V-proof cache under a concurrent double miss (P3-03 x P3-05)
# ---------------------------------------------------------------------------

_DOUBLE_MISS = textwrap.dedent(
    """
    import json, os, sys, time
    from fractions import Fraction
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, {src!r})
    from gf_framework.algebra.core import GeneralizedForm, GFTerm
    from gf_framework.verification.exact_gate import (
        exact_terms_from_gf, exact_verify_captures)
    from gf_framework.verification.exact_phase import (
        DiskVProofCache, ExactPhaseResult, v_input_hash, v_proof_fingerprint)

    root, record_p, bundle_p, barrier = sys.argv[1:5]
    record = json.loads(Path(record_p).read_text())
    bundle = json.loads(Path(bundle_p).read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]])
    grams = [{{"signs": tuple(int(s) for s in c["signs"]),
               "octant_index": int(c["octant_index"]),
               "eps": float(c["eps"]),
               "mu": tuple(int(v) for v in c["mu"]),
               "y_degree": int(c["y_degree"]),
               "basis": [tuple(int(e) for e in b) for b in c["basis"]],
               "gram": np.asarray(c["gram"], dtype=np.float64)}}
              for c in bundle["V_grams"]]
    terms = exact_terms_from_gf(V)
    exact = exact_verify_captures(terms, grams, 3)
    assert exact.ok
    result = ExactPhaseResult(
        v_resolve_feasible=True, v_exact=exact, v_grams=grams,
        v_input_hash=v_input_hash(V, 1e-6),
        v_fingerprint=v_proof_fingerprint(exact, grams))
    cache = DiskVProofCache(root, library="p3_03_double_miss")
    # Every process MISSES first (that is the case under test), then all of
    # them try to store the same key at once.
    hit, ev = cache.load(V, 1e-6, 3, v_terms=terms)
    assert hit is None
    assert ev.hit is False
    assert ev.reason == "vproof_cache_miss"
    # No process may store until all four have genuinely missed. A fixed
    # start time cannot ensure this when imports or verification are slow.
    barrier_dir = Path(barrier)
    (barrier_dir / (str(os.getpid()) + ".ready")).touch()
    deadline = time.monotonic() + 120.0
    while len(list(barrier_dir.glob("*.ready"))) != 4:
        if time.monotonic() >= deadline:
            raise TimeoutError("not all four cache misses reached the barrier")
        time.sleep(0.005)
    store = cache.store(V, 1e-6, 3, result)
    print(json.dumps({{"miss_reason": ev.reason, "hit": ev.hit,
                       "store_reason": store.store_reason,
                       "stored": store.stored}}))
    """
)


@pytest.mark.slow
def test_a_concurrent_double_miss_leaves_one_valid_entry_and_no_tmp(tmp_path):
    """Four processes miss the same key, then all store it simultaneously.

    This is exactly what ``--fanout all`` does in the P3-03 acceptance driver
    and what a naive corner fan-out does by accident. The requirement: the
    entry must not be corrupted or double-stored, and a writer that finds a
    valid entry must refuse to overwrite it.
    """
    if not BUNDLE.exists() or not RECORD.exists():
        pytest.skip("stored levant3d_2003print Gram bundle not present")
    script = tmp_path / "double_miss.py"
    script.write_text(_DOUBLE_MISS.format(src=str(REPO / "src")))
    root = tmp_path / "cache"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    procs = []
    outs = []
    try:
        for _ in range(4):
            procs.append(subprocess.Popen(
                [sys.executable, str(script), str(root), str(RECORD),
                 str(BUNDLE), str(barrier)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for p in procs:
            out, err = p.communicate(timeout=600)
            assert p.returncode == 0, err[-2000:]
            outs.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.communicate()

    # every process genuinely missed
    assert all(o["miss_reason"] == "vproof_cache_miss" for o in outs), outs
    assert all(o["hit"] is False for o in outs)
    # exactly one entry, no temporary file survived
    entries = sorted(root.glob("v_*.json"))
    assert len(entries) == 1, [p.name for p in entries]
    assert list(root.glob(".*tmp")) == []
    assert list(root.glob("*.tmp")) == []
    # at least one wrote it; the ones that lost the race refused to overwrite
    reasons = sorted(o["store_reason"] for o in outs)
    assert set(reasons) <= {"vproof_cache_stored",
                            "vproof_cache_store_skipped_valid"}, reasons
    assert reasons.count("vproof_cache_stored") >= 1
    # and the surviving entry is a usable proof, re-decided from scratch
    record = json.loads(RECORD.read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]])
    cache = ep.DiskVProofCache(root, library="p3_03_double_miss")
    hit, event = cache.load(V, 1e-6, 3, v_terms=exact_terms_from_gf(V))
    assert event.reason == "vproof_cache_hit" and event.revalidated is True
    assert hit is not None and hit.exact.ok


def test_a_second_writer_refuses_to_overwrite_a_valid_entry(tmp_path, captures):
    """The in-process half of the same rule, without four subprocesses."""
    terms, grams = captures
    record = json.loads(RECORD.read_text())
    V = GeneralizedForm(
        degree_m=record["V"]["degree_m"],
        weights_r=tuple(record["V"]["weights_r"]),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in record["V"]["terms"]])
    exact = exact_verify_captures(terms, grams, N_VARS)
    assert exact.ok
    result = ep.ExactPhaseResult(
        v_resolve_feasible=True, v_exact=exact, v_grams=grams,
        v_input_hash=ep.v_input_hash(V, 1e-6),
        v_fingerprint=ep.v_proof_fingerprint(exact, grams))
    cache = ep.DiskVProofCache(tmp_path / "c", library="p3_03_second_writer")
    first = cache.store(V, 1e-6, N_VARS, result)
    assert first.store_reason == "vproof_cache_stored" and first.stored
    path = cache.path_for(cache.key_for(V, 1e-6, N_VARS))
    before = path.read_bytes()
    second = cache.store(V, 1e-6, N_VARS, result)
    assert second.store_reason == "vproof_cache_store_skipped_valid"
    assert second.stored is False
    assert path.read_bytes() == before, "a valid entry was overwritten"
    # A DAMAGED entry is not a valid one, so it is replaced rather than kept.
    path.write_bytes(b"{not json")
    third = cache.store(V, 1e-6, N_VARS, result)
    assert third.store_reason == "vproof_cache_stored" and third.stored


# ---------------------------------------------------------------------------
# 7. The corner level, in review/reverify_exact.py
# ---------------------------------------------------------------------------

GATE = (REPO / "artifacts/campaign_2026-08/E1_matrix"
        / "validation_gate_2026-09-03/B_designpoint")


def _reverify():
    sys.path.insert(0, str(REPO / "review"))
    import reverify_exact  # noqa: E402
    return reverify_exact


def _gate_V(name):
    doc = json.loads((GATE / "design" / f"{name}.json").read_text())
    coeff = (lambda t: float.fromhex(t) if isinstance(t, str) else float(t))
    V = GeneralizedForm(
        degree_m=doc["V"]["degree_m"],
        weights_r=tuple(doc["V"]["weights_r"]),
        terms=[GFTerm(coefficient=coeff(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in doc["V"]["terms"]])
    return V, doc


def test_the_corner_worker_entry_point_is_module_level_and_picklable():
    import pickle
    rv = _reverify()
    assert pickle.loads(pickle.dumps(rv._corner_phase)) is rv._corner_phase
    assert rv.CORNER_WORKER_REASONS == {
        "corner_worker_exception", "corner_worker_died",
        "corner_pool_unavailable"}
    err = rv.CornerWorkerError("corner_worker_died", "d", corner=2)
    assert err.reason in rv.CORNER_WORKER_REASONS and err.corner == 2


@pytest.mark.slow
def test_corner_fanout_reproduces_the_serial_corner_box_exactly():
    """4 real bracket corners, MOSEK-free (the design Gram bundle witnesses).

    The corner box must come back in CORNER ORDER with identical per-octant
    exact results, whichever worker finished first, and corner 0's V proof
    must be the one every later corner reuses in both arms (R2's dedup
    crossing a process boundary by pickle).
    """
    if not (GATE / "design" / "levant4d.json").exists():
        pytest.skip("P2-06 gate design point not present")
    rv = _reverify()
    V, doc = _gate_V("levant4d")
    bundle = GATE / "bundles" / "levant4d.npz"
    common = dict(epsilon=doc["epsilon"], solver="MOSEK",
                  gram_bundle=bundle, fallback_solver=None)
    serial = rv.printed_rational_phases(V, "levant4d", jobs=1, **common)
    parallel = rv.printed_rational_phases(V, "levant4d", jobs=4, **common)
    assert len(serial) == len(parallel) == 4
    for (ed_a, pa), (ed_b, pb) in zip(serial, parallel):
        assert ed_a.corner_label == ed_b.corner_label
        assert digest(pa.v_exact) == digest(pb.v_exact)
        assert digest(pa.endpoints[0].exact) == digest(pb.endpoints[0].exact)
        assert pa.exact_pass == pb.exact_pass is True
        assert pa.v_fingerprint == pb.v_fingerprint
        assert pa.v_input_hash == pb.v_input_hash
    # corner 0 proves V, corners 1-3 reuse it, in BOTH arms
    assert [p.v_reused for _, p in serial] == [False, True, True, True]
    assert [p.v_reused for _, p in parallel] == [False, True, True, True]
    # and the provenance says where the budget went
    assert parallel[0][1].parallel["jobs"] == 4
    assert parallel[1][1].parallel["jobs"] == 1
