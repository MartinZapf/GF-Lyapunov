"""Shared orchestration of the Proposition-7 exact phase.

One canonical implementation of the flow that was previously replicated in
``review/reverify_exact.py`` and ``scripts/levant_2003_perturbed.py``:

    1. strict re-solve of V with Gram capture (independent solver, no fallback),
    2. per dynamics (nominal, or one per inclusion endpoint vertex):
       W = −Lie(V, f) re-solved with Gram capture, the exact W rebuilt from the
       rationalized V and dynamics, and both proved by the exact gate,
    3. a float-vs-exact Lie-derivative cross-check per dynamics (self-test of
       the exact mirror against the float pipeline).

For a perturbed inclusion pass ``dynamics_list=[f_plus, f_minus]`` (the
endpoint vertices): W is affine in the matched disturbance, so both endpoints
passing certifies the whole interval by convexity — see
``matched_disturbance_vertices``. Float δ₀/gains are read as exact dyadic
rationals (``Fraction(float)``); coefficients that are already ``Fraction``
(exact-rational gain injection, ``scripts/exact_gate_standard_gains.py``)
pass through untouched.

This module deliberately lives OUTSIDE ``exact_gate.py``: the gate stays a
pure exact-arithmetic module that never imports the float design/verify
pipeline; this orchestrator owns that boundary.

V deduplication across bracket corners — campaign item R2
---------------------------------------------------------
Under a rational gain bracket the exact phase runs once per corner of the
2^k box. V is GAIN-INDEPENDENT: ``exact_octant_target`` consumes only the V
terms, μ, the octant signs and ε, and no dynamics coefficient reaches it. So
the per-corner V re-solve and V exact proof reproduce corner 0 identically.
Measured on the stored levant4d record, that repeat is 54.0% of the whole
run (V work is 72%, three of four corners redundant).

``run_exact_phase`` therefore accepts a :class:`VProofCache`. The cache is
keyed by :func:`v_input_hash`, an exact content hash over the V terms as
rationals, the weights, and the exact ε — precisely the inputs that decide
the V proof. A corner whose inputs hash differently is REFUSED
(:class:`ExactPhaseError`), never silently accepted, so the dedup can only
skip work that was provably going to repeat. The reused proof additionally
carries a fingerprint over μ, the per-octant signs, bases and η̂ values, so an
artifact reader can see exactly which V proof was shared.

Drivers keep a ``--no-dedup`` escape hatch that restores the per-corner
re-proof. Provenance: PIPELINE.md §3 R2.

That cache lives in ONE process. An exact closure that runs one corner per
process and assembles the row afterwards never crosses corners with it
(measured 13830 / 11471 / 13238 s per levant4d E4 corner with no reuse,
against 831 s then 138 s in-process), so ``run_exact_phase`` also takes a
``v_proof_cache``: a :class:`DiskVProofCache` keyed by the same exact input,
which persists the V Gram captures and re-proves them on the next process's
hit. See "The V proof persisted ACROSS corner processes" below.

The design Gram as the witness — campaign item R3, decisions D1/D2
------------------------------------------------------------------
Historically the exact phase PROPOSED its own witness: it re-solved V and
every W with an independent float solver and exact-verified the Grams that
came back. That re-solve is the only float SDP left on the certifying route,
it goes through the CVXPY encoding (``verify_gf_sos_full``), and it is
expensive — the measured 831 s residual in corner 0 of the levant4d exact
row is exactly those two solves.

The DESIGN solve already produced a Gram per block. ``run_exact_phase`` can
therefore take that bundle (``JointSOSResult.gram_certificates``,
``joint_sos.GRAM_BUNDLE_SCHEMA``) as the witness and skip the re-solve
entirely, which makes the citability chain D → X → P: MOSEK-direct design,
exact re-proof on the design Gram, stdlib portable verifier.

What does NOT change, and is what keeps this sound:

  * the exact octant targets are still rebuilt FROM SCRATCH in ℚ
    (``exact_gate.exact_octant_target`` on the exactly recomputed V/W terms).
    The bundle contributes Q and its basis, nothing else. A bad Gram can only
    make the residual/charge grow, never make the gate accept;
  * the bundle is refused, loudly, when it does not belong to the target
    being built. Modelled on the R2 ``v_input_hash`` refusal above: every
    entry's content hash and the bundle fingerprint are re-derived
    (``gram_bundle_hash_mismatch``), and the bundle's own stored target,
    margin, μ, octant signs and clearing power are checked against the
    exact target assembled here (``gram_bundle_input_mismatch``);
  * ``gram_bundle=None`` (the default) restores the historical re-solve path,
    byte for byte.

A design run with the parity reduction ACTIVE emits only the representative
blocks (2^(n-1) V blocks). Their ``parity_class`` names the partner octant
and endpoint, so the missing half is expanded here from the representatives
and the record states, per octant, that the witness came from the partner.
The expansion is not trusted either: the partner's exact target is rebuilt
independently at its own signs and compared, so a parity identity that does
not actually hold is a refusal, not a proof.

Provenance: PLAN_2026-09-02 D1/D2/P2-03, PIPELINE.md §4.

Process parallelism over octants — campaign item P3-03
------------------------------------------------------
After R3 removed the float re-solve from the certifying route, the exact
rational arithmetic is what a row costs, and the 2ⁿ octants of a form are
independent: each rebuilds its own target in ℚ from the same terms and runs
its own LDLᵀ. ``run_exact_phase(..., jobs=N)`` spreads them over N worker
processes (``spawn``; see :data:`OCTANT_START_METHOD` for why not fork).

``jobs=1`` is the default and creates no pool: the serial loop in
``exact_gate.exact_verify_captures`` runs unchanged, so no stored baseline
moves. What the parallel path may NOT change is anything a record states, and
it does not:

  * the octant ORDER is the capture order, always. Workers yield in completion
    order and the gate assembles by index, so the per-octant digest, the
    truncation at the first failing octant and the recorded ``reason`` are the
    same objects a serial run produces;
  * a worker never decides anything. It receives the exact terms and one Gram
    and returns one ``ExactOctantCheck``; the verdict, the adoption ladder,
    the cache and the bundle refusals all stay in the parent;
  * a worker that raises or dies is a typed :class:`OctantWorkerError`
    (:data:`OCTANT_WORKER_REASONS`), which fails the witness attempt. It is
    never a shorter, passing certificate and never a hang.

``jobs`` and the start method are recorded in ``ExactPhaseResult.parallel``.

Which witness is adopted — campaign item P2-05, decision D1
-----------------------------------------------------------
There are THREE witness sources, tried in this order:

  1. the DESIGN Gram bundle (above), when one is supplied and accepted;
  2. a MOSEK re-solve (:data:`WITNESS_PRIMARY_SOLVER`, the default of
     ``run_exact_phase`` since P2-05);
  3. exactly ONE retry with CLARABEL (:data:`WITNESS_FALLBACK_SOLVER`).

A source is left behind when the exact phase it produced does not prove the
certificate (``exact_pass`` false) or when it proves it with less than a
factor :data:`WITNESS_CHARGE_TRIGGER` of headroom against the Proposition-7
acceptance cliff (worst ``charge/ε`` above 0.1 since WP P2-14; it was
``charge/(ε/2)`` above 0.1 before, so the trigger now fires at twice the
charge). Measured motivation (E5, 3 reps): worst charge/budget with a CLARABEL
witness 4.80e-7 (twisting), 8.26e-7 (levant3d), 1.26e-6 (pidsmc); with MOSEK
6.07e-2, 2.85e-1, 1.07e-2 — all pass, but MOSEK already burns 28.5% of the
ε/2 budget at n=3, m=5. Those three numbers are charge/(ε/2) ratios, measured
before the flip.

Two things this policy is NOT:

  * it is not a trust decision. A witness is adopted because the EXACT
    rational phase verified it, never because a solver reported a status.
    ``solver_infeasible`` stays typed float evidence (D6);
  * it is not a silent fallback. Every attempt is recorded — source, solver,
    verdict, worst charge ratio, typed shortfall — in
    ``ExactPhaseResult.witness_attempts``, and the adopted one is named by
    ``witness_source`` / ``witness_solver`` / ``witness_charge_ratio``.

A bundle refusal that is a DESIGN-side limit (:data:`BUNDLE_FALLTHROUGH_REASONS`:
the block carries no ε rows, or the weighted-sphere clearing changed its
polynomial — e.g. ``power_rate``'s odd-y-degree W block, P2-02/P2-04) falls
through to the re-solve as a recorded attempt. Every other refusal —
hash, input, coverage, schema, parity, empty — means the bundle is not the
object it claims to be and stays a hard stop that raises.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import copy
import errno
import gzip
import hashlib
import json
import multiprocessing
import os
import time
import uuid
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf, negate_gf
from gf_framework.algebra.symmetry import flip_endpoint, flip_octant, octant_index
from gf_framework.verification.exact_gate import (
    MARGIN_POLICIES,
    MARGIN_POLICY_ACHIEVED,
    MARGIN_POLICY_HALF,
    ExactCaptureResult,
    ExactGateError,
    resolve_margin_policy,
    exact_neg_lie_derivative,
    exact_octant_target,
    exact_terms_from_gf,
    exact_verify_captures,
    validate_one_capture,
)
from gf_framework.verification.failure_typing import classify_exception
from gf_framework.verification.joint_sos import (
    GRAM_BUNDLE_SCHEMA,
    gram_bundle_entry_hash,
    gram_bundle_fingerprint,
    verify_gf_sos_full,
)

#: ``ExactPhaseResult.witness_source`` when the phase proposed its own Gram
#: with an independent float re-solve (the historical path).
WITNESS_RESOLVE = "resolve"

#: ``ExactPhaseResult.witness_source`` when the DESIGN-stage Gram bundle was
#: the witness and no float solve ran on the certifying route (R3).
WITNESS_DESIGN_BUNDLE = "design_bundle"

#: Relative tolerance of the bundle-vs-exact target comparison. The two are
#: the same polynomial computed twice, once as a float dot product in the
#: design and once exactly here, so they agree to float round-off (measured
#: ≤ 1e-12 relative on the 12 nominal systems). The check exists to catch a
#: bundle from a DIFFERENT instance, where the disagreement is O(1); it is
#: deliberately not tight enough to reject a rational-bracket corner, whose
#: gains differ from the design floats in the 15th digit and whose (real)
#: discrepancy the exact gate charges against the margin anyway.
BUNDLE_TARGET_RTOL = 1e-6

#: Default float solver of the exact phase's own re-solve (P2-05, D1). The
#: strict CLARABEL gate left the citability chain; MOSEK is the design-stage
#: solver and now also the exact phase's primary witness proposer.
WITNESS_PRIMARY_SOLVER = "MOSEK"

#: The ONE retry, when the primary witness fails or leaves too little headroom
#: (D1: "a measured fallback witness inside X, retried once when the exact
#: charge ratio exceeds 0.1 or the MOSEK witness fails"). Set
#: ``fallback_solver=None`` to disable the retry entirely.
WITNESS_FALLBACK_SOLVER = "CLARABEL"

#: Worst exact ``charge/ε`` ratio above which a witness that DID prove the
#: certificate is still retried, because it sits within a factor 10 of the
#: Proposition-7 acceptance cliff ``charge = ε``. A retry never weakens a
#: proof: the better of the two verified witnesses is adopted and both are
#: recorded.
#:
#: WP P2-14 re-expressed the DENOMINATOR against ε (it used to be the margin
#: policy's own budget), so the trigger no longer moves when the policy does.
#: ``charge/ε > 0.1`` is 0.2 of the old ``charge ≤ ε/2`` budget, i.e. the
#: trigger now fires at twice the charge it fired at before the flip. That is
#: the outcome D1 asks for: on the levant4d E8 32-term row at ε = 1e-6 the
#: design bundle charges 0.0537 ε and is adopted in ONE attempt instead of
#: being sent to a re-solve it does not need (P2-13 §Acceptance).
WITNESS_CHARGE_TRIGGER = 0.1

#: ``witness_solver`` for a design-bundle witness. ``joint_sos`` populates
#: ``gram_certificates`` from the MOSEK_DIRECT Task-API backend only (P2-01),
#: so that is the solver that produced the Gram; a bundle document that
#: records its own ``solver`` in the metadata overrides this.
DESIGN_BUNDLE_SOLVER = "MOSEK_DIRECT"

#: :class:`GramBundleError` reasons that are DESIGN-side limits rather than a
#: wrong or tampered bundle: the block carries no ε rows (P2-02
#: ``eps_reason``, e.g. ``w_degree_not_positive_even`` on ``power_rate``), or
#: the weighted-sphere clearing means the block certifies a different
#: polynomial. There is simply no bundle witness for such a block, so the
#: phase falls through to the typed re-solve and records the refusal as a
#: failed attempt. Every other reason raises.
BUNDLE_FALLTHROUGH_REASONS = frozenset({
    "gram_bundle_margin_missing",
    "gram_bundle_clearing_active",
})


class ExactPhaseError(RuntimeError):
    """A cached V proof does not match the certificate it is being reused for."""


class GramBundleError(ExactPhaseError):
    """A supplied design Gram bundle cannot witness this exact target.

    Always carries a machine-readable ``reason``, so a driver can record WHY
    a bundle was refused instead of a prose string. The two reasons the
    protocol names explicitly:

    ``gram_bundle_hash_mismatch``
        an entry's content hash, or the bundle fingerprint, does not
        reproduce. The bundle is not the object it claims to be.
    ``gram_bundle_input_mismatch``
        the bundle is internally consistent but belongs to a different
        instance: its stored target, margin, μ, octant labels or block count
        disagree with the exact target being built here.

    The remaining reasons are structural refinements of the second one and
    are typed separately so an artifact can distinguish them:
    ``gram_bundle_empty``, ``gram_bundle_schema_unknown``,
    ``gram_bundle_coverage_incomplete``, ``gram_bundle_margin_missing``
    (the block carries no ε rows; the detail names the ``eps_reason`` from
    the design, e.g. ``w_degree_not_positive_even``),
    ``gram_bundle_clearing_active`` (the weighted-sphere clearing changed the
    block's polynomial, so it is not the exact target),
    ``gram_bundle_parity_inconsistent``.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def crosscheck_lie(w_exact_terms, W_gf_float) -> Dict[str, object]:
    """Self-test of the exact Lie mirror against the float pipeline.

    Compares support and coefficients of the exactly recomputed W against the
    float pipeline's W. A support mismatch or a large coefficient gap means
    the two Lie-derivative implementations have diverged — the exact gate
    would still be sound (it charges any discrepancy against the margin), but
    the mirror self-test is the early alarm.
    """
    fkeys = {(t.exponents, t.term_types): float(t.coefficient)
             for t in W_gf_float.terms}
    ekeys = {(e, t): float(c) for c, e, t in w_exact_terms}
    support_match = set(fkeys) == set(ekeys)
    max_diff = max(
        (abs(fkeys.get(k, 0.0) - ekeys.get(k, 0.0)) for k in set(fkeys) | set(ekeys)),
        default=0.0,
    )
    return {"support_match": support_match, "max_coeff_diff": max_diff}


def _exact_epsilon(epsilon) -> Fraction:
    """ε as the exact rational the pipeline actually uses (dyadic for floats)."""
    return epsilon if isinstance(epsilon, Fraction) else Fraction(float(epsilon))


def v_input_hash(V_gf, epsilon) -> str:
    """Exact content hash of everything the V exact proof depends on.

    Covers the V terms as exact rationals (coefficient, exponents, term
    types), the homogeneity weights, and the exact ε. μ and the Gram bases
    are deterministic functions of these plus the solver-independent lifting,
    and are pinned separately by :func:`v_proof_fingerprint`.

    Canonical: terms are sorted, so term ORDER in the reconstructed V cannot
    change the hash, while any coefficient, exponent, type, weight or margin
    change does.
    """
    h = hashlib.sha256()
    h.update(b"gf-exact-v-input/1\n")
    h.update(repr(tuple(int(r) for r in V_gf.weights_r)).encode())
    h.update(b"\n")
    eps = _exact_epsilon(epsilon)
    h.update(f"{eps.numerator}/{eps.denominator}\n".encode())
    rows = []
    for coeff, exps, types in exact_terms_from_gf(V_gf):
        rows.append(
            f"{coeff.numerator}/{coeff.denominator}|"
            + ",".join(f"{e.numerator}/{e.denominator}" for e in exps)
            + "|" + ",".join(types)
        )
    for row in sorted(rows):
        h.update(row.encode())
        h.update(b"\n")
    return h.hexdigest()


def v_proof_fingerprint(exact: ExactCaptureResult, grams: Optional[List[dict]]) -> str:
    """Fingerprint of the V proof OUTPUT (μ, ε, per-octant signs/basis/η̂).

    Recorded alongside the reuse so an artifact states which concrete proof
    was shared across corners. Not a security boundary: the refusal that
    matters is the :func:`v_input_hash` comparison.
    """
    h = hashlib.sha256()
    h.update(b"gf-exact-v-proof/1\n")
    h.update(f"mu={tuple(exact.mu)} eps={exact.eps}\n".encode())
    for chk in sorted(exact.octants, key=lambda c: tuple(c.signs)):
        h.update(f"{tuple(chk.signs)}|{chk.degree}|{chk.n_basis}|{chk.eta_hat}|"
                 f"{chk.n_blocks}|{chk.block_sizes}\n".encode())
    for cap in sorted(grams or [], key=lambda c: tuple(c["signs"])):
        h.update(f"{tuple(cap['signs'])}|{cap['y_degree']}|"
                 f"{len(cap['basis'])}\n".encode())
    return h.hexdigest()


@dataclass
class VProofCache:
    """A V exact proof, reusable across bracket corners (R2).

    Built from corner 0 by :func:`build_v_cache` and handed to every later
    ``run_exact_phase`` call. ``input_hash`` is re-derived and compared on
    each reuse; a mismatch raises instead of proceeding.
    """
    input_hash: str
    fingerprint: str
    resolve_feasible: bool
    exact: ExactCaptureResult
    grams: Optional[List[dict]] = None


def build_v_cache(V_gf, epsilon, result: "ExactPhaseResult") -> VProofCache:
    """Freeze the V half of a completed exact phase for reuse at other corners."""
    cache = VProofCache(
        input_hash=result.v_input_hash,
        fingerprint=v_proof_fingerprint(result.v_exact, result.v_grams),
        resolve_feasible=result.v_resolve_feasible,
        exact=result.v_exact,
        grams=result.v_grams,
    )
    return _checked_v_cache(cache, V_gf, epsilon, len(V_gf.weights_r),
                            result.v_exact.margin_policy)


def _checked_v_cache(cache, V_gf, epsilon, n_vars, margin_policy):
    """Bind a completed V proof to its input, coverage and requested policy.

    This checks a trusted in-process proof result, not an arbitrary serialized
    verdict. Disk inputs re-run exact_verify_captures by default. Changing the
    acceptance policy needs no new LDL: the exact achieved margins decide it.
    """
    policy = resolve_margin_policy(margin_policy)
    expected_hash = v_input_hash(V_gf, epsilon)
    if not cache.input_hash or cache.input_hash != expected_hash:
        raise ExactPhaseError("cached V proof does not belong to this certificate: "
                              "source input hash differs from V/weights/epsilon")
    exact = cache.exact
    full = set(product((-1, 1), repeat=n_vars))
    if (not cache.resolve_feasible or not exact.ok
            or exact.n_octants_expected != len(full)
            or len(exact.octants) != len(full)
            or {tuple(c.signs) for c in exact.octants} != full
            or not all(c.ok for c in exact.octants)):
        raise ExactPhaseError("cached V proof is not a complete passing octant proof")
    eps = _exact_epsilon(epsilon)
    try:
        if Fraction(exact.eps) != eps or eps <= 0:
            raise ValueError("proof epsilon differs")
        if (not cache.grams or len(cache.grams) != len(full)
                or {tuple(c["signs"]) for c in cache.grams} != full
                or any(_exact_epsilon(c["eps"]) != eps
                       or tuple(c["mu"]) != tuple(exact.mu) for c in cache.grams)):
            raise ValueError("Gram coverage, epsilon or lift differs")
        for c in exact.octants:
            certified = Fraction(c.eps_certified)
            if not 0 < certified <= eps:
                raise ValueError("missing or invalid exact achieved margin")
            if policy == MARGIN_POLICY_HALF and certified < eps / 2:
                raise ValueError("cached V margin does not satisfy half policy")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ExactPhaseError(f"cached V proof is not reusable: {exc}") from exc
    if v_proof_fingerprint(exact, cache.grams) != cache.fingerprint:
        raise ExactPhaseError("cached V proof fingerprint differs")
    # Keep the original proof unmodified and make all reported denominators
    # agree with the policy requested by this invocation.
    if exact.margin_policy != policy or any(c.margin_policy != policy for c in exact.octants):
        adjusted = copy.deepcopy(exact)
        adjusted.margin_policy = policy
        for c in adjusted.octants:
            c.margin_policy = policy
            c.budget_float = float(eps / 2 if policy == MARGIN_POLICY_HALF else eps)
        cache = copy.copy(cache)
        cache.exact = adjusted
    return cache


# ---------------------------------------------------------------------------
# Design Gram bundle: serialization, hash refusal, parity expansion (R3)
# ---------------------------------------------------------------------------

def gram_bundle_to_jsonable(entries: Sequence[dict], **meta: Any) -> Dict[str, Any]:
    """JSON-ready document for a design Gram bundle.

    Round trips through :func:`gram_bundle_from_jsonable` with the per-entry
    hashes intact: tuples become lists, the ``target`` dict becomes a list of
    ``[exponents, coefficient]`` pairs sorted by exponent (the same rendering
    ``joint_sos._canonical_hash_value`` hashes), and the Gram becomes nested
    float lists. Python's float repr round trips exactly, so the reloaded
    bundle hashes to the same digest as the in-memory one.

    Private keys (``_``-prefixed, e.g. the parity provenance added by the
    expansion below) are dropped: they are derived, not part of the object.
    """
    out: List[Dict[str, Any]] = []
    for entry in entries:
        doc = {k: v for k, v in entry.items() if not k.startswith("_")}
        doc["signs"] = [int(s) for s in entry["signs"]]
        doc["mu"] = [int(v) for v in entry["mu"]]
        doc["basis"] = [[int(e) for e in b] for b in entry["basis"]]
        doc["target"] = [[[int(e) for e in mono], float(c)]
                         for mono, c in sorted(entry["target"].items())]
        doc["gram"] = [[float(x) for x in row]
                       for row in np.asarray(entry["gram"], dtype=np.float64)]
        out.append(doc)
    document = {
        "schema": GRAM_BUNDLE_SCHEMA,
        "fingerprint": gram_bundle_fingerprint(entries),
        "n_entries": len(out),
        "entries": out,
    }
    document.update(meta)
    return document


def _decode_bundle_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """One serialized entry back into the in-memory shape the gate consumes."""
    entry = dict(raw)
    entry["signs"] = tuple(int(s) for s in raw["signs"])
    entry["mu"] = tuple(int(v) for v in raw["mu"])
    entry["basis"] = [tuple(int(e) for e in b) for b in raw["basis"]]
    target = raw["target"]
    if isinstance(target, dict):
        entry["target"] = {tuple(int(e) for e in k): float(v)
                           for k, v in target.items()}
    else:
        entry["target"] = {tuple(int(e) for e in mono): float(c)
                           for mono, c in target}
    entry["gram"] = np.asarray(raw["gram"], dtype=np.float64)
    return entry


def gram_bundle_from_jsonable(document: Dict[str, Any]) -> Tuple[List[dict], dict]:
    """Inverse of :func:`gram_bundle_to_jsonable`; returns ``(entries, meta)``."""
    entries = [_decode_bundle_entry(e) for e in document.get("entries") or []]
    meta = {k: v for k, v in document.items() if k != "entries"}
    return entries, meta


def save_gram_bundle(path, entries: Sequence[dict], **meta: Any) -> Path:
    """Write a design Gram bundle to ``.json``, ``.json.gz`` or ``.npz``.

    ``.npz`` keeps the Grams as float64 arrays and the metadata as one JSON
    string, which is what makes a large bundle (levant4d: 2^4 · 2 blocks)
    cheap to store and to reload; the JSON forms stay human-readable. All
    three reload to the same entries and the same fingerprint.
    """
    path = Path(path)
    document = gram_bundle_to_jsonable(entries, **meta)
    if path.suffix == ".npz":
        grams = {}
        for i, doc in enumerate(document["entries"]):
            grams[f"gram_{i}"] = np.asarray(doc.pop("gram"), dtype=np.float64)
        np.savez_compressed(path, _document=np.array(json.dumps(document)), **grams)
    elif path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(document, fh)
    else:
        path.write_text(json.dumps(document), encoding="utf-8")
    return path


def load_gram_bundle(source) -> Tuple[List[dict], dict]:
    """Normalize any accepted bundle argument to ``(entries, meta)``.

    Accepts, in this order: a path (``str``/``Path``) to ``.json``,
    ``.json.gz`` or ``.npz``; a packaged document dict (``{"entries": [...]}``
    or ``{"gram_certificates": [...]}``); any object exposing
    ``.gram_certificates`` (a :class:`~gf_framework.verification.joint_sos.JointSOSResult`);
    or the plain list of entries. Nothing is verified here — that is
    :func:`prepare_gram_bundle`'s job.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                document = json.loads(str(z["_document"].item()))
                for i, doc in enumerate(document.get("entries") or []):
                    doc["gram"] = z[f"gram_{i}"].tolist()
        elif path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                document = json.load(fh)
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
        entries, meta = gram_bundle_from_jsonable(document)
        meta["source_path"] = str(path)
        return entries, meta
    if isinstance(source, dict):
        raw = source.get("entries")
        if raw is None:
            raw = source.get("gram_certificates")
        if raw is None:
            raise GramBundleError(
                "gram_bundle_empty",
                "bundle dict carries neither 'entries' nor 'gram_certificates'")
        meta = {k: v for k, v in source.items()
                if k not in ("entries", "gram_certificates")}
        return [_maybe_decode(e) for e in raw], meta
    if hasattr(source, "gram_certificates"):
        entries = source.gram_certificates
        if not entries:
            raise GramBundleError(
                "gram_bundle_empty",
                f"{type(source).__name__}.gram_certificates is empty; the "
                "design captured no Gram (not MOSEK_DIRECT, infeasible, or "
                "capture_design_grams=False)")
        return [_maybe_decode(e) for e in entries], {}
    return [_maybe_decode(e) for e in source], {}


def _maybe_decode(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Decode an entry that came through JSON; pass an in-memory one through."""
    if isinstance(entry.get("gram"), np.ndarray) and isinstance(
            entry.get("target"), dict) and isinstance(entry.get("signs"), tuple):
        return entry
    return _decode_bundle_entry(entry)


def _bundle_witness_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Per-octant witness provenance, JSON-ready, for the exact record."""
    source = entry.get("_parity_source")
    return {
        "octant_index": int(entry["octant_index"]),
        "signs": [int(s) for s in entry["signs"]],
        "block_name": entry.get("name"),
        "entry_hash": entry.get("hash"),
        "from_parity_partner": source is not None,
        "parity_source": source,
    }


def _expand_parity_partners(entries: Sequence[dict], n_vars: int) -> List[dict]:
    """Cover all 2^n octants of a parity-REDUCED bundle from its representatives.

    A design run with the parity reduction active emits one block per orbit of
    the involution ``(s, σ) ↦ (−s, −σ)`` (``algebra.symmetry``), so the bundle
    holds 2^(n-1) V blocks and one endpoint of each W pair. Each such entry
    carries ``parity_class`` naming its partner, and the partner is certified
    by the SAME Gram: the associated forms of the two labels coincide
    (Φ_{−s}^V = Φ_s^V for an even V, Φ_{−s}^{W,σ} = Φ_s^{W,−σ}).

    This function only RELABELS a copy of the representative entry. Nothing
    here is trusted: the partner's exact target is rebuilt from scratch at the
    partner's own signs (and, for W, from the partner endpoint's dynamics) and
    compared in :func:`_check_bundle_group`, so a parity identity that does
    not hold on this instance surfaces as ``gram_bundle_input_mismatch``, and
    any residual it does leave is charged against the margin by the gate.

    Entries with ``parity_class is None`` (an unreduced design) pass through
    untouched and nothing is added.
    """
    out: List[dict] = []
    for entry in entries:
        out.append(entry)
        parity = entry.get("parity_class")
        if not parity:
            continue
        if parity.get("role") != "representative":
            raise GramBundleError(
                "gram_bundle_parity_inconsistent",
                f"block {entry.get('name')!r} carries parity_class role "
                f"{parity.get('role')!r}; only 'representative' blocks are "
                "emitted by a reduced design")
        signs = tuple(int(s) for s in entry["signs"])
        partner_signs = flip_octant(signs)
        partner_index = octant_index(partner_signs)
        declared = parity.get("partner_octant_index")
        if declared is not None and int(declared) != partner_index:
            raise GramBundleError(
                "gram_bundle_parity_inconsistent",
                f"block {entry.get('name')!r} declares partner octant "
                f"{int(declared)}, but the antipodal octant of {signs} is "
                f"{partner_index}")
        endpoint = entry.get("endpoint")
        if entry.get("block") == "W" and endpoint is not None:
            if parity.get("mode") == "nominal":
                # A nominal reduction has no endpoint involution: w_blocks()
                # labels the single dynamics as endpoint None while the block
                # itself is vertex 0, so the partner stays on vertex 0.
                partner_endpoint: Optional[int] = int(endpoint)
            else:
                try:
                    partner_endpoint = flip_endpoint(int(endpoint))
                except Exception as exc:  # ParityReductionError and friends
                    raise GramBundleError(
                        "gram_bundle_parity_inconsistent",
                        f"block {entry.get('name')!r} carries endpoint "
                        f"{endpoint!r}, which has no partner under the "
                        f"two-endpoint involution ({exc})") from exc
                declared_ep = parity.get("partner_endpoint")
                if declared_ep is not None and int(declared_ep) != partner_endpoint:
                    raise GramBundleError(
                        "gram_bundle_parity_inconsistent",
                        f"block {entry.get('name')!r} declares partner endpoint "
                        f"{int(declared_ep)}, the endpoint involution gives "
                        f"{partner_endpoint}")
        else:
            partner_endpoint = endpoint
        partner = dict(entry)
        partner["signs"] = partner_signs
        partner["octant_index"] = partner_index
        partner["endpoint"] = partner_endpoint
        partner["name"] = f"{entry.get('name')}#parity"
        # A relabelled copy is NOT the hashed object, so it must not claim the
        # representative's digest: it names its source instead.
        partner["hash"] = None
        partner["_parity_source"] = {
            "octant_index": int(entry["octant_index"]),
            "signs": [int(s) for s in signs],
            "endpoint": None if endpoint is None else int(endpoint),
            "block_name": entry.get("name"),
            "entry_hash": entry.get("hash"),
            "mode": parity.get("mode"),
        }
        out.append(partner)
    return out


@dataclass
class PreparedGramBundle:
    """A hash-verified, parity-expanded design bundle, split by block.

    ``v`` holds the 2^n V entries and ``w`` maps an inclusion-vertex index to
    its 2^n W entries, both sorted by octant index. ``fingerprint`` is over
    the bundle AS SUPPLIED (before expansion), which is what an artifact
    records and what a re-check reproduces.
    """
    fingerprint: str
    v: List[dict] = field(default_factory=list)
    w: Dict[int, List[dict]] = field(default_factory=dict)
    n_entries: int = 0
    parity_expanded: bool = False
    meta: dict = field(default_factory=dict)


def prepare_gram_bundle(
    source,
    n_vars: int,
    *,
    expected_fingerprint: Optional[str] = None,
) -> PreparedGramBundle:
    """Load, hash-verify and parity-expand a design Gram bundle.

    Refusals raised here are structural — they do not yet look at the exact
    target, which :func:`_check_bundle_group` does once the terms are built.
    Modelled on the R2 ``v_input_hash`` guard: a bundle that does not verify
    RAISES, it is never silently ignored or partially used.
    """
    entries, meta = load_gram_bundle(source)
    if not entries:
        raise GramBundleError("gram_bundle_empty", "the bundle holds no entries")

    for i, entry in enumerate(entries):
        schema = entry.get("schema")
        if schema != GRAM_BUNDLE_SCHEMA:
            raise GramBundleError(
                "gram_bundle_schema_unknown",
                f"entry {i} ({entry.get('name')!r}) carries schema {schema!r}, "
                f"this consumer implements {GRAM_BUNDLE_SCHEMA!r}")
        stored = entry.get("hash")
        if not stored:
            raise GramBundleError(
                "gram_bundle_hash_mismatch",
                f"entry {i} ({entry.get('name')!r}) carries no content hash; a "
                "witness that does not identify itself is not admissible")
        recomputed = gram_bundle_entry_hash(entry)
        if recomputed != stored:
            raise GramBundleError(
                "gram_bundle_hash_mismatch",
                f"entry {i} ({entry.get('name')!r}) hashes to {recomputed}, the "
                f"bundle records {stored}; the entry was modified after packaging")
        signs = entry.get("signs") or ()
        if len(signs) != n_vars:
            raise GramBundleError(
                "gram_bundle_input_mismatch",
                f"entry {i} ({entry.get('name')!r}) carries {len(signs)} octant "
                f"signs, this certificate has n_vars={n_vars}")

    fingerprint = gram_bundle_fingerprint(entries)
    for label, claimed in (("bundle document", meta.get("fingerprint")),
                           ("caller", expected_fingerprint)):
        if claimed and str(claimed) != fingerprint:
            raise GramBundleError(
                "gram_bundle_hash_mismatch",
                f"the {label} declares fingerprint {claimed}, the entries hash "
                f"to {fingerprint}")

    expanded = _expand_parity_partners(entries, n_vars)
    prepared = PreparedGramBundle(
        fingerprint=fingerprint,
        n_entries=len(entries),
        parity_expanded=len(expanded) != len(entries),
        meta=meta,
    )
    for entry in expanded:
        block = entry.get("block")
        if block == "V":
            prepared.v.append(entry)
        elif block == "W":
            endpoint = entry.get("endpoint")
            prepared.w.setdefault(0 if endpoint is None else int(endpoint),
                                  []).append(entry)
        else:
            raise GramBundleError(
                "gram_bundle_schema_unknown",
                f"entry {entry.get('name')!r} declares block {block!r}, "
                "expected 'V' or 'W'")
    prepared.v.sort(key=lambda e: int(e["octant_index"]))
    for group in prepared.w.values():
        group.sort(key=lambda e: int(e["octant_index"]))
    return prepared


def _check_bundle_group(
    entries: Sequence[dict],
    terms,
    n_vars: int,
    eps_exact: Fraction,
    label: str,
    *,
    rtol: float = BUNDLE_TARGET_RTOL,
) -> None:
    """Refuse a bundle group that does not belong to this exact target.

    The substantive check: the target the DESIGN stored for a block is
    rebuilt here from scratch in exact arithmetic at the block's own μ, octant
    signs and margin, and the two must agree to float round-off. A bundle from
    another system, another degree, another gain point or another ε disagrees
    by O(1) and is refused. This is the input-hash refusal of R2 transposed to
    an object the design did not hash against the certificate.
    """
    expected = 2 ** n_vars
    seen = sorted(tuple(int(s) for s in e["signs"]) for e in entries)
    if len(entries) != expected or len(set(seen)) != expected:
        raise GramBundleError(
            "gram_bundle_coverage_incomplete",
            f"{label}: {len(set(seen))} distinct octants over {len(entries)} "
            f"entries, need all {expected}. A parity-reduced bundle is expanded "
            "from parity_class; this one covers less than the full set")
    for entry in entries:
        name = entry.get("name")
        clearing = int(entry.get("clearing_power") or 0)
        if clearing:
            raise GramBundleError(
                "gram_bundle_clearing_active",
                f"{label} block {name!r} was designed with weighted-sphere "
                f"clearing k={clearing}; the block certifies "
                "∏ᵢ|xᵢ|^{k·ρ/rᵢ}·W, not W, so it is not a witness for this "
                "exact target")
        eps_entry = float(entry.get("eps") or 0.0)
        if eps_entry <= 0.0:
            raise GramBundleError(
                "gram_bundle_margin_missing",
                f"{label} block {name!r} carries eps={eps_entry!r} "
                f"(design reason: {entry.get('eps_reason') or 'unrecorded'}); "
                "Proposition 7's acceptance threshold would be zero, so the "
                "block proves no strict margin and cannot be a witness")
        if Fraction(eps_entry) != eps_exact:
            raise GramBundleError(
                "gram_bundle_input_mismatch",
                f"{label} block {name!r} subtracted eps={eps_entry!r}, this "
                f"exact phase is building the target at eps={float(eps_exact)!r}")
        signs = tuple(int(s) for s in entry["signs"])
        mu = tuple(int(v) for v in entry["mu"])
        try:
            target, _d = exact_octant_target(terms, mu, signs, eps_exact)
        except ExactGateError as exc:
            raise GramBundleError(
                "gram_bundle_input_mismatch",
                f"{label} block {name!r}: the exact target does not assemble at "
                f"the bundle's mu={mu}, signs={signs} ({exc})") from exc
        stored = entry.get("target") or {}
        scale = max(
            max((abs(float(v)) for v in target.values()), default=0.0),
            max((abs(float(v)) for v in stored.values()), default=0.0),
        )
        tol = rtol * scale
        worst, worst_mono = 0.0, None
        for mono in set(target) | set(stored):
            gap = abs(float(target.get(mono, 0)) - float(stored.get(mono, 0.0)))
            if gap > worst:
                worst, worst_mono = gap, mono
        if worst > tol:
            raise GramBundleError(
                "gram_bundle_input_mismatch",
                f"{label} block {name!r}: the design target and the exact "
                f"target differ by {worst:.6g} at monomial {worst_mono} "
                f"(tolerance {tol:.6g} = {rtol:g} x {scale:.6g}); the bundle "
                "belongs to a different system, degree, gain point or margin")


# ---------------------------------------------------------------------------
# The V proof persisted ACROSS corner processes — campaign item P3-05
# ---------------------------------------------------------------------------
#
# :class:`VProofCache` above lives in one process. An exact closure that
# writes one JSON per corner and assembles the ROW afterwards (the E4 levant4d
# pattern, ``scripts/campaign/exact_closure/``) therefore never reuses the V
# proof: measured 13830 / 11471 / 13238 s per corner with no reuse, against
# 831 s then 138 s for the same work in one process
# (``artifacts/campaign_2026-08/E4_levant4d/README.md``). Checkpointing per
# corner exists for a reason (timeout and OOM kills), so the fix is to persist
# the cache, not to merge the processes.
#
# What is persisted is the WITNESS, not the verdict: the per-octant Gram
# captures of the V half. On a hit the exact rational gate runs again on them
# (``revalidate=True``, the default), so a cached proof is re-decided by the
# current verifier and a stale, foreign or damaged Gram can only make the
# residual grow — never make the gate accept. That is the same soundness
# argument as R3's design-bundle witness. Measured cost of the revalidation on
# levant4d (n = 4, m = 7, ε = 1e-6, 16 octants): 1.74 s against 395.12 s for
# the V float re-solve it replaces, i.e. 0.4% of what it saves.
#
# ``revalidate=False`` restores the in-process semantics exactly (the stored
# ``ExactCaptureResult`` is adopted after the key and integrity checks alone).
# It is NOT the default and it is recorded in the result, because it trusts a
# verdict written by another process.

#: Schema of a persisted V-proof cache entry. Bumped whenever the stored
#: document's meaning changes; an entry with any other schema is refused.
VPROOF_CACHE_SCHEMA = "gf.vproof_cache/1"

#: Default directory NAME under a run's artifact/output directory
#: (:func:`resolve_vproof_cache_root`).
VPROOF_CACHE_DIRNAME = "vproof_cache"

#: Environment variable that overrides the cache root, so a corner wrapper
#: script (``run_corner.sh``) can point every process of a row at one cache
#: without changing the driver's arguments.
VPROOF_CACHE_ENV = "GF_VPROOF_CACHE_DIR"

#: Every typed reason a lookup can decline to reuse a stored proof. All of
#: them fall back to the full proof; none of them raises, because a cache is
#: an optimization and refusing it must never be able to fail a certificate.
#: (An INPUT error — a bad root, an unwritable directory — is a different
#: thing and does raise, at construction time.)
VPROOF_CACHE_REASONS = frozenset({
    "vproof_cache_hit",                  # not a refusal; the recorded hit
    "vproof_cache_stored",               # store_reason: the entry was written
    "vproof_cache_read_only",            # store_reason: writes are disabled
    "vproof_cache_disabled",             # no cache configured
    "vproof_cache_in_process_hit",       # an in-process v_cache took priority
    "vproof_cache_bundle_witness",       # a design bundle witnesses V; nothing to save
    "vproof_cache_miss",                 # no entry at this key
    "vproof_cache_unreadable",           # I/O or JSON decode failure (truncated, garbage)
    "vproof_cache_schema_unknown",       # entry schema is not VPROOF_CACHE_SCHEMA
    "vproof_cache_payload_corrupt",      # the entry's own content digest does not reproduce
    "vproof_cache_key_mismatch",         # stored key/v_input_hash/n_vars/library/extra differ
    "vproof_cache_code_version_mismatch",  # the verifier's source changed since the entry was written
    "vproof_cache_gram_mismatch",        # the Gram bundle fingerprint does not reproduce
    "vproof_cache_not_proved",           # the entry does not record a PROVED V half
    "vproof_cache_revalidation_failed",  # the exact gate rejected the stored Grams
    "vproof_cache_revalidation_error",   # the exact gate raised on the stored Grams
    "vproof_cache_fingerprint_mismatch",  # the revalidated proof is not the one stored
    # store_reason (P3-03): a VALID entry for this key already existed when
    # this process finished recomputing the same proof, so the write was
    # skipped. This is the concurrent double-miss case — k workers launched at
    # once all miss, all recompute, and only the first one to finish writes.
    "vproof_cache_store_skipped_valid",
})

_VERIFIER_CODE_VERSION: Optional[str] = None


def verifier_code_version() -> str:
    """Content hash of the code that decides an exact proof.

    Covers the source of :mod:`~gf_framework.verification.exact_gate` (the
    exact rational gate itself) and of this module (the orchestration and the
    serialization format), so ANY edit to either invalidates every persisted
    entry. Deliberately coarse: it over-invalidates (a docstring change
    invalidates the cache) rather than risking a stale proof surviving a
    change to the verifier. Falls back to the schema string when the sources
    are not readable (a frozen or zipped install), which is safe because the
    default lookup revalidates the stored Grams with the CURRENT code anyway.
    """
    global _VERIFIER_CODE_VERSION
    if _VERIFIER_CODE_VERSION is None:
        h = hashlib.sha256()
        h.update(b"gf-exact-verifier-code/1\n")
        h.update(VPROOF_CACHE_SCHEMA.encode())
        h.update(b"\n")
        for module in ("exact_gate", "exact_phase"):
            try:
                src = (Path(__file__).resolve().parent / f"{module}.py").read_bytes()
            except OSError:
                src = b"<unreadable>"
            h.update(f"{module}:{len(src)}\n".encode())
            h.update(hashlib.sha256(src).hexdigest().encode())
            h.update(b"\n")
        _VERIFIER_CODE_VERSION = h.hexdigest()
    return _VERIFIER_CODE_VERSION


def _canonical_json(payload: Any) -> str:
    """One rendering of a JSON body, so a digest over it is reproducible."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def vproof_cache_key(
    V_gf,
    epsilon,
    n_vars: int,
    *,
    library: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Cache key of one V exact proof: the FULL exact input, hashed.

    Components, all of them:

    * :func:`v_input_hash` — the V terms as exact rationals (coefficient,
      exponents, term types), the homogeneity weights, and the exact ε. This
      is the same content hash the in-process cache refuses on;
    * ``n_vars`` — the octant count 2ⁿ the proof must cover;
    * ``library`` — a caller-supplied tag for the term library / design point
      the V came from. Two runs that disagree about it never share an entry
      even when the V terms coincide;
    * ``extra`` — anything else the caller must not share across, rendered
      canonically. The parity class / reduction mode of the run goes here;
    * :func:`verifier_code_version` — the source hash of the exact gate and
      of this module.

    A collision therefore requires a sha256 collision on all of it. A
    MISMATCH on any of it is a miss (a different key is a different file), and
    the stored document repeats every component so a file that somehow lands
    at the wrong path is still refused on read.
    """
    h = hashlib.sha256()
    h.update(b"gf-exact-vproof-cache/1\n")
    h.update(f"schema={VPROOF_CACHE_SCHEMA}\n".encode())
    h.update(f"v_input={v_input_hash(V_gf, epsilon)}\n".encode())
    h.update(f"n_vars={int(n_vars)}\n".encode())
    h.update(f"library={library}\n".encode())
    h.update(f"extra={_canonical_json(extra or {})}\n".encode())
    h.update(f"code={verifier_code_version()}\n".encode())
    return h.hexdigest()


def resolve_vproof_cache_root(output_dir=None) -> Path:
    """Where the cache lives: ``$GF_VPROOF_CACHE_DIR``, else under the run.

    The default is a ``vproof_cache/`` directory inside the run's artifact or
    output directory, so a cache belongs to the campaign block that produced
    it and is thrown away with it. The environment variable overrides that for
    a set of corner processes that share one row but write their records
    separately.
    """
    env = os.environ.get(VPROOF_CACHE_ENV, "").strip()
    if env:
        return Path(env)
    return Path(output_dir if output_dir is not None else ".") / VPROOF_CACHE_DIRNAME


def _vproof_grams_to_jsonable(entries: Sequence[dict]) -> Dict[str, Any]:
    """Serialize V Gram captures, tolerating a capture with no stored target.

    ``verify_gf_sos_full`` captures carry ``target``; the stored campaign
    bundles do not, and the exact gate does not need it (it rebuilds every
    octant target in ℚ from the V terms). Filling in an empty target keeps one
    serializer for both, and the entry hash is taken over the DECODED form so
    that a round trip reproduces it exactly.
    """
    prepared = []
    for entry in entries:
        doc = dict(entry)
        doc.setdefault("target", {})
        prepared.append(doc)
    return gram_bundle_to_jsonable(prepared)


def _vproof_grams_fingerprint(document: Dict[str, Any]) -> str:
    """Bundle fingerprint of a serialized V Gram set, as it reloads."""
    entries, _meta = gram_bundle_from_jsonable(document)
    return gram_bundle_fingerprint(entries)


def _vproof_exact_to_jsonable(exact: ExactCaptureResult) -> Dict[str, Any]:
    """The exact V verdict as JSON, for the ``revalidate=False`` path."""
    return {
        "ok": bool(exact.ok),
        "reason": exact.reason,
        "n_octants_expected": int(exact.n_octants_expected),
        "mu": [int(v) for v in exact.mu],
        "eps": str(exact.eps),
        "margin_policy": exact.margin_policy,
        "octants": [
            {
                "ok": bool(c.ok), "signs": [int(s) for s in c.signs],
                "reason": c.reason, "degree": int(c.degree),
                "n_basis": int(c.n_basis), "eta_hat": str(c.eta_hat),
                "eta_hat_float": float(c.eta_hat_float),
                "resid_l1_float": float(c.resid_l1_float),
                "charge_float": float(c.charge_float),
                "budget_float": float(c.budget_float),
                "margin_policy": c.margin_policy,
                "eps_float": c.eps_float,
                "eps_certified": c.eps_certified,
                "eps_certified_float": c.eps_certified_float,
                "min_eig_float": float(c.min_eig_float),
                "target_all_even": bool(c.target_all_even),
                "sym_group_dim": int(c.sym_group_dim),
                "n_blocks": int(c.n_blocks),
                "block_sizes": [int(b) for b in c.block_sizes],
            }
            for c in exact.octants
        ],
    }


def _vproof_exact_from_jsonable(doc: Dict[str, Any]) -> ExactCaptureResult:
    """Inverse of :func:`_vproof_exact_to_jsonable`."""
    from gf_framework.verification.exact_gate import ExactOctantCheck
    res = ExactCaptureResult(
        ok=bool(doc.get("ok")),
        reason=str(doc.get("reason") or ""),
        n_octants_expected=int(doc.get("n_octants_expected") or 0),
        mu=tuple(int(v) for v in doc.get("mu") or ()),
        eps=str(doc.get("eps") or ""),
        margin_policy=resolve_margin_policy(doc.get("margin_policy", MARGIN_POLICY_HALF)),
    )
    for c in doc.get("octants") or []:
        res.octants.append(ExactOctantCheck(
            ok=bool(c.get("ok")),
            signs=tuple(int(s) for s in c.get("signs") or ()),
            reason=str(c.get("reason") or ""),
            degree=int(c.get("degree") or 0),
            n_basis=int(c.get("n_basis") or 0),
            eta_hat=str(c.get("eta_hat") or ""),
            eta_hat_float=float(c.get("eta_hat_float") or 0.0),
            resid_l1_float=float(c.get("resid_l1_float") or 0.0),
            charge_float=float(c.get("charge_float") or 0.0),
            budget_float=float(c.get("budget_float") or 0.0),
            margin_policy=resolve_margin_policy(c.get("margin_policy", MARGIN_POLICY_HALF)),
            eps_float=float(c.get("eps_float") or 0.0),
            eps_certified=str(c.get("eps_certified") or ""),
            eps_certified_float=float(c.get("eps_certified_float") or 0.0),
            min_eig_float=float(c.get("min_eig_float") or 0.0),
            target_all_even=bool(c.get("target_all_even")),
            sym_group_dim=int(c.get("sym_group_dim") or 0),
            n_blocks=int(c.get("n_blocks") or 1),
            block_sizes=tuple(int(b) for b in c.get("block_sizes") or ()),
        ))
    return res


@dataclass
class VProofCacheEvent:
    """What the disk cache did for one :func:`run_exact_phase` call.

    Always produced, including when no cache was configured, so a record can
    never be silent about it. ``reason`` is the LOOKUP outcome and
    ``store_reason`` the write outcome; both are drawn from
    :data:`VPROOF_CACHE_REASONS`.
    """
    enabled: bool = False
    hit: bool = False
    stored: bool = False
    reason: str = "vproof_cache_disabled"
    store_reason: str = ""
    detail: str = ""
    root: str = ""
    path: str = ""
    key: str = ""
    revalidated: bool = False
    code_version: str = ""
    written_by_pid: Optional[int] = None
    written_at: str = ""
    entry_bytes: int = 0
    load_s: float = 0.0
    revalidate_s: float = 0.0
    store_s: float = 0.0

    def as_record(self) -> Dict[str, object]:
        """JSON-ready provenance row for the exact record."""
        return {
            "enabled": bool(self.enabled),
            "hit": bool(self.hit),
            "stored": bool(self.stored),
            "reason": self.reason,
            "store_reason": self.store_reason,
            "detail": self.detail,
            "root": self.root,
            "path": self.path,
            "key": self.key,
            "revalidated": bool(self.revalidated),
            "code_version": self.code_version,
            "written_by_pid": self.written_by_pid,
            "written_at": self.written_at,
            "entry_bytes": int(self.entry_bytes),
            "load_s": round(float(self.load_s), 3),
            "revalidate_s": round(float(self.revalidate_s), 3),
            "store_s": round(float(self.store_s), 3),
        }


_CACHE_IO_RETRY_DELAYS = (0.005, 0.01, 0.02, 0.04, 0.08)


def _transient_cache_access_error(exc):
    # Windows can briefly deny opens/replaces while another process publishes
    # an atomic replacement. Never retry missing files, invalid JSON, or broad
    # I/O failures. A persistent denial still fails after 155 ms of backoff.
    return os.name == "nt" and (
        getattr(exc, "winerror", None) in (5, 32, 33)
        or isinstance(exc, PermissionError) and exc.errno == errno.EACCES)


def _cache_io(operation):
    for attempt in range(len(_CACHE_IO_RETRY_DELAYS) + 1):
        try:
            return operation()
        except OSError as exc:
            if (attempt == len(_CACHE_IO_RETRY_DELAYS)
                    or not _transient_cache_access_error(exc)):
                raise
            time.sleep(_CACHE_IO_RETRY_DELAYS[attempt])


def _cache_io_detail(exc):
    return (f"{type(exc).__name__}: errno={getattr(exc, 'errno', None)}, "
            f"winerror={getattr(exc, 'winerror', None)}: {exc}")[:300]


class DiskVProofCache:
    """A :class:`VProofCache` persisted on disk, shared across processes.

    One file per key, named by the key, written by atomic rename
    (``os.replace`` over a temporary file in the same directory), so a corner
    process can never read a half-written entry and two corner processes
    writing the same key concurrently both succeed with the last rename
    winning. Every entry carries its own content digest, so a file damaged
    after the rename (a truncated copy, a partial restore) is refused rather
    than read.

    ``library`` and ``extra`` enter the KEY (:func:`vproof_cache_key`): give
    them the run's design point / term library and its parity mode, and a
    proof can never migrate between two runs that disagree about either.

    ``revalidate`` (default ``True``) re-runs the exact rational gate on the
    stored Gram captures. Leave it on: it is what makes a persisted proof no
    more trusted than a design Gram, and it costs a fraction of a percent of
    the float re-solve it replaces.

    ``read_only`` looks up but never writes (a shared, curated cache).
    """

    def __init__(
        self,
        root,
        *,
        library: str = "",
        extra: Optional[Dict[str, Any]] = None,
        revalidate: bool = True,
        read_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.library = str(library)
        self.extra = dict(extra or {})
        self.revalidate = bool(revalidate)
        self.read_only = bool(read_only)

    # -- paths -------------------------------------------------------------

    def key_for(self, V_gf, epsilon, n_vars: int) -> str:
        return vproof_cache_key(V_gf, epsilon, n_vars,
                                library=self.library, extra=self.extra)

    def path_for(self, key: str) -> Path:
        return self.root / f"v_{key}.json"

    @property
    def margin_policy(self):
        # The unadorned key is historical half; achieved keys say so explicitly.
        return resolve_margin_policy(self.extra.get("margin_policy", MARGIN_POLICY_HALF))

    # -- read --------------------------------------------------------------

    def load(
        self,
        V_gf,
        epsilon,
        n_vars: int,
        *,
        v_terms=None,
        input_hash: Optional[str] = None,
    ) -> Tuple[Optional[VProofCache], VProofCacheEvent]:
        """Look up one V proof. Returns ``(cache_or_None, event)``.

        Never raises on a bad entry: every failure mode is a typed reason in
        the event and a fall-back to the full proof. ``v_terms`` is the exact
        V (:func:`~gf_framework.verification.exact_gate.exact_terms_from_gf`)
        and is required when ``revalidate`` is on.
        """
        key = self.key_for(V_gf, epsilon, n_vars)
        path = self.path_for(key)
        event = VProofCacheEvent(
            enabled=True, reason="vproof_cache_miss", root=str(self.root),
            path=str(path), key=key, code_version=verifier_code_version(),
            revalidated=False)
        t0 = time.time()
        try:
            raw = _cache_io(path.read_bytes)
        except FileNotFoundError:
            event.load_s = time.time() - t0
            return None, event
        except OSError as exc:
            event.reason, event.detail = "vproof_cache_unreadable", _cache_io_detail(exc)
            event.load_s = time.time() - t0
            return None, event
        event.entry_bytes = len(raw)
        try:
            document = json.loads(raw.decode("utf-8"))
            if not isinstance(document, dict):
                raise ValueError("entry is not a JSON object")
        except (UnicodeDecodeError, ValueError) as exc:
            event.reason = "vproof_cache_unreadable"
            event.detail = f"{type(exc).__name__}: {exc}"[:300]
            event.load_s = time.time() - t0
            return None, event
        event.written_by_pid = document.get("written_by_pid")
        event.written_at = str(document.get("written_at") or "")
        cache, reason, detail = self._accept(
            document, V_gf, epsilon, n_vars, key,
            v_terms=v_terms, input_hash=input_hash, event=event)
        event.reason, event.detail = reason, detail
        event.hit = cache is not None
        event.load_s = time.time() - t0
        return cache, event

    def _accept(self, document, V_gf, epsilon, n_vars, key, *,
                v_terms, input_hash, event) -> Tuple[Optional[VProofCache], str, str]:
        """The refusal ladder. One typed reason per way of not being reusable."""
        if document.get("schema") != VPROOF_CACHE_SCHEMA:
            return None, "vproof_cache_schema_unknown", (
                f"entry schema {document.get('schema')!r}, expected "
                f"{VPROOF_CACHE_SCHEMA!r}")

        stored_digest = document.get("payload_sha256")
        body = {k: v for k, v in document.items() if k != "payload_sha256"}
        actual_digest = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
        if stored_digest != actual_digest:
            return None, "vproof_cache_payload_corrupt", (
                f"content digest {actual_digest[:16]} does not reproduce the "
                f"stored {str(stored_digest)[:16]}")

        if document.get("code_version") != verifier_code_version():
            return None, "vproof_cache_code_version_mismatch", (
                f"entry written by verifier {str(document.get('code_version'))[:16]}, "
                f"this one is {verifier_code_version()[:16]}")

        expected_input = input_hash if input_hash is not None else v_input_hash(
            V_gf, epsilon)
        mismatches = []
        if document.get("key") != key:
            mismatches.append(f"key {str(document.get('key'))[:16]} != {key[:16]}")
        if document.get("v_input_hash") != expected_input:
            mismatches.append(
                f"v_input_hash {str(document.get('v_input_hash'))[:16]} != "
                f"{expected_input[:16]}")
        if int(document.get("n_vars") or -1) != int(n_vars):
            mismatches.append(f"n_vars {document.get('n_vars')} != {n_vars}")
        if str(document.get("library") or "") != self.library:
            mismatches.append(
                f"library {document.get('library')!r} != {self.library!r}")
        if _canonical_json(document.get("extra") or {}) != _canonical_json(self.extra):
            mismatches.append("extra differs")
        if mismatches:
            return None, "vproof_cache_key_mismatch", "; ".join(mismatches)[:300]

        if not document.get("resolve_feasible") or not (
                (document.get("exact") or {}).get("ok")):
            return None, "vproof_cache_not_proved", (
                "the entry does not record a proved V half "
                f"(resolve_feasible={document.get('resolve_feasible')}, "
                f"exact.ok={(document.get('exact') or {}).get('ok')})")

        try:
            grams, _meta = gram_bundle_from_jsonable(document.get("grams") or {})
        except Exception as exc:  # noqa: BLE001 - a malformed entry, never fatal
            return None, "vproof_cache_unreadable", f"{type(exc).__name__}: {exc}"[:300]
        if not grams:
            return None, "vproof_cache_not_proved", "the entry carries no Gram captures"
        if gram_bundle_fingerprint(grams) != document.get("gram_fingerprint"):
            return None, "vproof_cache_gram_mismatch", (
                "the reloaded Gram captures do not reproduce the stored "
                "bundle fingerprint")

        stored_fp = str(document.get("fingerprint") or "")
        if self.revalidate:
            if v_terms is None:
                return None, "vproof_cache_revalidation_error", (
                    "revalidate=True but the caller supplied no exact V terms")
            t0 = time.time()
            try:
                exact = exact_verify_captures(v_terms, grams, n_vars,
                                               margin_policy=self.margin_policy)
            except Exception as exc:  # noqa: BLE001 - typed, never fatal
                event.revalidate_s = time.time() - t0
                return None, "vproof_cache_revalidation_error", (
                    f"{type(exc).__name__}: {exc}"[:300])
            event.revalidate_s = time.time() - t0
            event.revalidated = True
            if not exact.ok:
                return None, "vproof_cache_revalidation_failed", (
                    f"the exact gate rejected the stored Grams: "
                    f"{exact.reason or 'failed'}")
            fingerprint = v_proof_fingerprint(exact, grams)
            if stored_fp and fingerprint != stored_fp:
                return None, "vproof_cache_fingerprint_mismatch", (
                    f"revalidated proof {fingerprint[:16]} is not the stored "
                    f"{stored_fp[:16]}")
        else:
            try:
                exact = _vproof_exact_from_jsonable(document.get("exact") or {})
            except (TypeError, ValueError, KeyError, ExactGateError) as exc:
                return None, "vproof_cache_unreadable", (
                    f"invalid stored exact verdict: {type(exc).__name__}: {exc}"[:300])
            if not exact.ok:
                return None, "vproof_cache_not_proved", (
                    "the stored verdict is not a pass")
            fingerprint = stored_fp

        cache = VProofCache(
            input_hash=expected_input,
            fingerprint=fingerprint,
            resolve_feasible=bool(document.get("resolve_feasible")),
            exact=exact,
            grams=grams,
        )
        try:
            cache = _checked_v_cache(cache, V_gf, epsilon, n_vars, self.margin_policy)
        except ExactPhaseError as exc:
            return None, "vproof_cache_not_proved", str(exc)[:300]
        return cache, "vproof_cache_hit", ""

    # -- write -------------------------------------------------------------

    def _valid_entry_exists(self, path: Path, key: str) -> bool:
        """Is there already a self-consistent, current entry at ``key``?

        The structural half of :meth:`_accept`, with no revalidation: schema,
        the entry's own content digest (which covers the Grams, so integrity
        is established without re-deriving the bundle fingerprint), the
        verifier code version, the key it claims, and a proved V half. Used
        only to decide whether a concurrent writer already stored this proof
        (P3-03); a False here just means "write it", never "the entry is bad",
        so every failure mode collapses to False and nothing raises.
        """
        try:
            document = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return False
        if not isinstance(document, dict):
            return False
        if document.get("schema") != VPROOF_CACHE_SCHEMA:
            return False
        body = {k: v for k, v in document.items() if k != "payload_sha256"}
        try:
            digest = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
        except (TypeError, ValueError):
            return False
        if digest != document.get("payload_sha256"):
            return False
        if document.get("code_version") != verifier_code_version():
            return False
        if str(document.get("key") or "") != key:
            return False
        if str(document.get("library") or "") != self.library:
            return False
        if _canonical_json(document.get("extra") or {}) != _canonical_json(self.extra):
            return False
        return bool(document.get("resolve_feasible")
                    and (document.get("exact") or {}).get("ok")
                    and document.get("grams"))

    def store(
        self,
        V_gf,
        epsilon,
        n_vars: int,
        result: "ExactPhaseResult",
        *,
        key: Optional[str] = None,
        into: Optional[VProofCacheEvent] = None,
        skip_if_valid: bool = True,
    ) -> VProofCacheEvent:
        """Persist the V half of a completed phase. Atomic, never partial.

        Refuses to write anything but a PROVED V half: the point of the entry
        is to save a re-solve, and an entry that records a failure would only
        be re-read and refused. ``into`` records the outcome on an existing
        lookup event (``store_reason`` / ``stored``) instead of a fresh one,
        so one record carries both halves.

        Concurrent double miss (P3-03). When k corner processes are launched
        at once they ALL miss and all recompute the same V proof. Before
        writing, this checks whether a VALID entry has appeared at the key in
        the meantime and, if so, REFUSES to overwrite it
        (``vproof_cache_store_skipped_valid``): what is on disk is already a
        proof of the same object. The check is a re-read plus the entry's own
        integrity tests, never a revalidation, so it costs a file read and not
        a proof. It is an optimisation, not a lock — two writers that pass it
        at the same instant both write, which stays safe for P3-05's reason:
        the write is ``fsync`` + ``os.replace``, so a reader sees a complete
        entry or none, and the two payloads are the same proof.
        ``skip_if_valid=False`` restores the unconditional overwrite, which is
        what a test that WANTS write traffic asks for.
        """
        key = key or self.key_for(V_gf, epsilon, n_vars)
        path = self.path_for(key)
        event = into if into is not None else VProofCacheEvent(
            enabled=True, reason="vproof_cache_miss", root=str(self.root),
            path=str(path), key=key, code_version=verifier_code_version())
        if self.read_only:
            event.store_reason = "vproof_cache_read_only"
            return event
        if not (result.v_resolve_feasible and result.v_exact.ok and result.v_grams):
            event.store_reason = "vproof_cache_not_proved"
            event.detail = (
                f"not storing: resolve_feasible={result.v_resolve_feasible}, "
                f"v_exact.ok={result.v_exact.ok}, "
                f"n_grams={len(result.v_grams or [])}")
            return event
        try:
            proof = _checked_v_cache(build_v_cache(V_gf, epsilon, result),
                                     V_gf, epsilon, n_vars, self.margin_policy)
        except ExactPhaseError as exc:
            event.store_reason = "vproof_cache_not_proved"
            event.detail = str(exc)[:300]
            return event
        if skip_if_valid and self._valid_entry_exists(path, key):
            # P3-03: another worker that missed the same key at the same time
            # got there first with the same proof. Never overwrite it.
            event.store_reason = "vproof_cache_store_skipped_valid"
            event.detail = (
                "a valid entry for this key appeared while this process was "
                "recomputing the same V proof; not overwriting it")
            return event
        t0 = time.time()
        eps_exact = _exact_epsilon(epsilon)
        body: Dict[str, Any] = {
            "schema": VPROOF_CACHE_SCHEMA,
            "key": key,
            "v_input_hash": proof.input_hash,
            "n_vars": int(n_vars),
            "library": self.library,
            "extra": self.extra,
            "code_version": verifier_code_version(),
            "epsilon": f"{eps_exact.numerator}/{eps_exact.denominator}",
            "epsilon_float": float(epsilon),
            "fingerprint": proof.fingerprint,
            "resolve_feasible": bool(result.v_resolve_feasible),
            "witness_source": result.witness_source,
            "witness_solver": result.witness_solver,
            "worst_charge_ratio": proof.exact.worst_charge_ratio,
            "worst_charge_over_eps": proof.exact.worst_charge_over_eps,
            "n_grams": len(result.v_grams),
            "grams": _vproof_grams_to_jsonable(result.v_grams),
            "exact": _vproof_exact_to_jsonable(proof.exact),
            "written_by_pid": int(os.getpid()),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        body["gram_fingerprint"] = _vproof_grams_fingerprint(body["grams"])
        body["payload_sha256"] = hashlib.sha256(
            _canonical_json(body).encode()).hexdigest()
        payload = _canonical_json(body).encode()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            _cache_io(lambda: os.replace(tmp, path))  # atomic within the directory
        except OSError as exc:
            event.store_reason = "vproof_cache_unreadable"
            event.detail = f"write failed: {_cache_io_detail(exc)}"[:300]
            event.store_s = time.time() - t0
            try:
                tmp.unlink()
            except (OSError, NameError, UnboundLocalError):
                pass
            return event
        event.stored = True
        event.store_reason = "vproof_cache_stored"
        event.entry_bytes = len(payload)
        event.written_by_pid = body["written_by_pid"]
        event.written_at = body["written_at"]
        event.store_s = time.time() - t0
        return event


def as_disk_vproof_cache(spec) -> Optional[DiskVProofCache]:
    """Normalize the ``v_proof_cache`` argument of :func:`run_exact_phase`.

    ``None``/``False`` disables it, ``True`` resolves the default root
    (:func:`resolve_vproof_cache_root`), a path or string names a root, and a
    :class:`DiskVProofCache` is used as it stands.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return DiskVProofCache(resolve_vproof_cache_root())
    if isinstance(spec, DiskVProofCache):
        return spec
    if isinstance(spec, (str, Path)):
        return DiskVProofCache(Path(spec))
    raise TypeError(
        "v_proof_cache must be None, True, a path, or a DiskVProofCache, "
        f"got {type(spec).__name__}")


def vproof_cache_for_margin_policy(
        cache: Optional["DiskVProofCache"],
        margin_policy: str) -> Optional["DiskVProofCache"]:
    """The disk cache a run under ``margin_policy`` actually keys against.

    A V proof accepted under a LOOSER acceptance threshold must never be
    reusable by a run under a stricter one, so the policy enters the cache
    KEY (WP P2-13), through :attr:`DiskVProofCache.extra`. The rule, and the
    reason it is asymmetric:

    * ``"half"`` adds NO ``extra`` component, so every cache entry written
      before the policy existed keeps its key byte for byte;
    * every other policy adds ``{"margin_policy": <policy>}``, so its entries
      live under a different key and the two never share one.

    The input cache is never mutated: a shallow copy carries the amended
    ``extra``. ``None`` in, ``None`` out. This is the function
    :func:`run_exact_phase` applies internally, exported so a caller (or a
    test) can look up the same entry the phase will, rather than
    re-deriving the rule.
    """
    if cache is None:
        return None
    if resolve_margin_policy(margin_policy) == MARGIN_POLICY_HALF:
        if "margin_policy" not in cache.extra:
            return cache
        amended = copy.copy(cache)
        amended.extra = dict(cache.extra)
        amended.extra.pop("margin_policy")
        return amended
    amended = copy.copy(cache)
    amended.extra = dict(cache.extra)
    amended.extra["margin_policy"] = resolve_margin_policy(margin_policy)
    return amended


# ---------------------------------------------------------------------------
# Process parallelism over octants — campaign item P3-03
# ---------------------------------------------------------------------------

#: The multiprocessing start method for octant workers. **spawn**, not fork.
#:
#: A worker inherits nothing: it is a fresh interpreter that imports
#: ``exact_gate`` and unpickles its arguments. ``fork`` would copy the parent's
#: address space, and by the time the exact phase runs that space contains a
#: loaded MOSEK library (the witness re-solve tier, and the design stage before
#: it) and an initialized OpenBLAS thread pool. Neither survives a fork
#: intact: MOSEK holds a licence handle and its own threads, OpenBLAS's worker
#: threads do not exist in the child, and the documented failure is a hang or a
#: silent wrong answer, not an exception. ``forkserver`` is better but still
#: forks from a server process that has already imported this module (hence
#: numpy), and it is POSIX-only. ``spawn`` is the only start method that is
#: correct on every platform and after any solver has run in the parent; its
#: cost is one interpreter start per worker, which the process-level pool below
#: pays once and amortizes over every octant of every system in the run.
OCTANT_START_METHOD = "spawn"

#: The ONLY function that crosses the process boundary, and the whole of what
#: a worker is trusted to do: rebuild one octant's exact target in ℚ from the
#: terms it is handed and run the Proposition-7 chain on one Gram. It is pure,
#: it holds no state, and it pickles by reference (module-level in
#: ``exact_gate``), so nothing about the parent's solver, cache or bundle
#: reaches a worker. Its result is an ``ExactOctantCheck``, which the parent
#: assembles — a worker never decides a verdict, an ordering or a reason.
OCTANT_WORKER_FN = validate_one_capture

#: ``jobs`` default. 1 means "no pool, no worker process, the historical serial
#: loop" — stored baselines cannot move unless a caller asks for more (D1's
#: rule that a default flip needs the P2-06 gate).
DEFAULT_JOBS = 1

#: Typed :class:`OctantWorkerError` reasons. A worker failure is never
#: swallowed and never silently retried in-process: it fails the witness
#: attempt with one of these, which ``run_exact_phase`` records as the
#: attempt's ``shortfall`` (and re-raises when no witness is left).
OCTANT_WORKER_REASONS = frozenset({
    # a worker raised while validating one octant (the octant is named)
    "octant_worker_exception",
    # a worker process died — OOM kill, segfault, an exec that never started
    "octant_worker_died",
    # a worker did not return within ``timeout_s``
    "octant_worker_timeout",
    # the pool itself could not be created or could not accept work
    "octant_pool_unavailable",
})


class OctantWorkerError(ExactPhaseError):
    """A parallel octant validation did not produce a result.

    This is a HARNESS failure, never a verdict about a certificate. It carries
    a machine-readable ``reason`` from :data:`OCTANT_WORKER_REASONS`, the
    octant ``index`` when one is identifiable, and the pool geometry, so a
    record states that the row failed because a worker died rather than
    because the mathematics failed. ``run_exact_phase`` types the witness
    attempt with ``reason`` instead of the generic ``unexpected_error``.
    """

    def __init__(self, reason: str, detail: str, *, jobs: int = 0,
                 start_method: str = "", index: Optional[int] = None) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.jobs = int(jobs)
        self.start_method = str(start_method)
        self.index = index


def resolve_jobs(jobs) -> int:
    """Normalize a ``jobs`` request to a positive worker count.

    ``None``, ``0`` and negatives all mean "serial" (1). A value above the
    core count is NOT clamped: oversubscription is the caller's decision and
    the measured effect belongs in the record, not in a silent correction.
    """
    if jobs is None:
        return DEFAULT_JOBS
    if isinstance(jobs, bool):  # True/False are ints in Python; refuse them
        raise ExactPhaseError(f"jobs must be an integer, got {jobs!r}")
    try:
        n = int(jobs)
    except (TypeError, ValueError) as exc:
        raise ExactPhaseError(f"jobs must be an integer, got {jobs!r}") from exc
    return max(1, n)


class OctantPool:
    """A reusable :mod:`concurrent.futures` process pool over octant jobs.

    One instance per (jobs, start method) per process, created lazily on the
    first parallel batch and reused for every later one, so the spawn cost is
    paid once and not per form or per system.

    :meth:`map` is the mapper contract of
    ``exact_gate.exact_verify_captures``: it yields ``(index, result)`` pairs
    in COMPLETION order, and the gate assembles them by index. Ordering is
    therefore not a property of this class at all, which is what makes the
    determinism testable without a pool (see the shuffling mapper in
    ``tests/test_octant_parallel.py``).
    """

    def __init__(self, jobs: int, start_method: str = OCTANT_START_METHOD,
                 timeout_s: Optional[float] = None) -> None:
        self.jobs = resolve_jobs(jobs)
        self.start_method = str(start_method)
        self.timeout_s = timeout_s
        self.n_tasks = 0
        self.n_batches = 0
        self.broken = False
        self._ex: Optional[concurrent.futures.Executor] = None

    # -- lifecycle ---------------------------------------------------------
    def _executor(self) -> concurrent.futures.Executor:
        if self._ex is None:
            try:
                ctx = multiprocessing.get_context(self.start_method)
                self._ex = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.jobs, mp_context=ctx)
            except Exception as exc:  # noqa: BLE001 - typed and re-raised
                self.broken = True
                raise OctantWorkerError(
                    "octant_pool_unavailable",
                    f"{type(exc).__name__}: {exc}",
                    jobs=self.jobs, start_method=self.start_method) from exc
        return self._ex

    def shutdown(self) -> None:
        ex, self._ex = self._ex, None
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass

    # -- the mapper --------------------------------------------------------
    def map(self, fn, items):
        """Yield ``(index, fn(*item))`` for every item, in completion order.

        Raises :class:`OctantWorkerError` — typed, never swallowed — as soon
        as any worker raises, dies or times out, after cancelling the work
        that has not started. A dead worker cannot hang the caller: the
        executor turns it into ``BrokenProcessPool`` on every pending future,
        which is what ``octant_worker_died`` reports.
        """
        items = list(items)
        if not items:
            return
        ex = self._executor()
        self.n_batches += 1
        self.n_tasks += len(items)
        futures: Dict[concurrent.futures.Future, int] = {}
        try:
            for i, item in enumerate(items):
                futures[ex.submit(fn, *item)] = i
            for fut in concurrent.futures.as_completed(
                    futures, timeout=self.timeout_s):
                yield futures[fut], fut.result()
        except Exception as exc:  # noqa: BLE001 - typed below, always raised
            index = next((i for f, i in futures.items()
                          if f.done() and not f.cancelled()
                          and f.exception() is not None), None)
            for fut in futures:
                fut.cancel()
            if isinstance(exc, OctantWorkerError):
                raise
            if isinstance(exc, BrokenProcessPool):
                self.broken = True
                self.shutdown()
                raise OctantWorkerError(
                    "octant_worker_died",
                    f"a worker process died while validating "
                    f"{len(items)} octant(s): {type(exc).__name__}: {exc}",
                    jobs=self.jobs, start_method=self.start_method,
                    index=index) from exc
            if isinstance(exc, concurrent.futures.TimeoutError):
                self.broken = True
                self.shutdown()
                raise OctantWorkerError(
                    "octant_worker_timeout",
                    f"no result within {self.timeout_s} s for "
                    f"{len(items)} octant(s)",
                    jobs=self.jobs, start_method=self.start_method) from exc
            raise OctantWorkerError(
                "octant_worker_exception",
                f"octant {index}: {type(exc).__name__}: {exc}"[:400],
                jobs=self.jobs, start_method=self.start_method,
                index=index) from exc


#: The process-level pool. Reused across systems, corners and forms so the
#: spawn cost is amortized; rebuilt when the geometry changes or a worker died.
_OCTANT_POOL: Optional[OctantPool] = None
_OCTANT_POOL_KEY: Optional[Tuple[int, str]] = None


def octant_pool(jobs, start_method: str = OCTANT_START_METHOD
                ) -> Optional[OctantPool]:
    """The shared :class:`OctantPool` for ``jobs`` workers, or ``None``.

    ``None`` for ``jobs <= 1``: the serial path must not create, touch or
    depend on a pool at all.
    """
    global _OCTANT_POOL, _OCTANT_POOL_KEY
    n = resolve_jobs(jobs)
    if n <= 1:
        return None
    key = (n, str(start_method))
    if (_OCTANT_POOL is None or _OCTANT_POOL_KEY != key
            or _OCTANT_POOL.broken):
        shutdown_octant_pool()
        _OCTANT_POOL = OctantPool(n, start_method)
        _OCTANT_POOL_KEY = key
    return _OCTANT_POOL


def shutdown_octant_pool() -> None:
    """Tear the shared pool down (registered with :mod:`atexit`)."""
    global _OCTANT_POOL, _OCTANT_POOL_KEY
    if _OCTANT_POOL is not None:
        _OCTANT_POOL.shutdown()
    _OCTANT_POOL = None
    _OCTANT_POOL_KEY = None


atexit.register(shutdown_octant_pool)


def octant_mapper(jobs, start_method: str = OCTANT_START_METHOD):
    """``(mapper, pool)`` for ``jobs``; ``(None, None)`` when serial.

    The mapper is what ``exact_gate.exact_verify_captures`` consumes. The
    function it will be handed is ``exact_gate.validate_one_capture``, a
    module-level pure function, so it pickles by reference and no closure
    crosses the process boundary.
    """
    pool = octant_pool(jobs, start_method)
    if pool is None:
        return None, None
    return pool.map, pool


def parallel_record(jobs, start_method: str, pool: Optional[OctantPool],
                    *, n_batches: int = 0, n_tasks: int = 0
                    ) -> Dict[str, object]:
    """JSON-ready provenance of HOW the exact arithmetic was scheduled.

    Recorded on every result, including the serial default, so no record is
    silent about it: ``{"jobs": 1, "enabled": false, "start_method": ""}`` is
    the statement that the historical serial loop ran.
    """
    n = resolve_jobs(jobs)
    enabled = pool is not None
    return {
        "scope": "octant",
        "jobs": n,
        "enabled": bool(enabled),
        "start_method": str(start_method) if enabled else "",
        "max_workers": int(pool.jobs) if enabled else 1,
        "n_octant_batches": int(n_batches),
        "n_octants": int(n_tasks),
        "cpu_count": os.cpu_count(),
    }


@dataclass
class WitnessAttempt:
    """One tried witness source, adopted or not (P2-05).

    Recorded for EVERY attempt, including the ones that were left behind, so
    an artifact states what was tried and why the adopted witness was
    adopted. ``shortfall`` is the typed reason the attempt was not good
    enough: ``""`` for the adopted-on-the-spot case, a bundle refusal reason
    (``gram_bundle_margin_missing``, ...), a
    :mod:`~gf_framework.verification.failure_typing` class for a raised
    exception, ``v_exact_...`` / ``w<i>_exact_...`` for an exact-gate
    rejection, or ``charge_over_eps_<value>_above_<trigger>`` for the headroom
    trigger (WP P2-14 renamed it from ``charge_ratio_`` when the denominator
    moved from the policy's budget to ε).
    """
    index: int
    source: str
    solver: str
    ok: bool
    #: The ENCODING that actually ran, from ``JointSOSResult.solver_used``
    #: (P2-10): ``"MOSEK_DIRECT"`` for the Task-API branch against
    #: ``"MOSEK"`` for the CVXPY interface, both of which ``solver`` above
    #: records identically as the REQUESTED ``"MOSEK"``. Empty when nothing
    #: was solved for this attempt (a design-bundle refusal, a raised
    #: re-solve). Several solves feed one attempt (V and one W per endpoint);
    #: when they disagree the distinct names are joined with ``+``.
    solver_used: str = ""
    worst_charge_ratio: Optional[float] = None
    #: The same headroom measured against ε instead of against the policy's
    #: budget (WP P2-14). Policy-free, and the number the headroom trigger
    #: actually compares — the field name states its denominator.
    worst_charge_over_eps: Optional[float] = None
    shortfall: str = ""
    detail: str = ""
    adopted: bool = False
    time_s: float = 0.0

    def as_record(self) -> Dict[str, object]:
        """JSON-ready row for the exact record."""
        return {
            "index": int(self.index),
            "source": self.source,
            "solver": self.solver,
            "solver_used": self.solver_used,
            "ok": bool(self.ok),
            "worst_charge_ratio": (None if self.worst_charge_ratio is None
                                   else float(self.worst_charge_ratio)),
            "worst_charge_over_eps": (None if self.worst_charge_over_eps is None
                                      else float(self.worst_charge_over_eps)),
            "shortfall": self.shortfall,
            "detail": self.detail,
            "adopted": bool(self.adopted),
            "time_s": round(float(self.time_s), 3),
        }


@dataclass
class EndpointExactResult:
    """Exact-phase outcome for one dynamics (nominal or one inclusion vertex)."""
    resolve_feasible: bool
    exact: ExactCaptureResult
    lie_crosscheck: Dict[str, object]
    # Raw per-octant Gram captures ({signs, octant_index, eps, mu, y_degree,
    # target, basis, gram}) from the W re-solve — retained so drivers can
    # persist them (--dump-grams / standalone-verifier bundles).
    grams: Optional[List[dict]] = None
    # --- R3 provenance ------------------------------------------------------
    # Which witness proved this endpoint: WITNESS_RESOLVE (a float re-solve
    # proposed the Gram) or WITNESS_DESIGN_BUNDLE (the design Gram was the
    # witness and no float solve ran). ``witnesses`` names the concrete entry
    # behind every octant, including the ones expanded from a parity partner.
    witness_source: str = WITNESS_RESOLVE
    witnesses: List[dict] = field(default_factory=list)
    # Which solver proposed this endpoint's Gram: the re-solve's solver, or
    # DESIGN_BUNDLE_SOLVER when the design bundle was the witness (P2-05).
    witness_solver: str = ""
    # The ENCODING that actually ran for this endpoint's W re-solve
    # (JointSOSResult.solver_used, P2-10/P3-05): "MOSEK_DIRECT" for the Task
    # API against "MOSEK" for the CVXPY interface, which witness_solver above
    # cannot distinguish because it records the REQUEST.
    witness_solver_used: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.resolve_feasible and self.exact.ok)


@dataclass
class ExactPhaseResult:
    """Aggregate Proposition-7 verdict for V and all supplied dynamics."""
    v_resolve_feasible: bool
    v_exact: ExactCaptureResult
    endpoints: List[EndpointExactResult] = field(default_factory=list)
    time_s: float = 0.0
    # Raw per-octant Gram captures from the V re-solve (see EndpointExactResult).
    v_grams: Optional[List[dict]] = None
    # --- R2 provenance ------------------------------------------------------
    # v_reused is True when the V half came from a VProofCache instead of a
    # fresh re-solve + exact proof. The hashes are recorded either way.
    v_reused: bool = False
    v_input_hash: str = ""
    v_fingerprint: str = ""
    # --- R3 provenance ------------------------------------------------------
    # witness_source is WITNESS_DESIGN_BUNDLE when the DESIGN-stage Gram bundle
    # witnessed this phase (no float solve ran on the certifying route) and
    # WITNESS_RESOLVE for the historical independent re-solve. On the bundle
    # path v_resolve_feasible is True by construction: a bundle exists only on
    # a FEASIBLE design solve, and that solve is what the flag reports.
    witness_source: str = WITNESS_RESOLVE
    gram_bundle_fingerprint: str = ""
    gram_bundle_parity_expanded: bool = False
    v_witnesses: List[dict] = field(default_factory=list)
    # --- P2-05 provenance: WHICH witness was adopted, and what else was tried
    # witness_solver is the solver that proposed the adopted Gram
    # (DESIGN_BUNDLE_SOLVER on the bundle path, else the re-solve's solver).
    # witness_charge_ratio is the adopted phase's worst exact charge/budget
    # ratio — the headroom the recorded proof actually has, not a solver
    # statistic. witness_attempts lists every source tried, in order.
    witness_solver: str = ""
    # The ENCODING behind the adopted witness (P2-10's follow-up, landed by
    # P3-05). ``witness_solver`` names the solver that was REQUESTED, and
    # since P2-10 a request for MOSEK can be served by either the CVXPY
    # interface ("MOSEK") or the Task-API branch ("MOSEK_DIRECT"); only
    # ``JointSOSResult.solver_used`` distinguishes them, and this is where it
    # survives the phase boundary into the record. Distinct names across the
    # V and W solves of one attempt are joined with ``+``.
    witness_solver_used: str = ""
    witness_charge_ratio: Optional[float] = None
    witness_attempts: List[dict] = field(default_factory=list)
    # --- P3-05 provenance: what the PERSISTED V-proof cache did -------------
    # Always populated, including "no cache was configured", so no record can
    # be silent about a reused proof. See VProofCacheEvent.
    v_proof_cache: Dict[str, object] = field(default_factory=dict)
    # --- P3-03 provenance: HOW the exact arithmetic was scheduled -----------
    # Always populated, including the serial default ({"jobs": 1, "enabled":
    # false, "start_method": ""}). Scheduling cannot change a verdict — the
    # gate assembles octants by index and truncates at the first failure
    # regardless of completion order — so this is a cost record, not a
    # soundness one; it is stored because a wall-clock number without it is
    # not reproducible. See parallel_record().
    parallel: Dict[str, object] = field(default_factory=dict)
    # --- P2-13 provenance: WHICH acceptance threshold this phase ran under --
    # ``"half"`` (default, historical) accepted at charge ≤ ε/2; ``"achieved"``
    # at charge < ε. Recorded because the two prove DIFFERENT margins from the
    # same Gram, and ``eps_certified_float`` below is the margin proved. A
    # record without this key was written before P2-13 and is "half".
    margin_policy: str = MARGIN_POLICY_ACHIEVED

    def witness_record(self) -> Dict[str, object]:
        """JSON-ready witness provenance for an artifact (R3).

        States, in one dict, which object proved this phase: the witness
        source, the fingerprint of the design bundle when one was used,
        whether any octant was covered by expanding a parity partner, and the
        per-octant entry names/hashes for V and every endpoint.

        ``v_reused`` is repeated here because it, not ``source``, is what says
        where the V half came from when R2's dedup and R3's bundle are used
        together: the V witnesses list is then empty and the V proof is the
        one corner 0 recorded under ``v_fingerprint``.

        ``source`` keeps P2-03's name and semantics (readers roll it up as
        ``witness.source``); ``witness_source`` repeats it under the explicit
        P2-05 name so one grep over a record finds the whole adopted-witness
        block (``witness_solver``, ``witness_source``,
        ``witness_charge_ratio``, ``witness_attempts``).
        """
        return {
            "source": self.witness_source,
            "witness_source": self.witness_source,
            "witness_solver": self.witness_solver,
            "witness_solver_used": self.witness_solver_used,
            "witness_charge_ratio": self.witness_charge_ratio,
            "witness_attempts": list(self.witness_attempts),
            "n_witness_attempts": len(self.witness_attempts),
            "gram_bundle_fingerprint": self.gram_bundle_fingerprint,
            "parity_expanded": bool(self.gram_bundle_parity_expanded),
            "v_reused": bool(self.v_reused),
            # P3-05: hit/miss, the typed refusal reason, and the cache path.
            # A sibling field, so a reader that ignores unknown keys is
            # unaffected and a record written before P3-05 simply has none.
            "v_proof_cache": dict(self.v_proof_cache),
            # P3-03: jobs and the start method behind the per-octant schedule.
            # Same sibling-key discipline as v_proof_cache above.
            "parallel": dict(self.parallel),
            # P2-13: the acceptance threshold in force and the margin PROVED.
            # ``eps_certified`` is min over V and every endpoint of ε − charge;
            # it is what a certificate claim quotes, never the requested ε.
            "margin_policy": self.margin_policy,
            "eps_certified": self.eps_certified_float,
            # P2-14: ratios must name their denominator. ``charge_over_eps``
            # is policy-free; ``worst_charge_ratio`` above is against the
            # POLICY's budget and is only comparable within one policy.
            "worst_charge_over_eps": self.worst_charge_over_eps,
            "note_half_margin_holds": self.note_half_margin_holds,
            "V": list(self.v_witnesses),
            "endpoints": [list(e.witnesses) for e in self.endpoints],
        }

    @property
    def exact_pass(self) -> bool:
        return bool(
            self.v_resolve_feasible and self.v_exact.ok
            and self.endpoints and all(e.ok for e in self.endpoints)
        )

    @property
    def worst_charge_ratio(self) -> Optional[float]:
        ratios = [r for r in (
            [self.v_exact.worst_charge_ratio]
            + [e.exact.worst_charge_ratio for e in self.endpoints]
        ) if r is not None]
        return max(ratios) if ratios else None

    @property
    def worst_charge_over_eps(self) -> Optional[float]:
        """max over V and every endpoint of ``charge / ε`` (WP P2-14).

        The POLICY-FREE ratio: its denominator is the requested ε and not the
        threshold in force, so it is comparable across ``margin_policy`` values
        and across records. ``< 1`` is the ``"achieved"`` acceptance test and
        ``≤ 1/2`` is the frozen note's printed ε/2 inequality. The D1 headroom
        trigger is expressed against THIS number since P2-14.
        """
        ratios = [r for r in (
            [self.v_exact.worst_charge_over_eps]
            + [e.exact.worst_charge_over_eps for e in self.endpoints]
        ) if r is not None]
        return max(ratios) if ratios else None

    @property
    def note_half_margin_holds(self) -> Optional[bool]:
        """Whether every octant of every form satisfies ``charge ≤ ε/2``.

        Reported, never enforced (acceptance is ``charge < ε``). ``None`` when
        no octant produced a charge.
        """
        worst = self.worst_charge_over_eps
        return None if worst is None else worst <= 0.5

    @property
    def eps_certified_float(self) -> Optional[float]:
        """min over V and every endpoint of ``ε − charge`` (WP P2-13).

        The margin this phase PROVED, on the worst octant of the worst form.
        Policy-free (the same Gram gives the same number under both policies),
        and for a bracketed row the caller must take the min over corners as
        well before quoting it — Lemma 9.1's convexity argument holds at the
        smallest corner margin, not at ε.
        """
        vals = [v for v in (
            [self.v_exact.eps_certified_float]
            + [e.exact.eps_certified_float for e in self.endpoints]
        ) if v is not None]
        return min(vals) if vals else None


def _exact_phase_once(
    V_gf,
    dynamics_list: Sequence[Sequence],
    n_vars: int,
    *,
    epsilon,
    eps_exact: Fraction,
    v_terms,
    input_hash: str,
    solver: str,
    v_cache: Optional[VProofCache],
    bundle: Optional[PreparedGramBundle],
    bundle_target_rtol: float,
    mapper=None,
    margin_policy: str = MARGIN_POLICY_ACHIEVED,
) -> ExactPhaseResult:
    """ONE witness attempt: the historical body of :func:`run_exact_phase`.

    ``bundle`` is an already hash-verified :class:`PreparedGramBundle` (the
    design witness) or ``None`` (re-solve with ``solver``). Sets the per-attempt
    provenance (``witness_source``, ``witness_solver``); the ladder in
    :func:`run_exact_phase` decides which attempt is adopted and records the
    rest.

    ``mapper`` (P3-03) parallelises the per-octant EXACT validation of V and of
    every endpoint's W. The endpoint loop itself stays serial on purpose: it
    interleaves the float re-solve (``verify_gf_sos_full``, i.e. MOSEK), which
    must not run concurrently with itself — the octants are the finer and
    safer grain (2ⁿ ≥ 16 jobs against 2 endpoint jobs at n = 4) and they cover
    100% of the exact arithmetic either way.
    """
    witness_solver = (bundle.meta.get("solver") or DESIGN_BUNDLE_SOLVER
                      if bundle is not None else solver)
    # The ENCODINGS that actually ran, in call order (P2-10's follow-up):
    # ``JointSOSResult.solver_used`` says MOSEK_DIRECT (Task API) or MOSEK
    # (CVXPY interface) where ``solver`` above only says what was requested.
    used_encodings: List[str] = []
    if v_cache is not None:
        # The input-hash refusal itself is hoisted into run_exact_phase: it is
        # an input check on the caller's arguments, so it must raise once,
        # before any witness is tried, not once per attempt.
        result = ExactPhaseResult(
            v_resolve_feasible=v_cache.resolve_feasible,
            v_exact=v_cache.exact,
            v_grams=v_cache.grams,
            v_reused=True,
            v_input_hash=input_hash,
            v_fingerprint=v_cache.fingerprint,
        )
    elif bundle is not None:
        # R3: the design Gram is the witness. The exact target is still
        # rebuilt from scratch below (inside exact_verify_captures); the
        # bundle contributes Q and its basis and nothing else.
        _check_bundle_group(bundle.v, v_terms, n_vars, eps_exact, "V",
                            rtol=bundle_target_rtol)
        v_exact = exact_verify_captures(v_terms, bundle.v, n_vars, mapper,
                                        margin_policy)
        result = ExactPhaseResult(
            # The design solve that emitted this bundle was feasible; no float
            # solve runs here (see the docstring of ``witness_source``).
            v_resolve_feasible=True,
            v_exact=v_exact,
            v_grams=bundle.v,
            v_reused=False,
            v_input_hash=input_hash,
            v_fingerprint=v_proof_fingerprint(v_exact, bundle.v),
            v_witnesses=[_bundle_witness_row(e) for e in bundle.v],
        )
    else:
        res_v = verify_gf_sos_full(V_gf, epsilon=epsilon, solver=solver,
                                   strict_solver=True, allow_solver_fallback=False,
                                   capture_grams=True)
        used_encodings.append(str(getattr(res_v, "solver_used", "") or ""))
        v_exact = exact_verify_captures(v_terms, res_v.gram_certificates,
                                        n_vars, mapper, margin_policy)
        result = ExactPhaseResult(
            v_resolve_feasible=bool(res_v.feasible),
            v_exact=v_exact,
            v_grams=res_v.gram_certificates,
            v_reused=False,
            v_input_hash=input_hash,
            v_fingerprint=v_proof_fingerprint(v_exact, res_v.gram_certificates),
        )
    result.margin_policy = margin_policy
    result.witness_solver = witness_solver
    if bundle is not None:
        result.witness_source = WITNESS_DESIGN_BUNDLE
        result.gram_bundle_fingerprint = bundle.fingerprint
        result.gram_bundle_parity_expanded = bundle.parity_expanded
    for endpoint, dyn in enumerate(dynamics_list):
        W_gf = negate_gf(compute_lie_derivative_gf(V_gf, dyn, validate=False))
        f_terms = [exact_terms_from_gf(f_i) for f_i in dyn]
        w_terms = exact_neg_lie_derivative(v_terms, f_terms)
        if bundle is not None:
            caps = bundle.w.get(endpoint)
            if not caps:
                raise GramBundleError(
                    "gram_bundle_coverage_incomplete",
                    f"the bundle carries no W blocks for inclusion vertex "
                    f"{endpoint}; endpoints present: {sorted(bundle.w)}")
            _check_bundle_group(caps, w_terms, n_vars, eps_exact,
                                f"W[endpoint {endpoint}]",
                                rtol=bundle_target_rtol)
            w_exact = exact_verify_captures(w_terms, caps, n_vars, mapper,
                                            margin_policy)
            result.endpoints.append(EndpointExactResult(
                resolve_feasible=True,
                exact=w_exact,
                lie_crosscheck=crosscheck_lie(w_terms, W_gf),
                grams=caps,
                witness_source=WITNESS_DESIGN_BUNDLE,
                witnesses=[_bundle_witness_row(e) for e in caps],
                witness_solver=witness_solver,
                witness_solver_used=witness_solver,
            ))
            continue
        res_w = verify_gf_sos_full(W_gf, epsilon=epsilon, solver=solver,
                                   strict_solver=True,
                                   allow_solver_fallback=False,
                                   capture_grams=True)
        w_used = str(getattr(res_w, "solver_used", "") or "")
        used_encodings.append(w_used)
        w_exact = exact_verify_captures(w_terms, res_w.gram_certificates,
                                        n_vars, mapper, margin_policy)
        result.endpoints.append(EndpointExactResult(
            resolve_feasible=bool(res_w.feasible),
            exact=w_exact,
            lie_crosscheck=crosscheck_lie(w_terms, W_gf),
            grams=res_w.gram_certificates,
            witness_solver=witness_solver,
            witness_solver_used=w_used,
        ))
    # On the bundle path nothing was solved here: the encoding of record is
    # the one that produced the DESIGN Gram (MOSEK_DIRECT, or whatever the
    # bundle metadata names).
    result.witness_solver_used = (_join_encodings(used_encodings)
                                  or (witness_solver if bundle is not None else ""))
    return result


def _join_encodings(used: Sequence[str]) -> str:
    """The distinct ``solver_used`` names of one attempt, in call order.

    One attempt runs several solves (V, then one W per inclusion endpoint) and
    each returns its own ``JointSOSResult.solver_used``. They normally agree;
    when they do not — a MOSEK_DIRECT V and a CVXPY-MOSEK W, say — the record
    must say so rather than pick one, so the names are joined with ``+``.
    Empty when nothing solved (a reused V half with no endpoints).
    """
    seen: List[str] = []
    for name in used:
        if name and name not in seen:
            seen.append(name)
    return "+".join(seen)


def _witness_shortfall(result: ExactPhaseResult, trigger: float) -> str:
    """Typed reason this witness is not good enough, ``""`` when it is.

    Only the halves THIS attempt produced are judged. A V half reused from a
    bracket corner via :class:`VProofCache` is not re-solved by a retry, so it
    can never be the reason to retry — corner 0's own ladder already decided
    it. Everything else is: an infeasible re-solve, an exact-gate rejection,
    or a proof whose ``charge/ε`` exceeds ``trigger`` (WP P2-14: the ratio is
    measured against ε, so the trigger no longer depends on the margin
    policy).
    """
    if not result.v_reused:
        if not result.v_resolve_feasible:
            return "v_resolve_infeasible"
        if not result.v_exact.ok:
            return f"v_exact_{result.v_exact.reason or 'failed'}"
    for i, ep in enumerate(result.endpoints):
        if not ep.resolve_feasible:
            return f"w{i}_resolve_infeasible"
        if not ep.exact.ok:
            return f"w{i}_exact_{ep.exact.reason or 'failed'}"
    ratios = [r for r in (
        ([] if result.v_reused else [result.v_exact.worst_charge_over_eps])
        + [e.exact.worst_charge_over_eps for e in result.endpoints]
    ) if r is not None]
    worst = max(ratios) if ratios else None
    if worst is not None and worst > trigger:
        return f"charge_over_eps_{worst:.3e}_above_{trigger:g}"
    return ""


def _witness_plan(
    solver: str,
    fallback_solver: Optional[str],
    has_bundle: bool,
) -> List[Tuple[str, str]]:
    """The ordered ``(source, solver)`` witness ladder (P2-05, D1).

    Design bundle first when one was supplied, then the primary re-solve,
    then exactly one retry with the fallback solver. The retry is dropped
    when it would repeat the primary (a caller that already pins CLARABEL
    keeps its single-attempt behaviour byte for byte) or when the caller
    passed ``fallback_solver=None``.
    """
    primary = str(solver).strip().upper()
    plan: List[Tuple[str, str]] = []
    if has_bundle:
        plan.append((WITNESS_DESIGN_BUNDLE, ""))
    plan.append((WITNESS_RESOLVE, primary))
    if fallback_solver:
        retry = str(fallback_solver).strip().upper()
        if retry != primary:
            plan.append((WITNESS_RESOLVE, retry))
    return plan


def run_exact_phase(
    V_gf,
    dynamics_list: Sequence[Sequence],
    n_vars: int,
    *,
    epsilon: float = 1e-6,
    solver: str = WITNESS_PRIMARY_SOLVER,
    v_cache: Optional[VProofCache] = None,
    v_proof_cache=None,
    gram_bundle=None,
    expect_bundle_fingerprint: Optional[str] = None,
    bundle_target_rtol: float = BUNDLE_TARGET_RTOL,
    fallback_solver: Optional[str] = WITNESS_FALLBACK_SOLVER,
    charge_headroom_trigger: float = WITNESS_CHARGE_TRIGGER,
    jobs: int = DEFAULT_JOBS,
    start_method: str = OCTANT_START_METHOD,
    margin_policy: str = MARGIN_POLICY_ACHIEVED,
) -> ExactPhaseResult:
    """Run the full Proposition-7 exact phase for one certificate.

    ``dynamics_list`` holds one dynamics (list of GeneralizedForm per state)
    for the nominal case, or the endpoint-vertex dynamics for a perturbed
    inclusion. V is re-solved once; each dynamics gets its own W re-solve and
    exact proof.

    ``v_cache`` reuses a V proof from an earlier bracket corner (R2). The
    cache's :func:`v_input_hash` must match the V/ε handed in here, otherwise
    :class:`ExactPhaseError` is raised: reuse is only ever allowed for a
    provably identical V proof. Pass ``None`` (the default, and what
    ``--no-dedup`` restores) to re-solve and re-prove V from scratch.

    ``v_proof_cache`` (P3-05) is the same reuse ACROSS PROCESSES: a
    :class:`DiskVProofCache`, a path to its root, or ``True`` for the default
    root (:func:`resolve_vproof_cache_root`). A corner process that finds an
    entry skips the V float re-solve; by default it still re-runs the exact
    rational gate on the stored Gram captures, so nothing about the stored
    proof is trusted. Every outcome — hit, miss, or a typed refusal from
    :data:`VPROOF_CACHE_REASONS` — is recorded in
    ``ExactPhaseResult.v_proof_cache`` and in ``witness_record()``, and every
    refusal falls back to the full proof rather than failing the row. An
    in-process ``v_cache`` takes priority; a ``gram_bundle`` disables the disk
    cache, since with a design witness there is no re-solve to save.

    ``gram_bundle`` (R3, decisions D1/D2) makes the DESIGN-stage Grams the
    witness instead of a fresh float re-solve: pass
    ``JointSOSResult.gram_certificates``, the result object itself, a packaged
    document, or a path to a ``.json``/``.json.gz``/``.npz`` bundle written by
    :func:`save_gram_bundle`. When it is supplied **no float solve runs on the
    certifying route** — ``verify_gf_sos_full`` is not called at all. The
    exact targets are still rebuilt from scratch in ℚ, the bundle is
    hash-verified entry by entry and against its fingerprint
    (``gram_bundle_hash_mismatch``), and its stored targets, margins, μ and
    octant labels are checked against the exact target being built
    (``gram_bundle_input_mismatch``); either refusal RAISES
    :class:`GramBundleError`. ``epsilon`` must be the margin the design
    subtracted, since that is the margin the bundle can witness. A
    parity-reduced bundle is expanded from its ``parity_class`` and the record
    says which octants were covered by a partner.

    ``expect_bundle_fingerprint`` is an optional expected fingerprint: supply
    the value an artifact recorded and a substituted bundle is refused before
    anything is proved. (The name avoids shadowing the module-level
    ``gram_bundle_fingerprint`` helper inside this function.)

    ``solver`` (P2-05, decision D1) is the PRIMARY re-solve solver, MOSEK by
    default. When the witness it proposes does not prove the certificate, or
    proves it with a worst exact ``charge/ε`` ratio above
    ``charge_headroom_trigger`` (0.1), the phase is retried EXACTLY once with
    ``fallback_solver`` (CLARABEL) and the better verified witness is adopted.
    Pass ``fallback_solver=None`` for the single-attempt behaviour; a caller
    that pins ``solver="CLARABEL"`` also gets a single attempt, since the
    retry would repeat it. Every attempt — including a design bundle refused
    for want of a margin — is recorded in ``witness_attempts``, and
    ``witness_source`` / ``witness_solver`` / ``witness_charge_ratio`` name
    the adopted one. Adoption is never a trust decision about a solver: an
    attempt can only be adopted because the EXACT rational phase verified the
    Gram it proposed.

    ``jobs`` (P3-03) is the number of WORKER PROCESSES the per-octant exact
    validation is spread over; 1 (the default) is the historical serial loop
    and creates no pool. Only the exact rational arithmetic is parallelised —
    the float re-solve and the endpoint loop stay serial, and the witness
    ladder, the disk V-proof cache and every recorded field are unchanged.
    Scheduling cannot move a verdict or a margin: octant results are assembled
    by index and truncated at the first failure regardless of completion
    order, and a worker that raises or dies fails the attempt with a typed
    :class:`OctantWorkerError` instead of returning a short certificate.
    ``start_method`` is ``"spawn"`` and should stay so
    (:data:`OCTANT_START_METHOD` explains why fork is unsafe here). Both are
    recorded in ``ExactPhaseResult.parallel`` and in ``witness_record()``.

    ``margin_policy`` is the acceptance THRESHOLD of the exact gate and
    nothing else: ``"achieved"`` (DEFAULT since P2-14, owner decision D-l)
    accepts an octant at ``charge < ε`` strictly, ``"half"`` (historical, the
    frozen note's printed inequality) at ``charge ≤ ε/2``. Every other step of
    Proposition 7 is identical, and both prove ``F₀ ≥ (ε − charge) Σᵢ yᵢᵈ``;
    the policy only decides how much of ε the certificate must retain. Read
    the proved margin off ``eps_certified_float`` (and quote THAT, never the
    requested ε); ``note_half_margin_holds`` says whether the row also
    satisfies the note's ε/2 sentence. Two consequences a caller must know:
    ``worst_charge_ratio`` is measured against the policy's own budget and is
    only comparable within one policy (use ``worst_charge_over_eps``, which
    the D1 headroom trigger is expressed against since P2-14, so the trigger
    no longer moves with the policy); and the persisted V-proof cache is keyed
    per policy, so a proof accepted at ``"achieved"`` can never be reused by a
    ``"half"`` run, and the historical ``"half"`` cache keys stay byte for byte
    what they were.
    """
    t0 = time.time()
    margin_policy = resolve_margin_policy(margin_policy)
    v_terms = exact_terms_from_gf(V_gf)
    input_hash = v_input_hash(V_gf, epsilon)
    eps_exact = _exact_epsilon(epsilon)

    if v_cache is not None and v_cache.input_hash != input_hash:
        raise ExactPhaseError(
            "cached V proof does not belong to this certificate: "
            f"expected input hash {v_cache.input_hash}, got {input_hash}. "
            "V terms, weights or epsilon differ across corners; refusing "
            "to reuse the proof (re-run with --no-dedup to prove each "
            "corner independently)"
        )
    if v_cache is not None:
        v_cache = _checked_v_cache(v_cache, V_gf, epsilon, n_vars, margin_policy)

    # P3-05: the persisted V proof, consulted before any witness is tried.
    # P2-13: a V proof accepted under a LOOSER threshold must never be
    # reusable by a run under the historical one, so the policy goes into the
    # cache key (``extra``) for every policy except "half", which keeps its
    # historical keys byte for byte. The rule lives in ONE place, exported, so
    # a caller can look up the entry this run will use.
    disk = vproof_cache_for_margin_policy(
        as_disk_vproof_cache(v_proof_cache), margin_policy)
    cache_event = VProofCacheEvent()
    disk_key: Optional[str] = None
    if disk is not None:
        disk_key = disk.key_for(V_gf, epsilon, n_vars)
        if v_cache is not None:
            cache_event = VProofCacheEvent(
                enabled=True, reason="vproof_cache_in_process_hit",
                detail="an in-process VProofCache was supplied and takes priority",
                root=str(disk.root), path=str(disk.path_for(disk_key)),
                key=disk_key, code_version=verifier_code_version())
        elif gram_bundle is not None:
            cache_event = VProofCacheEvent(
                enabled=True, reason="vproof_cache_bundle_witness",
                detail="a design Gram bundle witnesses V; there is no re-solve to save",
                root=str(disk.root), path=str(disk.path_for(disk_key)),
                key=disk_key, code_version=verifier_code_version())
        else:
            v_cache, cache_event = disk.load(
                V_gf, epsilon, n_vars, v_terms=v_terms, input_hash=input_hash)

    bundle: Optional[PreparedGramBundle] = None
    if gram_bundle is not None:
        bundle = prepare_gram_bundle(
            gram_bundle, n_vars, expected_fingerprint=expect_bundle_fingerprint)
        n_groups = len(bundle.w)
        if n_groups != len(dynamics_list):
            raise GramBundleError(
                "gram_bundle_input_mismatch",
                f"the bundle carries W blocks for {n_groups} inclusion "
                f"vertex/vertices {sorted(bundle.w)}, this phase was handed "
                f"{len(dynamics_list)} dynamics; a bundle may not witness a "
                "different inclusion structure than the one being proved")

    # P3-03: the per-octant schedule. Resolved ONCE for the whole ladder, so
    # every witness attempt of one call is scheduled identically and the
    # recorded counts cover the attempts that actually ran.
    mapper, pool = octant_mapper(jobs, start_method)
    batches0 = pool.n_batches if pool is not None else 0
    tasks0 = pool.n_tasks if pool is not None else 0

    plan = _witness_plan(solver, fallback_solver, bundle is not None)
    attempts: List[WitnessAttempt] = []
    results: List[Optional[ExactPhaseResult]] = []
    for index, (source, attempt_solver) in enumerate(plan):
        last = index == len(plan) - 1
        t_attempt = time.time()
        try:
            attempt_result = _exact_phase_once(
                V_gf, dynamics_list, n_vars,
                epsilon=epsilon, eps_exact=eps_exact, v_terms=v_terms,
                input_hash=input_hash, solver=attempt_solver, v_cache=v_cache,
                bundle=bundle if source == WITNESS_DESIGN_BUNDLE else None,
                bundle_target_rtol=bundle_target_rtol, mapper=mapper,
                margin_policy=margin_policy)
        except GramBundleError as exc:
            # A design-side limit (no ε rows, clearing active) means there is
            # no bundle witness for this block, which is a recorded attempt,
            # not a failure. Anything else means the bundle is not the object
            # it claims to be: hard stop, as P2-03 specified.
            if exc.reason not in BUNDLE_FALLTHROUGH_REASONS:
                raise
            attempts.append(WitnessAttempt(
                index=index, source=source,
                solver=(bundle.meta.get("solver") or DESIGN_BUNDLE_SOLVER)
                       if bundle is not None else attempt_solver,
                ok=False, shortfall=exc.reason, detail=exc.detail[:300],
                time_s=time.time() - t_attempt))
            results.append(None)
            continue
        except Exception as exc:  # noqa: BLE001 - typed, then re-raised if last
            # A re-solve that blew up (solver missing, out of memory, ...) is
            # a typed failed attempt as long as another witness is left; the
            # last attempt's exception propagates, because then nothing proved
            # anything and a silent None would be a false negative.
            if last:
                raise
            attempts.append(WitnessAttempt(
                index=index, source=source, solver=attempt_solver, ok=False,
                # A P3-03 worker failure carries its own typed reason
                # (octant_worker_died, ...); everything else goes through the
                # P3-02 exception classes.
                shortfall=(exc.reason if isinstance(exc, OctantWorkerError)
                           else classify_exception(exc)),
                detail=f"{type(exc).__name__}: {exc}"[:300],
                time_s=time.time() - t_attempt))
            results.append(None)
            continue
        shortfall = _witness_shortfall(attempt_result, charge_headroom_trigger)
        attempts.append(WitnessAttempt(
            index=index, source=attempt_result.witness_source,
            solver=attempt_result.witness_solver,
            solver_used=attempt_result.witness_solver_used,
            ok=bool(attempt_result.exact_pass),
            worst_charge_ratio=attempt_result.worst_charge_ratio,
            worst_charge_over_eps=attempt_result.worst_charge_over_eps,
            shortfall=shortfall, time_s=time.time() - t_attempt))
        results.append(attempt_result)
        if not shortfall:
            break

    adopted = _adopt_witness(attempts, results)
    result = results[adopted]
    assert result is not None  # _adopt_witness only returns a real result
    attempts[adopted].adopted = True
    result.witness_charge_ratio = result.worst_charge_ratio
    result.witness_attempts = [a.as_record() for a in attempts]
    # P3-05: seed the persisted cache from a V half that was proved HERE. A
    # reused half (in-process or from the cache itself) is already stored, and
    # a failed one would only be re-read and refused.
    if disk is not None and not cache_event.hit and not result.v_reused \
            and cache_event.reason not in ("vproof_cache_in_process_hit",
                                           "vproof_cache_bundle_witness"):
        disk.store(V_gf, epsilon, n_vars, result, key=disk_key, into=cache_event)
    result.v_proof_cache = cache_event.as_record()
    result.parallel = parallel_record(
        jobs, start_method, pool,
        n_batches=(pool.n_batches - batches0) if pool is not None else 0,
        n_tasks=(pool.n_tasks - tasks0) if pool is not None else 0)
    result.time_s = round(time.time() - t0, 2)
    return result


def _adopt_witness(
    attempts: Sequence[WitnessAttempt],
    results: Sequence[Optional[ExactPhaseResult]],
) -> int:
    """Index of the witness to adopt.

    In order: the first attempt that proved the certificate with the required
    headroom (the ladder stops there, so it is the last one tried); otherwise
    the VERIFIED attempt with the smallest exact charge ratio, because a proof
    with more headroom is the better record of the same fact; otherwise the
    last attempt that returned at all, so the record carries the failure that
    was actually reached rather than an earlier one.
    """
    for i, attempt in enumerate(attempts):
        if results[i] is not None and attempt.ok and not attempt.shortfall:
            return i
    verified = [i for i, a in enumerate(attempts)
                if results[i] is not None and a.ok]
    if verified:
        return min(verified, key=lambda i: (
            attempts[i].worst_charge_ratio
            if attempts[i].worst_charge_ratio is not None else float("inf")))
    returned = [i for i, r in enumerate(results) if r is not None]
    if returned:
        return returned[-1]
    raise ExactPhaseError(
        "no witness produced an exact phase: tried "
        + ", ".join(f"{a.source}/{a.solver or '-'} ({a.shortfall})"
                    for a in attempts))
