"""Paper-machinery GF-SOS discovery (Sanchez-Moreno 2018), full-library.

The flow is exactly the paper's method, automated:

  Input dynamics → build the C¹ GF term library at degree m → keep the
  μ-compatible terms (the tractability mechanism — see README) →
  **solve the SOS problem** for the coefficients over the FULL library
  (associated forms + standard SOS, §5.3 / Algorithm 8) → strict re-gate →
  escalate the degree m until a certificate is found.

The coefficients are always *designed* by SOS, and every accepted candidate is
re-gated strictly by ``certify_discovered_gf`` (standard SOS of V>0 and
W=−V̇>0 with a-posteriori Gram validation, plus the Lemma-5 C¹ surface
classification; with inclusion vertices, W is gated per extremal dynamics).
There is no regional/grid fallback: a system that does not certify at the
reachable degrees is reported as an honest negative.

C¹ (restriction R2) is kept because Lemma 5 — the continuous-time bridge for
these discontinuous Filippov systems — requires V ∈ C¹ (which is also what
makes Clarke unnecessary). The classical LF of some systems is non-smooth, and
escalating m is the route to an SOS-certifiable C¹ GF LF (converse theorem).

Do not read the escalation as a statement about the SYSTEM. This docstring used
to say the CTA Lyapunov function is "smooth-but-not-SOS-certifiable at its
degree"; that is false. WP P5-01 verified the published 6-term CTA function at
m = 5 exactly and shipped a portable record for it. What is true is the weaker
sentence about the DESIGN: this escalation does not reach m = 5 on CTA at any
margin tested (negative N20) and stops at the first degree that passes the
strict gate, m = 8. Owner decision D-j (2026-09-04) pins 8 as the canonical CTA
design degree.
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from ..algebra.core import GeneralizedForm
from ..algebra.enumerate import enumerate_diagonal_terms, enumerate_cross_terms, TermSpec
from ..verification.joint_sos import (
    EpsilonRung, JointSOSOptimizer, _normalize_solver_name,
    compute_mu_exponents_with_dynamics, get_design_epsilon_sweep_default,
    gram_bundle_fingerprint, resolve_epsilon_ladder)
from ..verification.verifier import certify_discovered_gf
from .c1_discovery import _degree_candidates, _spans_all_vars


def forced_mu(diag, weights_r, dynamics_gf):
    """Minimal μ forced by the un-droppable part of the problem: the diagonal terms
    (one pure power per axis, required for positive-definiteness) plus the dynamics.

    μ controls the y-degree of the associated forms and hence the SOS Gram size. The
    diagonals + dynamics pin μ to its smallest value (μ ∝ r); see
    ``compute_mu_exponents_with_dynamics``.
    """
    dspecs = [TermSpec(exponents=t.exponents, term_types=t.term_types) for t in diag]
    _, mu = compute_mu_exponents_with_dynamics(dspecs, weights_r, dynamics_gf)
    return mu


def mu_compatible(term, mu) -> bool:
    """True iff every variable's y-exponent ``μ_i·a_i`` is a non-negative even integer.

    A cross term that is NOT compatible forces a larger μ on (via the global μ ∝ r
    rule) EVERY axis — e.g. an ``x₃^{4/3}`` factor on the unit-weight axis triples μ
    and explodes the per-octant Gram from 36 to 91 rows, turning a sub-second SOS
    into a multi-minute one. The paper's hand-built Lyapunov functions only ever use
    μ-compatible terms; restricting the library to them (and escalating the degree m
    to recover expressiveness) is what keeps the SDP the size the paper's runs are.
    """
    for i, e in enumerate(term.exponents):
        if e == 0:
            continue
        ye = mu[i] * e
        if ye != int(ye) or int(ye) % 2 != 0:
            return False
    return True


def complete_lattice_cap(mu) -> int:
    """Exhaustive denominator cap for the μ-compatible slice at any degree.

    μ-compatibility requires μ_i·a_i to be an (even) integer, so a reduced
    exponent p/q on axis i needs q | μ_i — every μ-compatible denominator
    divides some μ_i, and enumerating with max_denominator = max(μ) provably
    exhausts the slice. A fixed cap of 4 silently truncated the n=5 slices
    (Levant-5D m=7: 59 of 61 terms; m=8: 103 of 111) — never hard-code it.
    """
    return max(int(v) for v in mu)


def mu_filtered_cross(m, weights_r, diag, dynamics_gf):
    """Enumerate the COMPLETE μ-compatible cross-term slice at degree m.

    Returns ``(mu, cross_full, cross)`` where ``cross_full`` is the unfiltered
    C¹ enumeration at the exhaustive cap and ``cross`` the μ-compatible subset.
    This is the one canonical way to build the library ``discover_gf`` solves
    over — scripts must call this instead of re-rolling the enumeration.
    """
    mu = forced_mu(diag, weights_r, dynamics_gf)
    cross_full = enumerate_cross_terms(m, weights_r,
                                       max_denominator=complete_lattice_cap(mu),
                                       include_signed=True, minimal=False,
                                       include_abs_variants=True, c1_only=True)
    return mu, cross_full, [t for t in cross_full if mu_compatible(t, mu)]


# Backwards-compatible aliases (scripts written before the names went public).
_forced_mu = forced_mu
_mu_compatible = mu_compatible


@dataclass
class GFDiscoveryResult:
    system: str = ""
    certified: bool = False
    # path ∈ {gf_sos, none}
    path: str = "none"
    # method ∈ {gf_sos_full_library, none}
    method: str = "none"
    claim_level: str = "candidate"
    V_gf: Optional[GeneralizedForm] = None
    coefficients: Optional[dict] = None
    degree_used: int = 0
    n_terms: int = 0
    # "cheap" evidence: how small the certified structure is vs. the full C¹ library
    c1_library_diag: int = 0
    c1_library_cross: int = 0           # μ-compatible cross terms actually used by the SDP
    c1_library_cross_full: int = 0      # full enumerated C¹ cross library (pre-μ-filter)
    mu: tuple = ()                      # μ exponents (small ⇒ small SOS; μ ∝ r)
    regularity: str = ""
    surface_scope: str = ""
    degrees_tried: list = field(default_factory=list)
    sdp_solves: int = 0
    total_time: float = 0.0
    message: str = ""
    # Numerical-soundness evidence from the strict gate (worst octant): smallest
    # SOS Gram eigenvalue, largest ℓ1 equality residual, largest Gram condition
    # number, ε-margin actually used.
    gram_min_eig: Optional[float] = None
    gram_residual_l1: Optional[float] = None
    gram_cond: Optional[float] = None
    epsilon_used: Optional[str] = None
    # Sparsification provenance (sparsify="reweighted_l1"): dense term count
    # before sparsification, the sparsifier used ("none" if disabled or failed),
    # reweighted-ℓ₁ iterations run, whether the support stabilized, and the
    # reason if the sparse candidate was rejected (dense certificate kept).
    n_terms_dense: int = 0
    sparsifier: str = "none"
    reweight_iters: int = 0
    support_stable: bool = False
    sparsify_failed_reason: str = ""
    # Per-degree attempt log: one entry per degree tried, each
    # {"degree": m, "status": "certified"|"no_certificate"|"unexpected_error",
    # "message": str, ["traceback": str]}. Lets run JSONs distinguish an honest
    # SOS/gate negative from a solver crash at that degree (see
    # ``_design_full_library``).
    degree_attempts: list = field(default_factory=list)
    # DESIGN Gram bundle of the solve that produced the certified V (P2-01
    # schema ``gf.design_grams/1``, one entry per emitted PSD block). This is
    # the witness the exact phase re-proves on
    # (``run_exact_phase(..., gram_bundle=...)``, R3/P2-03) and it is the SAME
    # object the design SDP solved, not a re-solve: ``_design_full_library``
    # keeps its ``JointSOSResult`` and hands the bundle up.
    #
    # ``None`` with a reason in ``gram_bundle_reason`` when the design Grams
    # cannot be attributed to the certified V: the backend was not
    # MOSEK_DIRECT, the design was infeasible, ``capture_design_grams=False``,
    # or the certified V is NOT the design proposal (the weighted-sphere
    # redesign inside ``certify_discovered_gf``, or a sparsified support). A
    # bundle from a different V would be refused downstream
    # (``gram_bundle_input_mismatch``); refusing to attach it here says why.
    gram_certificates: Optional[list] = None
    gram_bundle_fingerprint: Optional[str] = None
    gram_bundle_reason: str = ""
    # The margin the DESIGN subtracted (``JointSOSOptimizer.epsilon``, 1e-4 by
    # default), AFTER the V normalisation factor: the exact phase must be given
    # exactly this ε, because it is the only margin these Grams can witness
    # (P2-03). With no normalisation it equals ``design_epsilon_requested``.
    design_epsilon: Optional[float] = None
    # WP P5-02, the design-epsilon sweep. ``design_epsilon_ladder`` is the
    # ladder that was actually walked (empty when the sweep was off);
    # ``design_epsilon_requested`` is the rung that certified, BEFORE the
    # normalisation factor; ``v_normalization`` is that rung's normalisation
    # record (``verification.v_normalization.VNormalization.as_record``);
    # ``epsilon_rungs`` is one record per (degree, rung) attempted, so a run
    # JSON shows which margins were tried and what each of them returned.
    design_epsilon_ladder: tuple = ()
    design_epsilon_requested: Optional[float] = None
    v_normalization: Optional[dict] = None
    epsilon_rungs: list = field(default_factory=list)


def _design_full_library(weights_r, m, terms, dynamics_gf, solver, verbose,
                         dynamics_gf_vertices=None, allow_fallback=True,
                         degree_attempts: Optional[list] = None,
                         design_out: Optional[dict] = None):
    """Solve the joint V+W SOS problem over the FULL C¹ term library.

    Standard SOS *designs* the coefficients; returns the proposed V (re-gated by
    the caller) or None if the SDP is infeasible / errors. With inclusion
    vertices the W block is duplicated per vertex (robust design).
    ``allow_fallback=False`` pins the requested solver — required for
    MOSEK_DIRECT at large n, where a fallback to the CVXPY path would re-enter
    the canonicalization memory wall.

    An unexpected exception here is NOT an ordinary SOS negative — it can
    silently poison publication-grade "not certified" claims if swallowed. It
    is always reported to stderr and, when ``degree_attempts`` is given (the
    caller's per-degree log), recorded there with status "unexpected_error"
    and a full traceback, so the run JSON shows a crash, not a solver
    negative, occurred at this degree. The return-None contract is unchanged:
    the caller still treats this degree as having produced no candidate.

    ``design_out`` is an optional dict the caller owns; when given it receives
    ``{"result": JointSOSResult|None, "epsilon": float}``. The RETURN value is
    still the proposed ``V_gf`` (two scripts and the orchestration tests bind
    it that way), but the ``JointSOSResult`` is no longer dropped: its
    ``gram_certificates`` are the design Gram bundle stage X re-proves on
    (P2-11), and ``epsilon`` is the margin those Grams witness.
    """
    opt = JointSOSOptimizer(weights_r=weights_r, solver=solver, verbose=False,
                            allow_solver_fallback=allow_fallback)
    if design_out is not None:
        design_out["result"] = None
        design_out["epsilon"] = float(opt.epsilon)
    try:
        res = opt.find_lyapunov_with_dynamics(
            degree_m=m, dynamics_gf=dynamics_gf, active_terms=list(terms),
            dynamics_gf_vertices=dynamics_gf_vertices,
        )
    except Exception as e:  # keep the degree sweep alive
        tb = traceback.format_exc()
        print(f"[gf_discovery] m={m}: full-library SOS error: {e}\n{tb}",
              file=sys.stderr, flush=True)
        if verbose:
            print(f"  m={m}: full-library SOS error: {e}", flush=True)
        if degree_attempts is not None:
            degree_attempts.append({
                "degree": m, "status": "unexpected_error",
                "message": f"full-library SOS error: {e}", "traceback": tb,
            })
        return None
    if design_out is not None:
        design_out["result"] = res
    return res.V_gf if res.feasible else None


def _design_rungs(
    weights_r,
    m,
    terms,
    dynamics_gf,
    solver,
    verbose,
    *,
    dynamics_gf_vertices=None,
    allow_fallback=True,
    degree_attempts: Optional[list] = None,
    epsilon_ladder: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> Iterator[EpsilonRung]:
    """The design proposals to offer the gate at ONE degree (WP P5-02).

    ``epsilon_ladder is None`` reproduces the shipped path exactly: one call to
    :func:`_design_full_library`, at the optimizer's own ``epsilon``, with no
    normalisation and with the design screen's refusal behaving as it always
    did (a refused design yields no candidate). That branch is what
    ``tests/test_epsilon_sweep.py`` pins as byte-identical.

    With a ladder, the rungs come from
    :meth:`JointSOSOptimizer.iter_designs_over_epsilons`, which normalises each
    candidate for the gate and types a screen refusal instead of swallowing it.
    Because it is a generator, the caller stops after the first rung that
    gates: an unswept-equivalent system pays for exactly one solve.
    """
    if epsilon_ladder is None:
        design_out: dict = {}
        V_proposal = _design_full_library(
            weights_r, m, terms, dynamics_gf, solver, verbose,
            dynamics_gf_vertices=dynamics_gf_vertices,
            allow_fallback=allow_fallback, degree_attempts=degree_attempts,
            design_out=design_out)
        result = design_out.get("result")
        eps = design_out.get("epsilon")
        entries = list(getattr(result, "gram_certificates", None) or []) or None
        yield EpsilonRung(
            index=0,
            epsilon=float(eps) if eps is not None else 0.0,
            epsilon_effective=float(eps) if eps is not None else 0.0,
            result=result, V_gf=V_proposal,
            feasible=bool(getattr(result, "feasible", False)),
            screen_refused=False,
            solver_status=str(getattr(result, "solver_status", "") or ""),
            message=str(getattr(result, "message", "") or ""),
            gram_certificates=entries)
        return

    opt = JointSOSOptimizer(weights_r=weights_r, solver=solver, verbose=False,
                            allow_solver_fallback=allow_fallback)
    for rung in opt.iter_designs_over_epsilons(
            epsilon_ladder, normalize=normalize,
            degree_m=m, dynamics_gf=dynamics_gf, active_terms=list(terms),
            dynamics_gf_vertices=dynamics_gf_vertices):
        if rung.solver_status == "design_error" and degree_attempts is not None:
            # Same rule as ``_design_full_library``: a crash is recorded as a
            # crash, never as an SOS negative.
            degree_attempts.append({
                "degree": m, "status": "unexpected_error",
                "epsilon": float(rung.epsilon),
                "message": f"full-library SOS error: {rung.message}",
                "traceback": rung.traceback,
            })
        yield rung


def _adopt_design_bundle(design_result, V, V_proposal, *,
                         entries=None, refusal: str = ""):
    """The design Gram bundle, but only when it belongs to the certified V.

    Returns ``(entries, fingerprint, reason)``. ``entries`` is ``None``
    whenever the bundle cannot be attributed, and ``reason`` says which case
    it was. The exact phase would refuse a mismatched bundle anyway
    (``gram_bundle_input_mismatch``, P2-03), but a typed reason recorded at the
    source is what tells a reader whether the row HAS a design witness, and
    why not when it does not.

    ``refusal`` (WP P5-02) short-circuits with a reason the CALLER already
    knows, which is how a design-screen refusal on a swept rung reaches the
    record: the candidate certified, its Grams did not pass the screen, so the
    row has no design witness and the exact phase must re-solve. Never silent.

    ``entries`` overrides the bundle read off ``design_result``, which is what
    the V normalisation needs: the entries that witness the SCALED V are the
    scaled ones, not the ones the solver returned.
    """
    if refusal:
        return None, None, refusal
    if design_result is None:
        return None, None, "design_result_unavailable"
    if V is not V_proposal:
        # ``certify_discovered_gf`` returns a redesigned V on the
        # weighted-sphere clearing path, and the sparsifier replaces V with a
        # pruned support. Either way the Grams below certify a DIFFERENT V.
        return None, None, "certified_V_is_not_the_design_proposal"
    if entries is None:
        entries = getattr(design_result, "gram_certificates", None)
    if not entries:
        return None, None, ("no_design_grams (not MOSEK_DIRECT, or "
                            "capture_design_grams=False)")
    return list(entries), gram_bundle_fingerprint(entries), ""


def discover_gf(
    spec,
    solver: str = "MOSEK_DIRECT",
    time_budget: float = 1800.0,
    max_extra_degree: int = 4,
    mu_filter: bool = True,
    sparsify: str = "none",
    verbose: bool = False,
    gate_solver: Optional[str] = None,
    design_fallback: bool = True,
    gate_fallback: bool = True,
    epsilon_ladder: Optional[Sequence[float]] = None,
    v_normalization: bool = True,
) -> GFDiscoveryResult:
    """Run paper-machinery GF-SOS discovery for one controller.

    Per escalating degree m: build the C¹ GF library, keep the μ-compatible
    terms, solve the joint V+W SOS problem over the full library to design the
    coefficients, and accept the first candidate that passes the strict
    ``certify_discovered_gf`` gate (global asymptotic stability via Lemma 5).
    No certificate over the reachable degrees ⇒ honest negative.

    A certified result carries the DESIGN Gram bundle of the solve that
    produced it (``gram_certificates``, ``gram_bundle_fingerprint``,
    ``design_epsilon``; P2-11). That bundle is the witness the exact phase
    re-proves on, so the chain D -> X needs no second design solve. It is
    attached only when the certified V IS the design proposal — see
    ``_adopt_design_bundle`` and ``gram_bundle_reason``.

    ``sparsify="reweighted_l1"``: at the first m where the dense candidate
    passes the strict gate, run the reweighted-ℓ₁ sparse design
    (``verification.sparse_sos``) over the same μ-filtered library and re-gate
    the sparse candidate; on success it replaces the dense certificate, on
    failure the dense certificate is kept (``sparsify_failed_reason`` records
    why). Sparsification never changes which m certifies.

    ``solver`` defaults to ``MOSEK_DIRECT`` (decision D1: D is the MOSEK direct
    Task-API design), flipped from ``"MOSEK"`` by the P2-06 validation gate.
    Measured before the flip over the 12 nominal rows: identical certification
    verdict, identical certifying degree and identical dense term count on
    12/12, 660.9 s -> 45.2 s, and no 48 GB CVXPY canonicalization on levant4d
    (``artifacts/campaign_2026-08/E1_matrix/validation_gate_2026-09-03/``).

    ``gate_solver``: solver for the strict gate; it defaults to ``solver``
    EXCEPT when ``solver`` is ``MOSEK_DIRECT``, where it defaults to ``MOSEK``.
    ``verify_gf_sos_full`` does not know the name ``MOSEK_DIRECT``, and since
    P2-10 its ``MOSEK`` rung already IS the direct Task-API poly-SOS backend, so
    ``MOSEK`` is how the gate asks for the same encoding. Pass ``gate_solver``
    explicitly to split them any other way. ``design_fallback=False`` pins the
    design solver (see
    ``_design_full_library``); ``gate_fallback=False`` pins the strict gate
    to exactly ``gate_solver`` (``strict_solver=True`` in
    ``certify_discovered_gf``) — campaign rule 6 requires both pins
    (closes the C19(a) gap).

    ``epsilon_ladder`` (WP P5-02) retries the design at several margins BEFORE
    escalating the degree, coarsest first
    (``joint_sos.DESIGN_EPSILON_LADDER`` is the measured ladder
    ``(1e-4, 1e-6, 1e-8)``). ``None`` takes the process-wide default
    (``joint_sos.set_design_epsilon_sweep_default``), which ships as ``None``
    and means the historical one-design-per-degree loop. The rungs are
    generated lazily, so the first rung that gates ends the ladder and a system
    that certifies at the first margin pays for exactly one solve. Why it
    matters: the design margin is a real tightening of the SDP (a strict margin
    on every pure-power axis), so a primal-infeasible solve at 1e-4 says
    nothing about the degree, and escalating m on it can walk past a
    lower-degree certificate. Every rung's proposal still faces the same strict
    gate; the sweep changes which candidates are offered, never what certifies.

    ``v_normalization`` (WP P5-02, active only under a ladder) multiplies the
    candidate, its design Gram bundle and its design margin by one exact power
    of two before the gate, so a design whose smallest pure-power coefficient
    has drifted below the gate's ABSOLUTE margin is still visible to it. Scaling
    V by a positive constant changes no Lyapunov claim (see
    ``verification.v_normalization``); it changes the numeric margin, which is
    why the bundle and ``design_epsilon`` move with it. It is a measured no-op
    on all twelve nominal rows at the shipped 1e-4 margin.
    """
    t0 = time.time()
    gate_solver = gate_solver or ("MOSEK" if _normalize_solver_name(solver)
                                  == "MOSEK_DIRECT" else solver)
    weights_r = tuple(spec.weights_r)
    n = len(weights_r)
    out = GFDiscoveryResult(system=spec.name)
    degrees = _degree_candidates(spec.degree_m, weights_r, max_extra_degree)
    ladder = (get_design_epsilon_sweep_default() if epsilon_ladder is None
              else resolve_epsilon_ladder(epsilon_ladder))
    out.design_epsilon_ladder = tuple(ladder or ())

    for m in degrees:
        if time.time() - t0 > time_budget:
            break
        # Skip degrees where the C¹ library cannot even span every variable
        # (e.g. STA at m=2: the |x1|^1 diagonal is non-C¹) — escalate instead.
        diag = enumerate_diagonal_terms(m, weights_r, c1_only=True)
        if not _spans_all_vars(diag, n):
            if verbose:
                print(f"[{spec.name}] m={m}: C¹ diagonals don't span all vars → escalate")
            continue
        # Keep only μ-compatible cross terms. A single μ-inflating term (e.g. an
        # x₃^{4/3} factor on the unit-weight axis) would triple μ and explode the
        # per-octant SOS Gram (36→91 rows for 3D), turning a sub-second solve into a
        # multi-minute one — yet the paper's Lyapunov functions never use such terms.
        # Restricting to μ-compatible terms keeps the SDP the size the paper's runs
        # are; degree escalation recovers any expressiveness lost. See mu_compatible.
        if mu_filter:
            # The enumeration cap is derived from μ (complete_lattice_cap), so the
            # filtered library is the COMPLETE lattice slice at degree m — this is
            # what makes Proposition 3's relative completeness hold as shipped.
            mu_min, cross_full, cross = mu_filtered_cross(
                m, weights_r, diag, spec.dynamics_gf)
        else:
            # Completeness experiment mode: keep the μ-inflating terms too (the
            # designer/gate recompute the inflated μ themselves). Expect much
            # larger Grams — this exists to MEASURE the cost the filter avoids
            # and to check the filter does not exclude certifiable systems. The
            # cap-4 bound here is a search-breadth choice for the experiment, not
            # a lattice-completeness claim (off-lattice terms have no finite cap).
            mu_min = forced_mu(diag, weights_r, spec.dynamics_gf)
            cross_full = enumerate_cross_terms(m, weights_r, max_denominator=4,
                                               include_signed=True, minimal=False,
                                               include_abs_variants=True,
                                               c1_only=True)
            cross = list(cross_full)
        out.degrees_tried.append(m)
        if verbose and len(cross) < len(cross_full):
            print(f"[{spec.name}] m={m}: μ={mu_min}; cross library "
                  f"{len(cross_full)} → {len(cross)} μ-compatible", flush=True)

        method = "gf_sos_full_library"
        if verbose:
            print(f"[{spec.name}] m={m}: full-library GF-SOS "
                  f"(library {len(diag)} diag + {len(cross)} cross) ...", flush=True)
        n_attempts_before = len(out.degree_attempts)
        # ``_design_rungs`` keeps the JointSOSResult the design solve produced.
        # Its ``gram_certificates`` ARE the witness stage X re-proves on, so
        # dropping them (as this call did before P2-11) forced the publication
        # driver to re-solve the same SDP in a sibling script just to get them
        # back to disk. With no ``epsilon_ladder`` it yields exactly one rung,
        # from the same ``_design_full_library`` call as before (P5-02).
        for rung in _design_rungs(
                weights_r, m, list(diag) + list(cross), spec.dynamics_gf,
                solver, verbose,
                dynamics_gf_vertices=getattr(spec, "dynamics_gf_vertices", None),
                allow_fallback=design_fallback,
                degree_attempts=out.degree_attempts,
                epsilon_ladder=ladder, normalize=v_normalization):
            if ladder is not None:
                out.epsilon_rungs.append({"degree": m, **rung.as_record()})
            V_proposal: Optional[GeneralizedForm] = rung.V_gf
            if rung.epsilon > 0.0:
                out.design_epsilon = float(rung.epsilon_effective)
                out.design_epsilon_requested = float(rung.epsilon)
            out.v_normalization = rung.normalization

            if V_proposal is None:
                if verbose:
                    print(f"[{spec.name}] m={m}: no SOS-feasible candidate"
                          + (f" at eps={rung.epsilon:g}" if ladder is not None else ""))
                if len(out.degree_attempts) == n_attempts_before:
                    # No exception was logged by the design step → an ordinary
                    # SOS infeasibility at this degree, not an error.
                    entry = {
                        "degree": m, "status": "no_certificate",
                        "message": "full-library SOS infeasible",
                    }
                    if ladder is not None:
                        entry["epsilon"] = float(rung.epsilon)
                        entry["message"] = (
                            f"full-library SOS not usable at eps={rung.epsilon:g} "
                            f"({rung.solver_status or 'infeasible'})")
                    out.degree_attempts.append(entry)
                n_attempts_before = len(out.degree_attempts)
                continue

            # Authoritative strict gate (standard SOS of V>0 and W>0 + Lemma-5 C¹;
            # with inclusion vertices, W is gated for every extremal dynamics).
            out.sdp_solves += 1
            cert = certify_discovered_gf(
                V_proposal, weights_r, m, spec.dynamics_gf, solver=gate_solver,
                strict_solver=not gate_fallback,
                dynamics_gf_vertices=getattr(spec, "dynamics_gf_vertices", None),
                disturbance_bound=getattr(spec, "disturbance_bound", None),
            )
            if not cert.surface_certified:
                entry = {
                    "degree": m, "status": "no_certificate",
                    "message": f"candidate not certified (reg={cert.regularity}, "
                               f"scope={cert.surface_scope})",
                }
                if ladder is not None:
                    entry["epsilon"] = float(rung.epsilon)
                out.degree_attempts.append(entry)
                if verbose:
                    print(f"[{spec.name}] m={m}: candidate not certified "
                          f"(reg={cert.regularity}, scope={cert.surface_scope})")
                n_attempts_before = len(out.degree_attempts)
                continue

            V = cert.V_gf if cert.V_gf is not None else V_proposal
            out.n_terms_dense = len(V.terms)

            # Optional sparsification: same library, same gate. The dense
            # certificate at this m is the prerequisite and the fallback. The
            # gate closure makes the sparsifier walk the support ladder and
            # return the smallest CERTIFIED support.
            if sparsify == "reweighted_l1":
                from ..verification.sparse_sos import design_sparse_lyapunov

                def _gate(V_candidate):
                    out.sdp_solves += 1
                    return certify_discovered_gf(
                        V_candidate, weights_r, m, spec.dynamics_gf, solver=gate_solver,
                        strict_solver=not gate_fallback,
                        dynamics_gf_vertices=getattr(spec, "dynamics_gf_vertices", None),
                        disturbance_bound=getattr(spec, "disturbance_bound", None),
                    )

                sres = design_sparse_lyapunov(
                    weights_r, m, spec.dynamics_gf, list(diag) + list(cross),
                    dynamics_gf_vertices=getattr(spec, "dynamics_gf_vertices", None),
                    solver=solver, gate_fn=_gate, verbose=verbose,
                )
                out.reweight_iters = len(sres.iterations)
                out.support_stable = sres.support_stable
                if sres.gated and sres.V_gf is not None:
                    scert = sres.gate_result
                    V = scert.V_gf if scert.V_gf is not None else sres.V_gf
                    cert = scert
                    method = "gf_sos_reweighted_l1"
                    out.sparsifier = "reweighted_l1"
                else:
                    out.sparsify_failed_reason = sres.message or "sparse design failed"
                if verbose and out.sparsify_failed_reason:
                    print(f"[{spec.name}] m={m}: sparsify kept dense certificate: "
                          f"{out.sparsify_failed_reason}")

            (out.gram_certificates, out.gram_bundle_fingerprint,
             out.gram_bundle_reason) = _adopt_design_bundle(
                rung.result, V, V_proposal,
                entries=rung.gram_certificates, refusal=rung.bundle_reason)

            out.certified = True
            out.path = "gf_sos"
            out.method = method
            out.claim_level = "lemma5_global_asymptotic_stability"
            out.V_gf = V
            out.coefficients = {f"alpha{i}": float(t.coefficient) for i, t in enumerate(V.terms)}
            out.degree_used = m
            out.n_terms = len(V.terms)
            out.c1_library_diag = len(diag)
            out.c1_library_cross = len(cross)
            out.c1_library_cross_full = len(cross_full)
            out.mu = tuple(int(x) for x in mu_min)
            out.regularity = cert.regularity
            out.surface_scope = cert.surface_scope
            out.gram_min_eig = getattr(cert, "gram_min_eig", None)
            out.gram_residual_l1 = getattr(cert, "gram_residual_l1", None)
            out.gram_cond = getattr(cert, "gram_cond", None)
            out.epsilon_used = getattr(cert, "epsilon_used", None)
            out.total_time = time.time() - t0
            out.message = (
                f"{method}: C¹ GF certified globally via Lemma 5 (m={m}, "
                f"{out.n_terms} terms from a C¹ library of {len(diag) + len(cross)}"
                + (f"; sparsified from {out.n_terms_dense} dense terms"
                   if out.sparsifier != "none" else "")
                + (f"; design eps {rung.epsilon:g}"
                   + (f" x V-scale {rung.normalization['factor']:g}"
                      if (rung.normalization or {}).get("applied") else "")
                   if ladder is not None else "") + ").")
            certified_entry = {
                "degree": m, "status": "certified",
                "message": "strict Lemma-5 gate passed",
            }
            if ladder is not None:
                certified_entry["epsilon"] = float(rung.epsilon)
            out.degree_attempts.append(certified_entry)
            return out

    # ---------------------------------------------------------------- honest negative
    out.certified = False
    out.path = "none"
    out.claim_level = "candidate"
    out.total_time = time.time() - t0
    out.message = (f"No GF-SOS certificate over degrees {out.degrees_tried} "
                   f"(paper machinery, μ-filtered full library).")
    return out
