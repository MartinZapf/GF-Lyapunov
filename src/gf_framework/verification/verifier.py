"""
Verification Module for the GF Framework.

Main entry point for rigorous analytical verification via SOS optimization
and Pólya's theorem. Provides proof certificates for V > 0 and W = -dV/dt > 0.

Two-stage verification:
1. Try Pólya (fast, complete for polynomial GFs)
2. Fall back to Joint SOS with weighted-sphere clearing for negative exponents
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
import time
import warnings

from gf_framework.interfaces import (
    SelectionResult,
    SystemSpec,
    VerificationConfig,
    Certificate,
)
from gf_framework.algebra.core import GeneralizedForm
from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf
from .polya import PolyaVerifier, verify_gf_polya, PolyaResult
from .joint_sos import (
    JointSOSOptimizer,
    JointSOSResult,
    verify_gf_sos_full,
    analyze_w_exponents,
    verify_hyperplane_continuity,
)
from .failure_typing import (
    FailureClass,
    classify_failure,
    stage_report,
    worst_failure_class,
)


def _default_solver_metadata(config: VerificationConfig) -> dict:
    """Build default solver metadata payload for certificates."""
    return {
        "solver_requested": str(config.solver),
        "solver_used": "",
        "solver_fallback_used": False,
        "solver_attempts": [],
        "solver_status": "",
        "gram_min_eig": None,
        "gram_residual_l1": None,
        "gram_cond": None,
        "epsilon_used": None,
        # Typed failure information (WP P3-02). Empty until a stage fails.
        "failure_class": "",
        "exception_type": "",
        "exception_message": "",
        "normalized_retry_attempts": [],
    }


def _type_failure(meta: dict, exc: Optional[BaseException] = None) -> dict:
    """Stamp a typed failure class onto a solver-metadata payload, in place.

    This is the single place a rejection acquires its class. Before P3-02 the
    caller returned the literal method string ``"failed"`` and nothing else, so
    solver infeasibility, ``solver_unknown``, an exception and an a-posteriori
    Gram rejection were indistinguishable downstream.
    """
    meta["failure_class"] = classify_failure(
        verified=False,
        solver_status=str(meta.get("solver_status", "") or ""),
        attempts=meta.get("solver_attempts") or [],
        exception=exc,
    )
    if exc is not None:
        meta["exception_type"] = type(exc).__name__
        meta["exception_message"] = str(exc)[:400]
    return meta


def _solver_metadata_from_result(result: object, config: VerificationConfig) -> dict:
    """Extract normalized solver metadata from SOS result objects."""
    base = _default_solver_metadata(config)
    if result is None:
        return base

    attempts = getattr(result, "solver_attempts", None)
    if attempts is None:
        attempts = []
    base.update(
        {
            "solver_status": str(getattr(result, "solver_status", "") or ""),
            "solver_used": str(getattr(result, "solver_used", "") or ""),
            "solver_fallback_used": bool(getattr(result, "solver_fallback_used", False)),
            "solver_attempts": [str(a) for a in attempts],
            "gram_min_eig": getattr(result, "gram_min_eig", None),
            "gram_residual_l1": getattr(result, "gram_residual_l1", None),
            "gram_cond": getattr(result, "gram_cond", None),
            "epsilon_used": getattr(result, "epsilon_used", None),
        }
    )
    return base


def _merge_gram_metadata(*metas: dict) -> dict:
    """Worst-case Gram diagnostics across SOS stages (V stage + one W stage per
    inclusion vertex).

    Worst case means the most negative Gram eigenvalue, the largest equality
    residual, and the smallest (weakest) strictness margin actually used.
    """
    def _pick(key: str, fn):
        vals = [m.get(key) for m in metas if m.get(key) is not None]
        return fn(vals) if vals else None

    return {
        "gram_min_eig": _pick("gram_min_eig", min),
        "gram_residual_l1": _pick("gram_residual_l1", max),
        "gram_cond": _pick("gram_cond", max),
        "epsilon_used": _pick("epsilon_used", lambda v: min(v, key=float)),
    }


def _merge_solver_metadata(primary: dict, fallback: dict) -> dict:
    """
    Pick the most informative solver metadata, preferring the primary payload.
    """
    if primary and (
        primary.get("solver_used")
        or primary.get("solver_attempts")
        or primary.get("solver_status")
    ):
        return primary
    return fallback


def verify_lyapunov(
    selection: SelectionResult,
    system_spec: SystemSpec,
    config: Optional[VerificationConfig] = None,
    verbose: bool = True
) -> Certificate:
    """
    Main entry point for Verification Module.

    Verifies that V > 0 and W = -dV/dt > 0 using Pólya theorem
    or SOS optimization as fallback.

    Args:
        selection: Selection result containing V_gf
        system_spec: System specification with dynamics
        config: Verification configuration
        verbose: Print progress

    Returns:
        Certificate with verification status and details
    """
    if config is None:
        config = VerificationConfig()

    V_gf = selection.V_gf

    if verbose:
        print("\n=== Verification Module ===")
        print(f"Verifying {V_gf.n_terms} terms")

    # Stage 1: Verify V > 0
    if verbose:
        print("\n--- Stage 1: Verifying V > 0 ---")

    v_result = _verify_positivity(V_gf, config, verbose)
    if len(v_result) == 3:  # Backward-compatible for monkeypatched tests
        V_verified, V_method, polya_p_V = v_result
        v_solver_meta = _default_solver_metadata(config)
    else:
        V_verified, V_method, polya_p_V, v_solver_meta = v_result

    if not V_verified:
        v_class = str(v_solver_meta.get("failure_class", "") or "") or classify_failure(
            solver_status=str(v_solver_meta.get("solver_status", "") or ""),
            attempts=v_solver_meta.get("solver_attempts") or [],
        )
        v_report = stage_report("V", verified=False, method=V_method, meta=v_solver_meta)
        return Certificate(
            V_verified=False,
            W_verified=False,
            V_method=V_method,
            W_method="skipped",
            V_gf=V_gf,
            solver_status=str(v_solver_meta.get("solver_status", "") or ""),
            solver_requested=str(v_solver_meta.get("solver_requested", "") or ""),
            solver_used=str(v_solver_meta.get("solver_used", "") or ""),
            solver_fallback_used=bool(v_solver_meta.get("solver_fallback_used", False)),
            solver_attempts=list(v_solver_meta.get("solver_attempts", [])),
            gram_min_eig=v_solver_meta.get("gram_min_eig"),
            gram_residual_l1=v_solver_meta.get("gram_residual_l1"),
            gram_cond=v_solver_meta.get("gram_cond"),
            V_failure_class=v_class,
            W_failure_class=FailureClass.SKIPPED,
            failure_class=v_class,
            stage_reports=[v_report],
            message=f"V verification failed [{v_class}]"
        )

    if verbose:
        print(f"  V > 0 VERIFIED via {V_method}")
        if polya_p_V is not None:
            print(f"  Pólya exponent: p = {polya_p_V}")

    # Stage 2: Compute and verify W = -dV/dt
    if verbose:
        print("\n--- Stage 2: Verifying W = -dV/dt > 0 ---")

    if system_spec.dynamics_gf is None:
        # Cannot verify W without dynamics
        return Certificate(
            V_verified=True,
            W_verified=False,
            V_method=V_method,
            W_method="no_dynamics",
            V_gf=V_gf,
            polya_p_V=polya_p_V,
            solver_status=str(v_solver_meta.get("solver_status", "") or ""),
            solver_requested=str(v_solver_meta.get("solver_requested", "") or ""),
            solver_used=str(v_solver_meta.get("solver_used", "") or ""),
            solver_fallback_used=bool(v_solver_meta.get("solver_fallback_used", False)),
            solver_attempts=list(v_solver_meta.get("solver_attempts", [])),
            V_failure_class=FailureClass.NONE,
            W_failure_class=FailureClass.SKIPPED,
            failure_class=FailureClass.SKIPPED,
            stage_reports=[
                stage_report("V", verified=True, method=V_method, meta=v_solver_meta)
            ],
            message="V verified, W skipped (no dynamics provided)"
        )

    # Differential-inclusion vertices: W = -V̇ must be certified for EVERY
    # extremal dynamics (nominal-only when no vertices are given). W is affine
    # in a matched disturbance, so endpoint vertices cover the whole interval.
    vertices = system_spec.dynamics_gf_vertices or [system_spec.dynamics_gf]

    W_gf = None
    W_verified, W_method, polya_p_W, clearing_power = True, "", None, 0
    certified_V_from_joint_sos = None
    w_solver_meta = _default_solver_metadata(config)
    w_metas = []
    # One (verified, method, meta) triple per vertex actually attempted, so the
    # record can say which vertex failed and why (P3-02).
    w_stage_info = []
    for v_idx, dyn_k in enumerate(vertices):
        # Compute dV/dt using GF-native Lie derivative
        dVdt_gf = compute_lie_derivative_gf(V_gf, dyn_k)

        # Negate to get W = -dV/dt (we want W > 0)
        W_gf_k = _negate_gf(dVdt_gf)
        if W_gf is None:
            W_gf = W_gf_k  # certificate payload: first vertex's W

        if verbose:
            tag = f" (vertex {v_idx + 1}/{len(vertices)})" if len(vertices) > 1 else ""
            print(f"  W has {W_gf_k.n_terms} terms{tag}")

        # Check for negative exponents in W (requires weighted-sphere clearing)
        mu = V_gf.mu_exponents
        w_analysis = analyze_w_exponents(W_gf_k.terms, mu)

        if w_analysis['has_negative']:
            if verbose:
                print(f"  W has negative y-exponents (min={w_analysis['min_y_exp']:.2f})")
                print(f"  Using weighted-sphere SOS clearing (k={w_analysis['clearing_power']})")

        # Verify W > 0
        # NOTE: In weighted-sphere mode, joint SOS may return a new V with
        # certified coefficients for the same term structure.
        w_result = _verify_w_positivity(W_gf_k, V_gf, system_spec, config, w_analysis, verbose)
        if len(w_result) == 5:  # Backward-compatible for monkeypatched tests
            W_verified_k, W_method, polya_p_W, clearing_power_k, certified_V_from_joint_sos = w_result
            w_solver_meta = _default_solver_metadata(config)
        else:
            (
                W_verified_k,
                W_method,
                polya_p_W,
                clearing_power_k,
                certified_V_from_joint_sos,
                w_solver_meta,
            ) = w_result
        w_metas.append(w_solver_meta)
        clearing_power = max(clearing_power, clearing_power_k)

        # The weighted-sphere redesign path returns a NEW V certified against a
        # single vertex; that V is not proven for the other vertices, so it is
        # only admissible in the nominal (single-vertex) case.
        if certified_V_from_joint_sos is not None and len(vertices) > 1:
            if verbose:
                print("  Joint-SOS V redesign is incompatible with inclusion "
                      "vertices — rejecting honestly")
            W_verified_k = False
            certified_V_from_joint_sos = None
            # Not a solver outcome: we refused an otherwise feasible solve.
            w_solver_meta["failure_class"] = FailureClass.POLICY_REJECTED

        w_stage_info.append((bool(W_verified_k), W_method, w_solver_meta, v_idx))

        if not W_verified_k:
            W_verified = False
            break

    # If joint SOS produced a certified V with updated coefficients, use it —
    # but ONLY after re-running the complete Stage-1 positivity gate on it.
    #
    # Regression from the 2026-09-05 external scientific review, finding 3: the
    # weighted-sphere branch could replace V_gf in the payload while
    # V_verified=True still referred to the ORIGINAL candidate checked in stage
    # 1. The invariant enforced here is: the V in the returned Certificate is
    # the V whose positivity was established.
    v_redesign_report = None
    v_redesign_meta = None
    if W_verified and certified_V_from_joint_sos is not None:
        if verbose:
            print("  Joint SOS returned a redesigned V — re-running the V "
                  "positivity gate on it")

        rv_result = _verify_positivity(certified_V_from_joint_sos, config, verbose)
        if len(rv_result) == 3:  # Backward-compatible for monkeypatched tests
            rv_verified, rv_method, rv_polya_p = rv_result
            rv_meta = _default_solver_metadata(config)
        else:
            rv_verified, rv_method, rv_polya_p, rv_meta = rv_result

        if rv_verified:
            if verbose:
                print(f"  Redesigned V re-verified via {rv_method}")

            V_gf = certified_V_from_joint_sos
            V_method = rv_method
            polya_p_V = rv_polya_p

            # Recompute W from the certified V so certificate payload is coherent.
            dVdt_gf = compute_lie_derivative_gf(V_gf, system_spec.dynamics_gf)
            W_gf = _negate_gf(dVdt_gf)

            v_redesign_meta = rv_meta
            v_redesign_report = stage_report(
                "V_redesigned", verified=True, method=rv_method, meta=rv_meta
            )
            v_redesign_report["v_redesigned"] = True
            v_redesign_report["note"] = (
                "W stage returned a redesigned V; the complete V positivity gate "
                "was re-run on it and passed. Certificate payload (V_gf, W_gf, "
                "regularity, surface scope) is derived from the redesigned V."
            )
        else:
            if verbose:
                print("  Redesigned V FAILED the V positivity gate — refusing "
                      "the weighted-sphere result honestly")

            # Keep the ORIGINAL V (the one whose positivity was established) and
            # its W; report the W stage as refused. Not a solver outcome: the
            # solve was feasible, we refused its output (same treatment as the
            # multi-vertex refusal above).
            W_verified = False
            certified_V_from_joint_sos = None
            rv_meta = dict(rv_meta)
            rv_meta["failure_class"] = FailureClass.POLICY_REJECTED
            if w_stage_info:
                last_ok, last_method, last_meta, last_idx = w_stage_info[-1]
                last_meta = dict(last_meta)
                last_meta["failure_class"] = FailureClass.POLICY_REJECTED
                w_stage_info[-1] = (False, last_method, last_meta, last_idx)
            v_redesign_report = stage_report(
                "V_redesigned", verified=False, method=rv_method, meta=rv_meta
            )
            v_redesign_report["v_redesigned"] = True
            v_redesign_report["note"] = (
                "W stage returned a redesigned V that did NOT pass the V "
                "positivity gate. The certificate is reported unverified and the "
                "payload keeps the original, actually-verified V."
            )

    if verbose:
        if W_verified:
            print(f"  W > 0 VERIFIED via {W_method}")
            if clearing_power > 0:
                print(f"  Clearing power: k = {clearing_power}")
        else:
            print(f"  W > 0 FAILED via {W_method}")

    # Optional numerical sanity check near coordinate hyperplanes.
    # The formal certificate is on open orthants (Lemma 5).
    hyperplane_warning = ""
    if W_verified and clearing_power > 0:
        hp_valid, hp_msg = verify_hyperplane_continuity(
            W_gf, system_spec.weights_r, verbose=verbose
        )
        if not hp_valid:
            hyperplane_warning = f" WARNING: {hp_msg}"
            warnings.warn(
                f"Hyperplane continuity sanity check failed: {hp_msg}. "
                "Certificate remains open-orthant SOS; inspect near-hyperplane behavior."
            )
            if verbose:
                print(f"  {hyperplane_warning}")

    # Typed failure classes (P3-02). "none" when the stage passed.
    V_failure_class = FailureClass.NONE
    if not V_verified:
        V_failure_class = str(v_solver_meta.get("failure_class", "") or "") or classify_failure(
            solver_status=str(v_solver_meta.get("solver_status", "") or ""),
            attempts=v_solver_meta.get("solver_attempts") or [],
        )
    stage_reports = [
        stage_report("V", verified=V_verified, method=V_method, meta=v_solver_meta)
    ]
    if v_redesign_report is not None:
        # A failed re-verification lands in the W failure classes below, which is
        # correct: the W stage is what proposed the redesigned V.
        stage_reports.append(v_redesign_report)
    for w_ok, w_method_k, w_meta_k, w_vertex in w_stage_info:
        stage_reports.append(
            stage_report(
                "W", verified=w_ok, method=w_method_k, meta=w_meta_k,
                vertex_index=w_vertex,
            )
        )
    w_failure_classes = [
        r["failure_class"] for r in stage_reports[1:]
        if not r["verified"] and r["failure_class"] != FailureClass.NONE
    ]
    if W_verified:
        W_failure_class = FailureClass.NONE
    elif w_failure_classes:
        W_failure_class = worst_failure_class(w_failure_classes)
    else:
        W_failure_class = FailureClass.SOLVER_UNKNOWN
    if V_verified and W_verified:
        overall_failure_class = FailureClass.NONE
    elif not V_verified:
        overall_failure_class = V_failure_class
    else:
        overall_failure_class = W_failure_class

    base_message = _build_message(
        V_verified, W_verified, V_method, W_method,
        V_failure_class=V_failure_class, W_failure_class=W_failure_class,
    )
    if W_verified and len(vertices) > 1:
        full_suffix = f" (robust: W > 0 for all {len(vertices)} inclusion vertices)"
    else:
        full_suffix = ""
    full_message = base_message + full_suffix + hyperplane_warning
    solver_meta = _merge_solver_metadata(w_solver_meta, v_solver_meta)
    # The redesigned-V re-verification is a real proof stage, so its Gram
    # diagnostics take part in the worst-case merge (review 2026-09-05, f. 3).
    redesign_metas = [v_redesign_meta] if v_redesign_meta is not None else []
    gram_meta = _merge_gram_metadata(v_solver_meta, *redesign_metas, *w_metas)

    # Switching-surface scope via the Sanchez-Moreno bridge — Lemma 5 (Filippov),
    # NOT Clarke (Clarke belongs to the out-of-scope discretization setting).
    # Lemma 5: if V ∈ C¹ is PD and ∂V/∂x·f < 0 off the discontinuity set S of f
    # (i.e. on the open hyperoctants, which the exact V>0 / W>0 proofs cover), then
    # the origin is asymptotically stable. C¹-ness of V makes ∇V continuous across
    # the surfaces, so no separate switching-surface analysis is needed; moreover a
    # C¹ V yields W with non-negative exponents, so no clearing is required either.
    smooth = classify_smoothness(V_gf)
    regularity = smooth["regularity"]
    if not (V_verified and W_verified):
        surface_scope = "open_orthant_only"
        surface_certified = False
    elif regularity == "C1":
        # Full continuous-time asymptotic-stability proof (Lemma 5).
        surface_scope = "lemma5_filippov"
        surface_certified = True
    else:
        # Non-C¹ V is outside the continuous-time Lemma-5 method; a rigorous claim
        # would require non-smooth analysis (the out-of-scope discretization route).
        surface_scope = "non_c1_out_of_scope"
        surface_certified = False

    return Certificate(
        V_verified=V_verified,
        W_verified=W_verified,
        V_method=V_method,
        W_method=W_method,
        V_gf=V_gf,
        W_gf=W_gf,
        regularity=regularity,
        surface_scope=surface_scope,
        surface_certified=surface_certified,
        solver_status=str(solver_meta.get("solver_status", "") or "completed"),
        solver_requested=str(solver_meta.get("solver_requested", "") or ""),
        solver_used=str(solver_meta.get("solver_used", "") or ""),
        solver_fallback_used=bool(solver_meta.get("solver_fallback_used", False)),
        solver_attempts=list(solver_meta.get("solver_attempts", [])),
        gram_min_eig=gram_meta.get("gram_min_eig"),
        gram_residual_l1=gram_meta.get("gram_residual_l1"),
        gram_cond=gram_meta.get("gram_cond"),
        epsilon_used=gram_meta.get("epsilon_used"),
        n_vertices=len(vertices),
        clearing_power=clearing_power,
        polya_p_V=polya_p_V,
        polya_p_W=polya_p_W,
        V_failure_class=V_failure_class,
        W_failure_class=W_failure_class,
        failure_class=overall_failure_class,
        stage_reports=stage_reports,
        message=full_message
    )


def classify_smoothness(V_gf: GeneralizedForm) -> dict:
    """Classify the regularity of a GF Lyapunov candidate at coordinate hyperplanes.

    This decides whether the Sanchez-Moreno continuous-time bridge (Lemma 5,
    Filippov) applies. Lemma 5 requires V ∈ C¹ (restriction R2: m > maxᵢ rᵢ;
    abs-exponents ∈ {0}∪(1,∞); signed-exponents ≥ 1). Only the ``C1`` class is in
    scope; the non-C¹ classes are reported honestly as out of scope.

      * ``C1``           — V ∈ C¹: every term is continuously differentiable, so
                           ∇V is continuous across the switching surfaces and
                           Lemma 5 gives asymptotic stability with no separate
                           surface analysis.
      * ``lipschitz_kink`` — V is locally Lipschitz but non-smooth (``|x_i|^1``
                           kinks). Outside Lemma 5 (V ∉ C¹); a rigorous claim would
                           need non-smooth analysis, which is out of scope here.
      * ``sub_lipschitz`` — V contains ``|x_i|^{ρ}`` / ``⌈x_i⌋^{ρ}`` with 0<ρ<1
                           (e.g. STA's ``|x_1|^{1/2}``): V is not even locally
                           Lipschitz. Out of scope for the continuous-time GF method.
      * ``discontinuous`` — V has a ``sign(x_i)`` term (should not occur for a
                           Lyapunov candidate).

    Returns a dict with the overall ``regularity`` (worst case) and the per-axis
    ``kink_axes`` / ``sub_lipschitz_axes``.
    """
    # Per-factor regularity comes from the canonical algebra-level predicate so the
    # certification gate and the discovery basis (enumerate.py c1_only) cannot drift.
    from gf_framework.algebra.enumerate import classify_term_regularity

    order = {"C1": 0, "lipschitz_kink": 1, "sub_lipschitz": 2, "discontinuous": 3}
    worst = "C1"
    kink_axes, sub_axes, disc_axes = set(), set(), set()

    for term in V_gf.terms:
        if float(getattr(term, "coefficient", 0.0)) == 0.0:
            continue
        for i, (exp, typ) in enumerate(zip(term.exponents, term.term_types)):
            cls = classify_term_regularity(exp, typ)
            if cls == "discontinuous":
                disc_axes.add(i)
            elif cls == "sub_lipschitz":
                sub_axes.add(i)
            elif cls == "lipschitz_kink":
                kink_axes.add(i)
            if order[cls] > order[worst]:
                worst = cls

    return {
        "regularity": worst,
        "is_C1": worst == "C1",
        "kink_axes": sorted(kink_axes),
        "sub_lipschitz_axes": sorted(sub_axes),
        "discontinuous_axes": sorted(disc_axes),
    }


def certify_discovered_gf(
    V_gf: GeneralizedForm,
    weights_r,
    degree_m: int,
    dynamics_gf,
    solver: str = "MOSEK",
    strict_solver: bool = False,
    dynamics_gf_vertices=None,
    disturbance_bound: float = None,
    verbose: bool = False,
) -> Certificate:
    """Authoritative standard-SOS re-certification of a discovered GF candidate.

    This is the *honest gate*: the emitted publication claim must be derived from
    here, never from an internal discovery flag. It runs the full standard-SOS
    proof of V > 0 and W = -dV/dt > 0 (per-octant associated forms with a strict
    ε-margin, Gram P ⪰ 0; the SDP solver is the accepted oracle — Sanchez-Moreno
    2018 §5.3) guarded by the necessary ``min_V_on_sphere`` PD pre-check, plus the
    Lemma-5 (Filippov) C¹ surface classification.

    The returned Certificate's ``surface_certified=True`` means a *complete*
    continuous-time asymptotic-stability proof: V ∈ C¹ (restriction R2), V > 0,
    and V̇ < 0 off the discontinuity set. Non-C¹ V is reported honestly as
    out of scope (``surface_scope='non_c1_out_of_scope'``).

    Thin adapter over :func:`verify_lyapunov`; kept separate so discovery scripts
    and tests share one authoritative entry point.

    ``strict_solver=True`` pins the certification to exactly the requested
    solver (no silent fallback chain). Independent re-verification — e.g.
    ``review/verify_c1_certificates.py`` under CLARABEL — MUST set this,
    otherwise a CLARABEL failure would silently fall back to MOSEK and the
    "independent solver" claim would be vacuous.
    """
    sel = SelectionResult(
        active_terms=[], coefficients={}, V_gf=V_gf,
        numerical_valid=True, max_V_violation=0.0, max_W_violation=0.0,
    )
    spec = SystemSpec(
        weights_r=tuple(int(r) for r in weights_r),
        degree_m=int(degree_m),
        dynamics_gf=dynamics_gf,
        dynamics_gf_vertices=dynamics_gf_vertices,
    )
    # Construct (not mutate) so __post_init__ ties allow_solver_fallback to
    # strict_solver consistently.
    cfg = VerificationConfig(solver=solver, strict_solver=strict_solver)
    cert = verify_lyapunov(sel, spec, cfg, verbose=verbose)
    if disturbance_bound is not None:
        cert.disturbance_bound = float(disturbance_bound)
    return cert


def _has_cross_terms(gf: GeneralizedForm) -> bool:
    """
    Check if a GF contains any cross-terms (multi-variable terms).

    Cross-terms have sign-dependent structures that change across octants,
    making them non-polynomial in y-space. Pólya's theorem only applies to
    polynomials, so it cannot verify GFs with cross-terms.

    Args:
        gf: Generalized Form to check

    Returns:
        True if any term involves multiple variables
    """
    for term in gf.terms:
        # Count non-zero exponents in this term
        nonzero_count = sum(1 for exp in term.exponents if exp != 0)
        if nonzero_count > 1:
            return True
    return False


def _verify_positivity(
    gf: GeneralizedForm,
    config: VerificationConfig,
    verbose: bool
) -> Tuple[bool, str, Optional[int], dict]:
    """
    Verify that a GF is positive definite.

    Routing logic:
    - For diagonal-only GFs: Try Pólya first (fast, exact), then SOS fallback
    - For GFs with cross-terms: Skip Pólya (will fail), go directly to SOS

    This optimization avoids wasted computation since Pólya cannot verify
    cross-term GFs (they have sign-dependent structures across octants).

    Args:
        gf: Generalized Form to verify
        config: Verification configuration
        verbose: Print progress

    Returns:
        (verified, method, polya_p, solver_metadata)
    """
    # Check for cross-terms to decide verification strategy
    has_cross = _has_cross_terms(gf)

    def _run_sos(target_gf: GeneralizedForm):
        return verify_gf_sos_full(
            target_gf,
            epsilon=config.epsilon,
            solver=config.solver,
            strict_solver=config.strict_solver,
            allow_solver_fallback=config.allow_solver_fallback,
            verbose=verbose,
        )

    if has_cross:
        if verbose:
            print("  GF has cross-terms, skipping Pólya (will fail for non-polynomial)")
            print("  Using SOS verification directly...")

        # Go directly to SOS for cross-term GFs
        fail_meta = _default_solver_metadata(config)
        exc: Optional[BaseException] = None
        try:
            sos_result = _run_sos(gf)
            fail_meta = _solver_metadata_from_result(sos_result, config)

            if sos_result.feasible:
                return True, "sos_cross_term", None, fail_meta

            if verbose:
                print(f"  SOS failed: {sos_result.message}")

            # Numerical fallback: normalize coefficients and retry SOS.
            # This is solver-preconditioning only (same term structure).
            gf_norm, scale = _normalize_gf_coefficients(gf)
            if scale != 1.0:
                if verbose:
                    print(f"  Retrying SOS with normalized coefficients (scale={scale:.3e})...")
                sos_norm = _run_sos(gf_norm)
                if sos_norm.feasible:
                    return True, "sos_cross_term_normalized", None, _solver_metadata_from_result(sos_norm, config)
                fail_meta["normalized_retry_attempts"] = [
                    str(a) for a in (sos_norm.solver_attempts or [])
                ]
                if verbose:
                    print(f"  Normalized SOS failed: {sos_norm.message}")

        except Exception as e:
            exc = e
            if verbose:
                print(f"  SOS error [{type(e).__name__}]: {e}")

        # Failure keeps the attempt trace AND its typed class so re-verification
        # artifacts can show WHY the stage failed (not just that it did).
        _type_failure(fail_meta, exc)
        return False, "failed", None, fail_meta

    # Diagonal-only GF: Try Pólya first (faster, exact for polynomials)
    polya_verifier = PolyaVerifier(
        max_p=config.max_polya_p,
        tol=config.polya_tol
    )

    if verbose:
        print("  Trying Pólya verification (diagonal-only GF)...")

    polya_result = polya_verifier.verify(gf)

    if polya_result.verified:
        return True, "polya", polya_result.polya_p, _default_solver_metadata(config)

    if verbose and not polya_result.verified:
        print(f"  Pólya failed: {polya_result.message}")

    # Try SOS fallback
    if verbose:
        print("  Trying SOS verification...")

    fail_meta = _default_solver_metadata(config)
    exc: Optional[BaseException] = None
    try:
        sos_result = _run_sos(gf)
        fail_meta = _solver_metadata_from_result(sos_result, config)

        if sos_result.feasible:
            return True, "sos", None, fail_meta

        if verbose:
            print(f"  SOS failed: {sos_result.message}")

        gf_norm, scale = _normalize_gf_coefficients(gf)
        if scale != 1.0:
            if verbose:
                print(f"  Retrying SOS with normalized coefficients (scale={scale:.3e})...")
            sos_norm = _run_sos(gf_norm)
            if sos_norm.feasible:
                return True, "sos_normalized", None, _solver_metadata_from_result(sos_norm, config)
            fail_meta["normalized_retry_attempts"] = [
                str(a) for a in (sos_norm.solver_attempts or [])
            ]
            if verbose:
                print(f"  Normalized SOS failed: {sos_norm.message}")

    except Exception as e:
        exc = e
        if verbose:
            print(f"  SOS error [{type(e).__name__}]: {e}")

    _type_failure(fail_meta, exc)
    return False, "failed", None, fail_meta


def _verify_w_positivity(
    W_gf: GeneralizedForm,
    V_gf: GeneralizedForm,
    system_spec: SystemSpec,
    config: VerificationConfig,
    w_analysis: dict,
    verbose: bool
) -> Tuple[bool, str, Optional[int], int, Optional[GeneralizedForm], dict]:
    """
    Verify that W = -dV/dt is positive definite.

    Handles negative exponents via weighted-sphere clearing.

    Args:
        W_gf: Negative Lie derivative (already negated, should be > 0)
        V_gf: Lyapunov candidate
        system_spec: System specification
        config: Verification configuration
        w_analysis: Analysis of W exponents
        verbose: Print progress

    Returns:
        (verified, method, polya_p, clearing_power, certified_V_gf, solver_metadata)
        where certified_V_gf is populated when joint SOS solves coefficients.
    """
    clearing_power = 0
    solver_meta = _default_solver_metadata(config)

    if not w_analysis['has_negative']:
        # Standard verification (no clearing needed)
        polya_verifier = PolyaVerifier(
            max_p=config.max_polya_p,
            tol=config.polya_tol
        )

        if verbose:
            print("  Trying Pólya verification for W...")

        polya_result = polya_verifier.verify(W_gf)

        if polya_result.verified:
            return True, "polya", polya_result.polya_p, 0, None, solver_meta

        # Try SOS
        if verbose:
            print(f"  Pólya failed: {polya_result.message}")
            print("  Trying SOS verification for W...")

        exc: Optional[BaseException] = None
        try:
            sos_result = verify_gf_sos_full(
                W_gf,
                epsilon=config.epsilon,
                solver=config.solver,
                strict_solver=config.strict_solver,
                allow_solver_fallback=config.allow_solver_fallback,
                verbose=verbose,
            )
            # Keep the attempt trace even on failure so re-verification
            # artifacts can show WHY the W stage failed.
            solver_meta = _solver_metadata_from_result(sos_result, config)

            if sos_result.feasible:
                return True, "sos", None, 0, None, solver_meta

        except Exception as e:
            exc = e
            if verbose:
                print(f"  SOS error [{type(e).__name__}]: {e}")

        _type_failure(solver_meta, exc)
        return False, "failed", None, 0, None, solver_meta

    # Need weighted-sphere clearing for negative exponents
    clearing_power = w_analysis['clearing_power']

    if verbose:
        print(f"  W has negative exponents, using weighted-sphere SOS (k={clearing_power})")

    # Use the joint SOS optimizer with weighted-sphere clearing
    ws_exc: Optional[BaseException] = None
    try:
        from gf_framework.algebra.enumerate import TermSpec
        from gf_framework.verification.joint_sos import JointSOSOptimizer

        optimizer = JointSOSOptimizer(
            weights_r=system_spec.weights_r,
            epsilon=config.epsilon,
            solver=config.solver,
            strict_solver=config.strict_solver,
            allow_solver_fallback=config.allow_solver_fallback,
            verbose=verbose,
        )

        # Convert V_gf terms to TermSpecs
        active_terms = [
            TermSpec(exponents=t.exponents, term_types=t.term_types)
            for t in V_gf.terms
        ]

        # Run joint verification with dynamics
        result = optimizer.find_lyapunov_with_dynamics(
            degree_m=V_gf.degree_m,
            dynamics_gf=system_spec.dynamics_gf,
            active_terms=active_terms,
            max_denominator=6
        )

        if result.feasible:
            return (
                True,
                "sos_weighted_sphere",
                None,
                clearing_power,
                result.V_gf,
                _solver_metadata_from_result(result, config),
            )

        # A failed weighted-sphere solve used to drop its metadata entirely, so
        # the record could not say WHY (P3-02). Keep it.
        solver_meta = _solver_metadata_from_result(result, config)

        if verbose:
            print(f"  Weighted-sphere SOS failed: {result.message}")

    except Exception as e:
        ws_exc = e
        if verbose:
            print(f"  Weighted-sphere SOS error [{type(e).__name__}]: {e}")

    # NO numerical fallback - formal certificates only
    # If SOS/Pólya fail, we report failure (not "numerical success")
    _type_failure(solver_meta, ws_exc)
    return False, "sos_failed", None, clearing_power, None, solver_meta


def _negate_gf(gf: GeneralizedForm) -> GeneralizedForm:
    """Negate all coefficients of a GF."""
    from gf_framework.algebra.core import GFTerm

    new_terms = [
        GFTerm(
            coefficient=-t.coefficient,
            exponents=t.exponents,
            term_types=t.term_types
        )
        for t in gf.terms
    ]

    return GeneralizedForm(
        degree_m=gf.degree_m,
        weights_r=gf.weights_r,
        terms=new_terms
    )


def _normalize_gf_coefficients(gf: GeneralizedForm) -> Tuple[GeneralizedForm, float]:
    """
    Normalize GF coefficients by max |coef| for solver conditioning.

    Returns:
        (normalized_gf, scale) where normalized_gf = gf / scale and scale >= 1e-300.
        If no normalization is needed, returns (gf, 1.0).
    """
    if gf is None or not gf.terms:
        return gf, 1.0

    max_abs = max(abs(float(t.coefficient)) for t in gf.terms)
    if max_abs <= 0.0:
        return gf, 1.0
    if np.isclose(max_abs, 1.0):
        return gf, 1.0

    from gf_framework.algebra.core import GFTerm

    scaled_terms = [
        GFTerm(
            coefficient=float(t.coefficient) / max_abs,
            exponents=t.exponents,
            term_types=t.term_types,
        )
        for t in gf.terms
    ]
    return GeneralizedForm(degree_m=gf.degree_m, weights_r=gf.weights_r, terms=scaled_terms), float(max_abs)


def _build_message(
    V_verified: bool,
    W_verified: bool,
    V_method: str,
    W_method: str,
    V_failure_class: str = "",
    W_failure_class: str = "",
) -> str:
    """Build human-readable status message.

    P3-02: a failing stage now names its typed failure class in square
    brackets. The old text said "W failed via failed", which is compatible with
    solver infeasibility, a solver "unknown", a raised exception and a Gram
    rejected by ``_validate_gram_certificate`` alike.
    """
    def _tag(cls: str) -> str:
        cls = str(cls or "").strip()
        return f" [{cls}]" if cls and cls != FailureClass.NONE else ""

    if V_verified and W_verified:
        return f"Fully verified: V via {V_method}, W via {W_method}"
    elif V_verified:
        return (
            f"Partial: V verified via {V_method}, "
            f"W failed via {W_method}{_tag(W_failure_class)}"
        )
    else:
        return f"Failed: V not verified via {V_method}{_tag(V_failure_class)}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def verify_v_only(
    V_gf: GeneralizedForm,
    config: Optional[VerificationConfig] = None,
    verbose: bool = True
) -> Tuple[bool, str, Optional[int]]:
    """
    Verify only V > 0 (without dynamics).

    Args:
        V_gf: Lyapunov candidate
        config: Verification configuration
        verbose: Print progress

    Returns:
        (verified, method, polya_p)
    """
    if config is None:
        config = VerificationConfig()

    verified, method, polya_p, _solver_meta = _verify_positivity(V_gf, config, verbose)
    return verified, method, polya_p


def quick_verify(
    V_gf: GeneralizedForm,
    dynamics_gf: Optional[List[GeneralizedForm]] = None,
    weights_r: Optional[Tuple[int, ...]] = None
) -> Certificate:
    """
    Quick verification with default settings.

    Args:
        V_gf: Lyapunov candidate
        dynamics_gf: Optional dynamics for W verification
        weights_r: Weights (defaults to V_gf.weights_r)

    Returns:
        Certificate
    """
    if weights_r is None:
        weights_r = V_gf.weights_r

    spec = SystemSpec(
        weights_r=weights_r,
        degree_m=V_gf.degree_m,
        dynamics_gf=dynamics_gf
    )

    selection = SelectionResult(
        active_terms=[],
        coefficients={},
        V_gf=V_gf,
        numerical_valid=True,
        max_V_violation=0.0,
        max_W_violation=0.0
    )

    return verify_lyapunov(
        selection,
        spec,
        config=VerificationConfig(),
        verbose=False
    )


__all__ = [
    'verify_lyapunov',
    'verify_v_only',
    'quick_verify',
]
