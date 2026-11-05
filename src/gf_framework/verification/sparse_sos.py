"""Sparse-by-design SOS via reweighted ℓ₁ (Candès–Wakin–Boyd 2008).

The joint V+W SOS designer (``JointSOSOptimizer.find_lyapunov_with_dynamics``)
is linear in the term coefficients α, so sparse certificate design is a convex
program: minimize the weighted ℓ₁ norm Σⱼ wⱼ|αⱼ| of the CROSS-term
coefficients subject to the full V+W SOS feasibility constraints, and iterate
the weights wⱼ ← nⱼ/(|αⱼ|+δ) until the support stabilizes. The diagonal
coefficients stay unpenalized — they are pinned by the designer's scale
normalization (α_diag₀ = 1) and ε lower bound, which is exactly what makes the
ℓ₁ values comparable across iterations.

Atom normalization (the fair-ℓ₁ fix): each term's penalty weight starts from
the Euclidean norm nⱼ of its stacked associated-form coefficient column over
all octants — V columns AND W columns, since the W columns carry the dynamics
gains where scale unfairness actually lives.

Weighting arms: the initial weights are a design choice, not part of the proof
(the gate decides either way). ``SparseDesignConfig.weighting`` selects between
``"normalized"`` (the shipped atom scales, the default) and ``"uniform"``
(wⱼ = 1). Measured on the 12-system matrix the two arms give identical supports
on sta, twisting and levant3d; on pidsmc the uniform arm found 8 terms where
the atom normalization found 9. Running both and keeping the smaller GATING
support therefore costs minutes and can only help. The arm actually used is
recorded in ``SparseDesignResult.weighting``.

Two-stage design (the soundness guard): the ℓ₁ optimum hugs the feasibility
boundary, so after the support stabilizes the surviving terms are re-solved
WITHOUT the ℓ₁ objective (plain feasibility at the designer's standard ε
margin). Only the stage-2 candidate is handed to the strict gate
(``certify_discovered_gf``) — the designer proposes, the gate decides, exactly
the trust boundary of the dense pipeline.

μ-preservation guard (``config.mu_guard``, default ``"reject"``): a support can
drop the terms that pin one axis's exponent denominator, which lowers the V-only
μ that every downstream V-only path recomputes. The measured case is pidsmc
m = 6, where the 9-term ℓ₁ support takes μ from (6,4,2) to (6,2,2), breaking
proportionality to r; the float gate still certifies it but the exact phase
refuses target assembly. Such a support is now skipped exactly like a
gate-rejected one, with a typed reason in ``rejected``. See ``mu_guard``.

Backward elimination (``config.backward_elimination``, **off by default**): an
optional post-rl1 pass that drops the surviving cross terms one at a time,
smallest |coefficient| first, with a full design re-solve at every drop. A drop
is kept only when **all four** of these hold: the μ guard passes on the reduced
support, the design re-solves feasibly, the caller's gate certifies the new
candidate, and the **exact charge/budget ratio stays below
``config.backward_max_charge_ratio``** (default 1e-3). The last condition is the
headroom guard, and it is the reason the pass ships off by default: measured on
the levant3d 2003-print δ₀ = 1 row, going 16 → 14 terms bought two terms at 11.0x
of the exact charge like for like (3.805e-05 → 4.175e-04 under the FINDINGS
protocol, both supports freshly designed; the earlier 287x priced the published
16-term vector, 3.40e-06, against a 14-term charge of 9.75e-04 that did not
reproduce, see ``docs/papers/big_draft/DO_NOT_PURSUE.md`` DNP-12), and the
analogous 25 → 23 reduction on the 2-variant row lost the strict MOSEK gate
outright. Parsimony past the rl1 support is paid for in margin, so the caller has
to ask for it and the record carries the price. Every elimination step is recorded with a typed outcome
(``kept``, ``refused_mu``, ``refused_headroom``, ``infeasible``,
``gate_rejected``, ``charge_unavailable``, ``budget_exhausted``) and the measured
charge ratio. Decision D-k (owner's): no backward-eliminated row is adopted as a
published certificate on this evidence.

The charge comes from the exact gate run on the candidate's OWN design Grams
(``exact_phase.run_exact_phase(..., gram_bundle=design_result)``), which needs a
MOSEK-direct design with ``capture_design_grams``. No float re-solve, and in
particular no CVXPY solve, runs on that route. A candidate whose charge cannot be
measured is REFUSED (``charge_unavailable``), never kept on the benefit of the
doubt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..algebra.core import GeneralizedForm
from ..algebra.enumerate import TermSpec
from .joint_sos import (
    JointSOSOptimizer,
    JointSOSResult,
    build_v_coefficient_matrix,
    compute_mu_exponents_with_dynamics,
    get_hyperoctant_signs,
    get_monomial_basis,
)
from .mu_guard import (
    check_mu_preservation,
    design_mu,
    resolve_mu_guard,
    specs_from_gf,
    v_only_mu,
)


# The two initial-weight arms of the reweighted-ℓ₁ loop. Both are design
# heuristics; the gate decides in either case.
WEIGHTING_ARMS = ("normalized", "uniform")

# Drop orderings of the backward-elimination pass. Measured (FINDINGS §3.5): the
# two converged to the identical support in all 12 runs, so ``abs_coef`` — the
# cheap one, no atom scales needed — is the default.
BACKWARD_ORDERS = ("abs_coef", "coef_x_scale")

# Typed outcomes of ONE elimination step (one candidate drop).
BWD_KEPT = "kept"                            # all four conditions held
BWD_REFUSED_MU = "refused_mu"                # μ guard; no solve was spent
BWD_REFUSED_HEADROOM = "refused_headroom"    # exact charge ratio over the bound
BWD_INFEASIBLE = "infeasible"                # design re-solve infeasible
BWD_GATE_REJECTED = "gate_rejected"          # design feasible, float gate refused
BWD_CHARGE_UNAVAILABLE = "charge_unavailable"  # no measurable charge ⇒ refused
BWD_BUDGET_EXHAUSTED = "budget_exhausted"    # wall-clock budget hit mid-pass
BACKWARD_OUTCOMES = (
    BWD_KEPT, BWD_REFUSED_MU, BWD_REFUSED_HEADROOM, BWD_INFEASIBLE,
    BWD_GATE_REJECTED, BWD_CHARGE_UNAVAILABLE, BWD_BUDGET_EXHAUSTED,
)


@dataclass
class SparseDesignConfig:
    """Knobs of the reweighted-ℓ₁ loop (defaults follow the exam study)."""
    max_reweight_iters: int = 8
    # δⱼ = delta_rel · max_cross|α| (scale-free; the exam used absolute 1e-3
    # under its cᵀα = 1 normalization).
    delta_rel: float = 1e-3
    # Cap wⱼ ≤ w_max_factor · nⱼ so weights cannot blow up as αⱼ → 0.
    w_max_factor: float = 1e6
    # |αⱼ| < support_threshold_rel · max_cross|α| ⇒ term counted inactive.
    support_threshold_rel: float = 1e-5
    # Stop when the support is unchanged for this many consecutive iterations.
    stable_window: int = 2
    # Initial weights = atom scales (fair ℓ₁); False ⇒ uniform weights.
    # Legacy knob, kept for the existing callers and grid studies. It is an
    # alias of ``weighting``: normalize_atoms=False means weighting="uniform".
    normalize_atoms: bool = True
    # Initial ℓ₁ weighting arm: "normalized" (atom scales, the shipped default)
    # or "uniform" (wⱼ = 1). Resolved together with ``normalize_atoms`` by
    # ``resolved_weighting``; whichever of the two asks for uniform wins.
    weighting: str = "normalized"
    # Stage 2: re-solve on the fixed support without the ℓ₁ objective so the
    # gated candidate is not the boundary-hugging ℓ₁ optimum.
    remargin: bool = True
    # μ-preservation guard on each candidate support (see ``mu_guard``):
    # "reject" (default) skips a μ-breaking support like a gate rejection,
    # "warn" keeps it but records the typed reason, "off" disables the check.
    mu_guard: str = "reject"
    # --- optional post-rl1 backward elimination (see the module docstring) ---
    # OFF by default. Enabling it asks for a support smaller than rl1's at a
    # measured cost in exact headroom; the two guards below bound that cost.
    backward_elimination: bool = False
    # A drop is kept only if the candidate's exact charge/budget ratio stays
    # strictly below this. The default sits an order of magnitude under the
    # ε/2 cliff and just above the measured 16 → 14 levant3d cost, so the plan's
    # marginal case is visible rather than silently excluded.
    backward_max_charge_ratio: float = 1e-3
    # Drop ordering, ascending = tried for removal first.
    backward_order: str = "abs_coef"
    # Max full sweeps of the term list. Each accepted drop restarts a sweep;
    # measured convergence is 1 to 3 sweeps.
    backward_max_passes: int = 8
    # Wall-clock budget for the whole pass. Hitting it stops with
    # ``budget_exhausted`` and ``converged = False``; nothing already kept is
    # given back.
    backward_budget_s: float = 900.0
    # Design ε for the backward pass's re-solves, and therefore the margin its
    # Grams witness and the exact charge is measured against. None keeps the
    # designer's own default (1e-4). Scoped to this pass: it does NOT touch the
    # rl1 ladder above. Measured reason it exists: the levant3d 2003-print
    # δ₀ = 1 m = 7 support re-solves ``unknown`` at 1e-4 (Gram screening,
    # residual 1.2e-01) and cleanly at 1e-6, which is also the ε the published
    # exact-phase numbers were taken at.
    backward_epsilon: Optional[float] = None
    # ``strict_solver`` for those re-solves. None keeps the designer's default.
    # True is what a MOSEK-direct pass wants: a silent CVXPY fallback captures
    # no Gram, so the candidate could not be priced and would be refused anyway,
    # after paying for the fallback solve.
    backward_strict_solver: Optional[bool] = None
    # Design ε of the reweighted-ℓ₁ ladder solves and of the stage-2 remargin
    # re-solves. None (the default) keeps the designer's own default (1e-4),
    # byte for byte the behaviour before this knob existed. A caller that
    # designs its certificate at another ε (``certify_gain_point.py
    # --design-epsilon``) passes the same value here, so the support is
    # selected on the SDP that D then solves. Measured reason it exists
    # (WP X5, 2026-09-25): the perturbed pidsmc design is empty at 1e-4 under
    # the W-block ε rows and feasible at 1e-6, so a selection pinned at 1e-4
    # failed at iteration 0. Scoped to the ladder and remargin only: the
    # backward pass keeps ``backward_epsilon``.
    design_epsilon: Optional[float] = None

    def resolved_mu_guard(self) -> str:
        """The effective μ-guard mode, validated ("reject" | "warn" | "off")."""
        return resolve_mu_guard(self.mu_guard)

    def resolved_backward_order(self) -> str:
        """The effective drop ordering, validated (never silently defaulted)."""
        if self.backward_order not in BACKWARD_ORDERS:
            raise ValueError(
                f"unknown backward_order {self.backward_order!r}; "
                f"expected one of {sorted(BACKWARD_ORDERS)}")
        return self.backward_order

    def resolved_weighting(self) -> str:
        """The effective weighting arm: "normalized" or "uniform"."""
        if self.weighting not in WEIGHTING_ARMS:
            raise ValueError(
                f"unknown weighting {self.weighting!r}; "
                f"expected one of {sorted(WEIGHTING_ARMS)}")
        if self.weighting == "uniform" or not self.normalize_atoms:
            return "uniform"
        return "normalized"


@dataclass
class SparseDesignResult:
    feasible: bool = False
    V_gf: Optional[GeneralizedForm] = None
    support: List[int] = field(default_factory=list)   # indices into active_terms
    n_terms_dense: int = 0
    n_terms_sparse: int = 0
    # Per-iteration diagnostics: support set, ℓ₁ value, solve time, status.
    iterations: List[dict] = field(default_factory=list)
    atom_scales: Optional[List[float]] = None
    # Which initial-weight arm produced this record: "normalized" or "uniform".
    weighting: str = ""
    # μ bookkeeping (see ``mu_guard``). ``mu_design`` is the designer's μ = k·r
    # from the library AND the dynamics; ``mu_dense_v_only`` is the same library
    # scored the way every V-only path scores it, and is the guard's reference;
    # ``mu_sparse_v_only`` is the adopted candidate's. ``mu_checks`` carries one
    # ``MuCheck.as_record()`` per support the ladder visited.
    mu_guard: str = ""
    mu_design: Optional[List[int]] = None
    mu_dense_v_only: Optional[List[int]] = None
    mu_sparse_v_only: Optional[List[int]] = None
    mu_checks: List[dict] = field(default_factory=list)
    remargin_result: Optional[JointSOSResult] = None
    support_stable: bool = False
    # Backward-elimination bookkeeping (empty/None when the pass is off, which
    # is the default). ``backward_steps`` carries one ``BackwardStep.as_record()``
    # per candidate drop, in the order they were tried; ``backward_charge_ratio``
    # is the adopted support's exact charge/budget ratio and
    # ``backward_start_charge_ratio`` the rl1 support's, so the record states the
    # price of the reduction and not only its size.
    backward_elimination: bool = False
    backward_steps: List[dict] = field(default_factory=list)
    backward_start_support: List[int] = field(default_factory=list)
    backward_start_charge_ratio: Optional[float] = None
    backward_charge_ratio: Optional[float] = None
    backward_max_charge_ratio: Optional[float] = None
    backward_n_dropped: int = 0
    backward_solves: int = 0
    backward_passes: int = 0
    backward_converged: bool = False
    backward_time_s: float = 0.0
    # With gate_fn: whether the returned candidate passed the gate, and the
    # gate's return value (e.g. the Certificate) for the accepted candidate.
    gated: bool = False
    gate_result: Optional[object] = None
    message: str = ""
    total_time: float = 0.0


def atom_scales(
    term_specs: Sequence[TermSpec],
    weights_r,
    dynamics_gf: List[GeneralizedForm],
    optimizer: Optional[JointSOSOptimizer] = None,
    dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]] = None,
) -> np.ndarray:
    """Per-term associated-form contribution scales nⱼ for fair ℓ₁ weighting.

    nⱼ = ‖stacked column j of the V coefficient matrix AND the W coefficient
    matrix over all octants (and inclusion vertices)‖₂. The V entries are ±1
    sign factors (≈ constant per atom); the W entries carry the dynamics gains
    from the Lie derivative — the part where coefficient-scale unfairness in an
    unnormalized ℓ₁ actually lives (Review1's criticism of the exam study).
    """
    n_vars = len(weights_r)
    opt = optimizer or JointSOSOptimizer(weights_r=tuple(weights_r), verbose=False)
    specs = list(term_specs)
    n_terms = len(specs)

    _, mu = compute_mu_exponents_with_dynamics(
        term_specs=specs, weights_r=tuple(weights_r), dynamics_gf=dynamics_gf,
        ensure_even_exponents=opt.ensure_even_exponents,
    )
    v_y_degree = 0
    for spec in specs:
        deg = sum(mu[i] * float(spec.exponents[i]) for i in range(n_vars))
        v_y_degree = max(v_y_degree, int(deg))
    all_v_monomials = get_monomial_basis(n_vars, v_y_degree)

    vertices = dynamics_gf_vertices if dynamics_gf_vertices else [dynamics_gf]
    sq = np.zeros(n_terms)
    for signs in get_hyperoctant_signs(n_vars):
        M_V = build_v_coefficient_matrix(specs, mu, signs, all_v_monomials)
        sq += np.sum(M_V ** 2, axis=0)
        for dyn in vertices:
            M_W, _, _ = opt._compute_w_coefficient_matrix_weighted_sphere(
                specs, mu, signs, dyn)
            if M_W is not None:
                sq += np.sum(np.asarray(M_W) ** 2, axis=0)
    scales = np.sqrt(sq)
    # A zero column would mean the term contributes nothing anywhere; weight it
    # at the median scale so the ℓ₁ drives it to zero rather than dividing by 0.
    zero = scales <= 0
    if np.any(zero):
        fill = float(np.median(scales[~zero])) if np.any(~zero) else 1.0
        scales[zero] = fill
    return scales


def _support(specs: Sequence[TermSpec], V_gf: GeneralizedForm,
             threshold_rel: float) -> List[int]:
    """Indices of active terms: diagonals always; cross terms above threshold."""
    coefs = np.array([float(t.coefficient) for t in V_gf.terms])
    cross = [i for i, s in enumerate(specs) if not s.is_diagonal()]
    max_cross = max((abs(coefs[i]) for i in cross), default=0.0)
    thr = threshold_rel * max_cross if max_cross > 0 else 0.0
    keep = []
    for i, s in enumerate(specs):
        if s.is_diagonal() or abs(coefs[i]) > thr:
            keep.append(i)
    return keep


@dataclass(frozen=True)
class ChargeProbe:
    """One measurement of a candidate's exact charge/budget ratio.

    ``ratio is None`` means the charge could NOT be measured (no design Gram
    bundle, a refused bundle, a target that does not assemble). That is a
    refusal, not a pass: the caller must not keep a candidate it could not
    price. ``exact_ok`` is the Proposition-7 verdict on the same run; a ratio
    can be present and the verdict still False (the failing octant is the one
    that produced the ratio).
    """
    ratio: Optional[float]
    exact_ok: bool
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True)
class BackwardStep:
    """Typed outcome of ONE candidate drop in the backward-elimination pass."""
    pass_index: int
    outcome: str
    term_index: int
    term: str
    support_size_before: int
    support_size_after: int
    charge_ratio: Optional[float] = None
    charge_bound: Optional[float] = None
    exact_ok: Optional[bool] = None
    solved: bool = False
    detail: str = ""
    time_s: float = 0.0

    def as_record(self) -> dict:
        """JSON-serializable form for an artifact record."""
        return {
            "pass": int(self.pass_index),
            "outcome": self.outcome,
            "term_index": int(self.term_index),
            "term": self.term,
            "support_size_before": int(self.support_size_before),
            "support_size_after": int(self.support_size_after),
            "charge_ratio": self.charge_ratio,
            "charge_bound": self.charge_bound,
            "exact_ok": self.exact_ok,
            "solved": bool(self.solved),
            "detail": self.detail,
            "time_s": round(float(self.time_s), 3),
        }


@dataclass
class BackwardEliminationResult:
    """Outcome of the whole backward pass (see ``backward_eliminate``)."""
    support: List[int] = field(default_factory=list)
    V_gf: Optional[GeneralizedForm] = None
    gate_result: Optional[object] = None
    design_result: Optional[JointSOSResult] = None
    steps: List[BackwardStep] = field(default_factory=list)
    start_support: List[int] = field(default_factory=list)
    start_charge_ratio: Optional[float] = None
    charge_ratio: Optional[float] = None
    charge_bound: float = 0.0
    order: str = ""
    n_dropped: int = 0
    n_solves: int = 0
    n_passes: int = 0
    converged: bool = False
    time_s: float = 0.0
    message: str = ""

    def records(self) -> List[dict]:
        return [s.as_record() for s in self.steps]


def format_term_spec(spec) -> str:
    """Compact human label of a term: ``(e1/e2/e3)[types]`` (``s``/``a``)."""
    exps = "/".join(str(e) for e in spec.exponents)
    types = "".join("s" if t == "signed" else "a" for t in spec.term_types)
    return f"({exps})[{types}]"


def design_gram_charge_probe(
    V_gf: GeneralizedForm,
    design_result: JointSOSResult,
    dynamics_list: Sequence[Sequence[GeneralizedForm]],
    n_vars: int,
    epsilon: float,
) -> ChargeProbe:
    """Exact charge/budget ratio of a candidate, from its OWN design Grams.

    This is the D → X step of the citability chain: the MOSEK-direct design
    already emitted a Gram per PSD block, so Proposition 7 is re-proved in ℚ
    over those Grams and no float solve — in particular no CVXPY solve — runs
    on this route (``exact_phase.run_exact_phase(..., gram_bundle=...)``, which
    also hash-verifies the bundle and rebuilds every octant target from
    scratch).

    ``epsilon`` must be the margin the DESIGN subtracted; that is the only
    margin its Grams can witness, and ``run_exact_phase`` refuses any other.
    """
    from .exact_phase import GramBundleError, run_exact_phase

    if not getattr(design_result, "gram_certificates", None):
        return ChargeProbe(
            None, False, "no_design_gram_bundle",
            "the design captured no Gram bundle (backend is not MOSEK_DIRECT, "
            "or capture_design_grams=False); the exact charge cannot be "
            "measured without a re-solve, so the candidate is refused")
    try:
        exact = run_exact_phase(V_gf, list(dynamics_list), n_vars,
                                epsilon=epsilon, gram_bundle=design_result)
    except GramBundleError as exc:
        return ChargeProbe(None, False, exc.reason, str(exc)[:400])
    ratio = exact.worst_charge_ratio
    reason = ""
    if not exact.exact_pass:
        reason = exact.v_exact.reason or next(
            (e.exact.reason for e in exact.endpoints if not e.ok), "exact_failed")
    return ChargeProbe(ratio, bool(exact.exact_pass), reason)


def backward_eliminate(
    weights_r,
    degree_m: int,
    dynamics_gf: List[GeneralizedForm],
    active_terms: Sequence[TermSpec],
    start_support: Sequence[int],
    *,
    dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]] = None,
    solver: str = "MOSEK_DIRECT",
    config: Optional[SparseDesignConfig] = None,
    gate_fn=None,
    charge_fn: Optional[Callable[[GeneralizedForm, JointSOSResult], ChargeProbe]] = None,
    coefficients: Optional[dict] = None,
    scales: Optional[Sequence[float]] = None,
    mu_reference: Optional[Sequence[int]] = None,
    start_V_gf: Optional[GeneralizedForm] = None,
    start_design_result: Optional[JointSOSResult] = None,
    start_gate_result: Optional[object] = None,
    verbose: bool = False,
) -> BackwardEliminationResult:
    """Drop cross terms one at a time from ``start_support``, re-solving each time.

    Greedy, smallest-|coefficient| first, with a full design re-solve on every
    candidate support. A drop is ACCEPTED only when all of the following hold,
    checked in this order (cheapest first, so a refusal costs as little as
    possible):

    1. the μ-preservation guard passes on the reduced support (integer
       arithmetic, no solve — ``mu_guard.check_mu_preservation``);
    2. the design re-solve on the reduced support is feasible;
    3. ``gate_fn`` certifies the re-solved candidate;
    4. the exact charge/budget ratio is measurable AND below
       ``config.backward_max_charge_ratio``.

    Anything else records a typed step and moves on to the next term. After an
    accepted drop the sweep restarts from the smallest coefficient of the NEW
    candidate, up to ``config.backward_max_passes`` sweeps or
    ``config.backward_budget_s`` seconds.

    This function makes no stability claim and adopts nothing: it returns the
    smallest support that survived all four conditions, together with the price
    it paid in exact headroom (``start_charge_ratio`` → ``charge_ratio``).
    """
    t0 = time.time()
    cfg = config or SparseDesignConfig()
    order = cfg.resolved_backward_order()
    mu_guard_mode = cfg.resolved_mu_guard()
    specs = list(active_terms)
    n_vars = len(tuple(weights_r))
    bound = float(cfg.backward_max_charge_ratio)

    solve_kw = {}
    if cfg.backward_epsilon is not None:
        solve_kw["epsilon"] = float(cfg.backward_epsilon)
    if cfg.backward_strict_solver is not None:
        solve_kw["strict_solver"] = bool(cfg.backward_strict_solver)

    probe_opt = JointSOSOptimizer(weights_r=tuple(weights_r), verbose=False,
                                  **solve_kw)
    ensure_even = probe_opt.ensure_even_exponents
    design_epsilon = float(probe_opt.epsilon)
    if mu_reference is None:
        mu_reference = v_only_mu(specs, weights_r, ensure_even)

    dynamics_list = (list(dynamics_gf_vertices) if dynamics_gf_vertices
                     else [list(dynamics_gf)])
    if charge_fn is None:
        def charge_fn(V, design_result):  # noqa: E306
            return design_gram_charge_probe(V, design_result, dynamics_list,
                                            n_vars, design_epsilon)

    out = BackwardEliminationResult(
        support=list(start_support), start_support=list(start_support),
        V_gf=start_V_gf, design_result=start_design_result,
        gate_result=start_gate_result, charge_bound=bound, order=order)

    def _solve(support):
        opt = JointSOSOptimizer(weights_r=tuple(weights_r), solver=solver,
                                verbose=False, **solve_kw)
        return opt.find_lyapunov_with_dynamics(
            degree_m=degree_m, dynamics_gf=dynamics_gf,
            active_terms=[specs[i] for i in support],
            dynamics_gf_vertices=dynamics_gf_vertices)

    # Baseline: the price of the support we start from, measured the same way
    # every candidate is measured, so the two numbers are comparable.
    if start_design_result is not None and start_V_gf is not None:
        base = charge_fn(start_V_gf, start_design_result)
        out.start_charge_ratio = base.ratio
        out.charge_ratio = base.ratio
        if base.ratio is None and verbose:
            print(f"  backward: baseline charge unmeasurable ({base.reason})")

    coefs = {int(k): abs(float(v)) for k, v in (coefficients or {}).items()}
    scale_vec = (None if scales is None
                 else [float(s) for s in scales])

    def sort_key(i):
        c = coefs.get(i, 0.0)
        if order == "coef_x_scale" and scale_vec is not None:
            return c * scale_vec[i]
        return c

    diag_idx = {i for i, s in enumerate(specs) if s.is_diagonal()}
    S = list(out.support)
    stopped = False
    for pass_index in range(1, int(cfg.backward_max_passes) + 1):
        out.n_passes = pass_index
        improved = False
        for i in sorted([j for j in S if j not in diag_idx], key=sort_key):
            t_step = time.time()
            if time.time() - t0 > cfg.backward_budget_s:
                out.steps.append(BackwardStep(
                    pass_index, BWD_BUDGET_EXHAUSTED, i, format_term_spec(specs[i]),
                    len(S), len(S), charge_bound=bound,
                    detail=(f"wall-clock budget {cfg.backward_budget_s:g}s "
                            f"exhausted after {out.n_solves} design solves")))
                stopped = True
                break
            trial = [j for j in S if j != i]
            label = format_term_spec(specs[i])

            # 1. μ guard on the reduced SUPPORT — the V-only μ is a function of
            # the term exponents alone, so this costs no solve.
            if mu_guard_mode != "off":
                chk = check_mu_preservation([specs[j] for j in trial],
                                            weights_r, mu_reference, ensure_even)
                if not chk.ok and mu_guard_mode == "reject":
                    out.steps.append(BackwardStep(
                        pass_index, BWD_REFUSED_MU, i, label, len(S), len(trial),
                        charge_bound=bound, detail=f"{chk.reason}: {chk.detail}",
                        time_s=time.time() - t_step))
                    if verbose:
                        print(f"  backward: drop {label} refused by the mu guard "
                              f"({chk.reason})")
                    continue

            # 2. full design re-solve on the reduced support.
            res = _solve(trial)
            out.n_solves += 1
            if not res.feasible or res.V_gf is None:
                out.steps.append(BackwardStep(
                    pass_index, BWD_INFEASIBLE, i, label, len(S), len(trial),
                    charge_bound=bound, solved=True,
                    detail=str(res.solver_status), time_s=time.time() - t_step))
                continue
            V = res.V_gf

            # 2b. μ re-check on the MATERIALIZED candidate. Same set of specs,
            # so this must agree with step 1; it is defense in depth against a
            # designer that returns a different term list than it was handed.
            if mu_guard_mode != "off":
                chk2 = check_mu_preservation(specs_from_gf(V), weights_r,
                                             mu_reference, ensure_even)
                if not chk2.ok and mu_guard_mode == "reject":
                    out.steps.append(BackwardStep(
                        pass_index, BWD_REFUSED_MU, i, label, len(S), len(trial),
                        charge_bound=bound, solved=True,
                        detail=f"{chk2.reason}: {chk2.detail}",
                        time_s=time.time() - t_step))
                    continue

            # 3. the float gate stays the authority on the candidate.
            gate_ret = None
            if gate_fn is not None:
                gate_ret = gate_fn(V)
                if not bool(getattr(gate_ret, "surface_certified", gate_ret)):
                    out.steps.append(BackwardStep(
                        pass_index, BWD_GATE_REJECTED, i, label, len(S), len(trial),
                        charge_bound=bound, solved=True,
                        detail=str(getattr(gate_ret, "message", ""))[:300],
                        time_s=time.time() - t_step))
                    if verbose:
                        print(f"  backward: drop {label} failed the gate")
                    continue

            # 4. headroom guard.
            probe = charge_fn(V, res)
            if probe.ratio is None:
                out.steps.append(BackwardStep(
                    pass_index, BWD_CHARGE_UNAVAILABLE, i, label, len(S),
                    len(trial), charge_ratio=None, charge_bound=bound,
                    exact_ok=probe.exact_ok, solved=True,
                    detail=f"{probe.reason}: {probe.detail}"[:400],
                    time_s=time.time() - t_step))
                if verbose:
                    print(f"  backward: drop {label} refused, charge "
                          f"unmeasurable ({probe.reason})")
                continue
            if (not probe.exact_ok) or probe.ratio >= bound:
                out.steps.append(BackwardStep(
                    pass_index, BWD_REFUSED_HEADROOM, i, label, len(S), len(trial),
                    charge_ratio=probe.ratio, charge_bound=bound,
                    exact_ok=probe.exact_ok, solved=True,
                    detail=(probe.reason or
                            f"charge/budget {probe.ratio:.4g} >= bound {bound:g}"),
                    time_s=time.time() - t_step))
                if verbose:
                    print(f"  backward: drop {label} refused by the headroom "
                          f"guard (charge {probe.ratio:.4g} vs bound {bound:g})")
                continue

            # kept.
            out.steps.append(BackwardStep(
                pass_index, BWD_KEPT, i, label, len(S), len(trial),
                charge_ratio=probe.ratio, charge_bound=bound, exact_ok=True,
                solved=True, time_s=time.time() - t_step))
            S = trial
            out.support = list(S)
            out.V_gf = V
            out.design_result = res
            out.gate_result = gate_ret if gate_fn is not None else out.gate_result
            out.charge_ratio = probe.ratio
            out.n_dropped += 1
            # The new candidate's own coefficients order the next sweep.
            coefs = {idx: abs(float(t.coefficient))
                     for idx, t in zip(S, V.terms)}
            improved = True
            if verbose:
                print(f"  backward: dropped {label} -> |S|={len(S)} "
                      f"(charge {probe.ratio:.4g})")
            break
        if stopped:
            break
        if not improved:
            out.converged = True
            break

    out.time_s = time.time() - t0
    out.message = (
        f"backward elimination [{order}]: {len(out.start_support)} -> "
        f"{len(out.support)} terms in {out.n_solves} design solves / "
        f"{out.n_passes} pass(es)"
        + ("" if out.converged else " (stopped before convergence)")
        + f"; charge/budget {out.start_charge_ratio!r} -> {out.charge_ratio!r} "
          f"at bound {bound:g}")
    return out


def design_sparse_lyapunov(
    weights_r,
    degree_m: int,
    dynamics_gf: List[GeneralizedForm],
    active_terms: Sequence[TermSpec],
    dynamics_gf_vertices: Optional[List[List[GeneralizedForm]]] = None,
    solver: str = "MOSEK",
    config: Optional[SparseDesignConfig] = None,
    gate_fn=None,
    verbose: bool = False,
) -> SparseDesignResult:
    """Reweighted-ℓ₁ sparse design of a GF Lyapunov candidate.

    Runs the Candès–Wakin–Boyd loop over the SAME μ-filtered library and the
    SAME joint V+W SOS constraints as the dense designer, then (stage 2)
    re-solves on the stabilized support without the ℓ₁ objective. The returned
    candidate must still pass ``certify_discovered_gf`` — this function makes
    no stability claims itself.

    ``gate_fn``: optional caller-supplied gate (e.g. a ``certify_discovered_gf``
    closure; success is ``ret.surface_certified`` or truthiness). When given,
    the distinct supports visited by the reweighting are tried smallest-first
    and the smallest CERTIFIED support wins — the minimal ℓ₁ support can be a
    boundary case the strict gate rejects, and the next-larger support
    typically certifies. The gate stays the authority either way.

    ``config.weighting`` picks the initial-weight arm ("normalized", the
    default atom scales, or "uniform"); the arm used is echoed in
    ``SparseDesignResult.weighting`` and in the result message. Running both
    arms and keeping the smaller gating support is a caller-side decision.

    ``config.mu_guard`` ("reject" by default) refuses a candidate support whose
    V-only μ differs from the dense library's, or is not proportional to r. Such
    a support can still pass the float gate but can never be re-proven exactly
    (``exact_octant_target`` refuses a non-homogeneous target). The refusal is
    typed and lands in the same ``rejected`` list as a gate rejection, so the
    ladder walks on to the next-larger support.

    ``config.backward_elimination`` (**False** by default) runs
    :func:`backward_eliminate` on the support the ladder adopted, keeping a
    further drop only when the μ guard, the design re-solve, ``gate_fn`` and the
    exact-headroom bound all accept it. Off by default because past the rl1
    support parsimony is paid for in exact margin (module docstring); the steps
    and both charge ratios land in ``SparseDesignResult.backward_*`` either way.
    """
    t0 = time.time()
    cfg = config or SparseDesignConfig()
    weighting = cfg.resolved_weighting()
    mu_guard_mode = cfg.resolved_mu_guard()
    specs = list(active_terms)
    out = SparseDesignResult(n_terms_dense=len(specs), weighting=weighting,
                             mu_guard=mu_guard_mode)

    cross_idx = [i for i, s in enumerate(specs) if not s.is_diagonal()]
    if not cross_idx:
        out.message = "no cross terms to sparsify"
        out.total_time = time.time() - t0
        return out

    # μ reference for the guard, computed once on the dense library. The
    # designer's μ (with dynamics) is recorded alongside it: the two agree on
    # every measured system, and a disagreement is itself worth seeing in the
    # record rather than being averaged away.
    ensure_even = JointSOSOptimizer(
        weights_r=tuple(weights_r), verbose=False).ensure_even_exponents
    mu_dense_v_only = v_only_mu(specs, weights_r, ensure_even)
    out.mu_dense_v_only = list(mu_dense_v_only)
    out.mu_design = list(design_mu(specs, weights_r, dynamics_gf, ensure_even))

    scales = atom_scales(specs, weights_r, dynamics_gf,
                         dynamics_gf_vertices=dynamics_gf_vertices)
    out.atom_scales = [float(s) for s in scales]
    base_w = scales if weighting == "normalized" else np.ones(len(specs))
    w = base_w.copy()
    w_cap = cfg.w_max_factor * base_w

    last_supports: List[tuple] = []
    best: Optional[JointSOSResult] = None
    best_support: Optional[List[int]] = None

    # ``SparseDesignConfig.design_epsilon``: empty unless set, so the default
    # construction below is unchanged.
    eps_kw = ({} if cfg.design_epsilon is None
              else {"epsilon": float(cfg.design_epsilon)})

    for it in range(cfg.max_reweight_iters):
        opt = JointSOSOptimizer(weights_r=tuple(weights_r), solver=solver,
                                verbose=False, l1_weights=w, **eps_kw)
        res = opt.find_lyapunov_with_dynamics(
            degree_m=degree_m, dynamics_gf=dynamics_gf, active_terms=specs,
            dynamics_gf_vertices=dynamics_gf_vertices,
        )
        if not res.feasible or res.V_gf is None:
            out.iterations.append({
                "iter": it, "feasible": False, "status": res.solver_status,
                "solve_time": res.solve_time, "solver": solver,
            })
            out.message = (f"weighted-ℓ₁ SDP infeasible at iteration {it} "
                           f"({res.solver_status})")
            break

        coefs = np.array([float(t.coefficient) for t in res.V_gf.terms])
        support = _support(specs, res.V_gf, cfg.support_threshold_rel)
        l1_val = float(np.sum(base_w[cross_idx] * np.abs(coefs[cross_idx])))
        out.iterations.append({
            "iter": it, "feasible": True, "status": res.solver_status,
            "solve_time": res.solve_time, "support": list(support),
            "n_active": len(support), "l1_weighted": l1_val, "solver": solver,
        })
        if verbose:
            print(f"  reweight iter {it}: {len(support)}/{len(specs)} active, "
                  f"weighted ℓ₁ = {l1_val:.4g}")

        if best is None or len(support) <= len(best_support):
            best, best_support = res, support

        last_supports.append(tuple(sorted(support)))
        if (len(last_supports) > cfg.stable_window
                and len(set(last_supports[-(cfg.stable_window + 1):])) == 1):
            out.support_stable = True
            break

        # Reweight on TRUE coefficients: wⱼ ← nⱼ / (|αⱼ| + δ), capped.
        max_cross = max(abs(coefs[i]) for i in cross_idx)
        delta = cfg.delta_rel * max_cross if max_cross > 0 else cfg.delta_rel
        w = np.minimum(base_w / (np.abs(coefs) + delta), w_cap)

    if best is None:
        out.total_time = time.time() - t0
        if not out.message:
            out.message = "no feasible weighted-ℓ₁ solve"
        return out

    def _pruned_l1_candidate(support):
        kept = set(support)
        terms = [t for i, t in enumerate(best.V_gf.terms) if i in kept]
        return GeneralizedForm(degree_m=degree_m, weights_r=tuple(weights_r),
                               terms=terms)

    def _remargin(support):
        """Stage 2: re-solve the fixed support without the ℓ₁ objective."""
        opt2 = JointSOSOptimizer(weights_r=tuple(weights_r), solver=solver,
                                 verbose=False, **eps_kw)
        return opt2.find_lyapunov_with_dynamics(
            degree_m=degree_m, dynamics_gf=dynamics_gf,
            active_terms=[specs[i] for i in support],
            dynamics_gf_vertices=dynamics_gf_vertices,
        )

    # Candidate supports, smallest first. The minimal ℓ₁ support can be a
    # boundary case the strict gate rejects (W ≥ 0 in design vs W > 0 at the
    # gate), so with a gate available we walk the ladder of distinct supports
    # the reweighting visited and keep the smallest CERTIFIED one.
    if gate_fn is not None:
        seen, ladder = set(), []
        for it in out.iterations:
            if it.get("feasible") and "support" in it:
                key = tuple(sorted(it["support"]))
                if key not in seen:
                    seen.add(key)
                    ladder.append(list(key))
        ladder.sort(key=len)
    else:
        ladder = [best_support]

    rejected = []
    for support in ladder:
        # The design result that materialized THIS V, or None when V is the
        # pruned ℓ₁ candidate. Only a design result whose Grams belong to the
        # candidate can price it, so the backward pass refuses the None case
        # rather than measuring a bundle from a different support.
        design_res: Optional[JointSOSResult] = None
        if cfg.remargin:
            res2 = _remargin(support)
            out.remargin_result = res2
            if res2.feasible and res2.V_gf is not None:
                V = res2.V_gf
                design_res = res2
            elif tuple(sorted(support)) == tuple(sorted(best_support)):
                V = _pruned_l1_candidate(support)
                out.message = (f"remargin on {len(support)}-term support "
                               f"infeasible ({res2.solver_status}); "
                               f"returning ℓ₁ candidate")
            else:
                rejected.append((len(support), "remargin_infeasible"))
                continue
        else:
            if tuple(sorted(support)) != tuple(sorted(best_support)):
                continue  # without remargin only the ℓ₁ optimum is materializable
            V = _pruned_l1_candidate(support)

        # μ-preservation guard, on the MATERIALIZED candidate (the object the
        # gate and every downstream V-only path actually score), before the
        # gate: a μ-breaking support can pass the float gate and still be
        # unprovable exactly, so refusing it here is cheaper and more honest
        # than discovering it at the exact phase.
        if mu_guard_mode != "off":
            chk = check_mu_preservation(
                specs_from_gf(V), weights_r, mu_dense_v_only, ensure_even)
            rec = chk.as_record()
            rec["n_terms"] = len(support)
            rec["support"] = list(support)
            rec["enforced"] = (mu_guard_mode == "reject")
            out.mu_checks.append(rec)
            if not chk.ok:
                if mu_guard_mode == "reject":
                    rejected.append((len(support), chk.reason))
                    if verbose:
                        print(f"  support of {len(support)} breaks mu "
                              f"({chk.mu_candidate} vs dense {chk.mu_reference}): "
                              f"{chk.reason}; trying next larger support")
                    continue
                note = (f"mu guard WARNING on the {len(support)}-term support: "
                        f"{chk.reason} ({chk.detail})")
                out.message = f"{out.message}; {note}" if out.message else note

        if gate_fn is not None:
            ret = gate_fn(V)
            ok = bool(getattr(ret, "surface_certified", ret))
            out.gate_result = ret
            if not ok:
                rejected.append((len(support), "gate_rejected"))
                if verbose:
                    print(f"  support of {len(support)} failed the gate; "
                          f"trying next larger support")
                continue
            out.gated = True

        # Optional post-rl1 backward elimination (off by default). It runs on
        # the support the ladder just adopted and can only SHRINK it: every
        # candidate must clear the μ guard, the design re-solve, the same
        # gate_fn and the exact-headroom bound. If nothing survives, the ladder
        # support and its certificate are returned unchanged.
        if cfg.backward_elimination:
            bwd = backward_eliminate(
                weights_r, degree_m, dynamics_gf, specs, support,
                dynamics_gf_vertices=dynamics_gf_vertices, solver=solver,
                config=cfg, gate_fn=gate_fn,
                coefficients={idx: abs(float(t.coefficient))
                              for idx, t in zip(support, V.terms)},
                scales=scales, mu_reference=mu_dense_v_only,
                start_V_gf=V, start_design_result=design_res,
                start_gate_result=out.gate_result, verbose=verbose)
            out.backward_elimination = True
            out.backward_steps = bwd.records()
            out.backward_start_support = list(bwd.start_support)
            out.backward_start_charge_ratio = bwd.start_charge_ratio
            out.backward_charge_ratio = bwd.charge_ratio
            out.backward_max_charge_ratio = bwd.charge_bound
            out.backward_n_dropped = bwd.n_dropped
            out.backward_solves = bwd.n_solves
            out.backward_passes = bwd.n_passes
            out.backward_converged = bwd.converged
            out.backward_time_s = bwd.time_s
            if bwd.n_dropped and bwd.V_gf is not None:
                support = list(bwd.support)
                V = bwd.V_gf
                if gate_fn is not None and bwd.gate_result is not None:
                    out.gate_result = bwd.gate_result
                if bwd.design_result is not None:
                    out.remargin_result = bwd.design_result
            note = bwd.message
            out.message = f"{out.message}; {note}" if out.message else note

        out.V_gf = V
        out.support = list(support)
        out.n_terms_sparse = len(support)
        out.mu_sparse_v_only = list(v_only_mu(specs_from_gf(V), weights_r,
                                              ensure_even))
        out.feasible = True
        # Preserve whatever note is already on out.message (e.g. "remargin on
        # N-term support infeasible; returning l1 candidate" for THIS support)
        # as a prefix, but always append the final outcome summary — including
        # `rejected`, which is where an earlier support's explicit GATE
        # REJECTION lives. Unconditionally overwriting out.message here (the
        # old `if not out.message` guard) silently dropped that final gate
        # outcome whenever an earlier support had already set a message.
        note = out.message
        summary = (
            f"sparse design [{weighting} weights]: "
            f"{out.n_terms_sparse}/{out.n_terms_dense} terms "
            f"after {len(out.iterations)} reweighted-ℓ₁ solves"
            + (" (support stable)" if out.support_stable else "")
            + (f"; {len(rejected)} smaller support(s) rejected ({rejected})"
               if rejected else ""))
        out.message = f"{note}; {summary}" if note else summary
        out.total_time = time.time() - t0
        return out

    # Nothing materialized/gated: report the ℓ₁ optimum honestly (ungated). Its
    # μ is recorded too, since this is exactly the object a caller might publish
    # as "the sparse support" without a certificate behind it.
    out.support = best_support
    out.n_terms_sparse = len(best_support)
    out.V_gf = _pruned_l1_candidate(best_support)
    out.mu_sparse_v_only = list(v_only_mu(specs_from_gf(out.V_gf), weights_r,
                                          ensure_even))
    out.feasible = True
    out.gated = False
    out.message = (f"no support certified by the gate or the mu guard "
                   f"(rejected: {rejected}); returning ungated ℓ₁ candidate")
    out.total_time = time.time() - t0
    return out
