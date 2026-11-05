"""Fail-fast screen on the DESIGN output, before anything can enter X.

Toolchain D -> X -> P (decision D1). ``D`` is the MOSEK Task-API design solve,
``X`` the exact-rational re-proof that now consumes D's Gram bundle directly
(P2-01/P2-02), ``P`` the stdlib portable verifier. Between D and X there used
to be nothing but the solver's own status.

This module wires the two cheap filters D1 names onto the packaged design
result:

1. **Sphere pre-check** (``sphere_check.min_V_on_sphere``). ``V`` is
   generalized-homogeneous, so its sign is constant along dilation orbits; a
   dense sample of unit directions exposes negativity if there is any. Finding
   ``V(x) < 0`` refutes ``V > 0`` outright, so no SOS certificate can exist.
   Dynamics-independent, float, no solver.
2. **A-posteriori Gram validation** (``joint_sos._validate_gram_certificate``)
   on EVERY bundle entry, V and W, with the entry's own ``eps`` as the budget:
   the PSD shift ``η = max(0, −λ_min)`` and the equality residual ``‖E‖₁`` are
   charged against the strictness margin as ``charge = η·C₁ + ‖E‖₁``.

   **Active default threshold: ``charge < eps``** (strict), i.e.
   ``budget_policy="achieved"``, the default of every entry point in this
   module since WP P2-14 (owner decision D-l, 2026-09-04). It is the weakest
   demand under which the proof line still yields a strictly positive
   certified margin ``eps − charge``.

   The historical threshold ``charge ≤ eps/2`` is ``budget_policy="half"``.
   It was the default before P2-14 and is the inequality the frozen Automatica
   note prints; it is still selectable, but it is NOT what an accepted design
   witness satisfies unless it was screened under that policy explicitly. Do
   not read acceptance here as the frozen note's half-margin.

   Both names and both thresholds mirror ``exact_gate.MARGIN_POLICIES``
   verbatim (``BUDGET_POLICIES`` below), so the cheap float screen and the
   exact gate agree about which bundles are worth proving.
   ``tests/test_design_screen.py`` pins the two defaults to each other.

Neither filter is the proof. Both can only ever REJECT: passing them says the
design output is worth handing to the exact phase, nothing more. The proof is
X (exact rational) and P.

**Never accept on solver status.** The MOSEK design backend has its own
acceptance policy (``mosek_direct.py``: ``solsta.unknown`` tolerated at
``eig_tol = 1e-7``, ``res_tol = 1e-5``, three to five orders looser than the
exact budget). That policy is deliberately NOT imported here. A design that the
backend typed ``unknown_accepted`` still has to pass this screen before its
Grams reach a certificate record.

Scope of the refusal
--------------------
An entry is *X-eligible* when it carries a positive margin (``eps > 0``, i.e.
``eps_reason == ""``). Those are the entries ``exact_gate.exact_verify_captures``
will actually consume; a failure on one of them is a refusal.

Entries with ``eps == 0.0`` carry a typed ``eps_reason`` (P2-02) and are
refused by the exact gate before it looks at their Gram, so they can never
enter X. They are still validated and reported — with their numbers and their
``eps_reason`` — but they do not trigger the refusal, because there is no
margin to charge a residual against and the legacy absolute policy is not the
right instrument for them.

``power_rate`` used to be the measured instance of exactly that: its W target
was ``3·y³`` (odd degree, ``w_degree_not_positive_even``) on the
non-homogeneous Gram basis ``[1, y, y²]``, so ``zᵀQz`` carried the monomial
``y⁴`` outside the target support, which the design encoding never
constrained — ``Q[2,2] = 6.16e+04`` and ``‖E‖₁ = 6.16e+04``
(``docs/execution_2026-09/reports/P2-04.md``, contradiction 1). **WP P2-09
removed both halves of that**: the design now doubles μ for a W block whose
y-degree comes out odd (``mu_policy == "w_mu_doubled_odd_y_degree"``, reported
per block below) and emits an explicit zero row for every basis product
outside the target support, so no Gram entry is ever a free variable. The
twelve nominal systems now have ``eps > 0`` and ``eps_reason == ""`` on every
block, and every block is X-eligible.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# How a caller reacts to a screen refusal. "reject" is the default.
SCREEN_MODES = ("reject", "warn", "off")

# Typed reasons. Empty string means the screen passed.
SCREEN_OK = ""
SPHERE_PRECHECK_FAILED = "sphere_precheck_failed"
GRAM_VALIDATION_FAILED = "gram_validation_failed"

#: Default absolute PSD tolerance handed to ``_validate_gram_certificate`` for
#: the margin-free (legacy) policy. Matches the verification-stage default.
DEFAULT_PSD_TOL = 1e-7

# Budget policy (WP P2-13). The screen's whole job is to predict, cheaply and
# in float, what the EXACT gate will say about a bundle. That only works while
# the two use the SAME acceptance threshold, so this vocabulary is
# ``exact_gate.MARGIN_POLICIES`` verbatim:
#
#   "achieved"  accept a block at ``charge < ε``     (DEFAULT since P2-14)
#   "half"      accept a block at ``charge ≤ ε/2``   (historical; the frozen
#                                                     note's printed threshold)
#
# Decoupling the screen from ε entirely (an absolute residual tolerance tied
# to solver accuracy) was the other candidate P2-13 measured and is NOT
# offered here: it makes the screen pass bundles the exact gate then refuses,
# which is strictly worse than the refusal it was meant to avoid. What removed
# the false refusals was moving BOTH thresholds, not unhooking one of them.
BUDGET_POLICIES = ("half", "achieved")

# Which residual the screen CHARGES (WP P2-13), the second half of "predict
# the exact gate":
#
#   "dense"         expand zᵀQz over every basis pair   (default). Since
#                   2026-09-25 this IS the residual both exact
#                   implementations charge: exact_gate (X) adopted the
#                   portable verifier's (P) dense rule, fix (a) of
#                   reports/XVC_RESIDUAL_2026-09-25.md. "dense" is the model
#                   that predicts X and P.
#   "deck_blocked"  DEPRECATED, OPTIMISTIC. Expands it over same-parity-class
#                   pairs only, i.e. the residual of the Reynolds projection
#                   Q̃. X charged that residual until 2026-09-25; it no longer
#                   does, and P never did. ``‖Ẽ‖₁ ≤ ‖E‖₁`` always, so this
#                   model can pass a block that X and P then charge more (by
#                   exactly the cross-class mass ‖C‖₁) and refuse. Kept only
#                   so that historical records and callers stay readable;
#                   selecting it emits ``OptimisticResidualModelWarning``.
#
# Measured on the CTA m = 5 design at ε = 1e-7 (P2-13): dense 7.36e-08
# against a projected 9.09e-15 on the same block; after fix (a) the dense
# number is the one the exact gate charges. Neither choice is a proof, and
# neither moves a design point — the screen runs on an already-solved result.
RESIDUAL_MODELS = ("dense", "deck_blocked")
OPTIMISTIC_RESIDUAL_MODELS = ("deck_blocked",)


class OptimisticResidualModelWarning(UserWarning):
    """A float screen residual model that under-charges against X and P.

    Raised (as a warning) when ``residual_model="deck_blocked"`` is selected.
    Since 2026-09-25 both exact implementations charge the DENSE residual of
    the persisted Gram, so the projected screen residual is a lower bound on
    what the proof charges, not a prediction of it. A screen PASS under it can
    be followed by an X or P refusal. Use ``"dense"`` (the default).
    """


def warn_optimistic_residual_model(model: str, stacklevel: int = 3) -> None:
    """Warn loudly when an optimistic screen residual model is selected."""
    if model in OPTIMISTIC_RESIDUAL_MODELS:
        warnings.warn(
            f"design screen residual_model={model!r} is DEPRECATED and "
            "OPTIMISTIC: it charges the parity-projected residual, which the "
            "exact gate (X) no longer charges and the portable verifier (P) "
            "never charged (both charge the dense residual since 2026-09-25, "
            "reports/XVC_RESIDUAL_2026-09-25.md). A screen pass under it "
            "does not predict an X or P pass; use residual_model='dense'.",
            OptimisticResidualModelWarning, stacklevel=stacklevel)


def resolve_budget_policy(policy: str) -> str:
    """Validate a screen budget policy, raising on an unknown one."""
    if policy not in BUDGET_POLICIES:
        raise ValueError(
            f"unknown design screen budget_policy {policy!r}; "
            f"expected one of {sorted(BUDGET_POLICIES)}")
    return policy


def resolve_residual_model(model: str) -> str:
    """Validate a screen residual model, raising on an unknown one.

    ``"deck_blocked"`` is accepted but warns
    (``OptimisticResidualModelWarning``): it is optimistic against both exact
    implementations since 2026-09-25.
    """
    if model not in RESIDUAL_MODELS:
        raise ValueError(
            f"unknown design screen residual_model {model!r}; "
            f"expected one of {sorted(RESIDUAL_MODELS)}")
    warn_optimistic_residual_model(model)
    return model


def resolve_screen_mode(mode: str) -> str:
    """Validate a screen mode, raising on an unknown one (never silently off)."""
    if mode not in SCREEN_MODES:
        raise ValueError(
            f"unknown design_screen mode {mode!r}; "
            f"expected one of {sorted(SCREEN_MODES)}")
    return mode


@dataclass(frozen=True)
class BlockScreen:
    """A-posteriori validation outcome for ONE Gram block of the bundle."""
    name: str
    block: str                 # "V" or "W"
    octant_index: Optional[int]
    endpoint: Optional[int]
    dim: int
    eps: float
    eps_reason: str
    gateable: bool             # eps > 0 -> the exact gate will consume it
    ok: bool
    min_eig: float
    residual_l1: float
    reason: str                # "" when ok
    mu: Tuple[int, ...] = ()   # the y-lift the block was BUILT at
    mu_policy: str = ""        # "" | "w_mu_doubled_odd_y_degree" (P2-09)
    # WP P5-04: how the DESIGN posed this block. "blocked" means the design
    # Gram was restricted to be block-diagonal over the parity classes of its
    # basis (lossless in α by group averaging); the entry validated here is the
    # DENSE reassembly, so this screen is unaffected by the choice and the
    # numbers stay comparable across the flip. ``y_parity_reason`` carries the
    # typed refusal on a dense block.
    y_parity_mode: str = ""            # "" | "blocked" | "dense"
    y_parity_reason: str = ""
    y_parity_n_classes: int = 0
    y_parity_psd_scalars_dense: int = 0
    y_parity_psd_scalars_blocked: int = 0
    # WP P2-13: the Proposition-6 numbers behind ``ok``, so a refusal is
    # legible without re-running the validation, and ``eps_certified`` =
    # ε − charge, the margin THIS block would prove if the exact gate
    # reproduces the float charge. It is a float prediction, never a claim.
    budget_policy: str = "achieved"
    residual_model: str = "dense"
    charge: float = 0.0
    budget: float = 0.0
    eps_certified: float = 0.0
    #: The dense residual, recorded even when a projected one was charged, so
    #: the two models stay comparable inside one record.
    residual_l1_dense: float = 0.0

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "block": self.block,
            "octant_index": self.octant_index,
            "endpoint": self.endpoint,
            "dim": int(self.dim),
            "eps": float(self.eps),
            "eps_reason": self.eps_reason,
            "mu": [int(v) for v in self.mu],
            "mu_policy": self.mu_policy,
            "y_parity_mode": self.y_parity_mode,
            "y_parity_reason": self.y_parity_reason,
            "y_parity_n_classes": int(self.y_parity_n_classes),
            "y_parity_psd_scalars_dense": int(self.y_parity_psd_scalars_dense),
            "y_parity_psd_scalars_blocked": int(self.y_parity_psd_scalars_blocked),
            "gateable": bool(self.gateable),
            "ok": bool(self.ok),
            "gram_min_eig": float(self.min_eig),
            "gram_residual_l1": float(self.residual_l1),
            "budget_policy": self.budget_policy,
            "residual_model": self.residual_model,
            "gram_residual_l1_dense": float(self.residual_l1_dense),
            "charge": float(self.charge),
            "budget": float(self.budget),
            "eps_certified": float(self.eps_certified),
            "reason": self.reason,
        }


@dataclass
class DesignScreen:
    """Typed verdict of the design screen (both filters)."""
    ok: bool
    reason: str                       # "" | SPHERE_PRECHECK_FAILED | GRAM_VALIDATION_FAILED
    detail: str = ""
    # Sphere pre-check
    sphere_ran: bool = False
    sphere_min_V: Optional[float] = None
    sphere_argmin: Optional[List[float]] = None
    sphere_n_samples: int = 0
    sphere_seconds: float = 0.0
    # Gram validation
    gram_ran: bool = False
    gram_skipped: str = ""            # typed reason no Gram was validated
    blocks: List[BlockScreen] = field(default_factory=list)
    worst_block: Optional[str] = None
    gram_min_eig: Optional[float] = None      # min over blocks
    gram_residual_l1: Optional[float] = None  # max over blocks
    gram_seconds: float = 0.0
    # WP P2-13: which acceptance threshold this screen ran under, and the
    # smallest ε − charge over the X-eligible blocks (the margin the design
    # output is predicted to prove). ``None`` when no gateable block produced
    # a charge.
    budget_policy: str = "achieved"
    residual_model: str = "dense"
    eps_certified: Optional[float] = None

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_blocks_gateable(self) -> int:
        return sum(1 for b in self.blocks if b.gateable)

    @property
    def n_blocks_failed(self) -> int:
        return sum(1 for b in self.blocks if not b.ok)

    @property
    def seconds(self) -> float:
        return self.sphere_seconds + self.gram_seconds

    def as_record(self) -> dict:
        """JSON-serializable form for an artifact record."""
        return {
            "ok": bool(self.ok),
            "reason": self.reason,
            "detail": self.detail,
            "sphere": {
                "ran": bool(self.sphere_ran),
                "min_V": (None if self.sphere_min_V is None
                          else float(self.sphere_min_V)),
                "argmin": self.sphere_argmin,
                "n_samples": int(self.sphere_n_samples),
                "seconds": float(self.sphere_seconds),
            },
            "budget_policy": self.budget_policy,
            "residual_model": self.residual_model,
            "eps_certified": (None if self.eps_certified is None
                              else float(self.eps_certified)),
            "gram": {
                "ran": bool(self.gram_ran),
                "skipped": self.gram_skipped,
                "n_blocks": self.n_blocks,
                "n_blocks_gateable": self.n_blocks_gateable,
                "n_blocks_failed": self.n_blocks_failed,
                "worst_block": self.worst_block,
                "gram_min_eig": (None if self.gram_min_eig is None
                                 else float(self.gram_min_eig)),
                "gram_residual_l1": (None if self.gram_residual_l1 is None
                                     else float(self.gram_residual_l1)),
                "seconds": float(self.gram_seconds),
                "blocks": [b.as_record() for b in self.blocks],
            },
            "seconds": float(self.seconds),
        }


def screen_gram_bundle(
    entries: Optional[Sequence[Dict[str, Any]]],
    *,
    psd_tol: float = DEFAULT_PSD_TOL,
    budget_policy: str = "achieved",
    residual_model: str = "dense",
) -> List[BlockScreen]:
    """Validate every entry of a design Gram bundle (GRAM_BUNDLE_SCHEMA).

    One ``BlockScreen`` per entry, in bundle order. The verdict per entry is
    ``joint_sos._validate_gram_certificate`` and nothing else: no solver status
    is consulted, and no threshold from the design backend is reused.

    ``budget_policy`` (WP P2-13) is passed straight through and must be the
    policy the exact phase will run under; see :data:`BUDGET_POLICIES`.
    """
    from .joint_sos import _validate_gram_certificate  # lazy: avoids a cycle

    budget_policy = resolve_budget_policy(budget_policy)
    residual_model = resolve_residual_model(residual_model)
    out: List[BlockScreen] = []
    for entry in entries or ():
        signs = entry.get("signs") or ()
        n_vars = len(signs)
        basis = list(entry["basis"])
        if n_vars == 0:
            n_vars = len(basis[0]) if basis else 0
        eps = float(entry.get("eps", 0.0) or 0.0)
        diag: Dict[str, float] = {}
        ok, min_eig, resid_l1, reason = _validate_gram_certificate(
            entry["target"],
            np.asarray(entry["gram"], dtype=np.float64),
            basis,
            n_vars,
            eps,
            float(psd_tol),
            budget_policy=budget_policy,
            residual_model=residual_model,
            diag=diag,
        )
        yp = entry.get("y_parity") or {}
        out.append(BlockScreen(
            name=str(entry.get("name", "?")),
            block=str(entry.get("block", "?")),
            octant_index=entry.get("octant_index"),
            endpoint=entry.get("endpoint"),
            dim=int(entry.get("dim", len(basis))),
            eps=eps,
            eps_reason=str(entry.get("eps_reason", "")),
            mu=tuple(int(v) for v in (entry.get("mu") or ())),
            mu_policy=str(entry.get("mu_policy", "")),
            y_parity_mode=str(yp.get("mode", "")),
            y_parity_reason=str(yp.get("reason", "")),
            y_parity_n_classes=int(yp.get("n_classes", 0) or 0),
            y_parity_psd_scalars_dense=int(yp.get("psd_scalars_dense", 0) or 0),
            y_parity_psd_scalars_blocked=int(
                yp.get("psd_scalars_blocked", 0) or 0),
            gateable=bool(eps > 0.0),
            ok=bool(ok),
            min_eig=float(min_eig),
            residual_l1=float(resid_l1),
            budget_policy=budget_policy,
            residual_model=residual_model,
            residual_l1_dense=float(diag.get("resid_l1_dense", 0.0)),
            charge=float(diag.get("charge", 0.0)),
            budget=float(diag.get("budget", 0.0)),
            eps_certified=float(diag.get("eps_certified", 0.0)),
            reason=reason,
        ))
    return out


def _worst(blocks: Sequence[BlockScreen]) -> Optional[BlockScreen]:
    """Worst block: lowest ``min_eig``, ties broken by the larger residual."""
    if not blocks:
        return None
    return min(blocks, key=lambda b: (b.min_eig, -b.residual_l1))


def screen_design_candidate(
    V_gf,
    gram_certificates: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    sphere_samples: int = 200_000,
    sphere_tol: float = 1e-9,
    sphere_seed: int = 0,
    psd_tol: float = DEFAULT_PSD_TOL,
    run_sphere: bool = True,
    budget_policy: str = "achieved",
    residual_model: str = "dense",
) -> DesignScreen:
    """Run both filters on a design candidate and type the outcome.

    ``V_gf`` is the packaged candidate; ``gram_certificates`` its design Gram
    bundle (``None`` when the backend recovered none — the Gram half is then
    typed ``skipped`` and the screen cannot refuse on it).

    The sphere pre-check runs first: it is the cheaper of the two and it
    refutes the whole "certified V is not even PD" class before any Gram
    arithmetic.
    """
    from .sphere_check import min_V_on_sphere  # lazy: keeps import cost off

    budget_policy = resolve_budget_policy(budget_policy)
    residual_model = resolve_residual_model(residual_model)
    screen = DesignScreen(ok=True, reason=SCREEN_OK,
                          budget_policy=budget_policy,
                          residual_model=residual_model)

    if run_sphere and V_gf is not None:
        t0 = time.time()
        sc = min_V_on_sphere(V_gf, n_samples=sphere_samples,
                             tol=sphere_tol, seed=sphere_seed)
        screen.sphere_seconds = time.time() - t0
        screen.sphere_ran = True
        screen.sphere_min_V = float(sc.min_V)
        screen.sphere_argmin = list(sc.argmin)
        screen.sphere_n_samples = int(sc.n_samples)
        if sc.refutes_pd:
            screen.ok = False
            screen.reason = SPHERE_PRECHECK_FAILED
            screen.detail = (
                f"{SPHERE_PRECHECK_FAILED}(min_V={sc.min_V:.4e}): V is NOT "
                f"positive definite at {sc.argmin} over {sc.n_samples} unit "
                f"directions, so no SOS certificate can exist"
            )
            return screen
    elif V_gf is None:
        screen.sphere_ran = False

    if not gram_certificates:
        screen.gram_skipped = ("no_gram_bundle" if gram_certificates is None
                               else "empty_gram_bundle")
        return screen

    t0 = time.time()
    blocks = screen_gram_bundle(gram_certificates, psd_tol=psd_tol,
                                budget_policy=budget_policy,
                                residual_model=residual_model)
    screen.gram_seconds = time.time() - t0
    screen.gram_ran = True
    screen.blocks = blocks
    screen.gram_min_eig = min((b.min_eig for b in blocks), default=None)
    screen.gram_residual_l1 = max((b.residual_l1 for b in blocks), default=None)
    screen.eps_certified = min(
        (b.eps_certified for b in blocks if b.gateable and b.budget > 0.0),
        default=None)

    # Only X-eligible blocks (eps > 0) can refuse: the others are refused by
    # the exact gate on their typed eps_reason before their Gram is read, and
    # carry no margin to charge a residual against. They are still reported.
    failed_gateable = [b for b in blocks if b.gateable and not b.ok]
    w = _worst(failed_gateable) or _worst(blocks)
    screen.worst_block = w.name if w is not None else None
    if failed_gateable:
        screen.ok = False
        screen.reason = GRAM_VALIDATION_FAILED
        screen.detail = (
            f"{GRAM_VALIDATION_FAILED}(block={w.name},{w.reason}): "
            f"{len(failed_gateable)}/{sum(1 for b in blocks if b.gateable)} "
            f"gateable Gram blocks fail a-posteriori validation "
            f"(worst min_eig={w.min_eig:.3e}, resid_l1={w.residual_l1:.3e}, "
            f"eps={w.eps:.3e})"
        )
    return screen


def screen_design_result(
    result,
    *,
    sphere_samples: int = 200_000,
    sphere_tol: float = 1e-9,
    sphere_seed: int = 0,
    psd_tol: float = DEFAULT_PSD_TOL,
    budget_policy: str = "achieved",
    residual_model: str = "dense",
) -> DesignScreen:
    """``screen_design_candidate`` on a packaged ``JointSOSResult``.

    An infeasible result carries no candidate and is not screened (there is
    nothing that could enter X): the screen is returned ``ok`` with the Gram
    half typed ``not_feasible``.
    """
    if result is None or not getattr(result, "feasible", False):
        screen = DesignScreen(
            ok=True, reason=SCREEN_OK,
            budget_policy=resolve_budget_policy(budget_policy),
            residual_model=resolve_residual_model(residual_model))
        screen.gram_skipped = "not_feasible"
        return screen
    return screen_design_candidate(
        result.V_gf,
        result.gram_certificates,
        sphere_samples=sphere_samples,
        sphere_tol=sphere_tol,
        sphere_seed=sphere_seed,
        psd_tol=psd_tol,
        budget_policy=budget_policy,
        residual_model=residual_model,
    )
