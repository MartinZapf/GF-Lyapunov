"""Typed failure classes for the certification path (WP P3-02).

Why this module exists
----------------------
Before P3-02 every rejection collapsed to one word. ``verifier._build_message``
emitted the literal ``"failed"`` for solver infeasibility, a solver that
returned ``unknown``, a raised exception and a Gram that
``joint_sos._validate_gram_certificate`` rejected a-posteriori, and
``gate_candidate.py`` persisted only ``str(cert.message)[:400]``. The recorded
sentence "Partial: V verified via sos_cross_term, W failed via failed" is
therefore compatible with four substantively different outcomes, one of which
(the rejected Gram) is a *proposed but invalid* certificate and one of which
(``unexpected_error``) is not evidence about the mathematics at all. N14 and
N15 in ``docs/papers/big_draft/NEGATIVES.md`` are the recorded cost of that
collapse: a deliberate ``return 1`` came back as the watchdog string
``unexpected_error`` with no solver status persisted anywhere.

Claim hygiene (binding decision D6)
-----------------------------------
``solver_infeasible`` is typed FLOAT evidence, never proof. N12 records a case
where exact covariant scaling flipped one. ``gram_rejected`` is likewise
evidence that the *proposal* was invalid, not that no certificate exists.
Nothing in this module makes a positive claim; it only makes rejections
distinguishable.

Design constraints
------------------
This module is deliberately dependency-free (stdlib only, no numpy, no cvxpy)
and imports nothing from the rest of the package, so it can be imported from a
driver script, from the verifier, and from a portable checker alike. It does
not modify the solve path: the per-octant diagnostics are *parsed back* out of
the solver-attempt trace that ``joint_sos._numeric_poly_is_sos`` already
writes, so no producer had to change.

Attempt-string grammar consumed here (all produced by ``joint_sos``, unchanged)::

    [oct<i>:]<solver>:<cvxpy status>[:postcheck_ok(min_eig=<f>,resid_l1=<f>)]
    [oct<i>:]<solver>:<cvxpy status>:gram_margin_exceeded(eta=..,C1=..,resid_l1=<f>,budget=..)
    [oct<i>:]<solver>:<cvxpy status>:gram_not_psd(min_eig=<f>)
    [oct<i>:]<solver>:<cvxpy status>:gram_residual(resid_l1=<f>)
    [oct<i>:]<solver>:error:<exception text>
    [oct<i>:]<solver>:unavailable
    [oct<i>:]cvxpy_unavailable | odd_homogeneous_degree | monomial_outside_basis
    [oct<i>:]empty_basis
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "FailureClass",
    "FAILURE_CLASSES",
    "classify_exception",
    "classify_attempts",
    "classify_failure",
    "parse_octant_diagnostics",
    "stage_report",
    "worst_failure_class",
]


class FailureClass:
    """The typed vocabulary. String constants, so records stay JSON-native."""

    #: The stage succeeded. Never written as a failure.
    NONE = "none"
    #: The stage was not attempted (upstream stage already failed, no dynamics).
    SKIPPED = "skipped"

    # --- resource / harness: says nothing about the mathematics -------------
    #: Out of memory (MemoryError, or a watchdog RSS kill).
    OOM = "oom"
    #: Wall-clock budget exhausted.
    TIMEOUT = "timeout"
    #: The harness could not run the problem at all (missing CVXPY, bad input,
    #: I/O). Not evidence about the candidate.
    HARNESS_ERROR = "harness_error"
    #: Operator interrupt (SIGINT / KeyboardInterrupt).
    INTERRUPTED = "interrupted"
    #: An exception nothing above explains. Deliberately last-resort: N14/N15
    #: show what happens when everything lands here.
    UNEXPECTED_ERROR = "unexpected_error"

    # --- solver-level: typed FLOAT evidence, never proof (D6) ---------------
    #: The SDP solver reported primal infeasibility. Typed float evidence.
    SOLVER_INFEASIBLE = "solver_infeasible"
    #: The solver returned unknown / stalled / unbounded / no usable status.
    SOLVER_UNKNOWN = "solver_unknown"
    #: The solver raised (cvxpy SolverError, numeric failure inside the solve).
    SOLVER_ERROR = "solver_error"
    #: The requested solver is not installed / not licensed on this box.
    SOLVER_UNAVAILABLE = "solver_unavailable"

    # --- a-posteriori rejection: the solver answered, we refused the answer --
    #: A Gram was recovered but ``_validate_gram_certificate`` rejected it
    #: (PSD margin / equality residual charged against the epsilon budget).
    #: The strongest failure signal available: the proposal was invalid.
    GRAM_REJECTED = "gram_rejected"

    # --- refutations: structural, no solver needed ---------------------------
    #: ``min_V_on_sphere`` refuted positive-definiteness. No SOS certificate
    #: can exist for this V; this one IS a refutation, not merely evidence.
    NOT_POSITIVE_DEFINITE = "not_positive_definite"
    #: The target cannot be SOS on structural grounds (odd homogeneous degree,
    #: a monomial outside any admissible basis, empty basis).
    STRUCTURAL_REFUTATION = "structural_refutation"
    #: W could not be assembled on some octant, or cleared W was identically
    #: zero. A modelling/assembly outcome, not a solver outcome.
    ASSEMBLY_ERROR = "assembly_error"
    #: The solve succeeded but the result was refused on scope grounds (the
    #: joint-SOS V redesign is certified against ONE vertex, so it is refused
    #: whenever inclusion vertices are present). An honest refusal, not a
    #: solver outcome and not evidence against the candidate.
    POLICY_REJECTED = "policy_rejected"


FAILURE_CLASSES = frozenset(
    v for k, v in vars(FailureClass).items() if not k.startswith("_") and isinstance(v, str)
)

# Most informative first. ``worst_failure_class`` and the per-attempt scan both
# use this order, so a stage that both stalled and had a Gram rejected reports
# the Gram rejection (the substantive fact) rather than the stall.
_PRECEDENCE: List[str] = [
    FailureClass.NOT_POSITIVE_DEFINITE,
    FailureClass.STRUCTURAL_REFUTATION,
    FailureClass.GRAM_REJECTED,
    FailureClass.SOLVER_INFEASIBLE,
    FailureClass.ASSEMBLY_ERROR,
    FailureClass.POLICY_REJECTED,
    FailureClass.SOLVER_ERROR,
    FailureClass.OOM,
    FailureClass.TIMEOUT,
    FailureClass.INTERRUPTED,
    FailureClass.SOLVER_UNAVAILABLE,
    FailureClass.HARNESS_ERROR,
    FailureClass.UNEXPECTED_ERROR,
    FailureClass.SOLVER_UNKNOWN,
    FailureClass.SKIPPED,
    FailureClass.NONE,
]

_RANK = {name: i for i, name in enumerate(_PRECEDENCE)}

# Solver-status keywords, checked in order (longest/most specific first).
_STATUS_MAP = [
    ("infeasible_inaccurate", FailureClass.SOLVER_INFEASIBLE),
    ("infeasible", FailureClass.SOLVER_INFEASIBLE),
    ("unbounded", FailureClass.SOLVER_UNKNOWN),
    ("solver_error", FailureClass.SOLVER_ERROR),
    ("unknown", FailureClass.SOLVER_UNKNOWN),
    ("stalled", FailureClass.SOLVER_UNKNOWN),
]

# Rejection reasons emitted by joint_sos._validate_gram_certificate.
_GRAM_REJECT_TOKENS = ("gram_margin_exceeded", "gram_not_psd", "gram_residual(")

_STRUCTURAL_TOKENS = ("odd_homogeneous_degree", "monomial_outside_basis", "empty_basis")

_ASSEMBLY_STATUSES = ("w_assembly_failed", "w_identically_zero")

_OCT_RE = re.compile(r"^oct(\d+):")
_MIN_EIG_RE = re.compile(r"min_eig=(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
_RESID_RE = re.compile(r"resid_l1=(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
_ETA_RE = re.compile(r"eta=(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
_BUDGET_RE = re.compile(r"budget=(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")


def _to_float(match) -> Optional[float]:
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):  # pragma: no cover - regex guarantees shape
        return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
def classify_exception(exc: BaseException) -> str:
    """Type a raised exception.

    ``MemoryError`` is ``oom``; ``TimeoutError`` (and
    ``subprocess.TimeoutExpired``, which subclasses ``SubprocessError`` and is
    matched by name) is ``timeout``; ``KeyboardInterrupt`` is ``interrupted``;
    an import/OS/value problem in the harness is ``harness_error``; a cvxpy
    ``SolverError`` is ``solver_error``. Everything else is
    ``unexpected_error``, which is the class this module exists to make rare.
    """
    if isinstance(exc, MemoryError):
        return FailureClass.OOM
    if isinstance(exc, KeyboardInterrupt):
        return FailureClass.INTERRUPTED
    if isinstance(exc, TimeoutError):
        return FailureClass.TIMEOUT

    name = type(exc).__name__
    if name in ("TimeoutExpired", "FunctionTimedOut", "Timeout"):
        return FailureClass.TIMEOUT
    if name == "SolverError":
        return FailureClass.SOLVER_ERROR

    text = f"{name}: {exc}".lower()
    if "out of memory" in text or "cannot allocate memory" in text:
        return FailureClass.OOM
    if "timed out" in text or "timeout" in text:
        return FailureClass.TIMEOUT
    if isinstance(exc, (ImportError, ModuleNotFoundError, OSError, FileNotFoundError)):
        return FailureClass.HARNESS_ERROR
    return FailureClass.UNEXPECTED_ERROR


# ---------------------------------------------------------------------------
# Attempt traces
# ---------------------------------------------------------------------------
def parse_octant_diagnostics(attempts: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    """Recover per-octant Gram diagnostics from a solver-attempt trace.

    ``joint_sos`` aggregates Gram diagnostics to worst-case scalars before they
    reach a result object, but it writes the per-octant numbers into the attempt
    strings. This parses them back, so a record can carry
    ``gram_min_eig`` / ``gram_residual_l1`` *per octant* without any change to
    the solve path.

    Returns one dict per attempt, in trace order, with keys ``octant_index``
    (``None`` when the producer wrote no ``oct<i>:`` prefix, e.g.
    ``verify_w_given_numeric``), ``solver``, ``solver_status``, ``accepted``,
    ``gram_min_eig``, ``gram_min_eig_source``, ``gram_residual_l1``,
    ``gram_eta``, ``eps_budget``, ``reject_reason``, ``failure_class`` and
    ``raw``.
    """
    out: List[Dict[str, Any]] = []
    for raw_attempt in attempts or []:
        raw = str(raw_attempt)
        rest = raw
        octant_index: Optional[int] = None
        m = _OCT_RE.match(rest)
        if m:
            octant_index = int(m.group(1))
            rest = rest[m.end():]

        solver = ""
        status = ""
        detail = ""
        if rest in _STRUCTURAL_TOKENS or rest in ("cvxpy_unavailable",):
            status = rest
        else:
            parts = rest.split(":", 2)
            if len(parts) == 1:
                status = parts[0]
            else:
                solver, status = parts[0], parts[1]
                detail = parts[2] if len(parts) > 2 else ""

        accepted = detail.startswith("postcheck_ok")
        reject_reason = "" if accepted else detail

        min_eig = _to_float(_MIN_EIG_RE.search(raw))
        eta = _to_float(_ETA_RE.search(raw))
        # gram_margin_exceeded reports eta = max(0, -λ_min) instead of λ_min.
        # For eta > 0 that inverts exactly, so the octant still gets its
        # gram_min_eig; for eta == 0 only "λ_min ≥ 0" is known, so we leave it
        # None rather than inventing a zero.
        min_eig_source = "min_eig" if min_eig is not None else ""
        if min_eig is None and eta is not None and eta > 0.0:
            min_eig = -eta
            min_eig_source = "eta"

        entry = {
            "octant_index": octant_index,
            "solver": solver,
            "solver_status": status,
            "accepted": accepted,
            "gram_min_eig": min_eig,
            "gram_min_eig_source": min_eig_source,
            "gram_residual_l1": _to_float(_RESID_RE.search(raw)),
            "gram_eta": eta,
            "eps_budget": _to_float(_BUDGET_RE.search(raw)),
            "reject_reason": reject_reason,
            "failure_class": _classify_attempt(status, detail, accepted),
            "raw": raw,
        }
        out.append(entry)
    return out


def _classify_attempt(status: str, detail: str, accepted: bool) -> str:
    """Type ONE parsed attempt. ``none`` when the attempt was accepted."""
    if accepted:
        return FailureClass.NONE

    status_low = str(status or "").lower()
    detail_low = str(detail or "").lower()

    # The a-posteriori Gram rejection is the most informative outcome: the
    # solver answered and we refused the answer.
    if any(tok in detail_low for tok in _GRAM_REJECT_TOKENS):
        return FailureClass.GRAM_REJECTED
    if status_low in _STRUCTURAL_TOKENS:
        return FailureClass.STRUCTURAL_REFUTATION
    if status_low == "cvxpy_unavailable":
        return FailureClass.HARNESS_ERROR
    if status_low == "unavailable":
        return FailureClass.SOLVER_UNAVAILABLE
    if status_low == "error":
        if "memory" in detail_low:
            return FailureClass.OOM
        if "timed out" in detail_low or "timeout" in detail_low:
            return FailureClass.TIMEOUT
        return FailureClass.SOLVER_ERROR
    for token, cls in _STATUS_MAP:
        if token in status_low:
            return cls
    # "optimal" with no accepted post-check: no usable Gram came back.
    return FailureClass.SOLVER_UNKNOWN


def classify_attempts(attempts: Optional[Iterable[str]]) -> Optional[str]:
    """Type a whole attempt trace, or ``None`` when there is nothing to type.

    Only the attempts of the LAST octant present are consulted: the octant loop
    breaks at the first failing octant, so the tail of the trace is the failure.
    Within that octant the most informative class wins (see ``_PRECEDENCE``).
    """
    entries = parse_octant_diagnostics(attempts)
    if not entries:
        return None

    indices = [e["octant_index"] for e in entries if e["octant_index"] is not None]
    if indices:
        last = max(indices)
        entries = [e for e in entries if e["octant_index"] == last]

    classes = [e["failure_class"] for e in entries if e["failure_class"] != FailureClass.NONE]
    if not classes:
        return None
    return worst_failure_class(classes)


def worst_failure_class(classes: Iterable[str]) -> str:
    """Pick the most informative class from an iterable (see ``_PRECEDENCE``)."""
    known = [c for c in classes if c in _RANK]
    if not known:
        return FailureClass.UNEXPECTED_ERROR
    return min(known, key=lambda c: _RANK[c])


# ---------------------------------------------------------------------------
# Stage-level classification
# ---------------------------------------------------------------------------
def classify_failure(
    *,
    verified: bool = False,
    solver_status: str = "",
    attempts: Optional[Iterable[str]] = None,
    exception: Optional[BaseException] = None,
    default: str = FailureClass.SOLVER_UNKNOWN,
) -> str:
    """Type one verification stage.

    Precedence: a raised exception wins (the stage did not produce a verdict),
    then the result-level ``solver_status``, then the attempt trace, then
    ``default``. ``verified=True`` short-circuits to ``none``.
    """
    if verified:
        return FailureClass.NONE
    if exception is not None:
        return classify_exception(exception)

    status = str(solver_status or "").strip().lower()
    if status == "sphere_refutation":
        return FailureClass.NOT_POSITIVE_DEFINITE
    if status in _ASSEMBLY_STATUSES:
        return FailureClass.ASSEMBLY_ERROR

    from_attempts = classify_attempts(attempts)
    if from_attempts is not None:
        return from_attempts

    if status == "standard_sos_infeasible":
        # No attempt trace survived, so we cannot say WHICH solver behaviour
        # produced it. Report the honest, weaker class.
        return FailureClass.SOLVER_UNKNOWN
    if status:
        for token, cls in _STATUS_MAP:
            if token in status:
                return cls
    return default


def stage_report(
    stage: str,
    *,
    verified: bool,
    method: str = "",
    meta: Optional[Dict[str, Any]] = None,
    vertex_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one JSON-native per-stage record from a solver-metadata dict.

    ``meta`` is the payload ``verifier._solver_metadata_from_result`` produces,
    optionally carrying ``failure_class`` / ``exception_type`` /
    ``exception_message`` written by the verifier's exception handlers.
    """
    meta = dict(meta or {})
    attempts = [str(a) for a in (meta.get("solver_attempts") or [])]
    octants = parse_octant_diagnostics(attempts)

    failure_class = meta.get("failure_class") or ""
    if not failure_class:
        failure_class = classify_failure(
            verified=verified,
            solver_status=str(meta.get("solver_status", "") or ""),
            attempts=attempts,
        )
    elif verified:
        failure_class = FailureClass.NONE

    return {
        "stage": stage,
        "vertex_index": vertex_index,
        "verified": bool(verified),
        "method": str(method or ""),
        "failure_class": str(failure_class),
        "solver_requested": str(meta.get("solver_requested", "") or ""),
        "solver_used": str(meta.get("solver_used", "") or ""),
        "solver_status": str(meta.get("solver_status", "") or ""),
        "solver_fallback_used": bool(meta.get("solver_fallback_used", False)),
        "solver_attempts": attempts,
        "gram_min_eig": meta.get("gram_min_eig"),
        "gram_residual_l1": meta.get("gram_residual_l1"),
        "gram_cond": meta.get("gram_cond"),
        "epsilon_used": meta.get("epsilon_used"),
        "exception_type": str(meta.get("exception_type", "") or ""),
        "exception_message": str(meta.get("exception_message", "") or ""),
        "octants": octants,
    }
