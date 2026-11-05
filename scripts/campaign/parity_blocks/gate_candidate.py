#!/usr/bin/env python3
"""Deferred UNREDUCED strict gate for persisted yparity/orbit candidates.

Loads a yparity_driver record with candidate_V_terms, rebuilds V, and runs
certify_discovered_gf (CLARABEL, strict, vertex-aware, both endpoints) at the
record's EFFECTIVE (possibly orbit-scaled) gains and delta0. Appends the gate
result to a new JSON next to the input record. Claim hygiene: a passing gate
makes the candidate a strict-gate-verified certificate AT THE SCALED GAINS;
the exact phase and the exact-scaling transfer statement are separate stages.

WP P3-02 — failure typing survives into the record
--------------------------------------------------
This writer used to persist ``str(cert.message)[:400]`` and nothing else, so a
rejection recorded "Partial: V verified via sos_cross_term, W failed via
failed": one string covering solver infeasibility, a solver ``unknown``, a
raised exception and a Gram that ``_validate_gram_certificate`` refused
a-posteriori. It now persists

  * ``gate.failure_class`` plus ``gate.V_failure_class`` /
    ``gate.W_failure_class`` — the typed class from
    ``gf_framework.verification.failure_typing.FailureClass``;
  * ``gate.solver_status`` and ``gate.solver_attempts`` — the raw solver
    status strings, never collapsed;
  * ``gate.gram_min_eig`` / ``gate.gram_residual_l1`` / ``gate.gram_cond`` —
    the worst-case a-posteriori Gram diagnostics, and ``gate.octants`` with
    the SAME two numbers PER OCTANT;
  * ``gate.stages`` — one entry per verification stage (V, then W per
    inclusion vertex), each with its own attempts and octants;
  * ``gate.exception_type`` / ``gate.exception_message`` /
    ``gate.traceback`` — populated when the harness itself failed.

A run that raises now still writes its record (typed ``oom`` / ``timeout`` /
``harness_error`` / ``unexpected_error``) and exits 2; a run that produced a
verdict exits 0 whether or not it certified. That is the N14/N15 lesson: a
deliberate non-pass and a crash must not arrive as the same thing.

Claim hygiene (D6): ``solver_infeasible`` in these records is typed FLOAT
evidence, never proof.

Fields are ADDITIVE. Every key the old writer emitted is still emitted with
the same name and meaning, so existing readers of ``*_GATE.json`` keep working.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

# P1-04 rescue edit: the two hard-coded absolute candidates (Linux $HOME and
# a Windows OneDrive path) are replaced by the repository-relative src/ of
# the checkout this file now lives in (scripts/campaign/<group>/).
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gf_framework.discovery.universal_discovery as ud  # noqa: E402
from gf_framework.algebra.core import GFTerm, GeneralizedForm  # noqa: E402
from gf_framework.verification.failure_typing import (  # noqa: E402
    FailureClass,
    classify_exception,
    stage_report,
)
from gf_framework.verification.verifier import certify_discovered_gf  # noqa: E402

#: Bumped whenever the gate record gains fields. Readers should treat a
#: missing key as "written before that version" (records are additive).
RECORD_SCHEMA_VERSION = "p3-02"

NOTE = (
    "UNREDUCED strict gate on a persisted orbit-representative candidate; "
    "certificate scope is the scaled gain point; exact-phase closure and the "
    "exact covariant transfer to the printed gains are separate, pending stages"
)


def _to_frac(s):
    return Fraction(s) if "/" in s else (int(s) if s.lstrip("-").isdigit()
                                         else Fraction(s))


def build_candidate(r: dict):
    """Rebuild (V, weights, m, delta0_effective, spec) from a driver record."""
    assert r.get("candidate_V_terms"), "record has no persisted candidate"
    m = int(r["m"])
    delta0_eff = float(r["delta0_effective"])
    qp = Fraction(r["qscale_p"]) if r.get("qscale_p") else None
    gains = {}
    if qp is not None:
        from gf_framework.systems.levant4d import _L1, _L2, _L3, _L4
        p = float(qp)
        gains = dict(lambda1=_L1 * p, lambda2=_L2 * p ** 2,
                     lambda3=_L3 * p ** 3, lambda4=_L4 * p ** 4)
    spec = ud.create_levant4d_spec(delta0=delta0_eff, **gains)
    weights = tuple(int(w) for w in spec.weights_r)
    V = GeneralizedForm(
        degree_m=m, weights_r=weights,
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(_to_frac(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in r["candidate_V_terms"]])
    return V, weights, m, delta0_eff, spec


def _stages_from_certificate(cert) -> list:
    """Per-stage reports, reconstructed for pre-P3-02 Certificate objects."""
    stages = getattr(cert, "stage_reports", None)
    if stages:
        return [dict(s) for s in stages]
    # Backward compatible: a Certificate without stage_reports still carries a
    # flat attempt trace, so emit one synthetic stage rather than nothing.
    return [
        stage_report(
            "certificate",
            verified=bool(getattr(cert, "V_verified", False))
            and bool(getattr(cert, "W_verified", False)),
            method=f"{getattr(cert, 'V_method', '')}/{getattr(cert, 'W_method', '')}",
            meta={
                "solver_requested": getattr(cert, "solver_requested", ""),
                "solver_used": getattr(cert, "solver_used", ""),
                "solver_status": getattr(cert, "solver_status", ""),
                "solver_fallback_used": getattr(cert, "solver_fallback_used", False),
                "solver_attempts": list(getattr(cert, "solver_attempts", []) or []),
                "gram_min_eig": getattr(cert, "gram_min_eig", None),
                "gram_residual_l1": getattr(cert, "gram_residual_l1", None),
                "gram_cond": getattr(cert, "gram_cond", None),
                "epsilon_used": getattr(cert, "epsilon_used", None),
            },
        )
    ]


def build_gate_block(cert, elapsed_s: float, solver: str = "CLARABEL") -> dict:
    """The ``gate`` block of the record, for a run that produced a verdict."""
    stages = _stages_from_certificate(cert)
    octants = []
    for st in stages:
        for oc in st.get("octants") or []:
            entry = dict(oc)
            entry["stage"] = st.get("stage", "")
            entry["vertex_index"] = st.get("vertex_index")
            octants.append(entry)

    certified = bool(getattr(cert, "surface_certified", False))
    failure_class = str(getattr(cert, "failure_class", "") or "")
    if not failure_class:
        failure_class = FailureClass.NONE if certified else FailureClass.SOLVER_UNKNOWN

    return {
        # --- fields the pre-P3-02 writer emitted, unchanged ---------------
        "certified": certified,
        "solver": solver,
        "strict": True,
        "vertex_aware": True,
        "time_s": round(elapsed_s, 1),
        "message": str(getattr(cert, "message", ""))[:400],
        # --- P3-02: the typed failure, never collapsed --------------------
        "failure_class": failure_class,
        "V_failure_class": str(getattr(cert, "V_failure_class", "") or ""),
        "W_failure_class": str(getattr(cert, "W_failure_class", "") or ""),
        "V_verified": bool(getattr(cert, "V_verified", False)),
        "W_verified": bool(getattr(cert, "W_verified", False)),
        "V_method": str(getattr(cert, "V_method", "") or ""),
        "W_method": str(getattr(cert, "W_method", "") or ""),
        # --- P3-02: the raw solver status strings -------------------------
        "solver_status": str(getattr(cert, "solver_status", "") or ""),
        "solver_requested": str(getattr(cert, "solver_requested", "") or ""),
        "solver_used": str(getattr(cert, "solver_used", "") or ""),
        "solver_fallback_used": bool(getattr(cert, "solver_fallback_used", False)),
        "solver_attempts": [str(a) for a in (getattr(cert, "solver_attempts", []) or [])],
        # --- P3-02: Gram residuals, worst case and per octant -------------
        "gram_min_eig": getattr(cert, "gram_min_eig", None),
        "gram_residual_l1": getattr(cert, "gram_residual_l1", None),
        "gram_cond": getattr(cert, "gram_cond", None),
        "epsilon_used": getattr(cert, "epsilon_used", None),
        "octants": octants,
        "stages": stages,
        # --- context -------------------------------------------------------
        "n_vertices": int(getattr(cert, "n_vertices", 1) or 1),
        "clearing_power": int(getattr(cert, "clearing_power", 0) or 0),
        "regularity": str(getattr(cert, "regularity", "") or ""),
        "surface_scope": str(getattr(cert, "surface_scope", "") or ""),
        "exception_type": "",
        "exception_message": "",
        "traceback": "",
    }


def build_failed_gate_block(exc: BaseException, elapsed_s: float,
                            solver: str = "CLARABEL") -> dict:
    """The ``gate`` block for a run the harness could not finish.

    Before P3-02 this run wrote NO record at all: the traceback escaped, the
    watchdog typed the non-zero exit as ``unexpected_error`` and the solver
    status was lost (N14, N15). The typed class is now on disk.
    """
    cls = classify_exception(exc)
    return {
        "certified": False,
        "solver": solver,
        "strict": True,
        "vertex_aware": True,
        "time_s": round(elapsed_s, 1),
        "message": f"gate did not complete: {type(exc).__name__}: {exc}"[:400],
        "failure_class": cls,
        "V_failure_class": cls,
        "W_failure_class": FailureClass.SKIPPED,
        "V_verified": False,
        "W_verified": False,
        "V_method": "",
        "W_method": "",
        "solver_status": "",
        "solver_requested": solver,
        "solver_used": "",
        "solver_fallback_used": False,
        "solver_attempts": [],
        "gram_min_eig": None,
        "gram_residual_l1": None,
        "gram_cond": None,
        "epsilon_used": None,
        "octants": [],
        "stages": [],
        "n_vertices": 0,
        "clearing_power": 0,
        "regularity": "",
        "surface_scope": "open_orthant_only",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc)[:400],
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:],
    }


def _arm_timeout(seconds: float):
    """Best-effort wall-clock budget; returns a disarm callable.

    A budget that fires raises ``TimeoutError``, which types as ``timeout``
    rather than as an anonymous kill. On platforms without SIGALRM (or off the
    main thread) this is a no-op and the run is simply not bounded here.
    """
    if not seconds or seconds <= 0:
        return lambda: None
    try:
        import signal

        def _fire(_signum, _frame):
            raise TimeoutError(f"gate exceeded its {seconds:g} s budget")

        previous = signal.signal(signal.SIGALRM, _fire)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))

        def _disarm():
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)

        return _disarm
    except (ImportError, AttributeError, ValueError):
        return lambda: None


def run_gate(rec_path: Path, timeout_s: float = 0.0, verbose: bool = False) -> dict:
    """Run the gate for one driver record and return the full record dict.

    Never raises for a solver outcome OR for a harness failure: both come back
    as a record with a typed ``gate.failure_class``.
    """
    rec_path = Path(rec_path).expanduser()
    r: dict = {}
    m, delta0_eff = 0, 0.0
    t0 = time.time()
    disarm = _arm_timeout(timeout_s)
    try:
        r = json.loads(rec_path.read_text())
        V, weights, m, delta0_eff, spec = build_candidate(r)
        cert = certify_discovered_gf(
            V, weights, m, spec.dynamics_gf, solver="CLARABEL",
            strict_solver=True,
            dynamics_gf_vertices=spec.dynamics_gf_vertices,
            disturbance_bound=delta0_eff,
            verbose=verbose)
        gate = build_gate_block(cert, time.time() - t0)
    except BaseException as exc:  # noqa: BLE001 - typed, re-raised as a record
        if isinstance(exc, SystemExit):
            raise
        gate = build_failed_gate_block(exc, time.time() - t0)
        m = int(r.get("m", 0) or 0)
        delta0_eff = float(r.get("delta0_effective", 0.0) or 0.0)
    finally:
        disarm()

    return dict(
        input_record=str(rec_path.name), m=m,
        delta0_effective=delta0_eff, qscale_p=r.get("qscale_p"),
        gate=gate,
        note=NOTE,
        record_schema_version=RECORD_SCHEMA_VERSION,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


def gate_output_path(rec_path: Path) -> Path:
    rec_path = Path(rec_path)
    return rec_path.with_name(rec_path.stem + "_GATE.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", help="yparity driver record with candidate_V_terms")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="wall-clock budget in seconds; on expiry the record is "
                         "written with failure_class=timeout (0 = unbounded)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    rec_path = Path(args.record).expanduser()
    out = run_gate(rec_path, timeout_s=args.timeout, verbose=args.verbose)
    out_path = gate_output_path(rec_path)
    out_path.write_text(json.dumps(out, indent=1))

    gate = out["gate"]
    print(f"gate certified={gate['certified']} "
          f"failure_class={gate['failure_class']} "
          f"solver_status={gate['solver_status'] or '-'} "
          f"({gate['time_s']}s) -> {out_path}", flush=True)
    # A non-certifying gate is a datum (exit 0). A harness failure is not: it
    # exits 2 so a lane script can tell the two apart, with the record on disk
    # either way.
    harness_failed = bool(gate.get("exception_type"))
    return 2 if harness_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
