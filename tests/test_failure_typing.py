"""WP P3-02: a typed failure class and the raw solver status must survive
into the record.

Before P3-02 every rejection collapsed to the literal method string "failed"
(``verifier._build_message``) and to ``str(cert.message)[:400]``
(``gate_candidate.py``). Solver infeasibility, a solver ``unknown``, a raised
exception and an a-posteriori Gram rejection were indistinguishable on disk,
which is the failure mode N14/N15 record.

Three deliberate failures are exercised end to end into the JSON record:
a forced TIMEOUT, a forced EXCEPTION, and a genuine INFEASIBLE small cell
(solved for real, CLARABEL-first, no MOSEK required).

Claim hygiene (D6): nothing here asserts that a typed ``solver_infeasible`` is
a proof. It is typed float evidence, and these tests only pin that the type
is recorded rather than collapsed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.verification.failure_typing import (
    FailureClass,
    classify_attempts,
    classify_exception,
    classify_failure,
    parse_octant_diagnostics,
    stage_report,
    worst_failure_class,
)
from gf_framework.verification.verifier import certify_discovered_gf

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_CANDIDATE_PY = REPO_ROOT / "scripts" / "campaign" / "parity_blocks" / "gate_candidate.py"


def _load_gate_candidate():
    """Import the rescued driver by path (scripts/ is outside testpaths)."""
    spec = importlib.util.spec_from_file_location("gate_candidate_p302", GATE_CANDIDATE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# 1. The vocabulary and the attempt-trace parser
# =============================================================================
class TestAttemptParsing:
    """joint_sos writes per-octant Gram numbers into the attempt strings; the
    parser recovers them without any change to the solve path."""

    def test_accepted_attempt_carries_both_gram_numbers(self):
        entries = parse_octant_diagnostics(
            ["oct3:MOSEK:optimal:postcheck_ok(min_eig=1.23e-04,resid_l1=5.00e-13)"]
        )
        assert len(entries) == 1
        e = entries[0]
        assert e["octant_index"] == 3
        assert e["solver"] == "MOSEK"
        assert e["solver_status"] == "optimal"
        assert e["accepted"] is True
        assert e["failure_class"] == FailureClass.NONE
        assert e["gram_min_eig"] == pytest.approx(1.23e-4)
        assert e["gram_residual_l1"] == pytest.approx(5.0e-13)

    def test_margin_rejection_is_gram_rejected_and_inverts_eta(self):
        e = parse_octant_diagnostics(
            ["oct0:CLARABEL:optimal:gram_margin_exceeded("
             "eta=2.60e-06,C1=1.5,resid_l1=1.05e-07,budget=5.00e-07)"]
        )[0]
        assert e["failure_class"] == FailureClass.GRAM_REJECTED
        assert e["gram_eta"] == pytest.approx(2.60e-6)
        # eta = max(0, -lambda_min), so lambda_min is recovered exactly.
        assert e["gram_min_eig"] == pytest.approx(-2.60e-6)
        assert e["gram_min_eig_source"] == "eta"
        assert e["gram_residual_l1"] == pytest.approx(1.05e-7)
        assert e["eps_budget"] == pytest.approx(5.00e-7)

    def test_each_raw_solver_status_gets_its_own_class(self):
        cases = {
            "oct0:CLARABEL:infeasible": FailureClass.SOLVER_INFEASIBLE,
            "oct0:MOSEK:infeasible_inaccurate": FailureClass.SOLVER_INFEASIBLE,
            "oct0:SCS:unbounded": FailureClass.SOLVER_UNKNOWN,
            "oct0:MOSEK:unknown": FailureClass.SOLVER_UNKNOWN,
            "oct0:MOSEK:solver_error": FailureClass.SOLVER_ERROR,
            "oct0:MOSEK:error:license expired": FailureClass.SOLVER_ERROR,
            "MOSEK:unavailable": FailureClass.SOLVER_UNAVAILABLE,
            "cvxpy_unavailable": FailureClass.HARNESS_ERROR,
            "odd_homogeneous_degree": FailureClass.STRUCTURAL_REFUTATION,
            "monomial_outside_basis": FailureClass.STRUCTURAL_REFUTATION,
            "oct1:CLARABEL:optimal:gram_not_psd(min_eig=-3.10e-05)":
                FailureClass.GRAM_REJECTED,
            "oct1:CLARABEL:optimal:gram_residual(resid_l1=2.24e-11)":
                FailureClass.GRAM_REJECTED,
        }
        for attempt, expected in cases.items():
            got = parse_octant_diagnostics([attempt])[0]["failure_class"]
            assert got == expected, f"{attempt!r} -> {got}, expected {expected}"

    def test_only_the_last_octant_decides(self):
        # The octant loop breaks at the first failure, so the tail is the failure.
        attempts = [
            "oct0:CLARABEL:optimal:postcheck_ok(min_eig=1e-3,resid_l1=1e-14)",
            "oct1:CLARABEL:infeasible",
        ]
        assert classify_attempts(attempts) == FailureClass.SOLVER_INFEASIBLE

    def test_gram_rejection_outranks_infeasible_in_the_same_octant(self):
        attempts = [
            "oct0:CLARABEL:infeasible",
            "oct0:MOSEK:optimal:gram_margin_exceeded("
            "eta=1e-9,C1=1.5,resid_l1=4e-12,budget=5e-13)",
        ]
        assert classify_attempts(attempts) == FailureClass.GRAM_REJECTED
        assert worst_failure_class(
            [FailureClass.SOLVER_UNKNOWN, FailureClass.GRAM_REJECTED]
        ) == FailureClass.GRAM_REJECTED

    def test_empty_trace_types_nothing(self):
        assert classify_attempts([]) is None
        assert classify_attempts(None) is None


class TestExceptionTyping:
    def test_oom_timeout_interrupt_and_last_resort(self):
        assert classify_exception(MemoryError("out of memory")) == FailureClass.OOM
        assert classify_exception(TimeoutError("budget")) == FailureClass.TIMEOUT
        assert classify_exception(KeyboardInterrupt()) == FailureClass.INTERRUPTED
        assert classify_exception(ImportError("no mosek")) == FailureClass.HARNESS_ERROR
        assert classify_exception(ValueError("nope")) == FailureClass.UNEXPECTED_ERROR

    def test_subprocess_timeout_is_typed_by_name(self):
        import subprocess

        exc = subprocess.TimeoutExpired(cmd="x", timeout=1.0)
        assert classify_exception(exc) == FailureClass.TIMEOUT

    def test_exception_beats_status_and_attempts(self):
        cls = classify_failure(
            solver_status="standard_sos_infeasible",
            attempts=["oct0:CLARABEL:infeasible"],
            exception=MemoryError(),
        )
        assert cls == FailureClass.OOM

    def test_sphere_refutation_is_a_refutation_not_a_solver_outcome(self):
        assert classify_failure(solver_status="sphere_refutation") == \
            FailureClass.NOT_POSITIVE_DEFINITE
        assert classify_failure(solver_status="w_assembly_failed") == \
            FailureClass.ASSEMBLY_ERROR

    def test_verified_stage_is_never_a_failure(self):
        assert classify_failure(verified=True, solver_status="anything") == \
            FailureClass.NONE


class TestStageReport:
    def test_report_is_json_native_and_carries_octants(self):
        rep = stage_report(
            "W", verified=False, method="failed", vertex_index=1,
            meta={
                "solver_requested": "CLARABEL",
                "solver_status": "standard_sos_infeasible",
                "solver_attempts": [
                    "oct0:CLARABEL:optimal:postcheck_ok(min_eig=1e-3,resid_l1=1e-14)",
                    "oct1:CLARABEL:infeasible",
                ],
                "gram_min_eig": -1.0e-6,
                "gram_residual_l1": 2.0e-9,
            },
        )
        json.dumps(rep)  # must survive a record round-trip
        assert rep["stage"] == "W"
        assert rep["vertex_index"] == 1
        assert rep["failure_class"] == FailureClass.SOLVER_INFEASIBLE
        assert rep["solver_status"] == "standard_sos_infeasible"
        assert len(rep["octants"]) == 2
        assert rep["octants"][0]["gram_min_eig"] == pytest.approx(1e-3)


# =============================================================================
# 2. The verifier: a real, deliberately infeasible small cell
# =============================================================================
def _indefinite_candidate():
    """V = (x1 - x2)^2 with r = (1,1), m = 2.

    PSD but NOT positive definite (it vanishes on x1 = x2), so no strict
    epsilon-margin SOS certificate exists: the solver reports infeasible on
    the top rung and the recovered Grams on the lower rungs are rejected
    a-posteriori by _validate_gram_certificate. A small, fast, deterministic
    rejection that goes all the way through the SDP.
    """
    return GeneralizedForm(2, (1, 1), [
        GFTerm(1.0, (F(2), F(0)), ("abs", "abs")),
        GFTerm(-2.0, (F(1), F(1)), ("signed", "signed")),
        GFTerm(1.0, (F(0), F(2)), ("abs", "abs")),
    ])


def _linear_stable_dynamics():
    f1 = GeneralizedForm(1, (1, 1), [GFTerm(1.0, (F(0), F(1)), ("abs", "signed"))])
    f2 = GeneralizedForm(1, (1, 1), [
        GFTerm(-1.0, (F(1), F(0)), ("signed", "abs")),
        GFTerm(-1.0, (F(0), F(1)), ("abs", "signed")),
    ])
    return [f1, f2]


@pytest.fixture(scope="module")
def infeasible_cell_certificate():
    return certify_discovered_gf(
        _indefinite_candidate(), (1, 1), 2, _linear_stable_dynamics(),
        solver="CLARABEL",
    )


class TestInfeasibleSmallCell:
    def test_stage_fails_with_a_specific_class_not_the_word_failed(
        self, infeasible_cell_certificate
    ):
        cert = infeasible_cell_certificate
        assert cert.V_verified is False
        # The rejection is specific: either the solver said infeasible or the
        # recovered Gram was refused a-posteriori. Never a bare "failed".
        assert cert.V_failure_class in (
            FailureClass.SOLVER_INFEASIBLE, FailureClass.GRAM_REJECTED
        )
        assert cert.failure_class == cert.V_failure_class
        assert cert.W_failure_class == FailureClass.SKIPPED

    def test_raw_solver_status_survives(self, infeasible_cell_certificate):
        assert infeasible_cell_certificate.solver_status == "standard_sos_infeasible"

    def test_message_names_the_class(self, infeasible_cell_certificate):
        msg = infeasible_cell_certificate.message
        assert f"[{infeasible_cell_certificate.V_failure_class}]" in msg

    def test_per_octant_gram_numbers_reach_the_stage_report(
        self, infeasible_cell_certificate
    ):
        reports = infeasible_cell_certificate.stage_reports
        assert reports and reports[0]["stage"] == "V"
        octants = reports[0]["octants"]
        assert octants, "the failing stage must carry its per-octant trace"
        assert any(o["gram_residual_l1"] is not None for o in octants)
        assert any(o["gram_min_eig"] is not None for o in octants)
        assert all(o["octant_index"] is not None for o in octants)

    def test_a_passing_stage_is_typed_none(self):
        # Same machinery on a candidate that DOES certify: no false failure.
        #
        # The dynamics here are diagonal (x1' = -x1, x2' = -x2), NOT
        # ``_linear_stable_dynamics`` (x1' = x2, x2' = -x1 - x2). With the latter,
        # V = x1^2 + x2^2 gives W = 2*x2^2, which vanishes on the whole x1 axis:
        # a LaSalle situation, not a strict certificate. The old Pólya branch
        # accepted it because it only inspected the monomials PRESENT in the
        # sparse form (regression from the 2026-09-05 external scientific review,
        # finding 1); the closed-orthant rule now refuses it, correctly.
        V = GeneralizedForm(2, (1, 1), [
            GFTerm(1.0, (F(2), F(0)), ("abs", "abs")),
            GFTerm(1.0, (F(0), F(2)), ("abs", "abs")),
        ])
        diagonal_stable = [
            GeneralizedForm(1, (1, 1), [GFTerm(-1.0, (F(1), F(0)), ("signed", "abs"))]),
            GeneralizedForm(1, (1, 1), [GFTerm(-1.0, (F(0), F(1)), ("abs", "signed"))]),
        ]
        cert = certify_discovered_gf(
            V, (1, 1), 2, diagonal_stable, solver="CLARABEL"
        )
        assert cert.V_verified and cert.W_verified
        assert cert.failure_class == FailureClass.NONE
        assert cert.V_failure_class == FailureClass.NONE
        assert cert.W_failure_class == FailureClass.NONE


class TestVerifierExceptionTyping:
    """A raised exception must not come back as an anonymous rejection."""

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (TimeoutError("wall clock"), FailureClass.TIMEOUT),
            (MemoryError("oom"), FailureClass.OOM),
            (RuntimeError("boom"), FailureClass.UNEXPECTED_ERROR),
        ],
    )
    def test_stage_exception_is_typed_in_the_certificate(self, monkeypatch, exc, expected):
        import gf_framework.verification.verifier as vmod

        def _raise(*_a, **_k):
            raise exc

        monkeypatch.setattr(vmod, "verify_gf_sos_full", _raise)
        cert = certify_discovered_gf(
            _indefinite_candidate(), (1, 1), 2, _linear_stable_dynamics(),
            solver="CLARABEL",
        )
        assert cert.V_verified is False
        assert cert.V_failure_class == expected
        assert cert.failure_class == expected
        assert cert.stage_reports[0]["exception_type"] == type(exc).__name__
        assert f"[{expected}]" in cert.message


# =============================================================================
# 3. The record writer: the acceptance check of P3-02
# =============================================================================
def _driver_record(tmp_path: Path) -> Path:
    """Minimal yparity-driver-shaped record (the gate is stubbed in these
    tests, so the candidate terms only have to be well formed)."""
    rec = {
        "m": 9,
        "delta0_effective": 1.0,
        "qscale_p": "1/2",
        "candidate_V_terms": [
            {"coefficient": 1.0, "exponents": ["9/4", "0", "0", "0"],
             "term_types": ["abs", "abs", "abs", "abs"]},
        ],
    }
    path = tmp_path / "yparity_levant4d_m9_test.json"
    path.write_text(json.dumps(rec))
    return path


class TestGateRecord:
    def test_forced_timeout_is_typed_timeout_in_the_json(self, tmp_path, monkeypatch):
        gc = _load_gate_candidate()

        def _timeout(*_a, **_k):
            raise TimeoutError("gate exceeded its 1 s budget")

        monkeypatch.setattr(gc, "certify_discovered_gf", _timeout)
        rec_path = _driver_record(tmp_path)
        rc = gc.main([str(rec_path)])

        out = json.loads(gc.gate_output_path(rec_path).read_text())
        gate = out["gate"]
        assert gate["failure_class"] == FailureClass.TIMEOUT
        assert gate["exception_type"] == "TimeoutError"
        assert gate["certified"] is False
        assert gate["traceback"]
        # A harness failure is not a datum: it must be distinguishable by exit
        # code from a gate that ran and did not certify.
        assert rc == 2

    def test_forced_exception_is_typed_unexpected_error(self, tmp_path, monkeypatch):
        gc = _load_gate_candidate()

        def _boom(*_a, **_k):
            raise RuntimeError("solver harness blew up")

        monkeypatch.setattr(gc, "certify_discovered_gf", _boom)
        rec_path = _driver_record(tmp_path)
        rc = gc.main([str(rec_path)])

        gate = json.loads(gc.gate_output_path(rec_path).read_text())["gate"]
        assert gate["failure_class"] == FailureClass.UNEXPECTED_ERROR
        assert gate["exception_type"] == "RuntimeError"
        assert "solver harness blew up" in gate["exception_message"]
        assert rc == 2

    def test_forced_oom_is_typed_oom(self, tmp_path, monkeypatch):
        gc = _load_gate_candidate()
        monkeypatch.setattr(
            gc, "certify_discovered_gf",
            lambda *_a, **_k: (_ for _ in ()).throw(MemoryError()),
        )
        rec_path = _driver_record(tmp_path)
        gc.main([str(rec_path)])
        gate = json.loads(gc.gate_output_path(rec_path).read_text())["gate"]
        assert gate["failure_class"] == FailureClass.OOM

    def test_infeasible_small_cell_writes_status_attempts_and_octant_grams(
        self, tmp_path, monkeypatch, infeasible_cell_certificate
    ):
        """P3-02 acceptance: a deliberately failing run writes solver_status,
        solver_attempts and per-octant gram_min_eig / gram_residual_l1."""
        gc = _load_gate_candidate()
        monkeypatch.setattr(
            gc, "certify_discovered_gf",
            lambda *_a, **_k: infeasible_cell_certificate,
        )
        rec_path = _driver_record(tmp_path)
        rc = gc.main([str(rec_path)])
        assert rc == 0, "a gate that ran and did not certify is a datum, exit 0"

        gate = json.loads(gc.gate_output_path(rec_path).read_text())["gate"]
        assert gate["certified"] is False
        assert gate["failure_class"] in (
            FailureClass.SOLVER_INFEASIBLE, FailureClass.GRAM_REJECTED
        )
        assert gate["solver_status"] == "standard_sos_infeasible"
        assert gate["solver_attempts"], "the raw attempt trace must be persisted"
        assert any("infeasible" in a for a in gate["solver_attempts"])
        assert gate["octants"], "per-octant diagnostics must be persisted"
        assert any(o["gram_min_eig"] is not None for o in gate["octants"])
        assert any(o["gram_residual_l1"] is not None for o in gate["octants"])
        assert gate["stages"], "one entry per verification stage"
        assert gate["stages"][0]["stage"] == "V"

    def test_record_stays_backward_compatible(self, tmp_path, monkeypatch,
                                              infeasible_cell_certificate):
        """Every key the pre-P3-02 writer emitted is still emitted."""
        gc = _load_gate_candidate()
        monkeypatch.setattr(
            gc, "certify_discovered_gf",
            lambda *_a, **_k: infeasible_cell_certificate,
        )
        rec_path = _driver_record(tmp_path)
        gc.main([str(rec_path)])
        out = json.loads(gc.gate_output_path(rec_path).read_text())

        for key in ("input_record", "m", "delta0_effective", "qscale_p",
                    "gate", "note", "timestamp_utc"):
            assert key in out, f"pre-P3-02 record key {key!r} disappeared"
        for key in ("certified", "solver", "strict", "vertex_aware",
                    "time_s", "message"):
            assert key in out["gate"], f"pre-P3-02 gate key {key!r} disappeared"
        assert out["gate"]["solver"] == "CLARABEL"
        assert out["gate"]["strict"] is True
        assert out["gate"]["vertex_aware"] is True
        assert out["record_schema_version"] == gc.RECORD_SCHEMA_VERSION

    def test_old_records_still_load(self, tmp_path):
        """A pre-P3-02 *_GATE.json has none of the new keys; readers must cope
        by treating them as absent, which is what .get() does."""
        old = {
            "input_record": "yparity_levant4d_m9_old.json", "m": 9,
            "delta0_effective": 1.0, "qscale_p": "1/2",
            "gate": {"certified": False, "solver": "CLARABEL", "strict": True,
                     "vertex_aware": True, "time_s": 8431.3,
                     "message": "Partial: V verified via sos_cross_term, "
                                "W failed via failed"},
            "note": "...", "timestamp_utc": "20260803T190239Z",
        }
        path = tmp_path / "old_GATE.json"
        path.write_text(json.dumps(old))
        loaded = json.loads(path.read_text())
        assert loaded["gate"].get("failure_class", "") == ""
        assert loaded["gate"].get("octants", []) == []
        assert loaded["gate"]["certified"] is False


class TestCertificateBackwardCompatibility:
    def test_new_certificate_fields_default_to_empty(self):
        from gf_framework.interfaces import Certificate

        cert = Certificate(V_verified=True, W_verified=True,
                           V_method="polya", W_method="polya")
        assert cert.V_failure_class == ""
        assert cert.W_failure_class == ""
        assert cert.failure_class == ""
        assert cert.stage_reports is None

    def test_gate_block_handles_a_certificate_without_stage_reports(self):
        from gf_framework.interfaces import Certificate

        gc = _load_gate_candidate()
        cert = Certificate(
            V_verified=True, W_verified=False, V_method="sos_cross_term",
            W_method="failed", solver_status="standard_sos_infeasible",
            solver_attempts=["oct0:CLARABEL:infeasible"],
        )
        cert.stage_reports = None
        block = gc.build_gate_block(cert, 12.0)
        assert block["certified"] is False
        assert block["solver_status"] == "standard_sos_infeasible"
        assert block["stages"], "a synthetic stage is emitted for old objects"
        assert block["octants"][0]["failure_class"] == FailureClass.SOLVER_INFEASIBLE
        # No stage_reports means no typed class on the object itself; the
        # writer must still name an honest, weaker class rather than "none".
        assert block["failure_class"] == FailureClass.SOLVER_UNKNOWN
