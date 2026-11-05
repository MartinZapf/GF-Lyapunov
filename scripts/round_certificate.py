#!/usr/bin/env python3
"""X9: certified printed coefficients (tier A, and the tier-B helpers).

STDLIB ONLY. Standalone tool. Verdicts come from the portable verifier
(verification/standalone/verify_certificate.py, "VC") run as a SUBPROCESS,
exactly the way a reader would run it, on a record file this tool wrote to
disk. VC itself is never edited or monkeypatched for a verdict.

Tier A (2026-09-24, extended 2026-09-25)
----------------------------------------
Round V's coefficients to N significant digits, write them as exact rational
strings, keep the Gram sidecar byte for byte, and run VC. Increase N until
VERIFIED. A VERIFIED run is a complete P proof of the printed record.

Why the SAME Gram can prove a rounded V: the record schema has no hash or
other identity token binding V's coefficients to the Gram sidecar
(``source_artifacts`` sha256 entries are provenance VC never reads back). VC
RECOMPUTES every octant target from whatever V the record states and checks
the exact inequality

    charge = eta*C1 + ||E||_1 < eps,           E = target - z^T Q z,

so the record is bound to its Gram only by that arithmetic fact. Rounding a
coefficient moves the target, hence E, by a bounded amount; while the charge
stays below eps the unchanged Gram still witnesses the rounded target. Past
that point VC reports FAILED; nothing here weakens that check.

Units of N. ``--unit decimal`` (the 2026-09-24 demo): round to N significant
DECIMAL digits and print the coarsest dyadic within half a decimal ULP of
that decimal. ``--unit bits`` (2026-09-25, the printed-format decision:
dyadic): round to N significant BINARY digits, i.e. |c| = p * 2^e with
2^(N-1) <= p < 2^N (round half to even, exact), printed as the reduced
fraction ``p'/2^k``. Both are dyadic, so no step needs TC (NEXT_STEPS WP TC):
every such value with N <= 53 is also an exact float64.

Margin declarations. A portable record (EM, schema 2) declares its proved
margin (``eps_certified``, ``eps_certified_exact`` ...). VC refuses a record
whose declared margin exceeds the margin it derives (``MARGIN CLAIM EXCEEDS
PROOF``). Rounding V lowers the margin, so a rounded candidate must not carry
the source's declaration: :func:`printed_record` moves every declaration into
``provenance.printed_coefficients.source_margin_declarations`` (unverified
provenance, ignored by VC). After VC has VERIFIED the candidate,
:func:`declare_margin` writes VC's OWN derived margin back into the final
record (only if the source record had declarations, so the record keeps its
shape), and the final record is run through VC once more, independently.

The exact residual screen (``--strategy predict``)
-------------------------------------------------
For an expensive record (levant5d: about 10 h per VC run) a linear VC scan is
not affordable. The screen computes, for each N, the exact rational residual
``||E||_1`` of every octant of every form, with VC's own code for everything
(record parsing, dynamics, corners, targets, coverage), replacing only
``validate_octant`` by a residual-only copy that stops before the PSD test.
Since ``eta * C1 >= 0``, ``charge >= ||E||_1``; so a residual ratio
``||E||_1 / eps >= 1`` on any octant PROVES that VC fails at that N (VC's
ladder returns ``budget_exceeded`` at the first rung). A ratio < 1 proves
nothing by itself (the PSD shift may still add charge; it is zero whenever the
captured Gram is PSD, which is the case on every stored record). The screen
therefore only decides where VC is RUN: at the smallest N whose screen ratio
is < 1, and upward from there until VC VERIFIES. The screen imports VC as a
module in THIS process; its numbers are never a verdict.

Tier-B helpers
--------------
``round-row`` writes a row JSON (``V_terms``) with the rounded coefficients as
float64 numbers, asserting that each is the exact dyadic value (so the
fixed-V drivers, which read coefficients with ``float()``, consume exactly the
printed value). ``canonicalize`` rewrites the V coefficients of an EM record
(hex floats) as exact ``p/q`` strings, asserting exact equality with the
rounded row, so the printed string and the record string are identical.

Usage
-----
    # tier A (legacy flat CLI, decimal digits, one record)
    python scripts/round_certificate.py --record RECORD.json[.gz] \\
        --out-dir DIR [--sig-min 1] [--sig-max 20] [--python PY] [--json OUT]

    # tier A, one FUNCTION (all records that carry the same V), binary digits
    python scripts/round_certificate.py tier-a --label ID \\
        --record R1 [--record R2 ...] --out-dir DIR --unit bits \\
        --n-min 1 --n-max 53 --strategy scan|predict --finalize \\
        --python /usr/bin/python3 --json OUT

    python scripts/round_certificate.py predict --record R --n 12 --unit bits
    python scripts/round_certificate.py round-row --row ROW --n 8 --unit bits --out ROW_OUT
    python scripts/round_certificate.py canonicalize --record EMITTED \\
        --row ROW_OUT --out PRINTED.record.json
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

UNITS = ("decimal", "bits")

#: Top-level fields of a portable record that DECLARE a proved margin (VC
#: checks them against its own derivation) or report numbers that describe
#: the source V's proof. They are stale for a rounded V.
MARGIN_DECLARATION_FIELDS = (
    "eps_certified",
    "eps_certified_exact",
    "eps_certified_exact_is_float_transcription",
    "eps_certified_source",
    "note_half_margin_holds",
    "worst_charge_over_eps",
    "verifier",
)


class RoundCertificateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Exact rational helpers.
# ---------------------------------------------------------------------------

def decimal_exponent(x: Fraction) -> int:
    """The integer e with 10**e <= x < 10**(e+1), for x > 0, exact."""
    if x <= 0:
        raise RoundCertificateError(f"decimal_exponent: need x > 0, got {x}")
    try:
        e = int(math.floor(math.log10(float(x))))
    except (OverflowError, ValueError):
        e = 0
    while Fraction(10) ** (e + 1) <= x:
        e += 1
    while x < Fraction(10) ** e:
        e -= 1
    return e


def binary_exponent(x: Fraction) -> int:
    """The integer e with 2**e <= x < 2**(e+1), for x > 0, exact."""
    if x <= 0:
        raise RoundCertificateError(f"binary_exponent: need x > 0, got {x}")
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    while x < Fraction(2) ** e:
        e -= 1
    return e


def round_nearest_int(x: Fraction) -> int:
    """Round a nonnegative Fraction to the nearest integer (ties up), exact."""
    if x < 0:
        raise RoundCertificateError("round_nearest_int expects x >= 0")
    num, den = x.numerator, x.denominator
    return (2 * num + den) // (2 * den)


def round_to_dyadic_sig(c: Fraction, sig: int) -> Fraction:
    """Round ``c`` to ``sig`` significant DECIMAL digits, returned as the
    coarsest dyadic (power-of-2 denominator) rational within half a
    decimal-ULP of that N-significant-digit decimal value. Exact throughout.
    """
    if sig < 1:
        raise RoundCertificateError(f"sig must be >= 1, got {sig}")
    if c == 0:
        return Fraction(0)
    sign = 1 if c > 0 else -1
    a = abs(c)
    e = decimal_exponent(a)
    tol = Fraction(1, 2) * Fraction(10) ** (e - sig + 1)  # half-ULP at `sig` digits
    k = 0
    while Fraction(1, 2 ** (k + 1)) >= tol:
        k += 1
    p = round_nearest_int(a * (2 ** k))
    value = Fraction(p, 2 ** k)
    return sign * value


def round_to_dyadic_bits(c: Fraction, bits: int) -> Fraction:
    """Round ``c`` to ``bits`` significant BINARY digits, exact.

    With 2^e <= |c| < 2^(e+1) the grid is 2^(e-bits+1); ``|c|`` is rounded to
    the nearest grid point, ties to even (Python's exact ``round`` on a
    Fraction). The result is ``p * 2^(e-bits+1)`` with ``p <= 2^bits``, a
    dyadic rational, and ``|result - c| <= 2^(e-bits)``.
    """
    if bits < 1:
        raise RoundCertificateError(f"bits must be >= 1, got {bits}")
    c = Fraction(c)
    if c == 0:
        return Fraction(0)
    sign = 1 if c > 0 else -1
    a = abs(c)
    e = binary_exponent(a)
    ulp = Fraction(2) ** (e - bits + 1)
    p = round(a / ulp)  # exact, half to even
    return sign * p * ulp


def round_coefficient(c: Fraction, n: int, unit: str) -> Fraction:
    if unit == "decimal":
        return round_to_dyadic_sig(c, n)
    if unit == "bits":
        return round_to_dyadic_bits(c, n)
    raise RoundCertificateError(f"unknown unit {unit!r}; expected one of {UNITS}")


def parse_coefficient(raw) -> Fraction:
    """A V coefficient as VC reads it: JSON number, hex-float string or an
    exact rational string. Exact."""
    if isinstance(raw, bool):
        raise RoundCertificateError(f"boolean coefficient {raw!r}")
    if isinstance(raw, str) and raw.strip().lower().lstrip("+-").startswith("0x"):
        return Fraction(float.fromhex(raw))
    if isinstance(raw, float):
        return Fraction(raw)
    return Fraction(raw)


def exact_string(x: Fraction) -> str:
    x = Fraction(x)
    return f"{x.numerator}/{x.denominator}"


def is_dyadic(x: Fraction) -> bool:
    d = Fraction(x).denominator
    return d & (d - 1) == 0


def significant_bits(x: Fraction) -> int:
    """Bits of the odd numerator of a dyadic ``x`` (0 for x = 0)."""
    x = Fraction(x)
    if x == 0:
        return 0
    if not is_dyadic(x):
        raise RoundCertificateError(f"{x} is not dyadic")
    p = abs(x.numerator)
    while p % 2 == 0:
        p //= 2
    return p.bit_length()


# ---------------------------------------------------------------------------
# Record I/O.
# ---------------------------------------------------------------------------

def read_json_maybe_gz(path: Path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_grams_path(record_path: Path, record: dict) -> Path:
    name = record.get("grams_file")
    if not name:
        raise RoundCertificateError("record has no 'grams_file'")
    directory = Path(record_path).parent
    for candidate in (directory / name, directory / (name + ".gz"),
                      directory / name.replace(".gz", "")):
        if candidate.exists():
            return candidate
    raise RoundCertificateError(f"grams bundle {name!r} not found next to {record_path}")


def round_record_v(record: dict, sig: int, unit: str = "decimal"
                   ) -> tuple[dict, list[dict]]:
    """Return (new_record, per_term_report). V's terms are rewritten with
    ``coefficient`` as an exact 'p/q' string of a dyadic rational; every
    other field is passed through unchanged (deep-copied via json round trip
    to avoid accidental aliasing)."""
    new_record = json.loads(json.dumps(record))
    terms = new_record["V"]["terms"]
    reports = []
    for i, term in enumerate(terms):
        original = parse_coefficient(term["coefficient"])
        rounded = round_coefficient(original, sig, unit)
        term["coefficient"] = exact_string(rounded)
        reports.append({
            "term": i,
            "original": str(original),
            "rounded": term["coefficient"],
            "abs_error": str(abs(original - rounded)),
        })
    return new_record, reports


def v_signature(record: dict) -> tuple:
    """Canonical exact identity of a record's V (degree, weights, and per term
    exact coefficient, exponents and term types)."""
    V = record["V"]
    return (int(V["degree_m"]), tuple(int(w) for w in V["weights_r"]),
            tuple((parse_coefficient(t["coefficient"]),
                   tuple(Fraction(e) for e in t["exponents"]),
                   tuple(t["term_types"])) for t in V["terms"]))


def strip_margin_declarations(record: dict) -> tuple[dict, dict]:
    new_record = json.loads(json.dumps(record))
    removed = {k: new_record.pop(k) for k in MARGIN_DECLARATION_FIELDS
               if k in new_record}
    return new_record, removed


def printed_record(record: dict, n: int, unit: str, source: dict
                   ) -> tuple[dict, list[dict]]:
    """A tier-A candidate: V rounded, margin declarations moved to provenance,
    and a ``provenance.printed_coefficients`` entry naming the source."""
    rounded, report = round_record_v(record, n, unit)
    rounded, removed = strip_margin_declarations(rounded)
    prov = rounded.get("provenance")
    if not isinstance(prov, dict):
        prov = {"note": "unverified provenance, not part of the proved "
                        "statement; the portable verifier ignores every key "
                        "in this object"}
    prov["printed_coefficients"] = {
        "tool": "scripts/round_certificate.py",
        "tier": "A (same Gram sidecar, byte for byte)",
        "unit": unit,
        "n": n,
        "source_record": source.get("record"),
        "source_record_sha256": source.get("record_sha256"),
        "source_grams_sha256": source.get("grams_sha256"),
        "source_margin_declarations": removed,
        "note": ("V's coefficients were rounded to exact dyadic rationals; "
                 "every other proved field is the source record's. Fields "
                 "outside 'provenance' that describe the source proof "
                 "(witness, source_artifacts, env) are the source's and "
                 "name the Gram's origin, not this V's margin; this V's "
                 "margin is the one VC derives (and, in a final record, "
                 "declares)."),
    }
    rounded["provenance"] = prov
    return rounded, report


def declare_margin(record: dict, summary: dict) -> dict:
    """Write VC's own derived margin into a record (final printed record)."""
    out = json.loads(json.dumps(record))
    exact = Fraction(summary["eps_certified_exact"])
    out["eps_certified"] = float(exact)
    out["eps_certified_exact"] = exact_string(exact)
    out["eps_certified_exact_is_float_transcription"] = False
    out["eps_certified_source"] = "portable_verifier"
    out["note_half_margin_holds"] = bool(summary["note_half_margin_holds"]["overall"])
    out["worst_charge_over_eps"] = summary["worst_charge_over_eps"]
    out["verifier"] = {
        "acceptance_rule": summary.get("acceptance_rule"),
        "eps_certified_exact": exact_string(exact),
        "worst_charge_over_eps": summary["worst_charge_over_eps"],
    }
    return out


def write_json_file(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driving VC as a subprocess.
# ---------------------------------------------------------------------------

def run_vc(python_exe: str, vc_path: Path, record_path: Path, grams_path: Path,
           json_out: Path, dense: bool, timeout: float | None = 600,
           time_v: Path | None = None, log_path: Path | None = None
           ) -> tuple[bool, dict | None, str]:
    cmd = [python_exe, str(vc_path), str(record_path), "--grams", str(grams_path),
           "--json", str(json_out)]
    if dense:
        cmd.append("--dense")
    if time_v is not None and Path("/usr/bin/time").exists():
        cmd = ["/usr/bin/time", "-v", "-o", str(time_v)] + cmd
    if json_out.exists():
        json_out.unlink()
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.monotonic() - t0
    ok = proc.returncode == 0
    summary = None
    if json_out.exists():
        try:
            summary = json.loads(json_out.read_text())
        except json.JSONDecodeError:
            summary = None
    if ok and (summary is None or summary.get("verdict") != "VERIFIED"):
        ok = False
    log = proc.stdout + proc.stderr
    if log_path is not None:
        Path(log_path).write_text(f"$ {' '.join(cmd)}\n{log}\nEXIT {proc.returncode}\n"
                                  f"WALL_S {wall:.3f}\n")
    if summary is not None:
        summary = dict(summary)
        summary["_wall_s"] = wall
    return ok, summary, log


def find_smallest_verified_sig(record_path: Path, out_dir: Path, python_exe: str,
                               vc_path: Path, sig_min: int, sig_max: int,
                               dense: bool) -> dict:
    """The 2026-09-24 demo scan (decimal digits, one record). Kept as is."""
    record = read_json_maybe_gz(record_path)
    for field in ("V", "grams_file"):
        if field not in record:
            raise RoundCertificateError(f"record has no {field!r}")
    grams_path = resolve_grams_path(record_path, record)
    system = record.get("system", record_path.stem)

    out_dir.mkdir(parents=True, exist_ok=True)
    scan = []
    winner = None
    for sig in range(sig_min, sig_max + 1):
        rounded_record, term_report = round_record_v(record, sig)
        stem = f"{system}_sig{sig:02d}"
        record_out = out_dir / f"{stem}.record.json"
        grams_out = out_dir / f"{stem}.grams{''.join(grams_path.suffixes)}"
        if not grams_out.exists():
            shutil.copyfile(grams_path, grams_out)
        rounded_record["grams_file"] = grams_out.name
        record_out.write_text(json.dumps(rounded_record, indent=1))

        verify_json = out_dir / f"{stem}.verification.json"
        ok, summary, log = run_vc(python_exe, vc_path, record_out, grams_out,
                                  verify_json, dense)
        entry = {
            "sig": sig,
            "verdict": "VERIFIED" if ok else "FAILED",
            "worst_charge_over_eps": summary.get("worst_charge_over_eps") if summary else None,
            "eps_certified_float": summary.get("eps_certified_float") if summary else None,
            "record": str(record_out),
            "verification": str(verify_json),
        }
        scan.append(entry)
        if ok:
            winner = entry
            winner["term_report"] = term_report
            break
    if winner is None:
        raise RoundCertificateError(
            f"no N in [{sig_min}, {sig_max}] verified for {system}; "
            f"widen --sig-max or this record has no tier-A rounding in range")

    for entry in scan:
        if entry["sig"] < winner["sig"]:
            for p in (Path(entry["record"]), Path(entry["verification"])):
                if p.exists():
                    p.unlink()
            grams_for_entry = out_dir / f"{system}_sig{entry['sig']:02d}.grams{''.join(grams_path.suffixes)}"
            if grams_for_entry.exists():
                grams_for_entry.unlink()

    return {
        "tool": "scripts/round_certificate.py",
        "record": str(record_path),
        "grams": str(grams_path),
        "system": system,
        "smallest_verified_sig": winner["sig"],
        "winner": winner,
        "scan": scan,
    }


# ---------------------------------------------------------------------------
# The exact residual screen (a lower bound on VC's charge; never a verdict).
# ---------------------------------------------------------------------------

def _load_vc_module(vc_path: Path):
    spec = importlib.util.spec_from_file_location("_x9_vc_screen", str(vc_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _residual_only_validate_octant(vc, sink: list):
    """A copy of VC's ``validate_octant`` up to and including the dense
    residual ``||E||_1`` (verbatim arithmetic), without the PSD ladder. It
    always "accepts" and returns ``||E||_1 / eps`` so VC's plumbing runs to
    the end; the ratio is recorded in ``sink``."""
    def validate_octant(target, d, gram_rows, basis, eps, blocked=True,
                        cap=None, label=""):
        what = label or "octant"
        nv = vc.validate_target(target, d, what)
        basis = vc.validate_basis(basis, d, nv, what)
        n_basis = len(basis)
        if len(gram_rows) != n_basis or any(len(r) != n_basis for r in gram_rows):
            raise vc.VerifyError(f"{what}: gram shape mismatch")
        Q = [[(vc.q_from(gram_rows[i][j]) + vc.q_from(gram_rows[j][i])) / 2
              for j in range(n_basis)] for i in range(n_basis)]
        E = dict(target)
        for i in range(n_basis):
            for j in range(n_basis):
                q = Q[i][j]
                if q == 0:
                    continue
                mono = tuple(basis[i][k] + basis[j][k] for k in range(nv))
                E[mono] = E.get(mono, Fraction(0)) - q
        resid_l1 = sum(abs(v) for v in E.values())
        ratio = resid_l1 / eps
        sink.append((label, ratio))
        return True, ratio, "", (n_basis, 1, n_basis)
    return validate_octant


def predict_residual(record_path: Path, grams_path: Path, vc_path: Path) -> dict:
    """Exact per-form maxima of ``||E||_1 / eps`` for one record, computed by
    VC's own code with ``validate_octant`` replaced by the residual-only copy.

    A value >= 1 PROVES that VC rejects the record (charge >= ||E||_1). A
    value < 1 is necessary, not sufficient, for VERIFIED."""
    vc = _load_vc_module(Path(vc_path))
    sink: list = []
    vc.validate_octant = _residual_only_validate_octant(vc, sink)
    argv_saved = sys.argv
    sys.argv = ["verify_certificate.py", str(record_path), "--grams", str(grams_path)]
    buf = io.StringIO()
    t0 = time.monotonic()
    try:
        with contextlib.redirect_stdout(buf):
            vc.main()
    except vc.VerifyError as exc:
        return {"screen_error": str(exc), "time_s": time.monotonic() - t0}
    finally:
        sys.argv = argv_saved
    worst = {"V": Fraction(0), "W": Fraction(0)}
    for label, ratio in sink:
        key = "V" if label.startswith("V") else "W"
        worst[key] = max(worst[key], ratio)
    overall = max(worst.values())
    return {"V": worst["V"], "W": worst["W"], "overall": overall,
            "n_octants": len(sink), "time_s": time.monotonic() - t0,
            "proves_fail": overall >= 1}


def predict_residual_all_n(record_path: Path, grams_path: Path, vc_path: Path,
                           ns: list[int], unit: str) -> dict:
    """The same exact screen as :func:`predict_residual`, for MANY n at once.

    Every octant target is linear in V's coefficients (the association is
    linear, and W = -dV/dt is linear in V), so per octant

        E(c) = sum_i c_i * A_i + M - z^T Q z,

    with A_i the octant association of term i's contribution (of V itself,
    or of -dV/dt along the dynamics VC rebuilt for that corner and endpoint),
    M the -eps margin on the pure powers, and z^T Q z computed ONCE. VC's own
    code drives everything (record parsing, dynamics, corners, coverage);
    only ``verify_captures`` is replaced by a recorder that evaluates ``E``
    for every requested n, and ``exact_neg_lie_derivative`` is wrapped to
    learn which dynamics each W form was built from. The Gram rationalization
    and the residual are VC's formulas. Returns, per n, the exact maxima of
    ``||E||_1 / eps`` over V and W octants."""
    vc = _load_vc_module(Path(vc_path))
    record = read_json_maybe_gz(Path(record_path))
    coeffs0 = [parse_coefficient(t["coefficient"]) for t in record["V"]["terms"]]
    rounded = {n: [round_coefficient(c, n, unit) for c in coeffs0] for n in ns}
    worst = {n: {"V": Fraction(0), "W": Fraction(0)} for n in ns}
    state = {"dyn": None, "n_octants": 0, "base_cache": {}}
    orig_lie = vc.exact_neg_lie_derivative

    def lie(v_terms, dynamics_terms):
        state["dyn"] = dynamics_terms
        return orig_lie(v_terms, dynamics_terms)

    def assoc(terms, mu, signs):
        out = {}
        for c, exps, types in terms:
            key = tuple(int(mu[i] * exps[i]) for i in range(len(mu)))
            sign = 1
            for i, ty in enumerate(types):
                if ty == "signed":
                    sign *= signs[i]
            out[key] = out.get(key, Fraction(0)) + sign * c
        return out

    def verify_captures(terms, captures, n_vars, label, expected_eps, blocked=True):
        kind = "V" if label.startswith("V") else "W"
        v_terms = state["v_terms"]
        if kind == "V":
            contrib = [[(Fraction(1), e, t)] for (_, e, t) in v_terms]
        else:
            contrib = [orig_lie([(Fraction(1), e, t)], state["dyn"]) for (_, e, t) in v_terms]
        for cap in captures:
            mu = [int(m) for m in cap["mu"]]
            signs = [int(s) for s in cap["signs"]]
            eps = vc.q_from(cap["eps"])
            basis = [tuple(int(b) for b in beta) for beta in cap["basis"]]
            g = cap["gram"]
            nb = len(basis)
            # a SHARED nominal W witness is the same capture object at every
            # corner: -z^T Q z - margin is the same, compute it once
            ck = (id(g), tuple(basis), eps)
            if ck in state["base_cache"]:
                base, base_l1_cached = state["base_cache"][ck]
            else:
                base = {}
                for i in range(nb):
                    for j in range(nb):
                        q = (vc.q_from(g[i][j]) + vc.q_from(g[j][i])) / 2
                        if q == 0:
                            continue
                        mono = tuple(basis[i][k] + basis[j][k] for k in range(n_vars))
                        base[mono] = base.get(mono, Fraction(0)) - q
                d = sum(basis[0]) * 2 if basis else 0
                for i in range(n_vars):
                    pure = tuple(d if t == i else 0 for t in range(n_vars))
                    base[pure] = base.get(pure, Fraction(0)) - eps
                base_l1_cached = sum(abs(v) for v in base.values())
                if kind == "W":
                    state["base_cache"][ck] = (base, base_l1_cached)
            parts = [assoc(ct, mu, signs) for ct in contrib]
            # ||E||_1 = ||base||_1 + sum over the keys the terms touch of
            # (|base_k + sum_i c_i part_i[k]| - |base_k|): exact, and it avoids
            # copying the (large) z^T Q z dictionary once per n
            touched = set()
            for part in parts:
                touched.update(part)
            base_l1 = base_l1_cached
            base_touched = {k: base.get(k, Fraction(0)) for k in touched}
            for n in ns:
                acc = dict(base_touched)
                for c, part in zip(rounded[n], parts):
                    for k, v in part.items():
                        acc[k] += c * v
                l1 = base_l1 + sum(abs(acc[k]) - abs(base_touched[k]) for k in touched)
                r = l1 / eps
                if r > worst[n][kind]:
                    worst[n][kind] = r
            state["n_octants"] += 1
        return Fraction(0), "", {"n_octants": len(captures), "n_note_half_margin_ok": 0}

    orig_validate_terms = vc.validate_gf_terms

    def validate_gf_terms(raw_terms, n_vars, what):
        out = orig_validate_terms(raw_terms, n_vars, what)
        if what == "V":
            state["v_terms"] = out
        return out

    vc.exact_neg_lie_derivative = lie
    vc.verify_captures = verify_captures
    vc.validate_gf_terms = validate_gf_terms
    argv_saved = sys.argv
    sys.argv = ["verify_certificate.py", str(record_path), "--grams", str(grams_path)]
    buf = io.StringIO()
    t0 = time.monotonic()
    try:
        with contextlib.redirect_stdout(buf):
            vc.main()
    except vc.VerifyError as exc:
        return {"screen_error": str(exc), "time_s": time.monotonic() - t0}
    finally:
        sys.argv = argv_saved
    if "v_terms" not in state:
        raise RoundCertificateError("screen could not capture V's parsed terms")
    if [c for c, _, _ in state["v_terms"]] != coeffs0:
        raise RoundCertificateError("screen: VC's parsed V differs from the record's V")
    out = {}
    for n in ns:
        overall = max(worst[n].values())
        out[n] = {"V": worst[n]["V"], "W": worst[n]["W"], "overall": overall,
                  "proves_fail": overall >= 1}
    return {"per_n": out, "n_octants": state["n_octants"],
            "time_s": time.monotonic() - t0}


# ---------------------------------------------------------------------------
# Tier A for one FUNCTION (all records that carry the same V).
# ---------------------------------------------------------------------------

def _record_stem(path: Path) -> str:
    name = Path(path).name
    for suffix in (".record.json.gz", ".record.json", ".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def load_function(record_paths: list[Path], work_dir: Path) -> list[dict]:
    """Read every record, check that they carry the SAME V exactly, and copy
    each Gram sidecar byte for byte into ``work_dir``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    sig0 = None
    for path in record_paths:
        path = Path(path)
        record = read_json_maybe_gz(path)
        for field in ("V", "grams_file"):
            if field not in record:
                raise RoundCertificateError(f"{path}: record has no {field!r}")
        sig = v_signature(record)
        if sig0 is None:
            sig0 = sig
        elif sig != sig0:
            raise RoundCertificateError(
                f"{path}: V differs from the first record's V; one function "
                f"means one V (exact comparison)")
        grams = resolve_grams_path(path, record)
        grams_copy = work_dir / grams.name
        if not grams_copy.exists() or sha256_file(grams_copy) != sha256_file(grams):
            shutil.copyfile(grams, grams_copy)
        loaded.append({
            "path": path, "record": record, "stem": _record_stem(path),
            "grams_src": grams, "grams": grams_copy,
            "source": {"record": str(path), "record_sha256": sha256_file(path),
                       "grams": str(grams), "grams_sha256": sha256_file(grams)},
        })
    return loaded


def _candidate_paths(work_dir: Path, item: dict, unit: str, n: int, final=False):
    tag = f"{'b' if unit == 'bits' else 'd'}{n:02d}"
    stem = f"{item['stem']}.printed_{tag}" + ("" if final else ".candidate")
    if not final:  # scan candidates and their logs live in a subdirectory
        work_dir = work_dir / "scan"
        work_dir.mkdir(parents=True, exist_ok=True)
    return (work_dir / f"{stem}.record.json",
            work_dir / f"{stem}.verification.json",
            work_dir / f"{stem}.verification.log",
            work_dir / f"{stem}.time_v.txt")


def write_candidates(items: list[dict], work_dir: Path, n: int, unit: str
                     ) -> tuple[list[Path], list[dict]]:
    paths, report = [], None
    for item in items:
        rec, rep = printed_record(item["record"], n, unit, item["source"])
        rec["grams_file"] = item["grams"].name
        rec_path = _candidate_paths(work_dir, item, unit, n)[0]
        write_json_file(rec_path, rec)
        paths.append(rec_path)
        report = rep
    return paths, report


def tier_a_function(label: str, record_paths: list[Path], out_dir: Path,
                    python_exe: str, vc_path: Path, n_min: int, n_max: int,
                    unit: str, strategy: str, dense: bool, finalize: bool,
                    vc_timeout: float | None, screen_all: bool = False,
                    single_p: bool = False, batch_screen: bool = False,
                    log=print) -> dict:
    """Smallest N at which EVERY record of the function VERIFIES with its
    unchanged Gram sidecar; optionally the final printed records."""
    work_dir = Path(out_dir)
    items = load_function(record_paths, work_dir)
    result = {"tool": "scripts/round_certificate.py", "mode": "tier-a",
              "label": label, "unit": unit, "strategy": strategy,
              "n_range": [n_min, n_max], "vc": str(vc_path), "python": python_exe,
              "records": [i["source"] for i in items],
              "terms": len(items[0]["record"]["V"]["terms"]),
              "screen": [], "vc_runs": [], "smallest_verified_n": None}

    batch = {}
    if batch_screen:
        # all n at once, per record (exactly equal to the per-n screen)
        ns = list(range(n_min, n_max + 1))
        for item in items:
            b = predict_residual_all_n(item["path"], item["grams"], vc_path, ns, unit)
            if "screen_error" in b:
                raise RoundCertificateError(f"{label}: batch screen error: "
                                            f"{b['screen_error']}")
            batch[item["stem"]] = b
            result.setdefault("batch_screen", []).append(
                {"record": item["stem"], "time_s": b["time_s"],
                 "n_octants": b["n_octants"]})
            log(f"[X9A {label}] batch screen {item['stem']}: n in [{n_min}, {n_max}] "
                f"in {b['time_s']:.1f} s over {b['n_octants']} octants")

    def screen(n: int) -> dict:
        paths, _ = write_candidates(items, work_dir, n, unit)
        per = []
        for item, p in zip(items, paths):
            if batch:
                r = dict(batch[item["stem"]]["per_n"][n])
                r.pop("proves_fail", None)
            else:
                r = predict_residual(p, item["grams"], vc_path)
            per.append({"record": item["stem"],
                        **{k: (exact_string(v) if isinstance(v, Fraction) else v)
                           for k, v in r.items()},
                        "overall_float": float(r["overall"]) if "overall" in r else None})
        worst = max((Fraction(x["overall"]) for x in per if "overall" in x),
                    default=None)
        entry = {"n": n, "per_record": per,
                 "worst_residual_over_eps": exact_string(worst) if worst is not None else None,
                 "worst_residual_over_eps_float": float(worst) if worst is not None else None,
                 "proves_fail": worst is not None and worst >= 1}
        result["screen"].append(entry)
        if worst is None:
            log(f"[X9A {label}] screen n={n}: screen error "
                f"{[x.get('screen_error') for x in per]}")
        else:
            log(f"[X9A {label}] screen n={n}: worst ||E||_1/eps = "
                f"{float(worst):.4e}"
                f"{'  (proves VC FAILS)' if entry['proves_fail'] else ''}")
        return entry

    def vc_all(n: int, final: bool = False) -> dict:
        if not final:
            paths, report = write_candidates(items, work_dir, n, unit)
        else:
            paths, report = result["_final_paths"], None
        runs = []
        all_ok = True
        for item, rec_path in zip(items, paths):
            _, js, lg, tv = _candidate_paths(work_dir, item, unit, n, final=final)
            ok, summary, _ = run_vc(python_exe, vc_path, rec_path, item["grams"], js,
                                    dense, timeout=vc_timeout, time_v=tv, log_path=lg)
            runs.append({"record": str(rec_path), "verdict": "VERIFIED" if ok else "FAILED",
                         "verification": str(js) if ok else None, "log": str(lg),
                         "wall_s": (summary or {}).get("_wall_s"),
                         "eps_certified_exact": (summary or {}).get("eps_certified_exact"),
                         "worst_charge_over_eps": (summary or {}).get("worst_charge_over_eps"),
                         "_summary": summary})
            all_ok = all_ok and ok
            log(f"[X9A {label}] VC n={n}{' FINAL' if final else ''} "
                f"{item['stem']}: {'VERIFIED' if ok else 'FAILED'}"
                + (f" (worst charge/eps {summary['worst_charge_over_eps']['overall']:.4e}, "
                   f"{summary['_wall_s']:.1f} s)" if ok and summary else ""))
            if not ok and not final:
                break  # the function fails at this n; no need to run the rest
        entry = {"n": n, "final": final, "all_verified": all_ok,
                 "runs": [{k: v for k, v in r.items() if k != "_summary"} for r in runs]}
        result["vc_runs"].append(entry)
        return {"ok": all_ok, "runs": runs, "report": report, "paths": paths}

    def cleanup(keep_n: int) -> None:
        """Remove candidate records of every other n (their verdicts and
        screen values stay in the summary)."""
        ns = {e["n"] for e in result["vc_runs"] if not e["final"]} | \
            {e["n"] for e in result["screen"]}
        for other in ns - {keep_n}:
            for item in items:
                p = _candidate_paths(work_dir, item, unit, other)[0]
                if p.exists():
                    p.unlink()

    winner = None
    if strategy == "scan":
        for n in range(n_min, n_max + 1):
            if screen_all:
                screen(n)
            got = vc_all(n)
            if got["ok"]:
                winner = (n, got)
                break
    elif strategy == "predict":
        start = None
        for n in range(n_min, n_max + 1):
            entry = screen(n)
            if not entry["proves_fail"] and entry["worst_residual_over_eps"] is not None:
                start = n
                break
        if start is None:
            raise RoundCertificateError(
                f"{label}: the residual screen proves VC fails for every n in "
                f"[{n_min}, {n_max}]")
        result["screen_start_n"] = start
        if single_p:
            # ONE VC run in all (records whose VC run takes hours): the final
            # record is written at the screen's n with the SCREEN's margin
            # declared, and VC runs once on it. VC accepts an exact declared
            # margin only if it is <= the margin VC derives; the screen margin
            # is >= VC's (charge >= ||E||_1), so a VERIFIED run proves the
            # two are equal. A failed run is reported; no further n is tried.
            n = start
            entry = next(e for e in result["screen"] if e["n"] == n)
            final_paths = []
            for item, per in zip(items, entry["per_record"]):
                cand = json.loads(_candidate_paths(work_dir, item, unit, n)[0].read_text())
                had_decl = any(k in item["record"] for k in MARGIN_DECLARATION_FIELDS)
                if had_decl:
                    eps = Fraction(item["record"]["epsilon_exact"])
                    worst = {k: Fraction(per[k]) for k in ("V", "W", "overall")}
                    summary = {
                        "eps_certified_exact": exact_string(eps * (1 - worst["overall"])),
                        "note_half_margin_holds": {"overall": worst["overall"] <= Fraction(1, 2)},
                        "worst_charge_over_eps": {k: float(v) for k, v in worst.items()},
                        "acceptance_rule": "charge < eps (strict, exact rational); "
                                           "WP P2-14 / decision D-l",
                    }
                    cand = declare_margin(cand, summary)
                cand["provenance"]["printed_coefficients"]["declared_margin_source"] = (
                    "exact residual screen (eps * (1 - max ||E||_1/eps)); VC "
                    "accepts it only if it equals the margin VC derives"
                    if had_decl else None)
                fpath = _candidate_paths(work_dir, item, unit, n, final=True)[0]
                write_json_file(fpath, cand)
                final_paths.append(fpath)
            result["_final_paths"] = final_paths
            fin = vc_all(n, final=True)
            result.pop("_final_paths")
            result["final"] = {
                "all_verified": fin["ok"], "single_p": True,
                "records": [{"record": r["record"], "verdict": r["verdict"],
                             "verification": r["verification"], "wall_s": r["wall_s"],
                             "eps_certified_exact": r["eps_certified_exact"],
                             "worst_charge_over_eps": r["worst_charge_over_eps"]}
                            for r in fin["runs"]],
            }
            result["acceptance"] = acceptance_check(items, final_paths, n, unit)
            if fin["ok"]:
                result["smallest_verified_n"] = n
                result["below_winner_evidence"] = (
                    f"n in [{n_min}, {n - 1}]: residual screen ||E||_1/eps >= 1 "
                    f"proves VC fails")
            cleanup(n)
            write_json_file(work_dir / f"{label}.tierA_summary.json", result)
            if not fin["ok"]:
                raise RoundCertificateError(
                    f"{label}: single VC run at n = {n} FAILED; not retried")
            return result
        for n in range(start, n_max + 1):
            got = vc_all(n)
            if got["ok"]:
                winner = (n, got)
                break
    else:
        raise RoundCertificateError(f"unknown strategy {strategy!r}")

    if winner is None:
        write_json_file(work_dir / f"{label}.tierA_summary.json", result)
        raise RoundCertificateError(
            f"no N in [{n_min}, {n_max}] verified for {label}")
    n, got = winner
    result["smallest_verified_n"] = n
    result["term_report"] = got["report"]
    # below-the-winner rigour: every n < winner is FAILED by VC (scan) or
    # proved to fail by the residual screen (predict)
    if strategy == "predict":
        result["below_winner_evidence"] = (
            f"n in [{n_min}, {result['screen_start_n'] - 1}]: residual screen "
            f"||E||_1/eps >= 1 proves VC fails; n in [{result['screen_start_n']}, "
            f"{n - 1}]: VC FAILED")
    else:
        result["below_winner_evidence"] = f"n in [{n_min}, {n - 1}]: VC FAILED"

    if finalize:
        final_paths = []
        for item, run in zip(items, got["runs"]):
            cand = json.loads(Path(run["record"]).read_text())
            had_decl = any(k in item["record"] for k in MARGIN_DECLARATION_FIELDS)
            final = declare_margin(cand, run["_summary"]) if had_decl else cand
            final["provenance"]["printed_coefficients"]["declared_margin_source"] = (
                "VC --json summary of the candidate run (the verifier's own "
                "derivation); re-checked by the independent final VC run"
                if had_decl else None)
            fpath = _candidate_paths(work_dir, item, unit, n, final=True)[0]
            write_json_file(fpath, final)
            final_paths.append(fpath)
        result["_final_paths"] = final_paths
        fin = vc_all(n, final=True)
        result.pop("_final_paths")
        result["final"] = {
            "all_verified": fin["ok"],
            "records": [{"record": r["record"], "verdict": r["verdict"],
                         "verification": r["verification"], "wall_s": r["wall_s"],
                         "eps_certified_exact": r["eps_certified_exact"],
                         "worst_charge_over_eps": r["worst_charge_over_eps"]}
                        for r in fin["runs"]],
        }
        # acceptance: printed strings equal record strings; nothing else moved
        result["acceptance"] = acceptance_check(items, final_paths, n, unit)
    cleanup(n)
    write_json_file(work_dir / f"{label}.tierA_summary.json", result)
    return result


PROVED_FIELDS = ("system", "rational_gains", "delta0_exact", "epsilon_exact",
                 "brackets", "corners", "nominal", "channel", "weights_r", "kappa")


def acceptance_check(items: list[dict], final_paths: list[Path], n: int,
                     unit: str) -> dict:
    """Every printed coefficient string-equal to the record's; gains,
    brackets, delta0, epsilon, corners unchanged; exponents and term types
    unchanged; Gram sidecar byte-identical to the source's."""
    out = {"ok": True, "records": []}
    printed = None
    for item, fpath in zip(items, final_paths):
        final = json.loads(Path(fpath).read_text())
        src = item["record"]
        checks = {}
        for f in PROVED_FIELDS:
            if f in src or f in final:
                checks[f] = src.get(f) == final.get(f)
        v_ok = (src["V"]["degree_m"] == final["V"]["degree_m"]
                and src["V"]["weights_r"] == final["V"]["weights_r"]
                and len(src["V"]["terms"]) == len(final["V"]["terms"]))
        strings = []
        for a, b in zip(src["V"]["terms"], final["V"]["terms"]):
            v_ok = v_ok and a["exponents"] == b["exponents"] \
                and a["term_types"] == b["term_types"]
            expect = exact_string(round_coefficient(parse_coefficient(a["coefficient"]),
                                                    n, unit))
            v_ok = v_ok and b["coefficient"] == expect
            strings.append(b["coefficient"])
        if printed is None:
            printed = strings
        same_v = strings == printed
        grams_same = sha256_file(item["grams"]) == item["source"]["grams_sha256"]
        rec_ok = all(checks.values()) and v_ok and same_v and grams_same
        out["records"].append({"record": str(fpath), "fields_unchanged": checks,
                               "v_structure_and_strings_ok": v_ok,
                               "same_printed_v_as_first_record": same_v,
                               "grams_byte_identical": grams_same, "ok": rec_ok})
        out["ok"] = out["ok"] and rec_ok
    out["printed_coefficients"] = printed
    out["max_significant_bits"] = max((significant_bits(Fraction(s)) for s in printed),
                                      default=0)
    return out


# ---------------------------------------------------------------------------
# Tier-B helpers.
# ---------------------------------------------------------------------------

def round_row(row: dict, n: int, unit: str) -> tuple[dict, list[str]]:
    """A row JSON with ``V_terms`` coefficients rounded; each written as a
    float64 that IS the dyadic value (asserted), plus the exact strings."""
    out = json.loads(json.dumps(row))
    strings = []
    for t in out["V_terms"]:
        r = round_coefficient(parse_coefficient(t["coefficient"]), n, unit)
        f = float(r)
        if Fraction(f) != r:
            raise RoundCertificateError(
                f"rounded coefficient {r} is not an exact float64; the fixed-V "
                f"drivers read coefficients with float(), so this n is out of "
                f"the dyadic-without-TC range")
        t["coefficient"] = f
        strings.append(exact_string(r))
    out["printed_coefficients"] = {"tool": "scripts/round_certificate.py round-row",
                                   "unit": unit, "n": n, "exact": strings}
    return out, strings


def canonicalize_record(record: dict, strings: list[str]) -> dict:
    """Rewrite V's coefficients (hex floats from EM) as the exact strings,
    asserting exact equality term by term. Nothing else changes."""
    out = json.loads(json.dumps(record))
    terms = out["V"]["terms"]
    if len(terms) != len(strings):
        raise RoundCertificateError(f"term count {len(terms)} != {len(strings)}")
    for t, s in zip(terms, strings):
        if parse_coefficient(t["coefficient"]) != Fraction(s):
            raise RoundCertificateError(
                f"record coefficient {t['coefficient']} != printed {s}")
        t["coefficient"] = s
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _default_vc() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "verification" / "standalone" / "verify_certificate.py"


def legacy_main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sig-min", type=int, default=1)
    ap.add_argument("--sig-max", type=int, default=20)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to run verify_certificate.py with "
                         "(default: this script's own interpreter)")
    ap.add_argument("--vc", default=None,
                    help="path to verify_certificate.py "
                         "(default: verification/standalone/verify_certificate.py "
                         "next to this repo)")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    vc_path = Path(args.vc) if args.vc else _default_vc()
    if not vc_path.is_file():
        print(f"verify_certificate.py not found at {vc_path}", file=sys.stderr)
        return 2

    try:
        result = find_smallest_verified_sig(
            Path(args.record), Path(args.out_dir), args.python, vc_path,
            args.sig_min, args.sig_max, args.dense)
    except RoundCertificateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"tool": "scripts/round_certificate.py", "record": args.record,
                 "refused": True, "reason": str(exc)}, indent=1) + "\n")
        return 1

    print(f"system = {result['system']}")
    print("N (sig) -> verdict:")
    for entry in result["scan"]:
        print(f"  {entry['sig']:2d} -> {entry['verdict']}"
              + (f" (worst charge/eps overall {entry['worst_charge_over_eps']['overall']:.3e})"
                 if entry["worst_charge_over_eps"] else ""))
    print(f"smallest verified N (significant decimal digits) = {result['smallest_verified_sig']}")
    print(f"winner record: {result['winner']['record']}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
        print(f"summary written to {args.json}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        return legacy_main(argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("tier-a", help="tier A for one function")
    a.add_argument("--label", required=True)
    a.add_argument("--record", action="append", required=True,
                   help="a record of the function (repeatable; all must carry "
                        "the same V exactly)")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--unit", choices=UNITS, default="bits")
    a.add_argument("--n-min", type=int, default=1)
    a.add_argument("--n-max", type=int, default=53)
    a.add_argument("--strategy", choices=("scan", "predict"), default="scan")
    a.add_argument("--screen-all", action="store_true",
                   help="with --strategy scan: also run the residual screen at "
                        "every n (a cross-check of the screen against VC)")
    a.add_argument("--finalize", action="store_true",
                   help="write the final printed records (VC's margin declared "
                        "when the source declared one) and run VC on them again")
    a.add_argument("--single-p", action="store_true",
                   help="with --strategy predict: write the final record at the "
                        "screen's n with the screen margin declared and run VC "
                        "ONCE (for records whose VC run takes hours)")
    a.add_argument("--batch-screen", action="store_true",
                   help="compute the residual screen for every n of the range in "
                        "one pass per record (linearity of the octant targets in "
                        "V's coefficients); identical numbers, one Gram pass")
    a.add_argument("--python", default=sys.executable)
    a.add_argument("--vc", default=None)
    a.add_argument("--dense", action="store_true")
    a.add_argument("--vc-timeout", type=float, default=None,
                   help="seconds per VC run (default: none)")

    p = sub.add_parser("predict", help="exact residual screen of one rounding")
    p.add_argument("--record", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--unit", choices=UNITS, default="bits")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vc", default=None)

    r = sub.add_parser("round-row", help="tier B: write a rounded row JSON")
    r.add_argument("--row", required=True)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--unit", choices=UNITS, default="bits")
    r.add_argument("--out", required=True)

    c = sub.add_parser("canonicalize",
                       help="tier B: rewrite an emitted record's V as exact strings")
    c.add_argument("--record", required=True)
    c.add_argument("--row", required=True, help="the rounded row (round-row output)")
    c.add_argument("--out", required=True)

    for sp in (a, p, r, c):
        sp.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    vc_path = Path(getattr(args, "vc", None) or _default_vc())

    try:
        if args.cmd == "tier-a":
            res = tier_a_function(args.label, [Path(x) for x in args.record],
                                  Path(args.out_dir), args.python, vc_path,
                                  args.n_min, args.n_max, args.unit, args.strategy,
                                  args.dense, args.finalize, args.vc_timeout,
                                  screen_all=args.screen_all,
                                  single_p=args.single_p,
                                  batch_screen=args.batch_screen,
                                  log=lambda s: print(s, flush=True))
            print(f"[X9A {args.label}] smallest verified n ({args.unit}) = "
                  f"{res['smallest_verified_n']}; final "
                  f"{(res.get('final') or {}).get('all_verified')}; acceptance "
                  f"{(res.get('acceptance') or {}).get('ok')}", flush=True)
            if args.json:
                write_json_file(Path(args.json), res)
            ok = res.get("final", {"all_verified": True})["all_verified"] and \
                res.get("acceptance", {"ok": True})["ok"]
            return 0 if ok else 3
        if args.cmd == "predict":
            items = load_function([Path(args.record)], Path(args.out_dir))
            paths, _ = write_candidates(items, Path(args.out_dir), args.n, args.unit)
            res = predict_residual(paths[0], items[0]["grams"], vc_path)
            res = {k: (exact_string(v) if isinstance(v, Fraction) else v)
                   for k, v in res.items()}
            print(json.dumps(res, indent=1))
            if args.json:
                write_json_file(Path(args.json), res)
            return 0
        if args.cmd == "round-row":
            row = read_json_maybe_gz(Path(args.row))
            out, strings = round_row(row, args.n, args.unit)
            write_json_file(Path(args.out), out)
            print(json.dumps(strings))
            return 0
        if args.cmd == "canonicalize":
            rec = read_json_maybe_gz(Path(args.record))
            row = read_json_maybe_gz(Path(args.row))
            strings = row["printed_coefficients"]["exact"]
            out = canonicalize_record(rec, strings)
            prov = out.setdefault("provenance", {})
            prov["printed_coefficients"] = {
                "tool": "scripts/round_certificate.py canonicalize",
                "tier": "B (fixed-V re-solve witness)",
                "unit": row["printed_coefficients"]["unit"],
                "n": row["printed_coefficients"]["n"],
                "emitted_record": str(args.record),
                "emitted_record_sha256": sha256_file(Path(args.record)),
                "rounded_row": str(args.row),
                "note": ("V's coefficients rewritten from EM's hex floats to the "
                         "exact printed strings; the values are identical "
                         "(asserted exactly), nothing else changed."),
            }
            write_json_file(Path(args.out), out)
            return 0
    except RoundCertificateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        if getattr(args, "json", None):
            write_json_file(Path(args.json), {"refused": True, "reason": str(exc)})
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
