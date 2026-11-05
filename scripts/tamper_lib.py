#!/usr/bin/env python3
"""Shared helpers for tampering with gf-certificate-record JSON and running
them through the standalone verifier.

This module exists so ``scripts/tamper_battery.py`` (WP X6, the tamper
battery) and ``tests/test_standalone_verifier.py`` (the ``_run_tampered``
helper around L411) do not each keep their own copy of "write a tampered
record+grams pair to disk, then subprocess the verifier on it" or their own
copy of exact-Fraction Gram arithmetic. Both import this module instead.

Standard library only, like the verifier itself, so it can be imported from
a bare ``python3`` the same way ``verification/standalone/verify_certificate.py``
is invoked -- though unlike that file this module is a repository-internal
convenience, not a shipped artifact.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Exact-rational helpers (record/grams number encodings: JSON float, hex
# float string, or "p/q" / decimal rational string -- see
# ``verify_certificate.q_from`` / ``q_exact``).
# ---------------------------------------------------------------------------

def fr(x: Any) -> Fraction:
    """Exact ``Fraction`` of a record/grams number in any accepted encoding."""
    if isinstance(x, str):
        if x.lower().lstrip("+-").startswith("0x"):
            return Fraction(float.fromhex(x))
        return Fraction(x)
    return Fraction(x)


def qs(q: Fraction) -> str:
    """Render a Fraction as the "p/q" exact-rational string the records use."""
    q = Fraction(q)
    return f"{q.numerator}/{q.denominator}"


# ---------------------------------------------------------------------------
# Gram-matrix mutation primitives (symmetric by construction: every capture
# stores a full, not triangular, matrix).
# ---------------------------------------------------------------------------

def add_to_gram(cap: Dict[str, Any], i: int, j: int, delta: Fraction,
                 symmetric: bool = True) -> None:
    g = cap["gram"]
    g[i][j] = qs(fr(g[i][j]) + delta)
    if symmetric and i != j:
        g[j][i] = qs(fr(g[j][i]) + delta)


def flip_gram(cap: Dict[str, Any], i: int, j: int) -> None:
    g = cap["gram"]
    g[i][j] = qs(-fr(g[i][j]))
    if i != j:
        g[j][i] = qs(-fr(g[j][i]))


def largest_offdiag(cap: Dict[str, Any]) -> Tuple[int, int, float]:
    """The (i, j, |value|) of the largest-magnitude off-diagonal Gram entry."""
    g = cap["gram"]
    n = len(g)
    best = max(((abs(fr(g[i][j])), i, j)
                for i in range(n) for j in range(n) if i != j))
    return best[1], best[2], float(best[0])


# ---------------------------------------------------------------------------
# JSON I/O (records and grams bundles may be gzipped; see
# ``verify_certificate.read_json``'s magic-byte sniff, which this mirrors).
# ---------------------------------------------------------------------------

def load_json(path) -> Any:
    b = Path(path).read_bytes()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return json.loads(b)


# ---------------------------------------------------------------------------
# Writing a tampered (record, grams) pair and running the verifier on it.
# This is the generalisation of the ``_run_tampered`` helper that used to
# live only in tests/test_standalone_verifier.py: same file-writing and
# subprocess shape, parameterised over the verifier path and output
# directory instead of a hard-coded module-level ``VERIFIER`` and pytest's
# ``tmp_path`` fixture.
# ---------------------------------------------------------------------------

def write_tampered_pair(out_dir, rec: Dict[str, Any], bundle: Dict[str, Any],
                         record_name: str = "rec.json",
                         grams_name: str = "grams.json") -> Tuple[Path, Path]:
    """Write ``rec``/``bundle`` (already mutated by the caller) to
    ``out_dir``, pointing ``rec['grams_file']`` at the sibling grams file.

    Mutates ``rec`` in place (sets ``grams_file``), matching the behaviour of
    the original ``_run_tampered`` so existing call sites see no difference.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_p = out_dir / record_name
    gr_p = out_dir / grams_name
    rec["grams_file"] = gr_p.name
    rec_p.write_text(json.dumps(rec))
    gr_p.write_text(json.dumps(bundle))
    return rec_p, gr_p


def run_tampered(verifier, rec: Dict[str, Any], bundle: Dict[str, Any],
                  out_dir, extra_args: Sequence[str] = (),
                  timeout: Optional[float] = None,
                  record_name: str = "rec.json",
                  grams_name: str = "grams.json") -> subprocess.CompletedProcess:
    """Write the tampered pair under ``out_dir`` and run the verifier on it.

    Returns the ``subprocess.CompletedProcess``. The verifier's exit codes
    (its module docstring, "Outcome classes"): 0 = VERIFIED (ACCEPT); 1 =
    REJECTED, a mathematical check failed, ``VERIFICATION FAILED: ...`` on
    stderr; 2 = INVALID INPUT, ``INVALID INPUT: ...`` on stderr; 3 = ERROR,
    no verdict. A tamper is caught by any nonzero code.
    """
    rec_p, _ = write_tampered_pair(out_dir, rec, bundle, record_name, grams_name)
    cmd = [sys.executable, str(verifier), str(rec_p), *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_record(verifier, record_path, extra_args: Sequence[str] = (),
                timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run the verifier directly on an on-disk record (no tampering, no
    rewrite) -- used for the "every untampered base still verifies" check."""
    cmd = [sys.executable, str(verifier), str(record_path), *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def reason(text: str) -> str:
    """Pull the one-line verdict out of a verifier subprocess's combined
    stdout+stderr, for compact reporting."""
    for line in text.splitlines():
        if "VERIFICATION FAILED" in line or line.startswith("INVALID INPUT:"):
            return line.strip()
    lines = [l for l in text.splitlines() if l.strip()]
    tb = [l for l in lines if "Error" in l or "Exception" in l]
    if tb:
        return "UNCAUGHT: " + tb[-1].strip()
    for line in lines:
        if "eps_certified" in line and "worst" in line:
            return line.strip()
    return lines[-1].strip() if lines else ""
