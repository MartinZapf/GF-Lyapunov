#!/usr/bin/env python3
"""Tamper battery: adversarial negative controls for the standalone verifier.

WP X6 (``docs/papers/big_draft/NEXT_STEPS.md``, "Tamper battery" paragraph).
Ported from ``review/audit_2026-09-23/negative_controls.py``, generalised
from that script's fixed six-record ``BASES`` table and hard-coded scratch
path to arbitrary ``--base`` records, and made shape-aware so one mutation
library covers nominal and perturbed, bracketed and unbracketed records
without per-base-name special-casing.

For each ``--base NAME=RECORD[:GRAMS]``, this script:

1. loads RECORD and its grams bundle (GRAMS, or the record's own
   ``grams_file`` resolved next to RECORD if GRAMS is omitted);
2. runs the untampered pair through ``--verifier`` once, and requires
   VERIFIED (a base that does not verify on its own makes every tamper
   result on it meaningless);
3. for every mutation whose ``requires(rec, grams)`` predicate is true of
   that base, writes a tampered copy under ``--out-dir``, runs the verifier
   on it, and compares the verdict against the mutation's expectation:

   - ``REJECT``: the tamper must be caught (a nonzero returncode: 1 =
     REJECTED with "VERIFICATION FAILED" on stderr, or 2 = INVALID INPUT
     with "INVALID INPUT" on stderr, for a tamper that breaks the record
     schema; the verifier's outcome classes, 2026-11-05);
   - ``ACCEPT-legit``: the tamper leaves a mathematically valid certificate
     (e.g. a Gram entry nudged by far less than the record's own epsilon),
     so the verifier's exact re-derivation is SUPPOSED to accept it, at a
     silently changed margin;
   - ``ACCEPT-info``: a label-only change the verifier is not designed to
     see (it does not pin the published system label or inject a bogus
     extra field) -- a documented verifier limit, not a soundness gap.
     Reported as "accepted (documented limit)", never as a failure.
   - ``ACCEPT-same-margin``: a metadata-only change that the verifier's
     rules say must still verify, and that cannot touch the proof: the
     power-target boundary cases whose new target lies strictly inside
     ``(lo^k, hi^k)``. Expected: ACCEPT, and the re-derived exact margin
     must EQUAL the untampered base's re-derived exact margin (the corners,
     gains and Grams are unchanged, so any change of margin is a mismatch).
     This is a real oracle, not a rubber stamp.
   - ``ACCEPT-or-safe-REJECT``: a mutation that genuinely changes the
     dynamics being certified (``gain-tiny``'s free-gain shift by eps/1e9)
     rather than corrupting the proof of a FIXED claim. There is NO oracle
     for whether the mutated gains ought to verify: the verifier's
     re-derivation from the mutated inputs may honestly land on either
     side. Each side is checked on its own terms (review 2026-09-25, m1):

     * ACCEPT is checked by re-establishing, independently of the
       verifier's own post-checks and from the exact values it prints and
       the mutated record we wrote, the two invariants any acceptance must
       satisfy: (a) the re-derived margin ``eps_certified = eps - charge``
       (parsed as an exact fraction) satisfies ``0 < derived <= epsilon``
       of the record, and (b) every margin the MUTATED record declares
       (``eps_certified_exact``, ``eps_certified``) is NOT above the
       re-derived one, with the verifier's own rounding allowance only for
       a float image (``eps_certified``, or an exact value flagged
       ``eps_certified_exact_is_float_transcription``): equal float64
       image. The mutation keeps the base's declared margin, so (b) is the
       substantive check: a shift that LOWERS the true margin below the
       declared one must not be accepted. (a) alone could never fail, since
       the verifier prints the line only after accepting every octant at
       charge < eps; it is kept as a parse guard. Neither check can fail
       unless the verifier's acceptance logic is wrong, and that is exactly
       what a tamper battery is for; a failing ACCEPT is therefore flagged
       as a SOUNDNESS ALARM, not an ordinary mismatch. Bases that declare no
       margin at all (the 2026-07-12 exactgains records, example1) get (a)
       only, and the result says so ("no declared margin").
     * REJECT is safe only for the two reasons a pure free-gain shift can
       legitimately produce: the re-derived margin fell below the declared
       one (``MARGIN CLAIM EXCEEDS PROOF: stated A, derived B``, parsed, and
       required to satisfy ``A > B > 0``), or an octant's SOS residual
       charge now exceeds its budget (``FAILED (budget_exceeded``). Guard
       messages ("is outside the case these equations describe",
       "contradicts the published system") are deliberately NOT safe: a
       free-gain shift touches neither a structural parameter (those are
       excluded by ``_FIXED_STRUCTURAL_GAIN_NAMES``) nor the system label,
       family, channel or weights, so such a message would mean a
       battery or verifier defect and is reported as a mismatch to
       investigate. See ``_SAFE_GAIN_TINY_REJECT_MARKERS``.

Power-target boundary cases (review 2026-09-25, m1 coverage note). For
every bracket of a base, six cases move ``power_target`` to ``lo^k``,
``hi^k`` (on the boundary), ``lo^k + d``, ``hi^k - d`` (just inside) and
``lo^k - d``, ``hi^k + d`` (just outside), with
``d = min(1/10^12, (hi^k - lo^k)/1000)`` so "just inside" stays inside even
for the ~1e-15-wide ratio-1 brackets. The expectation of each is DERIVED in
Q from the verifier's strict rule ``0 < lo < hi and lo^k < q < hi^k``
(``_bracket_rule_holds``): violated -> REJECT, holds -> ACCEPT-same-margin.

Exit code: 0 iff every REJECT-expected tamper was rejected, every
ACCEPT-expected tamper (legit, same-margin or documented-limit) was
accepted and passed its checks, every ACCEPT-or-safe-REJECT tamper landed on
one of its two honest outcomes, and every untampered base verified.
Otherwise 1, with the mismatches -- and, most importantly, any
REJECT-expected tamper that was instead ACCEPTed, or any
ACCEPT-or-safe-REJECT acceptance that breaks the acceptance invariants,
which would be a soundness issue in the verifier, not a battery bug --
printed and flagged as SOUNDNESS ALARM.

Standard library only (subprocess-invokes the verifier; does not import it
or the framework).
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tamper_lib import (  # noqa: E402
    add_to_gram,
    flip_gram,
    fr,
    largest_offdiag,
    load_json,
    qs,
    reason,
    run_record,
    run_tampered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERIFIER = REPO_ROOT / "verification" / "standalone" / "verify_certificate.py"

REJECT = "REJECT"
ACCEPT_LEGIT = "ACCEPT-legit"
ACCEPT_INFO = "ACCEPT-info"
ACCEPT_OR_SAFE_REJECT = "ACCEPT-or-safe-REJECT"
ACCEPT_SAME_MARGIN = "ACCEPT-same-margin"

#: The ONLY reasons a REJECT of a pure free-gain shift (``gain-tiny``) counts
#: as safe (review 2026-09-25, m1): the re-derived margin fell below the one
#: the record declares, or an octant's SOS charge exceeded its budget. Guard
#: messages ("is outside the case these equations describe", "contradicts
#: the published system") were removed from this list on purpose: a
#: free-gain shift should never trigger them, so if one fires it is a
#: mismatch to investigate, not a pass. Matched against the one
#: "VERIFICATION FAILED" line only (``reason``), never anywhere in the output.
_SAFE_GAIN_TINY_REJECT_MARKERS = (
    "MARGIN CLAIM EXCEEDS PROOF",
    "FAILED (budget_exceeded",
)

_FRAC = r"(-?\d+(?:/\d+)?)"
_DERIVED_RE = re.compile(r"eps_certified = eps - charge = (-?\d+)/(\d+)")
_OVERCLAIM_RE = re.compile(
    rf"MARGIN CLAIM EXCEEDS PROOF: stated {_FRAC}, derived {_FRAC}")


def parse_derived_margin(out: str) -> Optional[Fraction]:
    """The exact ``eps_certified = eps - charge = P/Q`` fraction the verifier
    prints on ACCEPT, or ``None`` if the line is absent (the caller never
    treats "could not parse" as a pass)."""
    m = _DERIVED_RE.search(out)
    if not m:
        return None
    return Fraction(int(m.group(1)), int(m.group(2)))


def _margin_is_positive(out: str) -> Optional[bool]:
    """Whether the parsed re-derived margin is positive (``None`` if absent).
    Kept as a parse guard only: see the module docstring, it cannot fail on
    a genuine acceptance."""
    d = parse_derived_margin(out)
    return None if d is None else d > 0


def _declared_margins(rec) -> List[Tuple[str, Fraction, bool]]:
    """Every margin the record declares, as ``(field, value, float_image)``.
    ``float_image`` marks a value the verifier lets round up to the same
    float64 image of the derived margin: ``eps_certified`` always, and
    ``eps_certified_exact`` only when flagged as a float transcription."""
    out = []
    transcription = rec.get("eps_certified_exact_is_float_transcription") is True
    if rec.get("eps_certified_exact") is not None:
        out.append(("eps_certified_exact", fr(rec["eps_certified_exact"]),
                    transcription))
    if rec.get("eps_certified") is not None:
        out.append(("eps_certified", fr(rec["eps_certified"]), True))
    return out


def check_accept_invariants(rec, out: str) -> Tuple[bool, str]:
    """The ACCEPT side of ``ACCEPT-or-safe-REJECT`` (module docstring):
    ``0 < derived <= epsilon_exact`` and no declared margin of the (mutated)
    record ``rec`` above ``derived``. Returns ``(ok, note)``."""
    derived = parse_derived_margin(out)
    if derived is None:
        return False, "ACCEPT without a parseable eps_certified line"
    eps = fr(rec["epsilon_exact"])
    if not (0 < derived <= eps):
        return False, f"derived margin {derived} not in (0, epsilon = {eps}]"
    declared = _declared_margins(rec)
    if not declared:
        return True, (f"derived {qs(derived)} in (0, epsilon]; no declared "
                      "margin (no oracle beyond positivity)")
    for field, value, float_image in declared:
        if value > derived and not (float_image and float(value) == float(derived)):
            return False, (f"declared {field} = {qs(value)} exceeds the "
                           f"re-derived margin {qs(derived)}")
    return True, (f"derived {qs(derived)} in (0, epsilon]; declared "
                  f"{', '.join(f for f, _, _ in declared)} <= derived")


def check_safe_reject(out: str) -> Tuple[bool, str]:
    """The REJECT side of ``ACCEPT-or-safe-REJECT``: the verdict line must
    name one of ``_SAFE_GAIN_TINY_REJECT_MARKERS``; a margin-overclaim
    refusal must also be internally consistent (``stated > derived > 0``,
    parsed exactly). Returns ``(ok, note)``."""
    line = reason(out)
    if not line.startswith("VERIFICATION FAILED"):
        return False, f"no VERIFICATION FAILED line: {line!r}"
    if _SAFE_GAIN_TINY_REJECT_MARKERS[0] in line:
        m = _OVERCLAIM_RE.search(line)
        if not m:
            return False, "MARGIN CLAIM EXCEEDS PROOF line not parseable"
        stated, derived = Fraction(m.group(1)), Fraction(m.group(2))
        if not (stated > derived > 0):
            return False, (f"inconsistent overclaim refusal: stated {stated}, "
                           f"derived {derived}")
        return True, "safe: declared margin above the re-derived one"
    if _SAFE_GAIN_TINY_REJECT_MARKERS[1] in line:
        return True, "safe: an octant's SOS charge exceeds its budget"
    return False, "REJECT for a reason a free-gain shift should not produce"


def _is_safe_gain_tiny_reject(out: str) -> bool:
    return check_safe_reject(out)[0]

# A record/grams pair used to key off ``--base`` and feed the mutation
# library. Grams is the sidecar bundle (V_grams + corners), kept separate
# from the record because that mirrors both the on-disk shape and the
# verifier's own ``--grams`` split.
Base = Tuple[str, Dict[str, Any], Dict[str, Any], Path, Path]  # name, rec, grams, rec_path, grams_path


# ---------------------------------------------------------------------------
# Shape predicates. A base's applicable case set is derived from what its
# own record/grams pair actually contains, not from the base's name -- so
# the same mutation library works unmodified on nominal and perturbed,
# bracketed and unbracketed records.
# ---------------------------------------------------------------------------

def _has_v_octant_pair(rec, g) -> bool:
    vg = g.get("V_grams") or []
    return bool(vg) and len(vg[0].get("gram", [])) >= 2


def _has_multi_v_octants(rec, g) -> bool:
    return len(g.get("V_grams") or []) >= 2


def _has_endpoints(rec, g) -> bool:
    corners = g.get("corners") or []
    return bool(corners) and "endpoints" in corners[0]


def _has_w_grams(rec, g) -> bool:
    corners = g.get("corners") or []
    return bool(corners) and "W_grams" in corners[0]


def _has_multi_corners(rec, g) -> bool:
    return len(g.get("corners") or []) >= 2 and len(rec.get("corners") or []) >= 2


def _has_brackets(rec, g) -> bool:
    return bool(rec.get("brackets"))


#: rational_gains field names that are FIXED structural parameters of a
#: system's own equations, not free design gains -- perturbing one trips a
#: hard dynamics guard unrelated to margin slack (pidsmc's equations
#: describe delta = 1 only, guard R6, and 'delta' sorts first
#: alphabetically ahead of every free pidsmc gain). m_gain_wrong and
#: m_gain_tiny must never pick one of these as "the" gain to perturb; add a
#: name here, with the reason, if another system's registry turns up one.
_FIXED_STRUCTURAL_GAIN_NAMES = {"delta"}  # pidsmc: fixed at 1, guard R6


def _free_gain_names(rec) -> List[str]:
    """``rec``'s rational-gain field names, sorted, excluding fixed
    structural parameters. May be empty; ``_has_gains`` is the predicate
    that checks for at least one before a mutation is offered."""
    return sorted(n for n in (rec.get("rational_gains") or {})
                  if n not in _FIXED_STRUCTURAL_GAIN_NAMES)


def _has_gains(rec, g) -> bool:
    return bool(_free_gain_names(rec))


def _has_embedded_v(rec, g) -> bool:
    return isinstance(rec.get("V"), dict) and len(rec["V"].get("terms") or []) >= 1


def _always(rec, g) -> bool:
    return True


def _has_field(field):
    return lambda rec, g: field in rec


# System-label aliases that share IDENTICAL hand-written dynamics in the
# verifier's SYSTEMS registry (same equations, different published-gain
# tuning convention) -- relabelling to one of these is the documented
# ACCEPT-info case, not a soundness gap. See verify_certificate.py SYSTEMS:
# "levant3d_2003print" is the 2003-print gain convention of "levant3d", not
# a different system.
_SYSTEM_LABEL_ALIASES = {
    "levant3d_2003print": "levant3d",
    "levant4d_2003print": "levant4d",
}


# ---------------------------------------------------------------------------
# Mutations. Each entry mutates (rec, g) IN PLACE (both must already be a
# fresh deep copy of the base -- the caller reloads from disk per case, it
# never reuses a mutated dict) and returns a one-line description.
# ---------------------------------------------------------------------------

def m_gram_flip_V(rec, g):
    cap = g["V_grams"][0]
    i, j, v = largest_offdiag(cap)
    flip_gram(cap, i, j)
    return f"V octant {cap['signs']}: sign of Gram[{i}][{j}] (|q|={v:.3g}) and its mirror flipped"


def m_gram_tiny_V(rec, g):
    eps = fr(rec["epsilon_exact"])
    delta = eps / 10**9
    add_to_gram(g["V_grams"][0], 0, 0, delta)
    return f"V octant 0: Gram[0][0] += eps/1e9 ({float(delta):.3g}, exact rational, well inside slack)"


def m_gram_above_V(rec, g):
    eps = fr(rec["epsilon_exact"])
    delta = eps * 2
    add_to_gram(g["V_grams"][0], 0, 0, delta)
    return f"V octant 0: Gram[0][0] += 2*epsilon ({float(delta):.3g}, exceeds any slack)"


def m_gram_flip_W(rec, g):
    corner = g["corners"][-1]
    cap = corner["endpoints"][1][-1] if "endpoints" in corner else corner["W_grams"][-1]
    i, j, v = largest_offdiag(cap)
    flip_gram(cap, i, j)
    return f"W capture (last corner, octant {cap['signs']}): sign of Gram[{i}][{j}] (|q|={v:.3g}) flipped"


def m_gram_flip_Wnom(rec, g):
    cap = g["corners"][0]["W_grams"][0]
    i, j, v = largest_offdiag(cap)
    flip_gram(cap, i, j)
    return f"nominal W octant {cap['signs']}: sign of Gram[{i}][{j}] (|q|={v:.3g}) flipped"


def m_drop_V_octant(rec, g):
    s = g["V_grams"][-1]["signs"]
    g["V_grams"] = g["V_grams"][:-1]
    return f"V octant {s} capture deleted"


def m_dup_V_label(rec, g):
    old = g["V_grams"][-1]["signs"]
    g["V_grams"][-1]["signs"] = list(g["V_grams"][0]["signs"])
    return f"V octant label {old} overwritten with {g['V_grams'][0]['signs']} (duplicate label)"


def m_swap_V_grams(rec, g):
    a, b = g["V_grams"][0], g["V_grams"][1]
    a["gram"], b["gram"] = b["gram"], a["gram"]
    return f"Gram matrices of V octants {a['signs']} and {b['signs']} swapped (labels kept)"


def m_cap_eps_V(rec, g):
    g["V_grams"][0]["eps"] = qs(fr(rec["epsilon_exact"]) * 2)
    return "V octant 0 capture eps doubled (!= record epsilon_exact)"


def m_mu_frac_V(rec, g):
    for c in g["V_grams"]:
        c["mu"] = [2.9 if k == 0 else v for k, v in enumerate(c["mu"])]
    return "V captures mu[0] -> 2.9 (non-integer lift)"


def m_laurent_V(rec, g):
    c = g["V_grams"][0]
    b0 = c["basis"][0]
    half = sum(b0)
    c["basis"][0] = [half + 1, -1] + [0] * (len(b0) - 2)
    return f"V octant 0 basis[0] {b0} -> {c['basis'][0]} (Laurent monomial, same weighted degree)"


def m_declared_blocks(rec, g):
    c = g["V_grams"][0]
    n = len(c["basis"])
    c["blocks"] = [list(range(n))[::2], list(range(n))[1::2][:1]]
    return "V octant 0 declares a wrong parity-block partition"


def m_drop_endpoint(rec, g):
    g["corners"][0]["endpoints"] = g["corners"][0]["endpoints"][:1]
    return "corner 0: W- endpoint capture set deleted (only W+ kept)"


def m_dup_endpoint(rec, g):
    g["corners"][0]["endpoints"][1] = copy.deepcopy(g["corners"][0]["endpoints"][0])
    return "corner 0: W- captures replaced by a copy of W+ captures"


def m_swap_endpoints(rec, g):
    e = g["corners"][0]["endpoints"]
    e[0], e[1] = e[1], e[0]
    return "corner 0: W+ and W- capture sets swapped"


def m_drop_W_octant(rec, g):
    eps_caps = g["corners"][0]["endpoints"][1]
    s = eps_caps[-1]["signs"]
    g["corners"][0]["endpoints"][1] = eps_caps[:-1]
    return f"corner 0 W- octant {s} capture deleted"


def m_drop_corner(rec, g):
    rec["corners"] = rec["corners"][:-1]
    g["corners"] = g["corners"][:-1]
    return "last bracket corner deleted from record and grams"


def m_dup_corner(rec, g):
    rec["corners"][-1] = copy.deepcopy(rec["corners"][0])
    g["corners"][-1] = copy.deepcopy(g["corners"][0])
    return "last bracket corner replaced by a copy of corner 0 (record and grams)"


def m_power_target(rec, g):
    """Move the bracket's power_target OUTSIDE its own valid range
    ``lo^power_k < power_target < hi^power_k``, computed in Q from THIS
    record's own lo/hi/power_k -- not a fixed absolute shift. A fixed '+1'
    (the original form) can land back INSIDE a wide bracket's valid range
    (e.g. a box-derived bracket several tenths wide: lo^2 = 4.498641,
    hi^2 = 6.754801, target 4.5 -> 5.5 is still inside), which is the same
    class of imprecision already documented for m_shift_bracket /
    m_widen_bracket. hi^power_k + 1 is unconditionally above the upper
    bound for every bracket, independent of its width."""
    name, b = next(iter(rec["brackets"].items()))
    lo, hi, k = fr(b["lo"]), fr(b["hi"]), int(b["power_k"])
    lo_k, hi_k = lo ** k, hi ** k
    old = b["power_target"]
    new = hi_k + 1
    assert not (lo_k < new < hi_k), (
        f"m_power_target must produce a target outside [{lo_k}, {hi_k}], "
        f"got {new}")
    b["power_target"] = qs(new)
    return (f"bracket {name} power_target {old} -> {qs(new)} "
            f"(outside [{qs(lo_k)}, {qs(hi_k)}], unconditionally on this "
            "bracket's own width)")


def _bracket_rule_holds(lo: Fraction, hi: Fraction, k: int, q: Fraction) -> bool:
    """The verifier's bracket rule, strict on both sides, in Q:
    ``0 < lo < hi and lo^k < q < hi^k`` (verify_certificate.py, "[2/5]")."""
    return 0 < lo < hi and lo ** k < q < hi ** k


#: Power-target boundary kinds: name -> target as a function of
#: (lo^k, hi^k, d). The expectation is NOT listed here; it is derived per
#: base from ``_bracket_rule_holds`` at case-build time.
_POWER_TARGET_BOUNDARY = {
    "at-lo": lambda lo_k, hi_k, d: lo_k,
    "at-hi": lambda lo_k, hi_k, d: hi_k,
    "inside-lo": lambda lo_k, hi_k, d: lo_k + d,
    "inside-hi": lambda lo_k, hi_k, d: hi_k - d,
    "outside-lo": lambda lo_k, hi_k, d: lo_k - d,
    "outside-hi": lambda lo_k, hi_k, d: hi_k + d,
}


def _boundary_step(lo_k: Fraction, hi_k: Fraction) -> Fraction:
    """``d = min(1/10^12, (hi^k - lo^k)/1000)``: the reviewer's 1e-12, but
    never so large that "just inside" leaves a ~1e-15-wide ratio-1 bracket."""
    return min(Fraction(1, 10**12), (hi_k - lo_k) / 1000)


def power_target_boundary_target(b: Dict[str, Any], kind: str) -> Fraction:
    lo, hi, k = fr(b["lo"]), fr(b["hi"]), int(b["power_k"])
    lo_k, hi_k = lo ** k, hi ** k
    return _POWER_TARGET_BOUNDARY[kind](lo_k, hi_k, _boundary_step(lo_k, hi_k))


def power_target_boundary_expect(b: Dict[str, Any], kind: str) -> str:
    """The expectation of one boundary case, derived in Q from the
    verifier's strict rule: violated -> REJECT, holds -> ACCEPT-same-margin
    (the change is metadata only; corners, gains and Grams are untouched)."""
    lo, hi, k = fr(b["lo"]), fr(b["hi"]), int(b["power_k"])
    q = power_target_boundary_target(b, kind)
    return ACCEPT_SAME_MARGIN if _bracket_rule_holds(lo, hi, k, q) else REJECT


def make_power_target_boundary(bname: str, kind: str) -> Callable:
    def mutate(rec, g):
        b = rec["brackets"][bname]
        old = b["power_target"]
        new = power_target_boundary_target(b, kind)
        b["power_target"] = qs(new)
        k = int(b["power_k"])
        side = ("inside" if _bracket_rule_holds(fr(b["lo"]), fr(b["hi"]), k, new)
                else "violates")
        return (f"bracket {bname} power_target {old} -> {qs(new)} ({kind}; "
                f"{side} the strict rule lo^{k} < q < hi^{k})")
    mutate.__name__ = f"m_power_target_{kind.replace('-', '_')}"
    return mutate


#: A FIXED absolute shift/widen amount, deliberately NOT scaled to the
#: bracket's own width: several of these brackets pin an interior irrational
#: gain to within ~1e-15 of it (X1's ratio-1 cells), so a width-proportional
#: delta is imperceptible and the tamper would land back in ACCEPT-legit
#: territory by accident. 1/1000 is ~1000x every base's epsilon (~1e-6), the
#: same margin that makes the fixed-delta ``m_gain_wrong`` reliable.
_BRACKET_DELTA = Fraction(1, 1000)


def m_shift_bracket(rec, g):
    name, b = next(iter(rec["brackets"].items()))
    lo, hi = fr(b["lo"]), fr(b["hi"])
    delta = _BRACKET_DELTA
    nlo, nhi = lo + delta, hi + delta
    b["lo"], b["hi"] = qs(nlo), qs(nhi)
    mp = {lo: nlo, hi: nhi}
    for c in rec["corners"] + g["corners"]:
        c["assignment"][name] = qs(mp[fr(c["assignment"][name])])
    return (f"bracket {name} shifted by {float(delta):.3g} to "
            f"[{float(nlo):.6f},{float(nhi):.6f}] (power_target kept, corners "
            "moved to the shifted endpoints: claims dynamics far from what "
            "the captured Grams were built for)")


def m_widen_bracket(rec, g):
    name, b = next(iter(rec["brackets"].items()))
    lo, hi = fr(b["lo"]), fr(b["hi"])
    delta = _BRACKET_DELTA
    nlo, nhi = lo - delta, hi + delta
    b["lo"], b["hi"] = qs(nlo), qs(nhi)
    mp = {lo: nlo, hi: nhi}
    for c in rec["corners"] + g["corners"]:
        c["assignment"][name] = qs(mp[fr(c["assignment"][name])])
    return (f"bracket {name} widened by {float(delta):.3g} on each side to "
            f"[{qs(nlo)}, {qs(nhi)}] (still contains the published gain), "
            "corners moved to the new, far endpoints")


def m_delta0_wrong(rec, g):
    old = fr(rec["delta0_exact"])
    new = old + Fraction(1, 10)
    rec["delta0_exact"] = qs(new)
    return f"record delta0_exact: {old} -> {new}"


def m_gain_wrong(rec, g):
    name = _free_gain_names(rec)[0]
    old = fr(rec["rational_gains"][name])
    new = old + Fraction(1, 1000)
    rec["rational_gains"][name] = qs(new)
    return f"rational gain {name}: {old} -> {new} (shift 1e3x the record's epsilon)"


def m_gain_tiny(rec, g):
    """Shift a genuine free design gain by a tiny amount (eps/1e9). Unlike
    a Gram-entry nudge, this changes the actual dynamics the certificate
    claims to cover, so the honest outcome is ACCEPT_OR_SAFE_REJECT, not a
    guaranteed ACCEPT_LEGIT: the verifier's own re-derivation from the
    mutated gain may find a still-positive margin (ACCEPT) or may refuse
    for a named safe reason (REJECT) -- see _SAFE_GAIN_TINY_REJECT_MARKERS.
    The field is always a free gain (_free_gain_names excludes fixed
    structural parameters like pidsmc's delta = 1), never a case guard."""
    name = _free_gain_names(rec)[0]
    eps = fr(rec["epsilon_exact"])
    old = fr(rec["rational_gains"][name])
    delta = eps / 10**9
    new = old + delta
    rec["rational_gains"][name] = qs(new)
    return f"rational gain {name}: {old} += eps/1e9 (exact rational, changes the dynamics claimed)"


def m_v_negate(rec, g):
    for t in rec["V"]["terms"]:
        t["coefficient"] = qs(-fr(t["coefficient"]))
    return "every V coefficient negated"


def m_margin_overclaim(rec, g):
    rec["eps_certified_exact"] = rec["epsilon_exact"]
    rec["eps_certified"] = float(fr(rec["epsilon_exact"]))
    return "record claims eps_certified = epsilon_exact (more than eps - charge)"


def m_inject_W(rec, g):
    rec["W"] = {"terms": [{"coefficient": 1.0, "exponents": ["6", "0"],
                           "term_types": ["abs", "abs"]}]}
    rec["W_terms"] = rec["W"]["terms"]
    return "bogus 'W'/'W_terms' fields injected into record (V, gains, Grams unchanged)"


def m_system_relabel(rec, g):
    old = rec["system"]
    alias = _SYSTEM_LABEL_ALIASES.get(old)
    if alias is not None:
        new = alias
    else:
        new = "sta" if old != "sta" else "twisting"
    rec["system"] = new
    return f"record system {old!r} -> {new!r}"


def m_schema_bad(rec, g):
    old = rec.get("schema_version", "<absent>")
    rec["schema_version"] = "gf-certificate-record/99"
    return f"record schema_version {old!r} -> 'gf-certificate-record/99'"


def m_flip_nominal(rec, g):
    old = rec.get("nominal", False)
    rec["nominal"] = not old
    if rec["nominal"]:
        rec.pop("channel", None)
    return f"record nominal: {old!r} -> {rec['nominal']!r} (grams shape kept, now contradicts it)"


def m_channel_wrong(rec, g):
    old = rec.get("channel")
    new = (old or 0) + 1
    rec["channel"] = new
    return f"record channel {old!r} -> {new!r}"


def m_kappa_wrong(rec, g):
    old = rec.get("kappa")
    rec["kappa"] = "999"
    return f"record kappa {old!r} -> '999'"


def m_weights_wrong(rec, g):
    old = rec.get("weights_r")
    new = [int(w) + 1 for w in old]
    rec["weights_r"] = new
    return f"record weights_r {old!r} -> {new!r}"


def m_system_family_wrong(rec, g):
    old = rec.get("system_family")
    new = "not_a_real_family"
    rec["system_family"] = new
    return f"record system_family {old!r} -> {new!r}"



# (suffix, requires, mutate, expect)
MUTATIONS: List[Tuple[str, Callable, Callable, str]] = [
    ("gram-flip-V", _has_v_octant_pair, m_gram_flip_V, REJECT),
    ("gram-tiny-V", _has_v_octant_pair, m_gram_tiny_V, ACCEPT_LEGIT),
    ("gram-above-V", _has_v_octant_pair, m_gram_above_V, REJECT),
    ("drop-V-octant", _has_multi_v_octants, m_drop_V_octant, REJECT),
    ("dup-V-label", _has_multi_v_octants, m_dup_V_label, REJECT),
    ("swap-V-grams", _has_multi_v_octants, m_swap_V_grams, REJECT),
    ("cap-eps-V", lambda r, g: bool(g.get("V_grams")), m_cap_eps_V, REJECT),
    ("mu-frac-V", lambda r, g: bool(g.get("V_grams")), m_mu_frac_V, REJECT),
    ("laurent-V", _has_v_octant_pair, m_laurent_V, REJECT),
    ("declared-blocks", _has_v_octant_pair, m_declared_blocks, REJECT),
    ("gram-flip-W", lambda r, g: _has_endpoints(r, g) or _has_w_grams(r, g), m_gram_flip_W, REJECT),
    ("gram-flip-Wnom", _has_w_grams, m_gram_flip_Wnom, REJECT),
    ("drop-endpoint", _has_endpoints, m_drop_endpoint, REJECT),
    ("dup-endpoint", _has_endpoints, m_dup_endpoint, REJECT),
    ("swap-endpoints", _has_endpoints, m_swap_endpoints, REJECT),
    ("drop-W-octant", _has_endpoints, m_drop_W_octant, REJECT),
    ("drop-corner", _has_multi_corners, m_drop_corner, REJECT),
    ("dup-corner", _has_multi_corners, m_dup_corner, REJECT),
    ("power-target", _has_brackets, m_power_target, REJECT),
    ("shift-bracket", _has_brackets, m_shift_bracket, REJECT),
    ("widen-bracket", _has_brackets, m_widen_bracket, REJECT),
    ("delta0-wrong", _always, m_delta0_wrong, REJECT),
    ("gain-wrong", _has_gains, m_gain_wrong, REJECT),
    ("gain-tiny", _has_gains, m_gain_tiny, ACCEPT_OR_SAFE_REJECT),
    ("v-negate", _has_embedded_v, m_v_negate, REJECT),
    ("margin-overclaim", _always, m_margin_overclaim, REJECT),
    ("inject-W", _always, m_inject_W, ACCEPT_INFO),
    ("system-relabel", _always, m_system_relabel,
     None),  # expectation resolved per-base at case-build time (see build_cases)
    ("schema-bad", _always, m_schema_bad, REJECT),
    ("flip-nominal", _always, m_flip_nominal, REJECT),
    ("channel-wrong", _has_field("channel"), m_channel_wrong, REJECT),
    ("kappa-wrong", _has_field("kappa"), m_kappa_wrong, REJECT),
    ("weights-wrong", _has_field("weights_r"), m_weights_wrong, REJECT),
    ("system-family-wrong", _has_field("system_family"), m_system_family_wrong, REJECT),
]


def build_cases(name: str, rec: Dict[str, Any], g: Dict[str, Any]
                 ) -> List[Tuple[str, Callable, str]]:
    """The (case_id, mutate, expect) triples applicable to one base, decided
    from what the base's own record/grams actually contain. Case ids are
    ``{name}-{kind}``; the power-target boundary kinds are generated per
    bracket as ``power-target-{boundary}-{bracket}``."""
    cases = []
    for suffix, requires, mutate, expect in MUTATIONS:
        if not requires(rec, g):
            continue
        if suffix == "system-relabel":
            expect = (ACCEPT_INFO if rec["system"] in _SYSTEM_LABEL_ALIASES
                      else REJECT)
        cases.append((f"{name}-{suffix}", mutate, expect))
    for bname, b in (rec.get("brackets") or {}).items():
        for kind in _POWER_TARGET_BOUNDARY:
            cases.append((f"{name}-power-target-{kind}-{bname}",
                          make_power_target_boundary(bname, kind),
                          power_target_boundary_expect(b, kind)))
    return cases


def filter_cases(name: str, cases, only=None, kinds=None):
    """Apply ``--cases`` (case id or base NAME) and ``--kinds`` (fnmatch
    patterns on the kind, i.e. the case id without ``{name}-``)."""
    if only:
        cases = [c for c in cases if c[0] in only or name in only]
    if kinds:
        cases = [c for c in cases
                 if any(fnmatch.fnmatchcase(c[0][len(name) + 1:], pat)
                        for pat in kinds)]
    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _split_record_grams(rest: str) -> Tuple[str, Optional[str]]:
    """Split "RECORD[:GRAMS]" on the FIRST ':' that is not part of a Windows
    drive letter. A drive-letter ':' follows a single letter that starts a
    path (the whole string, or the part after the separator) and is followed
    by '/' or '\\\\', e.g. "C:\\a\\record.json". Any other ':' is a
    RECORD/GRAMS separator, including one followed by a POSIX absolute path
    ("/x/record.json:/x/grams.json"). Raises if more than one non-drive ':'
    is found -- the caller is told to use forward slashes or drop the extra
    ':'."""
    def is_drive(i: int) -> bool:
        return (i >= 1 and rest[i - 1].isascii() and rest[i - 1].isalpha()
                and (i == 1 or rest[i - 2] == ":")
                and i + 1 < len(rest) and rest[i + 1] in "/\\")

    seps = [i for i, ch in enumerate(rest) if ch == ":" and not is_drive(i)]
    if not seps:
        return rest, None
    if len(seps) > 1:
        raise ValueError(
            f"ambiguous --base value {rest!r}: found {len(seps)} possible "
            "RECORD:GRAMS separators; use forward slashes, or omit GRAMS "
            "and let it resolve from the record's own 'grams_file'")
    i = seps[0]
    return rest[:i], rest[i + 1:]


def parse_base_arg(value: str) -> Tuple[str, str, Optional[str]]:
    if "=" not in value:
        raise ValueError(f"--base {value!r} must be NAME=RECORD[:GRAMS]")
    name, rest = value.split("=", 1)
    if not name:
        raise ValueError(f"--base {value!r}: empty NAME")
    record, grams = _split_record_grams(rest)
    if not record:
        raise ValueError(f"--base {value!r}: empty RECORD")
    return name, record, (grams or None)


def load_base(name: str, record_arg: str, grams_arg: Optional[str]) -> Base:
    rec_path = Path(record_arg)
    if not rec_path.exists():
        raise FileNotFoundError(f"--base {name}: record not found: {rec_path}")
    rec = load_json(rec_path)
    if grams_arg:
        grams_path = Path(grams_arg)
    else:
        gf = rec.get("grams_file")
        if not gf:
            raise ValueError(f"--base {name}: record has no 'grams_file' "
                             "and no GRAMS was given")
        grams_path = rec_path.parent / gf
    if not grams_path.exists():
        raise FileNotFoundError(f"--base {name}: grams not found: {grams_path}")
    g = load_json(grams_path)
    return name, rec, g, rec_path, grams_path


def parse_cases_filter(spec: Optional[str]):
    if not spec:
        return None
    return {tok.strip() for tok in spec.replace(",", " ").split() if tok.strip()}


def classify(expect: str, verdict: str, out: str, mutated_rec,
             base_margin: Optional[Fraction]) -> Tuple[bool, str]:
    """``(as_expected, check-note)`` for one tamper result, per the
    expectation classes of the module docstring."""
    if verdict == "ERROR":
        return False, "no verdict (timeout or OS error)"
    if expect == ACCEPT_OR_SAFE_REJECT:
        if verdict == "ACCEPT":
            return check_accept_invariants(mutated_rec, out)
        return check_safe_reject(out)
    if expect == ACCEPT_SAME_MARGIN:
        if verdict != "ACCEPT":
            return False, "metadata-only change inside the bracket rule rejected"
        derived = parse_derived_margin(out)
        if derived is None or base_margin is None:
            return False, "margin not parseable (tampered or untampered run)"
        if derived != base_margin:
            return False, (f"margin changed: untampered {qs(base_margin)}, "
                           f"tampered {qs(derived)}")
        return True, "ACCEPT at the untampered base's exact margin"
    expect_verdict = "REJECT" if expect == REJECT else "ACCEPT"
    return verdict == expect_verdict, ""


def is_alarm(res: Dict[str, Any]) -> bool:
    """A mismatch that is a soundness issue: a REJECT-expected tamper that
    was accepted, or an ACCEPT of a genuine-dynamics change that breaks the
    acceptance invariants (``check_accept_invariants``)."""
    return (res.get("kind") == "tamper" and not res["as_expected"]
            and res["verdict"] == "ACCEPT"
            and res["expect"] in (REJECT, ACCEPT_OR_SAFE_REJECT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--verifier", default=str(DEFAULT_VERIFIER),
                    help=f"path to verify_certificate.py (default: {DEFAULT_VERIFIER})")
    ap.add_argument("--base", action="append", default=[], metavar="NAME=RECORD[:GRAMS]",
                    help="repeatable; RECORD's own 'grams_file' is used if "
                         "GRAMS is omitted")
    ap.add_argument("--cases", default=None,
                    help="comma/space-separated allow-list of case IDs "
                         "(e.g. STA-gram-flip-V) or base NAMEs (e.g. STA); "
                         "default: run every applicable case on every base")
    ap.add_argument("--kinds", default=None,
                    help="comma/space-separated fnmatch patterns on the case "
                         "KIND (the case id without the 'NAME-' prefix), e.g. "
                         "'gain-wrong,gain-tiny,power-target*'; combined with "
                         "--cases by intersection")
    ap.add_argument("--out-dir", required=True,
                    help="scratch directory for tampered record/grams pairs "
                         "and the results JSON (created if missing)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-invocation subprocess timeout in seconds (default: 120)")
    args = ap.parse_args()

    verifier = Path(args.verifier)
    if not verifier.exists():
        print(f"--verifier not found: {verifier}", file=sys.stderr)
        return 2
    if not args.base:
        print("at least one --base NAME=RECORD[:GRAMS] is required", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = parse_cases_filter(args.cases)
    kinds = parse_cases_filter(args.kinds)

    bases: List[Base] = []
    for spec in args.base:
        try:
            name, record_arg, grams_arg = parse_base_arg(spec)
            bases.append(load_base(name, record_arg, grams_arg))
        except (ValueError, FileNotFoundError) as exc:
            print(f"--base {spec!r}: {exc}", file=sys.stderr)
            return 2

    results = []
    untampered_ok = True

    # Cases are filtered up front, per base, so the untampered check below
    # can tell "this base has nothing selected, skip it" apart from "this
    # base has selected cases, so it must also be confirmed VERIFIED on its
    # own" -- filtering by a specific case ID (not the base NAME) must still
    # pull in that base's untampered check, or a --cases run could silently
    # never exercise guarantee 1 (untampered bases still verify).
    selected = {}
    for name, rec, g, rec_path, grams_path in bases:
        selected[name] = filter_cases(name, build_cases(name, rec, g),
                                      only, kinds)

    # 1. Every untampered base whose filter selected it (by name or by at
    # least one of its cases) must verify on its own.
    base_margin: Dict[str, Optional[Fraction]] = {}
    for name, rec, g, rec_path, grams_path in bases:
        if (only or kinds) and not selected[name] and not (
                only and name in only and not kinds):
            continue
        t0 = time.time()
        try:
            p = run_record(verifier, rec_path, timeout=args.timeout)
            rc, out = p.returncode, p.stdout + p.stderr
        except Exception as exc:  # subprocess.TimeoutExpired or OS error
            rc, out = f"error:{exc}", ""
        wall = time.time() - t0
        verdict = "VERIFIED" if rc == 0 else "NOT-VERIFIED"
        base_margin[name] = parse_derived_margin(out) if rc == 0 else None
        res = {"id": f"{name}-UNTAMPERED", "base": name, "kind": "untampered",
              "expect": "VERIFIED", "exit_code": rc, "verdict": verdict,
              "as_expected": verdict == "VERIFIED", "wall_s": round(wall, 2),
              "reason": reason(out),
              "derived_margin": (qs(base_margin[name])
                                 if base_margin[name] is not None else None)}
        results.append(res)
        print(json.dumps(res), flush=True)
        if not res["as_expected"]:
            untampered_ok = False

    # 2. Tamper cases, built per base from what that base's shape supports.
    for name, rec, g, rec_path, grams_path in bases:
        for case_id, mutate, expect in selected[name]:
            r = json.loads(json.dumps(rec))
            gg = json.loads(json.dumps(g))
            desc = mutate(r, gg)
            case_dir = out_dir / "tamper" / case_id
            t0 = time.time()
            try:
                p = run_tampered(verifier, r, gg, case_dir, timeout=args.timeout)
                rc, out = p.returncode, p.stdout + p.stderr
            except Exception as exc:
                rc, out = f"error:{exc}", ""
            wall = time.time() - t0
            (case_dir / "verifier_output.txt").write_text(out)
            # A timeout or OS error is not a verdict: it never satisfies a
            # REJECT expectation (it is reported as a mismatch, verdict ERROR).
            verdict = ("ERROR" if not isinstance(rc, int)
                       else "ACCEPT" if rc == 0 else "REJECT")
            ok, check = classify(expect, verdict, out, r, base_margin.get(name))
            derived = parse_derived_margin(out) if verdict == "ACCEPT" else None
            res = {"id": case_id, "base": name, "kind": "tamper", "tamper": desc,
                  "expect": expect, "exit_code": rc, "verdict": verdict,
                  "as_expected": ok, "check": check,
                  "derived_margin": qs(derived) if derived is not None else None,
                  "wall_s": round(wall, 2), "reason": reason(out)}
            results.append(res)
            print(json.dumps(res), flush=True)

    # --- summary --------------------------------------------------------
    mismatches = [r for r in results if not r["as_expected"]]
    alarms = [r for r in mismatches if is_alarm(r)]
    documented_limits = [r for r in results
                         if r.get("kind") == "tamper" and r["expect"] == ACCEPT_INFO
                         and r["as_expected"]]

    print()
    print(f"{len(results)} checks: {len(results) - len(mismatches)} as expected, "
         f"{len(mismatches)} mismatched")
    for r in documented_limits:
        print(f"  {r['id']}: accepted (documented limit) -- {r['tamper']}")
    for r in mismatches:
        tag = "SOUNDNESS ALARM" if r in alarms else "MISMATCH"
        print(f"  [{tag}] {r['id']}: expected {r['expect']}, got {r['verdict']} "
             f"-- {r.get('tamper', '(untampered base)')} :: {r['reason']}"
             + (f" :: check: {r['check']}" if r.get("check") else ""))

    (out_dir / "tamper_battery_results.json").write_text(
        json.dumps(results, indent=1))

    if alarms:
        print(f"\n{len(alarms)} SOUNDNESS ALARM(S): a tamper that should have "
             "been rejected was accepted, or an acceptance broke the "
             "acceptance invariants. See above.", file=sys.stderr)
    ok = untampered_ok and not mismatches
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
