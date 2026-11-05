#!/usr/bin/env python3
"""Portable-record emitter (campaign item C9, block E3, binding rule 8).

Binding rule 8 of ``docs/papers/big_draft/PRE_WRITING_CAMPAIGN.md``: a
certificate is citable in the draft only when it also ships as a portable
stdlib-only record that ``verification/standalone/verify_certificate.py``
re-proves without importing the framework. This script turns a certified run
into exactly that record.

What it does and, more importantly, does not do
-----------------------------------------------
It TRANSCRIBES. It never re-derives, re-solves, or repairs anything:

* V comes from the row JSON of the run that certified it, verbatim;
* the gains come from the printed-gain registry
  (``gf_framework.systems.printed_gains``, campaign item C20), so the record
  binds 27/10 and 11/10 rather than their float64 images, with rational
  brackets and the full 2^j corner box for the genuinely irrational ones;
* the per-octant Gram captures come from the exact-phase artifact of that
  same run.

If any required Gram capture is missing, the emitter REFUSES with a typed
reason and writes a refusal record naming what a re-run has to produce. It
never emits a partial record: a record with a hole in its coverage would look
citable while proving less than it claims, and the completeness rules
(2^n octants for V, 2^n per dynamics for W, 2^k bracket corners, both
inclusion endpoints for a perturbed row) are exactly what makes the affinity
argument close.

Completeness, per the E3 card
-----------------------------
* NOMINAL row: one dynamics, 2^n octant captures for V and 2^n for W, at each
  of the 2^k bracket corners, all sign patterns distinct.
* PERTURBED row: 2^n octant captures for V, and per bracket corner 2 inclusion
  endpoints x 2^n octants, all distinct.

The shared nominal W witness (``--share-w-grams``, WP R-05)
-----------------------------------------------------------
A nominal W witness is one per-octant Gram set from the design solver. The
Gram does not depend on the gains, only the octant TARGET does, so the same
2^n Grams can witness every one of the 2^k bracket corners. When that is the
case the captures are written ONCE, under the bundle's ``shared_W_grams``, and
each corner carries ``"W_grams_shared": true`` instead of its own copy. The
portable verifier reads exactly this shape (``nominal_w_captures``) and still
rebuilds every corner's dynamics from ITS OWN exact gains and proves all 2^n
octants against the shared Gram: storage aliasing, never a dropped obligation.

Sharing is never inferred. It happens only when it is asked for, and only
after the emitter has established that the corners really do name the same
Grams: with a design-bundle source, by recomputing every entry hash over the
Gram bytes and requiring it to equal the hash the exact phase recorded for
that corner; otherwise by hashing the encoded captures themselves. One
differing corner is a refusal (``shared_w_mismatch``), never a silent pick.

Margin policy and provenance (WP R-05, review 2026-09-05 finding 4)
-------------------------------------------------------------------
``margin_policy`` and ``eps_certified`` are never defaulted. The policy comes
from the exact-phase witness document, or from an explicit
``--margin-policy`` override that is then recorded as an override in
``provenance``; without either, the emitter refuses. ``eps_certified`` comes
from the exact phase, or from the portable verifier's own derived output at
self-verification time; there is no third source and no hand-typed number.

Which margin the record STATES (WP R-08, follow-up review 2026-09-05 B)
-----------------------------------------------------------------------
The follow-up review changed one source's reported ``witness.eps_certified``
to 999 and got a record that said 999 and self-verified: the emitter took the
verifier's own margin only when the source field was absent, so a number no
proof stage supported shipped beside a successful re-proof. Now:

* when self-verification runs, ``eps_certified``, ``eps_certified_exact``,
  ``note_half_margin_holds`` and ``worst_charge_over_eps`` ALWAYS come from
  the portable verifier's summary, whatever the source reported. The exact
  rational is the verifier's own string; the float is derived from it;
* the source stage's number is kept, labelled unverified, under
  ``provenance.source_reported_eps_certified``;
* a source that reports MORE than its own proof supports is a broken
  producer, not a rounding question: the emission refuses with
  ``margin_claim_exceeds_proof`` rather than quietly publish the smaller
  proved number under a producer that claims the larger one;
* without self-verification (``--no-self-verify``) the record carries the
  source's number under ``eps_certified_source =
  "exact_phase_unverified_by_portable_stage"``, and, when the source stores
  only a float64, ``eps_certified_exact_is_float_transcription: true``. That
  string is then the exact value of a rounded float, NOT an exact proof
  margin. When the portable proof of such a record finishes later,
  ``--attach-verify-summary`` writes the verifier's exact margin into a
  ``<stem>.verified_margin.json`` sidecar beside the record, leaving the
  verified bytes untouched.

Everything the record says that is NOT part of the proved statement lives
under the record's ``provenance`` object: free-text claims, the exact stage's
own reported ratios, the emitter command line. It is UNVERIFIED PROVENANCE,
the verifier ignores every key in it, and nothing in it may be quoted as a
result. The proved statement is what the portable verifier prints.

Record size (decision of 2026-07-31): everything stays in git, gzipped. The
record and its grams bundle are written as ``.json.gz`` with stdlib gzip (the
verifier reads them with stdlib gzip too, so no dependency is added anywhere),
floats are encoded as exact ``float.hex()`` strings, and every published value
(gains, delta0, margin, bracket endpoints) is an exact rational string.

Typical use
-----------
    # nominal benchmark row from the campaign matrix
    python scripts/emit_certificate_record.py \\
        --row artifacts/campaign_2026-08/E1_matrix/systems/fosmc.json \\
        --exact artifacts/campaign_2026-08/E1_matrix/reverify_exact_printed_*.json \\
        --grams <grams bundle from the same run> \\
        --campaign-block E1_matrix --out-dir records/

    # perturbed flagship row
    python scripts/emit_certificate_record.py \\
        --row artifacts/.../sta_delta1_row.json \\
        --exact artifacts/.../sta_exactgains_*.json \\
        --gain-point flagship --out-dir records/

    # a large nominal row whose exact phase wrote a per-corner CELL and whose
    # Grams live in the design bundle it proved on (levant5d, m = 11)
    python scripts/emit_certificate_record.py \\
        --exact-cell artifacts/.../l5d_m11_eps1e-4 \\
        --design-record artifacts/.../p4_02_l5d_m11_eps1e-4.json \\
        --design-bundle-npz artifacts/.../p4_02_l5d_m11_eps1e-4_grams.npz \\
        --gain-point flagship --share-w-grams --stem-tag m11 \\
        --no-self-verify --out-dir records/

    # a PERTURBED cell at a covariant representative p (WP TA): the levant4d
    # ratio-1 cells, whose gains are the flagship point scaled by p^i and
    # whose disturbance bound is delta0 = p^4
    python scripts/emit_certificate_record.py \\
        --exact-cell artifacts/.../cell_x_p1_2_eps1e-4/exact \\
        --design-record artifacts/.../p4_05_l4d_m9_p1_2_eps1e-4.json \\
        --design-bundle-npz artifacts/.../p4_05_l4d_m9_p1_2_eps1e-4_grams.npz \\
        --gain-point flagship --representative-p 1/2 --stem-tag p1_2_m9 \\
        --out-dir records/

Representative-p cells (WP TA)
------------------------------
A cell produced at a covariant representative ``p`` proves the dynamics at
the gains ``lambda_i * p^i`` and the disturbance bound it states in its header
(``delta0_exact``). ``--representative-p P`` builds exactly that point from
``FLAGSHIP_GAIN_POINTS[system]``, in Q: rational gains times ``p^i``, bracket
endpoints times ``p^i``, and the bracket power targets times ``p^(i k)``, so
``lo^k < target < hi^k`` still holds and is re-checked here. Every corner's
``exact_gains`` is pinned against that scaled point, and the cell header's own
``bracket_checks_in_Q`` and ``qscale_p`` are pinned against it too. The base
point and ``p`` are recorded under ``provenance``; the verifier never sees
``p``, it proves the scaled gains, brackets and delta0 it reads as data.

A perturbed cell names its two inclusion endpoints by design-bundle block
name: ``W_oct<i>_v0`` is the ``+delta0`` vertex and ``W_oct<i>_v1`` the
``-delta0`` vertex (``matched_disturbance_vertices`` returns them in that
order). The emitter maps them to W+ and W- by that suffix, never by list
position, and refuses a list that mixes suffixes.

Exact-record gain points (WP TB emitter change, option a)
--------------------------------------------------------
``--gain-point exact-record`` emits a gain point that is NOT in the
printed-gain registry: a gain box, a scaled continuous-twisting point, a new
tuning. The gains, brackets, box, delta0 and corner box are read from the
``--exact`` record itself, which must be an exact-gains record of
``scripts/certify_gain_point.py`` (``gain_source`` = its command line,
``gain_binding`` = ``printed_rationals``). Nothing is taken on trust: every
value is re-checked in Q against what the portable verifier checks (its gain
table, fixed parameters, positivity, ``lo^k < q < hi^k``, the disturbance
channel), and every VARYING gain must enter the verifier's equations
affinely (guard R6 declaration plus an exact probe of the verifier's own
builder), because the verifier's interval coverage rests on an affinity
argument it does not check in code. Anything else is refused with
``unverifiable_gain_point``. A box edge ``[lo, hi]`` becomes a verifier
bracket with ``power_k = 1`` and ``power_target = (lo + hi)/2``;
``--bracket-target NAME:K:Q`` names the gain inside it instead (checked as
``lo^K < Q < hi^K``); ``--corner-subset NAME=VALUE`` fixes a box gain at one
edge and emits only those corners. The record carries no registry source
artifact; ``provenance.gain_point_source`` names the exact record.

The same affinity check runs for a gain point taken from the printed-gain
registry (``--gain-point flagship|benchmark``, also a representative-p
point; item C2, 2026-09-28): every bracketed gain must be a declared affine
coordinate and pass the exact probe of the verifier's builder, at the gains
the record carries, or the emission refuses with ``unverifiable_gain_point``.
The check writes nothing into the record.

    # the X2 per-tuning record A: lambda1 = 2, lambda2 bracketed by 9/2
    python scripts/emit_certificate_record.py \\
        --row <run>/m8/row.json --exact <run>/m8/exact.json \\
        --gain-point exact-record --corner-subset lambda1=2 \\
        --bracket-target lambda2:2:9/2 --stem-tag tb_m8 --out-dir <run>/m8/portable

Exit codes: 0 emitted (and self-verified), 3 refused with a typed reason,
2 usage error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gf_framework.systems.printed_gains import (  # noqa: E402
    BENCHMARK_GAIN_POINTS,
    FLAGSHIP_GAIN_POINTS,
    brackets as registry_brackets,
    corner_assignments,
    corner_labels,
    exact_dynamics,
    gain_point,
    gain_point_record,
)

STANDALONE = REPO_ROOT / "verification" / "standalone" / "verify_certificate.py"

SCHEMA_VERSION = "gf-certificate-record/2"

#: Typed refusal reasons. Every one of them means the same thing operationally:
#: no record is emitted, and the campaign has an explicit item to re-run.
REFUSAL_REASONS = (
    "not_certified",             # the source row is not a positive
    "unknown_system",            # no hand-written builder in the verifier
    "missing_gram_captures",     # the exact artifact records verdicts, not Grams
    "incomplete_octant_coverage",   # captures present, sign patterns incomplete
    "incomplete_corner_box",     # fewer than 2^k distinct bracket corners
    "missing_endpoint_captures",    # perturbed row without both W+ and W-
    "nominal_perturbed_mismatch",   # endpoint captures on a nominal row, or vice versa
    "epsilon_ladder_mismatch",   # a capture certified at a lower eps rung
    "float_image_gains",         # the exact phase bound float64 images, not printed rationals
    "gain_mismatch",             # row gains disagree with the printed-gain registry
    "self_verification_failed",  # the emitted record did not verify
    "shared_w_mismatch",         # --share-w-grams asked for, corners name different Grams
    "missing_margin_policy",     # no margin policy / eps_certified, and no override
    "source_pin_mismatch",       # a source hash does not reproduce (V input, Gram entry)
    "proof_input_binding_missing",  # historical summary has no proof-time input hashes
    "proof_input_mismatch",      # attached inputs differ from the proof-time bytes
    "margin_claim_exceeds_proof",   # the source reports a margin its own proof does not support
    "margin_declaration_inconsistent",  # exact and floating declarations disagree
    "unverifiable_gain_point",   # a gain point (exact-record or registry) the verifier could not check
)


class Refusal(Exception):
    """A typed refusal: what is missing, and what a re-run must produce."""

    def __init__(self, reason, detail, required_rerun=()):
        if reason not in REFUSAL_REASONS:
            raise ValueError(f"untyped refusal reason {reason!r}")
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.required_rerun = list(required_rerun)

    def as_dict(self):
        return {"refused": True, "reason": self.reason, "detail": self.detail,
                "required_rerun": self.required_rerun}


# ---------------------------------------------------------------------------
# Covariant representatives (WP TA)
# ---------------------------------------------------------------------------

#: Systems with a registered covariant scaling. For Levant's differentiator
#: with n states, the map ``lambda_i -> p^i lambda_i``, ``delta0 -> p^n
#: delta0`` sends the inclusion at one gain point to an equivalent one at
#: every rational ``p > 0`` (the ratio delta0/L is preserved); the E4
#: ratio-1 cells were designed and proved at such representatives. A system
#: not listed here has no scaled gain point, and ``--representative-p`` on it
#: is a refusal, never a guess at the exponents.
COVARIANT_SCALING = {
    "levant4d": {
        "gain_index": {"lambda1": 1, "lambda2": 2, "lambda3": 3, "lambda4": 4},
        "delta0_power": 4,
    },
    # WP X10: the third-order differentiator at recursive (1.1, 1.5, 3, 5);
    # same equations as levant4d, hence the same covariant map.
    "levant4d_2003print": {
        "gain_index": {"lambda1": 1, "lambda2": 2, "lambda3": 3, "lambda4": 4},
        "delta0_power": 4,
    },
}


class ScaledGainPoint:
    """A registered gain point mapped to a covariant representative ``p``.

    Everything is exact: rational gains ``g * p^i``, bracket endpoints
    ``lo * p^i`` and ``hi * p^i``, and bracket power targets
    ``target * p^(i k)``. The scaled bracket's validity
    ``lo'^k < target' < hi'^k`` is re-checked in Q here (it holds for every
    ``p > 0`` because it is the base inequality times ``p^(i k)``; checking it
    costs nothing and a wrong exponent table would fail it). The corner order
    is the registry's, corner for corner.
    """

    def __init__(self, base, p):
        spec = COVARIANT_SCALING.get(base.system)
        if spec is None:
            raise Refusal(
                "unknown_system",
                f"{base.system}: no covariant scaling is registered, so a "
                f"representative-p gain point cannot be built for it",
                ["register the gain exponents of the system's covariant map "
                 "in COVARIANT_SCALING (scripts/emit_certificate_record.py) "
                 "and pin them in tests/test_representative_p_emission.py"])
        p = Fraction(p)
        if p <= 0:
            raise Refusal("gain_mismatch",
                          f"representative p must be positive (got {p})")
        if set(spec["gain_index"]) != set(base.gains):
            raise Refusal(
                "gain_mismatch",
                f"{base.system}: the covariant map scales "
                f"{sorted(spec['gain_index'])}, the registered point has "
                f"{sorted(base.gains)}")
        self.base = base
        self.p = p
        self.system = base.system
        self.channel = base.channel
        self.gain_index = dict(spec["gain_index"])
        self.delta0_power = int(spec["delta0_power"])
        self.scale = {n: p ** i for n, i in self.gain_index.items()}
        self._brackets, self._targets = {}, {}
        for name, (lo, hi) in registry_brackets(base).items():
            gain = base.gains[name]
            k = int(gain.power_k)
            s = self.scale[name]
            lo_s, hi_s = lo * s, hi * s
            target = gain.power_target * p ** (self.gain_index[name] * k)
            if not (0 < lo_s < hi_s and lo_s ** k < target < hi_s ** k):
                raise Refusal(
                    "gain_mismatch",
                    f"{base.system}.{name}: the scaled bracket "
                    f"[{lo_s}, {hi_s}] does not bracket the root of "
                    f"g^{k} = {target} in Q")
            self._brackets[name] = (lo_s, hi_s)
            self._targets[name] = (k, target)
        self._rational = {n: g.value * self.scale[n]
                          for n, g in base.gains.items() if not g.is_bracketed}
        self._assignments = [
            {n: v * self.scale[n] for n, v in a.items()}
            for a in corner_assignments(base)]

    @property
    def bracketed_names(self):
        return tuple(sorted(self._brackets))

    def brackets(self):
        return dict(self._brackets)

    def bracket_power(self, name):
        return self._targets[name]

    def rational_gains(self):
        return dict(self._rational)

    def corner_assignments(self):
        return [dict(a) for a in self._assignments]

    def corner_labels(self, assignment):
        out = {}
        for name, value in assignment.items():
            lo, hi = self._brackets[name]
            if value == lo:
                out[name] = "lo"
            elif value == hi:
                out[name] = "hi"
            else:
                raise Refusal(
                    "gain_mismatch",
                    f"{self.system}.{name}: {value} is not an endpoint of the "
                    f"scaled bracket [{lo}, {hi}]")
        return out

    def exact_gain_values(self, assignment):
        assignment = dict(assignment or {})
        if set(assignment) != set(self._brackets):
            raise Refusal(
                "gain_mismatch",
                f"{self.system}: a corner must assign exactly the bracketed "
                f"gains {sorted(self._brackets)}, got {sorted(assignment)}")
        values = dict(self._rational)
        values.update(assignment)
        return values

    def record(self):
        """The scaled point, in the shape of ``gain_point_record``."""
        base = gain_point_record(self.base)
        return {
            "system": self.system,
            "factory": self.base.factory,
            "module": self.base.module,
            "channel": self.channel,
            "note": (base["note"] + f" Covariant representative p = {self.p}: "
                     "lambda_i -> p^i lambda_i, bracket endpoints -> p^i x, "
                     "power targets -> p^(i k) x, delta0 -> p^"
                     f"{self.delta0_power} delta0."),
            "printed_rational": {n: str(v)
                                 for n, v in sorted(self._rational.items())},
            "roles": base["roles"],
            "provenance": {
                n: f"{text}; scaled by p^{self.gain_index[n]} with p = {self.p}"
                for n, text in base["provenance"].items()},
            "brackets": {
                n: {"lo": str(lo), "hi": str(hi),
                    "power_k": self._targets[n][0],
                    "power_target": str(self._targets[n][1])}
                for n, (lo, hi) in sorted(self._brackets.items())},
            "n_corners": 2 ** len(self._brackets),
            "bracket_digits": base["bracket_digits"],
            "representative_p": str(self.p),
        }

    def provenance(self, delta0):
        """``provenance.representative_p``: p, the map, and the base point."""
        base_delta0 = Fraction(delta0) / self.p ** self.delta0_power
        return {
            "note": ("the record's gains, brackets and delta0 are the scaled "
                     "point below; the verifier proves them as data and never "
                     "reads p"),
            "p": str(self.p),
            "gain_exponents": dict(sorted(self.gain_index.items())),
            "delta0_power": self.delta0_power,
            "map": ("rational gain g_i -> g_i p^i; bracket endpoint x -> x p^i; "
                    "bracket power target q -> q p^(i k); delta0 -> delta0 "
                    f"p^{self.delta0_power}"),
            "base_table": "FLAGSHIP_GAIN_POINTS",
            "base_gain_point": gain_point_record(self.base),
            "delta0_over_p_power": str(base_delta0),
        }


# ---------------------------------------------------------------------------
# Gain point read from an exact-gains record (WP TB emitter change, option a)
# ---------------------------------------------------------------------------

#: The only producer whose exact-gains record may supply the gain point. Its
#: exact phase injected exactly these rationals (``gain_binding`` is
#: ``printed_rationals``), so the record's gains are the proved gains; any
#: other document keeps the registry route.
EXACT_RECORD_GAIN_SOURCE = "scripts/certify_gain_point.py command line"

#: Systems whose ``create_*_spec`` factory has no printed-gain registry entry
#: (the same table as ``scripts/certify_gain_point.py::FACTORY_FALLBACK``).
EXACT_RECORD_FACTORY_FALLBACK = {
    "pidsmc4d": ("create_pidsmc4d_spec", "gf_framework.systems.pidsmc4d"),
}

#: Guard R6 coordinate aliases (``scripts/certify_gain_point.py::R6_ALIASES``):
#: the verifier names pidsmc's gains lambda1..3, the affinity declaration in
#: ``scripts/gain_region_fixedV.py`` names them k1..k3; they coincide at the
#: only delta the verifier's pidsmc equations describe, delta = 1.
EXACT_RECORD_R6_ALIASES = {
    "pidsmc": {"lambda1": "k1", "lambda2": "k2", "lambda3": "k3"},
}


def _exact_q(value, what):
    """A record rational: a ``"p/q"`` string or a JSON integer, never a float.

    A float here would be a float64 image, which is exactly what the
    exact-record route exists not to transcribe.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise Refusal("unverifiable_gain_point",
                      f"{what} = {value!r} is not an exact rational string")
    try:
        return Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise Refusal("unverifiable_gain_point",
                      f"{what} = {value!r} is not a rational: {exc}") from exc


def _exact_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("unverifiable_gain_point",
                      f"{what} = {value!r} is not an integer")
    return value


def parse_corner_subset(items):
    """``--corner-subset NAME=VALUE`` (repeatable) -> {name: Fraction}."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise Refusal("unverifiable_gain_point",
                          f"--corner-subset {item!r} is not NAME=VALUE")
        name, value = (s.strip() for s in item.split("=", 1))
        if name in out:
            raise Refusal("unverifiable_gain_point",
                          f"--corner-subset names {name} twice")
        out[name] = _exact_q(value, f"--corner-subset {name}")
    return out


def parse_bracket_targets(items):
    """``--bracket-target NAME:K:Q`` (repeatable) -> {name: (k, q)}."""
    out = {}
    for item in items or []:
        parts = item.split(":")
        if len(parts) != 3:
            raise Refusal("unverifiable_gain_point",
                          f"--bracket-target {item!r} is not NAME:K:Q")
        name = parts[0].strip()
        if name in out:
            raise Refusal("unverifiable_gain_point",
                          f"--bracket-target names {name} twice")
        try:
            k = int(parts[1])
        except ValueError as exc:
            raise Refusal("unverifiable_gain_point",
                          f"--bracket-target {name}: K = {parts[1]!r} is not "
                          f"an integer") from exc
        out[name] = (k, _exact_q(parts[2], f"--bracket-target {name} Q"))
    return out


# ---------------------------------------------------------------------------
# Affinity of a varying gain: guard R6 plus an exact probe of VC's builder
# ---------------------------------------------------------------------------

def _vc_build_system_dynamics():
    """The portable verifier's dynamics dispatcher (imported read-only)."""
    if str(STANDALONE.parent) not in sys.path:
        sys.path.insert(0, str(STANDALONE.parent))
    from verify_certificate import build_system_dynamics  # noqa: PLC0415
    return build_system_dynamics


def probe_builder_affinity(system, rational, brackets, name, build=None):
    """Exact probe: the verifier's equations are affine in ``name``.

    ``rational`` maps the fixed gains to exact values and ``brackets`` maps
    every varying gain to its exact interval ``(lo, hi)``. The other varying
    gains sit at their lower ends; ``name`` is set to lo, the midpoint and
    hi, and the rebuilt equations must have the same shape and support
    (exponents and term types) at all three, with every midpoint coefficient
    equal to the average of the two endpoint coefficients, all in exact
    arithmetic. ``build`` defaults to the portable verifier's own
    ``build_system_dynamics``; tests pass a synthetic builder. A failure is
    the refusal ``unverifiable_gain_point``.
    """
    if build is None:
        build = _vc_build_system_dynamics()
    lo, hi = (Fraction(v) for v in brackets[name][:2])
    base = {n: Fraction(v) for n, v in rational.items()}
    base.update({n: Fraction(b[0]) for n, b in brackets.items()})

    def terms(value):
        gains = dict(base)
        gains[name] = value
        try:
            dyn = build(system, gains, Fraction(0), None)
        except Exception as exc:  # noqa: BLE001 (any builder error refuses)
            raise Refusal(
                "unverifiable_gain_point",
                f"{system}.{name}: the verifier's builder refused the gains "
                f"{ {n: str(v) for n, v in sorted(gains.items())} } ({exc})"
            ) from exc
        return [[(tuple(e), tuple(t), c) for c, e, t in f] for f in dyn]

    at_lo, at_mid, at_hi = terms(lo), terms((lo + hi) / 2), terms(hi)
    if not (len(at_lo) == len(at_mid) == len(at_hi)):
        raise Refusal("unverifiable_gain_point",
                      f"{system}: the verifier's equations change shape "
                      f"across the interval of {name}")
    for f_lo, f_mid, f_hi in zip(at_lo, at_mid, at_hi):
        if not (len(f_lo) == len(f_mid) == len(f_hi)):
            raise Refusal("unverifiable_gain_point",
                          f"{system}: the verifier's equations "
                          f"change shape across the interval")
        for (e0, t0, c0), (e1, t1, c1), (e2, t2, c2) in zip(f_lo, f_mid,
                                                             f_hi):
            if not (e0 == e1 == e2 and t0 == t1 == t2) \
                    or c1 != (c0 + c2) / 2:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"{system}.{name}: the verifier's equations are "
                    f"not affine in {name} on [{lo}, {hi}]")
    return {"gain": name, "interval": [str(lo), str(hi)],
            "probe": "verifier builder at lo, midpoint and hi, exact"}


def check_varying_gain_affinity(system, rational, brackets, name, build=None):
    """Guard R6 for one varying gain: declaration, then the exact probe.

    The gain must be a declared affine coordinate of
    ``scripts/gain_region_fixedV.py`` (through ``EXACT_RECORD_R6_ALIASES``),
    and ``probe_builder_affinity`` must pass. The verifier's coverage of an
    interval by its end points rests on this affinity, which the verifier
    states but does not check in code.
    """
    import gain_region_fixedV as grf  # noqa: PLC0415 (read-only reuse)

    decl = grf.SYSTEMS.get(system)
    coord = EXACT_RECORD_R6_ALIASES.get(system, {}).get(name, name)
    if decl is None or coord not in decl.coords:
        raise Refusal(
            "unverifiable_gain_point",
            f"{system}.{name} varies over [{brackets[name][0]}, "
            f"{brackets[name][1]}] but is not a declared affine gain "
            f"coordinate (scripts/gain_region_fixedV.py, guard R6); the "
            f"verifier's interval coverage rests on affinity, which it "
            f"does not check itself")
    report = probe_builder_affinity(system, rational, brackets, name, build)
    report["declared_coordinate"] = coord
    return report


def check_registry_point_affinity(point, system, build=None):
    """Guard R6 and the exact probe for a gain point from the registry.

    C2 (2026-09-28). A registry point (``--gain-point flagship|benchmark``,
    also scaled to a representative ``p``) varies exactly its bracketed
    gains, and the verifier covers each bracket by its two ends. Every
    bracketed gain therefore gets the check an exact-record point gets
    (``check_varying_gain_affinity``), against the gains the record will
    carry (the scaled ones for a scaled point), before any record file is
    written. A point without brackets needs nothing. The record's bytes do
    not change: the check refuses or passes, it writes no field. Returns the
    per-gain reports.
    """
    brackets = point_brackets(point)
    if not brackets:
        return []
    rational = point_rational_gains(point)
    return [check_varying_gain_affinity(system, rational, brackets, name,
                                        build)
            for name in sorted(brackets)]


class ExactRecordGainPoint:
    """The gain point of an exact-gains record written by the TB driver.

    ``--gain-point exact-record``. The registry is not consulted for gain
    VALUES at all: the gains, brackets, box, delta0 and corner box are read
    from the exact-gains record (``exact.json`` of
    ``scripts/certify_gain_point.py``), whose exact phase injected exactly
    these rationals, and every one of them is re-checked here, in Q, against
    what the portable verifier can check:

    * the verifier's hand-written gain table is the authority on the names:
      every gain of its equations is present, nothing else is, and a fixed
      factory parameter (pidsmc ``delta``) is at the one value its equations
      describe and never varies;
    * every gain is a positive rational; a bracket has ``0 < lo < hi``, an
      integer ``k >= 1`` and ``lo^k < q < hi^k``; a box edge has
      ``0 < lo < hi``;
    * a gain that VARIES (bracket or box) must enter the dynamics affinely,
      because the verifier's coverage of the whole interval rests on the
      affinity argument, which it states but does not check in code (its
      docstring step 6). Two checks: it is a declared affine coordinate of
      ``scripts/gain_region_fixedV.py`` (guard R6), and the verifier's own
      builder is affine in it on the interval (same support at lo, the
      midpoint and hi, midpoint coefficients the average of the endpoint
      ones, all exact);
    * a positive delta0 needs the verifier's disturbance channel.

    A box edge ``[lo, hi]`` is written as a verifier bracket with
    ``power_k = 1`` and ``power_target = (lo + hi) / 2`` (the verifier accepts
    ``power_k >= 1``), unless ``--bracket-target NAME:K:Q`` names the gain
    the interval is meant to contain, which is then checked as
    ``lo^K < Q < hi^K``. ``--corner-subset NAME=VALUE`` fixes a BOX gain at
    one of its edges: the record then covers only the corners with that value,
    the gain moves to ``rational_gains`` and the bundle's other corners are
    dropped (never their V captures, which are gain-independent).
    """

    def __init__(self, exact_doc, system, cfg, corner_subset=None,
                 bracket_targets=None):
        self.system = system
        self.cfg = cfg
        self.channel = cfg.get("channel")
        source = exact_doc.get("gain_source")
        if source != EXACT_RECORD_GAIN_SOURCE:
            raise Refusal(
                "unverifiable_gain_point",
                f"--gain-point exact-record reads the gain point only from an "
                f"exact-gains record of the TB driver (gain_source = "
                f"{EXACT_RECORD_GAIN_SOURCE!r}); this one says {source!r}",
                ["emit a registry point with --gain-point flagship or "
                 "benchmark, or produce the exact record with "
                 "scripts/certify_gain_point.py"])
        if exact_doc.get("gain_binding") != "printed_rationals":
            raise Refusal(
                "float_image_gains",
                f"{system}: the exact record states gain_binding = "
                f"{exact_doc.get('gain_binding')!r}; only an exact phase that "
                f"injected the exact rationals can supply the gain point")
        if exact_doc.get("system") != system:
            raise Refusal("unverifiable_gain_point",
                          f"the exact record is about "
                          f"{exact_doc.get('system')!r}, the row about "
                          f"{system!r}")
        for key in ("rational_gains", "brackets", "box"):
            if not isinstance(exact_doc.get(key, {}), dict):
                raise Refusal("unverifiable_gain_point",
                              f"exact record {key} is not an object")
        rational = {n: _exact_q(v, f"gain {n}")
                    for n, v in (exact_doc.get("rational_gains") or {}).items()}
        brackets_in = {}
        for name, b in (exact_doc.get("brackets") or {}).items():
            if not isinstance(b, dict) or any(
                    f not in b for f in ("lo", "hi", "power_k", "power_target")):
                raise Refusal("unverifiable_gain_point",
                              f"bracket {name}: needs lo, hi, power_k and "
                              f"power_target")
            lo = _exact_q(b["lo"], f"bracket {name} lo")
            hi = _exact_q(b["hi"], f"bracket {name} hi")
            k = _exact_int(b["power_k"], f"bracket {name} power_k")
            q = _exact_q(b["power_target"], f"bracket {name} power_target")
            self._check_bracket(name, lo, hi, k, q)
            brackets_in[name] = (lo, hi, k, q)
        box_in = {}
        for name, b in (exact_doc.get("box") or {}).items():
            if not isinstance(b, dict) or "lo" not in b or "hi" not in b:
                raise Refusal("unverifiable_gain_point",
                              f"box {name}: needs lo and hi")
            lo = _exact_q(b["lo"], f"box {name} lo")
            hi = _exact_q(b["hi"], f"box {name} hi")
            if not 0 < lo < hi:
                raise Refusal("unverifiable_gain_point",
                              f"box {name}: need 0 < lo < hi, got [{lo}, {hi}]")
            box_in[name] = (lo, hi)
        for a, b in (("rational_gains", "brackets"), ("rational_gains", "box"),
                     ("brackets", "box")):
            left = rational if a == "rational_gains" else brackets_in
            right = box_in if b == "box" else brackets_in
            overlap = sorted(set(left) & set(right))
            if overlap:
                raise Refusal("unverifiable_gain_point",
                              f"gains {overlap} are declared in both {a} and "
                              f"{b}")
        self.source_rational = dict(rational)
        self.source_brackets = dict(brackets_in)
        self.source_box = dict(box_in)
        #: the full corner box the exact phase proved, in its order
        self.source_varying = sorted(set(brackets_in) | set(box_in))

        # --- corner subset: fix BOX gains at one edge --------------------
        self.corner_subset = dict(corner_subset or {})
        for name, value in self.corner_subset.items():
            if name in brackets_in:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"--corner-subset {name}: {name} is a bracket of an "
                    f"irrational gain; fixing it at an endpoint would change "
                    f"the gain the record names. Only a box gain can be fixed")
            if name not in box_in:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"--corner-subset {name}: not a box gain of this record "
                    f"(box: {sorted(box_in)})")
            if value not in box_in[name]:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"--corner-subset {name}={value}: not an edge of the box "
                    f"[{box_in[name][0]}, {box_in[name][1]}]; the exact phase "
                    f"proved only the edges")
            rational[name] = value
            del box_in[name]

        # --- bracket targets on the remaining box gains ------------------
        self.bracket_targets = dict(bracket_targets or {})
        self.kinds = {n: "irrational_root_bracket" for n in brackets_in}
        out_brackets = {n: v for n, v in brackets_in.items()}
        for name, (k, q) in self.bracket_targets.items():
            if name in self.corner_subset:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"--bracket-target {name}: the gain is fixed by "
                    f"--corner-subset; it has no interval left")
            if name not in box_in:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"--bracket-target {name}: only a box gain takes a target "
                    f"(box: {sorted(box_in)}; a bracket already has one)")
            lo, hi = box_in[name]
            self._check_bracket(name, lo, hi, k, q)
            out_brackets[name] = (lo, hi, k, q)
            self.kinds[name] = "box_edge_with_target"
        for name, (lo, hi) in box_in.items():
            if name in out_brackets:
                continue
            out_brackets[name] = (lo, hi, 1, (lo + hi) / 2)
            self.kinds[name] = "box_edge_midpoint_target"
        self._rational = rational
        self._brackets = out_brackets
        self.box = dict(box_in)

        self._check_against_verifier()
        self.delta0 = _exact_q(exact_doc.get("delta0_exact", "0"),
                               "exact record delta0_exact")
        if self.delta0 < 0:
            raise Refusal("unverifiable_gain_point",
                          f"delta0 = {self.delta0} is negative")
        if self.delta0 > 0 and self.channel is None:
            raise Refusal(
                "unverifiable_gain_point",
                f"{system}: delta0 = {self.delta0} but the verifier admits no "
                f"constant matched disturbance channel for it")
        self._assignments = [
            dict(zip(self.bracketed_names, combo)) for combo in _product(
                [self._brackets[n][:2] for n in self.bracketed_names])]

    # -- checks ------------------------------------------------------------

    @staticmethod
    def _check_bracket(name, lo, hi, k, q):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise Refusal("unverifiable_gain_point",
                          f"bracket {name}: power_k must be an integer >= 1 "
                          f"(got {k!r})")
        if not (0 < lo < hi and lo ** k < q < hi ** k):
            raise Refusal(
                "unverifiable_gain_point",
                f"bracket {name}: [{lo}, {hi}] does not bracket the root of "
                f"g^{k} = {q} (0 < lo < hi and lo^k < q < hi^k fail in Q)")

    def _check_against_verifier(self):
        """Names, fixed parameters, positivity and affinity, as VC sees them."""
        cfg, system = self.cfg, self.system
        names = set(self._rational) | set(self._brackets)
        missing = [g for g in cfg["gain_names"] if g not in names]
        if missing:
            raise Refusal("unverifiable_gain_point",
                          f"{system}: gains {missing} of the verifier's "
                          f"equations are not in the exact record")
        fixed = {n: Fraction(v) for n, v in (cfg.get("fixed") or {}).items()}
        extra = sorted(names - set(cfg["gain_names"]) - set(fixed))
        if extra:
            raise Refusal("unverifiable_gain_point",
                          f"{system}: {extra} are not parameters of the "
                          f"verifier's equations {cfg['gain_names']}")
        for name, value in fixed.items():
            if name in self._brackets:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"{system}.{name} is a fixed factory parameter; it cannot "
                    f"vary (guard R6)")
            if name in self._rational and self._rational[name] != value:
                raise Refusal(
                    "unverifiable_gain_point",
                    f"{system}.{name} = {self._rational[name]}: the verifier's "
                    f"equations describe {name} = {value} only")
        for name, value in self._rational.items():
            if value <= 0:
                raise Refusal("unverifiable_gain_point",
                              f"{system}.{name} = {value}: gains must be "
                              f"positive")
        for name in self.bracketed_names:
            self._check_affine(name)

    def _check_affine(self, name):
        """Guard R6 membership plus an exact probe of the verifier's builder."""
        check_varying_gain_affinity(
            self.system, self._rational,
            {n: v[:2] for n, v in self._brackets.items()}, name)

    # -- the interface build_record reads ----------------------------------

    @property
    def bracketed_names(self):
        return tuple(sorted(self._brackets))

    def brackets(self):
        return {n: (v[0], v[1]) for n, v in self._brackets.items()}

    def bracket_power(self, name):
        return self._brackets[name][2], self._brackets[name][3]

    def rational_gains(self):
        return dict(self._rational)

    def corner_assignments(self):
        return [dict(a) for a in self._assignments]

    def corner_labels(self, assignment):
        out = {}
        for name, value in assignment.items():
            lo, hi = self._brackets[name][:2]
            if value == lo:
                out[name] = "lo"
            elif value == hi:
                out[name] = "hi"
            else:
                raise Refusal("gain_mismatch",
                              f"{self.system}.{name}: {value} is not an "
                              f"endpoint of [{lo}, {hi}]")
        return out

    def exact_gain_values(self, assignment):
        values = dict(self._rational)
        values.update(assignment or {})
        return values

    def select_bundle(self, bundle):
        """The source bundle, restricted to the corner subset (if any).

        The source bundle must hold the FULL corner box of the exact record,
        in its order; a corner kept by the subset loses only the fixed
        coordinates from its assignment, after they are checked exactly.
        ``build_record`` then pins every remaining assignment against this
        point's corners, positionally.
        """
        corners = bundle.get("corners") or []
        if len(corners) != 2 ** len(self.source_varying):
            raise Refusal(
                "incomplete_corner_box",
                f"{self.system}: the exact record's bundle has {len(corners)} "
                f"corners, its {len(self.source_varying)} varying gains need "
                f"{2 ** len(self.source_varying)}")
        if not self.corner_subset:
            return bundle
        kept = []
        for bc in corners:
            assignment = bc.get("assignment")
            if not isinstance(assignment, dict) \
                    or set(assignment) != set(self.source_varying):
                raise Refusal(
                    "gain_mismatch",
                    f"{self.system}: a bundle corner assigns "
                    f"{sorted(assignment or {})}, the exact record varies "
                    f"{self.source_varying}")
            values = {n: _exact_q(v, f"bundle corner {n}")
                      for n, v in assignment.items()}
            if all(values[n] == v for n, v in self.corner_subset.items()):
                reduced = dict(bc)
                reduced["assignment"] = {
                    n: v for n, v in assignment.items()
                    if n not in self.corner_subset}
                kept.append(reduced)
        if len(kept) != len(self._assignments):
            raise Refusal(
                "incomplete_corner_box",
                f"{self.system}: the corner subset "
                f"{({n: str(v) for n, v in self.corner_subset.items()})} keeps "
                f"{len(kept)} bundle corners, the remaining box needs "
                f"{len(self._assignments)}")
        out = dict(bundle)
        out["corners"] = kept
        return out

    def check_row(self, row, exact_doc):
        """Pin the row (V and float gains) and delta0 to the exact record."""
        row_delta0 = Fraction(str(row.get("delta0", 0)))
        if row_delta0 != self.delta0:
            raise Refusal("source_pin_mismatch",
                          f"row delta0 = {row.get('delta0')!r}, the exact "
                          f"record states delta0_exact = {self.delta0}")
        exact_v = exact_doc.get("V")
        if exact_v is not None:
            def key(terms):
                return [(float(t["coefficient"]),
                         tuple(Fraction(str(e)) for e in t["exponents"]),
                         tuple(t["term_types"])) for t in terms]
            if exact_v.get("degree_m") != row.get("degree_used") \
                    or key(exact_v.get("terms") or []) != key(row["V_terms"]):
                raise Refusal(
                    "source_pin_mismatch",
                    f"{self.system}: the V of the row is not the V the exact "
                    f"record proved")
        row_gains = row.get("gains") or {}
        for name, got in row_gains.items():
            got = float(got)
            if name in self.source_rational:
                if got != float(self.source_rational[name]):
                    raise Refusal(
                        "gain_mismatch",
                        f"{self.system}: row gain {name} = {got!r}, the exact "
                        f"record has {self.source_rational[name]}")
            elif name in self.source_brackets or name in self.source_box:
                lo, hi = (self.source_brackets[name][:2]
                          if name in self.source_brackets
                          else self.source_box[name])
                if not float(lo) <= got <= float(hi):
                    raise Refusal(
                        "gain_mismatch",
                        f"{self.system}: row gain {name} = {got!r} lies "
                        f"outside [{lo}, {hi}]")
            else:
                raise Refusal("gain_mismatch",
                              f"{self.system}: row gain {name} is not a gain "
                              f"of the exact record")

    def kappa(self, row):
        """kappa and weights from the framework's exact dynamics at corner 0.

        Built through the system's ``create_*_spec`` factory at this point's
        own exact gains (the registry supplies only the factory name and the
        gain roles, never a value); the verifier derives kappa itself from its
        hand-written equations and refuses a disagreement.
        """
        from gf_framework.systems import printed_gains as pg  # noqa: PLC0415
        template = None
        for table in (BENCHMARK_GAIN_POINTS, FLAGSHIP_GAIN_POINTS):
            if self.system in table:
                template = table[self.system]
                break
        if template is not None:
            factory, module = template.factory, template.module
            roles = {n: g.role for n, g in template.gains.items()}
        elif self.system in EXACT_RECORD_FACTORY_FALLBACK:
            factory, module = EXACT_RECORD_FACTORY_FALLBACK[self.system]
            roles = {}
        else:
            raise Refusal("unknown_system",
                          f"{self.system}: no create_*_spec factory is known")
        values = self.exact_gain_values(self._assignments[0])
        point = pg.GainPoint(
            system=self.system, factory=factory, module=module,
            gains={n: pg.PrintedGain(Fraction(v), "exact-record gain point",
                                     role=roles.get(n, "gain"))
                   for n, v in values.items()},
            channel=self.channel, note="exact-record gain point")
        try:
            structure = exact_dynamics(point, {})
        except (pg.PrintedGainError, TypeError) as exc:
            raise Refusal("unverifiable_gain_point",
                          f"{self.system}: the dynamics cannot be built in "
                          f"exact arithmetic at {values}: {exc}") from exc
        if list(structure.weights_r) != list(row["weights_r"]):
            raise Refusal("gain_mismatch",
                          f"{self.system}: the factory has weights "
                          f"{list(structure.weights_r)}, the row "
                          f"{list(row['weights_r'])}")
        self.factory, self.module = factory, module
        return exact_str(Fraction(str(structure.kappa)))

    def record(self):
        """``gain_point``: metadata, ignored by the verifier."""
        return {
            "system": self.system,
            "factory": getattr(self, "factory", None),
            "module": getattr(self, "module", None),
            "channel": self.channel,
            "source": "exact_record",
            "note": ("command-line gain point, not a printed-gain registry "
                     "point: gains, brackets, box and delta0 were read from "
                     "the exact-gains record of scripts/certify_gain_point.py "
                     "and re-checked in Q against the portable verifier's "
                     "equations"),
            "rational_gains": {n: str(v)
                               for n, v in sorted(self._rational.items())},
            "brackets": {
                n: {"lo": str(lo), "hi": str(hi), "power_k": k,
                    "power_target": str(q), "kind": self.kinds[n]}
                for n, (lo, hi, k, q) in sorted(self._brackets.items())},
            "box_encoding": (
                "a box edge [lo, hi] is a verifier bracket with power_k = 1 "
                "and power_target = (lo + hi)/2 unless a target was named; "
                "the target only names a point inside the interval, the "
                "verifier proves every corner and the affinity argument "
                "covers the interval") if self.box or any(
                    k != "irrational_root_bracket"
                    for k in self.kinds.values()) else None,
            "corner_subset": {n: str(v)
                              for n, v in sorted(self.corner_subset.items())},
            "n_corners": len(self._assignments),
        }

    def provenance(self, exact_path, exact_doc):
        stage = exact_doc.get("design_stage") or {}
        return {
            "note": ("the gain point came from the exact-gains record named "
                     "here, NOT from the printed-gain registry "
                     "(gf_framework.systems.printed_gains); the verifier "
                     "proves the gains, brackets and delta0 it reads as data"),
            "kind": "exact_record",
            "exact_record": Path(exact_path).as_posix(),
            "exact_record_sha256": sha256_of(exact_path),
            "gain_source": exact_doc.get("gain_source"),
            "gain_binding": exact_doc.get("gain_binding"),
            "design_stage": {"label": stage.get("label"),
                             "citable_chain": stage.get("citable_chain")},
            "source_rational_gains": {
                n: str(v) for n, v in sorted(self.source_rational.items())},
            "source_brackets": {
                n: {"lo": str(lo), "hi": str(hi), "power_k": k,
                    "power_target": str(q)}
                for n, (lo, hi, k, q) in sorted(self.source_brackets.items())},
            "source_box": {n: {"lo": str(lo), "hi": str(hi)}
                           for n, (lo, hi) in sorted(self.source_box.items())},
            "source_n_corners": 2 ** len(self.source_varying),
            "corner_subset": {n: str(v)
                              for n, v in sorted(self.corner_subset.items())},
            "bracket_targets": {n: {"power_k": k, "power_target": str(q)}
                                for n, (k, q) in sorted(
                                    self.bracket_targets.items())},
            "exact_stage_scope": (
                "provenance.exact_stage numbers are over all "
                f"{2 ** len(self.source_varying)} corners of the source "
                "record" + (", not only this subset" if self.corner_subset
                            else "")),
        }

    def stem_suffix(self):
        """File-name tag of a corner subset, so two subsets never collide."""
        if not self.corner_subset:
            return ""
        return "sub_" + "_".join(
            f"{n}_{str(v).replace('/', '_')}"
            for n, v in sorted(self.corner_subset.items())) + "_"


def _product(intervals):
    """Lexicographic product of (lo, hi) pairs, first name slowest."""
    if not intervals:
        return [()]
    return [(a,) + rest for a in intervals[0] for rest in _product(intervals[1:])]


_DERIVED_POINTS = (ScaledGainPoint, ExactRecordGainPoint)


def point_brackets(point):
    """name -> (lo, hi) of a registry point, a scaled one or an exact-record one."""
    if isinstance(point, _DERIVED_POINTS):
        return point.brackets()
    return registry_brackets(point)


def point_corner_assignments(point):
    if isinstance(point, _DERIVED_POINTS):
        return point.corner_assignments()
    return corner_assignments(point)


def point_corner_labels(point, assignment):
    if isinstance(point, _DERIVED_POINTS):
        return point.corner_labels(assignment)
    return corner_labels(point, assignment)


def point_exact_gain_values(point, assignment):
    if isinstance(point, _DERIVED_POINTS):
        return point.exact_gain_values(assignment)
    from gf_framework.systems.printed_gains import (  # noqa: PLC0415
        exact_gain_values)
    return exact_gain_values(point, assignment)


def point_rational_gains(point):
    if isinstance(point, _DERIVED_POINTS):
        return point.rational_gains()
    return {n: g.value for n, g in point.gains.items() if not g.is_bracketed}


def point_bracket_power(point, name):
    """(power_k, power_target) of one bracketed gain."""
    if isinstance(point, _DERIVED_POINTS):
        return point.bracket_power(name)
    gain = point.gains[name]
    return gain.power_k, gain.power_target


def point_record(point):
    if isinstance(point, _DERIVED_POINTS):
        return point.record()
    return gain_point_record(point)


def base_point(point):
    """The registry entry behind a point (itself, unless it is scaled).

    An exact-record point has none: the registry is not its source.
    """
    if isinstance(point, ExactRecordGainPoint):
        return None
    return point.base if isinstance(point, ScaledGainPoint) else point


# ---------------------------------------------------------------------------
# Compact encoding
# ---------------------------------------------------------------------------

def hex_float(value):
    """Exact float64 as its ``float.hex()`` string.

    Chosen over a decimal repr because it round-trips bit for bit through
    every JSON writer and reader, and because it is visibly not a decimal: a
    reader cannot mistake a Gram entry for a published number.
    """
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise Refusal("missing_gram_captures",
                      f"non-finite value {value!r} in a Gram capture")
    return value.hex()


def exact_str(value):
    """Exact rational as ``"p/q"`` (or ``"p"``), never a float encoding."""
    frac = value if isinstance(value, Fraction) else Fraction(str(value))
    return str(frac)


def read_json(path):
    """Parse a JSON file that may be gzipped (same sniff as the verifier)."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def write_json_gz(path, payload):
    """Write compact JSON through stdlib gzip, deterministically.

    ``mtime=0`` keeps the bytes reproducible: re-emitting an unchanged record
    must not produce a different blob, or the determinism audit (E18) would
    see spurious diffs.

    Capture lists are streamed one capture at a time (see :class:`Captures`),
    so a document larger than memory is written without ever existing as one
    Python string. At n = 5, m = 11 the grams sidecar is 1.09 GB of text.
    """
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           mtime=0) as fh:
            counter = [0]

            def write(text):
                blob = text.encode("utf-8")
                counter[0] += len(blob)
                fh.write(blob)

            _stream_json(write, payload)
    return counter[0]


def _stream_json(write, node):
    """Compact JSON, emitted piecewise, with capture lists streamed."""
    if isinstance(node, Captures):
        write("[")
        for index, cap in enumerate(node):
            if index:
                write(",")
            write(encode_capture_json(cap, f"{node.label}[{index}]"))
        write("]")
    elif isinstance(node, dict):
        write("{")
        for index, (key, value) in enumerate(node.items()):
            if index:
                write(",")
            write(json.dumps(str(key), separators=(",", ":")) + ":")
            _stream_json(write, value)
        write("}")
    elif isinstance(node, (list, tuple)):
        write("[")
        for index, value in enumerate(node):
            if index:
                write(",")
            _stream_json(write, value)
        write("]")
    else:
        write(json.dumps(node, separators=(",", ":")))


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Gram captures: locate, check, encode
# ---------------------------------------------------------------------------

CAPTURE_KEYS = ("signs", "mu", "eps", "basis", "gram")

RERUN_HINT = [
    "re-run the exact phase with Gram capture enabled: "
    "run_exact_phase() already returns v_grams and endpoints[].grams "
    "(verification/exact_phase.py); the driver has to persist them",
    "serialize each capture with signs, octant_index, eps, mu, y_degree, "
    "basis and gram (see _serialize_grams in "
    "scripts/exact_gate_standard_gains.py)",
    "write the bundle next to the run JSON and pass it to this script "
    "with --grams",
]


class Captures(list):
    """A list of per-octant Gram captures, encoded one capture at a time.

    It is an ordinary list, so every completeness check reads it exactly as
    before. The writer treats it specially: it encodes and emits one capture,
    then drops it, instead of building the whole document in memory. One
    capture of the five-state degree-11 flagship is a 1365 x 1365 float64
    matrix, about 22 MB of hex text; 64 of them are 1.09 GB.
    """

    def __init__(self, caps=(), label="captures"):
        super().__init__(caps)
        self.label = label


class LazyCapture(dict):
    """A capture whose Gram is read from its source only when encoded.

    Everything except the matrix is an ordinary dict entry. ``gram`` is
    produced by the loader on access, so the 64 matrices of a design bundle
    are never all resident at once.
    """

    def __init__(self, meta, load_gram):
        super().__init__(meta)
        self._load_gram = load_gram

    def __contains__(self, key):
        return key == "gram" or dict.__contains__(self, key)

    def __getitem__(self, key):
        if key == "gram":
            return self._load_gram()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key == "gram":
            return self._load_gram()
        return dict.get(self, key, default)


def encode_capture_json(cap, label):
    """One per-octant Gram capture in the portable encoding, as JSON text.

    Only the fields the verifier's gate inequality consumes are kept. The
    framework's capture also carries the float target polynomial; that is
    derived data, and the verifier recomputes it exactly from V and the
    hand-written dynamics instead of trusting it.

    The text is built row by row rather than through a nested Python object,
    which is what keeps a 22 MB capture from becoming several times that in
    list and float objects first.
    """
    missing = [k for k in CAPTURE_KEYS
               if k != "gram" and cap.get(k) is None]
    if "gram" not in cap or (not isinstance(cap, LazyCapture)
                             and cap.get("gram") is None):
        missing.append("gram")
    if missing:
        raise Refusal(
            "missing_gram_captures",
            f"{label}: capture is missing {missing}; a verdict summary is not "
            f"a Gram certificate and cannot be replayed",
            RERUN_HINT)
    head = {
        "signs": [int(s) for s in cap["signs"]],
        "octant_index": int(cap.get("octant_index", -1)),
        "eps": hex_float(cap["eps"]),
        "mu": [int(v) for v in cap["mu"]],
        "y_degree": int(cap["y_degree"]) if cap.get("y_degree") is not None
                    else 2 * sum(int(b) for b in cap["basis"][0]),
        "basis": [[int(b) for b in beta] for beta in cap["basis"]],
    }
    prefix = json.dumps(head, separators=(",", ":"))[:-1]
    gram = cap["gram"]
    rows = gram.tolist() if hasattr(gram, "tolist") else gram
    body = ",".join("[" + ",".join('"' + hex_float(x) + '"' for x in row) + "]"
                    for row in rows)
    return prefix + ',"gram":[' + body + "]}"


def encode_capture(cap, label):
    """The same capture as a dict (small captures, tests, hashing)."""
    return json.loads(encode_capture_json(cap, label))


def capture_digest(cap, label):
    """sha256 over the ENCODED capture: metadata and every Gram entry.

    Two captures with the same digest are the same witness byte for byte in
    the portable encoding, which is what makes sharing them across bracket
    corners a storage decision and not a mathematical one.
    """
    return hashlib.sha256(
        encode_capture_json(cap, label).encode("utf-8")).hexdigest()


def check_octant_coverage(caps, n_vars, label):
    """The 2^n sign patterns must be present exactly once each."""
    expected = 2 ** n_vars
    seen = [tuple(int(s) for s in c["signs"]) for c in caps]
    full = {tuple(p) for p in _sign_patterns(n_vars)}
    if len(caps) != expected or set(seen) != full or len(set(seen)) != len(seen):
        raise Refusal(
            "incomplete_octant_coverage",
            f"{label}: {len(set(seen) & full)} of {expected} octants of "
            f"{{-1,+1}}^{n_vars} captured ({len(caps)} capture entries); the "
            f"octant gate proves nothing about an uncaptured octant",
            RERUN_HINT)


def _sign_patterns(n_vars):
    if n_vars == 0:
        return [()]
    return [p + (s,) for p in _sign_patterns(n_vars - 1) for s in (-1, 1)]


def check_epsilon(caps, epsilon, label):
    """Every capture must carry the margin the record advertises.

    ``verify_gf_sos_full`` walks a descending eps ladder internally and returns
    whichever rung certified. A record that advertises the top rung while one
    octant only certified at a lower one claims a stronger margin than was
    proved, so it is refused here rather than at re-verification time.
    """
    for cap in caps:
        if float(cap["eps"]) != float(epsilon):
            raise Refusal(
                "epsilon_ladder_mismatch",
                f"{label}: capture certified at eps={cap['eps']!r} but the "
                f"record's margin is {epsilon!r}; a lower ladder rung "
                f"certified while the top rung would have been recorded",
                ["re-run the exact phase at the margin the record should "
                 "advertise, or record the rung that actually certified"])


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def load_row(path):
    """The certified row: system, V, weights, and (perturbed rows) delta0."""
    row = read_json(path)
    if "V_terms" not in row or "weights_r" not in row:
        raise Refusal("not_certified",
                      f"{Path(path).name}: not a certificate row (no V_terms)")
    if row.get("certified") is False:
        raise Refusal("not_certified",
                      f"{Path(path).name}: row is not certified")
    return row


def is_nominal(row):
    """A row is nominal when it carries no disturbance bound, or bound zero.

    The bound may be a float (row JSONs) or an exact rational string such as
    ``"1/16"`` (exact-cell headers, WP TA), so it is read through Fraction.
    """
    return "delta0" not in row or Fraction(str(row["delta0"])) == 0


def registry_structure(point, assignments, row, system):
    """kappa and weights from the registered spec, cross-checked with the row.

    The row records the weights the certificate was solved at; the registry
    knows the system. They must agree, or the record would pair a V with a
    homogeneity structure it was not built for.

    Weights and kappa are structural, independent of the gain values, so a
    scaled representative point is checked through its registry base point.
    An exact-record point builds them at its own corner-0 gains.
    """
    if isinstance(point, ExactRecordGainPoint):
        return point.kappa(row)
    if isinstance(point, ScaledGainPoint):
        point = point.base
        assignments = corner_assignments(point)
    structure = exact_dynamics(point, assignments[0])
    if list(structure.weights_r) != list(row["weights_r"]):
        raise Refusal(
            "gain_mismatch",
            f"{system}: the registered spec has weights "
            f"{list(structure.weights_r)}, the row was certified at "
            f"{list(row['weights_r'])}")
    return exact_str(Fraction(str(structure.kappa)))


def find_grams(exact_doc, exact_path, system, explicit):
    """Locate the Gram bundle for ``system``, or refuse with the re-run list.

    Three shapes are accepted, in order: an explicit ``--grams`` bundle, a
    ``grams_file`` companion (named either at the top level, as the exactgains
    records do, or per system row, as ``review/reverify_exact.py
    --dump-grams`` writes), and a bundle embedded in the exact artifact
    itself. A benchmark exact artifact produced WITHOUT ``--dump-grams``
    matches none of them: it records per-octant VERDICTS (degree, n_basis,
    eta_hat, resid_l1, charge, budget) but not the basis and the Gram, so it
    cannot be replayed and this is where the refusal comes from.
    """
    if explicit:
        return read_json(explicit), str(explicit)
    named = exact_doc.get("grams_file")
    if not named:
        for entry in exact_doc.get("systems", []):
            if entry.get("system") == system:
                named = entry.get("grams_file")
                break
    if named:
        for candidate in (exact_path.parent / named,
                          exact_path.parent / (named + ".gz")):
            if candidate.exists():
                return read_json(candidate), str(candidate)
        raise Refusal("missing_gram_captures",
                      f"record names grams_file {named!r}, which is not next "
                      f"to it in {exact_path.parent}", RERUN_HINT)
    if "V_grams" in exact_doc and "corners" in exact_doc:
        return exact_doc, str(exact_path)
    raise Refusal(
        "missing_gram_captures",
        f"{exact_path.name} carries no Gram captures for {system}: it records "
        f"per-octant verdicts (n_basis, eta_hat, resid_l1, charge, budget) "
        f"but not the basis and the Gram matrix, so the octant gate cannot be "
        f"replayed from it",
        RERUN_HINT)


def exact_entry(exact_doc, system):
    """The per-system block of an exact artifact, whatever its shape.

    Three shapes are in the tree: ``scripts/exact_gate_standard_gains.py``
    writes one record per system; ``review/reverify_exact.py`` writes a matrix
    with a ``systems`` list; ``scripts/levant_2003_perturbed.py`` writes a run
    with a ``sweeps`` list whose exact phase sits under ``exact_gate``. The
    last one is understood here so that a campaign perturbed run refuses for
    the reason it actually has (no Gram captures) rather than for a wrong one.
    """
    if exact_doc.get("system") == system:
        return exact_doc
    for entry in exact_doc.get("systems", []):
        if entry.get("system") == system:
            return entry
    for sweep in exact_doc.get("sweeps", []):
        if sweep.get("system") != system:
            continue
        gate = sweep.get("exact_gate") or {}
        return {"system": system, "epsilon": gate.get("epsilon"),
                "exact_pass": gate.get("exact_pass"),
                "git_sha": exact_doc.get("git_sha")}
    raise Refusal("not_certified",
                  f"exact artifact has no entry for system {system!r}")


def check_gain_binding(exact_doc, entry, system):
    """The exact phase must have bound the PRINTED gains, not their images.

    The June re-proof read every gain as ``Fraction(float(gain))``, so what it
    certified carried 1.1 as a dyadic rather than 11/10 (footnote 1 of
    CANONICAL_STATE.md). A portable record built on such a run would carry
    that caveat into the draft under a green column, so an artifact that does
    not establish printed-rational binding is refused rather than assumed
    innocent. Two things establish it: the explicit ``gain_binding`` flag, and
    being an exactgains record, whose driver injects exact rationals by
    construction and prints the brackets it used.
    """
    binding = entry.get("gain_binding") or exact_doc.get("gain_binding")
    if binding == "printed_rationals":
        return
    if binding is None and "rational_gains" in exact_doc and "brackets" in exact_doc:
        return
    raise Refusal(
        "float_image_gains",
        f"{system}: the exact phase does not establish printed-rational gain "
        f"binding (gain_binding = {binding!r}), so its gains may be the "
        f"float64 images; a record built on it would put that caveat under a "
        f"green column",
        ["re-run the exact phase with printed-rational gain injection "
         "(review/reverify_exact.py --printed-rationals, or "
         "scripts/exact_gate_standard_gains.py), which also records "
         "gain_binding in the artifact"])


def check_row_gains(row, point, system):
    """The row's float gains must name the registry's published values.

    Bracketed gains are compared against the float seed of the irrational
    value, the rest against the exact float image. This is the join between
    "the run that certified" and "the numbers the record prints".
    """
    row_gains = row.get("gains")
    if not row_gains:
        return
    if isinstance(point, ScaledGainPoint):
        # Only the --exact-cell route takes a scaled point, and a cell row
        # carries no float gains; its corners are pinned exactly instead.
        raise Refusal("gain_mismatch",
                      f"{system}: a row with float gains cannot be joined to "
                      f"a representative-p point; use the exact-cell route")
    bracket_names = set(point.bracketed_names)
    for name, entry in point.gains.items():
        if name not in row_gains:
            continue
        got, want = float(row_gains[name]), entry.float_image()
        tol = 1e-12 if name in bracket_names else 0.0
        if abs(got - want) > tol:
            raise Refusal(
                "gain_mismatch",
                f"{system}: row gain {name} = {got!r} but the printed-gain "
                f"registry has {want!r}; the record would name a gain point "
                f"the row was not certified at",
                ["check the --gain-point choice (benchmark vs flagship) and "
                 "the registry entry in gf_framework/systems/printed_gains.py"])


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def witness_provenance(exact_doc, entry):
    """Name the ONE witness the exact phase used (campaign item R3).

    A portable record has to say which object proposed the Grams it carries.
    Two values, decided by decisions D1/D2:

    ``resolve``
        the exact phase re-solved V and W with an independent strict float
        solver and verified the Grams that came back. The historical path.
    ``design_bundle``
        the DESIGN-stage Gram bundle was the witness, hash-verified entry by
        entry, and no float solve ran on the certifying route.

    An exact artifact written before P2-03 carries no ``witness`` key; that is
    unambiguously the re-solve path, so it is reported as such rather than as
    unknown. The verifier ignores this key — it is provenance, not an input to
    the proof, which is why adding it does not move the record schema.

    Campaign item P2-05 adds WHICH witness was adopted and what else was
    tried: ``witness_solver`` (the solver that proposed the adopted Gram),
    ``witness_charge_ratio`` (the adopted proof's worst exact charge against
    the MARGIN POLICY's budget, so comparable only within one policy) and
    ``witness_attempts``. A record written between P2-03 and P2-05
    has the source but not the solver; that reads as ``None``, never as a
    guess. None of it is an input to the proof: a witness is adopted because
    the exact rational phase verified it, and a solver status is never
    evidence (D6).
    """
    recorded = entry.get("witness") or exact_doc.get("witness")
    if not recorded:
        return {"source": "resolve", "gram_bundle_fingerprint": None,
                "parity_expanded": False, "recorded_by_source": False,
                "witness_solver": None, "witness_solver_used": None,
                "witness_charge_ratio": None, "witness_charge_over_eps": None,
                "witness_attempts": [], "parallel": {}}
    return {
        "source": recorded.get("source", "resolve"),
        "witness_source": recorded.get("witness_source",
                                       recorded.get("source", "resolve")),
        "witness_solver": recorded.get("witness_solver") or None,
        # P2-10's follow-up, landed by P3-05: the ENCODING that actually ran
        # ("MOSEK_DIRECT" = the Task API, "MOSEK" = the CVXPY interface), which
        # "witness_solver" above cannot distinguish because it is the REQUEST.
        # None for a record written before this field existed — never a guess.
        "witness_solver_used": recorded.get("witness_solver_used") or None,
        "witness_charge_ratio": recorded.get("witness_charge_ratio"),
        # WP P2-14: every ratio names its denominator. ``witness_charge_ratio``
        # above is charge / (the policy's budget); this one is charge / eps and
        # is comparable across policies and across records. None for a record
        # written before P2-14 — never a guess, and never back-derived by
        # halving, because that would assume a policy the record does not name.
        "witness_charge_over_eps": recorded.get("worst_charge_over_eps"),
        "witness_attempts": list(recorded.get("witness_attempts") or []),
        "gram_bundle_fingerprint": recorded.get("gram_bundle_fingerprint") or None,
        "parity_expanded": bool(recorded.get("parity_expanded")),
        "v_reused": bool(recorded.get("v_reused")),
        # P3-05: whether the V half came off the PERSISTED cross-process
        # proof cache, and if not, the typed reason it did not. Empty for a
        # record written before P3-05 — never a guess.
        "v_proof_cache": dict(recorded.get("v_proof_cache") or {}),
        # P3-03: jobs and the multiprocessing start method behind the exact
        # phase's per-octant schedule. A cost record, not a proof input: the
        # octant order and every recorded number are assembled by index and are
        # independent of it. Empty for a record written before P3-03.
        "parallel": dict(recorded.get("parallel") or {}),
        "recorded_by_source": True,
    }


def collapse_shared_w(system, corner_w, out_corners):
    """Write one nominal W witness set instead of 2^k identical copies.

    ``corner_w`` is one ``(label, captures, entry_hashes)`` per bracket
    corner, in corner order. ``entry_hashes`` is the list the EXACT PHASE
    recorded for that corner, when the source has one: for a design-bundle
    source those hashes are recomputed over the Gram bytes and pinned against
    the exact phase's own record before they ever reach here, so comparing
    them compares the witnesses the proof actually used. Without them the
    encoded captures are hashed directly, which is a strictly byte-level test.

    Every corner must name the same witness, entry for entry and in the same
    order. One corner that does not is a refusal: choosing which of two
    disagreeing Gram sets is "the" witness is not a decision an emitter may
    make. Nothing about the proof obligations changes here; the verifier
    still rebuilds each corner's dynamics from its own exact gains and proves
    all 2^n octants of that corner against the shared Gram.
    """
    if not corner_w:
        raise Refusal("missing_gram_captures",
                      f"{system}: --share-w-grams with no nominal W captures")
    digests = []
    for label, caps, entry_hashes in corner_w:
        if entry_hashes is not None:
            if len(entry_hashes) != len(caps):
                raise Refusal(
                    "source_pin_mismatch",
                    f"{system} corner {label}: the exact phase recorded "
                    f"{len(entry_hashes)} W entry hashes for {len(caps)} "
                    f"captures; the witness cannot be pinned")
            digests.append(("exact_phase_entry_hash", list(entry_hashes)))
        else:
            digests.append(("encoded_capture_sha256",
                            [capture_digest(c, f"W@{label}[{i}]")
                             for i, c in enumerate(caps)]))
    base_kind, base = digests[0]
    base_label = corner_w[0][0]
    for (kind, got), (label, _, _) in zip(digests[1:], corner_w[1:]):
        if kind != base_kind:
            raise Refusal(
                "shared_w_mismatch",
                f"{system}: corner {label} pins its W witness by {kind} while "
                f"corner {base_label} pins it by {base_kind}; the two corners "
                f"are not comparable, so they are not shareable")
        if got != base:
            first = next((i for i, (a, b) in enumerate(zip(got, base))
                          if a != b), min(len(got), len(base)))
            raise Refusal(
                "shared_w_mismatch",
                f"{system}: corner {label} names a different W Gram than "
                f"corner {base_label} at capture {first} "
                f"({got[first] if first < len(got) else 'missing'} vs "
                f"{base[first] if first < len(base) else 'missing'}); the "
                f"witness is not shared and must be written per corner",
                ["re-emit without --share-w-grams, which writes every "
                 "corner's own W captures"])
    shared = Captures(corner_w[0][1], "shared W")
    for out in out_corners:
        out.pop("W_grams", None)
        out["W_grams_shared"] = True
    return shared


def margin_claim_exceeds(record, derived=None):
    """Validate exact/float margin declarations and compare every stated bound."""
    try:
        raw_exact = record.get("eps_certified_exact")
        raw_float = record.get("eps_certified")
        if raw_exact is not None and not isinstance(raw_exact, str):
            raise ValueError("eps_certified_exact must be a rational string")
        exact = Fraction(raw_exact) if raw_exact is not None else None
        if isinstance(raw_float, bool):
            raise ValueError("eps_certified must be numeric, not boolean")
        approximate = Fraction(float(raw_float)) if raw_float is not None else None
        transcription = record.get("eps_certified_exact_is_float_transcription", False)
        if not isinstance(transcription, bool):
            raise ValueError("float-transcription flag must be boolean")
        if exact is not None and approximate is not None and (
                float(exact) != float(approximate)
                or transcription and exact != approximate):
            raise ValueError("eps_certified_exact and eps_certified disagree")
        if transcription and (exact is None or approximate is None):
            raise ValueError("a float transcription requires both margin fields")
        if any(v is not None and v <= 0 for v in (exact, approximate)):
            raise ValueError("a declared certified margin must be positive")
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise Refusal("margin_declaration_inconsistent", str(exc)) from exc
    return derived is not None and any(
        value is not None and value > derived
        and not (rounded and float(value) == float(derived))
        for value, rounded in ((exact, transcription), (approximate, True)))


def resolve_margin(witness_doc, epsilon, args, system):
    """The acceptance rule this record was proved under, never a default.

    Before WP R-05 a witness document without ``margin_policy`` was recorded
    as ``"half"`` by construction. That silent default is what the 2026-09-05
    review found on the five-state flagship: a record can then advertise a
    policy nothing in its sources states. Now there are exactly two sources,
    both explicit:

    * the exact phase's witness document, which records the policy it ran
      under and the margin it achieved;
    * an operator's ``--margin-policy``, for an artifact written before the
      field existed. It is recorded as an override in ``provenance``, so a
      reader can see that the policy is asserted and not measured.

    Neither present is a refusal. ``eps_certified`` has no override at all:
    it is a number, and a number a record claims must come from a proof
    stage. When the exact phase did not record one, the value is left open
    here and filled from the portable verifier's own derived output at
    self-verification (see :func:`self_verify`); if that cannot run either,
    the emission refuses rather than publish a margin nobody derived.

    WP R-08: what this function returns for ``eps_certified`` is the SOURCE
    stage's report, and it is labelled as such
    (``exact_phase_unverified_by_portable_stage``). It survives into the
    shipped record only for an emission that skips self-verification. When
    self-verification runs, :func:`main` replaces it with the portable
    verifier's own derived margin, always, and refuses the emission outright
    if this reported number is larger than what that proof supports.
    """
    recorded = witness_doc.get("margin_policy")
    override = getattr(args, "margin_policy", None)
    if recorded is None and not override:
        raise Refusal(
            "missing_margin_policy",
            f"{system}: the exact-phase witness document records no "
            f"margin_policy, so the acceptance rule this record was proved "
            f"under is unknown; it is not assumed",
            ["re-run the exact phase with a driver that records "
             "margin_policy (WP P2-13 and later do), or pass "
             "--margin-policy explicitly, which is recorded as an override "
             "in the record's provenance"])
    if recorded is not None and override and override != recorded:
        raise Refusal(
            "missing_margin_policy",
            f"{system}: --margin-policy {override!r} contradicts the "
            f"{recorded!r} the exact phase recorded; the emitter does not "
            f"overrule a proof stage")
    policy = recorded if recorded is not None else override
    margin_claim_exceeds(witness_doc)
    eps_certified = witness_doc.get("eps_certified")
    if eps_certified is not None:
        eps_certified = float(eps_certified)
    # An exact phase that stored its margin in Q keeps it; one that stored
    # only a float64 (every artifact of the 2026-08/09 campaign, levant5d
    # m = 11 included) gets an honest label instead of a promotion.
    reported_exact = witness_doc.get("eps_certified_exact")
    is_transcription = witness_doc.get(
        "eps_certified_exact_is_float_transcription",
        reported_exact is None and eps_certified is not None)
    if reported_exact is None and eps_certified is not None:
        reported_exact = str(Fraction(eps_certified))
    margin_claim_exceeds({"eps_certified": eps_certified,
                         "eps_certified_exact": reported_exact,
                         "eps_certified_exact_is_float_transcription": is_transcription})
    note_half = witness_doc.get("note_half_margin_holds")
    if note_half is None and eps_certified is not None and epsilon:
        # charge = eps - eps_certified, so the note's charge <= eps/2 is
        # eps_certified >= eps/2. Derived only from numbers the record itself
        # carries, never from a policy assumption.
        note_half = bool(eps_certified >= 0.5 * epsilon)
    return {
        "margin_policy": policy,
        "margin_policy_source": "exact_phase" if recorded is not None
                                else "operator_override",
        "eps_certified": eps_certified,
        # The exact phase's own rational when it stored one, otherwise the
        # EXACT value of the float64 it stored, so a reader never has to
        # re-parse a decimal. Either way it is that stage's number, not a
        # re-derived bound: the bound the record stands on is the one the
        # portable verifier derives in Q from the captures.
        "eps_certified_exact": reported_exact,
        "eps_certified_exact_is_float_transcription": is_transcription,
        # WP R-08: the source stage's report is never called a verified
        # margin. Self-verification overwrites this with the verifier's own.
        "eps_certified_source": (
            None if eps_certified is None
            else "exact_phase_unverified_by_portable_stage"),
        # Kept separately so it survives into provenance even after the
        # verifier's number replaces the one above.
        "source_reported_eps_certified": eps_certified,
        "source_reported_eps_certified_exact": reported_exact,
        "note_half_margin_holds": note_half,
    }


def build_record(row, exact_doc, entry, bundle, point, args, n_vars):
    """Assemble the schema-v2 record and the portable grams bundle."""
    system = row["system"]
    nominal = is_nominal(row)
    delta0 = Fraction(0) if nominal else Fraction(str(row["delta0"]))
    epsilon = next((float(e) for e in (entry.get("epsilon"),
                                       exact_doc.get("epsilon"), args.epsilon)
                    if e is not None), None)
    if not epsilon or epsilon <= 0:
        raise Refusal("epsilon_ladder_mismatch",
                      f"{system}: the exact artifact records no positive "
                      f"margin (got {epsilon!r})")

    reg_brackets = point_brackets(point)
    assignments = point_corner_assignments(point)
    if len(assignments) != 2 ** len(reg_brackets):
        raise Refusal("incomplete_corner_box",
                      f"{system}: registry produced {len(assignments)} corners "
                      f"for {len(reg_brackets)} bracketed gains")
    kappa = registry_structure(point, assignments, row, system)

    # --- V captures -----------------------------------------------------
    v_caps = bundle.get("V_grams")
    if not v_caps:
        raise Refusal("missing_gram_captures",
                      f"{system}: the bundle has no V_grams", RERUN_HINT)
    check_octant_coverage(v_caps, n_vars, "V")
    check_epsilon(v_caps, epsilon, "V")
    v_encoded = Captures(v_caps, "V")

    # --- corners --------------------------------------------------------
    bundle_corners = bundle.get("corners") or []
    if len(bundle_corners) != len(assignments):
        raise Refusal(
            "incomplete_corner_box",
            f"{system}: bundle has {len(bundle_corners)} corners, the "
            f"{len(reg_brackets)} bracketed gains need "
            f"{2 ** len(reg_brackets)}; the multi-affine argument covers the "
            f"box only from all of its vertices",
            RERUN_HINT)

    share_w = bool(getattr(args, "share_w_grams", False))
    if share_w and not nominal:
        raise Refusal(
            "nominal_perturbed_mismatch",
            f"{system}: --share-w-grams is a NOMINAL shape. A perturbed W "
            f"depends on the inclusion endpoint as well, and the two endpoint "
            f"capture sets are exactly what a dropped-endpoint record would "
            f"fake; the portable verifier refuses the shape there too")

    rec_corners, out_corners, seen_corners = [], [], set()
    shared_w_source = []
    for assignment, bc in zip(assignments, bundle_corners):
        labels = point_corner_labels(point, assignment)
        label = ", ".join(f"{k}={labels[k]}" for k in sorted(labels)) \
            or "rational gains only"
        exact_assignment = {k: exact_str(v) for k, v in sorted(assignment.items())}
        key = tuple(sorted(exact_assignment.items()))
        if key in seen_corners:
            raise Refusal("incomplete_corner_box",
                          f"{system}: corner {label} is duplicated; the corner "
                          f"box needs distinct vertices")
        seen_corners.add(key)
        # The corner order is positional: assignment i of the registry is
        # paired with bundle corner i. When the source bundle names its own
        # assignment, insist that the two agree, or the captures of one corner
        # would be recorded under another corner's gains.
        source_assignment = bc.get("assignment")
        if source_assignment is not None:
            got = {k: exact_str(v) for k, v in source_assignment.items()}
            if got != exact_assignment:
                raise Refusal(
                    "gain_mismatch",
                    f"{system}: bundle corner {bc.get('corner_label', label)!r} "
                    f"assigns {got}, the registry's corner {label} assigns "
                    f"{exact_assignment}; the corner boxes are not the same box")
        rec_corners.append({"assignment": exact_assignment,
                            "corner_label": label})

        out = {"corner_label": label, "assignment": exact_assignment}
        if nominal:
            if bc.get("endpoints"):
                raise Refusal(
                    "nominal_perturbed_mismatch",
                    f"{system} corner {label}: the row is nominal but the "
                    f"bundle carries inclusion endpoints")
            caps = bc.get("W_grams") or bc.get("nominal")
            if not caps:
                raise Refusal("missing_gram_captures",
                              f"{system} corner {label}: no nominal W captures",
                              RERUN_HINT)
            check_octant_coverage(caps, n_vars, f"W@{label}")
            check_epsilon(caps, epsilon, f"W@{label}")
            out["W_grams"] = Captures(caps, f"W@{label}")
            shared_w_source.append((label, caps, bc.get("entry_hashes")))
        else:
            endpoints = bc.get("endpoints")
            if bc.get("W_grams"):
                raise Refusal(
                    "nominal_perturbed_mismatch",
                    f"{system} corner {label}: the row is perturbed but the "
                    f"bundle carries a single nominal capture set")
            if not endpoints or len(endpoints) != 2:
                raise Refusal(
                    "missing_endpoint_captures",
                    f"{system} corner {label}: "
                    f"{0 if not endpoints else len(endpoints)} endpoint "
                    f"capture sets, need 2 (W+ and W-); one endpoint leaves "
                    f"half the disturbance interval unproved",
                    RERUN_HINT)
            out["endpoints"] = []
            for caps, lbl in zip(endpoints, ("W+", "W-")):
                check_octant_coverage(caps, n_vars, f"{lbl}@{label}")
                check_epsilon(caps, epsilon, f"{lbl}@{label}")
                out["endpoints"].append(Captures(caps, f"{lbl}@{label}"))
        out_corners.append(out)

    # --- one shared nominal W witness, when it is asked for and true ------
    shared_w_grams = None
    if share_w:
        shared_w_grams = collapse_shared_w(system, shared_w_source, out_corners)

    # WP P2-14, tightened by WP R-05: no silent default, ever.
    witness_doc = entry.get("witness") or exact_doc.get("witness") or {}
    margin = resolve_margin(witness_doc, epsilon, args, system)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": timestamp,
        "git_sha": entry.get("git_sha") or exact_doc.get("git_sha"),
        "system": system,
        "system_family": args.system_family,
        "nominal": nominal,
        "channel": None if nominal else row.get("channel", point.channel),
        "weights_r": list(row["weights_r"]),
        "kappa": kappa,
        "campaign_block": args.campaign_block,
        "row_source": Path(args.row).name,
        "witness": witness_provenance(exact_doc, entry),
        # WP P2-14 (owner decision D-l): the acceptance rule this record was
        # proved under, and the margin it actually proves. Additive keys —
        # every verifier in the chain ignores keys it does not know, so no
        # SCHEMA_VERSION bump is needed (P2-13 §Recommendation 1). A record
        # without them predates P2-13 and was proved under "half".
        # ``eps_certified`` is min over octants, forms, corners and endpoints
        # of eps - charge, and it, never ``epsilon``, is what a claim quotes.
        # WP R-05: both are resolved, never defaulted, and each names where it
        # came from (the exact phase, an explicit override, or the portable
        # verifier's own derived output).
        "margin_policy": margin["margin_policy"],
        "margin_policy_source": margin["margin_policy_source"],
        # WP R-08: while the source stage's number stands here it is labelled
        # unverified, and its exact string says whether it is a proof margin
        # in Q or the exact value of a rounded float64.
        "eps_certified": margin["eps_certified"],
        "eps_certified_exact": margin["eps_certified_exact"],
        "eps_certified_exact_is_float_transcription":
            margin["eps_certified_exact_is_float_transcription"],
        "eps_certified_source": margin["eps_certified_source"],
        "note_half_margin_holds": margin["note_half_margin_holds"],
        "delta0_exact": exact_str(delta0),
        "epsilon": epsilon,
        "epsilon_exact": exact_str(Fraction(epsilon)),
        "rational_gains": {
            name: exact_str(value)
            for name, value in sorted(point_rational_gains(point).items())
        },
        "V": {"degree_m": row["degree_used"],
              "weights_r": list(row["weights_r"]),
              "terms": [{"coefficient": hex_float(t["coefficient"]),
                         "exponents": [str(Fraction(e)) for e in t["exponents"]],
                         "term_types": list(t["term_types"])}
                        for t in row["V_terms"]]},
        "brackets": {
            name: {"lo": exact_str(lo), "hi": exact_str(hi),
                   "power_k": point_bracket_power(point, name)[0],
                   "power_target": exact_str(point_bracket_power(point, name)[1])}
            for name, (lo, hi) in sorted(reg_brackets.items())
        },
        "corners": rec_corners,
        "gain_point": point_record(point),
        "encoding": {
            "float64": "hex (float.hex(), exact round trip)",
            "exact": "rational string p/q; gains, delta0, margin and bracket "
                     "endpoints are never float encoded",
            "container": "gzip (stdlib), compact JSON separators",
            "shared_W": ("the nominal W witness is one capture set under "
                         "shared_W_grams; each corner carries "
                         "W_grams_shared: true. Storage only: every corner "
                         "still proves all 2^n octants against it, from its "
                         "own gains") if shared_w_grams is not None else None,
        },
        "coverage": {
            "octants_per_form": 2 ** n_vars,
            "bracket_corners": len(assignments),
            "inclusion_endpoints": 1 if nominal else 2,
            "rule": ("nominal: one dynamics, 2^n octants for V and 2^n for W "
                     "per corner. perturbed: 2^n for V, and 2 endpoints x 2^n "
                     "octants per corner."),
        },
    }
    record["shared_W_witness"] = shared_w_grams is not None
    grams = {"schema_version": SCHEMA_VERSION, "timestamp_utc": timestamp,
             "system": system, "nominal": nominal,
             "V_grams": v_encoded}
    if shared_w_grams is not None:
        grams["shared_W_grams"] = shared_w_grams
    grams["corners"] = out_corners
    return record, grams, margin


def self_verify(record_path, summary_path=None):
    """Run the standalone verifier on what was just written.

    A record that the dependency-free verifier does not accept is not a
    record. Running it here means a refusal happens at emission time, in the
    block that produced the certificate, rather than at the cut-line audit.

    With ``summary_path`` the verifier also writes its machine-readable
    summary, which is the ONLY place a record may take a margin the exact
    phase did not record: those numbers are the portable verifier's own
    derived output, in exact rational arithmetic, not a transcription.
    """
    cmd = [sys.executable, str(STANDALONE), str(record_path)]
    if summary_path is not None:
        cmd += ["--json", str(summary_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        failure_text = (proc.stderr or proc.stdout).strip()
        raise Refusal(
            ("margin_claim_exceeds_proof" if "MARGIN CLAIM EXCEEDS PROOF" in failure_text
             else "margin_declaration_inconsistent" if "margin declarations disagree" in failure_text
             else "self_verification_failed"),
            f"the emitted record did not verify: "
            f"{(proc.stderr or proc.stdout).strip()[-500:]}",
            ["inspect the source artifact: either the captures do not prove "
             "their targets, or the hand-written dynamics in "
             "verification/standalone/verify_certificate.py disagree with the "
             "gain point the record names"])
    return proc.stdout


# ---------------------------------------------------------------------------
# Source shape: a design Gram bundle (npz) plus a per-corner exact cell
# ---------------------------------------------------------------------------

class CellSource:
    """The design-bundle route: ``--design-record``, ``--design-bundle-npz``,
    ``--exact-cell``.

    The exact phase of a large row does not write one artifact with the Grams
    in it. It writes a CELL: a header, a ROW verdict, and one record per
    bracket corner, each naming the design bundle entries it proved on by
    content hash, never by value. The Grams themselves live in the design
    bundle npz. This class joins the three back together, and it PINS them
    while doing so:

    * every Gram entry hash is recomputed over the little-endian float64
      bytes of the matrix and must equal the hash the exact phase recorded
      for that corner, on every corner, for V and for W;
    * the whole-bundle fingerprint is recomputed from those hashes and must
      equal the one the exact phase and the design record both name;
    * the V input hash is recomputed from the design record's terms and the
      exact epsilon and must equal the one the exact phase proved under;
    * every corner's exact gains must be the printed-gain registry's exact
      rationals at that corner of the bracket box.

    Any of those failing is a refusal. What is left after they pass is a pure
    transcription: nothing is re-designed, re-solved or re-proved here.
    """

    def __init__(self, args):
        self.design_record_path = Path(args.design_record)
        self.bundle_path = Path(args.design_bundle_npz)
        self.cell = Path(args.exact_cell)
        self.design = read_json(self.design_record_path)
        self.header = read_json(self.cell / "header.json")
        self.row_doc = read_json(self.cell / "ROW.json")
        self.corner_paths = sorted(self.cell.glob("corner_*.json"),
                                   key=lambda p: int(p.stem.split("_")[1]))
        self.corners = [read_json(p) for p in self.corner_paths]
        self._entry_hashes = None
        if not self.corners:
            raise Refusal("not_certified",
                          f"{self.cell}: no corner_*.json records")
        if not self.row_doc.get("exact_pass_all"):
            raise Refusal("not_certified",
                          f"{self.cell}/ROW.json: exact_pass_all is not true; "
                          f"there is nothing to make portable")
        for corner in self.corners:
            if not corner.get("exact_pass"):
                raise Refusal(
                    "not_certified",
                    f"{self.cell}: corner {corner['corner']['index']} is not "
                    f"an exact pass")
        self.representative_p = getattr(args, "representative_p", None)
        self.delta0 = self._resolve_delta0()
        self.nominal = self.delta0 == 0
        self._check_representative_p()

    # -- WP TA: the disturbance bound and the representative p -------------

    def _resolve_delta0(self):
        """The cell's disturbance bound, exact, pinned across its sources.

        Exact-phase headers written by the perturbed drivers carry
        ``delta0_exact`` (a rational string); an older nominal header carries
        at most a float ``delta0``. Every other place that states the bound
        (the float in the header, each corner record, the design record, the
        design bundle document) must agree with it, or the record would name
        a disturbance interval that some part of the proof did not use.
        """
        header = self.header
        exact = header.get("delta0_exact")
        loose = header.get("delta0")
        if exact is not None:
            try:
                delta0 = Fraction(str(exact))
            except (ValueError, ZeroDivisionError) as exc:
                raise Refusal("source_pin_mismatch",
                              f"header delta0_exact {exact!r} is not a "
                              f"rational: {exc}") from exc
        elif loose is not None:
            delta0 = Fraction(str(loose))
        else:
            delta0 = Fraction(0)
        if delta0 < 0:
            raise Refusal("source_pin_mismatch",
                          f"header states a negative disturbance bound {delta0}")
        stated = [("header delta0", loose),
                  ("header delta0_float", header.get("delta0_float")),
                  ("design record delta0_exact", self.design.get("delta0_exact")),
                  ("design record delta0_float", self.design.get("delta0_float"))]
        stated += [(f"corner {c['corner'].get('index', i)} delta0_exact",
                    c["corner"].get("delta0_exact"))
                   for i, c in enumerate(self.corners)]
        for label, value in stated:
            if value is None:
                continue
            if isinstance(value, float):
                agrees = value == float(delta0)
            else:
                agrees = Fraction(str(value)) == delta0
            if not agrees:
                raise Refusal(
                    "source_pin_mismatch",
                    f"{label} = {value!r} disagrees with the cell's "
                    f"disturbance bound {delta0}")
        return delta0

    def _check_representative_p(self):
        """``--representative-p`` must be the p the cell was proved at."""
        stated = self.header.get("qscale_p")
        for label, value in (("design record qscale_p",
                              self.design.get("qscale_p")),):
            if value is not None and stated is not None \
                    and Fraction(str(value)) != Fraction(str(stated)):
                raise Refusal("source_pin_mismatch",
                              f"{label} = {value!r}, the exact cell header "
                              f"states qscale_p = {stated!r}")
        given = self.representative_p
        if stated is None:
            return
        stated = Fraction(str(stated))
        if given is None:
            if stated != 1:
                raise Refusal(
                    "gain_mismatch",
                    f"{self.cell.name}: the cell was proved at the covariant "
                    f"representative p = {stated}, so its gains are the "
                    f"registry point scaled by p^i; the unscaled registry "
                    f"cannot pin them",
                    [f"pass --representative-p {stated}"])
            return
        if given != stated:
            raise Refusal(
                "gain_mismatch",
                f"--representative-p {given} contradicts the qscale_p = "
                f"{stated} the exact cell was proved at")

    def check_scaled_point(self, point):
        """Pin the emitter's scaled point against the driver's own Q checks.

        The exact-phase driver wrote the scaled bracket endpoints and power
        targets it verified into ``bracket_checks_in_Q``, and the ratio
        ``delta0 / p^n`` it ran at. Both were computed by a different program
        from this one; they must agree exactly.
        """
        if not isinstance(point, ScaledGainPoint):
            return
        checks = self.header.get("bracket_checks_in_Q") or {}
        for name, check in checks.items():
            if name not in point.bracketed_names:
                raise Refusal(
                    "gain_mismatch",
                    f"the cell checked a bracket for {name}, the scaled point "
                    f"brackets {list(point.bracketed_names)}")
            lo, hi = point.brackets()[name]
            k, target = point.bracket_power(name)
            got = (Fraction(check["lo_scaled"]), Fraction(check["hi_scaled"]),
                   int(check["power_k"]), Fraction(check["power_target_scaled"]))
            if got != (lo, hi, k, target) or check.get("valid_in_Q") is not True:
                raise Refusal(
                    "gain_mismatch",
                    f"bracket {name}: the cell proved [{got[0]}, {got[1]}] with "
                    f"g^{got[2]} = {got[3]}, the scaled point is [{lo}, {hi}] "
                    f"with g^{k} = {target}")
        ratio = self.header.get("ratio_delta0_over_L")
        if ratio is not None:
            want = Fraction(str(ratio)) * point.p ** point.delta0_power
            if want != self.delta0:
                raise Refusal(
                    "source_pin_mismatch",
                    f"the cell states ratio delta0/L = {ratio} at p = "
                    f"{point.p}, which is delta0 = {want}, but its bound is "
                    f"{self.delta0}")

    # -- the two documents the emitter's own pipeline consumes -------------

    def row(self):
        """The certified row, assembled from the design record and header."""
        terms = self.design.get("candidate_V_terms")
        if not terms:
            raise Refusal("not_certified",
                          f"{self.design_record_path.name}: no "
                          f"candidate_V_terms to transcribe")
        if self.header.get("n_v_terms") not in (None, len(terms)):
            raise Refusal(
                "source_pin_mismatch",
                f"the exact phase proved {self.header['n_v_terms']} V terms, "
                f"the design record holds {len(terms)}")
        return {"system": self.header["system"],
                "weights_r": list(self.header["weights_r"]),
                "degree_used": self.header["m"],
                # WP TA: the exact bound resolved and pinned in __init__
                # (``delta0_exact`` first), as a rational string.
                "delta0": str(self.delta0),
                "certified": True,
                "V_terms": terms}

    def exact_doc(self):
        """The exact artifact, in the shape ``exact_entry`` already reads.

        ``gain_binding`` is asserted as printed rationals because this source
        CHECKS it: every corner's gains are compared against the registry's
        exact rationals in :meth:`bundle`, so the binding is established here
        rather than believed from a flag.
        """
        witness = dict(self.corners[0].get("witness") or {})
        for key in ("V", "endpoints", "witness_attempts", "v_proof_cache",
                    "parallel"):
            witness.pop(key, None)
        policies = {c.get("margin_policy") for c in self.corners}
        policies.discard(None)
        if len(policies) > 1:
            raise Refusal(
                "missing_margin_policy",
                f"the corners of {self.cell.name} were proved under different "
                f"margin policies {sorted(policies)}; one record cannot name "
                f"them all")
        if policies:
            witness["margin_policy"] = policies.pop()
        elif "margin_policy" in witness:
            witness.pop("margin_policy")
        certified = [c.get("eps_certified_float") for c in self.corners]
        row_certified = self.row_doc.get("eps_certified_over_corners")
        if all(v is not None for v in certified):
            worst = min(float(v) for v in certified)
            if row_certified is not None and float(row_certified) != worst:
                raise Refusal(
                    "source_pin_mismatch",
                    f"ROW.json reports eps_certified {row_certified} over the "
                    f"corners, the corner records report {worst}")
            witness["eps_certified"] = worst
        return {"system": self.header["system"],
                "epsilon": self.header["epsilon"],
                "exact_pass_all": self.row_doc.get("exact_pass_all"),
                "git_sha": self.header.get("pinned_git_sha"),
                "gain_binding": "printed_rationals",
                "witness": witness}

    # -- the Gram bundle, pinned entry by entry ---------------------------

    def bundle(self, point, epsilon):
        import numpy as np  # noqa: PLC0415  (only this source shape needs it)
        from gf_framework.verification.joint_sos import (  # noqa: PLC0415
            gram_bundle_entry_hash, gram_bundle_fingerprint)

        expect_sha = ((self.design.get("bundle") or {}).get("sha256")
                      or self.header.get("bundle_sha256"))
        got_sha = sha256_of(self.bundle_path)
        if expect_sha and got_sha != expect_sha:
            raise Refusal(
                "source_pin_mismatch",
                f"{self.bundle_path.name}: sha256 {got_sha} is not the "
                f"{expect_sha} the design record names")
        self.bundle_sha256 = got_sha
        archive = np.load(self.bundle_path, allow_pickle=True)
        doc = json.loads(str(archive["_document"]))
        entries = doc["entries"]
        # WP TA: the design bundle states the bound and the representative it
        # was designed at; both must be the cell's.
        if doc.get("delta0") is not None \
                and float(doc["delta0"]) != float(self.delta0):
            raise Refusal(
                "source_pin_mismatch",
                f"{self.bundle_path.name} was designed at delta0 = "
                f"{doc['delta0']!r}, the cell proves delta0 = {self.delta0}")
        if doc.get("qscale_p") is not None \
                and self.header.get("qscale_p") is not None \
                and Fraction(str(doc["qscale_p"])) \
                != Fraction(str(self.header["qscale_p"])):
            raise Refusal(
                "source_pin_mismatch",
                f"{self.bundle_path.name} was designed at p = "
                f"{doc['qscale_p']}, the cell states p = "
                f"{self.header['qscale_p']}")
        self.check_scaled_point(point)

        # Recompute every entry hash over the Gram BYTES. This is what makes
        # the matrices shipped here the matrices the exact phase proved on,
        # and not merely matrices from a file of the same name.
        recomputed, by_name = {}, {}
        for index, meta in enumerate(entries):
            entry = dict(meta)
            entry["gram"] = archive[f"gram_{index}"]
            recomputed[meta["name"]] = gram_bundle_entry_hash(entry)
            by_name[meta["name"]] = index
        fingerprint = gram_bundle_fingerprint(
            [{"hash": recomputed[m["name"]]} for m in entries])
        expect_fp = (self.header.get("expect_fingerprint")
                     or (self.design.get("bundle") or {}).get("fingerprint"))
        if expect_fp and fingerprint != expect_fp:
            raise Refusal(
                "source_pin_mismatch",
                f"the recomputed bundle fingerprint {fingerprint} is not the "
                f"{expect_fp} the exact phase proved on")
        self.bundle_fingerprint = fingerprint
        self._entry_hashes = recomputed

        self._check_v_input_hash(epsilon)

        def capture(name):
            meta = entries[by_name[name]]
            index = by_name[name]
            return LazyCapture(
                {k: meta[k] for k in ("signs", "octant_index", "eps", "mu",
                                      "y_degree", "basis")},
                lambda index=index: archive[f"gram_{index}"])

        v_names, corners_out = None, []
        assignments = point_corner_assignments(point)
        if len(assignments) != len(self.corners):
            raise Refusal(
                "incomplete_corner_box",
                f"the registry has {len(assignments)} bracket corners, the "
                f"exact cell recorded {len(self.corners)}")
        for position, (assignment, corner) in enumerate(
                zip(assignments, self.corners)):
            index = corner["corner"]["index"]
            if index != position:
                raise Refusal(
                    "incomplete_corner_box",
                    f"corner record {position} calls itself corner {index}; "
                    f"the corner order is positional and must not be guessed")
            witness = corner["witness"]
            got_v = [(o["octant_index"], o["block_name"], o["entry_hash"])
                     for o in witness.get("V") or []]
            endpoints = witness.get("endpoints") or []
            if self.nominal:
                if len(endpoints) != 1:
                    raise Refusal(
                        "nominal_perturbed_mismatch",
                        f"corner {index}: {len(endpoints)} inclusion "
                        f"endpoints, but the cell's disturbance bound is 0; a "
                        f"nominal cell proves exactly one dynamics")
                w_sets = [[(o["octant_index"], o["block_name"], o["entry_hash"])
                           for o in endpoints[0]]]
            else:
                w_sets = self._perturbed_endpoints(index, endpoints, entries,
                                                   by_name)
            got_w = [item for w_set in w_sets for item in w_set]
            # V is gain-independent and is proved once, at the first corner;
            # later corners record ``v_reused`` and carry no V list. The
            # hashes are pinned where they exist, and the reuse flag is
            # REQUIRED where they do not, rather than accepting a hole.
            if not got_v and not corner.get("v_reused"):
                raise Refusal(
                    "source_pin_mismatch",
                    f"corner {index}: no V witness entries and v_reused is "
                    f"not set; the V witness of this corner is unaccounted for")
            for label, got in (("V", got_v), ("W", got_w)):
                for _, name, digest in got:
                    if recomputed.get(name) != digest:
                        raise Refusal(
                            "source_pin_mismatch",
                            f"corner {index} {label} block {name}: the exact "
                            f"phase recorded entry_hash {digest}, the bundle "
                            f"on disk hashes to {recomputed.get(name)}")
            if got_v:
                if v_names is not None and got_v != v_names:
                    raise Refusal(
                        "source_pin_mismatch",
                        f"corner {index} names a different V witness than the "
                        f"first corner")
                v_names = got_v
            self._check_corner_gains(point, assignment, corner)
            # The assignment comes from the CORNER RECORD's own exact gains,
            # not from the registry, so the emitter's corner cross-check
            # compares two independently obtained boxes rather than one with
            # itself.
            proved = corner["corner"]["exact_gains"]
            out = {
                "corner_label": corner["corner"].get("label", str(index)),
                "assignment": {name: exact_str(Fraction(proved[name]))
                               for name in sorted(assignment)},
                # What the exact phase recorded for THIS corner, already
                # pinned against the bytes above. --share-w-grams compares
                # these, so sharing is decided on the proof's own hashes.
                "entry_hashes": [digest for _, _, digest in got_w],
            }
            if self.nominal:
                out["W_grams"] = [capture(name) for _, name, _ in got_w]
            else:
                # [W+, W-]: the order build_record and the verifier read.
                out["endpoints"] = [[capture(name) for _, name, _ in w_set]
                                    for w_set in w_sets]
            corners_out.append(out)
        if v_names is None:
            raise Refusal("source_pin_mismatch",
                          "no corner records the V witness entries")
        return {"V_grams": [capture(name) for _, name, _ in v_names],
                "corners": corners_out}

    _W_BLOCK = re.compile(r"^W_oct(\d+)_v(\d+)$")

    def _perturbed_endpoints(self, index, endpoints, entries, by_name):
        """The two endpoint witness lists of a perturbed corner, as [W+, W-].

        The design bundle names each W block ``W_oct<i>_v<k>``, where ``k`` is
        the position of the dynamics in the vertex list the design and the
        exact phase were given, and ``matched_disturbance_vertices`` returns
        ``[f + delta0 e_ch, f - delta0 e_ch]``: ``v0`` is W+ and ``v1`` is W-.
        The mapping is by that suffix, not by list position, and every block
        of one list must carry the same suffix, the octant index its name
        states, and (when the bundle records it) the same ``endpoint`` field.
        Anything else is a refusal: the verifier would rebuild the W+ target
        against whatever the record calls W+, and a swapped pair is a record
        about the wrong vertex.
        """
        if len(endpoints) != 2:
            raise Refusal(
                "missing_endpoint_captures",
                f"corner {index}: {len(endpoints)} endpoint witness lists, the "
                f"cell's disturbance bound {self.delta0} > 0 needs 2 (W+ and "
                f"W-); one endpoint leaves half the interval unproved",
                RERUN_HINT)
        by_vertex = {}
        for position, endpoint in enumerate(endpoints):
            vertices = set()
            for item in endpoint:
                name = item["block_name"]
                match = self._W_BLOCK.match(name)
                if not match:
                    raise Refusal(
                        "source_pin_mismatch",
                        f"corner {index} endpoint list {position}: block "
                        f"{name!r} is not a W_oct<i>_v<k> block, so its "
                        f"inclusion vertex is unknown")
                octant, vertex = int(match.group(1)), int(match.group(2))
                if octant != int(item["octant_index"]):
                    raise Refusal(
                        "source_pin_mismatch",
                        f"corner {index}: block {name} is recorded for octant "
                        f"{item['octant_index']}")
                meta = entries[by_name[name]] if name in by_name else {}
                if meta.get("endpoint") is not None \
                        and int(meta["endpoint"]) != vertex:
                    raise Refusal(
                        "source_pin_mismatch",
                        f"corner {index}: block {name} is stored as endpoint "
                        f"{meta['endpoint']} in the design bundle")
                vertices.add(vertex)
            if len(vertices) != 1:
                raise Refusal(
                    "source_pin_mismatch",
                    f"corner {index} endpoint list {position} mixes inclusion "
                    f"vertices {sorted(vertices)}; one list must be one vertex")
            vertex = vertices.pop()
            if vertex in by_vertex:
                raise Refusal(
                    "missing_endpoint_captures",
                    f"corner {index}: both endpoint lists witness vertex "
                    f"v{vertex}; the other endpoint is unproved", RERUN_HINT)
            by_vertex[vertex] = [(o["octant_index"], o["block_name"],
                                  o["entry_hash"]) for o in endpoint]
        if set(by_vertex) != {0, 1}:
            raise Refusal(
                "missing_endpoint_captures",
                f"corner {index}: endpoint vertices {sorted(by_vertex)}, need "
                f"v0 (+delta0, W+) and v1 (-delta0, W-)", RERUN_HINT)
        return [by_vertex[0], by_vertex[1]]

    def _check_v_input_hash(self, epsilon):
        """Recompute the exact V input hash and pin it to the exact phase."""
        expect = self.header.get("v_input_hash")
        if not expect:
            return
        from fractions import Fraction as _F  # noqa: PLC0415
        from gf_framework.algebra.core import (  # noqa: PLC0415
            GeneralizedForm, GFTerm)
        from gf_framework.verification.exact_phase import (  # noqa: PLC0415
            v_input_hash)
        terms = []
        for term in self.design["candidate_V_terms"]:
            coeff = term["coefficient"]
            coeff = float.fromhex(coeff) if isinstance(coeff, str) \
                else float(coeff)
            terms.append(GFTerm(coefficient=coeff,
                                exponents=tuple(_F(str(e))
                                                for e in term["exponents"]),
                                term_types=tuple(term["term_types"])))
        gf = GeneralizedForm(degree_m=self.header["m"],
                             weights_r=tuple(self.header["weights_r"]),
                             terms=terms)
        got = v_input_hash(gf, epsilon)
        if got != expect:
            raise Refusal(
                "source_pin_mismatch",
                f"the V transcribed from {self.design_record_path.name} hashes "
                f"to {got}, the exact phase proved {expect}")

    def _check_corner_gains(self, point, assignment, corner):
        """The registry's exact gains at this corner are the proved gains.

        For a scaled representative point (WP TA) "the registry's exact
        gains" are the flagship point's, scaled by ``p^i`` in Q.
        """
        want = point_exact_gain_values(point, assignment)
        got = {name: Fraction(value)
               for name, value in corner["corner"]["exact_gains"].items()}
        if got != want:
            raise Refusal(
                "gain_mismatch",
                f"corner {corner['corner']['index']}: the exact phase proved "
                f"gains {({k: str(v) for k, v in got.items()})}, the printed "
                f"gain registry has "
                f"{({k: str(v) for k, v in want.items()})}")

    def source_artifacts(self):
        roles = [("design_record", self.design_record_path),
                 ("design_gram_bundle", self.bundle_path),
                 ("exact_cell_header", self.cell / "header.json"),
                 ("exact_cell_row", self.cell / "ROW.json")]
        roles += [(f"exact_corner_{index}", path)
                  for index, path in enumerate(self.corner_paths)]
        crosscheck = (self.header.get("corners_crosscheck") or {}).get("path")
        if crosscheck and Path(crosscheck).exists():
            roles.append(("gain_point_corners", Path(crosscheck)))
        return roles


def registry_source_artifact(point):
    """The printed-gain registry, as a source of the record's gain point.

    Two hashes: the module the gains are written in, and the digest of the
    gain point record this emission read out of it. The second is what a
    reader compares when the module has moved on.
    """
    import gf_framework.systems.printed_gains as printed  # noqa: PLC0415
    path = Path(printed.__file__)
    # Repo-relative when the package is the checkout's own (the usual
    # editable install): an absolute path would put a host's home directory
    # into a record that is read on other machines. The hash is unchanged.
    try:
        shown = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        shown = path
    digest = hashlib.sha256(json.dumps(gain_point_record(point),
                                       sort_keys=True,
                                       separators=(",", ":")).encode()
                            ).hexdigest()
    return {"role": "gain_point_registry", "path": shown.as_posix(),
            "sha256": sha256_of(path), "gain_point_digest": digest}


VERIFIED_MARGIN_SCHEMA = "gf-verified-margin/1"


def attach_verify_summary(record_path, summary_path, out_dir=None, grams_path=None):
    """Write the verifier's exact margin BESIDE an already emitted record.

    The five-state flagship is emitted with ``--no-self-verify`` because its
    portable proof takes about ten hours, so its record states the exact
    phase's own reported number, labelled unverified. When that proof
    finishes, this attaches the authoritative margin without touching a byte
    of the record the verifier accepted: re-writing the record would mean the
    shipped bytes are not the bytes that were proved.

    The sidecar names both inputs by content hash, carries the verifier's
    exact rational, and states plainly whether the record's own reported
    number is covered by the proof.
    """
    record_path = Path(record_path)
    summary_path = Path(summary_path)
    summary = read_json(summary_path)
    record = read_json(record_path)   # sniffs gzip, exactly as the verifier does

    if summary.get("verdict") != "VERIFIED":
        raise Refusal(
            "self_verification_failed",
            f"{summary_path.name} is not a successful verification "
            f"(verdict {summary.get('verdict')!r}); there is no proved margin "
            f"to attach",
            ["re-run verification/standalone/verify_certificate.py on this "
             "record with --json and attach the summary it writes"])
    binding = summary.get("proof_inputs")
    if not isinstance(binding, dict) or binding.get("schema") != "gf-proof-inputs/1":
        raise Refusal("proof_input_binding_missing",
                      "the summary has no proof-time input hashes; it cannot be attached. "
                      "Keep it as historical evidence or re-run the portable verifier")
    required = {"record": record_path,
                "verifier": Path(__file__).resolve().parents[1] /
                            "verification/standalone/verify_certificate.py"}
    if grams_path is not None:
        required["grams"] = Path(grams_path)
    elif isinstance(record.get("grams_file"), str):
        name = record["grams_file"]
        candidates = [record_path.parent / name, record_path.parent / (name + ".gz"),
                      record_path.parent / name.replace(".gz", "")]
        required["grams"] = next((p for p in candidates if p.is_file()), candidates[0])
    else:
        raise Refusal("proof_input_binding_missing", "no Gram input was supplied")
    if "V" not in record:
        row_binding = binding.get("row")
        if not isinstance(row_binding, dict) or not isinstance(row_binding.get("path"), str):
            raise Refusal("proof_input_binding_missing", "the external V row is not bound")
        required["row"] = Path(row_binding["path"])
    for role, input_path in required.items():
        pin = binding.get(role)
        if not isinstance(pin, dict) or not isinstance(pin.get("sha256"), str):
            raise Refusal("proof_input_binding_missing", f"no proof-time hash for {role}")
        if not input_path.is_file() or sha256_of(input_path) != pin["sha256"]:
            raise Refusal("proof_input_mismatch",
                          f"{role} differs from the bytes used by the portable proof: {input_path}")
    if summary.get("system") != record.get("system"):
        raise Refusal(
            "source_pin_mismatch",
            f"the summary verifies {summary.get('system')!r}, the record is "
            f"{record.get('system')!r}: this summary is not about this record")
    if Path(str(summary.get("record", ""))).name != record_path.name:
        raise Refusal(
            "source_pin_mismatch",
            f"the summary was written for {summary.get('record')!r}, not for "
            f"{record_path.name}")
    if (record.get("epsilon_exact") is not None
            and summary.get("epsilon_exact") is not None
            and Fraction(record["epsilon_exact"])
            != Fraction(summary["epsilon_exact"])):
        raise Refusal(
            "epsilon_ladder_mismatch",
            f"the summary proves at epsilon {summary['epsilon_exact']}, the "
            f"record is stated at {record['epsilon_exact']}")

    derived = Fraction(summary["eps_certified_exact"])
    reported = record.get("eps_certified")
    exceeds = margin_claim_exceeds(record, derived)
    sidecar = {
        "schema": VERIFIED_MARGIN_SCHEMA,
        "note": ("the portable verifier's own derived margin for a record "
                 "emitted with --no-self-verify. The record's bytes are NOT "
                 "modified: they are the bytes the verifier accepted, and "
                 "this file is what a claim about the achieved margin "
                 "quotes."),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "record": record_path.as_posix(),
        "record_sha256": sha256_of(record_path),
        "grams_file": record.get("grams_file"),
        "verify_summary": summary_path.as_posix(),
        "verify_summary_sha256": sha256_of(summary_path),
        "system": record.get("system"),
        "verdict": summary.get("verdict"),
        "proof_inputs": binding,
        "verifier": {
            "acceptance_rule": summary.get("acceptance_rule"),
            "margin_policy": summary.get("margin_policy"),
            "epsilon_exact": summary.get("epsilon_exact"),
            "eps_certified_exact": summary.get("eps_certified_exact"),
            "eps_certified_float": float(derived),
            "worst_charge_over_eps": summary.get("worst_charge_over_eps"),
            "note_half_margin_holds": summary.get("note_half_margin_holds"),
        },
        "record_reported": {
            "note": "what the record states, and where it says that came from",
            "eps_certified": reported,
            "eps_certified_exact": record.get("eps_certified_exact"),
            "eps_certified_exact_is_float_transcription": record.get(
                "eps_certified_exact_is_float_transcription"),
            "eps_certified_source": record.get("eps_certified_source"),
            "margin_policy": record.get("margin_policy"),
        },
        "margin_claim_exceeds_proof": bool(exceeds),
        "authoritative_eps_certified_exact": summary.get(
            "eps_certified_exact"),
    }
    stem = record_path.name
    for suffix in (".record.json.gz", ".json.gz", ".gz", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    target_dir = Path(out_dir) if out_dir else record_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = target_dir / f"{stem}.verified_margin.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=1) + "\n")
    print(json.dumps({
        "attached": True, "sidecar": str(sidecar_path),
        "system": record.get("system"),
        "eps_certified_exact": summary.get("eps_certified_exact"),
        "eps_certified_float": float(derived),
        "record_unmodified_sha256": sidecar["record_sha256"],
        "margin_claim_exceeds_proof": bool(exceeds),
    }, indent=1))
    if exceeds:
        print(f"MARGIN CLAIM EXCEEDS PROOF: the record states {reported!r}, "
              f"the proof gives {summary['eps_certified_exact']}; the record "
              f"must be re-emitted from a corrected source",
              file=sys.stderr)
        return 3
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--row", default=None,
                    help="certified row JSON (campaign systems/<name>.json for "
                         "a nominal row, *_delta<d>_row.json for a perturbed "
                         "one); required unless --exact-cell is used")
    ap.add_argument("--exact", default=None,
                    help="exact-phase artifact for that row; required unless "
                         "--exact-cell is used")
    ap.add_argument("--grams", default=None,
                    help="Gram bundle, when it is not reachable from --exact")
    ap.add_argument("--exact-cell", default=None,
                    help="per-corner exact-phase CELL (header.json, ROW.json, "
                         "corner_*.json) instead of a single --exact artifact; "
                         "needs --design-record and --design-bundle-npz")
    ap.add_argument("--design-record", default=None,
                    help="design-stage record holding candidate_V_terms, the "
                         "V source for an --exact-cell emission")
    ap.add_argument("--design-bundle-npz", default=None,
                    help="design Gram bundle (.npz) the exact phase proved on; "
                         "every entry hash is recomputed and pinned against "
                         "the corner records before anything is written")
    ap.add_argument("--share-w-grams", action="store_true",
                    help="write the nominal W witness ONCE under "
                         "shared_W_grams when every bracket corner names the "
                         "same Gram entries; refuses if any corner differs")
    ap.add_argument("--stem-tag", default=None,
                    help="tag inserted in the output file names, e.g. m11 for "
                         "levant5d_m11_nominal.record.json.gz; use it when one "
                         "system has records at more than one degree")
    ap.add_argument("--margin-policy", choices=("half", "achieved"),
                    default=None,
                    help="acceptance rule the exact phase ran under, for an "
                         "artifact written before the field existed; recorded "
                         "as an override in the record's provenance")
    ap.add_argument("--attach-verify-summary", default=None,
                    help="follow-up mode (WP R-08): given the --json summary "
                         "of a completed portable verification of an ALREADY "
                         "emitted --record, write the verifier's exact margin "
                         "into <stem>.verified_margin.json beside that "
                         "record, without modifying the verified bytes")
    ap.add_argument("--record", default=None,
                    help="the emitted record an --attach-verify-summary "
                         "belongs to")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for the record and its grams "
                         "sidecar; required except in "
                         "--attach-verify-summary mode, where it overrides "
                         "the record's own directory")
    ap.add_argument("--gain-point",
                    choices=("auto", "benchmark", "flagship", "exact-record"),
                    default="auto",
                    help="which registered gain point the row was certified "
                         "at; exact-record reads the gain point (gains, "
                         "brackets, box, delta0, corners) from the --exact "
                         "record of scripts/certify_gain_point.py instead of "
                         "the registry, and re-checks it in Q")
    ap.add_argument("--corner-subset", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="exact-record only: fix a BOX gain at one of its "
                         "edges and emit only the corners with that value "
                         "(repeatable); e.g. lambda1=2")
    ap.add_argument("--bracket-target", action="append", default=[],
                    metavar="NAME:K:Q",
                    help="exact-record only: write the box edge of NAME as a "
                         "bracket of the root of g^K = Q, checked as "
                         "lo^K < Q < hi^K in Q (default: K = 1, Q = the "
                         "interval midpoint); e.g. lambda2:2:9/2")
    ap.add_argument("--representative-p", default=None,
                    help="rational covariant representative p > 0 (WP TA): the "
                         "cell was proved at the flagship gains scaled by "
                         "p^i (brackets by p^i, power targets by p^(i k)); "
                         "needs --exact-cell, and must equal the cell's "
                         "qscale_p")
    ap.add_argument("--campaign-block", default=None,
                    help="campaign block that produced the source run, e.g. E1_matrix")
    ap.add_argument("--env", default=None,
                    help="env.json of the source run (provenance, embedded verbatim)")
    ap.add_argument("--epsilon", type=float, default=1e-6,
                    help="margin, only used when the artifact does not record it")
    ap.add_argument("--no-self-verify", action="store_true",
                    help="skip the standalone re-verification of the emitted "
                         "record (only for records too large to replay here)")
    args = ap.parse_args()

    if args.attach_verify_summary:
        if not args.record:
            ap.error("--attach-verify-summary needs --record")
        if args.row or args.exact or args.exact_cell:
            ap.error("--attach-verify-summary attaches to an existing record; "
                     "it emits nothing and takes no source artifacts")
        try:
            return attach_verify_summary(args.record,
                                         args.attach_verify_summary,
                                         args.out_dir, args.grams)
        except Refusal as refusal:
            print(json.dumps(refusal.as_dict(), indent=1))
            print(f"REFUSED ({refusal.reason}): nothing attached",
                  file=sys.stderr)
            return 3
    if not args.out_dir:
        ap.error("--out-dir is required")
    if args.record:
        ap.error("--record belongs to --attach-verify-summary")

    if args.exact_cell:
        if not (args.design_record and args.design_bundle_npz):
            ap.error("--exact-cell needs --design-record and "
                     "--design-bundle-npz")
        if args.row or args.exact:
            ap.error("--exact-cell replaces --row and --exact")
    elif not (args.row and args.exact):
        ap.error("--row and --exact are required without --exact-cell")
    exact_record_mode = args.gain_point == "exact-record"
    if exact_record_mode:
        if args.exact_cell:
            ap.error("--gain-point exact-record reads the gain point from an "
                     "--exact record of scripts/certify_gain_point.py; it "
                     "does not take --exact-cell")
        if args.representative_p is not None:
            ap.error("--gain-point exact-record carries its gains as proved; "
                     "it cannot be combined with --representative-p")
    elif args.corner_subset or args.bracket_target:
        ap.error("--corner-subset and --bracket-target need "
                 "--gain-point exact-record")
    if args.representative_p is not None:
        if not args.exact_cell:
            ap.error("--representative-p needs --exact-cell")
        if args.gain_point == "benchmark":
            ap.error("--representative-p scales the flagship gain point; it "
                     "cannot be combined with --gain-point benchmark")
        try:
            args.representative_p = Fraction(args.representative_p)
        except (ValueError, ZeroDivisionError):
            ap.error(f"--representative-p {args.representative_p!r} is not "
                     f"a rational number")
        if args.representative_p <= 0:
            ap.error("--representative-p must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = None
    row_path = Path(args.row) if args.row else Path(args.design_record)
    exact_path = Path(args.exact) if args.exact else Path(args.exact_cell)

    try:
        if args.exact_cell:
            cell = CellSource(args)
            row = cell.row()
            # ``row_source`` names where V was transcribed from.
            args.row = str(cell.design_record_path)
        else:
            row = load_row(row_path)
        system = row["system"]

        # The hand-written builder table of the standalone verifier is the
        # authority on which systems can be recorded at all.
        sys.path.insert(0, str(STANDALONE.parent))
        from verify_certificate import SYSTEMS as STANDALONE_SYSTEMS
        if system not in STANDALONE_SYSTEMS:
            raise Refusal(
                "unknown_system",
                f"{system}: the standalone verifier has no hand-written "
                f"dynamics builder for it, so no portable record can be "
                f"checked without the framework",
                [f"add a builder for {system} to "
                 f"verification/standalone/verify_certificate.py and pin it "
                 f"term for term in tests/test_standalone_verifier.py"])
        cfg = STANDALONE_SYSTEMS[system]
        n_vars = len(cfg["weights_r"])
        args.system_family = cfg["family"]
        if list(cfg["weights_r"]) != list(row["weights_r"]):
            raise Refusal(
                "gain_mismatch",
                f"{system}: the verifier's hand-written equations use weights "
                f"{list(cfg['weights_r'])}, the row was certified at "
                f"{list(row['weights_r'])}")

        representative_p = args.representative_p
        if exact_record_mode:
            # WP TB emitter change (option a): the gain point is the exact
            # record's own, re-checked in Q against the verifier's equations.
            exact_doc = read_json(exact_path)
            point = ExactRecordGainPoint(
                exact_doc, system, cfg,
                corner_subset=parse_corner_subset(args.corner_subset),
                bracket_targets=parse_bracket_targets(args.bracket_target))
            point.check_row(row, exact_doc)
        else:
            flagship = (args.gain_point == "flagship" or
                        representative_p is not None or
                        (args.gain_point == "auto" and not is_nominal(row)
                         and system in FLAGSHIP_GAIN_POINTS))
            if args.gain_point == "benchmark" \
                    and system not in BENCHMARK_GAIN_POINTS:
                raise Refusal("unknown_system",
                              f"{system}: no benchmark gain point registered")
            if flagship and system not in FLAGSHIP_GAIN_POINTS:
                raise Refusal("unknown_system",
                              f"{system}: no flagship gain point registered")
            point = gain_point(system, flagship=flagship)
            if representative_p is not None:
                # WP TA: the flagship point scaled to the representative p.
                point = ScaledGainPoint(point, representative_p)
            # C2 (2026-09-28): every bracketed gain of a registry point
            # gets the exact-record route's guard R6 declaration and the
            # exact probe of the verifier's own builder, before any record
            # file is written (refusal: unverifiable_gain_point).
            check_registry_point_affinity(point, system)
            check_row_gains(row, point, system)
            exact_doc = cell.exact_doc() if cell else read_json(exact_path)
        entry = exact_entry(exact_doc, system)
        check_gain_binding(exact_doc, entry, system)
        for verdict in ("exact_pass", "exact_pass_all"):
            if entry.get(verdict) is False or exact_doc.get(verdict) is False:
                raise Refusal("not_certified",
                              f"{system}: the exact phase did not pass "
                              f"({verdict} is false)")
        if cell:
            epsilon = float(exact_doc["epsilon"])
            bundle = cell.bundle(point, epsilon)
            sources = cell.source_artifacts()
        else:
            bundle, grams_source = find_grams(exact_doc, exact_path, system,
                                              args.grams)
            if exact_record_mode:
                bundle = point.select_bundle(bundle)
            sources = [("row", row_path), ("exact_phase", exact_path),
                       ("grams", grams_source)]

        record, grams, margin = build_record(row, exact_doc, entry, bundle,
                                             point, args, n_vars)
        # WP R-05 (c): every input this transcription read, by path and by
        # content hash, plus the registry the gain point came out of. POSIX
        # paths: a record is read on machines that did not produce it. An
        # exact-record gain point has no registry source: the exact record
        # (role exact_phase) is where it came from, and it is hashed above.
        record["source_artifacts"] = [
            {"role": role, "path": Path(path).as_posix(),
             "sha256": sha256_of(path)} for role, path in sources
        ] + ([] if exact_record_mode
             else [registry_source_artifact(base_point(point))])
        record["env"] = read_json(args.env) if args.env else None
        # WP R-05 (d): free text and the source stage's own reported numbers,
        # in one place, labelled. The verifier ignores all of it, and nothing
        # in it may be quoted as a result.
        record["provenance"] = {
            "note": ("unverified provenance, not part of the proved "
                     "statement; the portable verifier ignores every key in "
                     "this object"),
            "emitter": "scripts/emit_certificate_record.py",
            # The options that shape the CONTENT, not the command line: an
            # output directory in the record would make two emissions of one
            # certificate differ, and the determinism audit reads that as a
            # certificate difference.
            "emitter_options": {
                "gain_point": args.gain_point,
                "share_w_grams": bool(args.share_w_grams),
                "margin_policy": args.margin_policy,
                "source_shape": "exact_cell" if cell else "row_and_exact",
            },
            "margin_policy_override": args.margin_policy,
            # WP R-08: the source stage's OWN reported achieved margin, in
            # one labelled place. It is never the number the record states
            # once the portable verifier has derived its own, and it is
            # refused outright when it is larger than that derivation.
            "source_reported_eps_certified": {
                "note": ("the source stage's reported achieved margin, "
                         "UNVERIFIED by the portable stage; the record's own "
                         "eps_certified is the verifier's derivation whenever "
                         "self-verification ran"),
                "value": margin["source_reported_eps_certified"],
                "exact": margin["source_reported_eps_certified_exact"],
                "exact_is_float_transcription":
                    margin["eps_certified_exact_is_float_transcription"],
            },
            "shared_W_grams_requested": bool(args.share_w_grams),
            "source_claim": (cell.row_doc.get("claim") if cell
                             else exact_doc.get("claim")),
            "exact_stage": {
                "note": ("the exact phase's OWN reported numbers, kept for "
                         "audit; the portable verifier derives its own"),
                "worst_charge_ratio": (
                    cell.row_doc.get("worst_charge_ratio_over_corners")
                    if cell else entry.get("worst_charge_ratio")),
                "eps_certified": (
                    cell.row_doc.get("eps_certified_over_corners") if cell
                    else (entry.get("witness") or {}).get("eps_certified")),
                "v_input_hash": cell.header.get("v_input_hash") if cell
                                else None,
                "gram_bundle_sha256": getattr(cell, "bundle_sha256", None),
                "gram_bundle_fingerprint": getattr(
                    cell, "bundle_fingerprint", None),
            },
        }
        if isinstance(point, ScaledGainPoint):
            # WP TA: p, the covariant map and the unscaled base point. The
            # record's own gains, brackets and delta0 are the scaled ones.
            # Added only for a scaled emission, so every other record keeps
            # its bytes (determinism audit).
            record["provenance"]["emitter_options"]["representative_p"] = \
                str(point.p)
            record["provenance"]["representative_p"] = point.provenance(
                Fraction(record["delta0_exact"]))
        if isinstance(point, ExactRecordGainPoint):
            # WP TB emitter change: where the gain point came from. Added only
            # for this mode, so every other record keeps its bytes.
            options = record["provenance"]["emitter_options"]
            options["corner_subset"] = {
                n: str(v) for n, v in sorted(point.corner_subset.items())}
            options["bracket_targets"] = {
                n: f"{k}:{q}"
                for n, (k, q) in sorted(point.bracket_targets.items())}
            record["provenance"]["gain_point_source"] = point.provenance(
                exact_path, exact_doc)

        # WP R-05 (b): a record may not advertise a margin nobody derived.
        # The exact phase records one, or the portable verifier derives one
        # below; with neither, there is nothing to publish.
        if record["eps_certified"] is None and args.no_self_verify:
            raise Refusal(
                "missing_margin_policy",
                f"{system}: the exact phase records no eps_certified and "
                f"self-verification is skipped, so the margin this record "
                f"proves would be unknown",
                ["drop --no-self-verify, so the portable verifier derives "
                 "eps_certified itself, or emit from an exact artifact that "
                 "records it"])

        tag = f"{args.stem_tag}_" if args.stem_tag else ""
        if isinstance(point, ExactRecordGainPoint):
            tag += point.stem_suffix()
        stem = f"{system}_{tag}{'nominal' if record['nominal'] else 'delta' + record['delta0_exact'].replace('/', '_')}"
        grams_path = out_dir / f"{stem}.grams.json.gz"
        rec_path = out_dir / f"{stem}.record.json.gz"
        record["grams_file"] = grams_path.name
        try:
            n_grams = write_json_gz(grams_path, grams)
        except Refusal:
            # A capture that cannot be encoded is found while streaming, with
            # a partial sidecar already on disk. It must not survive.
            grams_path.unlink(missing_ok=True)
            raise
        n_rec = write_json_gz(rec_path, record)

        if not args.no_self_verify:
            summary_path = out_dir / f"{stem}.verify_summary.json"
            try:
                self_verify(rec_path, summary_path)
            except Refusal:
                # The files exist by now. A rejected record must not survive on
                # disk, or the next audit would find something that looks like
                # an emitted record and is not one.
                rec_path.unlink(missing_ok=True)
                grams_path.unlink(missing_ok=True)
                summary_path.unlink(missing_ok=True)
                raise
            summary = read_json(summary_path)
            # WP R-08 (follow-up review 2026-09-05 B): the ONE authority on
            # the margin a self-verified record states is the proof that just
            # ran. The source stage's report is compared against it and then
            # kept only as provenance; it is never the published number, and
            # a report larger than the proof is a refusal, because a producer
            # that advertises more than it proved is broken.
            try:
                derived_exact = Fraction(summary["eps_certified_exact"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                raise Refusal(
                    "self_verification_failed",
                    f"{system}: the portable verifier's summary carries no "
                    f"usable eps_certified_exact "
                    f"({summary.get('eps_certified_exact')!r}), so the "
                    f"margin this record would state is underived",
                    ["re-run verification/standalone/verify_certificate.py "
                     "with --json and inspect the summary"])
            reported = margin["source_reported_eps_certified"]
            if reported is not None and Fraction(reported) > derived_exact \
                    and float(reported) != float(derived_exact):
                # Not a rounding question: the float64 image of the derived
                # margin is accepted as a transcription of it, and anything
                # above that is a claim the proof does not carry.
                rec_path.unlink(missing_ok=True)
                grams_path.unlink(missing_ok=True)
                summary_path.unlink(missing_ok=True)
                raise Refusal(
                    "margin_claim_exceeds_proof",
                    f"{system}: the source stage reports eps_certified "
                    f"{reported!r}, the portable re-proof of this very record "
                    f"derives {summary['eps_certified_exact']} "
                    f"(~{float(derived_exact):.6g}); the source claims more "
                    f"than it proved and the emitter does not correct it "
                    f"silently",
                    ["fix the producer of the exact-phase artifact so its "
                     "reported eps_certified is the margin its own captures "
                     "prove, then re-run the exact phase and re-emit"])
            # Taking the verifier's numbers means the record changed after it
            # was verified, so it is verified again below; the bytes that
            # ship are always bytes the portable verifier accepted.
            record["eps_certified"] = float(derived_exact)
            record["eps_certified_exact"] = summary["eps_certified_exact"]
            record["eps_certified_exact_is_float_transcription"] = False
            record["eps_certified_source"] = "portable_verifier"
            record["note_half_margin_holds"] = (
                summary.get("note_half_margin_holds") or {}).get("overall")
            record["worst_charge_over_eps"] = summary.get(
                "worst_charge_over_eps")
            record["verifier"] = {
                "acceptance_rule": summary.get("acceptance_rule"),
                "eps_certified_exact": summary.get("eps_certified_exact"),
                "worst_charge_over_eps": summary.get(
                    "worst_charge_over_eps"),
            }
            n_rec = write_json_gz(rec_path, record)
            try:
                self_verify(rec_path, summary_path)
            except Refusal:
                rec_path.unlink(missing_ok=True)
                grams_path.unlink(missing_ok=True)
                summary_path.unlink(missing_ok=True)
                raise

        print(json.dumps({
            "emitted": True, "system": system, "schema": SCHEMA_VERSION,
            "nominal": record["nominal"], "record": str(rec_path),
            "grams": str(grams_path),
            "self_verified": not args.no_self_verify,
            "bytes_record_gz": rec_path.stat().st_size,
            "bytes_grams_gz": grams_path.stat().st_size,
            "bytes_uncompressed": n_rec + n_grams,
        }, indent=1))
        return 0
    except Refusal as refusal:
        payload = refusal.as_dict()
        payload["system"] = locals().get("system")
        payload["row"] = str(row_path)
        payload["exact"] = str(exact_path)
        payload["timestamp_utc"] = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
        name = f"{payload.get('system') or row_path.stem}.refusal.json"
        (out_dir / name).write_text(json.dumps(payload, indent=1))
        print(json.dumps(payload, indent=1))
        print(f"REFUSED ({refusal.reason}): no record emitted", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
