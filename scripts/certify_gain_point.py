#!/usr/bin/env python3
"""Generic gain-point driver: D -> X -> record -> EM (WP TB).

One command runs the citability chain for ONE controller at ONE gain point,
gain bracket or gain box, nominal or perturbed:

  D   joint V+W design over the float dynamics of EVERY corner x inclusion
      endpoint, passed as one vertex list to
      ``JointSOSOptimizer.find_lyapunov_with_dynamics`` (the W block is
      duplicated per vertex, so one V decreases along all of them). With
      ``--design-solver MOSEK_DIRECT`` (the default) the design runs strict,
      with no solver fallback, and its Gram bundle is persisted next to the
      run (``design_bundle_m<m>.npz``). That is the D of the chain.
  X   ONE ``run_exact_phase`` call over the EXACT dynamics of every corner x
      endpoint, in the same order as the design vertex list, witnessed by D's
      persisted bundle. Before X runs, the bundle's W-group count must equal
      the dynamics count (the same invariant ``run_exact_phase`` enforces,
      checked here first so the refusal is typed in this run's record).
      Every witness attempt, including a fallback re-solve, is recorded.
  rec the exactgains shape of ``scripts/exact_gate_standard_gains.py``
      (``exact.json`` + ``exact_grams.json.gz`` + ``row.json``), then
  EM  ``scripts/emit_certificate_record.py`` as a subprocess, which
      transcribes, self-verifies with the portable verifier and writes the
      portable record. When the CLI gain point IS a registered point
      (flagship or benchmark table, compared exactly, brackets included) EM
      binds the gains to the printed-gain registry (``--gain-point
      flagship|benchmark``). Any other point (a box, a scaled point, a new
      tuning) is emitted with ``--gain-point exact-record``: EM reads the
      gain point from ``exact.json`` itself, re-checks it in Q against the
      portable verifier (gain table, fixed parameters, brackets, box edges,
      affinity of every varying gain, disturbance channel) and records that
      the gain point came from the exact record, not the registry. Until
      2026-09-24 such a point stopped here with exit code 5.

Two-point (union) mode (2026-09-25)
-----------------------------------
``--point SPEC`` (repeatable) names several gain points, each with its own
fixed gains, brackets or box; ``--gains``/``--bracket``/``--box`` and
``--delta0`` are shared by all of them. Every point is checked on its own
(verifier gain table, brackets in Q, guard R6 for each varying gain, exact
dynamics). D designs ONE V over the corner x endpoint dynamics of ALL points
(one bundle, one W group per dynamics), X runs once over all of them, and the
records are then written per point (``m<m>/point<k>/``): each carries its own
gains, brackets and corners, the shared V captures and only the W captures of
its own dynamics, and is emitted and verified on its own (stem tag suffix
``_pt<k>``). The union of points is not a box: no statement is made about
gains between the points.

What is and is not D
--------------------
Only a strict MOSEK_DIRECT design with a persisted Gram bundle is D (NEXT_STEPS
Conventions, "Citability chain"). ``--design-solver CLARABEL`` exists for tests
and smoke runs on machines without MOSEK: the CVXPY design captures no Gram,
X then re-solves its witness, and every output of such a run is labelled
``design_stage.label = "clarabel_design_not_D"`` and ``citable_chain = false``.
``--v-row`` (fixed-V mode) skips design altogether and is labelled
``fixed_v_row_not_D``.

Soundness boundaries this driver does not move
----------------------------------------------
* The verdict is the exact phase's, never a solver status; the proved margin
  quoted anywhere is ``eps_certified`` of the exact phase and, once emitted,
  the portable verifier's own derivation.
* Guard R6: a gain that varies (``--bracket``, ``--box``) must be an affine
  gain coordinate of the system. The symbolic assertion of
  ``scripts/gain_region_fixedV.py::assert_gain_affinity`` is reused verbatim
  and runs before any solve. A fixed factory parameter (pidsmc ``delta``) is
  pinned to the value the portable verifier's equations describe.
* Brackets are validated in Q (``lo^k < q < hi^k``) before anything runs.
* Nothing is written into ``artifacts/ablation/**`` without
  ``--allow-frozen-root``.

Exit codes: 0 certified and emitted (or ``--no-emit``); 1 no degree passed
X; 3 EM refused or P failed; 4 input refusal (typed, before or instead of any
solve); 5 retired (was: emission blocked for a non-registered gain point);
6 unexpected error.

Examples
--------
    # MOSEK machine: D -> X -> P for the perturbed twisting controller
    python scripts/certify_gain_point.py --system twisting \\
        --gains r1=2,r2=3/2 --delta0 127/256 --degrees 5 6 7 \\
        --out-dir artifacts/campaign_2026-09/X4/delta127_256

    # MOSEK machine: ONE V for both standard tunings of the second-order
    # differentiator (two narrow lambda2 brackets), one record per tuning
    python scripts/certify_gain_point.py --system levant3d --gains lambda3=11/10 \\
        --point 'lambda1=2;bracket=lambda2:2121/1000:2122/1000:2:9/2' \\
        --point 'lambda1=3;bracket=lambda2:2598/1000:2599/1000:2:27/4' \\
        --delta0 1 --degrees 8 9 10 --design-epsilon 1e-6 --out-dir <dir>

    # MOSEK-free smoke (NOT D): STA at the standard tuning, delta0 = 1
    python scripts/certify_gain_point.py --system sta \\
        --gains k1=3/2,k2=11/10 --delta0 1 --degrees 6 --sparsify rl1 \\
        --design-solver CLARABEL --out-dir <dir>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gf_framework.systems import printed_gains as pg  # noqa: E402

EM_SCRIPT = REPO_ROOT / "scripts" / "emit_certificate_record.py"
VC_SCRIPT = REPO_ROOT / "verification" / "standalone" / "verify_certificate.py"

DESIGN_SOLVERS = ("MOSEK_DIRECT", "CLARABEL")
DEFAULT_DESIGN_SOLVER = "MOSEK_DIRECT"
DEFAULT_DESIGN_EPSILON = 1e-4

EXIT_OK = 0
EXIT_NOT_CERTIFIED = 1
EXIT_EM_REFUSED = 3
EXIT_REFUSED = 4
#: Retired 2026-09-24: a non-registered gain point is now emitted through
#: EM's ``--gain-point exact-record`` mode. Kept so the code is never reused
#: with another meaning.
EXIT_EMISSION_BLOCKED = 5
EXIT_ERROR = 6

#: EM's gain-point route for a point that is not in the printed-gain registry.
EXACT_RECORD_ROUTE = "exact-record"

#: Stage labels written into every output. Only ``D`` is the D of the chain.
STAGE_D = "D"
STAGE_CLARABEL = "clarabel_design_not_D"
STAGE_FIXED_V = "fixed_v_row_not_D"

#: Systems whose factory is not in the printed-gain registry.
FACTORY_FALLBACK = {
    "pidsmc4d": ("create_pidsmc4d_spec", "gf_framework.systems.pidsmc4d"),
}

#: Guard R6 coordinate aliases: the portable verifier names pidsmc's gains
#: lambda1..3, ``gain_region_fixedV`` declares the affine coordinates k1..k3.
#: They coincide exactly at delta = 1, the only case the verifier's pidsmc
#: equations describe (and which this driver enforces through ``fixed``).
R6_ALIASES = {
    "pidsmc": {"lambda1": "k1", "lambda2": "k2", "lambda3": "k3"},
}


class TBRefusal(Exception):
    """A typed refusal: the run cannot be performed as requested."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail

    def as_dict(self) -> dict:
        return {"refused": True, "kind": self.kind, "detail": self.detail}


# ---------------------------------------------------------------------------
# Gain specification
# ---------------------------------------------------------------------------

def parse_rational(text: str, what: str) -> Fraction:
    """An exact rational from ``p/q``, an integer or a finite decimal string."""
    try:
        value = Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise TBRefusal("bad_rational", f"{what}: {text!r} is not a rational "
                                        f"number ({exc})") from exc
    return value


@dataclass(frozen=True)
class Bracket:
    """A rational interval [lo, hi] with its algebraic pin lo^k < q < hi^k."""
    lo: Fraction
    hi: Fraction
    power_k: int
    power_target: Fraction

    def check(self, name: str) -> None:
        if self.power_k < 1:
            raise TBRefusal("bad_bracket", f"{name}: power_k must be >= 1 "
                                           f"(got {self.power_k})")
        if not (0 < self.lo < self.hi):
            raise TBRefusal("bad_bracket", f"{name}: need 0 < lo < hi, got "
                                           f"[{self.lo}, {self.hi}]")
        k = self.power_k
        if not (self.lo ** k < self.power_target < self.hi ** k):
            raise TBRefusal(
                "bad_bracket",
                f"{name}: [{self.lo}, {self.hi}] does not bracket the root of "
                f"g^{k} = {self.power_target} (lo^k < q < hi^k fails in Q)")

    def record(self) -> dict:
        return {"lo": str(self.lo), "hi": str(self.hi),
                "power_k": int(self.power_k),
                "power_target": str(self.power_target)}


@dataclass
class GainSpec:
    """The gain point of one run, all values exact."""
    system: str
    rational: Dict[str, Fraction] = field(default_factory=dict)
    brackets: Dict[str, Bracket] = field(default_factory=dict)
    box: Dict[str, Tuple[Fraction, Fraction]] = field(default_factory=dict)
    delta0: Fraction = Fraction(0)

    @property
    def varying(self) -> List[str]:
        return sorted(set(self.brackets) | set(self.box))

    @property
    def names(self) -> List[str]:
        return sorted(set(self.rational) | set(self.brackets) | set(self.box))

    @property
    def perturbed(self) -> bool:
        return self.delta0 > 0

    def interval(self, name: str) -> Tuple[Fraction, Fraction]:
        if name in self.brackets:
            b = self.brackets[name]
            return b.lo, b.hi
        return self.box[name]

    def corners(self) -> List[Dict[str, Fraction]]:
        """All 2^k vertices, sorted names, (lo, hi) per name.

        This is the registry's corner order (``printed_gains.corner_assignments``)
        whenever the varying gains are brackets, which is what lets EM pair
        its registry corners with these positionally.
        """
        names = self.varying
        if not names:
            return [{}]
        return [dict(zip(names, combo)) for combo in
                itertools.product(*(self.interval(n) for n in names))]

    def corner_label(self, assignment: Dict[str, Fraction]) -> str:
        if not assignment:
            return "rational gains only"
        parts = []
        for name in sorted(assignment):
            lo, hi = self.interval(name)
            parts.append(f"{name}={'lo' if assignment[name] == lo else 'hi'}")
        return ", ".join(parts)

    def values_at(self, assignment: Dict[str, Fraction]) -> Dict[str, Fraction]:
        values = dict(self.rational)
        values.update(assignment)
        return values

    def record(self) -> dict:
        return {
            "system": self.system,
            "rational_gains": {n: str(v) for n, v in sorted(self.rational.items())},
            "brackets": {n: b.record() for n, b in sorted(self.brackets.items())},
            "box": {n: {"lo": str(lo), "hi": str(hi)}
                    for n, (lo, hi) in sorted(self.box.items())},
            "delta0_exact": str(self.delta0),
            "n_corners": len(self.corners()),
            "inclusion_endpoints": 2 if self.perturbed else 1,
        }


def _split_assignments(values: Sequence[str]) -> List[str]:
    out = []
    for chunk in values or []:
        out.extend(p for p in (s.strip() for s in chunk.split(",")) if p)
    return out


def build_gain_spec(system: str, gains: Sequence[str] = (),
                    brackets: Sequence[str] = (), boxes: Sequence[str] = (),
                    delta0: str = "0", preset: Optional[str] = None) -> GainSpec:
    """Parse and validate the CLI gain point (no solve, no dynamics)."""
    spec = GainSpec(system=system,
                    delta0=parse_rational(delta0, "--delta0"))
    if spec.delta0 < 0:
        raise TBRefusal("bad_delta0", f"delta0 must be >= 0 (got {spec.delta0})")

    if preset:
        table = (pg.FLAGSHIP_GAIN_POINTS if preset == "flagship"
                 else pg.BENCHMARK_GAIN_POINTS)
        point = table.get(system)
        if point is None:
            raise TBRefusal("unknown_preset",
                            f"no {preset} gain point is registered for {system}")
        reg_br = pg.brackets(point)
        for name, g in point.gains.items():
            if g.is_bracketed:
                lo, hi = reg_br[name]
                spec.brackets[name] = Bracket(lo, hi, int(g.power_k),
                                              Fraction(g.power_target))
            else:
                spec.rational[name] = g.value

    def _claim(name: str) -> None:
        spec.rational.pop(name, None)
        spec.brackets.pop(name, None)
        spec.box.pop(name, None)

    seen: Dict[str, str] = {}

    def _once(name: str, flag: str) -> None:
        if name in seen:
            raise TBRefusal("duplicate_gain",
                            f"{name} given twice ({seen[name]} and {flag})")
        seen[name] = flag

    for item in _split_assignments(gains):
        if "=" not in item:
            raise TBRefusal("bad_gain", f"--gains entry {item!r} is not name=RATIONAL")
        name, value = (s.strip() for s in item.split("=", 1))
        _once(name, "--gains")
        _claim(name)
        spec.rational[name] = parse_rational(value, f"gain {name}")
    for item in brackets or []:
        parts = item.split(":")
        if len(parts) != 5:
            raise TBRefusal("bad_bracket",
                            f"--bracket {item!r} is not name:lo:hi:k:q")
        name = parts[0].strip()
        _once(name, "--bracket")
        try:
            k = int(parts[3])
        except ValueError as exc:
            raise TBRefusal("bad_bracket", f"{name}: power k {parts[3]!r} is "
                                           f"not an integer") from exc
        b = Bracket(parse_rational(parts[1], f"bracket {name} lo"),
                    parse_rational(parts[2], f"bracket {name} hi"), k,
                    parse_rational(parts[4], f"bracket {name} q"))
        b.check(name)
        _claim(name)
        spec.brackets[name] = b
    for item in boxes or []:
        parts = item.split(":")
        if len(parts) != 3:
            raise TBRefusal("bad_box", f"--box {item!r} is not name:lo:hi")
        name = parts[0].strip()
        _once(name, "--box")
        lo = parse_rational(parts[1], f"box {name} lo")
        hi = parse_rational(parts[2], f"box {name} hi")
        if not (0 < lo < hi):
            raise TBRefusal("bad_box", f"{name}: need 0 < lo < hi, got [{lo}, {hi}]")
        _claim(name)
        spec.box[name] = (lo, hi)
    for name, value in spec.rational.items():
        if value <= 0 and not _is_fixed_param(system, name):
            raise TBRefusal("bad_gain", f"{name} = {value}: gains must be positive")
    return spec


# ---------------------------------------------------------------------------
# System structure: the portable verifier's table is the authority
# ---------------------------------------------------------------------------

def vc_systems() -> dict:
    """The stdlib verifier's hand-written system table (import, read-only)."""
    sys.path.insert(0, str(VC_SCRIPT.parent))
    from verify_certificate import SYSTEMS  # noqa: PLC0415
    return SYSTEMS


def _is_fixed_param(system: str, name: str) -> bool:
    cfg = vc_systems().get(system) or {}
    return name in (cfg.get("fixed") or {})


def registry_template(system: str) -> Tuple[str, str, Dict[str, str]]:
    """(factory, module, roles) of a system, from the registry when possible."""
    for table in (pg.BENCHMARK_GAIN_POINTS, pg.FLAGSHIP_GAIN_POINTS):
        point = table.get(system)
        if point is not None:
            return (point.factory, point.module,
                    {n: g.role for n, g in point.gains.items()})
    if system in FACTORY_FALLBACK:
        factory, module = FACTORY_FALLBACK[system]
        return factory, module, {}
    raise TBRefusal("unknown_system",
                    f"{system}: no create_*_spec factory is known for it")


def check_gain_names(spec: GainSpec) -> dict:
    """Every verifier gain present, nothing unregistered, fixed params pinned.

    Adds a missing fixed parameter (pidsmc ``delta``) at its pinned value, so
    the record and the verifier describe the same equations.
    """
    table = vc_systems()
    if spec.system not in table:
        raise TBRefusal("unknown_system",
                        f"{spec.system}: the portable verifier has no builder "
                        f"for it (known: {sorted(table)})")
    cfg = table[spec.system]
    fixed = {n: Fraction(v) for n, v in (cfg.get("fixed") or {}).items()}
    names = set(spec.names)
    missing = [g for g in cfg["gain_names"] if g not in names]
    if missing:
        raise TBRefusal("missing_gain",
                        f"{spec.system}: gains {missing} are not given; every "
                        f"gain of the verifier's equations must be pinned")
    extra = sorted(names - set(cfg["gain_names"]) - set(fixed))
    if extra:
        raise TBRefusal("unregistered_gain",
                        f"{spec.system}: {extra} are not parameters of the "
                        f"verifier's equations {cfg['gain_names']}")
    for name, value in fixed.items():
        if name in spec.brackets or name in spec.box:
            raise TBRefusal(
                "non_affine_parameter",
                f"{spec.system}.{name} is a factory scaling parameter, not an "
                f"affine gain coordinate (guard R6); it cannot vary")
        given = spec.rational.get(name)
        if given is None:
            spec.rational[name] = value
        elif given != value:
            raise TBRefusal(
                "non_affine_parameter",
                f"{spec.system}.{name} = {given}: the verifier's equations "
                f"describe {name} = {value} only, and {name} scales the gains "
                f"non-affinely (guard R6)")
    if spec.perturbed and cfg.get("channel") is None:
        raise TBRefusal("no_disturbance_channel",
                        f"{spec.system}: no weighted-degree-0 channel admits a "
                        f"constant matched disturbance; delta0 must be 0")
    return cfg


def assert_affinity(spec: GainSpec) -> dict:
    """Guard R6, reused from ``scripts/gain_region_fixedV.py``.

    Only a gain that VARIES needs it: with a single corner the multi-affine
    box argument is not invoked. A varying gain must be a declared affine
    coordinate, and the symbolic assertion must pass for the whole system.
    """
    if not spec.varying:
        return {"required": False,
                "reason": "single gain point: no bracket or box is covered by "
                          "the vertex argument, so gain affinity is not used"}
    import gain_region_fixedV as grf  # noqa: PLC0415 (read-only reuse)

    decl = grf.SYSTEMS.get(spec.system)
    if decl is None:
        raise TBRefusal(
            "non_affine_parameter",
            f"{spec.system}: no affine gain coordinates are declared in "
            f"scripts/gain_region_fixedV.py, so a bracket or box cannot be "
            f"covered by its vertices (guard R6)")
    alias = R6_ALIASES.get(spec.system, {})
    for name in spec.varying:
        coord = alias.get(name, name)
        if coord not in decl.coords:
            raise TBRefusal(
                "non_affine_parameter",
                f"{spec.system}.{name} is not a declared affine gain "
                f"coordinate {list(decl.coords)} (guard R6)")
    try:
        report = grf.assert_gain_affinity(decl, float(spec.delta0))
    except grf.GainRegionRefusal as exc:
        raise TBRefusal("non_affine_parameter",
                        f"guard R6 refused: {exc.message}") from exc
    report = dict(report)
    report["required"] = True
    report["varying"] = list(spec.varying)
    report["aliases"] = {n: alias.get(n, n) for n in spec.varying}
    report["source"] = "scripts/gain_region_fixedV.py::assert_gain_affinity"
    return report


# ---------------------------------------------------------------------------
# Dynamics: exact per corner x endpoint, and the float design image
# ---------------------------------------------------------------------------

@dataclass
class Vertices:
    labels: List[str]
    corner_index: List[int]
    exact: List[list]
    floats: List[list]
    float_nominal: list
    weights_r: Tuple[int, ...]
    channel: Optional[int]
    corners: List[Dict[str, Fraction]]
    repairs: List[dict]


def _cli_point(spec: GainSpec, values: Dict[str, Fraction], channel):
    factory, module, roles = registry_template(spec.system)
    gains = {n: pg.PrintedGain(Fraction(v), "certify_gain_point CLI value",
                               role=roles.get(n, "gain"))
             for n, v in values.items()}
    return pg.GainPoint(system=spec.system, factory=factory, module=module,
                        gains=gains, channel=channel,
                        note="WP TB command-line gain point")


def build_vertices(spec: GainSpec, cfg: dict) -> Vertices:
    """Exact and float dynamics for every corner x inclusion endpoint.

    Order: corner-major, endpoint (+delta0, -delta0) minor; one dynamics per
    corner when nominal. The design vertex list and the exact-phase dynamics
    list use this SAME order, so W group i of D's bundle witnesses dynamics i.
    """
    from gf_framework.discovery import universal_discovery as ud  # noqa: PLC0415

    channel = cfg.get("channel")
    corners = spec.corners()
    labels, idx, exact, floats, repairs = [], [], [], [], []
    float_nominal = None
    weights = None
    factory = getattr(ud, registry_template(spec.system)[0])
    for ci, assignment in enumerate(corners):
        values = spec.values_at(assignment)
        try:
            ed = pg.exact_dynamics(_cli_point(spec, values, channel), {})
        except (pg.PrintedGainError, TypeError) as exc:
            raise TBRefusal("dynamics_not_exact",
                            f"corner {ci}: the dynamics cannot be built in "
                            f"exact arithmetic at {values}: {exc}") from exc
        weights = tuple(int(r) for r in ed.weights_r)
        if list(weights) != list(cfg["weights_r"]):
            raise TBRefusal("weights_mismatch",
                            f"factory weights {weights} vs verifier "
                            f"{list(cfg['weights_r'])}")
        fdyn = factory(**{n: float(v) for n, v in values.items()}).dynamics_gf
        if float_nominal is None:
            float_nominal = fdyn
        label = spec.corner_label(assignment)
        repairs.extend({"corner": ci, "equation": e, "term": t, "gain": g}
                       for e, t, g in ed.repairs)
        if spec.perturbed:
            ex_v = ud.matched_disturbance_vertices(ed.dynamics, channel, spec.delta0)
            fl_v = ud.matched_disturbance_vertices(fdyn, channel, float(spec.delta0))
            for sign, e_dyn, f_dyn in zip("+-", ex_v, fl_v):
                labels.append(f"corner {ci} [{label}] endpoint {sign}delta0")
                idx.append(ci)
                exact.append(e_dyn)
                floats.append(f_dyn)
        else:
            labels.append(f"corner {ci} [{label}] nominal")
            idx.append(ci)
            exact.append(ed.dynamics)
            floats.append(fdyn)
    for i, (e_dyn, f_dyn) in enumerate(zip(exact, floats)):
        _check_float_image(e_dyn, f_dyn, labels[i])
    return Vertices(labels=labels, corner_index=idx, exact=exact,
                    floats=floats, float_nominal=float_nominal,
                    weights_r=weights, channel=channel, corners=corners,
                    repairs=repairs)


def _check_float_image(e_dyn, f_dyn, label: str) -> None:
    """The float design dynamics must be the float image of the exact ones."""
    if len(e_dyn) != len(f_dyn):
        raise TBRefusal("dynamics_mismatch", f"{label}: equation count differs")
    for i, (ge, gf) in enumerate(zip(e_dyn, f_dyn)):
        if len(ge.terms) != len(gf.terms):
            raise TBRefusal("dynamics_mismatch",
                            f"{label} f{i}: term count differs")
        for te, tf in zip(ge.terms, gf.terms):
            if tuple(te.exponents) != tuple(tf.exponents) or \
                    tuple(te.term_types) != tuple(tf.term_types):
                raise TBRefusal("dynamics_mismatch",
                                f"{label} f{i}: support differs")
            a, b = float(te.coefficient), float(tf.coefficient)
            if abs(a - b) > 1e-12 * max(1.0, abs(a)):
                raise TBRefusal("dynamics_mismatch",
                                f"{label} f{i}: exact {te.coefficient} vs "
                                f"float {tf.coefficient}")


# ---------------------------------------------------------------------------
# Two-point (non-product) mode: a union of gain points, ONE V (2026-09-25)
# ---------------------------------------------------------------------------

def parse_point(text: str) -> Tuple[List[str], List[str], List[str]]:
    """``--point`` SPEC: ``;``-separated items, each one of

    * ``name=RATIONAL[,name=RATIONAL...]`` (fixed gains of this point),
    * ``bracket=name:lo:hi:k:q`` (a bracket of this point),
    * ``box=name:lo:hi`` (a box edge pair of this point).

    Returns (gains, brackets, boxes) in the syntax of ``--gains``,
    ``--bracket`` and ``--box``; ``build_gain_spec`` validates them.
    """
    gains, brackets, boxes = [], [], []
    for item in (s.strip() for s in str(text).split(";")):
        if not item:
            continue
        if item.startswith("bracket="):
            brackets.append(item[len("bracket="):])
        elif item.startswith("box="):
            boxes.append(item[len("box="):])
        else:
            gains.append(item)
    if not (gains or brackets or boxes):
        raise TBRefusal("bad_point", f"--point {text!r} is empty")
    return gains, brackets, boxes


@dataclass
class GainPointMember:
    """One gain point of a two-point (union) run and the dynamics it owns."""
    index: int
    spec: GainSpec
    verts: Vertices
    dyn_index: List[int]          # positions in the combined vertex list
    table: Optional[str]
    match_reason: str
    affinity: dict


def combine_vertices(members: List[GainPointMember]) -> Vertices:
    """The joint design's vertex list: point-major, each point's own order.

    Point k's dynamics occupy ``members[k].dyn_index`` in the combined list,
    so W group j of D's bundle witnesses combined dynamics j, and the exact
    phase's endpoint j belongs to exactly one point.
    """
    labels, idx, exact, floats, corners, repairs = [], [], [], [], [], []
    first = members[0].verts
    for mem in members:
        v = mem.verts
        if tuple(v.weights_r) != tuple(first.weights_r) or \
                v.channel != first.channel:
            raise TBRefusal("point_structure_mismatch",
                            f"point {mem.index}: weights or channel differ "
                            f"from point 0")
        start = len(exact)
        mem.dyn_index = list(range(start, start + len(v.exact)))
        offset = len(corners)
        labels.extend(f"point {mem.index}: {lab}" for lab in v.labels)
        idx.extend(offset + ci for ci in v.corner_index)
        exact.extend(v.exact)
        floats.extend(v.floats)
        corners.extend(v.corners)
        repairs.extend(dict(r, point=mem.index) for r in v.repairs)
    return Vertices(labels=labels, corner_index=idx, exact=exact,
                    floats=floats, float_nominal=first.float_nominal,
                    weights_r=first.weights_r, channel=first.channel,
                    corners=corners, repairs=repairs)


def build_members(args) -> List[GainPointMember]:
    """Parse, check (names, R6, dynamics) and match every ``--point``.

    The shared ``--gains``/``--bracket``/``--box`` and ``--delta0`` apply to
    every point; a name given both shared and per point is refused
    (``duplicate_gain``), and two identical points are refused.
    """
    members, seen = [], set()
    for k, text in enumerate(args.point):
        g, b, x = parse_point(text)
        spec = build_gain_spec(args.system, list(args.gains) + g,
                               list(args.bracket) + b, list(args.box) + x,
                               args.delta0, None)
        cfg = check_gain_names(spec)
        key = json.dumps(spec.record(), sort_keys=True)
        if key in seen:
            raise TBRefusal("duplicate_point",
                            f"point {k} repeats an earlier point")
        seen.add(key)
        affinity = assert_affinity(spec)
        verts = build_vertices(spec, cfg)
        table, reason = registry_match(spec)
        members.append(GainPointMember(k, spec, verts, [], table, reason,
                                       affinity))
    return members


def point_phase(phase, member: GainPointMember):
    """The exact phase restricted to one point's endpoints.

    V, its captures and the adopted witness are shared; only the endpoint
    list is cut, so ``exact_pass``, ``eps_certified_float`` and the charge
    ratios become the point's own (min / max over V and ITS endpoints).
    """
    sub = replace(phase, endpoints=[phase.endpoints[j]
                                    for j in member.dyn_index])
    # Review 2026-09-25 m10: ``witness_charge_ratio`` is a stored field of the
    # adopted attempt, i.e. the JOINT phase's worst over every point. A
    # per-point record states its OWN worst (V and its own endpoints, the same
    # definition: charge over the margin policy's budget); the joint value is
    # kept under ``design_stage.joint_design``.
    sub.witness_charge_ratio = sub.worst_charge_ratio
    return sub


# ---------------------------------------------------------------------------
# Registry match (EM binds gains to the printed-gain registry)
# ---------------------------------------------------------------------------

def registry_match(spec: GainSpec) -> Tuple[Optional[str], str]:
    """Which registered gain point the CLI point IS, compared exactly."""
    if spec.box:
        return None, ("a box gain is not a registered gain point; EM emits it "
                      "with --gain-point exact-record")
    reasons = []
    for table_name, table in (("flagship", pg.FLAGSHIP_GAIN_POINTS),
                              ("benchmark", pg.BENCHMARK_GAIN_POINTS)):
        point = table.get(spec.system)
        if point is None:
            continue
        if set(point.gains) != set(spec.names):
            reasons.append(f"{table_name}: gain names {sorted(point.gains)} "
                           f"vs {spec.names}")
            continue
        reg_br = pg.brackets(point)
        diff = []
        for name, g in point.gains.items():
            if g.is_bracketed:
                b = spec.brackets.get(name)
                if b is None or (b.lo, b.hi) != reg_br[name] or \
                        b.power_k != g.power_k or b.power_target != g.power_target:
                    diff.append(name)
            elif spec.rational.get(name) != g.value:
                diff.append(name)
        if not diff:
            return table_name, f"exact match with the {table_name} table"
        reasons.append(f"{table_name}: {diff} differ")
    return None, "; ".join(reasons) or f"no registered point for {spec.system}"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def v_terms_json(V) -> List[dict]:
    return [{"coefficient": float(t.coefficient),
             "exponents": [str(Fraction(e)) for e in t.exponents],
             "term_types": list(t.term_types)} for t in V.terms]


def reconstruct_v(row: dict):
    from gf_framework.algebra.core import GeneralizedForm, GFTerm  # noqa: PLC0415
    terms = [GFTerm(coefficient=float(t["coefficient"]),
                    exponents=tuple(Fraction(e) for e in t["exponents"]),
                    term_types=tuple(t["term_types"]))
             for t in row["V_terms"]]
    return GeneralizedForm(degree_m=row["degree_used"],
                           weights_r=tuple(row["weights_r"]), terms=terms)


def serialize_captures(caps, epsilon: float, label: str) -> List[dict]:
    """Per-octant captures in the exactgains shape, eps pinned to the record."""
    out = []
    for cap in caps or []:
        if float(cap["eps"]) != float(epsilon):
            raise TBRefusal(
                "epsilon_ladder_mismatch",
                f"{label}: a capture certified at eps={cap['eps']!r}, the "
                f"record's margin is {epsilon!r}")
        gram = cap["gram"]
        rows = gram.tolist() if hasattr(gram, "tolist") else gram
        basis = [[int(b) for b in beta] for beta in cap["basis"]]
        y_degree = cap.get("y_degree")
        out.append({
            "signs": [int(s) for s in cap["signs"]],
            "octant_index": int(cap["octant_index"]),
            "eps": float(cap["eps"]),
            "mu": [int(m) for m in cap["mu"]],
            "y_degree": int(y_degree) if y_degree is not None
                        else 2 * sum(basis[0]),
            "basis": basis,
            "gram": [[float(x) for x in row] for row in rows],
        })
    return out


def write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=1, default=_json_default),
                    encoding="utf-8")
    return path


def write_json_gz(path: Path, payload) -> Path:
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
            fh.write(json.dumps(payload, default=_json_default).encode("utf-8"))
    return path


def _json_default(obj):
    if isinstance(obj, Fraction):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(argv) -> dict:
    try:
        from _env import collect_env  # noqa: PLC0415
        env = collect_env()
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        env = {"error": repr(exc)}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)
                                             .as_posix())] + list(argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "env": env,
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def mosek_available() -> bool:
    try:
        import mosek  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def select_support(spec, verts, m, library, args, gate_solver) -> Tuple[list, dict]:
    """``--sparsify rl1``: robust-aware reweighted-l1 support selection.

    Support SELECTION only. It runs ``design_sparse_lyapunov`` over the same
    vertex list as D and a strict vertex-aware float gate; the Lyapunov
    function of record is then designed by D on the selected support, at the
    run's design epsilon, and proved by X. The gate here only picks a support.
    """
    from gf_framework.verification.sparse_sos import (  # noqa: PLC0415
        SparseDesignConfig, design_sparse_lyapunov)
    from gf_framework.verification.verifier import certify_discovered_gf  # noqa: PLC0415

    vertex_arg = verts.floats if len(verts.floats) > 1 else None
    weights = verts.weights_r

    def gate(V_gf):
        return certify_discovered_gf(
            V_gf, weights, m, verts.float_nominal, solver=gate_solver,
            strict_solver=True, dynamics_gf_vertices=vertex_arg)

    t0 = time.time()
    res = design_sparse_lyapunov(
        weights_r=weights, degree_m=m, dynamics_gf=verts.float_nominal,
        active_terms=library, dynamics_gf_vertices=vertex_arg,
        solver=args.design_solver, gate_fn=gate, verbose=False,
        # The ladder and remargin solves run at D's own margin (X5,
        # 2026-09-25): before this they ran at the designer default 1e-4
        # whatever --design-epsilon said.
        config=SparseDesignConfig(design_epsilon=args.design_epsilon))
    record = {
        "method": "design_sparse_lyapunov (robust-aware rl1 over the full "
                  "corner x endpoint vertex list, gate-aware support ladder, "
                  "backward elimination off)",
        "solver": args.design_solver, "gate_solver": gate_solver,
        "design_epsilon": args.design_epsilon,
        "feasible": bool(res.feasible), "gated": bool(res.gated),
        "n_terms_dense": int(res.n_terms_dense),
        "n_terms_sparse": int(res.n_terms_sparse) if res.feasible else None,
        "support": list(res.support),
        "n_rl1_iters": len(res.iterations),
        "weighting": res.weighting,
        "mu_guard": res.mu_guard,
        "message": res.message,
        "time_s": round(time.time() - t0, 2),
        "note": "support selection only; the gate is a float filter, never "
                "the proof. D re-designs on this support, X proves it.",
    }
    if not (res.feasible and res.gated and res.support):
        return [], record
    return [library[i] for i in res.support], record


def run_design(spec, verts, m, args, gate_solver) -> Tuple[Optional[object], dict, object]:
    """D (MOSEK_DIRECT, strict, bundle) or the CLARABEL non-D design."""
    from gf_framework.algebra.enumerate import enumerate_diagonal_terms  # noqa: PLC0415
    from gf_framework.discovery.gf_discovery import mu_filtered_cross  # noqa: PLC0415
    from gf_framework.verification.joint_sos import JointSOSOptimizer  # noqa: PLC0415

    weights = verts.weights_r
    diag = enumerate_diagonal_terms(m, weights, c1_only=True)
    mu, _, cross = mu_filtered_cross(m, weights, diag, verts.float_nominal)
    library = list(diag) + list(cross)
    rec = {"m": m, "mu": [int(x) for x in mu],
           "library": {"diag": len(diag), "cross": len(cross)},
           "design_solver": args.design_solver,
           "stage_label": STAGE_D if args.design_solver == "MOSEK_DIRECT"
                          else STAGE_CLARABEL,
           "design_epsilon": args.design_epsilon,
           "n_vertices": len(verts.floats), "vertex_labels": verts.labels}
    terms = library
    if args.sparsify == "rl1":
        terms, rec["sparsify"] = select_support(spec, verts, m, library, args,
                                                gate_solver)
        if not terms:
            rec["status"] = "sparsify_no_certified_support"
            return None, rec, None
    rec["n_terms_offered"] = len(terms)
    opt = JointSOSOptimizer(
        weights_r=weights, epsilon=args.design_epsilon,
        solver=args.design_solver, strict_solver=True,
        allow_solver_fallback=False, verbose=False,
        parity_reduction=False, capture_design_grams=True,
        design_screen="reject", design_screen_budget=args.margin_policy)
    t0 = time.time()
    res = opt.find_lyapunov_with_dynamics(
        degree_m=m, dynamics_gf=verts.float_nominal, active_terms=list(terms),
        dynamics_gf_vertices=verts.floats if len(verts.floats) > 1 else None)
    rec.update({
        "time_s": round(time.time() - t0, 2),
        "feasible": bool(res.feasible and res.V_gf is not None),
        "solver_status": str(res.solver_status),
        "solver_requested": str(res.solver_requested),
        "solver_used": str(res.solver_used),
        "solver_fallback_used": bool(res.solver_fallback_used),
        "design_screen": res.design_screen,
    })
    if not rec["feasible"]:
        rec["status"] = "design_infeasible_or_error"
        return None, rec, res
    if res.solver_fallback_used:
        rec["status"] = "design_solver_fell_back"
        return None, rec, res
    rec["n_terms"] = len(res.V_gf.terms)
    rec["V_terms"] = v_terms_json(res.V_gf)
    rec["status"] = "designed"
    return res.V_gf, rec, res


def persist_bundle(res, out: Path, m: int, spec, verts, args) -> dict:
    """Write D's Gram bundle and pin its W-group count to the dynamics count."""
    from gf_framework.verification.exact_phase import (  # noqa: PLC0415
        prepare_gram_bundle, save_gram_bundle)

    if not res.gram_certificates:
        raise TBRefusal("no_design_bundle",
                        "the MOSEK_DIRECT design returned no Gram bundle; X "
                        "cannot run on D's bundle and no fallback is taken")
    path = out / f"design_bundle_m{m}.npz"
    save_gram_bundle(path, res.gram_certificates, system=spec.system,
                     epsilon=args.design_epsilon, degree_m=m,
                     solver="MOSEK_DIRECT", solver_used=str(res.solver_used),
                     n_dynamics=len(verts.exact), vertex_labels=verts.labels,
                     produced_by="scripts/certify_gain_point.py")
    prepared = prepare_gram_bundle(path, len(verts.weights_r))
    return check_w_groups(prepared, len(verts.exact), path)


def check_w_groups(prepared, n_dynamics: int, path) -> dict:
    """The invariant of ``run_exact_phase`` (bundle W groups == dynamics)."""
    n_groups = len(prepared.w)
    if n_groups != n_dynamics:
        raise TBRefusal(
            "w_group_count_mismatch",
            f"the design bundle carries W blocks for {n_groups} vertex "
            f"group(s) {sorted(prepared.w)}, X is handed {n_dynamics} "
            f"corner x endpoint dynamics")
    return {"path": Path(path).name, "sha256": sha256_of(path),
            "fingerprint": prepared.fingerprint,
            "n_entries": prepared.n_entries,
            "n_w_groups": n_groups, "n_dynamics": n_dynamics,
            "parity_expanded": bool(prepared.parity_expanded)}


def run_x(V, verts, args, bundle: Optional[dict], out: Path, epsilon: float):
    from gf_framework.verification.exact_phase import run_exact_phase  # noqa: PLC0415

    kwargs = dict(epsilon=epsilon, solver=args.witness_solver,
                  fallback_solver=args.fallback_solver or None,
                  jobs=args.jobs, margin_policy=args.margin_policy)
    if bundle is not None:
        kwargs["gram_bundle"] = str(out / bundle["path"])
        kwargs["expect_bundle_fingerprint"] = bundle["fingerprint"]
    return run_exact_phase(V, verts.exact, len(verts.weights_r), **kwargs)


def claim_texts(stage: Optional[str], bundle: Optional[dict],
                witness: dict) -> Tuple[str, str]:
    """``claim`` and ``design_stage.note`` from what X ACTUALLY adopted.

    Until 2026-09-25 both strings were built from the stage label alone, so a
    D run whose exact phase adopted a fallback re-solve witness (the bundle
    proof left less than the headroom trigger) still said "D -> X on D's
    persisted MOSEK_DIRECT Gram bundle" (reports/X2.md, open issue 2). The
    text now names the adopted ``witness_source`` and solver. The wording of
    the two unchanged cases (D on its bundle; a run with no bundle) is kept,
    so existing readers that match on it still match.
    """
    source = witness.get("witness_source") or witness.get("source")
    solver = witness.get("witness_solver_used") or witness.get("witness_solver")
    n_att = witness.get("n_witness_attempts")
    adopted = f"{source} ({solver})" if solver else f"{source}"
    head = f"certify_gain_point (WP TB): design stage {stage}; "
    if stage == STAGE_D and bundle and source == "design_bundle":
        return (head + "D -> X on D's persisted MOSEK_DIRECT Gram bundle",
                "X consumed D's persisted Gram bundle")
    if stage == STAGE_D and bundle:
        return (head + "D designed V and persisted its MOSEK_DIRECT Gram "
                "bundle, but X did NOT adopt it: X adopted the recorded "
                f"fallback witness {adopted} after {n_att} attempt(s), see "
                "witness.witness_attempts",
                "D's persisted Gram bundle was offered to X; X adopted the "
                f"recorded fallback witness {adopted}")
    return (head + "NOT on the D -> X chain (no MOSEK_DIRECT design bundle); "
            f"the exact phase re-solved its own witness, adopted {adopted}",
            "no design bundle: X re-solved its witness; this run is NOT on "
            "the D -> X chain")


def load_supplied_provenance(path: Optional[str]) -> Optional[dict]:
    """``--provenance-json``: a JSON object, recorded verbatim and labelled.

    The driver checks only that it parses as a JSON object; its content is
    the caller's statement (for example the covariant scaling s, q and the
    base gains of an X3 run) and is never read by any proof stage.
    """
    if not path:
        return None
    raw = Path(path).read_bytes()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TBRefusal("bad_provenance_json",
                        f"{path}: not valid UTF-8 JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise TBRefusal("bad_provenance_json",
                        f"{path}: the top level must be a JSON object")
    return {"note": ("supplied by the caller with --provenance-json, copied "
                     "verbatim and UNVERIFIED; no proof stage reads it"),
            "source_file": _repo_rel(Path(path)),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "supplied": obj}


def exact_records(phase, spec, verts, V, m, epsilon, args, design_rec, bundle,
                  git_sha, joint: Optional[dict] = None) -> Tuple[dict, dict]:
    """The exactgains record and its Gram bundle, from ONE exact phase.

    ``joint`` (two-point mode, 2026-09-25): ``phase`` and ``verts`` are then
    the view of ONE gain point of a joint design, and ``joint`` describes the
    whole design (all points, the dynamics this point owns, the shared
    bundle). It is recorded under ``design_stage.joint_design`` and named in
    the claim.
    """
    v_caps = serialize_captures(phase.v_grams, epsilon, "V")
    corners, bundle_corners = [], []
    ep_names = ("W+", "W-") if spec.perturbed else ("W",)
    for ci, assignment in enumerate(verts.corners):
        js = [j for j, c in enumerate(verts.corner_index) if c == ci]
        eps = [phase.endpoints[j] for j in js]
        summary = {
            "corner_label": spec.corner_label(assignment),
            "assignment": {n: str(v) for n, v in sorted(assignment.items())},
            "dynamics_index": js,
            "exact_pass": bool(phase.v_exact.ok and all(e.ok for e in eps)),
        }
        for lbl, j, ep in zip(ep_names, js, eps):
            summary[lbl] = {
                "resolve_feasible": bool(ep.resolve_feasible),
                "exact_ok": bool(ep.exact.ok),
                "worst_charge_ratio": ep.exact.worst_charge_ratio,
                "worst_charge_over_eps": ep.exact.worst_charge_over_eps,
                "eps_certified": ep.exact.eps_certified_float,
                "lie_crosscheck": ep.lie_crosscheck,
                "witness_source": ep.witness_source,
            }
        corners.append(summary)
        bc = {"corner_label": summary["corner_label"],
              "assignment": summary["assignment"]}
        caps = [serialize_captures(ep.grams, epsilon, f"{lbl}@{ci}")
                for lbl, ep in zip(ep_names, eps)]
        if spec.perturbed:
            bc["endpoints"] = caps
        else:
            bc["W_grams"] = caps[0]
        bundle_corners.append(bc)
    stage = design_rec.get("stage_label")
    witness = phase.witness_record()
    claim, note = claim_texts(stage, bundle, witness)
    if joint and bundle:
        # Review 2026-09-25 m8: a per-point exact.json lives in
        # m<m>/point<k>/, one level below the joint design's bundle. The path
        # is relative to the directory of this exact.json, as in single-point
        # runs (where the two share a directory).
        bundle = dict(bundle, path=f"../{bundle['path']}",
                      path_base="the directory of this exact.json")
    if joint:
        # Review 2026-09-25 m10: the attempt list is the joint phase's; say
        # so on every entry (EM copies the list verbatim into the record).
        witness["witness_attempts"] = [
            dict(a, scope="joint phase over all gain points, not this "
                          "point's own numbers")
            for a in witness.get("witness_attempts") or []]
        witness["witness_charge_ratio_scope"] = (
            "this gain point (V and its own endpoints); the joint value is "
            "design_stage.joint_design.joint_witness_charge_ratio")
        claim += (f"; one V designed jointly over {joint['n_points']} gain "
                  f"points ({joint['n_dynamics_total']} dynamics); this "
                  f"record covers point {joint['point_index']} only")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": git_sha,
        "system": spec.system,
        "row_source": "row.json",
        "delta0_exact": str(spec.delta0),
        "epsilon": epsilon,
        "epsilon_exact": str(Fraction(epsilon)),
        # EM vocabulary: the exact phase injected exact rationals (here the
        # command-line values), never float64 images.
        "gain_binding": "printed_rationals",
        "gain_source": "scripts/certify_gain_point.py command line",
        "rational_gains": {n: str(v) for n, v in sorted(spec.rational.items())},
        "brackets": {n: b.record() for n, b in sorted(spec.brackets.items())},
        "box": {n: {"lo": str(lo), "hi": str(hi)}
                for n, (lo, hi) in sorted(spec.box.items())},
        "witness_solver": args.witness_solver,
        "witness_fallback_solver": args.fallback_solver or None,
        "design_stage": {
            "label": stage,
            "citable_chain": stage == STAGE_D,
            "solver": design_rec.get("design_solver"),
            "design_epsilon": design_rec.get("design_epsilon"),
            "bundle": bundle,
            "witness_source": witness.get("witness_source"),
            "note": note,
        },
        # EM copies this into the portable record's provenance.source_claim,
        # so the stage label survives into the record without a schema change.
        "claim": claim,
        "V": {"degree_m": m, "weights_r": list(verts.weights_r),
              "terms": v_terms_json(V)},
        "n_dynamics": len(verts.exact),
        "dynamics_labels": verts.labels,
        "corners": corners,
        "exact_pass": bool(phase.exact_pass),
        "exact_pass_all": bool(phase.exact_pass),
        "worst_charge_ratio": phase.worst_charge_ratio,
        "witness": witness,
        "time_s": phase.time_s,
        "grams_file": "exact_grams.json.gz",
    }
    if joint:
        record["design_stage"]["joint_design"] = joint
    supplied = getattr(args, "supplied_provenance", None)
    if supplied:
        # Review 2026-09-25 m6. EM has no provenance passthrough; the record
        # reaches this block through source_artifacts (role exact_phase, by
        # sha256) and the claim below names it.
        record["provenance"] = supplied
        record["claim"] += ("; caller provenance (unverified) in the source "
                            "exact.json under provenance.supplied, keys "
                            f"{sorted(supplied['supplied'])}")
    grams = {"timestamp_utc": record["timestamp_utc"], "system": spec.system,
             "V_grams": v_caps, "corners": bundle_corners}
    return record, grams


def row_record(spec, verts, V, m, design_rec) -> dict:
    gains = {n: float(v) for n, v in spec.rational.items()}
    gains.update({n: float(b.lo) for n, b in spec.brackets.items()})
    return {
        "system": spec.system, "certified": True, "degree_used": m,
        "n_terms": len(V.terms),
        "method": f"certify_gain_point ({design_rec.get('stage_label')})",
        "delta0": str(spec.delta0),
        "channel": verts.channel if spec.perturbed else None,
        "V_terms": v_terms_json(V),
        "weights_r": list(verts.weights_r),
        "gains": gains,
    }


def _repo_rel(path: Path) -> str:
    """A repo-relative POSIX path when possible (EM records the paths it is
    given in ``source_artifacts``; a relative path keeps host paths out)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def _run_child(cmd):
    """Run EM or VC from the repository root, UTF-8 on every platform.

    The verifier prints non-ASCII symbols; on a Windows console code page the
    child would otherwise encode them in cp1252 and the parent's decoder would
    drop the whole stream.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(REPO_ROOT), env=env)
    proc.stdout = proc.stdout or ""
    proc.stderr = proc.stderr or ""
    return proc


def default_stem_tag(m: int, stage: str) -> str:
    """EM stem tag; a run that is not D says so in its file names too."""
    return f"tb_m{m}" if stage == STAGE_D else f"tb_{stage}_m{m}"


def emit(out: Path, table: str, args, m: int, env_path: Path,
         stage: str) -> dict:
    """EM as a subprocess (read-only use; EM is not modified by TB)."""
    portable = out / "portable"
    portable.mkdir(exist_ok=True)
    tag = args.stem_tag or default_stem_tag(m, stage)
    cmd = [sys.executable, _repo_rel(EM_SCRIPT),
           "--row", _repo_rel(out / "row.json"),
           "--exact", _repo_rel(out / "exact.json"),
           "--out-dir", _repo_rel(portable),
           "--gain-point", table, "--campaign-block", args.campaign_block,
           "--stem-tag", tag, "--env", _repo_rel(env_path)]
    t0 = time.time()
    proc = _run_child(cmd)
    (out / "emission.log").write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n"
        f"{proc.stderr}", encoding="utf-8")
    result = {"command": cmd, "returncode": proc.returncode,
              "time_s": round(time.time() - t0, 2), "gain_point": table}
    try:
        payload = json.loads(proc.stdout[proc.stdout.index("{"):])
    except ValueError:
        payload = None
    result["em_output"] = payload
    if proc.returncode == 0 and payload and payload.get("record"):
        rec = Path(payload["record"])
        if not rec.is_absolute():
            rec = REPO_ROOT / rec
        result["record"] = rec.name
        result["record_sha256"] = sha256_of(rec)
        result["grams"] = Path(payload["grams"]).name
        summary = rec.parent / (rec.name.replace(".record.json.gz", "")
                                + ".verify_summary.json")
        if summary.exists():
            vs = json.loads(summary.read_text(encoding="utf-8"))
            result["self_verify"] = {
                k: vs.get(k) for k in ("verdict", "eps_certified_exact",
                                       "worst_charge_over_eps",
                                       "acceptance_rule")}
            result["self_verify_summary"] = summary.name
    return result


def run_p(out: Path, emission: dict, python: str) -> dict:
    """P: the stdlib verifier under the named interpreter, with --json."""
    rec = out / "portable" / emission["record"]
    stem = emission["record"].replace(".record.json.gz", "")
    js = out / "portable" / f"{stem}.verification.json"
    cmd = [python, _repo_rel(VC_SCRIPT), _repo_rel(rec), "--json", _repo_rel(js)]
    t0 = time.time()
    proc = _run_child(cmd)
    (out / "portable" / f"{stem}.verification.log").write_text(
        f"$ {' '.join(cmd)}\n\n{proc.stdout}{proc.stderr}", encoding="utf-8")
    result = {"command": cmd, "returncode": proc.returncode,
              "time_s": round(time.time() - t0, 2), "json": js.name}
    if js.exists():
        doc = json.loads(js.read_text(encoding="utf-8"))
        result.update({k: doc.get(k) for k in ("verdict", "eps_certified_exact",
                                               "worst_charge_over_eps")})
    return result


def finish_points(run, entry, out, members, phase, V, m, epsilon, args,
                  design_rec, bundle, git_sha, env_path) -> Tuple[str, int]:
    """Two-point mode after a joint X PASS: one exactgains record, one row
    and one portable record PER POINT, all with the same V.

    Each point's record carries its own gains, brackets and corners, the V
    captures (shared) and only the W captures of its own dynamics; EM and P
    run per point. Exit 0 only when every point is emitted and P VERIFIED.
    """
    stage = design_rec.get("stage_label")
    n_total = sum(len(mem.dyn_index) for mem in members)
    points_out, worst = [], EXIT_OK
    for mem in members:
        pout = out / f"point{mem.index}"
        pout.mkdir(exist_ok=True)
        sub = point_phase(phase, mem)
        joint = {"n_points": len(members), "point_index": mem.index,
                 "n_dynamics_total": n_total,
                 "dynamics_indices": list(mem.dyn_index),
                 "points": [x.spec.record() for x in members],
                 "joint_witness_charge_ratio": phase.witness_charge_ratio,
                 "joint_worst_charge_over_eps": phase.worst_charge_over_eps,
                 "joint_eps_certified": phase.eps_certified_float,
                 "note": ("ONE V designed and proved jointly over every point; "
                          "the bundle and the witness attempts are those of the "
                          "joint phase, the margins, charge ratios and verdict "
                          "here are this point's (V and its own endpoints)")}
        record, grams = exact_records(sub, mem.spec, mem.verts, V, m, epsilon,
                                      args, design_rec, bundle, git_sha,
                                      joint=joint)
        write_json(pout / "exact.json", record)
        write_json_gz(pout / "exact_grams.json.gz", grams)
        write_json(pout / "row.json", row_record(mem.spec, mem.verts, V, m,
                                                 design_rec))
        prec = {"point": mem.index, "dir": pout.name,
                "gain_point": mem.spec.record(),
                "exact_pass": bool(sub.exact_pass),
                "eps_certified": sub.eps_certified_float,
                "worst_charge_over_eps": sub.worst_charge_over_eps,
                "registry_match": {"table": mem.table,
                                   "reason": mem.match_reason},
                "outputs": {q.name: sha256_of(q) for q in
                            (pout / "row.json", pout / "exact.json",
                             pout / "exact_grams.json.gz")}}
        points_out.append(prec)
        print(f"[TB] m={m} point {mem.index}: exact_pass {sub.exact_pass}, "
              f"eps_certified {sub.eps_certified_float}", flush=True)
        if args.no_emit or not sub.exact_pass:
            if not sub.exact_pass:
                worst = EXIT_NOT_CERTIFIED
            continue
        route = mem.table if mem.table is not None else EXACT_RECORD_ROUTE
        tag = (args.stem_tag or default_stem_tag(m, stage)) + f"_pt{mem.index}"
        em_args = replace_namespace(args, stem_tag=tag)
        emission = emit(pout, route, em_args, m, env_path, stage)
        emission["registry_match"] = prec["registry_match"]
        prec["emission"] = emission
        if emission["returncode"] != 0:
            worst = max(worst, EXIT_EM_REFUSED)
            continue
        if args.p_python:
            prec["P"] = run_p(pout, emission, args.p_python)
            if prec["P"].get("verdict") != "VERIFIED":
                worst = max(worst, EXIT_EM_REFUSED)
    run["points_result"] = points_out
    entry["points"] = [{k: p_.get(k) for k in ("point", "exact_pass",
                                                "eps_certified")}
                       for p_ in points_out]
    if args.no_emit:
        return ("x_pass_not_emitted", EXIT_OK)
    if worst == EXIT_OK:
        return ("certified_and_emitted", EXIT_OK)
    if worst == EXIT_NOT_CERTIFIED:
        return ("point_not_certified", EXIT_NOT_CERTIFIED)
    return ("em_refused_or_P_failed", worst)


def replace_namespace(ns, **kw):
    """A shallow copy of an argparse namespace with some fields replaced."""
    out = argparse.Namespace(**vars(ns))
    for k, v in kw.items():
        setattr(out, k, v)
    return out


def run_plan(run, out_root, spec, verts, args, gate_solver, table,
             match_reason, env_path, prov,
             members: Optional[List[GainPointMember]] = None
             ) -> Tuple[str, int]:
    """The degree ladder: design, X, record, EM, P. Returns (status, code).

    With ``members`` (two-point mode) ``spec`` is point 0 and ``verts`` the
    combined vertex list; D and X run once over all of it, and the records
    are written per point by :func:`finish_points`.
    """
    git_sha = (prov.get("env") or {}).get("git_sha")
    if args.v_row:
        row = json.loads(Path(args.v_row).read_text(encoding="utf-8"))
        if row.get("system") != spec.system:
            run["refusal"] = {"kind": "v_row_system_mismatch",
                              "detail": f"{row.get('system')} vs {spec.system}"}
            return ("refused:v_row_system_mismatch", EXIT_REFUSED)
        plan = [(int(row["degree_used"]), reconstruct_v(row))]
    else:
        plan = [(m, None) for m in args.degrees]

    for m, V_fixed in plan:
        out = out_root / f"m{m}"
        out.mkdir(parents=True, exist_ok=True)
        entry = {"m": m, "dir": out.name}
        run["degrees"].append(entry)
        bundle = None
        try:
            if V_fixed is not None:
                V = V_fixed
                epsilon = args.epsilon
                design_rec = {"stage_label": STAGE_FIXED_V,
                              "design_solver": None, "design_epsilon": None,
                              "v_row": str(args.v_row),
                              "v_row_sha256": sha256_of(Path(args.v_row)),
                              "n_terms": len(V.terms)}
            else:
                epsilon = args.design_epsilon
                print(f"[TB] m={m}: design ({args.design_solver}) over "
                      f"{len(verts.floats)} vertices ...", flush=True)
                V, design_rec, res = run_design(spec, verts, m, args, gate_solver)
                if V is not None and args.design_solver == "MOSEK_DIRECT":
                    bundle = persist_bundle(res, out, m, spec, verts, args)
                    design_rec["bundle"] = bundle
            write_json(out / "design.json", design_rec)
            entry["design"] = {k: design_rec.get(k) for k in
                               ("stage_label", "status", "feasible",
                                "solver_status", "n_terms", "time_s")}
            if V is None:
                print(f"[TB] m={m}: no design ({design_rec.get('status')})",
                      flush=True)
                continue
            print(f"[TB] m={m}: {len(V.terms)} terms; X over {len(verts.exact)} "
                  f"dynamics ({'D bundle' if bundle else args.witness_solver + ' re-solve'}) ...",
                  flush=True)
            phase = run_x(V, verts, args, bundle, out, epsilon)
        except TBRefusal as exc:
            entry["refusal"] = exc.as_dict()
            run["refusal"] = exc.as_dict()
            print(f"[TB] REFUSED ({exc.kind}): {exc.detail}", file=sys.stderr)
            return (f"refused:{exc.kind}", EXIT_REFUSED)

        witness = phase.witness_record()
        if members:
            entry["exact"] = {"exact_pass": bool(phase.exact_pass),
                              "worst_charge_over_eps": phase.worst_charge_over_eps,
                              "eps_certified": phase.eps_certified_float,
                              "witness_source": witness.get("witness_source"),
                              "witness_solver": witness.get("witness_solver"),
                              "n_witness_attempts": witness.get("n_witness_attempts"),
                              "time_s": phase.time_s}
            print(f"[TB] m={m}: joint X {'PASS' if phase.exact_pass else 'FAIL'} "
                  f"over {len(verts.exact)} dynamics (worst charge/eps "
                  f"{phase.worst_charge_over_eps}, eps_certified "
                  f"{phase.eps_certified_float}, witness "
                  f"{witness.get('witness_source')})", flush=True)
            write_json(out / "joint_exact_summary.json", {
                "exact_pass": bool(phase.exact_pass),
                "eps_certified": phase.eps_certified_float,
                "worst_charge_over_eps": phase.worst_charge_over_eps,
                "n_dynamics": len(verts.exact), "dynamics_labels": verts.labels,
                "points": [{"point": x.index, "dynamics_indices": x.dyn_index,
                            "gain_point": x.spec.record()} for x in members],
                "witness": witness, "time_s": phase.time_s})
            if not phase.exact_pass:
                continue
            run["certified_degree"] = m
            run["stage_label"] = design_rec.get("stage_label")
            run["citable_chain"] = design_rec.get("stage_label") == STAGE_D
            return finish_points(run, entry, out, members, phase, V, m,
                                 epsilon, args, design_rec, bundle, git_sha,
                                 env_path)
        entry["exact"] = {"exact_pass": bool(phase.exact_pass),
                          "worst_charge_over_eps": phase.worst_charge_over_eps,
                          "eps_certified": phase.eps_certified_float,
                          "witness_source": witness.get("witness_source"),
                          "witness_solver": witness.get("witness_solver"),
                          "n_witness_attempts": witness.get("n_witness_attempts"),
                          "time_s": phase.time_s}
        try:
            record, grams = exact_records(phase, spec, verts, V, m, epsilon,
                                          args, design_rec, bundle, git_sha)
        except TBRefusal as exc:
            entry["refusal"] = exc.as_dict()
            continue
        write_json(out / "exact.json", record)
        write_json_gz(out / "exact_grams.json.gz", grams)
        print(f"[TB] m={m}: X {'PASS' if phase.exact_pass else 'FAIL'} "
              f"(worst charge/eps {phase.worst_charge_over_eps}, eps_certified "
              f"{phase.eps_certified_float})", flush=True)
        if not phase.exact_pass:
            continue
        write_json(out / "row.json", row_record(spec, verts, V, m, design_rec))
        entry["outputs"] = {p.name: sha256_of(p) for p in
                            (out / "design.json", out / "row.json",
                             out / "exact.json", out / "exact_grams.json.gz")}
        if bundle:
            entry["outputs"][bundle["path"]] = bundle["sha256"]
        run["certified_degree"] = m
        run["stage_label"] = design_rec.get("stage_label")
        run["citable_chain"] = design_rec.get("stage_label") == STAGE_D
        if args.no_emit:
            return ("x_pass_not_emitted", EXIT_OK)
        # A registered point goes through the registry route; any other point
        # (a box, a scaled point, a new tuning) is emitted with EM's
        # exact-record mode, which reads the gain point from exact.json and
        # re-checks it against the portable verifier. This was the exit-5
        # "emission blocked" path before the emitter change.
        route = table if table is not None else EXACT_RECORD_ROUTE
        emission = emit(out, route, args, m, env_path,
                        design_rec.get("stage_label"))
        emission["registry_match"] = {"table": table, "reason": match_reason}
        run["emission"] = emission
        if emission["returncode"] != 0:
            return ("em_refused", EXIT_EM_REFUSED)
        if args.p_python:
            run["P"] = run_p(out, emission, args.p_python)
            if run["P"].get("verdict") != "VERIFIED":
                return ("emitted_but_P_failed", EXIT_EM_REFUSED)
        return ("certified_and_emitted", EXIT_OK)
    return ("not_certified", EXIT_NOT_CERTIFIED)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True,
                    help="a system key of verification/standalone/verify_certificate.py")
    ap.add_argument("--gains", action="append", default=[],
                    help="name=RATIONAL,... exact gains (p/q, integer or finite "
                         "decimal); repeatable")
    ap.add_argument("--bracket", action="append", default=[],
                    help="name:lo:hi:k:q, a rational bracket of the positive "
                         "root of g^k = q (checked in Q); repeatable")
    ap.add_argument("--box", action="append", default=[],
                    help="name:lo:hi, an affine gain interval covered by its "
                         "two endpoints (guard R6 applies); repeatable")
    ap.add_argument("--point", action="append", default=[],
                    help="two-point (union) mode, repeatable: one gain point "
                         "as ';'-separated items 'name=RATIONAL[,...]', "
                         "'bracket=name:lo:hi:k:q', 'box=name:lo:hi'. ONE V "
                         "is designed and proved over every point's corner x "
                         "endpoint dynamics; one record is emitted per point. "
                         "--gains/--bracket/--box/--delta0 are shared")
    ap.add_argument("--provenance-json", default=None, metavar="FILE",
                    help="a JSON object of caller provenance (e.g. "
                         "{\"scaling\": {\"s\": ..., \"q\": ..., "
                         "\"base_gains\": {...}}}), copied verbatim, "
                         "UNVERIFIED, into run.json and every exact.json under "
                         "provenance.supplied. Not a proof input")
    ap.add_argument("--preset", choices=("flagship", "benchmark"), default=None,
                    help="start from a registered printed-gain point; --gains, "
                         "--bracket and --box override single names")
    ap.add_argument("--delta0", default="0",
                    help="matched disturbance bound, exact rational (0 = nominal)")
    ap.add_argument("--degrees", type=int, nargs="+", default=None,
                    help="degree ladder m ...; the first degree that passes X wins")
    ap.add_argument("--design-epsilon", type=float, default=DEFAULT_DESIGN_EPSILON,
                    help="design margin; X runs at the same margin, because D's "
                         "bundle witnesses exactly the margin it subtracted "
                         "(default %(default)g)")
    ap.add_argument("--design-solver", choices=DESIGN_SOLVERS,
                    default=DEFAULT_DESIGN_SOLVER,
                    help="MOSEK_DIRECT (default) is D: strict, no fallback, Gram "
                         "bundle persisted. CLARABEL is for tests and smoke runs "
                         "only: it captures no Gram, X re-solves, and every "
                         "output is labelled clarabel_design_not_D")
    ap.add_argument("--v-row", default=None,
                    help="fixed-V mode: take V from this row JSON and skip "
                         "design (outputs labelled fixed_v_row_not_D)")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="exact margin in --v-row mode (default: --design-epsilon)")
    ap.add_argument("--sparsify", choices=("rl1",), default=None,
                    help="robust-aware reweighted-l1 support selection before D")
    ap.add_argument("--gate-solver", default=None,
                    help="float gate of the --sparsify support ladder (default: "
                         "MOSEK with a MOSEK_DIRECT design, else CLARABEL)")
    ap.add_argument("--witness-solver", default=None,
                    help="X's re-solve witness when no bundle witnesses (and "
                         "the retry after a bundle); default MOSEK, or CLARABEL "
                         "with --design-solver CLARABEL")
    ap.add_argument("--fallback-solver", default="CLARABEL",
                    help="X's single labelled fallback witness retry "
                         "(default %(default)s; empty string disables)")
    ap.add_argument("--margin-policy", choices=("achieved", "half"),
                    default="achieved",
                    help="exact acceptance threshold (default %(default)s)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes of the per-octant exact arithmetic")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-frozen-root", action="store_true",
                    help="permit writing into artifacts/ablation/** (refused "
                         "by default)")
    ap.add_argument("--campaign-block", default="TB",
                    help="campaign block recorded by EM (default %(default)s)")
    ap.add_argument("--stem-tag", default=None,
                    help="EM stem tag (default tb_m<m>)")
    ap.add_argument("--no-emit", action="store_true",
                    help="stop after the exactgains record")
    ap.add_argument("--p-python", default=None,
                    help="after emission, run the stdlib verifier (P) under "
                         "this interpreter with --json, e.g. /usr/bin/python3")
    ap.add_argument("--check-only", action="store_true",
                    help="validate the gain point, guard R6 and the dynamics, "
                         "write plan.json, and stop before any solve")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)

    from _frozen_root import resolve_out_dir  # noqa: PLC0415
    out_root = resolve_out_dir(args.out_dir, Path(args.out_dir),
                               allow_frozen=args.allow_frozen_root)
    if args.witness_solver is None:
        args.witness_solver = ("CLARABEL" if args.design_solver == "CLARABEL"
                               and not args.v_row else "MOSEK")
    gate_solver = args.gate_solver or (
        "MOSEK" if args.design_solver == "MOSEK_DIRECT" else "CLARABEL")

    prov = provenance(argv)
    run = {"schema": "certify_gain_point/1", "provenance": prov,
           "args": vars(args).copy(), "degrees": [], "status": None}
    run_path = out_root / "run.json"
    env_path = out_root / "env.json"
    write_json(env_path, prov["env"])

    def finish(status: str, code: int) -> int:
        run["status"] = status
        run["exit_code"] = code
        write_json(run_path, run)
        print(f"[TB] {status} (exit {code}); run record: {run_path}", flush=True)
        return code

    try:
        if args.v_row and args.sparsify:
            raise TBRefusal("bad_usage", "--sparsify needs a design; it cannot "
                                         "be combined with --v-row")
        if not args.v_row and not args.degrees:
            raise TBRefusal("bad_usage", "--degrees is required without --v-row")
        if args.v_row and args.epsilon is None:
            args.epsilon = args.design_epsilon
        if not args.v_row and args.epsilon not in (None, args.design_epsilon):
            raise TBRefusal("bad_usage",
                            "--epsilon differs from --design-epsilon; X on D's "
                            "bundle runs at the design margin")
        args.supplied_provenance = load_supplied_provenance(
            args.provenance_json)
        if args.supplied_provenance:
            run["provenance"]["supplied"] = args.supplied_provenance
        members = None
        if args.point:
            if args.preset or args.v_row:
                raise TBRefusal("bad_usage", "--point cannot be combined with "
                                             "--preset or --v-row")
            members = build_members(args)
            verts = combine_vertices(members)
            spec = members[0].spec
            table, match_reason = members[0].table, members[0].match_reason
            affinity = [m_.affinity for m_ in members]
        else:
            spec = build_gain_spec(args.system, args.gains, args.bracket,
                                   args.box, args.delta0, args.preset)
            cfg = check_gain_names(spec)
            affinity = assert_affinity(spec)
            verts = build_vertices(spec, cfg)
            table, match_reason = registry_match(spec)
        run.update({
            "gain_point": spec.record(),
            "guard_r6": affinity,
            "dynamics": {"n": len(verts.exact), "labels": verts.labels,
                         "weights_r": list(verts.weights_r),
                         "channel": verts.channel,
                         "repaired_coefficients": verts.repairs},
            "registry_match": {"table": table, "reason": match_reason,
                               "emission_route": table or EXACT_RECORD_ROUTE},
        })
        if members:
            run["gain_point"] = {"points": [m_.spec.record() for m_ in members]}
            run["registry_match"] = {"points": [
                {"point": m_.index, "table": m_.table, "reason": m_.match_reason,
                 "emission_route": m_.table or EXACT_RECORD_ROUTE,
                 "dynamics_indices": m_.dyn_index} for m_ in members]}
            print(f"[TB] {spec.system}: {len(members)} gain points, "
                  f"{len(verts.corners)} corner(s) in all, {len(verts.exact)} "
                  f"dynamics; EM routes "
                  f"{[m_.table or EXACT_RECORD_ROUTE for m_ in members]}",
                  flush=True)
        else:
            print(f"[TB] {spec.system}: {len(verts.corners)} corner(s) x "
                  f"{2 if spec.perturbed else 1} endpoint(s) = {len(verts.exact)} "
                  f"dynamics; registry match: {table or 'none'} ({match_reason}); "
                  f"EM route: {table or EXACT_RECORD_ROUTE}",
                  flush=True)
        if args.check_only:
            write_json(out_root / "plan.json", run)
            return finish("checked_only", EXIT_OK)
        if not args.v_row and args.design_solver == "MOSEK_DIRECT" \
                and not mosek_available():
            raise TBRefusal(
                "design_solver_unavailable",
                "MOSEK_DIRECT was requested and mosek is not importable; D "
                "never falls back to another solver. Use --design-solver "
                "CLARABEL for a smoke run (not D)")
    except TBRefusal as exc:
        run["refusal"] = exc.as_dict()
        print(f"[TB] REFUSED ({exc.kind}): {exc.detail}", file=sys.stderr)
        return finish(f"refused:{exc.kind}", EXIT_REFUSED)

    try:
        status, code = run_plan(run, out_root, spec, verts, args,
                                gate_solver, table, match_reason, env_path,
                                prov, members=members)
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        import traceback  # noqa: PLC0415
        run["error"] = {"type": type(exc).__name__,
                        "detail": str(exc)[:2000],
                        "traceback": traceback.format_exc()[-6000:]}
        finish(f"unexpected_error:{type(exc).__name__}", EXIT_ERROR)
        raise
    return finish(status, code)


if __name__ == "__main__":
    sys.exit(main())
