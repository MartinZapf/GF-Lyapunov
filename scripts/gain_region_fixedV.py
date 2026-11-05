#!/usr/bin/env python3
"""Fixed-V certified gain regions (C3, campaign block E9; also E11.d and E4.a).

See docs/papers/big_draft/PRE_WRITING_CAMPAIGN.md, block E9.

Structural fact this driver exploits
------------------------------------
For a FIXED V (fixed basis, fixed coefficients, fixed margin) the cleared
octant target W = -grad(V).f is AFFINE in the gain vector, in the coordinates
where each gain multiplies exactly one dynamics term, and the SOS/PSD
constraints are convex. The set of gains certified by that fixed V is
therefore convex, so an SOS-PASS at every VERTEX of a gain box certifies the
whole box. For a perturbed run the same convexity argument in the disturbance
gives the second axis: both inclusion endpoints f +/- delta0.e_ch are checked
AT EVERY VERTEX, so the certified object is the full product
(gain box) x (|delta| <= delta0). This is the box generalization of the
irrational-gain bracket convention in scripts/exact_gate_standard_gains.py
and of the endpoint convention in
discovery.universal_discovery.matched_disturbance_vertices.

Guard R6 (the reason this driver refuses rather than assumes)
------------------------------------------------------------
Affinity holds in the k-coordinates, NOT in every parameterization. PID-SMC is
the standing counterexample: its factory takes lambda1, lambda2, lambda3 and a
scaling Delta, and the dynamics coefficients are
k1 = lambda1.Delta^(2/3), k2 = lambda2.Delta^(1/2), k3 = lambda3.Delta, which
is not affine in (lambda, Delta) jointly. This driver therefore DECLARES the
affine gain coordinates per system and ASSERTS the affinity SYMBOLICALLY
(sympy) before any solve: every declared gain must enter exactly one dynamics
term, with a coefficient of the form const*gain, and W must be linear in each
declared gain with no gain-gain products. A violation aborts that system with
a typed refusal. Every region this driver reports is stated in k-coordinates.

Verification path
-----------------
Every cell runs through the authoritative W path of the strict gate,
gf_framework.verification.joint_sos.verify_w_given_numeric: the cleared-W
octant polynomial of the STORED V (no coefficient re-optimization), common
monomial factored out, strict epsilon-margin SOS on the quotient, per open
orthant, with the recovered Gram validated a-posteriori. Nothing about the SOS
check is reimplemented here. The gain-independent V stage (V > 0 by standard
SOS over all octants, plus the C1 / Lemma-5 regularity classification) is run
ONCE per invocation, since V does not depend on the gains at all.

Solvers
-------
--solver MOSEK | MOSEK_DIRECT | CLARABEL, with no silent fallback anywhere
(strict_solver=True and allow_solver_fallback=False on every solve).
MOSEK_DIRECT is the direct Task-API ENCODING of MOSEK and exists only in the
DESIGN path (JointSOSOptimizer._find_lyapunov_direct); the fixed-V W check has
no direct encoding, so requesting it produces a typed refusal instead of a
silent degradation to the CVXPY encoding.

Typed statuses (campaign binding rule 5), one per (vertex x endpoint) cell:
certified | solver_infeasible | solver_unknown | timeout | oom |
unexpected_error. A failed cell means "not certified at this vertex", never
"unstable" and never a statement about the system.

Modes
-----
  1. box            certify an axis-aligned box (2^p vertices x endpoints)
  2. --grow         outward bisection: largest certified box + first failed shell
  3. --max-delta0   largest certified delta0 at a fixed gain point (E9.e)

Examples
--------
  python scripts/gain_region_fixedV.py --system sta \\
      --v-from matrix artifacts/publication/results_gf_sos_clean/gf_matrix_20260610_230219.json \\
      --delta0 0 --box "k1:1.2:1.8,k2:0.9:1.3" --solver MOSEK \\
      --output-dir artifacts/campaign_2026-08/E9_gain_regions/

  python scripts/gain_region_fixedV.py --system levant3d_2003print \\
      --v-from record artifacts/ablation/levant_2003_perturbed/levant3d_2003print_delta1_sparse_row.json \\
      --delta0 1 --rel-halfwidth 0.05 --grow --solver MOSEK \\
      --output-dir artifacts/campaign_2026-08/E9_gain_regions/

  python scripts/gain_region_fixedV.py --system sta --max-delta0 \\
      --v-from matrix .../gf_matrix_20260610_230219.json --solver CLARABEL \\
      --output-dir artifacts/campaign_2026-08/E9_gain_regions/
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import collect_env  # scripts/_env.py (read-only, C15)

from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.discovery.universal_discovery import matched_disturbance_vertices
from gf_framework.systems import printed_gains as pg

BLOCK = "E9"

#: Campaign binding rule 5, restricted to the outcomes a fixed-V W check can
#: produce (there is no separate design stage here, so "gate_rejected" of the
#: design-plus-gate drivers has no analogue: the gate IS the run).
TYPED_STATUSES = (
    "certified", "solver_infeasible", "solver_unknown", "timeout", "oom",
    "unexpected_error",
)

DEFAULT_EPSILON = 1e-6
#: Refusal exit code, distinct from argparse's 2 and from a clean run's 0.
REFUSAL_EXIT_CODE = 4


class GainRegionRefusal(Exception):
    """A typed refusal: the run cannot be performed as requested.

    ``kind`` is the machine-readable reason, recorded in the artifact so a
    refusal is as citable as a result.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


# ===========================================================================
# Per-system declaration of the AFFINE gain coordinates (guard R6)
# ===========================================================================

@dataclass(frozen=True)
class GainCoordinates:
    """The declared affine gain coordinates of one system.

    ``build`` maps {coordinate name -> value} to the nominal dynamics. The
    values may be floats (a real solve) or sympy Symbols (the affinity
    assertion); both go through the SAME system-module builder, so the
    assertion is made about the object the solver actually sees.
    """
    system: str
    coords: Tuple[str, ...]
    build: Callable[[Mapping[str, object]], List[GeneralizedForm]]
    weights_r: Tuple[int, ...]
    #: weighted-degree-0 channel carrying the matched disturbance.
    channel: int
    #: printed_gains registry key + table for the default centre.
    registry_key: str
    registry_flagship: bool = False
    #: coordinate-hygiene note printed and embedded in every artifact.
    note: str = ""
    #: registry gains -> declared k-coordinates (identity unless stated).
    center_from_registry: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None


def _b_fosmc(v):
    from gf_framework.systems.fosmc import get_fosmc_dynamics_gf
    return get_fosmc_dynamics_gf(v["k"])


def _b_sta(v):
    from gf_framework.systems.sta import get_sta_dynamics_gf
    return get_sta_dynamics_gf(v["k1"], v["k2"])


def _b_twisting(v):
    from gf_framework.systems.twisting import get_twisting_dynamics_gf
    return get_twisting_dynamics_gf(v["r1"], v["r2"])


def _b_dsta(v):
    from gf_framework.systems.dsta import get_dsta_dynamics_gf
    return get_dsta_dynamics_gf(v["k1"], v["k2"], v["k3"])


def _b_levant3d(v):
    from gf_framework.systems.levant3d import get_levant3d_dynamics_gf
    return get_levant3d_dynamics_gf(v["lambda1"], v["lambda2"], v["lambda3"])


def _b_levant4d(v):
    from gf_framework.systems.levant4d import get_levant4d_dynamics_gf
    return get_levant4d_dynamics_gf(v["lambda1"], v["lambda2"], v["lambda3"],
                                    v["lambda4"])


def _b_levant5d(v):
    from gf_framework.systems.levant5d import get_levant5d_dynamics_gf
    return get_levant5d_dynamics_gf(v["lambda1"], v["lambda2"], v["lambda3"],
                                    v["lambda4"], v["lambda5"])


def _b_pidsmc(v):
    """PID-SMC in its AFFINE (k) coordinates.

    The system module's builder takes (lambda1, lambda2, lambda3, Delta) and
    forms k1 = lambda1.Delta^(2/3), k2 = lambda2.Delta^(1/2), k3 = lambda3.Delta.
    Passing Delta = 1 makes that map the identity bit for bit (1.0**p == 1.0
    and x*1.0 == x for every float x), so the coefficients of the returned
    dynamics are exactly the declared k values. Delta is NOT a declared gain
    coordinate: k is not affine in (lambda, Delta) jointly, which is precisely
    the R6 trap.
    """
    from gf_framework.systems.pidsmc import get_pidsmc_dynamics_gf
    return get_pidsmc_dynamics_gf(v["k1"], v["k2"], v["k3"], delta=1.0)


def _b_cta(v):
    from gf_framework.systems.continuous_twisting import get_cta_dynamics_gf
    return get_cta_dynamics_gf(v["k1"], v["k2"], v["k3"], v["k4"])


def _b_fc_cta(v):
    from gf_framework.systems.fc_cta import get_fc_cta_dynamics_gf
    return get_fc_cta_dynamics_gf(v["k1"], v["k2"], v["k3"], v["k4"], v["k5"])


def _pidsmc_center(reg: Dict[str, float]) -> Dict[str, float]:
    """Registry (lambda, Delta) -> declared k-coordinates."""
    delta = float(reg.get("delta", 1.0))
    return {
        "k1": float(reg["lambda1"]) * delta ** (2.0 / 3.0),
        "k2": float(reg["lambda2"]) * delta ** 0.5,
        "k3": float(reg["lambda3"]) * delta,
    }


_LAMBDA_NOTE = (
    "Each lambda_i multiplies exactly one dynamics term, so the lambda "
    "coordinates ARE the affine k-coordinates for this system."
)

SYSTEMS: Dict[str, GainCoordinates] = {
    "fosmc": GainCoordinates(
        system="fosmc", coords=("k",), build=_b_fosmc, weights_r=(1,), channel=0,
        registry_key="fosmc",
        note="s' = -k.sign(s); k multiplies the single dynamics term.",
    ),
    "sta": GainCoordinates(
        system="sta", coords=("k1", "k2"), build=_b_sta, weights_r=(2, 1), channel=1,
        registry_key="sta",
        note="k1 multiplies ceil(x1)^(1/2) in f1, k2 multiplies sign(x1) in f2.",
    ),
    "twisting": GainCoordinates(
        system="twisting", coords=("r1", "r2"), build=_b_twisting, weights_r=(2, 1),
        channel=1, registry_key="twisting",
        note="r1 multiplies sign(x1), r2 multiplies sign(x2), both in f2. "
             "The region is reported in (r1, r2); the opposing-relay margin "
             "r1 - r2 is a derived quantity, not a box coordinate.",
    ),
    "dsta": GainCoordinates(
        system="dsta", coords=("k1", "k2", "k3"), build=_b_dsta, weights_r=(2, 1),
        channel=1, registry_key="dsta",
        note="k1 in f1; k2 (sign(x1)) and k3 (sign(x2)) in f2.",
    ),
    "levant3d": GainCoordinates(
        system="levant3d", coords=("lambda1", "lambda2", "lambda3"),
        build=_b_levant3d, weights_r=(3, 2, 1), channel=2,
        registry_key="levant3d", note=_LAMBDA_NOTE,
    ),
    "levant3d_2003print": GainCoordinates(
        system="levant3d_2003print", coords=("lambda1", "lambda2", "lambda3"),
        build=_b_levant3d, weights_r=(3, 2, 1), channel=2,
        registry_key="levant3d_2003print", registry_flagship=True,
        note=_LAMBDA_NOTE + " Same dynamics builder as levant3d; only the "
             "registered gain point differs (2003-print variant).",
    ),
    "levant4d": GainCoordinates(
        system="levant4d",
        coords=("lambda1", "lambda2", "lambda3", "lambda4"),
        build=_b_levant4d, weights_r=(4, 3, 2, 1), channel=3,
        registry_key="levant4d", note=_LAMBDA_NOTE,
    ),
    "levant4d_2003print": GainCoordinates(
        system="levant4d_2003print",
        coords=("lambda1", "lambda2", "lambda3", "lambda4"),
        build=_b_levant4d, weights_r=(4, 3, 2, 1), channel=3,
        registry_key="levant4d_2003print", registry_flagship=True,
        note=_LAMBDA_NOTE + " Same dynamics builder as levant4d; only the "
             "registered gain point differs (recursive (1.1, 1.5, 3, 5), WP X10).",
    ),
    "levant5d": GainCoordinates(
        system="levant5d",
        coords=("lambda1", "lambda2", "lambda3", "lambda4", "lambda5"),
        build=_b_levant5d, weights_r=(5, 4, 3, 2, 1), channel=4,
        registry_key="levant5d", registry_flagship=True,
        note=_LAMBDA_NOTE + " levant5d is the fourth-order differentiator "
             "(5 states) and is NOT in the benchmark matrix, so its printed "
             "gain point lives in the FLAGSHIP registry table; the centre is "
             "read from there.",
    ),
    "pidsmc": GainCoordinates(
        system="pidsmc", coords=("k1", "k2", "k3"), build=_b_pidsmc,
        weights_r=(3, 2, 1), channel=2, registry_key="pidsmc",
        center_from_registry=_pidsmc_center,
        note="R6 CASE. Declared coordinates are k1, k2, k3 (the dynamics "
             "coefficients), NOT lambda1, lambda2, lambda3: the factory forms "
             "k1 = lambda1.Delta^(2/3), k2 = lambda2.Delta^(1/2), "
             "k3 = lambda3.Delta, which is not affine in (lambda, Delta) "
             "jointly. Delta is fixed at 1 here, where k equals lambda "
             "numerically; the region is still stated in k-coordinates.",
    ),
    "continuous_twisting": GainCoordinates(
        system="continuous_twisting", coords=("k1", "k2", "k3", "k4"),
        build=_b_cta, weights_r=(3, 2, 1), channel=2,
        registry_key="continuous_twisting",
        note="CTA (Torres-Gonzalez, Fridman, Moreno 2017): k1, k2 in f2; "
             "k3 (sign(x1)) and k4 (sign(x2)) in f3.",
    ),
    "fc_cta": GainCoordinates(
        system="fc_cta", coords=("k1", "k2", "k3", "k4", "k5"), build=_b_fc_cta,
        weights_r=(3, 2, 1), channel=2, registry_key="fc_cta",
        note="FC-CTA: CTA plus the novel k5.sign(x3) relay in f3.",
    ),
}


def registry_center(decl: GainCoordinates) -> Tuple[Dict[str, float], dict]:
    """Default box centre in the DECLARED coordinates, from printed_gains.

    printed_gains is the single source of truth for gain values and brackets
    (campaign item C20); nothing numeric is duplicated here.
    """
    point = pg.gain_point(decl.registry_key, flagship=decl.registry_flagship)
    reg = pg.float_gain_values(point)
    if decl.center_from_registry is not None:
        center = decl.center_from_registry(reg)
    else:
        center = {n: float(reg[n]) for n in decl.coords if n in reg}
    missing = [n for n in decl.coords if n not in center]
    if missing:
        raise GainRegionRefusal(
            "center_incomplete",
            f"{decl.system}: printed_gains point "
            f"{'flagship' if decl.registry_flagship else 'benchmark'}:"
            f"{decl.registry_key} does not supply {missing}",
        )
    brackets = {n: (str(lo), str(hi)) for n, (lo, hi) in pg.brackets(point).items()}
    provenance = {
        "registry_table": "flagship" if decl.registry_flagship else "benchmark",
        "registry_key": decl.registry_key,
        "registry_values": {n: str(v) for n, v in sorted(reg.items())},
        "registry_roles": {n: g.role for n, g in sorted(point.gains.items())},
        "brackets": brackets,
        "note": point.note,
        "converted_to_k_coordinates": decl.center_from_registry is not None,
    }
    return center, provenance


# ===========================================================================
# Symbolic affinity assertion (guard R6) -- runs before any solve
# ===========================================================================

def _gf_lists_for_check(decl: GainCoordinates, values: Mapping[str, object],
                        delta0: float) -> List[Tuple[str, List[GeneralizedForm]]]:
    """The dynamics lists a run actually verifies: nominal, or both endpoints."""
    dyn = decl.build(values)
    if delta0 and float(delta0) > 0:
        verts = matched_disturbance_vertices(dyn, decl.channel, float(delta0))
        return [("endpoint_plus", verts[0]), ("endpoint_minus", verts[1])]
    return [("nominal", dyn)]


def assert_gain_affinity(decl: GainCoordinates, delta0: float = 0.0) -> dict:
    """Assert symbolically that W is affine in the DECLARED gain coordinates.

    Two independent conditions, both required by the campaign card:

      (1) structural: every declared gain enters exactly ONE dynamics term,
          with a coefficient of the form const*gain (const a nonzero number,
          free of the other gains). Every other coefficient is gain-free;
      (2) W-level: with grad(V) carried as gain-independent opaque symbols
          (V is fixed, so it cannot depend on the gains),
          W = -sum_i dV_i . f_i is linear in each declared gain and carries no
          gain-gain products.

    Raises GainRegionRefusal (kind "affinity_refused") on any violation, which
    aborts the system before a single SDP is solved.
    """
    import sympy as sp

    gains = {n: sp.Symbol(n, positive=True) for n in decl.coords}
    try:
        checked = _gf_lists_for_check(decl, gains, delta0)
    except Exception as exc:  # noqa: BLE001 -- turned into a typed refusal
        raise GainRegionRefusal(
            "affinity_refused",
            f"{decl.system}: could not build symbolic dynamics in the declared "
            f"coordinates {list(decl.coords)}: {type(exc).__name__}: {exc}",
        ) from exc

    syms = [gains[n] for n in decl.coords]
    per_gain: Dict[str, dict] = {}
    report_lists = []

    for label, dyn in checked:
        occurrences: Dict[str, List[Tuple[int, int]]] = {n: [] for n in decl.coords}
        atoms = []
        for i, f in enumerate(dyn):
            if getattr(f, "composite_terms", None):
                raise GainRegionRefusal(
                    "affinity_refused",
                    f"{decl.system} f{i}: composite GF terms are outside the "
                    f"fixed-V gain-region argument",
                )
            for t, term in enumerate(f.terms):
                coeff = sp.sympify(term.coefficient)
                try:
                    poly = sp.Poly(coeff, *syms)
                except sp.PolynomialError as exc:
                    raise GainRegionRefusal(
                        "affinity_refused",
                        f"{decl.system} f{i} term {t}: coefficient {coeff} is not "
                        f"polynomial in the declared coordinates "
                        f"{list(decl.coords)} ({exc}). Guard R6: affinity holds "
                        f"in the k-coordinates, not in every parameterization.",
                    ) from exc
                if poly.total_degree() > 1:
                    raise GainRegionRefusal(
                        "affinity_refused",
                        f"{decl.system} f{i} term {t}: coefficient {coeff} has "
                        f"total degree {poly.total_degree()} in the declared "
                        f"coordinates; W would not be affine in them.",
                    )
                for n in decl.coords:
                    d = sp.diff(coeff, gains[n])
                    if d == 0:
                        continue
                    if d.free_symbols:
                        raise GainRegionRefusal(
                            "affinity_refused",
                            f"{decl.system} f{i} term {t}: d(coefficient)/d({n}) "
                            f"= {d} is not a constant; the gain does not enter "
                            f"linearly on its own.",
                        )
                    if sp.simplify(coeff - d * gains[n]) != 0:
                        raise GainRegionRefusal(
                            "affinity_refused",
                            f"{decl.system} f{i} term {t}: coefficient {coeff} is "
                            f"not of the form const*{n}; the declared coordinate "
                            f"does not multiply this term cleanly.",
                        )
                    occurrences[n].append((i, t))
                    per_gain[n] = {"equation": i, "term": t,
                                   "coefficient": str(coeff),
                                   "d_coefficient_d_gain": float(d)}
                atoms.append((i, t, coeff))

        for n in decl.coords:
            if len(occurrences[n]) != 1:
                raise GainRegionRefusal(
                    "affinity_refused",
                    f"{decl.system}: declared coordinate {n!r} multiplies "
                    f"{len(occurrences[n])} dynamics terms {occurrences[n]} on "
                    f"{label}; the fixed-V vertex argument requires exactly one.",
                )

        # (2) W itself, with grad(V) and the x-atoms carried as opaque symbols.
        dV = {i: sp.Symbol(f"dV_{i + 1}") for i in range(len(dyn))}
        W = sp.Integer(0)
        for i, t, coeff in atoms:
            W -= dV[i] * coeff * sp.Symbol(f"phi_{i}_{t}")
        W = sp.expand(W)
        for n in decl.coords:
            if sp.diff(W, gains[n], 2) != 0:
                raise GainRegionRefusal(
                    "affinity_refused",
                    f"{decl.system}: W is not linear in {n!r} on {label}.",
                )
            if sp.diff(W, gains[n]) == 0:
                raise GainRegionRefusal(
                    "affinity_refused",
                    f"{decl.system}: declared coordinate {n!r} does not enter W "
                    f"on {label}; it is not a gain coordinate of this system.",
                )
        for a, b in itertools.combinations(decl.coords, 2):
            if sp.diff(W, gains[a], gains[b]) != 0:
                raise GainRegionRefusal(
                    "affinity_refused",
                    f"{decl.system}: W carries a {a}.{b} product on {label}; the "
                    f"certified set need not be convex in these coordinates.",
                )
        report_lists.append(label)

    return {
        "coordinates": list(decl.coords),
        "coordinate_note": decl.note,
        "checked_dynamics": report_lists,
        "delta0": float(delta0),
        "per_gain": per_gain,
        "each_gain_multiplies_exactly_one_term": True,
        "W_linear_in_each_declared_gain": True,
        "no_gain_gain_products_in_W": True,
        "method": "sympy on the dynamics built by the system modules themselves",
    }


def assert_symbolic_matches_numeric(decl: GainCoordinates,
                                    values: Mapping[str, float],
                                    rel_tol: float = 1e-12) -> dict:
    """The symbolic build, evaluated at ``values``, must reproduce the numeric one.

    Guards against a builder that branches on gain VALUES (which would make the
    symbolic affinity statement vacuous for the object actually solved).
    """
    import sympy as sp

    gains = {n: sp.Symbol(n, positive=True) for n in decl.coords}
    sym = decl.build(gains)
    num = decl.build(dict(values))
    subs = {gains[n]: sp.Float(float(values[n])) for n in decl.coords}
    worst = 0.0
    if len(sym) != len(num):
        raise GainRegionRefusal(
            "affinity_refused",
            f"{decl.system}: symbolic build has {len(sym)} equations, numeric "
            f"build has {len(num)}",
        )
    for i, (fs, fn) in enumerate(zip(sym, num)):
        if len(fs.terms) != len(fn.terms):
            raise GainRegionRefusal(
                "affinity_refused",
                f"{decl.system} f{i}: symbolic build has {len(fs.terms)} terms, "
                f"numeric build has {len(fn.terms)}",
            )
        for t, (ts, tn) in enumerate(zip(fs.terms, fn.terms)):
            if (tuple(ts.exponents) != tuple(tn.exponents)
                    or tuple(ts.term_types) != tuple(tn.term_types)):
                raise GainRegionRefusal(
                    "affinity_refused",
                    f"{decl.system} f{i} term {t}: support differs between the "
                    f"symbolic and numeric builds",
                )
            got = float(sp.sympify(ts.coefficient).subs(subs))
            want = float(tn.coefficient)
            scale = max(1.0, abs(want))
            worst = max(worst, abs(got - want) / scale)
    if worst > rel_tol:
        raise GainRegionRefusal(
            "affinity_refused",
            f"{decl.system}: symbolic build disagrees with the numeric build at "
            f"the centre by {worst:.3e} (relative), above {rel_tol:g}",
        )
    return {"worst_relative_deviation": worst, "tolerance": rel_tol}


def assert_same_support(decl: GainCoordinates, ref: List[GeneralizedForm],
                        other: List[GeneralizedForm], where: str) -> None:
    """Two gain points must give dynamics with identical support.

    A gain value that changed the term structure would break the affine
    argument silently; this makes it loud.
    """
    if len(ref) != len(other):
        raise GainRegionRefusal("support_changed",
                                f"{decl.system}: equation count changed at {where}")
    for i, (a, b) in enumerate(zip(ref, other)):
        if len(a.terms) != len(b.terms):
            raise GainRegionRefusal(
                "support_changed",
                f"{decl.system} f{i}: term count changed at {where}")
        for t, (ta, tb) in enumerate(zip(a.terms, b.terms)):
            if (tuple(ta.exponents) != tuple(tb.exponents)
                    or tuple(ta.term_types) != tuple(tb.term_types)):
                raise GainRegionRefusal(
                    "support_changed",
                    f"{decl.system} f{i} term {t}: support changed at {where}")


# ===========================================================================
# Loading the fixed V
# ===========================================================================

def _V_from_terms(terms: Sequence[dict], degree_m: int,
                  weights_r: Sequence[int]) -> GeneralizedForm:
    """Rebuild a stored V bit for bit (same convention as
    review/verify_c1_certificates._reconstruct_V and
    scripts/levant_2003_perturbed.reconstruct_V)."""
    return GeneralizedForm(
        degree_m=int(degree_m),
        weights_r=tuple(int(r) for r in weights_r),
        terms=[GFTerm(coefficient=float(t["coefficient"]),
                      exponents=tuple(Fraction(e) for e in t["exponents"]),
                      term_types=tuple(t["term_types"]))
               for t in terms],
    )


def _v_sha256(V: GeneralizedForm) -> str:
    payload = json.dumps({
        "degree_m": V.degree_m,
        "weights_r": list(V.weights_r),
        "terms": [{"coefficient": repr(float(t.coefficient)),
                   "exponents": [str(e) for e in t.exponents],
                   "term_types": list(t.term_types)} for t in V.terms],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_V(kind: str, path: Path, system: str) -> Tuple[GeneralizedForm, dict]:
    """Load the fixed V from a matrix JSON row or from a stored record.

    Supported shapes:
      * matrix: data["results"] rows (scripts/discover_gf.py output), keyed by
        the "system" field, carrying V_terms / degree_used / weights_r;
      * record: a row JSON with V_terms / degree_used / weights_r at the top
        level (the levant_2003_perturbed *_row.json and *_sparse_row.json
        records, e.g. the 25-term and 16-term order-2 V's), or an exact-gains
        record with the same fields nested under "V" as
        terms / degree_m / weights_r.
    """
    data = json.loads(Path(path).read_text())
    if kind == "matrix":
        rows = data.get("results")
        if not rows:
            raise GainRegionRefusal("v_source_invalid",
                                    f"{path}: no 'results' array (not a matrix JSON)")
        match = [r for r in rows if r.get("system") == system]
        if not match:
            raise GainRegionRefusal(
                "v_source_invalid",
                f"{path}: no row for system {system!r}; rows present: "
                f"{sorted(r.get('system') for r in rows)}")
        row = match[0]
        if not row.get("V_terms"):
            raise GainRegionRefusal("v_source_invalid",
                                    f"{path}: row {system!r} stores no V_terms")
        V = _V_from_terms(row["V_terms"], row["degree_used"], row["weights_r"])
        meta = {
            "kind": "matrix", "path": str(path), "system_row": system,
            "certified_in_source": bool(row.get("certified")),
            "degree_used": row.get("degree_used"),
            "n_terms": len(V.terms),
            "source_regularity": row.get("regularity"),
            "source_gains": None,
            "source_delta0": None,
            "source_timestamp": data.get("timestamp"),
            "source_solver": data.get("solver"),
        }
    elif kind == "record":
        if "V_terms" in data:
            V = _V_from_terms(data["V_terms"], data["degree_used"], data["weights_r"])
        elif isinstance(data.get("V"), dict) and "terms" in data["V"]:
            v = data["V"]
            V = _V_from_terms(v["terms"], v.get("degree_m") or v.get("degree_used"),
                              v["weights_r"])
        else:
            raise GainRegionRefusal(
                "v_source_invalid",
                f"{path}: no V_terms at the top level and no V.terms block")
        rec_system = data.get("system")
        if rec_system is not None and rec_system != system:
            raise GainRegionRefusal(
                "v_source_invalid",
                f"{path}: record is for system {rec_system!r} but --system is "
                f"{system!r}; refusing to certify one system's gains with "
                f"another system's V")
        meta = {
            "kind": "record", "path": str(path), "system_row": rec_system,
            "certified_in_source": bool(data.get("certified", True)),
            "degree_used": V.degree_m,
            "n_terms": len(V.terms),
            "source_regularity": None,
            "source_gains": {k: v for k, v in (data.get("gains") or {}).items()
                             if isinstance(v, (int, float))},
            "source_delta0": data.get("delta0"),
            "source_method": data.get("method"),
            "source_provenance": data.get("provenance"),
        }
    else:  # pragma: no cover -- argparse restricts the choices
        raise GainRegionRefusal("v_source_invalid", f"unknown --v-from kind {kind!r}")

    meta["v_sha256_16"] = _v_sha256(V)
    meta["V_id"] = (f"V[{system}, {meta['kind']}:{Path(path).name}, "
                    f"m={V.degree_m}, {len(V.terms)} terms, "
                    f"sha {meta['v_sha256_16']}]")
    return V, meta


# ===========================================================================
# One cell = one (gain vertex, disturbance endpoint) W check
# ===========================================================================

def _classify_sos_failure(res) -> Tuple[str, str]:
    """Typed status from an SOS result that did not certify.

    Never collapses a solver error, an unavailable solver, or a rejected Gram
    into "the gains do not work": each is reported for what it is.
    """
    attempts = [str(a) for a in (res.solver_attempts or [])]
    last = attempts[-1].lower() if attempts else ""
    status = (res.solver_status or "").lower()

    if status == "w_assembly_failed":
        return "unexpected_error", "W assembly failed on an octant"
    if status == "w_identically_zero":
        return ("solver_infeasible",
                "cleared W is identically zero on an octant (structural, not a "
                "solver decision): the strict margin cannot hold there")
    if "unavailable" in last:
        return ("unexpected_error",
                f"requested solver is not available to CVXPY here: {attempts[-1]}")
    if "cvxpy_unavailable" in last:
        return "unexpected_error", "cvxpy is not importable in this environment"
    if "empty_basis" in last:
        return "unexpected_error", "empty SOS monomial basis"
    if "monomial_outside_basis" in last or "odd_homogeneous_degree" in last:
        return ("solver_infeasible",
                f"structurally not SOS ({attempts[-1]}): no solver decision needed")
    if ":error:" in last:
        if "memory" in last or "alloc" in last:
            return "oom", f"solver raised a memory error: {attempts[-1]}"
        return "solver_unknown", f"solver raised: {attempts[-1]}"
    if "infeasible" in last and "unbounded" not in last:
        return "solver_infeasible", f"solver returned infeasible: {attempts[-1]}"
    if "optimal" in last:
        return ("solver_unknown",
                f"solver returned optimal but the recovered Gram failed the "
                f"a-posteriori validation: {attempts[-1]}")
    return ("solver_unknown",
            f"no conclusion (solver_status={res.solver_status!r}, "
            f"attempts={attempts})")


def _verify_w_cell(V: GeneralizedForm, dyn: List[GeneralizedForm],
                   weights_r: Tuple[int, ...], degree_m: int, solver: str,
                   epsilon: float) -> dict:
    """The authoritative fixed-V W check for ONE dynamics list.

    Thin wrapper over joint_sos.verify_w_given_numeric (strict, no fallback):
    the SOS check itself is never reimplemented here.
    """
    from gf_framework.verification.joint_sos import verify_w_given_numeric

    t0 = time.monotonic()
    try:
        res = verify_w_given_numeric(
            V, dyn, weights_r, degree_m, solver=solver, strict_solver=True,
            allow_solver_fallback=False, epsilon=epsilon,
        )
    except MemoryError as exc:
        return {"status": "oom", "message": f"W check out of memory: {exc}",
                "wall_time_s": time.monotonic() - t0}
    except TimeoutError as exc:
        return {"status": "timeout", "message": f"W check timed out: {exc}",
                "wall_time_s": time.monotonic() - t0}
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
        return {"status": "unexpected_error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_time_s": time.monotonic() - t0}

    if res.feasible:
        status, message = "certified", (res.message or "W > 0 on all open orthants")
    else:
        status, message = _classify_sos_failure(res)
    return {
        "status": status, "message": message,
        "w_solver_status": res.solver_status,
        "solver_used": res.solver_used,
        "solver_attempts": [str(a) for a in (res.solver_attempts or [])],
        "gram_min_eig": res.gram_min_eig,
        "gram_residual_l1": res.gram_residual_l1,
        "gram_cond": res.gram_cond,
        "wall_time_s": time.monotonic() - t0,
    }


def _cell_child(queue, V, dyn, weights_r, degree_m, solver, epsilon):
    """Entry point of the isolated cell process (--cell-timeout)."""
    try:
        queue.put(_verify_w_cell(V, dyn, weights_r, degree_m, solver, epsilon))
    except Exception as exc:  # noqa: BLE001
        queue.put({"status": "unexpected_error",
                   "message": f"child process failed: {type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "wall_time_s": 0.0})


def run_cell(V: GeneralizedForm, dyn: List[GeneralizedForm],
             weights_r: Tuple[int, ...], degree_m: int, solver: str,
             epsilon: float, cell_timeout: Optional[float] = None) -> dict:
    """One cell, optionally in an isolated process so a timeout can be enforced.

    Without --cell-timeout the check runs in-process (no spawn cost). With it,
    the cell runs in a child process that is terminated on expiry and recorded
    as status "timeout". A child that dies without a result is recorded as
    "unexpected_error": a process-level kill (a true OOM) is the external
    watchdog's job (scripts/_watchdog.py), never guessed at here.
    """
    if not cell_timeout or cell_timeout <= 0:
        return _verify_w_cell(V, dyn, weights_r, degree_m, solver, epsilon)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_cell_child,
                       args=(queue, V, dyn, weights_r, degree_m, solver, epsilon))
    t0 = time.monotonic()
    proc.start()
    proc.join(cell_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return {"status": "timeout",
                "message": f"cell exceeded --cell-timeout {cell_timeout:g} s and "
                           f"was terminated",
                "wall_time_s": time.monotonic() - t0}
    try:
        return queue.get_nowait()
    except Exception:  # noqa: BLE001 -- empty queue: the child produced nothing
        return {"status": "unexpected_error",
                "message": f"isolated cell process exited with code "
                           f"{proc.exitcode} without returning a result",
                "wall_time_s": time.monotonic() - t0}


# ===========================================================================
# Boxes and vertices
# ===========================================================================

def parse_box(spec: str, decl: GainCoordinates) -> "Dict[str, Tuple[float, float]]":
    """Parse --box "k1:1.2:1.8,k2:0.9:1.3" in the declared coordinates."""
    box: Dict[str, Tuple[float, float]] = {}
    for chunk in [c.strip() for c in spec.split(",") if c.strip()]:
        parts = chunk.split(":")
        if len(parts) != 3:
            raise GainRegionRefusal(
                "box_invalid",
                f"--box entry {chunk!r} is not NAME:LO:HI")
        name, lo_s, hi_s = parts
        name = name.strip()
        if name not in decl.coords:
            raise GainRegionRefusal(
                "box_invalid",
                f"--box names {name!r}, which is not a declared gain coordinate "
                f"of {decl.system}; declared: {list(decl.coords)}")
        try:
            lo, hi = float(lo_s), float(hi_s)
        except ValueError as exc:
            raise GainRegionRefusal("box_invalid",
                                    f"--box entry {chunk!r}: {exc}") from exc
        if hi < lo:
            raise GainRegionRefusal("box_invalid",
                                    f"--box entry {chunk!r}: hi < lo")
        if name in box:
            raise GainRegionRefusal("box_invalid",
                                    f"--box names {name!r} twice")
        box[name] = (lo, hi)
    if not box:
        raise GainRegionRefusal("box_invalid", "--box is empty")
    return box


def box_from_rel_halfwidth(center: Mapping[str, float], rel: float,
                           decl: GainCoordinates) -> "Dict[str, Tuple[float, float]]":
    return {n: (center[n] * (1.0 - rel), center[n] * (1.0 + rel))
            for n in decl.coords}


def box_vertices(box: Mapping[str, Tuple[float, float]],
                 fixed: Mapping[str, float],
                 decl: GainCoordinates) -> List[Dict[str, float]]:
    """All corners of the box, degenerate coordinates collapsed (deduplicated)."""
    names = [n for n in decl.coords if n in box]
    seen = set()
    out: List[Dict[str, float]] = []
    for combo in itertools.product(*[box[n] for n in names]):
        gains = dict(fixed)
        gains.update(dict(zip(names, combo)))
        key = tuple(gains[n] for n in decl.coords)
        if key in seen:
            continue
        seen.add(key)
        out.append({n: gains[n] for n in decl.coords})
    return out


def scale_box(box: Mapping[str, Tuple[float, float]],
              factor: float) -> "Dict[str, Tuple[float, float]]":
    """Scale a box about its own midpoint."""
    out = {}
    for n, (lo, hi) in box.items():
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out[n] = (mid - factor * half, mid + factor * half)
    return out


def rel_halfwidths(box: Mapping[str, Tuple[float, float]],
                   center: Mapping[str, float]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for n, (lo, hi) in box.items():
        c = float(center.get(n, 0.0))
        out[n] = (0.5 * (hi - lo) / abs(c)) if c else None
    return out


def center_in_interior(box: Mapping[str, Tuple[float, float]],
                       center: Mapping[str, float]) -> bool:
    return all(lo < float(center[n]) < hi for n, (lo, hi) in box.items())


def center_position(box: Mapping[str, Sequence[float]],
                    center: Mapping[str, float]) -> str:
    """Where the registered gain point sits relative to the box."""
    if all(lo == hi for lo, hi in box.values()):
        same = all(float(center[n]) == lo for n, (lo, _hi) in box.items())
        return "single_point_at_the_registered_tuning" if same else "single_point"
    if all(lo < float(center[n]) < hi for n, (lo, hi) in box.items()):
        return "interior"
    if all(lo <= float(center[n]) <= hi for n, (lo, hi) in box.items()):
        return "boundary"
    return "outside"


def certify_box(V: GeneralizedForm, decl: GainCoordinates,
                box: Mapping[str, Tuple[float, float]],
                fixed: Mapping[str, float], delta0: float, solver: str,
                epsilon: float, cell_timeout: Optional[float],
                center_dyn: List[GeneralizedForm],
                verbose: bool = True, label: str = "") -> dict:
    """Certify every (vertex x endpoint) cell of one box.

    The box verdict is "certified" only when EVERY cell certified; any other
    cell status makes the box not certified, and the per-cell typed status
    says why. A non-certified cell is a statement about this V at that vertex,
    never about the system.
    """
    vertices = box_vertices(box, fixed, decl)
    cells: List[dict] = []
    t0 = time.monotonic()
    for vi, gains in enumerate(vertices):
        dyn = decl.build(gains)
        assert_same_support(decl, center_dyn, dyn,
                            f"vertex {vi} {gains}")
        if delta0 > 0:
            try:
                verts = matched_disturbance_vertices(dyn, decl.channel,
                                                     float(delta0))
            except ValueError as exc:  # not a weighted-degree-0 channel
                raise GainRegionRefusal(
                    "endpoint_build_failed",
                    f"{decl.system}: the inclusion endpoints could not be built "
                    f"on channel {decl.channel}: {exc}") from exc
            if len(verts) != 2:
                raise GainRegionRefusal(
                    "endpoint_build_failed",
                    f"{decl.system}: expected 2 inclusion endpoints, got "
                    f"{len(verts)}")
            endpoints = list(zip(("plus", "minus"), verts))
        else:
            endpoints = [("nominal", dyn)]
        for ei, (ep_label, ep_dyn) in enumerate(endpoints):
            rec = run_cell(V, ep_dyn, decl.weights_r, V.degree_m, solver,
                           epsilon, cell_timeout)
            rec.update({
                "vertex_index": vi,
                "gains": {n: float(gains[n]) for n in decl.coords},
                "endpoint": ep_label,
                "endpoint_index": None if delta0 <= 0 else ei,
                "delta0": float(delta0),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            cells.append(rec)
            if verbose:
                gtxt = ", ".join(f"{n}={gains[n]:.6g}" for n in decl.coords)
                print(f"    [{label}] vertex {vi + 1}/{len(vertices)} ({gtxt}) "
                      f"endpoint {ep_label}: {rec['status']} "
                      f"({rec['wall_time_s']:.2f} s)", flush=True)

    counts = {s: sum(1 for c in cells if c["status"] == s) for s in TYPED_STATUSES}
    return {
        "box": {n: [float(lo), float(hi)] for n, (lo, hi) in box.items()},
        "fixed_gains": {n: float(v) for n, v in fixed.items() if n not in box},
        "n_vertices": len(vertices),
        "n_endpoints_per_vertex": 2 if delta0 > 0 else 1,
        "n_cells": len(cells),
        "n_cells_expected": len(vertices) * (2 if delta0 > 0 else 1),
        "certified": all(c["status"] == "certified" for c in cells),
        "status_counts": counts,
        "cells": cells,
        "wall_time_s": time.monotonic() - t0,
    }


# ===========================================================================
# Modes
# ===========================================================================

def run_box(ctx: dict) -> dict:
    box = ctx["box"]
    res = certify_box(ctx["V"], ctx["decl"], box, ctx["center"], ctx["delta0"],
                      ctx["solver"], ctx["epsilon"], ctx["cell_timeout"],
                      ctx["center_dyn"], label="box")
    res["rel_halfwidths"] = rel_halfwidths(box, ctx["center"])
    res["center_in_interior"] = center_in_interior(box, ctx["center"])
    res["center_position"] = center_position(box, ctx["center"])
    return {"mode": "box", "result": res}


def run_grow(ctx: dict) -> dict:
    """Outward ladder then bisection on the scale factor about the box midpoint.

    Every shell is recorded, certified or not (binding rule 5). If the base box
    itself does not certify, the search turns INWARD (bisecting between 0 and
    1) and says so, rather than reporting nothing.
    """
    base = ctx["box"]
    grow_max = float(ctx["grow_max"])
    tol = float(ctx["grow_tol"])
    shells: List[dict] = []

    def shell(factor: float) -> bool:
        b = scale_box(base, factor)
        res = certify_box(ctx["V"], ctx["decl"], b, ctx["center"], ctx["delta0"],
                          ctx["solver"], ctx["epsilon"], ctx["cell_timeout"],
                          ctx["center_dyn"], label=f"scale {factor:.4g}")
        res["scale"] = factor
        res["rel_halfwidths"] = rel_halfwidths(b, ctx["center"])
        res["center_in_interior"] = center_in_interior(b, ctx["center"])
        res["center_position"] = center_position(b, ctx["center"])
        shells.append(res)
        print(f"  scale {factor:.4g}: "
              f"{'certified' if res['certified'] else 'NOT certified'} "
              f"({res['status_counts']})", flush=True)
        return res["certified"]

    def blocked() -> bool:
        """A crash/timeout/oom in the last shell stops the search: it is not
        evidence that the shell is outside the certified set."""
        return any(c["status"] in ("timeout", "oom", "unexpected_error")
                   for c in shells[-1]["cells"])

    BLOCKED_NOTE = ("the search stopped early: a shell produced a "
                    "timeout/oom/unexpected_error cell, which is not evidence "
                    "about the gains")
    note = ""
    base_ok = shell(1.0)
    if blocked():
        return _grow_payload(base, direction="outward", grow_max=grow_max, tol=tol,
                             shells=shells, note=BLOCKED_NOTE)
    if base_ok:
        direction = "outward"
        lo, hi = 1.0, None
        factor = 2.0
        while factor <= grow_max + 1e-12:
            ok = shell(factor)
            if blocked():
                note = BLOCKED_NOTE
                break
            if not ok:
                hi = factor
                break
            lo = factor
            factor *= 2.0
        if hi is None and not note:
            note = (f"every tested shell up to the --grow-max cap {grow_max:g} "
                    f"certified; the reported box is the cap, not a boundary")
    else:
        direction = "inward"
        note = ("the base box did not certify, so the search bisected INWARD "
                "between scale 0 and 1")
        lo, hi = 0.0, 1.0

    if hi is not None and not blocked():
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            ok = shell(mid)
            if blocked():
                note = BLOCKED_NOTE
                break
            if ok:
                lo = mid
            else:
                hi = mid

    return _grow_payload(base, direction=direction, grow_max=grow_max, tol=tol,
                         shells=shells, note=note)


def _grow_payload(base, direction: str, grow_max: float, tol: float,
                  shells: List[dict], note: str) -> dict:
    """Assemble the --grow result: largest certified box, first failed shell,
    and every shell that was run (binding rule 5)."""
    certified_shell = None
    for s in shells:
        if s["certified"] and (certified_shell is None
                               or s["scale"] > certified_shell["scale"]):
            certified_shell = s
    failed_shells = [s for s in shells if not s["certified"]]
    first_failed = min(failed_shells, key=lambda s: s["scale"]) if failed_shells else None

    return {
        "mode": "grow",
        "result": {
            "base_box": {n: [float(a), float(b)] for n, (a, b) in base.items()},
            "direction": direction,
            "grow_max": grow_max,
            "grow_tol": tol,
            "n_shells": len(shells),
            "largest_certified_scale": (certified_shell["scale"]
                                        if certified_shell else None),
            "largest_certified_box": (certified_shell["box"]
                                      if certified_shell else None),
            "largest_certified_rel_halfwidths": (
                certified_shell["rel_halfwidths"] if certified_shell else None),
            "first_failed_scale": first_failed["scale"] if first_failed else None,
            "first_failed_box": first_failed["box"] if first_failed else None,
            "first_failed_status_counts": (first_failed["status_counts"]
                                           if first_failed else None),
            "note": note,
            "shells": shells,
        },
    }


def run_max_delta0(ctx: dict) -> dict:
    """Bisect the largest certified delta0 for the fixed V at a fixed gain point.

    Both inclusion endpoints are checked at EVERY trial (the record carries the
    endpoint count so this is auditable, not asserted).
    """
    decl = ctx["decl"]
    point = {n: float(ctx["center"][n]) for n in decl.coords}
    box = {n: (point[n], point[n]) for n in decl.coords}
    tol = float(ctx["delta0_tol"])
    hi_cap = float(ctx["delta0_hi"])
    trials: List[dict] = []

    def trial(d0: float) -> bool:
        res = certify_box(ctx["V"], decl, box, point, d0, ctx["solver"],
                          ctx["epsilon"], ctx["cell_timeout"], ctx["center_dyn"],
                          label=f"delta0 {d0:.4g}")
        res["delta0"] = d0
        trials.append(res)
        print(f"  delta0={d0:.4f}: "
              f"{'certified' if res['certified'] else 'NOT certified'} "
              f"({res['status_counts']})", flush=True)
        return res["certified"]

    def blocked() -> bool:
        return any(c["status"] in ("timeout", "oom", "unexpected_error")
                   for c in trials[-1]["cells"])

    BLOCKED_NOTE = ("the bisection stopped early on a "
                    "timeout/oom/unexpected_error trial, which is not evidence "
                    "about the disturbance bound")
    note = ""
    if not trial(0.0):
        return {
            "mode": "max_delta0",
            "result": {
                "gain_point": point, "delta0_max": None,
                "delta0_first_failed": 0.0,
                "note": "the fixed V did not certify the NOMINAL dynamics at "
                        "this gain point, so no disturbance bound is claimed",
                "tol": tol, "hi_cap": hi_cap,
                "n_trials": len(trials),
                "n_endpoint_cells": sum(t["n_cells"] for t in trials),
                "trials": trials,
            },
        }

    lo, hi = 0.0, None
    probe = min(0.25, hi_cap)
    while True:
        ok = trial(probe)
        if blocked():
            note = BLOCKED_NOTE
            break
        if not ok:
            hi = probe
            break
        lo = probe
        if probe >= hi_cap:
            note = (f"every tested delta0 up to the cap {hi_cap:g} certified; "
                    f"the reported value is the search cap, not a limit of V")
            break
        probe = min(probe * 2.0, hi_cap)

    if hi is not None and not blocked():
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            ok = trial(mid)
            if blocked():
                note = BLOCKED_NOTE
                break
            if ok:
                lo = mid
            else:
                hi = mid

    return {
        "mode": "max_delta0",
        "result": {
            "gain_point": point,
            "delta0_max": lo,
            "delta0_first_failed": hi,
            "tol": tol,
            "hi_cap": hi_cap,
            "note": note,
            "n_trials": len(trials),
            "n_endpoint_cells": sum(t["n_cells"] for t in trials),
            "endpoints_per_trial": [t["n_endpoints_per_vertex"] for t in trials],
            "cells_per_positive_delta0_trial": sorted(
                {t["n_cells"] for t in trials if t["delta0"] > 0}),
            "n_positive_delta0_trials": sum(1 for t in trials if t["delta0"] > 0),
            "trials": trials,
        },
    }


# ===========================================================================
# Pre-registered sentence templates (block E9), filled from the data
# ===========================================================================

def _fmt_box(box: Mapping[str, Sequence[float]]) -> str:
    return "{" + ", ".join(
        f"{n} in [{lo:.6g}, {hi:.6g}]" for n, (lo, hi) in box.items()) + "}"


def statements(payload: dict) -> List[str]:
    """Fill the block's pre-registered templates. Regions in k-coordinates,
    no em dashes, no claim beyond the tested set."""
    out: List[str] = []
    vid = payload["V_source"]["V_id"]
    d0 = payload["delta0"]
    coord_note = ("Gain coordinates: "
                  + ", ".join(payload["affinity"]["coordinates"])
                  + " (k-coordinates; " + payload["affinity"]["coordinate_note"] + ")")
    out.append(coord_note)
    mode = payload["mode"]
    res = payload["result"]

    if mode == "box":
        if res["certified"]:
            where = {
                "interior": " the registered tuning lies in its interior.",
                "boundary": " the registered tuning lies on its boundary, so no "
                            "interior claim is made.",
                "outside": " the registered tuning lies outside the box.",
                "single_point_at_the_registered_tuning":
                    " the box is the registered tuning itself, a single point.",
                "single_point": " the box is a single gain point.",
            }[res["center_position"]]
            out.append(f"For the fixed function {vid}, the entire gain box "
                       f"{_fmt_box(res['box'])} is certified at delta0 = {d0:g};"
                       + where)
        else:
            rel = [v for v in (res.get("rel_halfwidths") or {}).values()
                   if v is not None]
            relmax = max(rel) if rel else None
            counts = res["status_counts"]
            inconclusive = sum(counts.get(s, 0) for s in
                               ("timeout", "oom", "unexpected_error"))
            head = (f"For the tested {vid}, box parameterization "
                    f"{_fmt_box(res['box'])}, and disturbance level "
                    f"delta0 = {d0:g}, the box was not certified at every "
                    f"vertex"
                    + (f" (relative half-widths up to {relmax:.4g})"
                       if relmax else "")
                    + f"; the per-cell statuses were {counts}.")
            if inconclusive:
                out.append(
                    head + f" {inconclusive} of {res['n_cells']} cells did not "
                    f"reach a conclusion (timeout, oom, or unexpected_error). "
                    f"Those cells are statements about this run and this "
                    f"environment, not about the gains, so no region claim is "
                    f"made from this run.")
            else:
                out.append(
                    head + " Transfer of this fixed V across gains is limited "
                    "in this setting. A non-certified vertex is a statement "
                    "about this V at that vertex, not about the system.")
    elif mode == "grow":
        if res["largest_certified_box"]:
            rel = [v for v in (res["largest_certified_rel_halfwidths"] or {}).values()
                   if v is not None]
            relmax = max(rel) if rel else None
            out.append(
                f"For the fixed function {vid}, the entire gain box "
                f"{_fmt_box(res['largest_certified_box'])} is certified at "
                f"delta0 = {d0:g}"
                + (f" (relative half-widths up to {relmax:.4g})" if relmax else "")
                + ".")
        else:
            out.append(f"For the tested {vid} and box parameterization, no shell "
                       f"of the tested ladder certified at delta0 = {d0:g}.")
        if res["first_failed_box"]:
            out.append(
                f"The first shell that did not certify was "
                f"{_fmt_box(res['first_failed_box'])} at scale "
                f"{res['first_failed_scale']:.4g}, with per-cell statuses "
                f"{res['first_failed_status_counts']}. The certified region did "
                f"not extend beyond that shell under the tested procedure.")
        if res.get("note"):
            out.append(f"Search note: {res['note']}")
    elif mode == "max_delta0":
        if res["delta0_max"] is None:
            out.append(f"For the tested {vid} at gain point "
                       f"{res['gain_point']}, the nominal dynamics did not "
                       f"certify, so no disturbance bound is reported.")
        else:
            out.append(
                f"For the fixed function {vid} at the gain point "
                f"{res['gain_point']}, the largest disturbance bound certified "
                f"by bisection (tolerance {res['tol']:g}, both inclusion "
                f"endpoints per trial) is delta0 = {res['delta0_max']:.4g}"
                + (f"; the next tested value {res['delta0_first_failed']:.4g} "
                   f"did not certify." if res["delta0_first_failed"] is not None
                   else "."))
        if res.get("note"):
            out.append(f"Search note: {res['note']}")
    if payload.get("note"):
        out.append(payload["note"])
    return out


# ===========================================================================
# Output
# ===========================================================================

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _solver_version(solver_name: str, env: dict) -> str:
    s = (solver_name or "").upper()
    if s.startswith("MOSEK"):
        return env.get("mosek_version", "unknown")
    return env.get("packages", {}).get(s.lower(), "unknown")


def provenance(argv: Sequence[str], solver: str) -> dict:
    env = collect_env()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": env.get("git_sha"),
        "git_dirty": env.get("git_dirty"),
        "command_line": " ".join([sys.argv[0]] + list(argv)),
        "command_argv": [sys.executable, sys.argv[0]] + list(argv),
        "solver_name": solver,
        "solver_version": _solver_version(solver, env),
        "env": env,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _peak_rss_bytes():
    """Cheap OS-maintained peak RSS (never an active watchdog); same helper
    contract as robust_delta_sweep.py."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)
        return int(peak) if peak else int(mi.rss)
    except Exception:  # noqa: BLE001
        return None


def markdown(payload: dict) -> str:
    prov = payload.get("provenance", {})
    lines = [
        f"# {BLOCK} fixed-V gain region: {payload['system']} "
        f"({payload['mode']}, delta0 = {payload['delta0']:g})",
        "",
        f"- timestamp (UTC): {prov.get('timestamp_utc')}",
        f"- git SHA: `{prov.get('git_sha')}` (dirty: {prov.get('git_dirty')})",
        f"- command: `{prov.get('command_line')}`",
        f"- solver: {prov.get('solver_name')} {prov.get('solver_version')} "
        f"(strict, no fallback)",
        f"- V: {payload['V_source']['V_id']}",
        f"- V stage (gain-independent): {payload['v_stage']['status']}",
        "",
        "## Statements",
        "",
    ]
    for s in payload.get("statements", []):
        lines.append(f"- {s}")
    lines += ["", "## Cells", "", "| cell | gains | endpoint | status | s |",
              "|---|---|---|---|---|"]

    def emit(cells):
        for c in cells:
            gtxt = ", ".join(f"{n}={v:.6g}" for n, v in c["gains"].items())
            lines.append(
                f"| {c['vertex_index']} | {gtxt} | {c['endpoint']} | "
                f"{c['status']} | {c['wall_time_s']:.2f} |")

    res = payload["result"]
    if payload["mode"] == "box":
        emit(res["cells"])
    elif payload["mode"] == "grow":
        for s in res["shells"]:
            lines.append(f"| scale {s['scale']:.4g} | | | "
                         f"{'certified' if s['certified'] else 'not certified'} | "
                         f"{s['wall_time_s']:.1f} |")
            emit(s["cells"])
    else:
        for t in res["trials"]:
            lines.append(f"| delta0 {t['delta0']:.4g} | | | "
                         f"{'certified' if t['certified'] else 'not certified'} | "
                         f"{t['wall_time_s']:.1f} |")
            emit(t["cells"])
    lines.append("")
    return "\n".join(lines) + "\n"


def _json_default(obj):
    """Serialize numpy scalars and Fractions rather than losing a record.

    A diagnostic that cannot be encoded must never cost us the artifact; it is
    converted, and anything genuinely unknown is stringified with its type.
    """
    for cast in (float, int):
        try:
            return cast(obj)
        except (TypeError, ValueError):
            continue
    return f"{type(obj).__name__}:{obj}"


def write_outputs(payload: dict, output_dir: Path) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    what = f"{BLOCK}_gain_region_{payload['mode']}_{payload['system']}"
    json_path = output_dir / f"{what}_{stamp}.json"
    md_path = output_dir / f"{what}_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default),
                         encoding="utf-8")
    md_path.write_text(markdown(payload), encoding="utf-8")
    return json_path, md_path


def write_refusal(kind: str, message: str, argv: Sequence[str], solver: str,
                  system: Optional[str], output_dir: Optional[Path]) -> Optional[Path]:
    """A refusal is recorded too (binding rule 5: every run is recorded)."""
    if output_dir is None:
        return None
    payload = {
        "block": BLOCK, "refused": True, "refusal_kind": kind,
        "message": message, "system": system, "solver": solver,
        "provenance": provenance(argv, solver),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{BLOCK}_gain_region_REFUSED_{system or 'none'}_{_utc_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default),
                    encoding="utf-8")
    return path


# ===========================================================================
# The run
# ===========================================================================

def v_stage(V: GeneralizedForm, solver: str, epsilon: float) -> dict:
    """The gain-independent half of the gate: V > 0 and the C1 classification.

    V does not depend on the gains, so this is run ONCE per invocation and its
    verdict applies to every vertex. Same entry points as the strict gate
    (verify_gf_sos_full includes the necessary min_V_on_sphere pre-check;
    classify_smoothness decides the Lemma-5 scope).
    """
    from gf_framework.verification.joint_sos import verify_gf_sos_full
    from gf_framework.verification.verifier import classify_smoothness

    smooth = classify_smoothness(V)
    t0 = time.monotonic()
    try:
        res = verify_gf_sos_full(V, epsilon=epsilon, solver=solver,
                                 strict_solver=True, allow_solver_fallback=False,
                                 verbose=False)
    except MemoryError as exc:
        return {"status": "oom", "message": str(exc), "regularity": smooth["regularity"],
                "wall_time_s": time.monotonic() - t0}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unexpected_error",
                "message": f"{type(exc).__name__}: {exc}",
                "regularity": smooth["regularity"],
                "wall_time_s": time.monotonic() - t0}
    status = "certified" if res.feasible else _classify_sos_failure(res)[0]
    return {
        "status": status,
        "message": res.message,
        "solver_status": res.solver_status,
        "solver_attempts": [str(a) for a in (res.solver_attempts or [])],
        "gram_min_eig": res.gram_min_eig,
        "gram_residual_l1": res.gram_residual_l1,
        "epsilon_used": res.epsilon_used,
        "regularity": smooth["regularity"],
        "lemma5_in_scope": smooth["regularity"] == "C1",
        "wall_time_s": time.monotonic() - t0,
        "note": "gain-independent: V does not depend on the gains, so this "
                "stage is run once and applies to every vertex",
    }


def resolve_center(decl: GainCoordinates, args, v_meta: dict) -> Tuple[Dict[str, float], dict]:
    """Box centre in the declared coordinates, plus a cross-check against the V.

    The centre comes from the printed-gain registry unless --gains overrides
    it. When the V source records the gains it was designed at, they must agree
    with the centre: certifying a box around a gain point the V was not
    designed for is a silent category error, so it is refused rather than
    warned about.
    """
    if args.gains:
        center = {}
        for chunk in [c.strip() for c in args.gains.split(",") if c.strip()]:
            if "=" not in chunk:
                raise GainRegionRefusal("gains_invalid",
                                        f"--gains entry {chunk!r} is not NAME=VALUE")
            name, val = chunk.split("=", 1)
            name = name.strip()
            if name not in decl.coords:
                raise GainRegionRefusal(
                    "gains_invalid",
                    f"--gains names {name!r}, not a declared coordinate of "
                    f"{decl.system}; declared: {list(decl.coords)}")
            center[name] = float(val)
        missing = [n for n in decl.coords if n not in center]
        if missing:
            raise GainRegionRefusal(
                "gains_invalid",
                f"--gains must give every declared coordinate; missing {missing}")
        prov = {"source": "--gains (explicit override)"}
    else:
        center, prov = registry_center(decl)
        prov["source"] = "printed_gains registry"

    src = v_meta.get("source_gains") or {}
    compared, skipped = {}, []
    for name, value in src.items():
        if name in decl.coords:
            compared[name] = float(value)
        else:
            skipped.append(name)
    mismatch = {n: (v, center[n]) for n, v in compared.items()
                if abs(v - center[n]) > 1e-9 * max(1.0, abs(v))}
    if mismatch and not args.allow_center_mismatch:
        raise GainRegionRefusal(
            "center_mismatch",
            f"{decl.system}: the V source records gains {compared} but the box "
            f"centre is {center} (differing: {mismatch}). Pass --gain-point "
            f"flagship / --gains to centre the box on the gain point this V was "
            f"designed at, or --allow-center-mismatch to proceed deliberately.")
    prov["v_source_gains_compared"] = compared
    prov["v_source_gains_not_declared_coordinates"] = sorted(skipped)
    prov["center_mismatch_allowed"] = bool(args.allow_center_mismatch)
    prov["center_mismatch"] = {n: list(v) for n, v in mismatch.items()}
    return center, prov


def run(args, argv: Sequence[str]) -> dict:
    solver = args.solver.upper()
    if solver == "MOSEK_DIRECT":
        raise GainRegionRefusal(
            "unsupported_solver_encoding",
            "MOSEK_DIRECT is the direct Task-API ENCODING of the DESIGN SDP "
            "(JointSOSOptimizer._find_lyapunov_direct). The fixed-V W check "
            "used here has no direct encoding, so this driver refuses rather "
            "than silently running the CVXPY encoding under a MOSEK_DIRECT "
            "label. Use --solver MOSEK for the same solver through CVXPY, or "
            "--solver CLARABEL for the independent referee.")
    if solver not in ("MOSEK", "CLARABEL"):
        raise GainRegionRefusal("unsupported_solver",
                                f"unsupported --solver {args.solver!r}")

    decl = SYSTEMS[args.system]
    if args.gain_point:
        decl = replace(decl, registry_flagship=args.gain_point == "flagship")
    delta0 = float(args.delta0)
    if delta0 < 0:
        raise GainRegionRefusal("delta0_invalid", "--delta0 must be >= 0")

    V, v_meta = load_V(args.v_from[0], Path(args.v_from[1]), args.system)
    center, center_prov = resolve_center(decl, args, v_meta)

    # ---- guard R6, before any solve -------------------------------------
    affinity = assert_gain_affinity(decl, delta0 if args.mode != "max_delta0"
                                    else max(delta0, args.delta0_hi))
    affinity["symbolic_vs_numeric"] = assert_symbolic_matches_numeric(decl, center)
    print(f"affinity asserted in {list(decl.coords)} for {decl.system} "
          f"({', '.join(affinity['checked_dynamics'])})", flush=True)

    center_dyn = decl.build(center)

    # ---- the box (or the fixed point for --max-delta0) ------------------
    # Nothing is ever silently ignored: a flag that cannot act in the chosen
    # mode is a refusal, not a no-op.
    if args.box and args.rel_halfwidth is not None:
        raise GainRegionRefusal(
            "mode_conflict",
            "--box and --rel-halfwidth both given; pass exactly one")
    if args.mode == "max_delta0":
        ignored = [flag for flag, value in (("--box", args.box),
                                            ("--rel-halfwidth", args.rel_halfwidth))
                   if value is not None]
        if delta0 > 0:
            ignored.append("--delta0")
        if ignored:
            raise GainRegionRefusal(
                "mode_conflict",
                f"--max-delta0 bisects the disturbance bound at a single gain "
                f"point, so {ignored} would have no effect; use --gains to move "
                f"the gain point, --delta0-hi to move the search cap")
        box = None
    elif args.box:
        box = parse_box(args.box, decl)
    elif args.rel_halfwidth is not None:
        box = box_from_rel_halfwidth(center, float(args.rel_halfwidth), decl)
    else:
        raise GainRegionRefusal(
            "box_missing",
            "one of --box, --rel-halfwidth, or --max-delta0 is required")
    if box is not None:
        n_vertices = 2 ** sum(1 for n in box if box[n][0] != box[n][1])
        if n_vertices > args.max_vertices:
            raise GainRegionRefusal(
                "too_many_vertices",
                f"{n_vertices} vertices exceeds --max-vertices "
                f"{args.max_vertices}; reduce the number of box coordinates")

    ctx = {
        "V": V, "decl": decl, "box": box, "center": center, "delta0": delta0,
        "solver": solver, "epsilon": float(args.epsilon),
        "cell_timeout": args.cell_timeout, "center_dyn": center_dyn,
        "grow_max": args.grow_max, "grow_tol": args.grow_tol,
        "delta0_hi": args.delta0_hi, "delta0_tol": args.delta0_tol,
    }

    vs = ({"status": "skipped",
           "message": "--no-v-stage: V > 0 was not re-checked in this run, so "
                      "the emitted claim covers the W stage only",
           "regularity": None, "lemma5_in_scope": None}
          if args.no_v_stage else v_stage(V, solver, float(args.epsilon)))
    print(f"V stage (gain-independent): {vs['status']}", flush=True)

    t0 = time.monotonic()
    if args.mode == "box":
        out = run_box(ctx)
    elif args.mode == "grow":
        out = run_grow(ctx)
    else:
        out = run_max_delta0(ctx)

    payload = {
        "block": BLOCK,
        "driver": "scripts/gain_region_fixedV.py",
        "system": args.system,
        "mode": out["mode"],
        "delta0": delta0,
        "epsilon": float(args.epsilon),
        "solver": solver,
        "solver_policy": "strict_solver=True, allow_solver_fallback=False "
                         "(no fallback anywhere)",
        "weights_r": list(decl.weights_r),
        "channel": decl.channel,
        "degree_m": V.degree_m,
        "coordinates": list(decl.coords),
        "coordinate_system": "k-coordinates (each declared gain multiplies "
                             "exactly one dynamics term)",
        "center": {n: float(v) for n, v in center.items()},
        "center_provenance": center_prov,
        "V_source": v_meta,
        "affinity": affinity,
        "v_stage": vs,
        "verification_entry_point":
            "gf_framework.verification.joint_sos.verify_w_given_numeric "
            "(strict, no fallback) per (gain vertex x inclusion endpoint) cell; "
            "gf_framework.verification.joint_sos.verify_gf_sos_full plus "
            "verifier.classify_smoothness once for the gain-independent V stage",
        "typed_statuses": list(TYPED_STATUSES),
        "claim_scope":
            "Regions are stated in k-coordinates for THIS fixed V, this box "
            "parameterization, this degree, and this disturbance level. A cell "
            "that did not certify is a statement about this V at that vertex, "
            "never about the stability of the system.",
        "result": out["result"],
        "wall_time_s": time.monotonic() - t0,
    }
    payload["statements"] = statements(payload)
    payload["provenance"] = provenance(argv, solver)
    return payload


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/gain_region_fixedV.py",
        description=(
            "Fixed-V certified gain regions (C3, block E9). For a FIXED V the "
            "cleared W is affine in the declared k-coordinates and the SOS "
            "constraints are convex, so SOS-PASS at every vertex of a gain box "
            "(at BOTH inclusion endpoints per vertex when --delta0 > 0) "
            "certifies the whole box. Affinity is asserted symbolically in the "
            "declared coordinates before any solve (guard R6); a violation is a "
            "typed refusal. Regions are reported in k-coordinates."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--system", required=True, choices=sorted(SYSTEMS),
                   help="System whose declared affine gain coordinates are used.")
    p.add_argument("--v-from", required=True, nargs=2, metavar=("KIND", "PATH"),
                   help="Where the FIXED V comes from: 'matrix <gf_matrix.json>' "
                        "(row selected by --system) or 'record <row.json>' (the "
                        "levant_2003_perturbed row / sparse-row records, e.g. "
                        "the 25-term and 16-term order-2 V's).")
    p.add_argument("--output-dir", required=True,
                   help="REQUIRED. Campaign block directory for the "
                        "E9_gain_region_*_<UTC>.json/.md outputs.")
    p.add_argument("--solver", default="MOSEK",
                   choices=["MOSEK", "MOSEK_DIRECT", "CLARABEL"],
                   help="Solver for every SOS check, with NO silent fallback. "
                        "MOSEK_DIRECT names the direct Task-API encoding, which "
                        "exists only in the design path: requesting it here is a "
                        "typed refusal, never a silent switch to CVXPY MOSEK.")
    p.add_argument("--delta0", type=float, default=0.0,
                   help="Matched-disturbance bound. > 0 checks BOTH inclusion "
                        "endpoints f +/- delta0.e_ch at every gain vertex.")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help=f"Strict SOS margin (default {DEFAULT_EPSILON:g}).")

    g = p.add_argument_group("region")
    g.add_argument("--box", default=None,
                   help='Axis-aligned box in the declared coordinates, e.g. '
                        '"k1:1.2:1.8,k2:0.9:1.3". Coordinates left out are fixed '
                        'at the centre value.')
    g.add_argument("--rel-halfwidth", type=float, default=None,
                   help="Alternative to --box: a box of this relative half-width "
                        "around the registered gain point, in every coordinate.")
    g.add_argument("--gains", default=None,
                   help='Explicit gain point, e.g. "k1=1.5,k2=1.1". Overrides the '
                        'printed-gain registry centre.')
    g.add_argument("--gain-point", choices=["benchmark", "flagship"], default=None,
                   help="Which printed_gains table supplies the centre (default: "
                        "the table the system is registered in).")
    g.add_argument("--allow-center-mismatch", action="store_true",
                   help="Proceed even when the V source records different gains "
                        "than the box centre (default: refuse).")
    g.add_argument("--max-vertices", type=int, default=1024,
                   help="Refuse boxes with more than this many vertices.")

    m = p.add_argument_group("mode")
    m.add_argument("--grow", dest="mode", action="store_const", const="grow",
                   help="Outward bisection: scale the base box about its midpoint "
                        "(doubling ladder, then bisection) and report the largest "
                        "certified box and the first failed shell. Every shell is "
                        "recorded.")
    m.add_argument("--max-delta0", dest="mode", action="store_const",
                   const="max_delta0",
                   help="Bisect the largest certified delta0 for the fixed V at a "
                        "fixed gain point (E9.e), both endpoints per trial.")
    m.add_argument("--grow-max", type=float, default=8.0,
                   help="Cap on the scale factor for --grow (default 8).")
    m.add_argument("--grow-tol", type=float, default=0.05,
                   help="Scale-factor tolerance for --grow (default 0.05).")
    m.add_argument("--delta0-hi", type=float, default=2.0,
                   help="Search cap for --max-delta0 (default 2).")
    m.add_argument("--delta0-tol", type=float, default=0.005,
                   help="Bisection tolerance for --max-delta0 (default 0.005).")

    r = p.add_argument_group("runtime")
    r.add_argument("--cell-timeout", type=float, default=None,
                   help="Per-cell wall-clock budget in seconds. When set, each "
                        "cell runs in an isolated process that is terminated on "
                        "expiry and recorded with status 'timeout'.")
    r.add_argument("--no-v-stage", action="store_true",
                   help="Skip the gain-independent V > 0 stage (the emitted "
                        "claim then covers the W stage only).")
    r.add_argument("--dry-run", action="store_true",
                   help="Assert affinity and print the resolved plan without "
                        "solving anything.")
    p.set_defaults(mode="box")
    return p


def describe_plan(args) -> dict:
    """The resolved plan and the affinity report, WITHOUT any solve."""
    decl = SYSTEMS[args.system]
    if args.gain_point:
        decl = replace(decl, registry_flagship=args.gain_point == "flagship")
    V, v_meta = load_V(args.v_from[0], Path(args.v_from[1]), args.system)
    center, center_prov = resolve_center(decl, args, v_meta)
    affinity = assert_gain_affinity(decl, float(args.delta0))
    affinity["symbolic_vs_numeric"] = assert_symbolic_matches_numeric(decl, center)
    box = None
    if args.mode != "max_delta0":
        if args.box:
            box = parse_box(args.box, decl)
        elif args.rel_halfwidth is not None:
            box = box_from_rel_halfwidth(center, float(args.rel_halfwidth), decl)
    return {
        "block": BLOCK, "dry_run": True, "system": args.system,
        "mode": args.mode, "solver": args.solver.upper(),
        "delta0": float(args.delta0), "coordinates": list(decl.coords),
        "center": center, "center_provenance": center_prov,
        "box": ({n: list(v) for n, v in box.items()} if box else None),
        "n_vertices": (len(box_vertices(box, center, decl)) if box else 1),
        "n_endpoints_per_vertex": 2 if float(args.delta0) > 0 else 1,
        "V_source": v_meta, "affinity": affinity,
        "note": "dry run: nothing was solved; this is the resolved plan and the "
                "symbolic affinity assertion only.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.dry_run:
        try:
            print(json.dumps(describe_plan(args), indent=2))
        except GainRegionRefusal as exc:
            print(f"REFUSING [{exc.kind}]: {exc.message}", file=sys.stderr)
            return REFUSAL_EXIT_CODE
        return 0

    try:
        payload = run(args, argv)
    except GainRegionRefusal as exc:
        path = write_refusal(exc.kind, exc.message, argv, args.solver.upper(),
                             args.system, out_dir)
        print(f"REFUSING [{exc.kind}]: {exc.message}", file=sys.stderr)
        if path:
            print(f"wrote {path}")
        return REFUSAL_EXIT_CODE

    json_path, md_path = write_outputs(payload, out_dir)
    print("")
    for s in payload["statements"]:
        print(f"* {s}")
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
