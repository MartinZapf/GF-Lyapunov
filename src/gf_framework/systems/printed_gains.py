"""Printed-gain registry: one source of truth for the exact rational gains.

Campaign item C20 (docs/papers/big_draft/PRE_WRITING_CAMPAIGN.md, E1.3).

Why this module exists
----------------------
The June exact re-proof (``review/reverify_exact.py``) rebuilt every system
spec from its FLOAT defaults, so the object it proved carried the float64
IMAGES of the gains, not the printed values: 2.7 entered as the dyadic
Fraction(2.7) = 3039929748475085/1125899906842624, never as 27/10, and 1.1
entered as 2476979795053773/2251799813685248, never as 11/10. That is the
caveat behind footnote 1 of
``docs/papers/big_draft/CANONICAL_STATE.md``. The statement "certified at the
published gains" is only true once the gains enter the exact arithmetic as the
printed rationals themselves.

``scripts/exact_gate_standard_gains.py`` already does this correctly for the
four flagship rows. This registry generalizes it to the full 12-system
benchmark matrix and becomes the single place where a printed gain value is
written down.

What is registered
------------------
Per system, every parameter of the ``create_*_spec`` factory that influences
the dynamics (``delta0`` is excluded: it is the disturbance bound, supplied
per run, not a gain), with

* ``PrintedGain``: the exact ``Fraction`` of the printed decimal. Invariant,
  asserted in ``tests/test_printed_gains.py``: ``float(value)`` equals the
  float default the system module uses, bit for bit. So the printed rational
  and the float image name the same published number; only the arithmetic they
  are proved in differs.
* ``BracketedGain``: a genuinely irrational gain, registered by the algebraic
  identity ``value ** power_k == power_target`` plus a float seed. The exact
  object is then a rational interval [lo, hi] with ``lo**k < q < hi**k``
  decided in Q (``decimal_bracket``). With V fixed, each octant target is
  affine in each gain, so exact-PASS at both endpoints certifies every gain in
  [lo, hi], and with j bracketed gains all 2**j corners of the box are run.
  This is the bracket convention of ``exact_gate_standard_gains.py``, moved
  here unchanged.

Two gain points per system are possible and are kept apart deliberately:
``BENCHMARK_GAIN_POINTS`` holds the module-default point that produced the
benchmark matrix (levant3d: (3, 2, 11/10)), ``FLAGSHIP_GAIN_POINTS`` holds the
published standard tunings of the differentiator rows (levant3d:
(2, (3/2)sqrt(2), 11/10)). They are different gains, not different encodings of
one gain.

Rationalizing the dynamics
--------------------------
Passing ``Fraction`` gains into the factories is not sufficient by itself.
``create_pidsmc_spec`` scales its gains by ``delta ** (2/3)`` and
``delta ** (1/2)``; a fractional float power of a ``Fraction`` returns a float,
so lambda1 and lambda2 silently fall back to float64 even when exact values go
in. ``exact_dynamics`` therefore runs a repair pass over the assembled
dynamics: every coefficient must end up an exact ``Fraction``, either the
injected one, a structural unit coefficient (the +x_{i+1} chain terms), or a
value re-attributed to a registered gain by its float image. Anything that
cannot be attributed raises ``PrintedGainError``; nothing is silently left in
float64.

Every rationalized term is then checked against the same dynamics built from
the float gains: identical exponents, identical term types, and identical
float image of the coefficient (for a bracketed gain, agreement to within
``BRACKET_FLOAT_TOL``, since the endpoint is a decimal neighbour of the float
image of the true irrational). A mis-registered value cannot pass this check.
"""
from __future__ import annotations

import importlib
import inspect
import itertools
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Tuple, Union

from gf_framework.algebra.core import GeneralizedForm

#: Decimal digits of the rational bracket endpoints for irrational gains.
BRACKET_DIGITS = 15

#: Tolerance for "the bracket endpoint is the decimal neighbour of the float
#: image of the true irrational gain". The endpoints of a 15-digit decimal
#: bracket of a value of magnitude ~1..10 sit within ~1e-15 of it; the float
#: image itself may be up to half an ulp on the other side.
BRACKET_FLOAT_TOL = 1e-12


class PrintedGainError(Exception):
    """A gain point cannot be realized exactly, or contradicts the float path."""


# ---------------------------------------------------------------------------
# Registry entry types
# ---------------------------------------------------------------------------

#: ``gain``     enters a dynamics coefficient (the object of the repair pass).
#: ``exponent`` enters a GF exponent, which is an exact Fraction already.
#: ``scaling``  a factory-side scaling parameter, not itself a coefficient.
ROLES = ("gain", "exponent", "scaling")


@dataclass(frozen=True)
class PrintedGain:
    """A gain whose printed decimal is an exact rational."""
    value: Fraction
    provenance: str
    role: str = "gain"
    #: Module attribute holding the float default, when the factory signature
    #: default is ``None`` (levant4d resolves its gains from module constants).
    default_attr: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise PrintedGainError(f"unknown role {self.role!r}")
        if not isinstance(self.value, Fraction):
            raise PrintedGainError(f"printed value must be a Fraction, got {self.value!r}")

    @property
    def is_bracketed(self) -> bool:
        return False

    def float_image(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class BracketedGain:
    """A genuinely irrational gain, pinned by ``value ** power_k == power_target``.

    ``approx`` is a float seed only. It chooses where ``decimal_bracket`` starts
    looking; the bracket's validity is decided in exact rational arithmetic.
    """
    power_k: int
    power_target: Fraction
    approx: float
    provenance: str
    role: str = "gain"
    default_attr: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise PrintedGainError(f"unknown role {self.role!r}")
        if self.power_k < 2:
            raise PrintedGainError("power_k must be at least 2 for an irrational gain")
        if self.power_target <= 0 or self.approx <= 0:
            raise PrintedGainError("only positive gains are supported")

    @property
    def is_bracketed(self) -> bool:
        return True

    def float_image(self) -> float:
        return self.approx


GainEntry = Union[PrintedGain, BracketedGain]


@dataclass(frozen=True)
class GainPoint:
    """One named point in gain space for one system."""
    system: str
    #: ``create_*_spec`` factory name in ``discovery.universal_discovery``.
    factory: str
    #: System module the float defaults live in (provenance of the printed values).
    module: str
    gains: Mapping[str, GainEntry]
    #: Weighted-degree-0 sign channel, or None when no constant matched
    #: disturbance is admissible (kappa + r_ch != 0).
    channel: Optional[int]
    note: str = ""

    @property
    def bracketed_names(self) -> Tuple[str, ...]:
        return tuple(sorted(n for n, g in self.gains.items() if g.is_bracketed))

    @property
    def coefficient_gains(self) -> Dict[str, GainEntry]:
        """Entries that reach a dynamics coefficient (role ``gain``)."""
        return {n: g for n, g in self.gains.items() if g.role == "gain"}


@dataclass
class ExactDynamics:
    """Dynamics of one system at one bracket corner, in exact rational arithmetic."""
    system: str
    dynamics: List[GeneralizedForm]
    weights_r: Tuple[int, ...]
    kappa: float
    #: name -> exact value used for this corner (bracket endpoint if bracketed).
    gain_values: Dict[str, Fraction]
    #: name -> (lo, hi) for the bracketed gains of this point.
    brackets: Dict[str, Tuple[Fraction, Fraction]]
    #: name -> "lo"/"hi" for this corner (empty when nothing is bracketed).
    corner: Dict[str, str] = field(default_factory=dict)
    #: Coefficients the repair pass had to re-attribute (system, eq, term, gain).
    repairs: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def corner_label(self) -> str:
        if not self.corner:
            return "rational gains only"
        return ", ".join(f"{n}={self.corner[n]}" for n in sorted(self.corner))

    def summary(self) -> dict:
        """JSON-ready provenance block for run artifacts."""
        return {
            "system": self.system,
            "corner_label": self.corner_label,
            "corner": dict(self.corner),
            "gain_values": {n: str(v) for n, v in sorted(self.gain_values.items())},
            "repaired_coefficients": [
                {"equation": e, "term": t, "gain": g} for e, t, g in self.repairs
            ],
        }


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_LEVANT4D_NOTE = (
    "Non-recursive conversion of Levant's published recursive set "
    "(1.1, 1.5, 2, 3) at L = 1: lambda1 = 3, lambda2 = 2*3^(2/3), "
    "lambda3 = (3/2)*sqrt(lambda2), lambda4 = 11/10. lambda2 and lambda3 are "
    "irrational and enter through rational brackets."
)

_LEVANT5D_NOTE = (
    "Non-recursive conversion of Levant's published recursive set "
    "(1.1, 1.5, 2, 3, 5) at L = 1: lambda1 = 5, lambda2 = 3*5^(3/4), "
    "lambda3 = 2*lambda2^(2/3), lambda4 = (3/2)*sqrt(lambda3), "
    "lambda5 = 11/10. lambda2, lambda3 and lambda4 are irrational and enter "
    "through rational brackets, so the corner box of this point has "
    "2**3 = 8 corners. levant5d is the n = 5 scale-boundary probe and is NOT "
    "part of the 12-system benchmark matrix, so it is registered in the "
    "flagship table only; as for levant4d, its module defaults ARE the "
    "converted 2003 gains, which is why one entry serves both roles."
)

#: The 12-system benchmark matrix, at the gain point the ``create_*_spec``
#: factories use by default. This is the point that produced
#: ``artifacts/publication/results_gf_sos_clean/gf_matrix_*.json`` and the one
#: E1.1/E1.3 re-run.
BENCHMARK_GAIN_POINTS: Dict[str, GainPoint] = {
    "fosmc": GainPoint(
        system="fosmc",
        factory="create_fosmc_spec",
        module="gf_framework.systems.fosmc",
        channel=0,
        note="Switching gain of s' = -k sign(s).",
        gains={
            # systems/fosmc.py: get_fosmc_dynamics_gf(k=1.0)
            "k": PrintedGain(Fraction(1), "system module default (fosmc.py, k = 1.0)"),
        },
    ),
    "power_rate": GainPoint(
        system="power_rate",
        factory="create_power_rate_spec",
        module="gf_framework.systems.power_rate",
        channel=None,  # kappa + r_ch = alpha != 0: no constant matched disturbance
        note="Power-rate reaching law s' = -k ceil(s)^alpha. alpha is an "
             "exponent, exact as Fraction(1, 2) in the GF already.",
        gains={
            # systems/power_rate.py: PowerRateSystem(k=1.5, alpha=0.5),
            # get_power_rate_dynamics_gf(k=1.5, alpha=0.5)
            "k": PrintedGain(Fraction(3, 2), "system module default (power_rate.py, k = 1.5)"),
            "alpha": PrintedGain(
                Fraction(1, 2),
                "system module default (power_rate.py, POWER_RATE_ALPHA = 0.5)",
                role="exponent",
            ),
        },
    ),
    "sta": GainPoint(
        system="sta",
        factory="create_sta_spec",
        module="gf_framework.systems.sta",
        channel=1,
        note="Super-twisting standard tuning; the Seeber-Horn 2017 setting.",
        gains={
            # systems/sta.py: "Default gains: k1 = 1.5, k2 = 1.1"
            "k1": PrintedGain(Fraction(3, 2), "system module default (sta.py, k1 = 1.5)"),
            "k2": PrintedGain(Fraction(11, 10), "system module default (sta.py, k2 = 1.1)"),
        },
    ),
    "twisting": GainPoint(
        system="twisting",
        factory="create_twisting_spec",
        module="gf_framework.systems.twisting",
        channel=1,
        note="Twisting relay pair, r1 > r2 > 0 (Levant 1993).",
        gains={
            # systems/twisting.py: get_twisting_dynamics_gf(r1=2.0, r2=1.5)
            "r1": PrintedGain(Fraction(2), "system module default (twisting.py, r1 = 2.0)"),
            "r2": PrintedGain(Fraction(3, 2), "system module default (twisting.py, r2 = 1.5)"),
        },
    ),
    "example1": GainPoint(
        system="example1",
        factory="create_example1_spec",
        module="gf_framework.systems.example1",
        channel=None,  # no weighted-degree-0 channel
        note="Sanchez-Moreno 2019, Example 1 (eq. 4), unit gains.",
        gains={
            # systems/example1.py: get_example1_dynamics_gf(k1=1.0, k2=1.0)
            "k1": PrintedGain(Fraction(1), "system module default (example1.py, k1 = 1.0)"),
            "k2": PrintedGain(Fraction(1), "system module default (example1.py, k2 = 1.0)"),
        },
    ),
    "example3": GainPoint(
        system="example3",
        factory="create_example3_spec",
        module="gf_framework.systems.example3",
        channel=None,  # no weighted-degree-0 channel
        note="Sanchez-Moreno 2019, Example 3 (eq. 7). The paper prints the "
             "Algorithm-6 gains rounded (k1 approx 1.8, k2 approx 4.2); the "
             "registry binds the printed decimals, which are exactly what the "
             "system module implements.",
        gains={
            # systems/example3.py: get_example3_dynamics_gf(k1=1.8, k2=4.2)
            "k1": PrintedGain(Fraction(9, 5), "system module default (example3.py, k1 = 1.8)"),
            "k2": PrintedGain(Fraction(21, 5), "system module default (example3.py, k2 = 4.2)"),
        },
    ),
    "levant3d": GainPoint(
        system="levant3d",
        factory="create_levant3d_spec",
        module="gf_framework.systems.levant3d",
        channel=2,
        note="BENCHMARK gain point (3, 2, 11/10). Distinct from both published "
             "standard tunings of the second-order differentiator; see "
             "FLAGSHIP_GAIN_POINTS for those.",
        gains={
            # systems/levant3d.py: get_levant3d_dynamics_gf(3.0, 2.0, 1.1)
            "lambda1": PrintedGain(Fraction(3), "system module default (levant3d.py, lambda1 = 3.0)"),
            "lambda2": PrintedGain(Fraction(2), "system module default (levant3d.py, lambda2 = 2.0)"),
            "lambda3": PrintedGain(Fraction(11, 10), "system module default (levant3d.py, lambda3 = 1.1)"),
        },
    ),
    "pidsmc": GainPoint(
        system="pidsmc",
        factory="create_pidsmc_spec",
        module="gf_framework.systems.pidsmc",
        channel=2,
        note="PID-like SMC (Perez-Ventura et al. 2021). The factory scales the "
             "gains by delta**(2/3), delta**(1/2), delta; at delta = 1 the "
             "scaling is the identity, but the float power collapses Fractions "
             "back to float64, so the repair pass in exact_dynamics is what "
             "makes lambda1 and lambda2 exact here.",
        gains={
            # systems/pidsmc.py: get_pidsmc_dynamics_gf(2.7, 5.345, 1.1, delta=1.0)
            "lambda1": PrintedGain(Fraction(27, 10), "system module default (pidsmc.py, lambda1 = 2.7)"),
            "lambda2": PrintedGain(Fraction(1069, 200), "system module default (pidsmc.py, lambda2 = 5.345)"),
            "lambda3": PrintedGain(Fraction(11, 10), "system module default (pidsmc.py, lambda3 = 1.1)"),
            "delta": PrintedGain(
                Fraction(1),
                "system module default (pidsmc.py, delta = 1.0)",
                role="scaling",
            ),
        },
    ),
    "dsta": GainPoint(
        system="dsta",
        factory="create_dsta_spec",
        module="gf_framework.systems.dsta",
        channel=1,
        note="Damped super-twisting (novel controller): STA plus sign(x2) damping.",
        gains={
            # systems/dsta.py: DSTA_DEFAULT_GAINS = {'k1': 1.5, 'k2': 1.1, 'k3': 0.5}
            "k1": PrintedGain(Fraction(3, 2), "system module default (dsta.py, DSTA_DEFAULT_GAINS k1 = 1.5)"),
            "k2": PrintedGain(Fraction(11, 10), "system module default (dsta.py, DSTA_DEFAULT_GAINS k2 = 1.1)"),
            "k3": PrintedGain(Fraction(1, 2), "system module default (dsta.py, DSTA_DEFAULT_GAINS k3 = 0.5)"),
        },
    ),
    "fc_cta": GainPoint(
        system="fc_cta",
        factory="create_fc_cta_spec",
        module="gf_framework.systems.fc_cta",
        channel=2,
        note="Fully-connected CTA (novel controller): CTA gains plus the novel "
             "sign(x3) relay gain k5.",
        gains={
            # systems/fc_cta.py: FC_CTA_DEFAULT_GAINS
            "k1": PrintedGain(Fraction(274, 5), "system module default (fc_cta.py, k1 = 54.8; CTA benchmark value)"),
            "k2": PrintedGain(Fraction(27), "system module default (fc_cta.py, k2 = 27.0; CTA benchmark value)"),
            "k3": PrintedGain(Fraction(57, 10), "system module default (fc_cta.py, k3 = 5.7; CTA benchmark value)"),
            "k4": PrintedGain(Fraction(3), "system module default (fc_cta.py, k4 = 3.0; CTA benchmark value)"),
            "k5": PrintedGain(Fraction(2), "system module default (fc_cta.py, k5 = 2.0; novel relay gain)"),
        },
    ),
    "continuous_twisting": GainPoint(
        system="continuous_twisting",
        factory="create_cta_spec",
        module="gf_framework.systems.continuous_twisting",
        channel=2,
        note="Continuous twisting algorithm (Torres-Gonzalez, Sanchez, "
             "Fridman and Moreno 2017): module benchmark gains of unknown "
             "source, not a gain set of that paper (its Table 1 lists G1-G4).",
        gains={
            # systems/continuous_twisting.py: module default (benchmark) gains
            # k1 = 54.8, k2 = 27, k3 = 5.7, k4 = 3. Correction 2026-09-28: the
            # paper has no "Lemma 1"; see docs/papers/gf_ijrnc/reviews/cta_redo/.
            "k1": PrintedGain(Fraction(274, 5), "module default value (CTA benchmark gains, k1 = 54.8)"),
            "k2": PrintedGain(Fraction(27), "module default value (CTA benchmark gains, k2 = 27)"),
            "k3": PrintedGain(Fraction(57, 10), "module default value (CTA benchmark gains, k3 = 5.7)"),
            "k4": PrintedGain(Fraction(3), "module default value (CTA benchmark gains, k4 = 3)"),
        },
    ),
    "levant4d": GainPoint(
        system="levant4d",
        factory="create_levant4d_spec",
        module="gf_framework.systems.levant4d",
        channel=3,
        note=_LEVANT4D_NOTE,
        gains={
            # systems/levant4d.py: _L1 = 3.0, _L2 = 2*_L1**(2/3),
            # _L3 = 1.5*_L2**0.5, _L4 = 1.1
            "lambda1": PrintedGain(
                Fraction(3), "system module constant (levant4d.py, _L1 = 3.0)",
                default_attr="_L1",
            ),
            # lambda2 = 2*3^(2/3)  =>  lambda2^3 = 8*9 = 72
            "lambda2": BracketedGain(
                power_k=3, power_target=Fraction(72),
                approx=2.0 * 3.0 ** (2.0 / 3.0),
                provenance="system module constant (levant4d.py, _L2 = 2*_L1**(2/3)); "
                           "irrational, bracketed by lambda2^3 = 72",
                default_attr="_L2",
            ),
            # lambda3 = (3/2)*sqrt(lambda2)  =>  lambda3^6 = (3/2)^6 * lambda2^3
            #                                             = (729/64)*72 = 6561/8
            "lambda3": BracketedGain(
                power_k=6, power_target=Fraction(6561, 8),
                approx=1.5 * (2.0 * 3.0 ** (2.0 / 3.0)) ** 0.5,
                provenance="system module constant (levant4d.py, _L3 = 1.5*_L2**0.5); "
                           "irrational, bracketed by lambda3^6 = 6561/8",
                default_attr="_L3",
            ),
            "lambda4": PrintedGain(
                Fraction(11, 10), "system module constant (levant4d.py, _L4 = 1.1)",
                default_attr="_L4",
            ),
        },
    ),
}

#: Gains whose printed value is NOT the float default's exact value, because
#: the true value is irrational. Every other registry entry must satisfy
#: float(printed) == float default, bit for bit.
EXPECTED_FLOAT_IMAGE_DEVIATIONS = frozenset({
    ("levant4d", "lambda2"),
    ("levant4d", "lambda3"),
})

#: The published standard tunings of the flagship perturbed rows, consumed by
#: ``scripts/exact_gate_standard_gains.py``. These are DIFFERENT gain points
#: from the benchmark ones above, not different encodings of the same numbers.
FLAGSHIP_GAIN_POINTS: Dict[str, GainPoint] = {
    "sta": GainPoint(
        system="sta",
        factory="create_sta_spec",
        module="gf_framework.systems.sta",
        channel=1,
        note="Standard super-twisting tuning (3/2, 11/10); identical to the "
             "benchmark point for this system.",
        gains={
            "k1": PrintedGain(Fraction(3, 2), "published standard tuning (k1 = 1.5)"),
            "k2": PrintedGain(Fraction(11, 10), "published standard tuning (k2 = 1.1)"),
        },
    ),
    "levant3d": GainPoint(
        system="levant3d",
        factory="create_levant3d_spec",
        module="gf_framework.systems.levant3d",
        channel=2,
        note="Standard second-order differentiator tuning: conversion of the "
             "recursive set (1.1, 1.5, 2) at L = 1.",
        gains={
            "lambda1": PrintedGain(Fraction(2), "converted standard tuning (lambda1 = 2)"),
            # lambda2 = (3/2)sqrt(2)  =>  lambda2^2 = 9/2
            "lambda2": BracketedGain(
                power_k=2, power_target=Fraction(9, 2),
                approx=1.5 * math.sqrt(2.0),
                provenance="converted standard tuning (lambda2 = (3/2)sqrt(2)); "
                           "irrational, bracketed by lambda2^2 = 9/2",
            ),
            "lambda3": PrintedGain(Fraction(11, 10), "converted standard tuning (lambda3 = 1.1)"),
        },
    ),
    "levant3d_2003print": GainPoint(
        system="levant3d_2003print",
        factory="create_levant3d_spec",
        module="gf_framework.systems.levant3d",
        channel=2,
        note="Higher-gain printed variant of the second-order differentiator "
             "sequence (Levant 2003 print).",
        gains={
            "lambda1": PrintedGain(Fraction(3), "printed 2003 sequence (lambda1 = 3)"),
            # lambda2 = (3/2)sqrt(3)  =>  lambda2^2 = 27/4
            "lambda2": BracketedGain(
                power_k=2, power_target=Fraction(27, 4),
                approx=1.5 * math.sqrt(3.0),
                provenance="printed 2003 sequence (lambda2 = (3/2)sqrt(3)); "
                           "irrational, bracketed by lambda2^2 = 27/4",
            ),
            "lambda3": PrintedGain(Fraction(11, 10), "printed 2003 sequence (lambda3 = 1.1)"),
        },
    ),
    "levant4d": GainPoint(
        system="levant4d",
        factory="create_levant4d_spec",
        module="gf_framework.systems.levant4d",
        channel=3,
        note=_LEVANT4D_NOTE + " Identical to the benchmark point: the levant4d "
             "module defaults ARE the converted 2003 gains.",
        gains=BENCHMARK_GAIN_POINTS["levant4d"].gains,
    ),
    # WP X10 (2026-09-25): the third-order differentiator at the higher-gain
    # standard tuning, recursive (lambdahat_0..lambdahat_3) = (1.1, 1.5, 3, 5)
    # at L = 1. This is the order-3 tail of the recursive set Levant prints
    # in 2003 (Section 7.1, the fifth-order example 12, 8, 5, 3, 1.5, 1.1) and
    # 2005 (1.1, 1.5, 3, 5, 8, 12); the levant4d entry above is the other
    # standard tuning (1.1, 1.5, 2, 3). Same equations as levant4d (factory
    # create_levant4d_spec), different gains; the key follows the
    # levant3d_2003print precedent. Non-recursive conversion (exact, by
    # substituting each recursive stage into the next):
    #   lambda1 = lambdahat3                   = 5
    #   lambda2 = lambdahat2 * lambda1^(2/3)   = 3 * 5^(2/3)
    #   lambda3 = lambdahat1 * lambda2^(1/2)   = (3/2) * sqrt(lambda2)
    #   lambda4 = lambdahat0                   = 11/10
    "levant4d_2003print": GainPoint(
        system="levant4d_2003print",
        factory="create_levant4d_spec",
        module="gf_framework.systems.levant4d",
        channel=3,
        note="Higher-gain standard tuning of the third-order differentiator: "
             "conversion of the recursive set (1.1, 1.5, 3, 5) at L = 1 "
             "(Levant 2003/2005 print). Same equations as levant4d, different "
             "gains (WP X10).",
        gains={
            "lambda1": PrintedGain(
                Fraction(5), "converted recursive set (1.1, 1.5, 3, 5): "
                             "lambda1 = lambdahat3 = 5"),
            # lambda2 = 3*5^(2/3)  =>  lambda2^3 = 27*25 = 675
            "lambda2": BracketedGain(
                power_k=3, power_target=Fraction(675),
                approx=3.0 * 5.0 ** (2.0 / 3.0),
                provenance="converted recursive set (1.1, 1.5, 3, 5): "
                           "lambda2 = 3*lambda1^(2/3) = 3*5^(2/3); irrational, "
                           "bracketed by lambda2^3 = 675",
            ),
            # lambda3 = (3/2)*sqrt(lambda2)  =>  lambda3^6 = (3/2)^6 * lambda2^3
            #                                             = (729/64)*675 = 492075/64
            "lambda3": BracketedGain(
                power_k=6, power_target=Fraction(492075, 64),
                approx=1.5 * (3.0 * 5.0 ** (2.0 / 3.0)) ** 0.5,
                provenance="converted recursive set (1.1, 1.5, 3, 5): "
                           "lambda3 = (3/2)*lambda2^(1/2); irrational, "
                           "bracketed by lambda3^6 = 492075/64",
            ),
            "lambda4": PrintedGain(
                Fraction(11, 10), "converted recursive set (1.1, 1.5, 3, 5): "
                                  "lambda4 = lambdahat0 = 1.1"),
        },
    ),
    "levant5d": GainPoint(
        system="levant5d",
        factory="create_levant5d_spec",
        module="gf_framework.systems.levant5d",
        # weights r = (5, 4, 3, 2, 1), kappa = -1, so the weighted-degree-0
        # channel is the last equation: kappa + r_5 = 0.
        channel=4,
        note=_LEVANT5D_NOTE,
        gains={
            # Sources, per gain: the Levant 2003 print (recursive set
            # (lambdahat_0..lambdahat_4) = (1.1, 1.5, 2, 3, 5) at L = 1) and
            # the module constants of systems/levant5d.py that implement its
            # non-recursive conversion:
            #   _L1 = 5.0, _L2 = 3.0*_L1**(3/4), _L3 = 2.0*_L2**(2/3),
            #   _L4 = 1.5*_L3**0.5, _L5 = 1.1.
            # The float seeds below repeat those expressions literally, so
            # float(seed) is the module default bit for bit.
            "lambda1": PrintedGain(
                Fraction(5),
                "system module constant (levant5d.py, _L1 = 5.0) = Levant 2003 "
                "print lambdahat4 = 5 at L = 1",
                default_attr="_L1",
            ),
            # lambda2 = 3*5^(3/4)  =>  lambda2^4 = 3^4 * 5^3 = 81*125 = 10125
            "lambda2": BracketedGain(
                power_k=4, power_target=Fraction(10125),
                approx=3.0 * 5.0 ** (3.0 / 4.0),
                provenance="system module constant (levant5d.py, "
                           "_L2 = 3.0*_L1**(3/4)) = Levant 2003 print "
                           "lambdahat3*lambda1^(3/4); irrational, bracketed by "
                           "lambda2^4 = 10125",
                default_attr="_L2",
            ),
            # lambda3 = 2*lambda2^(2/3)  =>  lambda3^6 = 2^6 * lambda2^4
            #                                          = 64*10125 = 648000
            "lambda3": BracketedGain(
                power_k=6, power_target=Fraction(648000),
                approx=2.0 * (3.0 * 5.0 ** (3.0 / 4.0)) ** (2.0 / 3.0),
                provenance="system module constant (levant5d.py, "
                           "_L3 = 2.0*_L2**(2/3)) = Levant 2003 print "
                           "lambdahat2*lambda2^(2/3); irrational, bracketed by "
                           "lambda3^6 = 648000",
                default_attr="_L3",
            ),
            # lambda4 = (3/2)*sqrt(lambda3)  =>  lambda4^12 = (3/2)^12 * lambda3^6
            #                        = (531441/4096)*648000 = 5380840125/64
            "lambda4": BracketedGain(
                power_k=12, power_target=Fraction(5380840125, 64),
                approx=1.5 * (2.0 * (3.0 * 5.0 ** (3.0 / 4.0))
                              ** (2.0 / 3.0)) ** 0.5,
                provenance="system module constant (levant5d.py, "
                           "_L4 = 1.5*_L3**0.5) = Levant 2003 print "
                           "lambdahat1*lambda3^(1/2); irrational, bracketed by "
                           "lambda4^12 = 5380840125/64",
                default_attr="_L4",
            ),
            "lambda5": PrintedGain(
                Fraction(11, 10),
                "system module constant (levant5d.py, _L5 = 1.1) = Levant 2003 "
                "print lambdahat0 = 1.1 at L = 1",
                default_attr="_L5",
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Brackets and corners
# ---------------------------------------------------------------------------

def decimal_bracket(k: int, q: Fraction, approx: float,
                    digits: int = BRACKET_DIGITS) -> Tuple[Fraction, Fraction]:
    """Rational decimal bracket [lo, hi] of the positive real g with g**k == q.

    Seeded from the float approximation, then stepped in exact arithmetic until
    lo**k < q < hi**k holds. The float only chooses where to start looking; the
    bracket's validity is proved in Q.

    Moved here verbatim from ``scripts/exact_gate_standard_gains.py`` so that
    the benchmark matrix and the flagship driver share one bracket convention.
    """
    step = Fraction(1, 10 ** digits)
    lo = Fraction(round(approx * 10 ** digits), 10 ** digits)
    while lo ** k >= q:
        lo -= step
    hi = lo + step
    while hi ** k <= q:
        hi += step
    assert lo ** k < q < hi ** k and lo > 0
    return lo, hi


def gain_point(system: str, flagship: bool = False) -> GainPoint:
    """Look up a registered gain point by system name."""
    table = FLAGSHIP_GAIN_POINTS if flagship else BENCHMARK_GAIN_POINTS
    try:
        return table[system]
    except KeyError:
        raise PrintedGainError(
            f"no {'flagship' if flagship else 'benchmark'} gain point registered "
            f"for {system!r}; known: {sorted(table)}"
        ) from None


def brackets(point: GainPoint,
             digits: int = BRACKET_DIGITS) -> Dict[str, Tuple[Fraction, Fraction]]:
    """Rational brackets of every irrational gain of ``point``."""
    return {
        name: decimal_bracket(g.power_k, g.power_target, g.approx, digits)
        for name, g in point.gains.items() if g.is_bracketed
    }


def corner_assignments(point: GainPoint,
                       digits: int = BRACKET_DIGITS) -> List[Dict[str, Fraction]]:
    """All 2**j corners of the bracket box, as name -> exact endpoint.

    Returns ``[{}]`` when the point has no irrational gain, so callers can loop
    uniformly over corners.
    """
    bs = brackets(point, digits)
    names = sorted(bs)
    if not names:
        return [{}]
    return [dict(zip(names, combo))
            for combo in itertools.product(*(bs[n] for n in names))]


def corner_labels(point: GainPoint, assignment: Mapping[str, Fraction],
                  digits: int = BRACKET_DIGITS) -> Dict[str, str]:
    """name -> "lo"/"hi" for one corner assignment."""
    bs = brackets(point, digits)
    out = {}
    for name, value in assignment.items():
        lo, hi = bs[name]
        if value == lo:
            out[name] = "lo"
        elif value == hi:
            out[name] = "hi"
        else:
            raise PrintedGainError(
                f"{point.system}.{name}: {value} is not an endpoint of "
                f"[{lo}, {hi}]"
            )
    return out


def exact_gain_values(point: GainPoint, assignment: Optional[Mapping[str, Fraction]] = None,
                      digits: int = BRACKET_DIGITS) -> Dict[str, Fraction]:
    """Exact value of every registered parameter at one corner."""
    assignment = dict(assignment or {})
    values: Dict[str, Fraction] = {}
    for name, g in point.gains.items():
        if g.is_bracketed:
            if name not in assignment:
                raise PrintedGainError(
                    f"{point.system}.{name} is bracketed; a corner assignment "
                    f"is required (use corner_assignments)"
                )
            values[name] = assignment[name]
        else:
            values[name] = g.value
    unknown = set(assignment) - set(point.gains)
    if unknown:
        raise PrintedGainError(f"assignment names not registered: {sorted(unknown)}")
    return values


def float_gain_values(point: GainPoint) -> Dict[str, float]:
    """Float image of every registered parameter (the historical encoding)."""
    return {name: g.float_image() for name, g in point.gains.items()}


def module_float_defaults(point: GainPoint) -> Dict[str, float]:
    """The float default each parameter actually has in the system module.

    Signature default of the ``create_*_spec`` factory, or the module constant
    named by ``default_attr`` when the signature default is ``None``.
    """
    from gf_framework.discovery import universal_discovery as ud

    sig = inspect.signature(getattr(ud, point.factory))
    mod = importlib.import_module(point.module)
    out: Dict[str, float] = {}
    for name, g in point.gains.items():
        if name not in sig.parameters:
            raise PrintedGainError(
                f"{point.system}: registered gain {name!r} is not a parameter of "
                f"{point.factory}"
            )
        default = sig.parameters[name].default
        if default is None:
            if not g.default_attr:
                raise PrintedGainError(
                    f"{point.system}.{name}: factory default is None and no "
                    f"default_attr is registered"
                )
            default = getattr(mod, g.default_attr)
        out[name] = float(default)
    return out


# ---------------------------------------------------------------------------
# Exact dynamics assembly
# ---------------------------------------------------------------------------

def _attribute_float_coefficient(coeff: float, values: Mapping[str, Fraction],
                                 where: str) -> Tuple[Fraction, Optional[str]]:
    """Turn a leftover float dynamics coefficient into its exact rational.

    Structural unit coefficients (the +x_{i+1} chain terms, coefficient 1.0)
    are exact already. Any other float must be the float image of exactly one
    registered gain, entering with a sign; otherwise the gain point cannot be
    realized exactly and we refuse rather than fall back to the dyadic image.

    Two deliberate details:

    * unit coefficients are resolved structurally BEFORE gain attribution, so a
      structural +1 is never mislabelled as a gain. A system whose printed gain
      happens to be exactly 1 (example1) therefore does not appear in the
      repair list; its exact value is Fraction(1) either way, so no coefficient
      is affected.
    * a bracketed gain is matched by exact float equality, which a bracket
      endpoint need not satisfy (it is a decimal neighbour of the float image).
      A factory that started doing arithmetic on an irrational gain would
      therefore raise here instead of quietly producing a wrong value.
    """
    if coeff == 0.0:
        return Fraction(0), None
    if coeff in (1.0, -1.0):
        return Fraction(int(coeff)), None
    magnitude = abs(coeff)
    matches = {name: v for name, v in values.items() if float(v) == magnitude}
    distinct = set(matches.values())
    if not distinct:
        raise PrintedGainError(
            f"{where}: float coefficient {coeff!r} is not the image of any "
            f"registered gain {sorted(str(v) for v in values.values())}; "
            f"refusing to fall back to the float64 image"
        )
    if len(distinct) > 1:
        raise PrintedGainError(
            f"{where}: float coefficient {coeff!r} is ambiguous between "
            f"{sorted(matches)} with different exact values"
        )
    exact = next(iter(distinct))
    return (-exact if coeff < 0 else exact), sorted(matches)[0]


def _rationalize(gf_exact: GeneralizedForm, gf_float: GeneralizedForm,
                 values: Mapping[str, Fraction],
                 bracket_images: Mapping[Fraction, float],
                 eq_index: int, system: str) -> Tuple[GeneralizedForm, List[Tuple[int, int, str]]]:
    """Rationalize one equation and cross-check it against the float build."""
    if len(gf_exact.terms) != len(gf_float.terms):
        raise PrintedGainError(
            f"{system} f{eq_index}: term count differs between the exact and "
            f"float builds ({len(gf_exact.terms)} vs {len(gf_float.terms)})"
        )
    if getattr(gf_exact, "composite_terms", None):
        raise PrintedGainError(
            f"{system} f{eq_index}: composite GF terms are not supported by the "
            f"printed-rational path"
        )
    new_terms = []
    repairs: List[Tuple[int, int, str]] = []
    for t_index, (t, t0) in enumerate(zip(gf_exact.terms, gf_float.terms)):
        where = f"{system} f{eq_index} term {t_index}"
        if tuple(t.exponents) != tuple(t0.exponents) or tuple(t.term_types) != tuple(t0.term_types):
            raise PrintedGainError(
                f"{where}: support differs between the exact and float builds"
            )
        coeff = t.coefficient
        if isinstance(coeff, Fraction):
            exact = coeff
        else:
            exact, gain_name = _attribute_float_coefficient(float(coeff), values, where)
            if gain_name is not None:
                repairs.append((eq_index, t_index, gain_name))
        _check_float_image(exact, float(t0.coefficient), bracket_images, where)
        new_terms.append(replace(t, coefficient=exact))
    return replace(gf_exact, terms=new_terms), repairs


def _check_float_image(exact: Fraction, float_coeff: float,
                       bracket_images: Mapping[Fraction, float], where: str) -> None:
    """The exact coefficient must name the same published number as the float one."""
    if float(exact) == float_coeff:
        return
    reference = bracket_images.get(exact)
    if reference is None:
        raise PrintedGainError(
            f"{where}: exact coefficient {exact} has float image {float(exact)!r}, "
            f"but the float build has {float_coeff!r}, and it is not a "
            f"registered bracket endpoint"
        )
    if abs(reference - float_coeff) > BRACKET_FLOAT_TOL:
        raise PrintedGainError(
            f"{where}: bracket endpoint {exact} sits {abs(reference - float_coeff):.3e} "
            f"away from the float build's coefficient {float_coeff!r}, beyond the "
            f"tolerance {BRACKET_FLOAT_TOL:g}"
        )


def exact_dynamics(point: GainPoint,
                   assignment: Optional[Mapping[str, Fraction]] = None,
                   digits: int = BRACKET_DIGITS) -> ExactDynamics:
    """Build the NOMINAL dynamics of ``point`` in exact rational arithmetic.

    Every coefficient of the returned generalized forms is an exact
    ``Fraction``. For a perturbed run, pass the result to
    ``universal_discovery.matched_disturbance_vertices`` with an exact
    ``Fraction`` delta0; that function already preserves exact rationals.
    """
    from gf_framework.discovery import universal_discovery as ud

    factory = getattr(ud, point.factory)
    values = exact_gain_values(point, assignment, digits)
    spec_exact = factory(**values)
    spec_float = factory(**float_gain_values(point))

    if spec_exact.dynamics_gf_vertices is not None:
        raise PrintedGainError(
            f"{point.system}: exact_dynamics builds the nominal dynamics only; "
            f"the factory returned inclusion vertices"
        )

    bracket_images: Dict[Fraction, float] = {}
    for name, (lo, hi) in brackets(point, digits).items():
        image = point.gains[name].float_image()
        for endpoint in (lo, hi, -lo, -hi):
            bracket_images[endpoint] = image if endpoint > 0 else -image

    if len(spec_exact.dynamics_gf) != len(spec_float.dynamics_gf):
        raise PrintedGainError(f"{point.system}: equation count differs between builds")

    # Only role "gain" entries reach a dynamics coefficient; an exponent or a
    # scaling parameter must never be used to explain a leftover float.
    coefficient_values = {n: values[n] for n in point.coefficient_gains}

    dynamics = []
    repairs: List[Tuple[int, int, str]] = []
    for i, (g, g0) in enumerate(zip(spec_exact.dynamics_gf, spec_float.dynamics_gf)):
        gf, reps = _rationalize(g, g0, coefficient_values, bracket_images, i, point.system)
        dynamics.append(gf)
        repairs.extend(reps)

    for i, gf in enumerate(dynamics):
        for j, t in enumerate(gf.terms):
            if not isinstance(t.coefficient, Fraction):
                raise PrintedGainError(
                    f"{point.system} f{i} term {j}: coefficient is still "
                    f"{type(t.coefficient).__name__}, not an exact Fraction"
                )

    return ExactDynamics(
        system=point.system,
        dynamics=dynamics,
        weights_r=tuple(spec_exact.weights_r),
        kappa=spec_exact.kappa,
        gain_values=values,
        brackets=brackets(point, digits),
        corner=corner_labels(point, assignment or {}, digits),
        repairs=repairs,
    )


def exact_dynamics_corners(point: GainPoint,
                           digits: int = BRACKET_DIGITS) -> List[ExactDynamics]:
    """``exact_dynamics`` at every corner of the bracket box."""
    return [exact_dynamics(point, a, digits) for a in corner_assignments(point, digits)]


def gain_point_record(point: GainPoint,
                      digits: int = BRACKET_DIGITS) -> dict:
    """JSON-ready description of a registered gain point, for run artifacts."""
    bs = brackets(point, digits)
    return {
        "system": point.system,
        "factory": point.factory,
        "module": point.module,
        "channel": point.channel,
        "note": point.note,
        "printed_rational": {
            name: str(g.value) for name, g in sorted(point.gains.items())
            if not g.is_bracketed
        },
        "roles": {name: g.role for name, g in sorted(point.gains.items())},
        "provenance": {name: g.provenance for name, g in sorted(point.gains.items())},
        "brackets": {
            name: {
                "lo": str(bs[name][0]),
                "hi": str(bs[name][1]),
                "power_k": point.gains[name].power_k,
                "power_target": str(point.gains[name].power_target),
            }
            for name in sorted(bs)
        },
        "n_corners": 2 ** len(bs),
        "bracket_digits": digits,
    }


def legacy_gain_spec(point: GainPoint) -> dict:
    """``exact_gate_standard_gains.GAIN_SPECS`` entry for a flagship gain point.

    Keeps that driver's historical config shape (factory callable, channel,
    ``rational`` map, ``irrational`` map of (power, target, approx)) while the
    numbers come from this registry.
    """
    from gf_framework.discovery import universal_discovery as ud

    return {
        "factory": getattr(ud, point.factory),
        "channel": point.channel,
        "rational": {n: g.value for n, g in point.gains.items()
                     if not g.is_bracketed},
        "irrational": {n: (g.power_k, g.power_target, g.approx)
                       for n, g in point.gains.items() if g.is_bracketed},
    }
