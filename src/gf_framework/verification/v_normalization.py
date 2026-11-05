"""Scale normalisation of a designed V, before the strict gate (WP P5-02).

Why a scale step exists at all
------------------------------
The design SDP pins one diagonal coefficient (``alpha[diag_indices[0]] ==
coefficient_scale``) and bounds every other pure-power coefficient below by the
DESIGN margin ``epsilon``.  Lower the design margin -- which is exactly what
the P5-02 epsilon sweep does -- and the solver is free to push the remaining
pure powers down with it: at ``epsilon = 1e-4`` the twelve nominal benchmark
designs carry a smallest pure-power coefficient between 1.06e-04 and 1.0, and
at ``epsilon = 1e-8`` the CTA m = 5 design puts it at 5.18e-08.

The strict gate does not know where the design's margin was.  It re-solves
``V(y;s) - eps_gate * sum_i y_i^d`` SOS on a descending ABSOLUTE ladder
``[eps_gate, 1e-3*eps_gate, 1e-6*eps_gate]`` seeded from
``interfaces.VerificationConfig.epsilon`` (1e-6), and it then charges the
recovered Gram's l1 coefficient residual against an ABSOLUTE budget
``eps_gate/2``.  Both thresholds are absolute and the certificate is not: they
bracket V's coefficient magnitudes from below (the margin must fit under the
smallest pure power) and from above (the residual grows with the largest
coefficient).  So the gate has a SCALE WINDOW.

What the window actually looks like
-----------------------------------
Measured (``artifacts/analysis_2026-09-02/eps_sweep_escalation/run/
scale_window_p1.json``), gating ``c*V`` for ``c = 2**k``:

    continuous_twisting m=5, design eps 1e-8 : certified for k in [-4 .. +16]
    continuous_twisting m=6, design eps 1e-8 : certified for k in {0, +4, +12, +16}
    continuous_twisting m=7, design eps 1e-6 : certified for k in [-12 .. +12]
    continuous_twisting m=8, design eps 1e-4 : certified for k in [-16 .. 0]
    fc_cta              m=5, design eps 1e-6 : certified for k in [-8 .. +16]
    levant3d            m=5, design eps 1e-4 : certified for k in [-16 .. +16]
    pidsmc              m=6, design eps 1e-4 : certified for k in [-16 .. +20]

``k = 0`` -- the design's own scale -- is inside EVERY window, and the
dependence on ``k`` is NOT monotone (the gate re-solves the SDP for each
scaled V, so this is solver conditioning, not a threshold law: CTA m = 6 at
k = +8 came back with residual 7.8e+02 and a "solution may be inaccurate"
warning, between two certifying rungs).  The conclusion this module encodes is
therefore the conservative one: **do not move a design that is not degenerate**.
The step fires only when the coefficient log-centre defined below has drifted
more than ``2 ** NORMALIZATION_BAND_LOG2`` away from 1, which on every design
measured for WP P5-02 -- the twelve nominal rows at 1e-4, every sweep rung that
certified, and the levant3d 2-variant perturbed probe -- is never.  It is a
guard, not a routine step, and the measurement is why.

Why the statistic is the log-centre
-----------------------------------
The two absolute thresholds bracket the coefficient magnitudes from opposite
sides, so the quantity to centre is the geometric mean of the two ends:

    s = sqrt( min_i |alpha_i| over PURE POWER terms  x  max_j |alpha_j| )

Its smallest pure power is what the margin has to fit under; its largest
coefficient is what the residual tracks.  Centring ``s`` on 1 maximises the
room for the gate's ladder on both sides at once.  Pure arithmetic on the
coefficients: no solver, no sampling, no randomness, so it is reproducible
across processes by construction.

Why scaling is not a change of claim
------------------------------------
``V > 0 <=> cV > 0`` and ``-d(cV)/dt > 0 <=> -dV/dt > 0`` for any ``c > 0``, so
positive definiteness, the Filippov/Lemma-5 surface classification, C^1
regularity, r-homogeneity and the homogeneity degree m are all untouched: a
Lyapunov certificate is a statement about a POSITIVE CONE of functions and
``c`` moves V inside it.  What is NOT scale-invariant is the MARGIN: a
certificate of ``V >= eps * sum y^d`` becomes a certificate of
``cV >= (c*eps) * sum y^d``, which is why ``c`` multiplies the design margin
too, and why the number a certificate may quote is the charge/budget RATIO
(and ``eps_certified``, in the units of the V it belongs to), never a bare
epsilon compared across two differently scaled V's.  For an SOS block:

    V_s - eps*sum y^d = z^T Q z   =>   cV_s - (c*eps)*sum y^d = z^T (cQ) z

so ``cQ`` is PSD exactly when ``Q`` is, the exact residual
``E = z^T (cQ) z - c*target`` is exactly ``c`` times the old one, and with the
budget scaled by the same ``c`` the exact charge/budget ratio of Proposition 7
is INVARIANT.  Nothing is proved that was not proved before, and nothing that
was proved is lost.  (``eta_hat``, which floats only PROPOSE, is likewise
exactly ``c`` times the old one when the min-eigenvalue proposal is exact under
the scaling -- and even if a float proposal moved, the exact chain would still
decide the verdict, which is the whole point of X.)

Why the factor is a power of two
--------------------------------
``c = 2**k`` is exact in IEEE-754 binary64: every scaled coefficient, Gram
entry, target coefficient and margin is the exact product with no rounding, and
``Fraction(c*x) == c*Fraction(x)`` for every finite non-subnormal ``x``.  So the
exact-rational objects the X stage rebuilds are exactly ``c`` times the
unscaled ones, and the equality check the bundle route runs
(``exact_phase._check_bundle_group``: the stored margin must equal the rebuilt
margin as a ``Fraction``) still holds exactly.  A decimal factor would not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..algebra.core import GeneralizedForm, GFTerm

__all__ = [
    "NORMALIZATION_BAND_LOG2",
    "MAX_LOG2_FACTOR",
    "NORMALIZATION_REASONS",
    "VNormalization",
    "pure_power_indices",
    "scale_statistic",
    "plan_normalization",
    "scale_gf",
    "scale_gram_bundle",
    "normalize_for_gate",
]

#: Half-width, in binades, of the band around 1 in which a design's coefficient
#: log-centre is left alone.  Deliberately wide, and set from measurement
#: rather than derived, because the measurement says the design's own scale is
#: the RIGHT one wherever it has been checked:
#:
#:   * the twelve nominal rows at eps = 1e-4 have a log-centre between
#:     2**-6.09 (pidsmc, 1.47e-02) and 2**+0.70 (example1, 1.63);
#:   * every sweep rung that certified, down to eps = 1e-8, is inside
#:     2**-13.02 (continuous_twisting m = 6);
#:   * the gate's certifying scale window CONTAINS k = 0 on all seven cells of
#:     ``run/scale_window_p1.json``, and moving out of it in either direction
#:     loses the certificate.
#:
#: So the step is a last-resort guard against a design whose scale has run
#: away by more than sixteen binades (about five decades), a regime nothing
#: measured here reaches; inside the band the certificate keeps the scale its
#: designer gave it.  Widening or narrowing this is a measurement, not a
#: preference: re-run ``p5_02_measure.py scale`` before touching it.
NORMALIZATION_BAND_LOG2: int = 16

#: Refuse to scale by more than 2**MAX_LOG2_FACTOR.  A design needing more than
#: that is degenerate rather than drifted, and a factor near the float64
#: exponent range would risk overflow in the Gram rather than fix anything.
MAX_LOG2_FACTOR: int = 120

#: Typed reasons a normalisation did NOT happen.  ``""`` means it did.
NORMALIZATION_REASONS: Tuple[str, ...] = (
    "",
    "disabled",
    "within_band",
    "no_pure_power_terms",
    "composite_terms",
    "non_finite_coefficients",
    "non_positive_statistic",
    "factor_out_of_range",
)


@dataclass
class VNormalization:
    """What the normalisation step did, or the typed reason it did nothing."""

    applied: bool = False
    #: The scale factor, always an exact power of two (1.0 when not applied).
    factor: float = 1.0
    log2_factor: int = 0
    #: The log-centre before and after.
    statistic: float = float("nan")
    statistic_after: float = float("nan")
    #: The two ends the log-centre is built from, for the record.
    min_pure_power: float = float("nan")
    max_abs_coefficient: float = float("nan")
    reason: str = "disabled"
    n_pure_power_terms: int = 0
    n_terms: int = 0
    band_log2: int = NORMALIZATION_BAND_LOG2
    #: Design margin before and after (``None`` when no margin was handed in).
    epsilon: Optional[float] = None
    epsilon_scaled: Optional[float] = None
    #: Number of bundle entries rescaled (0 when there was no bundle).
    n_bundle_entries: int = 0

    def as_record(self) -> Dict[str, Any]:
        """JSON-ready record (every value a plain Python scalar)."""
        return {
            "applied": bool(self.applied),
            "factor": float(self.factor),
            "log2_factor": int(self.log2_factor),
            "statistic": float(self.statistic),
            "statistic_after": float(self.statistic_after),
            "min_pure_power": float(self.min_pure_power),
            "max_abs_coefficient": float(self.max_abs_coefficient),
            "reason": str(self.reason),
            "n_pure_power_terms": int(self.n_pure_power_terms),
            "n_terms": int(self.n_terms),
            "band_log2": int(self.band_log2),
            "epsilon": (None if self.epsilon is None else float(self.epsilon)),
            "epsilon_scaled": (None if self.epsilon_scaled is None
                               else float(self.epsilon_scaled)),
            "n_bundle_entries": int(self.n_bundle_entries),
        }


def pure_power_indices(V: GeneralizedForm) -> List[int]:
    """Indices of V's pure-power (diagonal) terms.

    The same predicate the designer uses to decide which coefficients get the
    explicit ``alpha[i] >= epsilon`` lower bound
    (``algebra.enumerate.TermSpec.is_diagonal``): exactly one nonzero exponent.
    These are the coefficients the gate's pure-power margin rows act on, which
    is why they, and not the cross terms, set the lower end of the statistic.
    """
    out: List[int] = []
    for i, term in enumerate(V.terms):
        if sum(1 for e in term.exponents if e != 0) == 1:
            out.append(i)
    return out


def scale_statistic(V: GeneralizedForm) -> float:
    """The coefficient log-centre of ``V``: ``sqrt(min pure power x max |a|)``.

    ``nan`` when V carries no pure-power term or no nonzero coefficient (then
    nothing here applies and the caller reports the typed reason).
    """
    idx = pure_power_indices(V)
    if not idx:
        return float("nan")
    lo = min(abs(float(V.terms[i].coefficient)) for i in idx)
    nz = [abs(float(t.coefficient)) for t in V.terms
          if float(t.coefficient) != 0.0]
    if not nz or not (lo > 0.0):
        return float("nan")
    return math.sqrt(lo * max(nz))


def plan_normalization(
    V: GeneralizedForm,
    *,
    band_log2: int = NORMALIZATION_BAND_LOG2,
    max_log2: int = MAX_LOG2_FACTOR,
    epsilon: Optional[float] = None,
) -> VNormalization:
    """Decide the scale factor without touching anything."""
    n_terms = len(V.terms)
    idx = pure_power_indices(V)
    base = VNormalization(
        applied=False, factor=1.0, log2_factor=0,
        statistic=float("nan"), statistic_after=float("nan"),
        n_pure_power_terms=len(idx), n_terms=n_terms,
        band_log2=int(band_log2),
        epsilon=(None if epsilon is None else float(epsilon)),
        epsilon_scaled=(None if epsilon is None else float(epsilon)),
    )
    if getattr(V, "composite_terms", None):
        # A composite term's coefficient is not the whole scale of the term
        # (the nested GF carries its own), so a flat coefficient multiply is
        # not a rescaling of V.  Refuse rather than guess.
        base.reason = "composite_terms"
        return base
    if not idx:
        base.reason = "no_pure_power_terms"
        return base
    coeffs = [float(t.coefficient) for t in V.terms]
    if not all(math.isfinite(c) for c in coeffs):
        base.reason = "non_finite_coefficients"
        return base
    lo = min(abs(coeffs[i]) for i in idx)
    nz = [abs(c) for c in coeffs if c != 0.0]
    base.min_pure_power = lo
    base.max_abs_coefficient = max(nz) if nz else float("nan")
    if not (lo > 0.0) or not nz:
        base.reason = "non_positive_statistic"
        return base
    s = math.sqrt(lo * max(nz))
    base.statistic = s
    base.statistic_after = s
    if not (s > 0.0) or not math.isfinite(s):
        base.reason = "non_positive_statistic"
        return base
    log2_s = math.log2(s)
    if abs(log2_s) <= float(band_log2):
        base.reason = "within_band"
        return base
    k = -int(round(log2_s))
    if k == 0:                       # unreachable while band_log2 >= 1, kept honest
        base.reason = "within_band"
        return base
    if abs(k) > int(max_log2):
        base.reason = "factor_out_of_range"
        base.log2_factor = k
        return base
    c = float(2.0 ** k)
    base.applied = True
    base.reason = ""
    base.factor = c
    base.log2_factor = k
    base.statistic_after = s * c
    if epsilon is not None:
        base.epsilon_scaled = float(epsilon) * c
    return base


def scale_gf(V: GeneralizedForm, factor: float) -> GeneralizedForm:
    """A NEW ``GeneralizedForm`` whose every coefficient is ``factor`` times.

    Exponents, term types, weights, degree and strictness are copied
    unchanged, so the scaled form is r-homogeneous of the same degree m and has
    the same support and the same C^1 classification.  ``factor == 1.0``
    returns ``V`` ITSELF, so a no-op normalisation cannot perturb an object
    identity the caller depends on (``gf_discovery._adopt_design_bundle``
    compares ``V is V_proposal``).
    """
    c = float(factor)
    if not (c > 0.0) or not math.isfinite(c):
        raise ValueError(f"normalisation factor must be finite and > 0, got {factor!r}")
    if c == 1.0:
        return V
    if getattr(V, "composite_terms", None):
        raise ValueError("scale_gf does not scale composite terms")
    terms = [
        GFTerm(coefficient=float(t.coefficient) * c,
               exponents=tuple(t.exponents),
               term_types=tuple(t.term_types))
        for t in V.terms
    ]
    return GeneralizedForm(
        degree_m=V.degree_m,
        weights_r=tuple(V.weights_r),
        terms=terms,
        composite_terms=[],
        strict_homogeneous=V.strict_homogeneous,
    )


def scale_gram_bundle(
    entries: Optional[Sequence[Dict[str, Any]]],
    factor: float,
) -> Optional[List[Dict[str, Any]]]:
    """Rescale a design Gram bundle so it witnesses ``factor * V``.

    ``V_s - eps*sum y^d = z^T Q z`` gives
    ``cV_s - (c*eps)*sum y^d = z^T (cQ) z``, so every entry's ``gram``,
    ``target`` and ``eps`` is multiplied by ``c`` and nothing else moves: the
    basis, the octant labels, mu, the y-degree, the parity records and the
    clearing power are properties of the block, not of the scale.  A block
    that carries no margin (``eps == 0.0``, the typed P2-02 cases) keeps
    ``eps = 0.0``, since ``c * 0 == 0``: scaling cannot manufacture a margin
    where the design proved none.

    The per-entry content hash is RECOMPUTED, so the returned bundle is
    self-consistent and the exact phase's hash check passes on it.  It is a
    different proof object from the unscaled one and its fingerprint says so.
    """
    if entries is None:
        return None
    from .joint_sos import gram_bundle_entry_hash  # local: avoids an import cycle

    c = float(factor)
    if not (c > 0.0) or not math.isfinite(c):
        raise ValueError(f"normalisation factor must be finite and > 0, got {factor!r}")
    out: List[Dict[str, Any]] = []
    for entry in entries:
        scaled = dict(entry)
        gram = entry.get("gram")
        if gram is not None:
            scaled["gram"] = np.asarray(gram, dtype=np.float64) * c
        target = entry.get("target")
        if target is not None:
            scaled["target"] = {k: float(v) * c for k, v in target.items()}
        scaled["eps"] = float(entry.get("eps") or 0.0) * c
        scaled.pop("hash", None)
        scaled["hash"] = gram_bundle_entry_hash(scaled)
        out.append(scaled)
    return out


def normalize_for_gate(
    V: GeneralizedForm,
    *,
    gram_bundle: Optional[Sequence[Dict[str, Any]]] = None,
    epsilon: Optional[float] = None,
    enabled: bool = True,
    band_log2: int = NORMALIZATION_BAND_LOG2,
    max_log2: int = MAX_LOG2_FACTOR,
) -> Tuple[GeneralizedForm, Optional[List[Dict[str, Any]]], Optional[float], VNormalization]:
    """Normalise ``V`` and everything that has to travel with it.

    Returns ``(V_out, bundle_out, epsilon_out, record)``.  When nothing is
    applied the SAME ``V`` object is returned (not a copy), so a disabled or
    in-band normalisation is a literal no-op.
    """
    if not enabled:
        plan = plan_normalization(V, band_log2=band_log2, max_log2=max_log2,
                                  epsilon=epsilon)
        plan.applied = False
        plan.factor = 1.0
        plan.log2_factor = 0
        plan.statistic_after = plan.statistic
        plan.epsilon_scaled = plan.epsilon
        plan.reason = "disabled"
        plan.n_bundle_entries = 0 if gram_bundle is None else len(gram_bundle)
        return V, (None if gram_bundle is None else list(gram_bundle)), epsilon, plan

    rec = plan_normalization(V, band_log2=band_log2, max_log2=max_log2,
                             epsilon=epsilon)
    rec.n_bundle_entries = 0 if gram_bundle is None else len(gram_bundle)
    if not rec.applied:
        return V, (None if gram_bundle is None else list(gram_bundle)), epsilon, rec
    V_out = scale_gf(V, rec.factor)
    bundle_out = scale_gram_bundle(gram_bundle, rec.factor)
    eps_out = None if epsilon is None else float(epsilon) * rec.factor
    return V_out, bundle_out, eps_out, rec
