#!/usr/bin/env python3
"""Standalone exact-rational verifier for the published Lyapunov certificates.

STDLIB ONLY — no numpy, no sympy, no solver, no gf_framework import. Runs on
a bare Python >= 3.9. This is the independent checker shipped as
supplementary material with the Automatica note: everything it trusts is
(a) this file, (b) the certificate record JSON, and (c) exact rational
arithmetic (fractions.Fraction).

What it re-proves, from scratch, for one *_exactgains_*.json record produced
by scripts/exact_gate_standard_gains.py (+ its *_grams_*.json bundle):

  0. Every RAW input the arithmetic rests on is validated BEFORE it is
     coerced. Basis exponents must be nonnegative integers of the right
     dimension and of degree d/2 (a Laurent exponent satisfies 2*sum = d and
     would still break the residual and PSD-shift bounds, which are bounds on
     POLYNOMIAL monomials); the lift mu must be a positive integer vector,
     the same one for every capture of one form; sign patterns must be
     exactly {-1,+1}^n; Gram entries must be finite numbers or exact
     rationals; and the record's booleans, integers and rationals must be
     exactly that. ``int()``/``bool()``/``float()`` on a record value is never
     a substitute for a check, because it silently redefines what the record
     says (2.9 -> 2). The verifier must not inherit a premise from the
     producer (external scientific review 2026-09-05, findings 2, 5 and 13).
  1. V is an admissible C^1 generalized form: per variable, 'abs' exponents
     lie in {0} u (1, inf) and 'signed' exponents in {0} u [1, inf) — with a
     'signed' exponent 0 forbidden in V (that would be sign(x), not C^1) —
     and every term has the same weighted degree (homogeneity).
  2. The lambda-brackets are valid: lo^k < q < hi^k in exact rational
     arithmetic, where q = gain^k is the rational power that pins the
     irrational published gain (e.g. (1.5*sqrt(2))^2 = 9/2).
  3. The differentiator error dynamics are rebuilt HERE (15 lines, from the
     published equations) with the exact rational gains of each bracket
     corner and the exact delta0, and W = -grad(V) . f is recomputed exactly
     for both inclusion endpoints f +- delta0*e_ch.
  4. Every per-octant Gram capture proves its target: with the exact octant
     association of V (resp. W) minus the margin, basis B of degree d/2,
     rationalized Gram Q, residual E = target - z^T Q z computed
     coefficient-wise in Q, the gate inequality
        charge = eta*C1 + ||E||_1 < eps   with   Q + eta*I  PSD (exact LDL^T)
     holds — which proves  form >= (eps - charge) * sum_i y_i^d  on the
     octant (Proposition "exact-rational gate" of the note). The accepted
     margin is REPORTED as eps_certified = eps - charge; it, and never the
     requested eps, is what the certificate proves.

     Acceptance rule (WP P2-14, owner decision D-l of 2026-09-04): accept at
     charge < eps, STRICTLY, and report the achieved margin. The note prints
     the stricter threshold charge <= eps/2, which yields the rounder
     statement form >= (eps/2) * sum_i y_i^d. That inequality is a CONVENTION
     about how much of the requested margin the certificate must retain, not
     a step of the proof, so this verifier does not enforce it — but it does
     compute it, per octant and rolled up, and prints it as
     note_half_margin_holds, so a reader can see at once whether a record is
     inside the note's printed sentence.

     The PSD proposal eta is searched over {0, decimal ladder, exact
     Gershgorin bound}; the LDL^T factorization is the sole authority, so a
     PASS is sound regardless of how eta was proposed. The PSD test runs per
     parity block of the target's deck group (derived here, not read from the
     record); ``--dense`` selects the single-block reference path, which is
     the CONSERVATIVE one: a dense PASS implies the blocked PASS at the same
     charge, never the other way round. The two agree exactly when the same
     eta rung is selected — as on every stored record, whose captured Gram is
     already PSD so both take eta = 0 — and a difference on an indefinite
     Gram is sound, not a bug. See ``validate_octant``.
  5. COVERAGE is complete: all 2^n octant sign patterns (for V and for every
     corner x endpoint W), all 2^k bracket corners (distinct assignments,
     every coordinate a bracket endpoint — by pigeonhole this IS the full
     {lo, hi}^k product), and exactly two inclusion endpoints per corner.
     An incomplete record is REJECTED, because step 6 needs completeness:
  6. Affinity closes the argument (in the note, not in code): W is affine in
     delta and in each gain, so PASS at both endpoints and at all bracket
     corners certifies every disturbance in [-delta0, delta0] and every gain
     value in the brackets — including the irrational published gains. Every
     capture's margin eps must equal the record's epsilon_exact.

Record schema versions
----------------------
``gf-certificate-record/1`` (the four Automatica-note records): perturbed
only, one system family (Levant's differentiator), JSON numbers.

``gf-certificate-record/2`` (campaign item C9, emitted by
``scripts/emit_certificate_record.py``): backwards compatible, and adds

  * ``system_family`` / ``channel`` / ``nominal`` / ``weights_r`` / ``kappa``,
    so a record states which published equations it is about instead of
    leaving that to the file name. ``channel`` is null exactly for nominal
    records and for systems that admit no constant matched disturbance;
  * NOMINAL records: ``delta0_exact = "0"``, no inclusion endpoints, one
    dynamics per bracket corner, whose captures live under ``W_grams``. The
    completeness rule becomes 2^n octants for V and 2^n per corner for W;
  * ``env`` / ``campaign_block`` / ``source_artifacts`` provenance;
  * a compact encoding: every float64 may appear as its exact hex string
    (``float.hex()``), every exact value as a rational string ``"p/q"``, and
    both record and grams bundle may be gzipped (``.json.gz``, stdlib gzip).
    Version 1 records keep working unchanged: JSON numbers still decode;
  * a SHARED nominal W witness (2026-09-04, WP P4-04): when one Gram bundle
    witnesses W at every bracket corner — which is what a design-solver
    bundle is, the Gram being gain-independent while the target is not — the
    captures may be written ONCE under the bundle's ``shared_W_grams`` and
    each corner may carry ``"W_grams_shared": true`` in place of its own
    ``W_grams`` list. This is storage aliasing on the decode side and nothing
    else: every corner still rebuilds its W from ITS OWN exact gains and
    still proves all 2^n octants against that Gram, so the proof obligations
    are unchanged and only the 2^k duplicate copies of the bytes are not
    written. Perturbed records may not use it (see ``nominal_w_captures``).

Nothing about the DYNAMICS is ever read from a record. The equations are
hand-written in this file, per family, from the published sources; the record
supplies V, the gains, delta0, the margin, and the Gram captures only.

Usage:
    python3 verify_certificate.py RECORD.json[.gz] [--grams GRAMS.json[.gz]]
                                  [--row ROW.json] [--dense] [--json OUT.json]

--row is only needed for records that predate the embedded-V format.

Outcome classes and exit codes (2026-11-05, Moreno's evaluation, Section 6)
----------------------------------------------------------------------------
Every run ends with one ``RESULT:`` line naming its class:

  0  VERIFIED       every check above passed. Exit code 0 iff everything
                    verifies; this is the only accepting outcome.
  1  REJECTED       the inputs were read as a certificate record and a
                    MATHEMATICAL check failed: V not an admissible C^1 form of
                    the declared degree, an invalid bracket, incomplete
                    coverage (octants, bracket corners, inclusion endpoints, a
                    form without a witness), an octant inequality (charge >= eps
                    or no PSD shift), a capture at a margin other than the
                    record's, a lift that does not make a form an even
                    polynomial, or a declared quantity (kappa, y_degree, parity
                    blocks, the certified margin: "MARGIN CLAIM EXCEEDS PROOF")
                    that contradicts what is derived here. stderr carries
                    ``VERIFICATION FAILED: <reason>``.
  2  INVALID INPUT  the inputs could not be read as a certificate record: a
                    missing or unreadable file, a truncated gzip stream,
                    malformed JSON, an unknown schema version or system, a
                    missing field, a value of the wrong type, shape or encoding,
                    or declarations that contradict each other or the system the
                    record names (family, weights, channel, nominal flag). Also
                    command-line usage errors (argparse). stderr carries
                    ``INVALID INPUT: <reason>``. Not a mathematical rejection.
  3  ERROR          an unexpected condition inside the verifier (for example
                    MemoryError); traceback on stderr, no verdict.

The classification is REPORTING ONLY. It decides the message and the exit
code, never whether a record is accepted: the arithmetic and every acceptance
condition are those of steps 0 to 6, and each refusal path refuses exactly as
before. A REJECTED verdict is a statement about the bytes that were read; a
record or Gram bundle that does not match its published sha256 is not the
published certificate. ``main()`` still raises :class:`VerifyError` (its
subclasses :class:`RejectedError` and :class:`InvalidInputError`) for in-process
callers; only the command-line wrapper :func:`cli` maps outcomes to exit codes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import traceback
import zlib
from fractions import Fraction
from itertools import product
from pathlib import Path

#: Record schemas this verifier accepts. An unknown version is refused rather
#: than read on a best-effort basis: a future field could carry a weakened
#: completeness convention that this code would silently not enforce.
SCHEMA_VERSIONS = ("gf-certificate-record/1", "gf-certificate-record/2")

# ---------------------------------------------------------------------------
# Record I/O: gzip transparency and the compact number encoding (schema v2).
# Both are decode-side only; nothing here decides anything about a proof.
# ---------------------------------------------------------------------------

def read_json(path, bindings=None, role=None):
    """Parse a JSON file that may be gzipped, detected by the gzip magic.

    Sniffing the magic rather than the suffix keeps a renamed bundle readable
    and keeps v1 records (plain .json) working untouched.
    """
    what = role or "input"
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read the {what} file {str(path)!r}: "
                                f"{exc.strerror or exc}")
    if bindings is not None:
        bindings[role] = {"path": str(Path(path).resolve()),
                          "sha256": hashlib.sha256(raw).hexdigest()}
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError, zlib.error) as exc:
            raise InvalidInputError(f"the {what} file {str(path)!r} is not a "
                                    f"readable gzip stream (truncated or "
                                    f"corrupted download?): {exc}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidInputError(f"the {what} file {str(path)!r} is not valid "
                                f"UTF-8 JSON: {exc}")


def resolve_sibling(directory, name):
    """Locate a bundle named in a record, allowing a .gz counterpart."""
    for candidate in (directory / name, directory / (name + ".gz"),
                      directory / name.replace(".gz", "")):
        if candidate.exists():
            return candidate
    raise InvalidInputError(f"grams bundle {name!r} not found next to the record")


def parse_rational(text, what):
    """``Fraction(text)`` with an unparseable string turned into a refusal.

    Every rational a record carries goes through here, so a malformed number
    leaves the same clean ``VERIFICATION FAILED`` line as any other refusal
    instead of a ValueError traceback.
    """
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        raise InvalidInputError(f"{what}: {text!r} is not a rational number ({exc})")


def q_from(value):
    """Exact Fraction from a record number, in any of the accepted encodings.

    * JSON number            -> the exact value of that float64
    * ``"0x1.8p+0"``         -> float.hex() round-trip, the exact float64 again
    * ``"3/2"`` / ``"1.5"``  -> the exact rational it denotes

    The hex form exists because it survives every JSON writer bit for bit;
    ``Fraction(float(...))`` and this function agree on v1 records.
    """
    if isinstance(value, bool):
        raise InvalidInputError(f"boolean {value!r} where a number was expected")
    if isinstance(value, str):
        text = value.strip()
        low = text.lower()
        if low.startswith(("0x", "-0x", "+0x")):
            parsed = float.fromhex(text)
            if parsed != parsed or parsed in (float("inf"), float("-inf")):
                raise InvalidInputError(f"non-finite hex float {text!r}")
            return Fraction(parsed)
        if any(bad in low for bad in ("inf", "nan")):
            raise InvalidInputError(f"non-finite value {text!r}")
        return parse_rational(text, "number")
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, float):
        # No best-effort float() of a list, dict or None: a Gram entry that is
        # not a number is a malformed capture, not a value to guess at.
        raise InvalidInputError(f"{type(value).__name__} {value!r} where a number "
                          f"was expected")
    if value != value or value in (float("inf"), float("-inf")):
        raise InvalidInputError(f"non-finite value {value!r}")
    return Fraction(value)


def q_exact(value, what):
    """Exact rational from a field that must NOT be a float64 in disguise.

    The gains, delta0, the margin and the bracket endpoints are the PUBLISHED
    numbers, and the whole point of the record is that they enter the
    arithmetic as themselves: 11/10, not the dyadic float64 nearest to 1.1.
    So the float encodings that ``q_from`` accepts for Gram entries are
    refused here, and 1.1 written as a hex float is an error, not a value.
    """
    if isinstance(value, bool):
        raise InvalidInputError(f"{what}: boolean where a number was expected")
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, str):
        raise InvalidInputError(f"{what} must be an exact rational string, got "
                          f"{type(value).__name__} {value!r}; a float64 cannot "
                          f"carry a published decimal exactly")
    text = value.strip()
    if text.lower().startswith(("0x", "-0x", "+0x")):
        raise InvalidInputError(f"{what} is hex float64 encoded ({text!r}); the "
                          f"published value must enter as an exact rational")
    return parse_rational(text, what)


# ---------------------------------------------------------------------------
# Raw input validation (external scientific review 2026-09-05, findings 2/5/13)
#
# The premises of the exact bounds are established HERE, on the raw record
# values, and never inherited from whoever produced them. Coercion is the
# defect these functions replace: ``int(beta)`` turns a Laurent exponent into
# a plausible-looking basis vector, ``int(mu_i)`` turns a declared lift of 2.9
# into 2, and ``bool(field)`` turns any truthy JSON value into True. Each of
# those silently changes the mathematical statement being proved.
# ---------------------------------------------------------------------------

def require_bool(value, what):
    """A JSON boolean, never a truthy value that ``bool()`` would accept."""
    if not isinstance(value, bool):
        raise InvalidInputError(f"{what} must be a JSON boolean, got "
                          f"{type(value).__name__} {value!r}")
    return value


def require_int(value, what):
    """An exact integer from a RAW record value, before any coercion.

    Accepts a JSON integer, a JSON float that is exactly integral (2.0), and a
    string denoting an exact integer ("2", "4/2"). Refuses booleans (True is
    not 1 here), floats with a fractional part (2.9 must never become 2),
    non-finite floats, and every other type.
    """
    if isinstance(value, bool):
        raise InvalidInputError(f"{what}: boolean {value!r} where an integer was "
                          f"expected")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidInputError(f"{what}: non-finite {value!r}")
        if not value.is_integer():
            raise InvalidInputError(f"{what}: {value!r} is not an integer")
        return int(value)
    if isinstance(value, str):
        q = parse_rational(value.strip(), what)
        if q.denominator != 1:
            raise InvalidInputError(f"{what}: {value!r} is not an integer")
        return int(q)
    raise InvalidInputError(f"{what}: {type(value).__name__} {value!r} where an "
                      f"integer was expected")


def require_vector(value, n, what):
    """A list/tuple of exactly ``n`` entries (a string is never a vector)."""
    if isinstance(value, (str, bytes, dict)) or not isinstance(
            value, (list, tuple)):
        raise InvalidInputError(f"{what} must be a list of {n} entries, got "
                          f"{type(value).__name__} {value!r}")
    if len(value) != n:
        raise InvalidInputError(f"{what} has {len(value)} entries, need {n}")
    return list(value)


def require_int_vector(value, n, what, minimum=None):
    """``n`` raw integers, optionally each ``>= minimum``."""
    out = []
    for i, entry in enumerate(require_vector(value, n, what)):
        v = require_int(entry, f"{what}[{i}]")
        if minimum is not None and v < minimum:
            raise InvalidInputError(f"{what}[{i}] = {v} violates >= {minimum}")
        out.append(v)
    return tuple(out)


def validate_mu(value, n_vars, what):
    """The declared association lift: exactly ``n_vars`` POSITIVE integers.

    x_i = s_i y_i^{mu_i} is a valid octant chart only for integral mu_i >= 1:
    mu_i = 0 collapses a coordinate, a negative mu_i is not a chart at all,
    and a fractional one is not the substitution the record declares. Finding
    5 of the 2026-09-05 review: this was ``tuple(int(v) for v in mu)``, so
    2.9 was read as 2 and the arithmetic proved a different statement than the
    record claimed.
    """
    return require_int_vector(value, n_vars, f"{what}: mu", minimum=1)


def validate_signs(value, n_vars, what, strict=False):
    """The octant sign label: exactly ``n_vars`` raw integers.

    ``strict`` additionally demands each entry be exactly +1 or -1. Callers
    inside :func:`verify_captures` leave it off because the coverage check
    there (the sign set must EQUAL {-1,+1}^n) is the stronger statement and
    owns that refusal message.
    """
    signs = require_int_vector(value, n_vars, f"{what}: signs")
    if strict and any(s not in (-1, 1) for s in signs):
        raise InvalidInputError(f"{what}: sign pattern {signs} is not in "
                          f"{{-1,+1}}^{n_vars}")
    return signs


def validate_basis(basis, d, n_vars, what):
    """The Gram basis: nonnegative integer exponent vectors of degree d/2.

    Finding 2 of the 2026-09-05 review. ``tuple(int(b) for b in beta)`` plus
    ``2*sum(beta) == d`` accepts a homogeneous LAURENT monomial such as
    (8, -4) at d = 8, and the gate's two bounds then do not hold:

      * the residual bound charges ||E||_1 and concludes
        |E(y)| <= ||E||_1 * sum_i y_i^d, which is an inequality about
        monomials y^alpha with alpha >= 0 and |alpha| = d. A monomial with a
        negative exponent is unbounded as that coordinate goes to zero;
      * the PSD shift charges eta*C1 because z^T z <= C1 * sum_i y_i^d for
        the vector z of basis monomials, which again needs alpha >= 0.

    So a Laurent basis lets a positive-definite Gram "prove" a form that is
    negative somewhere (the review's V = -x1^2 + 4 x2^4). Duplicates are
    refused too: a repeated basis vector is a degenerate Gram indexing that no
    honest capture produces, and it makes the derived parity blocks ambiguous.
    """
    if isinstance(basis, (str, bytes, dict)) or not isinstance(
            basis, (list, tuple)):
        raise InvalidInputError(f"{what}: basis must be a list, got "
                          f"{type(basis).__name__}")
    if len(basis) == 0:
        raise InvalidInputError(f"{what}: empty basis")
    out = []
    seen = set()
    for k, beta in enumerate(basis):
        b = require_int_vector(beta, n_vars, f"{what}: basis[{k}]", minimum=0)
        if 2 * sum(b) != d:
            raise InvalidInputError(f"{what}: basis[{k}] = {b} has degree {sum(b)}, "
                              f"need d/2 = {d}/2")
        if b in seen:
            raise InvalidInputError(f"{what}: basis[{k}] = {b} is a duplicate")
        seen.add(b)
        out.append(b)
    return out


def validate_target(target, d, what):
    """The octant target: nonnegative integer monomials, all of degree d.

    Returns the number of variables. The same premise as the basis check, on
    the other side of the residual: E = target - z^T Q z is charged in the
    coefficient 1-norm, and that norm bounds the polynomial only over
    monomials of a common nonnegative degree d.
    """
    if not isinstance(target, dict) or not target:
        raise RejectedError(f"{what}: octant target must be a non-empty dict")
    n_vars = None
    for mono, coeff in target.items():
        if not isinstance(mono, tuple):
            raise RejectedError(f"{what}: target key {mono!r} is not a tuple")
        if n_vars is None:
            n_vars = len(mono)
        mono_i = require_int_vector(mono, n_vars, f"{what}: target monomial",
                                    minimum=0)
        if sum(mono_i) != d:
            raise RejectedError(f"{what}: target monomial {mono_i} has degree "
                              f"{sum(mono_i)}, need {d}")
        if isinstance(coeff, bool) or not isinstance(coeff, (int, Fraction)):
            raise RejectedError(f"{what}: target coefficient {coeff!r} of "
                              f"{mono_i} is not an exact rational")
    if d <= 0 or d % 2 != 0:
        raise RejectedError(f"{what}: target degree {d} is not positive even")
    return n_vars


def validate_gf_exponent(value, what):
    """One GF exponent: an exact NONNEGATIVE rational, never a coerced float.

    Exponents carry the homogeneity of V and of W; a negative one would leave
    the polynomial world the octant lift maps into, and a string that is not a
    rational is a malformed record rather than something to guess at.
    """
    if isinstance(value, bool):
        raise InvalidInputError(f"{what}: boolean where an exponent was expected")
    if isinstance(value, Fraction):
        q = value
    elif isinstance(value, int):
        q = Fraction(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidInputError(f"{what}: non-finite exponent {value!r}")
        q = Fraction(value)
    elif isinstance(value, str):
        q = parse_rational(value.strip(), what)
    else:
        raise InvalidInputError(f"{what}: {type(value).__name__} {value!r} where an "
                          f"exponent was expected")
    if q < 0:
        raise InvalidInputError(f"{what}: negative exponent {q}")
    return q


def validate_gf_terms(raw_terms, n_vars, what):
    """Read the record's V as exact terms, checking shape before arithmetic.

    Every term needs exactly ``n_vars`` exponents and ``n_vars`` factor types
    drawn from {'abs', 'signed'} — the two the term algebra of this file
    implements. An unknown type name would otherwise be read as 'signed' by
    the ``!= 'abs'`` branches and silently change the form.
    """
    if isinstance(raw_terms, (str, bytes, dict)) or not isinstance(
            raw_terms, (list, tuple)):
        raise InvalidInputError(f"{what}: terms must be a list, got "
                          f"{type(raw_terms).__name__}")
    if not raw_terms:
        raise InvalidInputError(f"{what}: no terms")
    out = []
    for k, t in enumerate(raw_terms):
        if not isinstance(t, dict):
            raise InvalidInputError(f"{what}: term {k} is not an object")
        for field in ("coefficient", "exponents", "term_types"):
            if field not in t:
                raise InvalidInputError(f"{what}: term {k} has no {field!r}")
        exps = tuple(
            validate_gf_exponent(e, f"{what}: term {k} exponent {i}")
            for i, e in enumerate(
                require_vector(t["exponents"], n_vars,
                               f"{what}: term {k} exponents")))
        types = []
        for i, ty in enumerate(require_vector(t["term_types"], n_vars,
                                              f"{what}: term {k} term_types")):
            if ty not in ("abs", "signed"):
                raise InvalidInputError(f"{what}: term {k} factor type {ty!r} is "
                                  f"neither 'abs' nor 'signed'")
            types.append(ty)
        out.append((q_from(t["coefficient"]), exps, tuple(types)))
    return out


# ---------------------------------------------------------------------------
# Exact GF-term algebra (vendored from gf_framework.verification.exact_gate —
# kept dependency-free on purpose; a term is (coeff, exponents, types) with
# types in {'abs', 'signed'}: |x|^a resp. |x|^a*sign(x); exponent 0 with
# 'signed' means sign(x)).
# ---------------------------------------------------------------------------

def exact_partial(term, var_idx):
    coeff, exps, types = term
    e = exps[var_idx]
    if e == 0:
        return None
    new_exps = list(exps)
    new_exps[var_idx] = e - 1
    new_types = list(types)
    new_types[var_idx] = 'signed' if types[var_idx] == 'abs' else 'abs'
    return (coeff * e, tuple(new_exps), tuple(new_types))


def exact_multiply(t1, t2):
    c = t1[0] * t2[0]
    exps = tuple(a + b for a, b in zip(t1[1], t2[1]))
    types = tuple('abs' if a == b else 'signed' for a, b in zip(t1[2], t2[2]))
    return (c, exps, types)


def exact_collect(terms):
    acc = {}
    for c, e, t in terms:
        key = (e, t)
        acc[key] = acc.get(key, Fraction(0)) + c
    return [(c, e, t) for (e, t), c in acc.items() if c != 0]


def exact_neg_lie_derivative(v_terms, dynamics_terms):
    """W = -dV/dt = -sum_i (dV/dx_i) * f_i, exactly."""
    out = []
    for i, f_i in enumerate(dynamics_terms):
        for vt in v_terms:
            d = exact_partial(vt, i)
            if d is None:
                continue
            for ft in f_i:
                c, e, t = exact_multiply(d, ft)
                out.append((-c, e, t))
    return exact_collect(out)


def exact_octant_target(terms, mu, signs, eps):
    """Octant association (x_i = s_i y_i^{mu_i}) minus the margin eps.

    ``mu`` and ``signs`` are validated RAW (positive integers resp. exactly
    +-1, both of length n) before anything is coerced, and every term must
    carry n exponents and n factor types: the lift is a premise of the whole
    chain, not a hint from the producer (review 2026-09-05, finding 5).
    """
    if isinstance(mu, (str, bytes, dict)) or not isinstance(mu, (list, tuple)):
        raise InvalidInputError(f"mu must be a list, got {type(mu).__name__} {mu!r}")
    n = len(mu)
    mu = validate_mu(mu, n, "octant")
    signs = validate_signs(signs, n, "octant", strict=True)
    target = {}
    for c, exps, types in terms:
        require_vector(exps, n, "form term exponents")
        for ty in require_vector(types, n, "form term types"):
            if ty not in ("abs", "signed"):
                raise InvalidInputError(f"form term factor type {ty!r} is neither "
                                  f"'abs' nor 'signed'")
        y_exp = []
        sign = 1
        for i in range(n):
            ye = mu[i] * exps[i]
            if ye.denominator != 1 or ye < 0:
                raise RejectedError(f"non-integer/negative y-exponent {ye}")
            y_exp.append(int(ye))
            if types[i] == 'signed':
                sign *= signs[i]
        key = tuple(y_exp)
        target[key] = target.get(key, Fraction(0)) + sign * c
    target = {k: v for k, v in target.items() if v != 0}
    if not target:
        raise RejectedError("octant target identically zero")
    degrees = {sum(k) for k in target}
    if len(degrees) != 1:
        raise RejectedError(f"target not homogeneous: degrees {sorted(degrees)}")
    d = degrees.pop()
    if d <= 0 or d % 2 != 0:
        raise RejectedError(f"target degree {d} not positive even")
    for i in range(n):
        pure = tuple(d if t == i else 0 for t in range(n))
        target[pure] = target.get(pure, Fraction(0)) - eps
    return target, d


# ---------------------------------------------------------------------------
# Deck group of the octant target and the Gram parity blocks it induces.
#
# Pure integer arithmetic (bitmasks mod 2), ~70 lines, no floats, no imports.
# This is the only speed structure in the file, and it is derived HERE from the
# target support and the basis exponents. Nothing about it is read from the
# record: a record MAY state its block structure, but it is re-derived and the
# claim is checked against the derivation (see ``check_declared_blocks``),
# never trusted.
# ---------------------------------------------------------------------------

def _f2_rref(rows, n_vars):
    """Reduced row echelon form over F_2 of bitmask-encoded rows.

    Returns (reduced_rows, pivot_columns): one pivot per reduced row, zeros in
    every other pivot column, so the null space reads off directly.
    """
    work = [r for r in rows if r]
    pivots = []
    rank = 0
    for col in range(n_vars):
        pivot_row = None
        for i in range(rank, len(work)):
            if (work[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        for i in range(len(work)):
            if i != rank and ((work[i] >> col) & 1):
                work[i] ^= work[rank]
        pivots.append(col)
        rank += 1
        if rank == len(work):
            break
    return work[:rank], pivots


def deck_group_basis(support, n_vars):
    """Basis of H = {g in F_2^n : <g, alpha> = 0 (mod 2) for all alpha in support}.

    H is the group of sign flips y_i -> -y_i that leave the octant target
    invariant: it is the null space over F_2 of the support exponents reduced
    mod 2. An all-even support gives only zero rows, hence H = F_2^n. A basis
    of length 0 means no usable symmetry, and the caller falls back to the
    dense path (one block spanning every index).
    """
    rows = []
    for alpha in support:
        mask = 0
        for i in range(n_vars):
            if int(alpha[i]) & 1:
                mask |= 1 << i
        if mask:
            rows.append(mask)
    reduced, pivot_cols = _f2_rref(rows, n_vars)
    pivot_set = set(pivot_cols)
    basis = []
    for free in (c for c in range(n_vars) if c not in pivot_set):
        g = 1 << free
        for k, pcol in enumerate(pivot_cols):
            # row k reads  g_{pcol} + sum_{free c in row k} g_c = 0
            if (reduced[k] >> free) & 1:
                g |= 1 << pcol
        basis.append(tuple((g >> i) & 1 for i in range(n_vars)))
    return basis


def parity_blocks(basis, group_basis):
    """Partition Gram basis indices into the deck group's character classes.

    Basis exponents beta, beta' may pair in the H-symmetrized Gram iff
    <g, beta + beta'> = 0 (mod 2) for every g in H, i.e. iff they carry the
    same character (<g, beta> mod 2)_{g in H}. Returns one index list per
    non-empty class, ordered by class key, each list increasing (so a single
    class reproduces the dense matrix exactly, index for index).
    """
    if not group_basis:
        return [list(range(len(basis)))]
    classes = {}
    for idx, beta in enumerate(basis):
        key = tuple(sum(g[i] * int(beta[i]) for i in range(len(g))) & 1
                    for g in group_basis)
        classes.setdefault(key, []).append(idx)
    return [classes[k] for k in sorted(classes)]


#: Field names under which a record could claim a Gram block structure. None of
#: the shipped records carries one; the check exists so that a future record
#: that does cannot smuggle a partition past the derivation.
BLOCK_FIELDS = ("blocks", "parity_blocks", "block_structure", "block_sizes")


def check_declared_blocks(cap, blocks, label):
    """Refuse a record whose claimed block structure is not the derived one.

    The derivation above is the authority. A declared partition is compared
    against it and a mismatch is an error, so the record can never widen a
    block (which would hide a cross-class Gram entry from the residual).
    """
    for field in BLOCK_FIELDS:
        if field not in cap:
            continue
        claimed = cap[field]
        if field == "block_sizes":
            derived = [len(b) for b in blocks]
            ok = [int(v) for v in claimed] == derived
        else:
            derived = [sorted(int(i) for i in b) for b in blocks]
            ok = sorted([sorted(int(i) for i in b) for b in claimed]) == \
                sorted(derived)
        if not ok:
            raise RejectedError(
                f"{label}: record field {field!r} = {claimed!r} contradicts the "
                f"parity blocks derived here ({derived!r})")


def ldl_psd(A):
    """Exact PSD test: symmetric elimination, largest-diagonal pivoting.
    Negative pivot => not PSD; zero max diagonal => remaining block must be 0;
    else eliminate. All in Q. THE proof authority of this file."""
    n = len(A)
    M = [list(row) for row in A]
    remaining = list(range(n))
    while remaining:
        p = max(remaining, key=lambda i: M[i][i])
        piv = M[p][p]
        if piv < 0:
            return False
        if piv == 0:
            return all(M[i][j] == 0 for i in remaining for j in remaining)
        remaining.remove(p)
        col = {i: M[i][p] for i in remaining}
        for i in remaining:
            ci = col[i]
            if ci == 0:
                continue
            row_i = M[i]
            for j in remaining:
                cj = col[j]
                if cj != 0:
                    row_i[j] -= ci * cj / piv
    return True


class VerifyError(Exception):
    """Every refusal of this verifier; never raised directly (module
    docstring, "Outcome classes"). The two subclasses below carry the class
    of the refusal, which decides the message prefix and the exit code. The
    classification is reporting only: which subclass a check raises never
    changes WHETHER a record is accepted."""
    outcome = "REJECTED"
    exit_code = 1


class RejectedError(VerifyError):
    """REJECTED (exit 1): the inputs were read as a certificate record and a
    MATHEMATICAL check failed on them: V not an admissible C^1 form of the
    declared degree, a bracket that does not enclose its gain, incomplete
    coverage (octants, bracket corners, inclusion endpoints, a form with no
    witness), a form outside the premises of the Gram lemma, an octant
    inequality, a capture at another margin, or a declaration (kappa,
    y_degree, parity blocks, certified margin) that contradicts what this
    file derives."""
    outcome = "REJECTED"
    exit_code = 1


class InvalidInputError(VerifyError):
    """INVALID INPUT (exit 2): the inputs could not be read as a certificate
    record for a system this file knows: a missing or unreadable file, data
    that is not (gzipped) JSON, an unknown schema version or system, a
    missing field, a value of the wrong type, shape or encoding, or a record
    whose own declarations are inconsistent with each other or with the
    system it names. Nothing about the certificate is decided by it."""
    outcome = "INVALID INPUT"
    exit_code = 2


#: Exit codes of the command line (module docstring, "Outcome classes").
EXIT_VERIFIED = 0
EXIT_REJECTED = RejectedError.exit_code
EXIT_INVALID_INPUT = InvalidInputError.exit_code
EXIT_ERROR = 3


# ---------------------------------------------------------------------------
# The gate inequality for one captured octant (numpy-free: the eta proposal
# uses an exact Gershgorin bound instead of a float eigenvalue estimate)
# ---------------------------------------------------------------------------

def blocks_psd(Q, blocks, eta):
    """``Q_H + eta*I >= 0`` by exact LDL^T on each parity block.

    A block-diagonal matrix is PSD iff every block is, so the O(N^3) exact
    elimination is replaced by sum_b O(N_b^3). One block spanning every index
    IS the dense test on ``Q + eta*I``, entry for entry.
    """
    for idxs in blocks:
        sub = [[Q[i][j] + eta if i == j else Q[i][j] for j in idxs]
               for i in idxs]
        if not ldl_psd(sub):
            return False
    return True


def validate_octant(target, d, gram_rows, basis, eps, blocked=True, cap=None,
                    label=""):
    """Prove ``form >= (eps - charge) * sum_i y_i^d`` on one octant.

    Acceptance is ``charge = eta*C1 + ||E||_1 < eps``, strictly, in exact
    rational arithmetic (WP P2-14). The returned ratio is ``charge / eps``:
    ``< 1`` is this acceptance rule, and ``<= 1/2`` is the ``charge <= eps/2``
    inequality the frozen Automatica note prints. The denominator is named in
    every field and message, because before P2-14 the same ratio was taken
    against ``eps/2`` and is therefore twice this number in older records.

    ``blocked`` (default) exhibits the H-symmetrized Gram Q_H instead of Q: the
    cross-parity-class entries of Q are zeroed, which is the Reynolds
    projection of Q over the deck group H of ``target``. Two facts make this
    both sound and free:

      * pinching. Q_H is an average of D Q D over the sign-flip diagonals
        D in H, each of which is orthogonally similar to Q. Hence
        lambda_min(Q_H) >= lambda_min(Q): any shift eta that works for Q
        works for Q_H, and the eta ladder below is unchanged.
      * the residual is charged DENSELY, over every index pair, including the
        cross-class ones Q_H drops. The dropped entries can only hit monomials
        beta + beta' that are non-invariant under H, and an H-invariant target
        has no coefficient there, so ||target - z^T Q_H z||_1 <= the number
        charged here.

    What that does and does not imply about the two paths (corrected
    2026-09-05, external review finding 12). Both paths charge the SAME dense
    residual, so for one and the same eta they compute the same charge; what
    differs is only which eta passes the PSD test, and pinching makes that
    test easier for Q_H. Hence:

      * a DENSE acceptance implies the blocked acceptance, at the same eta and
        therefore at the same charge. The blocked path is the weaker test and
        the dense path the conservative reference;
      * the converse is FALSE, and that is sound, not a bug. On basis
        {(1,0),(0,1)}, target 0, eps = 1 and Q = [[0, 2/5], [2/5, 0]] the
        blocked path accepts at charge 4/5 (Q_H is the zero matrix, PSD at
        eta = 0) while the dense path needs the Gershgorin rung eta = 2/5 and
        rejects at charge 6/5. Both verdicts are correct: projection cannot
        lower the smallest eigenvalue, and the dense residual over-charges the
        projected one;
      * the two paths return the identical verdict and the identical rational
        charge exactly when the same eta rung is selected, since the charge at
        a given eta is the same number in both. Two sufficient conditions: the
        captured Q is itself PSD, so eta = 0 is taken in both (this is the
        case on every stored record — a healthy capture is PSD, and the charge
        is then the residual alone); or the cross-parity entries of Q are
        zero, so Q_H = Q and the two paths are literally the same test.
        Nothing forces agreement on an arbitrary indefinite Gram, and a
        difference there must not be "fixed" into an equality.

    Returns ``(ok, charge/eps as an exact Fraction or None, why, shape)``
    with ``shape = (n_basis, n_blocks, largest block)``; on acceptance,
    ``shape`` carries the accepted shift eta as a fourth entry (reported
    only, never part of the verdict: the paper's Gram lemma is the eta = 0
    case, and the summary states whether every octant met it).
    """
    # Raw validation of every premise, BEFORE any coercion (finding 2): the
    # target fixes the number of variables and the degree, the basis must be
    # nonnegative integer exponents of that dimension and of degree d/2, and
    # the Gram must be a square array of numbers of matching size.
    what = label or "octant"
    nv = validate_target(target, d, what)
    basis = validate_basis(basis, d, nv, what)
    n_basis = len(basis)
    if not hasattr(gram_rows, "__len__") or isinstance(gram_rows, (str, dict)):
        raise InvalidInputError(f"{what}: gram must be a square array, got "
                          f"{type(gram_rows).__name__}")
    if len(gram_rows) != n_basis or any(
            not hasattr(r, "__len__") or len(r) != n_basis for r in gram_rows):
        raise InvalidInputError(f"{what}: gram shape mismatch")

    # Symmetrized exact rationalization of the captured Gram
    Q = [[(q_from(gram_rows[i][j]) + q_from(gram_rows[j][i])) / 2
          for j in range(n_basis)] for i in range(n_basis)]

    # Parity blocks of the deck group of THIS target (never of a float-collected
    # one, and never a partition the record supplied). Derived in both paths, so
    # a record that claims a partition is contradicted either way.
    derived = parity_blocks(basis, deck_group_basis(list(target), nv))
    if cap is not None:
        check_declared_blocks(cap, derived, label)
    blocks = derived if blocked else [list(range(n_basis))]
    shape = (n_basis, len(blocks), max((len(b) for b in blocks), default=0))

    # E = target - z^T Q z, coefficient-wise, exact, over ALL pairs. Charging
    # the DENSE residual in the blocked path too is what makes the implication
    # one-way and sound (see the docstring, review finding 12): a dense PASS
    # implies the blocked PASS, at the same eta and hence at the same charge.
    # The charges and margins coincide only when both paths select the same
    # eta rung -- in particular on a captured Q that is already PSD, where
    # both take eta = 0 -- never as a general identity.
    E = dict(target)
    for i in range(n_basis):
        for j in range(n_basis):
            q = Q[i][j]
            if q == 0:
                continue
            mono = tuple(basis[i][k] + basis[j][k] for k in range(nv))
            E[mono] = E.get(mono, Fraction(0)) - q
    resid_l1 = sum(abs(v) for v in E.values())

    c1 = max((Fraction(sum(2 * beta[i] for beta in basis), d) for i in range(nv)),
             default=Fraction(0))
    # WP P2-14: the acceptance threshold IS eps, and the test below is strict,
    # so an accepted octant always has a strictly positive certified margin
    # eps - charge. The note's eps/2 is reported separately, never enforced.
    budget = eps

    # Exact Gershgorin lower bound on the spectrum: Q + eta_G*I is diagonally
    # dominant with nonnegative diagonal, hence PSD — the ladder's last rung.
    # Taken on the DENSE Q, so the ladder is the same in both paths; by pinching
    # it bounds Q_H too (at worst conservatively).
    eta_g = max((sum(abs(Q[i][j]) for j in range(n_basis) if j != i) - Q[i][i]
                 for i in range(n_basis)), default=Fraction(0))
    candidates = {Fraction(0)}
    candidates.update(Fraction(1, 10 ** k) for k in range(15, 8, -1))
    if eta_g > 0:
        candidates.add(eta_g)
    # Ascending order makes the early budget return sound AND ensures the
    # guaranteed-PSD Gershgorin rung is never preempted by a larger rung.
    for eta in sorted(candidates):
        charge = eta * c1 + resid_l1
        if charge >= budget:
            # Larger eta only increases the charge — every later rung fails too.
            return False, charge / budget, "budget_exceeded", shape
        if blocks_psd(Q, blocks, eta):
            return True, charge / budget, "", shape + (eta,)
    return False, None, "no_psd_shift_found", shape


def nominal_w_captures(bundle, bc):
    """The nominal W captures of one corner, resolving a SHARED witness set.

    A corner carries either its own ``W_grams`` list, or ``"W_grams_shared":
    true``, in which case the captures are the bundle's top-level
    ``shared_W_grams`` list. Exactly one of the two, always: a corner with
    both would leave it to this function to pick which Gram was the witness,
    and a corner with neither has no witness at all.

    Why the alias exists (WP P4-04). The witness of a nominal row is one
    per-octant Gram set from the design solver. The Gram does NOT depend on
    the gains — only the octant TARGET does, through W = -grad(V).f — so the
    same 2^n Grams witness every one of the 2^k bracket corners, and writing
    them out 2^k times multiplies the sidecar by 2^k without adding a single
    proof obligation. At n = 5, m = 11 that is the difference between a 73 MB
    sidecar and a 254 MB one.

    What it does NOT change. Every corner still rebuilds its dynamics from
    its own exact gains, recomputes W, derives every octant target itself and
    proves all 2^n of them against the shared Gram; the octant coverage check
    and the eps check run per corner, on the resolved list. Nothing is
    skipped, and a shared witness that fails at one corner fails there.

    Perturbed records are refused: their W depends on the inclusion endpoint
    as well, the two endpoint sets are the thing a dropped-endpoint record
    would fake, and there is no measured case that needs the alias. Keeping
    the surface at the nominal shape keeps the trust base smaller.
    """
    label = bc.get("corner_label", "?")
    own = bc.get("W_grams")
    shared_flag = bc.get("W_grams_shared")
    if own is not None and shared_flag:
        raise InvalidInputError(f"corner [{label}]: carries both its own W_grams and "
                          f"W_grams_shared; which Gram is the witness is not "
                          f"for this verifier to choose")
    if own is not None:
        return own
    if not shared_flag:
        raise RejectedError(f"corner [{label}]: nominal record has no W_grams "
                          f"captures")
    shared = bundle.get("shared_W_grams")
    if not shared:
        raise RejectedError(f"corner [{label}]: W_grams_shared is set but the "
                          f"bundle has no shared_W_grams capture set")
    return shared


#: The threshold the frozen Automatica note PRINTS (charge <= eps/2), as a
#: ratio against eps. Never an acceptance test here — see NOTE_HALF_RULE.
NOTE_HALF_RATIO = Fraction(1, 2)

ACCEPT_RULE = "charge < eps (strict, exact rational); WP P2-14 / decision D-l"
NOTE_HALF_RULE = ("charge <= eps/2, the inequality printed in the Automatica "
                  "note; reported, never enforced")


def verify_captures(terms, captures, n_vars, label, expected_eps, blocked=True):
    """Prove every octant of one form; return the worst ``charge/eps``.

    Returns ``(worst_charge_over_eps, shape_string, stats)`` where ``stats``
    counts the octants and how many of them also satisfy the note's printed
    ``charge <= eps/2``. Acceptance is ``charge < eps`` and nothing else; the
    note statistic is carried alongside so a record can be described honestly
    without re-running anything.
    """
    expected = 2 ** n_vars
    if isinstance(captures, (str, bytes, dict)) or not isinstance(
            captures, (list, tuple)):
        raise InvalidInputError(f"{label}: capture set must be a list, got "
                          f"{type(captures).__name__}")
    # Raw shape first: a capture is a dict carrying all five proof inputs.
    # Missing one is a refusal with a message, never a KeyError traceback.
    for i, cap in enumerate(captures):
        if not isinstance(cap, dict):
            raise InvalidInputError(f"{label}: capture {i} is not an object")
        for field in ("mu", "signs", "eps", "gram", "basis"):
            if field not in cap:
                raise InvalidInputError(f"{label}: capture {i} has no {field!r}")
    # The sign labels are read as RAW integers (2.9 is refused, not truncated
    # to 2); membership in {-1,+1}^n and completeness are then one and the
    # same statement, and the coverage check below owns it.
    labels = [validate_signs(c["signs"], n_vars, f"{label}: capture {i}")
              for i, c in enumerate(captures)]
    seen = set(labels)
    full = set(product((-1, 1), repeat=n_vars))
    if len(captures) != expected or seen != full:
        raise RejectedError(
            f"{label}: octant coverage {len(seen & full)}/{expected} "
            f"(the sign patterns must be exactly the full {{-1,+1}}^{n_vars} set)")
    # One form is proved under ONE lift. Every capture must declare the same
    # positive integral mu, or the 2^n octants would each be a chart of a
    # different substitution and would not cover R^n between them (finding 5).
    mus = {validate_mu(c["mu"], n_vars, f"{label}: capture {i}")
           for i, c in enumerate(captures)}
    if len(mus) != 1:
        raise RejectedError(f"{label}: captures declare {len(mus)} different "
                          f"lifts mu ({sorted(mus)}); one form is proved "
                          f"under one association")
    mu = next(iter(mus))
    worst = Fraction(0)
    shapes = set()
    n_note_half = 0
    # Accepted PSD shifts, reported only. None once an octant result carries
    # no shift (a caller that substitutes its own validate_octant).
    max_eta = Fraction(0)
    n_eta_positive = 0
    for cap, signs in zip(captures, labels):
        eps = q_from(cap["eps"])
        if eps != expected_eps:
            raise RejectedError(f"{label}: capture eps {eps} != record "
                              f"epsilon_exact {expected_eps}")
        target, d = exact_octant_target(terms, mu, signs, eps)
        if "y_degree" in cap and require_int(
                cap["y_degree"], f"{label}: capture y_degree") != d:
            raise RejectedError(f"{label}: octant {signs} declares y_degree "
                              f"{cap['y_degree']!r}, the target derived here "
                              f"has degree {d}")
        ok, ratio, why, shape = validate_octant(
            target, d, cap["gram"], cap["basis"], eps, blocked=blocked,
            cap=cap, label=f"{label} octant {signs}")
        if not ok:
            raise RejectedError(f"{label}: octant {signs} FAILED ({why}, "
                              f"charge/eps {float(ratio) if ratio is not None else None})")
        worst = max(worst, ratio)
        if ratio <= NOTE_HALF_RATIO:
            n_note_half += 1
        if len(shape) > 3 and max_eta is not None:
            max_eta = max(max_eta, shape[3])
            n_eta_positive += shape[3] > 0
        else:
            max_eta = None
        shapes.add(shape)
    n_basis = max(s[0] for s in shapes)
    n_blocks = min(s[1] for s in shapes)
    largest = max(s[2] for s in shapes)
    stats = {"n_octants": len(captures), "n_note_half_margin_ok": n_note_half,
             "max_eta": max_eta, "n_eta_positive": n_eta_positive}
    return (worst,
            f"{n_basis} basis -> {n_blocks} parity blocks, largest {largest}",
            stats)


# ---------------------------------------------------------------------------
# The published systems, rebuilt from the equations (nothing imported).
#
# ONE BUILDER PER FAMILY, hand-written from the published source, and the
# single reason this file can be trusted without the framework: a record
# supplies V, the gains, delta0, the margin and the Gram captures, never the
# dynamics. If the emitter's idea of a system ever drifts from the published
# equations, verification here fails instead of agreeing with the drift.
# ``tests/test_standalone_verifier.py`` pins every builder term for term
# against the framework's own GF builders.
#
# Convention shared by all builders:
#   gains          name -> exact Fraction (rational gain or bracket endpoint)
#   delta0         exact Fraction bound of the matched disturbance
#   endpoint_sign  +1 / -1 for the two inclusion vertices, None for NOMINAL
#                  dynamics (no disturbance term at all, delta0 must be 0)
# and the returned object is one list of exact terms per state equation.
# ---------------------------------------------------------------------------

def _term(n, coeff, *factors):
    """One exact GF term: coeff * prod_j |x_i|^e (* sign(x_i) if 'signed').

    ``factors`` are (variable index, exponent, type) triples; every unlisted
    variable gets exponent 0 with type 'abs', i.e. the factor 1.
    """
    exps = [Fraction(0)] * n
    types = ['abs'] * n
    for i, e, typ in factors:
        exps[i] = Fraction(e)
        types[i] = typ
    return (Fraction(coeff), tuple(exps), tuple(types))


def _disturbance(n, delta0, endpoint_sign):
    """One vertex of the matched constant disturbance, +- delta0."""
    return _term(n, endpoint_sign * delta0)


def build_dynamics(n, gains, delta0, endpoint_sign):
    """Levant's arbitrary-order differentiator, error dynamics (Levant 2003):

        e_i'   = -g_{i+1} |e_1|^{(n-1-i)/n} sign(e_1) + e_{i+2}   (i < n-1)
        e_{n-1}' = -g_n sign(e_1)                             (+- delta0)

    with weights (n, n-1, ..., 1). ``gains`` is the ORDERED list here, the one
    builder that keeps a positional signature (it is the v1 record path).
    The n=2 member is the super-twisting algorithm; the n=1 member is the
    first-order SMC reaching law s' = -k sign(s). The family is written for
    arbitrary n, so ``levant5d`` (the fourth-order differentiator, n = 5,
    added 2026-09-04 for WP P4-04) needs no new equations here: it is this
    recursion at n = 5, and ``tests/test_standalone_verifier.py`` pins it
    term for term against ``gf_framework.systems.levant5d``.
    """
    dyn = []
    for i in range(n):
        terms = [_term(n, -gains[i], (0, Fraction(n - 1 - i, n), 'signed'))]
        if i < n - 1:
            terms.append(_term(n, 1, (i + 1, 1, 'signed')))
        elif endpoint_sign is not None:
            terms.append(_disturbance(n, delta0, endpoint_sign))
        dyn.append(terms)
    return dyn


def build_power_rate(g, delta0, endpoint_sign):
    """Power-rate reaching law (Gao and Hung 1993): s' = -k |s|^alpha sign(s).

    Weighted degree kappa = -alpha, so kappa + r_ch != 0 and no constant
    matched disturbance is admissible: nominal only, by construction.
    """
    return [[_term(1, -g["k"], (0, g["alpha"], 'signed'))]]


def build_twisting(g, delta0, endpoint_sign):
    """Twisting algorithm (Levant 1993), weights r = (2, 1):

        x1' = x2,    x2' = -r1 sign(x1) - r2 sign(x2)   (+- delta0)
    """
    n = 2
    dyn = [[_term(n, 1, (1, 1, 'signed'))],
           [_term(n, -g["r1"], (0, 0, 'signed')),
            _term(n, -g["r2"], (1, 0, 'signed'))]]
    if endpoint_sign is not None:
        dyn[1].append(_disturbance(n, delta0, endpoint_sign))
    return dyn


def build_dsta(g, delta0, endpoint_sign):
    """Damped super-twisting (this work), weights r = (2, 1): the STA plus a
    sign(x2) damping relay,

        x1' = -k1 |x1|^{1/2} sign(x1) + x2
        x2' = -k2 sign(x1) - k3 sign(x2)               (+- delta0)
    """
    n = 2
    dyn = [[_term(n, -g["k1"], (0, Fraction(1, 2), 'signed')),
            _term(n, 1, (1, 1, 'signed'))],
           [_term(n, -g["k2"], (0, 0, 'signed')),
            _term(n, -g["k3"], (1, 0, 'signed'))]]
    if endpoint_sign is not None:
        dyn[1].append(_disturbance(n, delta0, endpoint_sign))
    return dyn


def build_example1(g, delta0, endpoint_sign):
    """Sanchez and Moreno 2019, Example 1 (eq. 4), weights r = (1, 3):

        x1' = -k1 |x1|^3 sign(x1) + x2,    x2' = -k2 |x1|^5 sign(x1)

    kappa = 2 > 0, so there is no weighted-degree-0 channel: nominal only.
    """
    n = 2
    return [[_term(n, -g["k1"], (0, 3, 'signed')), _term(n, 1, (1, 1, 'signed'))],
            [_term(n, -g["k2"], (0, 5, 'signed'))]]


def build_example3(g, delta0, endpoint_sign):
    """Sanchez and Moreno 2019, Example 3 (eq. 7), weights r = (3, 2):

        x1' = x2,   x2' = -k1 |x1|^{1/3} sign(x1) - k2 |x2|^{1/2} sign(x2)

    kappa + r_2 = 1 != 0, so no constant matched disturbance: nominal only.
    """
    n = 2
    return [[_term(n, 1, (1, 1, 'signed'))],
            [_term(n, -g["k1"], (0, Fraction(1, 3), 'signed')),
             _term(n, -g["k2"], (1, Fraction(1, 2), 'signed'))]]


def build_pidsmc(g, delta0, endpoint_sign):
    """PID-like SMC (Perez-Ventura and Fridman), weights r = (3, 2, 1):

        x1' = x2
        x2' = -l1 |x1|^{1/3} sign(x1) - l2 |x2|^{1/2} sign(x2) + x3
        x3' = -l3 sign(x1)                             (+- delta0)
    """
    n = 3
    dyn = [[_term(n, 1, (1, 1, 'signed'))],
           [_term(n, -g["lambda1"], (0, Fraction(1, 3), 'signed')),
            _term(n, -g["lambda2"], (1, Fraction(1, 2), 'signed')),
            _term(n, 1, (2, 1, 'signed'))],
           [_term(n, -g["lambda3"], (0, 0, 'signed'))]]
    if endpoint_sign is not None:
        dyn[2].append(_disturbance(n, delta0, endpoint_sign))
    return dyn


def build_continuous_twisting(g, delta0, endpoint_sign):
    """Continuous twisting algorithm (Torres-Gonzalez, Sanchez, Fridman and
    Moreno 2017), weights r = (3, 2, 1):

        x1' = x2
        x2' = -k1 |x1|^{1/3} sign(x1) - k2 |x2|^{1/2} sign(x2) + x3
        x3' = -k3 sign(x1) - k4 sign(x2)               (+- delta0)
    """
    n = 3
    dyn = [[_term(n, 1, (1, 1, 'signed'))],
           [_term(n, -g["k1"], (0, Fraction(1, 3), 'signed')),
            _term(n, -g["k2"], (1, Fraction(1, 2), 'signed')),
            _term(n, 1, (2, 1, 'signed'))],
           [_term(n, -g["k3"], (0, 0, 'signed')),
            _term(n, -g["k4"], (1, 0, 'signed'))]]
    if endpoint_sign is not None:
        dyn[2].append(_disturbance(n, delta0, endpoint_sign))
    return dyn


def build_fc_cta(g, delta0, endpoint_sign):
    """Fully connected CTA (this work), weights r = (3, 2, 1): the continuous
    twisting algorithm with the additional sign(x3) relay k5,

        x1' = x2
        x2' = -k1 |x1|^{1/3} sign(x1) - k2 |x2|^{1/2} sign(x2) + x3
        x3' = -k3 sign(x1) - k4 sign(x2) - k5 sign(x3)   (+- delta0)
    """
    n = 3
    dyn = build_continuous_twisting(g, delta0, None)
    dyn[2].append(_term(n, -g["k5"], (2, 0, 'signed')))
    if endpoint_sign is not None:
        dyn[2].append(_disturbance(n, delta0, endpoint_sign))
    return dyn


def build_pidsmc4d(g, delta0, endpoint_sign):
    """Fourth-order PID-like SMC (a best-guess extension, this work),
    weights r = (4, 3, 2, 1):

        x1' = x2
        x2' = x3
        x3' = -k1 |x1|^{1/4} sign(x1) - k2 |x2|^{1/3} sign(x2)
              - k3 |x3|^{1/2} sign(x3) + x4
        x4' = -k4 sign(x1)                             (+- delta0)

    CLAIM-HYGIENE QUARANTINE (carried verbatim from
    ``gf_framework/systems/pidsmc4d.py``): this formulation and its gains are a
    best-guess, natural relative-degree extension of the third-order PID-like
    SMC benchmark, NOT a confirmed reproduction of any specific prior
    construction. Nothing here or downstream may tie it to a named prior
    example. It is described only as "a fourth-order PID-like extension".

    Unlike the third-order ``pidsmc``, this family's factory takes no ``delta``
    scaling parameter (``create_pidsmc4d_spec`` forwards k1..k4 unscaled), so
    there is no delta = 1 case to pin: any ``delta`` key in a record is an
    unregistered extra gain and ``build_system_dynamics`` refuses it. That is
    why ``SYSTEMS["pidsmc4d"]`` carries no ``fixed`` entry.
    """
    n = 4
    dyn = [[_term(n, 1, (1, 1, 'signed'))],
           [_term(n, 1, (2, 1, 'signed'))],
           [_term(n, -g["k1"], (0, Fraction(1, 4), 'signed')),
            _term(n, -g["k2"], (1, Fraction(1, 3), 'signed')),
            _term(n, -g["k3"], (2, Fraction(1, 2), 'signed')),
            _term(n, 1, (3, 1, 'signed'))],
           [_term(n, -g["k4"], (0, 0, 'signed'))]]
    if endpoint_sign is not None:
        dyn[3].append(_disturbance(n, delta0, endpoint_sign))
    return dyn


#: family -> builder, for every family that is not the differentiator (whose
#: builder keeps the positional (n, ordered gains) signature of schema v1).
FAMILY_BUILDERS = {
    "power_rate": build_power_rate,
    "twisting": build_twisting,
    "dsta": build_dsta,
    "example1": build_example1,
    "example3": build_example3,
    "pidsmc": build_pidsmc,
    "continuous_twisting": build_continuous_twisting,
    "fc_cta": build_fc_cta,
    "pidsmc4d": build_pidsmc4d,
}

#: Per system: its family, its weight vector, the ORDER its gains enter the
#: builder in, the weighted-degree-0 disturbance channel (None where no
#: constant matched disturbance is admissible), and any factory parameter that
#: is NOT a coefficient of these equations. ``fixed`` is a soundness guard,
#: not documentation: pidsmc's factory scales its gains by delta**(2/3) and
#: delta**(1/2), so the equations above are the delta = 1 case only, and a
#: record carrying any other delta is refused instead of mis-verified.
SYSTEMS = {
    "fosmc": {"n": 1, "family": "levant_differentiator", "weights_r": (1,),
              "channel": 0, "gain_names": ["k"]},
    "power_rate": {"n": 1, "family": "power_rate", "weights_r": (1,),
                   "channel": None, "gain_names": ["k", "alpha"]},
    "sta": {"n": 2, "family": "levant_differentiator", "weights_r": (2, 1),
            "channel": 1, "gain_names": ["k1", "k2"]},
    "twisting": {"n": 2, "family": "twisting", "weights_r": (2, 1),
                 "channel": 1, "gain_names": ["r1", "r2"]},
    "dsta": {"n": 2, "family": "dsta", "weights_r": (2, 1),
             "channel": 1, "gain_names": ["k1", "k2", "k3"]},
    "example1": {"n": 2, "family": "example1", "weights_r": (1, 3),
                 "channel": None, "gain_names": ["k1", "k2"]},
    "example3": {"n": 2, "family": "example3", "weights_r": (3, 2),
                 "channel": None, "gain_names": ["k1", "k2"]},
    "levant3d": {"n": 3, "family": "levant_differentiator", "weights_r": (3, 2, 1),
                 "channel": 2, "gain_names": ["lambda1", "lambda2", "lambda3"]},
    "levant3d_2003print": {"n": 3, "family": "levant_differentiator",
                           "weights_r": (3, 2, 1), "channel": 2,
                           "gain_names": ["lambda1", "lambda2", "lambda3"]},
    "pidsmc": {"n": 3, "family": "pidsmc", "weights_r": (3, 2, 1), "channel": 2,
               "gain_names": ["lambda1", "lambda2", "lambda3"],
               "fixed": {"delta": "1"}},
    "continuous_twisting": {"n": 3, "family": "continuous_twisting",
                            "weights_r": (3, 2, 1), "channel": 2,
                            "gain_names": ["k1", "k2", "k3", "k4"]},
    "fc_cta": {"n": 3, "family": "fc_cta", "weights_r": (3, 2, 1), "channel": 2,
               "gain_names": ["k1", "k2", "k3", "k4", "k5"]},
    "levant4d": {"n": 4, "family": "levant_differentiator", "weights_r": (4, 3, 2, 1),
                 "channel": 3,
                 "gain_names": ["lambda1", "lambda2", "lambda3", "lambda4"]},
    # The third-order differentiator at the higher-gain standard tuning,
    # recursive (1.1, 1.5, 3, 5) (WP X10). Same equations as levant4d; the
    # gains are record data either way. The key exists so that a record names
    # the registered gain point it was emitted against, as levant3d_2003print
    # does for the second-order differentiator.
    "levant4d_2003print": {"n": 4, "family": "levant_differentiator",
                           "weights_r": (4, 3, 2, 1), "channel": 3,
                           "gain_names": ["lambda1", "lambda2", "lambda3",
                                          "lambda4"]},
    # Levant's FOURTH-ORDER differentiator, n = 5, r = (5,4,3,2,1), kappa = -1,
    # so kappa + r_5 = 0 and channel 4 carries the constant matched
    # disturbance. Same recursion as every other differentiator row, one
    # order up (WP P4-04, 2026-09-04); no new builder, only this entry.
    "levant5d": {"n": 5, "family": "levant_differentiator",
                 "weights_r": (5, 4, 3, 2, 1), "channel": 4,
                 "gain_names": ["lambda1", "lambda2", "lambda3", "lambda4",
                                "lambda5"]},
    # Best-guess fourth-order PID-like extension (quarantined, see
    # build_pidsmc4d). NO "fixed" entry on purpose: its factory takes no delta
    # scaling parameter, so a record carrying one is an unregistered extra gain
    # and is refused rather than pinned to a case.
    "pidsmc4d": {"n": 4, "family": "pidsmc4d", "weights_r": (4, 3, 2, 1),
                 "channel": 3, "gain_names": ["k1", "k2", "k3", "k4"]},
}


def build_system_dynamics(system, gains, delta0, endpoint_sign):
    """Dispatch to the family builder, after checking the record's gain set.

    Every builder input is accounted for: a missing gain, an unregistered
    extra gain, or a fixed factory parameter at a value these equations do not
    describe is an error, never a silently ignored field.
    """
    cfg = SYSTEMS[system]
    n = len(cfg["weights_r"])
    missing = [g for g in cfg["gain_names"] if g not in gains]
    if missing:
        raise InvalidInputError(f"{system}: record is missing gains {missing}")
    fixed = cfg.get("fixed", {})
    for name, value in gains.items():
        if name in cfg["gain_names"]:
            continue
        if name not in fixed:
            raise InvalidInputError(
                f"{system}: gain {name!r} is not a parameter of the published "
                f"equations rebuilt here; refusing to ignore it")
        if value != Fraction(fixed[name]):
            raise InvalidInputError(
                f"{system}: parameter {name} = {value} is outside the case "
                f"these equations describe ({name} = {fixed[name]})")
    if endpoint_sign is None:
        if delta0 != 0:
            raise InvalidInputError(f"{system}: nominal dynamics with delta0 = {delta0}")
    elif cfg["channel"] is None:
        raise InvalidInputError(
            f"{system}: no weighted-degree-0 channel admits a constant matched "
            f"disturbance, so a perturbed record cannot be built for it")
    if cfg["family"] == "levant_differentiator":
        return build_dynamics(n, [gains[g] for g in cfg["gain_names"]],
                              delta0, endpoint_sign)
    return FAMILY_BUILDERS[cfg["family"]](gains, delta0, endpoint_sign)


def derive_kappa(dyn, weights, what):
    """The weighted homogeneity degree of the REBUILT dynamics, derived here.

    f is r-homogeneous of degree kappa iff every term of f_i has weighted
    degree r_i + kappa. That is a property of the equations this file writes
    down, so it is computed from them term by term and never read from the
    record's ``kappa`` field — which finding 13 of the 2026-09-05 review
    showed could say 999 and still be VERIFIED. A record's declared kappa is
    then compared against this value, and the value is printed.

    A constant matched disturbance vertex is a term of weighted degree 0, so a
    perturbed system is homogeneous only if r_ch + kappa = 0 for its channel;
    that consistency is checked by the caller against the channel table.
    """
    kappa = None
    for i, f_i in enumerate(dyn):
        for _c, exps, _types in f_i:
            wdeg = sum(Fraction(w) * e for w, e in zip(weights, exps))
            k = wdeg - Fraction(weights[i])
            if kappa is None:
                kappa = k
            elif k != kappa:
                raise RejectedError(
                    f"{what}: the rebuilt dynamics are not weighted "
                    f"homogeneous: f_{i+1} has a term of weighted degree "
                    f"{wdeg}, i.e. kappa = {k}, against {kappa} elsewhere")
    if kappa is None:
        raise RejectedError(f"{what}: the rebuilt dynamics have no terms")
    return kappa


def check_v_admissible(v_terms, weights, degree_m):
    """R2 (C^1) admissibility + weighted homogeneity of the published V."""
    n = len(weights)
    if not v_terms:
        raise InvalidInputError("V has no terms")
    for c, exps, types in v_terms:
        require_vector(exps, n, "V term exponents")
        for ty in require_vector(types, n, "V term types"):
            if ty not in ("abs", "signed"):
                raise InvalidInputError(f"V factor type {ty!r} is neither 'abs' nor "
                                  f"'signed'")
        wdeg = sum(Fraction(w) * e for w, e in zip(weights, exps))
        if wdeg != degree_m:
            raise RejectedError(f"V term weighted degree {wdeg} != {degree_m}")
        for e, t in zip(exps, types):
            if t == 'abs':
                if not (e == 0 or e > 1):
                    raise RejectedError(f"V not C^1: abs exponent {e}")
            else:
                if e == 0:
                    raise RejectedError("V not C^1: bare sign(x) term")
                if e < 1:
                    raise RejectedError(f"V not C^1: signed exponent {e} < 1")
    if degree_m <= max(weights):
        raise RejectedError("V degree must exceed max weight (R2)")


def check_declared_margin(record, derived=None):
    """Every published margin declaration must agree and be covered by the proof.

    A marked float transcription may round upward to the same float image of
    the derived rational. An independently declared exact bound has no such
    rounding allowance. Missing declarations on historical records are fine.
    """
    exact_raw = record.get("eps_certified_exact")
    float_raw = record.get("eps_certified")
    exact = q_exact(exact_raw, "eps_certified_exact") if exact_raw is not None else None
    approx = q_from(float_raw) if float_raw is not None else None
    transcription = require_bool(record["eps_certified_exact_is_float_transcription"],
                                 "eps_certified_exact_is_float_transcription") \
        if "eps_certified_exact_is_float_transcription" in record else False
    if exact is not None and approx is not None:
        if float(exact) != float(approx) or transcription and exact != approx:
            raise InvalidInputError("margin declarations disagree: eps_certified_exact "
                              "does not match eps_certified")
    if transcription and (exact is None or approx is None):
        raise InvalidInputError("a margin float transcription needs both margin fields")
    for value, rounded in ((exact, transcription), (approx, True)):
        if value is None:
            continue
        if value <= 0:
            raise InvalidInputError("declared certified margin must be positive")
        if derived is not None and value > derived and not (
                rounded and float(value) == float(derived)):
            raise RejectedError(f"MARGIN CLAIM EXCEEDS PROOF: stated {value}, "
                              f"derived {derived}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", help="*_exactgains_*.json produced by "
                                   "scripts/exact_gate_standard_gains.py")
    ap.add_argument("--grams", default=None,
                    help="grams bundle (default: record['grams_file'] "
                         "next to the record)")
    ap.add_argument("--row", default=None,
                    help="row JSON with V_terms (only for records without "
                         "an embedded V)")
    ap.add_argument("--dense", action="store_true",
                    help="reference path: one dense exact LDL^T per octant, "
                         "no deck-group parity blocking. Much slower, and the "
                         "conservative one: a dense PASS implies the blocked "
                         "PASS at the same charge, not conversely. The two "
                         "agree whenever the same eta rung is selected, which "
                         "is the case on every stored record (their captured "
                         "Gram is PSD, so both take eta = 0)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write the verification summary as JSON: the "
                         "acceptance rule, margin_policy, the exact "
                         "eps_certified, the worst charge/eps (and the legacy "
                         "charge/(eps/2) for comparison with pre-P2-14 "
                         "records), and note_half_margin_holds per form and "
                         "rolled up")
    args = ap.parse_args()
    blocked = not args.dense

    # Bind the summary to the bytes actually parsed, before a potentially long proof.
    # Hashing paths again at the end would instead identify replacements made mid-run.
    proof_inputs = {"schema": "gf-proof-inputs/1", "verifier": {
        "path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}

    rec_path = Path(args.record)
    rec = read_json(rec_path, proof_inputs, "record")
    if not isinstance(rec, dict):
        raise InvalidInputError(f"record must be a JSON object, got "
                          f"{type(rec).__name__}")
    check_declared_margin(rec)
    for field in ("system", "rational_gains", "delta0_exact", "epsilon_exact",
                  "corners"):
        if field not in rec:
            raise InvalidInputError(f"record has no {field!r}")
    if args.grams is None and "grams_file" not in rec:
        raise InvalidInputError("record has no 'grams_file' and no --grams was given")
    grams_path = Path(args.grams) if args.grams else \
        resolve_sibling(rec_path.parent, rec["grams_file"])
    bundle = read_json(grams_path, proof_inputs, "grams")
    if not isinstance(bundle, dict) or "V_grams" not in bundle \
            or "corners" not in bundle:
        raise InvalidInputError("grams bundle must be a JSON object carrying "
                          "'V_grams' and 'corners'")

    schema = rec.get("schema_version", SCHEMA_VERSIONS[0])
    if schema not in SCHEMA_VERSIONS:
        raise InvalidInputError(f"unknown record schema {schema!r}; this verifier "
                          f"implements {list(SCHEMA_VERSIONS)}")

    system = rec["system"]
    if not isinstance(system, str):
        raise InvalidInputError(f"record system must be a string, got "
                          f"{type(system).__name__} {system!r}")
    if system not in SYSTEMS:
        raise InvalidInputError(f"no hand-written dynamics builder for system "
                          f"{system!r}; known: {sorted(SYSTEMS)}")
    cfg = SYSTEMS[system]
    weights = cfg["weights_r"]
    n = len(weights)

    # A v2 record names its own family and channel. Both are re-derived here
    # from the system name and must agree: a record that claims a different
    # family or a different disturbance channel than the equations this file
    # rebuilds is describing something else, and is refused rather than
    # verified against the wrong dynamics.
    if "system_family" in rec and rec["system_family"] != cfg["family"]:
        raise InvalidInputError(f"record system_family = {rec['system_family']!r} "
                          f"contradicts the published system {system} "
                          f"({cfg['family']!r})")
    if "weights_r" in rec and tuple(
            require_int_vector(rec["weights_r"], n, "record weights_r",
                               minimum=1)) != weights:
        raise InvalidInputError(f"record weights_r = {rec['weights_r']!r} "
                          f"contradicts the published system {system} "
                          f"({list(weights)!r})")
    # ``nominal`` decides which record SHAPE is proved (one dynamics per corner
    # against W_grams, or two inclusion endpoints), so a truthy string must not
    # become True here: it is a JSON boolean or it is a refusal.
    nominal = require_bool(rec["nominal"], "record nominal") \
        if "nominal" in rec else False
    expected_channel = None if nominal else cfg["channel"]
    if "channel" in rec:
        declared_channel = rec["channel"]
        if declared_channel is not None:
            declared_channel = require_int(declared_channel, "record channel")
        if declared_channel != expected_channel:
            raise InvalidInputError(f"record channel {rec['channel']!r} contradicts "
                              f"the published system {system} "
                              f"({expected_channel!r})")

    # --- V (published certificate) -------------------------------------
    if "V" in rec:
        vsrc = rec["V"]
    else:
        if not args.row:
            print("INVALID INPUT: record has no embedded V; pass --row ROW.json",
                  file=sys.stderr)
            return EXIT_INVALID_INPUT
        row = read_json(Path(args.row), proof_inputs, "row")
        vsrc = {"degree_m": row["degree_used"], "weights_r": row["weights_r"],
                "terms": row["V_terms"]}
    if not isinstance(vsrc, dict):
        raise InvalidInputError(f"V must be a JSON object, got "
                          f"{type(vsrc).__name__}")
    for field in ("degree_m", "weights_r", "terms"):
        if field not in vsrc:
            raise InvalidInputError(f"V has no {field!r}")
    if tuple(require_int_vector(vsrc["weights_r"], n, "V weights_r",
                                minimum=1)) != weights:
        raise InvalidInputError(f"unexpected weights {vsrc['weights_r']}")
    degree_m = require_int(vsrc["degree_m"], "V degree_m")
    if degree_m <= 0:
        raise InvalidInputError(f"V degree_m must be positive (got {degree_m})")
    v_terms = validate_gf_terms(vsrc["terms"], n, "V")
    check_v_admissible(v_terms, weights, Fraction(degree_m))
    print(f"[1/5] V admissible (C^1, homogeneous of degree {degree_m}"
          f", {len(v_terms)} terms)")

    # --- brackets --------------------------------------------------------
    brackets = {}
    raw_brackets = rec.get("brackets", {})
    if not isinstance(raw_brackets, dict):
        raise InvalidInputError(f"record brackets must be an object, got "
                          f"{type(raw_brackets).__name__}")
    for name, b in raw_brackets.items():
        if not isinstance(b, dict):
            raise InvalidInputError(f"bracket {name} is not an object")
        for field in ("lo", "hi", "power_k", "power_target"):
            if field not in b:
                raise InvalidInputError(f"bracket {name} has no {field!r}")
        lo = q_exact(b["lo"], f"bracket {name} lo")
        hi = q_exact(b["hi"], f"bracket {name} hi")
        k = require_int(b["power_k"], f"bracket {name} power_k")
        if k < 1:
            raise InvalidInputError(f"bracket {name} power_k must be >= 1 (got {k})")
        q = q_exact(b["power_target"], f"bracket {name} power_target")
        if not (0 < lo < hi and lo ** k < q < hi ** k):
            raise RejectedError(f"bracket {name} INVALID: need lo^{k} < {q} < hi^{k}")
        brackets[name] = (lo, hi)
        print(f"[2/5] bracket {name}: lo^{k} < {q} < hi^{k} verified in Q")
    if not brackets:
        print("[2/5] no irrational gains (all rational, injected exactly)")

    if not isinstance(rec["rational_gains"], dict):
        raise InvalidInputError(f"record rational_gains must be an object, got "
                          f"{type(rec['rational_gains']).__name__}")
    rational = {k: q_exact(v, f"gain {k}")
                for k, v in rec["rational_gains"].items()}
    overlap = set(rational) & set(brackets)
    if overlap:
        raise InvalidInputError(f"gains declared both rational and bracketed: {sorted(overlap)}")
    delta0 = q_exact(rec["delta0_exact"], "delta0_exact")
    epsilon_exact = q_exact(rec["epsilon_exact"], "epsilon_exact")
    if nominal and delta0 != 0:
        raise InvalidInputError(f"record is flagged nominal but carries "
                          f"delta0 = {delta0}")
    if not nominal and delta0 <= 0:
        raise InvalidInputError(f"perturbed record needs a positive delta0 "
                          f"(got {delta0})")
    if epsilon_exact <= 0:
        # eps = 0 makes the acceptance threshold zero in validate_octant,
        # which used to crash with a ZeroDivisionError computing charge/eps for
        # the first non-zero-residual octant (fail-closed, but not a clean
        # rejection). Reject explicitly instead, same fail-closed outcome.
        raise InvalidInputError(f"margin epsilon must be positive (got {epsilon_exact})")

    # --- COMPLETENESS of the corner box and the endpoint pairs ----------
    # The affinity argument (docstring step 6) needs ALL 2^k bracket corners
    # and BOTH inclusion endpoints; an incomplete record proves nothing about
    # the interior gain/disturbance values, so it is rejected here.
    expected_corners = 2 ** len(brackets)
    if len(rec["corners"]) != expected_corners \
            or len(bundle["corners"]) != expected_corners:
        raise RejectedError(
            f"corner box incomplete: record {len(rec['corners'])} / bundle "
            f"{len(bundle['corners'])}, need 2^{len(brackets)} = "
            f"{expected_corners}")
    corner_keys = set()
    for rc in rec["corners"]:
        if set(rc["assignment"]) != set(brackets):
            raise RejectedError(f"corner assigns {sorted(rc['assignment'])}, "
                              f"brackets are {sorted(brackets)}")
        for name, val in rc["assignment"].items():
            if q_exact(val, f"corner gain {name}") not in brackets[name]:
                raise RejectedError(f"corner gain {name}={val} is not a "
                                  f"bracket endpoint")
        corner_keys.add(tuple(sorted((m, q_exact(v, f"corner gain {m}"))
                                     for m, v in rc["assignment"].items())))
    if len(corner_keys) != expected_corners:
        # distinct + every coordinate in {lo, hi} + count 2^k  ==>  the full
        # {lo, hi}^k product (pigeonhole); duplicates mean a corner is missing
        raise RejectedError(f"only {len(corner_keys)} distinct corner "
                          f"assignments, need {expected_corners}")
    # A nominal record has ONE dynamics per corner, under its own key. The keys
    # are disjoint on purpose: reusing "endpoints" with a single entry would
    # let a perturbed bundle with a dropped W- endpoint pass as nominal by
    # flipping one flag. Here that bundle has no W_grams and is refused, and a
    # nominal bundle has no endpoints to drop.
    for bc in bundle["corners"]:
        if nominal:
            if "endpoints" in bc:
                raise InvalidInputError(f"corner [{bc['corner_label']}]: nominal "
                                  f"record carries inclusion endpoints")
            nominal_w_captures(bundle, bc)
        else:
            if "W_grams" in bc or "W_grams_shared" in bc \
                    or "shared_W_grams" in bundle:
                raise InvalidInputError(f"corner [{bc['corner_label']}]: perturbed "
                                  f"record carries nominal W_grams captures")
            if len(bc["endpoints"]) != 2:
                raise RejectedError(f"corner [{bc['corner_label']}]: "
                                  f"{len(bc['endpoints'])} endpoint capture "
                                  f"sets, need 2 (W+ and W-)")

    # --- kappa DERIVED from the rebuilt dynamics (review finding 13) -----
    # The record's kappa field is metadata until it is checked; a copy of the
    # FOSMC record with kappa = 999 used to verify. It is derived here from
    # the equations this file writes down (every term of f_i must have
    # weighted degree r_i + kappa), compared with the declaration, and
    # printed. The check runs before the expensive octant proofs, and is
    # repeated per corner inside the W loop so no corner can differ.
    first_gains = dict(rational)
    first_gains.update({k: q_exact(v, f"corner gain {k}")
                        for k, v in bundle["corners"][0]["assignment"].items()})
    kappa_derived = derive_kappa(
        build_system_dynamics(system, first_gains, Fraction(0), None),
        weights, system)
    if "kappa" in rec:
        declared_kappa = q_exact(rec["kappa"], "record kappa")
        if declared_kappa != kappa_derived:
            raise RejectedError(
                f"record kappa = {rec['kappa']!r} contradicts the weighted "
                f"homogeneity degree {kappa_derived} derived from the "
                f"dynamics rebuilt here for {system}")
    if not nominal:
        # A constant matched disturbance on channel ch is an r-homogeneous
        # vertex only if r_ch + kappa = 0; otherwise the perturbed inclusion
        # this record claims is not homogeneous and the octant argument does
        # not apply to it.
        r_ch = Fraction(weights[cfg["channel"]])
        if r_ch + kappa_derived != 0:
            raise RejectedError(
                f"{system}: channel {cfg['channel']} has weight {r_ch} and the "
                f"dynamics have kappa = {kappa_derived}, so r_ch + kappa = "
                f"{r_ch + kappa_derived} != 0: a constant matched disturbance "
                f"is not admissible for this perturbed record")

    # --- V positivity from the Gram captures ----------------------------
    worst_v, shape_v, stats_v = verify_captures(
        v_terms, bundle["V_grams"], n, "V", epsilon_exact, blocked=blocked)
    print(f"[3/5] V positivity: all {2**n} octants PASS "
          f"(worst charge/eps {float(worst_v):.2e}; {shape_v})")

    # --- W per corner per endpoint (perturbed) or per corner (nominal) ----
    worst_w = Fraction(0)
    shape_w = ""
    stats_w = {"n_octants": 0, "n_note_half_margin_ok": 0,
               "max_eta": Fraction(0), "n_eta_positive": 0}
    n_checked = 0
    shared_w = bool(nominal and bundle.get("shared_W_grams"))
    for rc, bc in zip(rec["corners"], bundle["corners"]):
        if rc["assignment"] != bc["assignment"]:
            raise InvalidInputError(f"record/bundle corner order mismatch: "
                              f"{rc['assignment']} vs {bc['assignment']}")
        gains_map = dict(rational)
        gains_map.update({k: q_exact(v, f"corner gain {k}")
                          for k, v in bc["assignment"].items()})
        # kappa is DERIVED from the equations rebuilt here, at this corner's
        # own gains, and every corner must agree (finding 13). The unperturbed
        # dynamics carry it: the disturbance vertex is the constant term whose
        # own admissibility is r_ch + kappa = 0, checked separately below.
        k_here = derive_kappa(
            build_system_dynamics(system, gains_map, Fraction(0), None),
            weights, f"{system} corner [{bc.get('corner_label', '?')}]")
        if k_here != kappa_derived:
            raise RejectedError(f"{system}: corner [{bc.get('corner_label','?')}]"
                              f" has homogeneity degree {k_here}, another "
                              f"corner {kappa_derived}")
        if nominal:
            work = [(nominal_w_captures(bundle, bc), None, "W")]
        else:
            work = list(zip(bc["endpoints"], (+1, -1), ("W+", "W-")))
        for caps, endpoint_sign, lbl in work:
            dyn = build_system_dynamics(system, gains_map, delta0, endpoint_sign)
            w_terms = exact_neg_lie_derivative(v_terms, dyn)
            worst, shape_w, st = verify_captures(
                w_terms, caps, n, f"{lbl}@{bc['corner_label']}",
                epsilon_exact, blocked=blocked)
            worst_w = max(worst_w, worst)
            stats_w["n_octants"] += st["n_octants"]
            stats_w["n_note_half_margin_ok"] += st["n_note_half_margin_ok"]
            # A substituted verify_captures (scripts/round_certificate.py)
            # reports no shift; the shift is then unknown, never assumed 0.
            stats_w["n_eta_positive"] += st.get("n_eta_positive", 0)
            eta_here = st.get("max_eta")
            stats_w["max_eta"] = (None if None in (stats_w["max_eta"], eta_here)
                                  else max(stats_w["max_eta"], eta_here))
            n_checked += 1
    scope = "corners" if nominal else "endpoint x corner combinations"
    witness = (", one shared witness Gram set proved against every corner's "
               "own W" if shared_w else "")
    print(f"[4/5] W: {n_checked} {scope}, all "
          f"{2**n} octants each, PASS (worst charge/eps "
          f"{float(worst_w):.2e}; {shape_w}{witness})")

    coverage = ("the nominal dynamics" if nominal else
                "both inclusion endpoints")
    path = "dense" if args.dense else "deck-group parity blocked"

    # WP P2-14. The margin actually PROVED is eps - charge on the worst octant
    # of the worst form, exactly, in Q. Every octant of every form was accepted
    # at charge < eps, so this is strictly positive.
    worst_all = max(worst_v, worst_w)
    eps_certified = epsilon_exact * (1 - worst_all)
    check_declared_margin(rec, eps_certified)
    note_half_v = worst_v <= NOTE_HALF_RATIO
    note_half_w = worst_w <= NOTE_HALF_RATIO
    note_half_all = note_half_v and note_half_w
    n_oct_total = stats_v["n_octants"] + stats_w["n_octants"]
    n_note_ok = (stats_v["n_note_half_margin_ok"]
                 + stats_w["n_note_half_margin_ok"])
    print(f"[5/5] {system}: certificate VERIFIED — V > 0 and "
          f"-dV/dt > 0 (margin form) on every octant, {coverage}, "
          f"every gain-bracket corner; exact rational arithmetic "
          f"throughout ({path} LDL^T), delta0 = {delta0}, "
          f"epsilon = {rec['epsilon_exact']}, "
          f"kappa = {kappa_derived} (derived from the rebuilt dynamics)")
    print(f"      accepted under: {ACCEPT_RULE}")
    print(f"      eps_certified = eps - charge = "
          f"{eps_certified.numerator}/{eps_certified.denominator} "
          f"(~{float(eps_certified):.6e}); worst charge/eps "
          f"{float(worst_all):.2e}")
    print(f"      note_half_margin_holds ({NOTE_HALF_RULE}): "
          f"{'yes' if note_half_all else 'NO'} "
          f"[{n_note_ok}/{n_oct_total} octants]")
    # The accepted PSD shift, reported only (the verdict above does not
    # depend on it). eta = 0 on every octant means the record also meets the
    # shift-free Gram lemma (Q_H itself PSD and ||E||_1 < eps).
    max_eta_v = stats_v.get("max_eta")
    max_eta_all = (None if None in (max_eta_v, stats_w["max_eta"])
                   else max(max_eta_v, stats_w["max_eta"]))
    n_eta_pos = stats_v.get("n_eta_positive", 0) + stats_w["n_eta_positive"]
    if max_eta_all is not None:
        if n_eta_pos == 0:
            print(f"      psd_shift: eta = 0 on all {n_oct_total} octants "
                  f"(the Gram itself is PSD; charge = ||E||_1)")
        else:
            print(f"      psd_shift: eta > 0 on {n_eta_pos}/{n_oct_total} "
                  f"octants, largest {max_eta_all}")

    if args.json:
        summary = {
            "verifier": "verification/standalone/verify_certificate.py",
            "verdict": "VERIFIED",
            "proof_inputs": proof_inputs,
            "system": system,
            "record": str(rec_path),
            "grams": str(grams_path),
            "schema_version": schema,
            "path": path,
            "nominal": bool(nominal),
            # Derived here from the rebuilt equations, not copied from the
            # record; a record that declares a different one is refused.
            "kappa_derived": f"{kappa_derived.numerator}/"
                             f"{kappa_derived.denominator}",
            "delta0_exact": f"{delta0.numerator}/{delta0.denominator}",
            "epsilon_exact": f"{epsilon_exact.numerator}/"
                             f"{epsilon_exact.denominator}",
            "epsilon_float": float(epsilon_exact),
            # The acceptance rule in force, named, plus the note's inequality
            # as a REPORTED property of this record.
            "margin_policy": "achieved",
            "acceptance_rule": ACCEPT_RULE,
            "eps_certified_exact": f"{eps_certified.numerator}/"
                                   f"{eps_certified.denominator}",
            "eps_certified_float": float(eps_certified),
            # Both ratios, each naming its denominator: the second is the
            # pre-P2-14 "charge/budget" and is exactly twice the first.
            "worst_charge_over_eps": {
                "V": float(worst_v), "W": float(worst_w),
                "overall": float(worst_all)},
            "worst_charge_over_half_eps_legacy": {
                "V": float(2 * worst_v), "W": float(2 * worst_w),
                "overall": float(2 * worst_all)},
            "note_half_margin_rule": NOTE_HALF_RULE,
            "note_half_margin_holds": {
                "V": bool(note_half_v), "W": bool(note_half_w),
                "overall": bool(note_half_all)},
            "note_half_margin_octants": {
                "ok": n_note_ok, "total": n_oct_total},
            # Reported only: the accepted PSD shift (0 on every octant means
            # the shift-free Gram lemma holds for this record).
            "psd_shift_eta": {
                "max_exact": (None if max_eta_all is None else
                              f"{max_eta_all.numerator}/"
                              f"{max_eta_all.denominator}"),
                "n_octants_positive": n_eta_pos,
                "n_octants": n_oct_total},
        }
        try:
            Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")
        except OSError as exc:
            raise InvalidInputError(f"the certificate verified, but the --json "
                                    f"summary {args.json!r} cannot be written: "
                                    f"{exc.strerror or exc}")
        print(f"      summary written to {args.json}")
    return EXIT_VERIFIED


#: The last line of every run names the outcome class (module docstring).
RESULT_LINES = {
    EXIT_VERIFIED: "RESULT: VERIFIED (exit code 0): every check passed",
    EXIT_REJECTED: (
        "RESULT: REJECTED (exit code 1): a mathematical check failed on these "
        "input files. The verdict concerns exactly these bytes: compare their "
        "sha256 with the published SHA256SUMS before reading it as a statement "
        "about a published certificate"),
    EXIT_INVALID_INPUT: (
        "RESULT: INVALID INPUT (exit code 2): the input files or arguments "
        "could not be read as a complete certificate record (missing or "
        "unreadable file, malformed JSON or gzip, schema or type violation). "
        "This is not a mathematical rejection; check the files and their "
        "sha256"),
    EXIT_ERROR: (
        "RESULT: ERROR (exit code 3): the verifier stopped on an unexpected "
        "condition (e.g. out of memory); no verdict was reached"),
}

#: Python exceptions that, raised while a record is read, mean that the record
#: or bundle is malformed in a way no explicit check names (a missing key, a
#: list where an object belongs, ...). They are INVALID INPUT, never REJECTED.
MALFORMED_INPUT_EXCEPTIONS = (KeyError, IndexError, TypeError, AttributeError,
                              ValueError, OSError, EOFError, zlib.error)


def cli():
    """Run :func:`main` and map its outcome to a message and an exit code.

    :func:`main` itself keeps raising :class:`VerifyError` (in-process
    callers such as ``scripts/round_certificate.py`` catch it); only this
    wrapper turns the outcome into the exit codes of the module docstring.
    """
    try:
        code = main()
    except SystemExit as exc:
        # argparse: usage errors exit 2 (INVALID INPUT), --help exits 0.
        if exc.code == EXIT_INVALID_INPUT:
            print(RESULT_LINES[EXIT_INVALID_INPUT], file=sys.stderr)
        raise
    except RejectedError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        print(RESULT_LINES[EXIT_REJECTED], file=sys.stderr)
        return EXIT_REJECTED
    except InvalidInputError as exc:
        print(f"INVALID INPUT: {exc}", file=sys.stderr)
        print(RESULT_LINES[EXIT_INVALID_INPUT], file=sys.stderr)
        return EXIT_INVALID_INPUT
    except VerifyError as exc:  # not raised by this file; kept conservative
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        print(RESULT_LINES[EXIT_REJECTED], file=sys.stderr)
        return EXIT_REJECTED
    except MALFORMED_INPUT_EXCEPTIONS as exc:
        print(f"INVALID INPUT: malformed record or Gram bundle: "
              f"{type(exc).__name__}: {exc} (no explicit check names this "
              f"defect; if the files match their published sha256, please "
              f"report it as a verifier bug)", file=sys.stderr)
        print(RESULT_LINES[EXIT_INVALID_INPUT], file=sys.stderr)
        return EXIT_INVALID_INPUT
    except Exception as exc:  # noqa: BLE001 -- e.g. MemoryError: no verdict
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(RESULT_LINES[EXIT_ERROR], file=sys.stderr)
        return EXIT_ERROR
    if code == EXIT_VERIFIED:
        print(RESULT_LINES[EXIT_VERIFIED])
    elif code in RESULT_LINES:
        print(RESULT_LINES[code], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(cli())
