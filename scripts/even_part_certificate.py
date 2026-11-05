#!/usr/bin/env python3
"""Even part of a certified generalized-form Lyapunov function (2026-11-05).

STDLIB ONLY. Verdicts come from the portable verifier
(verification/standalone/verify_certificate.py, "VC"), run as a subprocess on
the files this tool writes, exactly as a reader would run it.

Why the even part is certified (paper, Section 4.3, "Exact scaling and
symmetry"; Moreno's evaluation of 2026-11-04, Section 3.2)
--------------------------------------------------------------------------
Let the one-sided fields be odd, f_{-s}(-x) = -f_s(x) for every sign vector s,
which holds for every system in the paper (all terms are signed powers or
relays, and the disturbance interval is symmetric). For a GF V put
U(x) := V(-x) and V_e := (V + U)/2, the EVEN PART of V: a term with an even
number of signed factors is unchanged by x -> -x, a term with an odd number
changes sign, so V_e keeps exactly the even terms of V with their
coefficients and drops the odd ones.

On the orthant s, with x = s * y^mu,
    Phi_s[U](y)          = Phi_{-s}[V](y),
    Phi_s[W_U^{w,sig}]   = Phi_{-s}[W^{w,-sig}],
where W^{w,sig} = -grad V . (f(.; w) + sig*delta0*b) (oddness of f and
evenness of the constant disturbance term under the sign flip). Hence
    Phi_s[V_e]          = (Phi_s[V] + Phi_{-s}[V]) / 2,
    Phi_s[W_e^{w,sig}]  = (Phi_s[W^{w,sig}] + Phi_{-s}[W^{w,-sig}]) / 2,
and the averaged Gram matrices witness the averaged forms: the residual is the
average of the two residuals, so its l1 norm is at most the larger of the two
(< eps), and the parity-blocked Gram is an average of PSD matrices, hence PSD.
So an accepted certificate for V yields an accepted certificate for V_e. This
tool builds that certificate and lets VC decide it; nothing here is trusted.

Witness map written by this tool (capture lists are keyed by sign vector):
    V_grams[s]            := (V_grams[s] + V_grams[-s]) / 2
    perturbed, corner c:  endpoints[0] is +delta0, endpoints[1] is -delta0
      W+[s]               := (W+[s] + W-[-s]) / 2
      W-[s]               := (W-[s] + W+[-s]) / 2
    nominal, corner c:    W[s] := (W[s] + W[-s]) / 2
Paired captures must share mu, y_degree, eps and the basis (checked).

Usage (from GF Framework/):
    python scripts/even_part_certificate.py --record REC.json --out-dir DIR [--single-run]

Without --single-run, VC runs twice: on the candidate (no margin declared), then
on the final record that declares VC's derived margin, as round_certificate.py
does. --single-run skips the second, confirming run (for the order-four record).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round_certificate as rc  # noqa: E402  (stdlib-only helpers)

ROOT = HERE.parent
VC_PATH = ROOT / "verification" / "standalone" / "verify_certificate.py"

# Families whose one-sided fields are odd (f_{-s}(-x) = -f_s(x)); checked by
# hand against the builders in verify_certificate.py, which use only signed
# powers, relays and the linear chain terms.
ODD_FAMILIES = {"levant_differentiator", "twisting", "continuous_twisting",
                "pidsmc", "pidsmc4d"}


class EvenPartError(Exception):
    pass


def q(value) -> Fraction:
    if isinstance(value, str):
        t = value.strip()
        if t.lower().startswith(("0x", "-0x", "+0x")):
            return Fraction(float.fromhex(t))
        return Fraction(t)
    if isinstance(value, bool):
        raise EvenPartError("boolean where a number was expected")
    return Fraction(value)


def enc(x: Fraction) -> str:
    """Hex float when exact (compact), else an exact 'p/q' string."""
    f = float(x)
    if Fraction(f) == x:
        return f.hex()
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def term_parity(term: dict) -> int:
    n_signed = 0
    for e, t in zip(term["exponents"], term["term_types"]):
        if t != "abs" and Fraction(e) != 0:
            n_signed += 1
    return n_signed % 2


def by_signs(captures: list) -> dict:
    out = {}
    for cap in captures:
        key = tuple(int(s) for s in cap["signs"])
        if key in out:
            raise EvenPartError(f"duplicate capture for octant {key}")
        out[key] = cap
    return out


def average(cap_a: dict, cap_b: dict, signs: tuple) -> dict:
    for field in ("mu", "y_degree", "eps", "basis"):
        if cap_a[field] != cap_b[field]:
            raise EvenPartError(f"paired captures differ in {field!r} at {signs}")
    ga, gb = cap_a["gram"], cap_b["gram"]
    if len(ga) != len(gb):
        raise EvenPartError("paired Gram matrices differ in size")
    gram = [[enc((q(a) + q(b)) / 2) for a, b in zip(ra, rb)] for ra, rb in zip(ga, gb)]
    out = dict(cap_a)
    out["signs"] = list(signs)
    out["gram"] = gram
    return out


def neg(s: tuple) -> tuple:
    return tuple(-v for v in s)


def even_grams(grams: dict, nominal: bool) -> dict:
    out = {k: v for k, v in grams.items() if k not in ("V_grams", "corners", "shared_W_grams")}
    V = by_signs(grams["V_grams"])
    out["V_grams"] = [average(V[s], V[neg(s)], s) for s in V]
    if "shared_W_grams" in grams:
        if not nominal:
            raise EvenPartError("shared W grams on a perturbed bundle")
        W = by_signs(grams["shared_W_grams"])
        out["shared_W_grams"] = [average(W[s], W[neg(s)], s) for s in W]
    corners = []
    for corner in grams["corners"]:
        c = {k: v for k, v in corner.items() if k not in ("endpoints", "W_grams")}
        if "endpoints" in corner:
            if nominal or len(corner["endpoints"]) != 2:
                raise EvenPartError("endpoint structure does not match the record")
            plus, minus = (by_signs(e) for e in corner["endpoints"])
            c["endpoints"] = [
                [average(plus[s], minus[neg(s)], s) for s in plus],
                [average(minus[s], plus[neg(s)], s) for s in minus],
            ]
        elif "W_grams" in corner:
            W = by_signs(corner["W_grams"])
            c["W_grams"] = [average(W[s], W[neg(s)], s) for s in W]
        corners.append(c)
    out["corners"] = corners
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--single-run", action="store_true",
                    help="run VC once (on the candidate) and write the final record with "
                         "VC's derived margin, without the confirming second run; for "
                         "records whose VC run takes hours (order four: about 10 h)")
    args = ap.parse_args(argv)

    rec_path = Path(args.record)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = rc.read_json_maybe_gz(rec_path)
    grams_path = rc.resolve_grams_path(rec_path, record)
    grams = rc.read_json_maybe_gz(grams_path)
    if record.get("system_family") not in ODD_FAMILIES:
        raise EvenPartError(f"system family {record.get('system_family')!r} "
                            f"(system {record.get('system')!r}) not known to be odd")
    nominal = bool(record.get("nominal"))

    terms = record["V"]["terms"]
    even = [t for t in terms if term_parity(t) == 0]
    odd = [t for t in terms if term_parity(t) == 1]
    if not odd:
        print("V is already even; nothing to do")
        return 0

    stem = rec_path.name.split(".record.json")[0].replace(".json", "").replace(".gz", "")
    new_rec_path = out_dir / f"{stem}.even.record.json"
    new_grams_name = f"{stem}.even.grams.json.gz"
    new_grams_path = out_dir / new_grams_name

    new_record, removed = rc.strip_margin_declarations(record)
    new_record["V"] = dict(new_record["V"])
    new_record["V"]["terms"] = even
    new_record["grams_file"] = new_grams_name
    prov = new_record.get("provenance")
    if not isinstance(prov, dict):
        prov = {"note": "unverified provenance, not part of the proved statement; "
                        "the portable verifier ignores every key in this object"}
    max_odd = max(abs(q(t["coefficient"])) for t in odd)
    prov["even_part"] = {
        "tool": "scripts/even_part_certificate.py",
        "rule": "V_e = (V(x) + V(-x))/2 keeps the even terms; witnesses averaged "
                "over the orthants s and -s (W endpoints swapped), see the tool docstring",
        "source_record": str(rec_path),
        "source_record_sha256": rc.sha256_file(rec_path),
        "source_grams": str(grams_path),
        "source_grams_sha256": rc.sha256_file(grams_path),
        "terms_source": len(terms),
        "terms_even": len(even),
        "terms_dropped_odd": len(odd),
        "max_abs_dropped_coefficient": rc.exact_string(max_odd),
        "source_margin_declarations": removed,
    }
    new_record["provenance"] = prov

    new_grams = even_grams(grams, nominal)
    raw = json.dumps(new_grams, separators=(",", ":")).encode("utf-8")
    new_grams_path.write_bytes(gzip.compress(raw, mtime=0))
    rc.write_json_file(new_rec_path, new_record)

    py = sys.executable
    ok, summary, _ = rc.run_vc(py, VC_PATH, new_rec_path, new_grams_path,
                               out_dir / f"{stem}.even.candidate.verification.json",
                               dense=False, timeout=args.timeout,
                               log_path=out_dir / f"{stem}.even.candidate.log")
    if not ok:
        print(f"FAILED: VC rejected the even part of {rec_path.name}")
        return 1
    final = rc.declare_margin(new_record, summary)
    rc.write_json_file(new_rec_path, final)
    if args.single_run:
        for suffix in ("verification.json", "log"):
            src = out_dir / f"{stem}.even.candidate.{suffix}"
            if src.exists():
                src.replace(out_dir / f"{stem}.even.{suffix}")
        print(f"VERIFIED (single run, on the candidate) {new_rec_path} terms "
              f"{len(terms)} -> {len(even)} eps_certified "
              f"{float(Fraction(summary['eps_certified_exact'])):.4e} "
              f"worst charge/eps {summary['worst_charge_over_eps']['overall']:.4f} "
              f"wall {summary['_wall_s']:.1f} s")
        return 0
    ok, summary2, _ = rc.run_vc(py, VC_PATH, new_rec_path, new_grams_path,
                                out_dir / f"{stem}.even.verification.json",
                                dense=False, timeout=args.timeout,
                                log_path=out_dir / f"{stem}.even.verification.log")
    (out_dir / f"{stem}.even.candidate.verification.json").unlink(missing_ok=True)
    (out_dir / f"{stem}.even.candidate.log").unlink(missing_ok=True)
    if not ok:
        print(f"FAILED: final record of {rec_path.name} rejected")
        return 1
    print(f"VERIFIED {new_rec_path} terms {len(terms)} -> {len(even)} "
          f"eps_certified {float(Fraction(summary2['eps_certified_exact'])):.4e} "
          f"(source {float(q(record['eps_certified_exact'])) if 'eps_certified_exact' in record else 'n/a'}) "
          f"worst charge/eps {summary2['worst_charge_over_eps']['overall']:.4f} "
          f"wall {summary2['_wall_s']:.1f} s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvenPartError, rc.RoundCertificateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
