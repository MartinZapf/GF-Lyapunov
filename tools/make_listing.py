#!/usr/bin/env python3
"""Print the readable listing of the function in a certificate record.

STDLIB ONLY. Usage, from the root of the repository:

    python3 tools/make_listing.py certificates/twisting/twisting_x9b_b04_delta2047_4096.printed_b04.record.json
    python3 tools/make_listing.py RECORD --even      # only the even terms (Proposition 2)

One line per term: index, exact coefficient p/q, term, parity. Notation:
|x|^a = |x| to the power a, [x]^a = sign(x)|x|^a, x = [x]^1; variables e1, e2, ...
for Levant's differentiator, x1, x2, ... otherwise. The files in listings/ were
written by the same rules and are checked against the records by
tools/check_claims.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from fractions import Fraction
from pathlib import Path


def read_json(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def q(v) -> Fraction:
    if isinstance(v, str) and v.strip().lower().startswith(("0x", "-0x", "+0x")):
        return Fraction(float.fromhex(v.strip()))
    return Fraction(v)


def qs(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def exp_str(e: Fraction) -> str:
    return str(e.numerator) if e.denominator == 1 else f"({e.numerator}/{e.denominator})"


def term_str(exps, types, var) -> str:
    out = []
    for i, (e, t) in enumerate(zip(exps, types)):
        e = q(e)
        name = f"{var}{i + 1}"
        if e == 0:
            if t == "signed":
                out.append(f"[{name}]^0")
            continue
        if t == "signed":
            out.append(name if e == 1 else f"[{name}]^{exp_str(e)}")
        else:
            out.append(f"|{name}|^{exp_str(e)}")
    return " * ".join(out) if out else "1"


def n_signed(t) -> int:
    return sum(1 for e, ty in zip(t["exponents"], t["term_types"])
               if ty != "abs" and q(e) != 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record")
    ap.add_argument("--even", action="store_true", help="only the even terms")
    args = ap.parse_args(argv)
    r = read_json(args.record)
    var = "e" if str(r.get("system", "")).startswith("levant") else "x"
    terms = r["V"]["terms"]
    if args.even:
        terms = [t for t in terms if n_signed(t) % 2 == 0]
    print(f"# {Path(args.record).name}: system {r.get('system')}, weights "
          f"{tuple(r['V']['weights_r'])}, degree m = {r['V']['degree_m']}, {len(terms)} terms"
          + (" (even part)" if args.even else ""))
    print("# j\talpha_j\tt_j\tparity")
    for j, t in enumerate(terms, 1):
        print(f"{j}\t{qs(q(t['coefficient']))}\t"
              f"{term_str(t['exponents'], t['term_types'], var)}\t"
              f"{'odd' if n_signed(t) % 2 else 'even'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
