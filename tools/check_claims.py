#!/usr/bin/env python3
"""Exact side checks of the paper's computer-assisted claims.

STDLIB ONLY (Python >= 3.9). Run from the root of the release:

    python3 tools/check_claims.py

The verifier decides every certificate. This script checks, in exact rational
arithmetic, the statements AROUND the certificates that a reader would
otherwise have to take from the text:

  manifest   every record's sha256, system, weights, degree, term count,
             delta0, margin, gains and brackets equal MANIFEST.json, and every
             listing in listings/ equals its record term for term;
  library    the library of Definition 1 at each certified degree, built from
             the exponents of the field that the VERIFIER rebuilds: its size,
             its number of even terms, and that every dense function uses all
             of it and every even part all of its even terms;
  even       every function the paper reports is even (no term with an odd
             number of signed factors);
  gains      the rational gains and the bracket power targets of every
             Levant record equal (lambda_i p^i)^k for the non-recursive gains
             converted from Levant's recursive tunings (paper eq. (10),
             Table 1), with p = 1 (orders one, two, four) or p = 1/2 (order
             three, L0 = p^4 = 1/16), and delta0 = p^4 or 0;
  box        2.121 < (3/2) sqrt 2 and (3/2) sqrt 3 < 2.599, so the order-two
             box contains (c, (3/2) sqrt c, 1.1) for every 2 <= c <= 3;
  bounds     twisting 2047/4096 = (2047/2048)(c1 - c2); continuous twisting
             1199/1000 = (1199/1200)(k3 - k4) > k4; PID-like gains = D1;
  worked     the Gram sizes and parity blocks of Section 5 (7 and 6; 4+3 and
             3+3), read from the Gram bundle's bases;
  twisting   scripts/twisting_below_ceiling.py (Proposition 4: four boundary
             polynomials, Sturm sequences over Q) runs and passes, and its
             hard-coded V equals the twisting record's V.

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_certificate", ROOT / "verification" / "standalone" / "verify_certificate.py")
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)

FAILS: list = []


def check(ok: bool, what: str):
    print(f"[{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILS.append(what)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q(v) -> Fraction:
    return vc.q_from(v)


def term_key(t):
    """(exponents, types) with type 'abs' for a zero exponent."""
    exps = tuple(Fraction(e) if not isinstance(e, Fraction) else e
                 for e in (q(x) for x in t["exponents"]))
    types = tuple("abs" if e == 0 else ty for e, ty in zip(exps, t["term_types"]))
    return exps, types


def n_signed(key) -> int:
    exps, types = key
    return sum(1 for e, ty in zip(exps, types) if e != 0 and ty == "signed")


# ---------------------------------------------------------------------------
# Library of Definition 1, from the field the verifier rebuilds
# ---------------------------------------------------------------------------

def k_star(dyn, weights) -> int:
    """Smallest even k with k r_i a even for every exponent a of x_i in f."""
    exps = [set() for _ in weights]
    for f_i in dyn:
        for _c, e, _t in f_i:
            for i, a in enumerate(e):
                exps[i].add(Fraction(a))
    k = 2
    while True:
        if all((k * weights[i] * a).denominator == 1 and (k * weights[i] * a) % 2 == 0
               for i in range(len(weights)) for a in exps[i]):
            return k
        k += 2


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def library(weights, m, kstar):
    """Diagonal terms |x_i|^(m/r_i) and every admissible cross term of degree
    m with at least two positive exponents on the lattice mu* a_i in 2Z,
    mu* = k* r. On the lattice a_i = 2 j_i / (k* r_i) with sum j_i = k* m / 2."""
    n = len(weights)
    out = set()
    for i in range(n):
        exps = tuple(Fraction(m, weights[i]) if t == i else Fraction(0) for t in range(n))
        out.add((exps, ("abs",) * n))
    total = Fraction(kstar * m, 2)
    assert total.denominator == 1
    for j in compositions(int(total), n):
        a = tuple(Fraction(2 * j[i], kstar * weights[i]) for i in range(n))
        if sum(1 for x in a if x > 0) < 2:
            continue
        if any(0 < x < 1 for x in a):           # not C^1
            continue
        choices = []
        for x in a:
            if x == 0:
                choices.append(("abs",))
            elif x == 1:
                choices.append(("signed",))      # |x|^1 is not C^1 at 0
            else:
                choices.append(("abs", "signed"))
        for types in product(*choices):
            out.add((a, types))
    return out


# ---------------------------------------------------------------------------
# Levant's gains: the conversion (10) with exact exponents
# ---------------------------------------------------------------------------

def nonrecursive(recursive):
    """lambda_i as {index j of the recursive gain: exponent}, eq. (10):
    lambda_1 = h_{n-1}, lambda_i = h_{n-i} lambda_{i-1}^((n-i)/(n-i+1))."""
    n = len(recursive)
    lam = [{n - 1: Fraction(1)}]
    for i in range(2, n + 1):
        prev = lam[-1]
        cur = {j: e * Fraction(n - i, n - i + 1) for j, e in prev.items()}
        cur[n - i] = cur.get(n - i, Fraction(0)) + 1
        lam.append(cur)
    return lam


def power_value(rep, recursive, k, p_pow):
    """(prod h_j^(e_j))^k * p_pow as an exact rational, or None if irrational."""
    val = Fraction(1)
    for j, e in rep.items():
        ke = k * e
        if ke.denominator != 1:
            return None
        val *= Fraction(recursive[j]) ** int(ke)
    return val * p_pow


TUNINGS = {   # recursive (h_0, ..., h_{n-1}), paper Table 1
    "levant-order1": [Fraction(11, 10), Fraction(3, 2)],
    "levant-order2-a": [Fraction(11, 10), Fraction(3, 2), Fraction(2)],
    "levant-order2-b": [Fraction(11, 10), Fraction(3, 2), Fraction(3)],
    "levant-order3-a": [Fraction(11, 10), Fraction(3, 2), Fraction(2), Fraction(3)],
    "levant-order3-b": [Fraction(11, 10), Fraction(3, 2), Fraction(3), Fraction(5)],
    "levant-order4": [Fraction(11, 10), Fraction(3, 2), Fraction(2), Fraction(3),
                            Fraction(5)],
}
TUNINGS["levant-order3-a-dense"] = TUNINGS["levant-order3-a"]
TUNINGS["levant-order3-b-dense"] = TUNINGS["levant-order3-b"]
GAIN_NAMES = {"sta": ["k1", "k2"]}


def check_gains(cid, rec, p):
    rec_h = TUNINGS[cid]
    n = len(rec_h)
    names = GAIN_NAMES.get(rec["system"], [f"lambda{i}" for i in range(1, n + 1)])
    lam = nonrecursive(rec_h)
    ok = True
    for i, (name, rep) in enumerate(zip(names, lam), 1):
        if name in rec["rational_gains"]:
            v = power_value(rep, rec_h, 1, p ** i)
            ok &= v is not None and v == q(rec["rational_gains"][name])
        else:
            b = rec["brackets"][name]
            k = int(b["power_k"])
            v = power_value(rep, rec_h, k, p ** (i * k))
            ok &= v is not None and v == q(b["power_target"])
            lo, hi = q(b["lo"]), q(b["hi"])
            ok &= lo ** k < v < hi ** k
    return ok


# ---------------------------------------------------------------------------

def main() -> int:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    certs = {c["id"]: c for c in manifest["certificates"]}
    records = {}

    print("== manifest, records and listings")
    for cid, c in certs.items():
        files = c["files"]
        lst = ROOT / c["listing"]
        if True:
            rp = ROOT / files["record"]["path"]
            check(sha256(rp) == files["record"]["sha256"], f"{cid}: record sha256")
            gp = ROOT / files["grams"]["path"]
            if gp.exists():
                check(sha256(gp) == files["grams"]["sha256"], f"{cid}: Gram file sha256")
            else:
                check(False, f"{cid}: Gram file {files['grams']['path']} not present")
            r = vc.read_json(rp)
            records[cid] = r
            terms = r["V"]["terms"]
            same = (r["system"] == c["system"] and r["V"]["degree_m"] == c["degree_m"]
                    and list(r["V"]["weights_r"]) == list(c["weights_r"])
                    and len(terms) == c["n_terms"] and r["delta0_exact"] == c["delta0"]
                    and r["epsilon_exact"] == c["epsilon"]["exact"]
                    and r["rational_gains"] == c["gains_rational"]
                    and r.get("brackets", {}) == c["brackets"])
            check(same, f"{cid}: system, m, weights, {len(terms)} terms, delta0, eps, gains, "
                        f"brackets equal the manifest")
        rows = [l.split("\t") for l in lst.read_text(encoding="utf-8").splitlines()
                if l and not l.startswith("#")]
        coeffs_ok = len(rows) == len(terms) and all(
            Fraction(row[1]) == q(t["coefficient"]) for row, t in zip(rows, terms))
        check(coeffs_ok, f"{cid}: listing {c['listing']} equals the record term for term "
                         f"({len(rows)} rows)")

    print("== evenness of the reported functions")
    for cid, c in certs.items():
        if c["role"] != "reported" or cid not in records:
            continue
        odd = sum(1 for t in records[cid]["V"]["terms"] if n_signed(term_key(t)) % 2)
        if cid == "levant-order4":
            # Nominal and not printed: reported with every library term (paper Section 4.3
            # reports by their even parts only the dense functions WITH a disturbance).
            check(records[cid]["delta0_exact"] in ("0", "0/1") and odd > 0,
                  f"{cid}: nominal dense function reported with all its terms "
                  f"({len(records[cid]['V']['terms'])} terms, {odd} of them odd)")
            continue
        check(odd == 0, f"{cid}: reported function is even ({odd} odd terms)")

    print("== library of Definition 1 (from the field rebuilt by the verifier)")
    expect = {  # paper's numbers (source comments of main.tex and Section 5)
        "levant-order1": None, "levant-order2-box-dense": (189, 96),
        "levant-order3-a-dense": (163, 83), "levant-order3-b-dense": (163, 83),
        "levant-order4": (507, None), "continuous-twisting-frontier-dense": (101, 52),
    }
    for cid, r in records.items():
        gains = {k: q(v) for k, v in r["rational_gains"].items()}
        corner = r["corners"][0]["assignment"]
        gains.update({k: q(v) for k, v in corner.items()})
        dyn = vc.build_system_dynamics(r["system"], gains, Fraction(0), None)
        weights = [int(w) for w in r["V"]["weights_r"]]
        m = int(r["V"]["degree_m"])
        ks = k_star(dyn, weights)
        lib = library(weights, m, ks)
        even_lib = {t for t in lib if n_signed(t) % 2 == 0}
        support = {term_key(t) for t in r["V"]["terms"]}
        inside = support <= lib
        line = (f"{cid}: k* = {ks}, mu* = {tuple(ks * w for w in weights)}, library at m = {m}: "
                f"{len(lib)} terms, {len(even_lib)} even; V uses {len(support)}")
        check(inside, line + " (all on the library)")
        if cid in expect and expect[cid]:
            want_all, want_even = expect[cid]
            check(len(lib) == want_all and want_even in (None, len(even_lib)),
                  f"{cid}: library size {len(lib)} = paper {want_all}"
                  + (f", even terms {len(even_lib)} = paper {want_even}" if want_even else ""))
            check(support == lib, f"{cid}: the dense function uses every library term")
        if cid == "levant-order1":
            check(len(lib) == 14, f"{cid}: library of Section 5 has 14 terms")
        if cid.endswith("-dense"):
            continue
        if cid in ("levant-order2-box", "levant-order2-box-a", "levant-order2-box-b",
                   "levant-order3-a", "levant-order3-b", "continuous-twisting-frontier"):
            check(support == even_lib, f"{cid}: the even part uses every even library term "
                                       f"({len(support)} = {len(even_lib)})")

    print("== Levant gains, scale and delta0 (Table 1, eq. (10), Corollary 1)")
    for cid, p, d0 in (("levant-order1", Fraction(1), Fraction(1)),
                       ("levant-order2-a", Fraction(1), Fraction(1)),
                       ("levant-order2-b", Fraction(1), Fraction(1)),
                       ("levant-order3-a", Fraction(1, 2), Fraction(1, 16)),
                       ("levant-order3-b", Fraction(1, 2), Fraction(1, 16)),
                       ("levant-order3-a-dense", Fraction(1, 2), Fraction(1, 16)),
                       ("levant-order3-b-dense", Fraction(1, 2), Fraction(1, 16)),
                       ("levant-order4", Fraction(1), Fraction(0))):
        if cid not in records:
            continue
        r = records[cid]
        check(check_gains(cid, r, p), f"{cid}: gains = lambda_i p^i of Table 1 with p = {p} "
                                      f"(brackets enclose them, in Q)")
        check(q(r["delta0_exact"]) == d0, f"{cid}: delta0 = {d0}"
              + (f" = L0 = p^4" if p != 1 else ""))

    print("== order-two box")
    lo2, hi2 = Fraction(2121, 1000), Fraction(2599, 1000)
    check(lo2 ** 2 < Fraction(9, 2) and Fraction(27, 4) < hi2 ** 2,
          "2.121^2 < 9/2 and 27/4 < 2.599^2: the box contains (c, (3/2) sqrt c, 1.1), 2 <= c <= 3")
    if "levant-order2-box" in records:
        r = records["levant-order2-box"]
        b = r["brackets"]
        check(q(b["lambda1"]["lo"]) == 2 and q(b["lambda1"]["hi"]) == 3
              and q(b["lambda2"]["lo"]) == lo2 and q(b["lambda2"]["hi"]) == hi2
              and q(r["rational_gains"]["lambda3"]) == Fraction(11, 10),
              "levant-order2-box: corners span [2,3] x [2.121,2.599] x {11/10}")

    print("== disturbance bounds of the controllers")
    if "twisting" in records:
        r = records["twisting"]
        c1, c2, d0 = (q(r["rational_gains"]["r1"]), q(r["rational_gains"]["r2"]),
                      q(r["delta0_exact"]))
        check((c1, c2) == (2, Fraction(3, 2)) and d0 == Fraction(1, 2) - Fraction(1, 4096)
              and d0 == Fraction(2047, 2048) * (c1 - c2),
              "twisting: (c1, c2) = (2, 3/2), delta0 = 2047/4096 = 1/2 - 2^-12 = (2047/2048)(c1 - c2)")
    for cid in ("continuous-twisting", "continuous-twisting-frontier"):
        if cid in records:
            g = {k: q(v) for k, v in records[cid]["rational_gains"].items()}
            d0 = q(records[cid]["delta0_exact"])
            ok = g == {"k1": 25, "k2": 15, "k3": Fraction(23, 10), "k4": Fraction(11, 10)}
            if cid.endswith("frontier"):
                ok &= d0 == Fraction(1199, 1200) * (g["k3"] - g["k4"]) and d0 > g["k4"]
            else:
                ok &= d0 == 1
            check(ok, f"{cid}: G1 = (25, 15, 23/10, 11/10), delta0 = {d0}"
                      + (" = (1199/1200)(k3 - k4) > k4" if cid.endswith("frontier") else ""))
    if "pid-like" in records:
        g = {k: q(v) for k, v in records["pid-like"]["rational_gains"].items()}
        check(g == {"lambda1": Fraction(27, 10), "lambda2": Fraction(1069, 200),
                    "lambda3": Fraction(11, 10), "delta": 1}
              and q(records["pid-like"]["delta0_exact"]) == 1,
              "pid-like: D1 = (2.7, 5.345, 1.1) at Delta = delta0 = 1")

    print("== worked example, Section 5")
    c = certs.get("levant-order1")
    if c:
        g = vc.read_json(ROOT / c["files"]["grams"]["path"])
        from collections import Counter

        def blocks(cap):
            return sorted(Counter(tuple(int(v) % 2 for v in b) for b in cap["basis"]).values(),
                          reverse=True)
        v_sizes = {len(cap["basis"]) for cap in g["V_grams"]}
        w_caps = [cap for corner in g["corners"] for e in corner["endpoints"] for cap in e]
        w_sizes = {len(cap["basis"]) for cap in w_caps}
        n_mats = len(g["V_grams"]) + len(w_caps)
        check(v_sizes == {7} and w_sizes == {6} and n_mats == 12,
              f"levant-order1: 12 Gram matrices, size 7 for V and 6 for W ({n_mats}, {v_sizes}, {w_sizes})")
        check({tuple(blocks(cp)) for cp in g["V_grams"]} == {(4, 3)}
              and {tuple(blocks(cp)) for cp in w_caps} == {(3, 3)},
              "levant-order1: parity blocks 4+3 (V) and 3+3 (W)")

    print("== twisting below the ceiling (Proposition 4)")
    tool = ROOT / "scripts" / "twisting_below_ceiling.py"
    res = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True, cwd=ROOT)
    check(res.returncode == 0, f"scripts/twisting_below_ceiling.py exit {res.returncode}")
    # The four boundary polynomials as printed in the paper (Appendix A, proof of
    # Proposition 4), transcribed here; the script derives them from V and the field.
    printed = {"45u^4/32 + 5u^3 - 69u^2/16 - 15u/8 + 183/128",
               "15u^4/8 + 15u^3/2 - 87u^2/16 - 15u/8 + 57/32",
               "5u^3/2 - 15u^2/16 + 15u/8 + 3/8",
               "15u^4/32 - 33u^2/16 + 15u/8 + 3/128"}
    derived = {line.split("<-")[0].strip() for line in res.stdout.splitlines() if "<-" in line}
    check(derived == printed, "the four polynomials derived by the script are those printed in "
                              "the paper's proof of Proposition 4")
    if "twisting" in records:
        spec2 = importlib.util.spec_from_file_location("tbc", tool)
        tbc = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(tbc)
        script_v = set()
        for coeff, factors in tbc.V_TERMS:
            exps = tuple(Fraction(a) for a, _s in factors)
            types = tuple("abs" if a == 0 else ("signed" if s else "abs") for a, s in factors)
            script_v.add((Fraction(coeff), exps, types))
        rec_v = {(q(t["coefficient"]),) + term_key(t) for t in records["twisting"]["V"]["terms"]}
        check(script_v == rec_v, "the V of twisting_below_ceiling.py equals the twisting record's V")

    print(f"\n{'ALL CHECKS PASS' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
