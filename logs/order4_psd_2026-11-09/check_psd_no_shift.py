#!/usr/bin/env python3
"""Order four (levant-order4): every parity-projected Gram matrix is PSD without a shift.

The full verifier run of this certificate (logs/workstation/, about ten hours) predates the
verifier's `psd_shift` report line, so its log does not say which shift eta was accepted. This
script closes that gap: for each of the 32 Gram matrices of V and the 32 shared Gram matrices of
the decrease rates, it forms Q_hat exactly as `validate_octant` of the released verifier does and
calls the verifier's own exact test `blocks_psd(Q, blocks, 0)` (LDL^T over Q on every parity
block, no shift). It does not recompute the residuals: `charge < eps` on all 288 forms is the
result of the workstation run, and charge = eta * C1 + ||E||_1 >= ||E||_1, so with Q_hat >= 0
both conditions of the paper's Lemma 2 hold for every witness.

Usage, from the repository root (standard library only; about 9 GB RAM, 4 processes,
about two hours on a 4-core machine):
    python3 logs/order4_psd_2026-11-09/check_psd_no_shift.py
"""
import hashlib
import json
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verification" / "standalone"))
import verify_certificate as vc  # noqa: E402

GRAMS = ROOT / "certificates" / "levant-order4" / "levant5d_m11_nominal.grams.json.gz"
BUNDLE = None


def work(job):
    kind, k = job
    cap = (BUNDLE["V_grams"] if kind == "V" else BUNDLE["shared_W_grams"])[k]
    t0 = time.time()
    basis = [tuple(int(e) for e in b) for b in cap["basis"]]
    nb, nv = len(basis), len(basis[0])
    g = cap["gram"]
    Q = [[(vc.q_from(g[i][j]) + vc.q_from(g[j][i])) / 2 for j in range(nb)] for i in range(nb)]
    # The associated forms have even exponents only (Lemma 1), so the sign-flip group is the full
    # one and the parity blocks are the classes of the basis exponents modulo 2.
    support = [tuple(2 * e for e in b) for b in basis]
    blocks = vc.parity_blocks(basis, vc.deck_group_basis(support, nv))
    classes = {}
    for i, b in enumerate(basis):
        classes.setdefault(tuple(e % 2 for e in b), []).append(i)
    same = sorted(map(sorted, blocks)) == sorted(map(sorted, classes.values()))
    ok = vc.blocks_psd(Q, blocks, Fraction(0))
    return dict(kind=kind, octant=cap["octant_index"], signs=cap["signs"], n_basis=nb,
                n_blocks=len(blocks), largest=max(map(len, blocks)),
                blocks_are_parity_classes=same, psd_eta0=bool(ok),
                seconds=round(time.time() - t0, 1))


def main():
    global BUNDLE
    BUNDLE = vc.read_json(str(GRAMS))
    jobs = ([("V", k) for k in range(len(BUNDLE["V_grams"]))]
            + [("W", k) for k in range(len(BUNDLE["shared_W_grams"]))])
    out = []
    with Pool(4) as pool:  # fork: the workers share BUNDLE
        for r in pool.imap_unordered(work, jobs):
            out.append(r)
            print(json.dumps(r), flush=True)
    res = dict(
        verifier_sha256=hashlib.sha256(Path(vc.__file__).read_bytes()).hexdigest(),
        grams_sha256=hashlib.sha256(GRAMS.read_bytes()).hexdigest(),
        n_checked=len(out),
        all_psd_eta0=all(r["psd_eta0"] for r in out),
        all_blocks_parity=all(r["blocks_are_parity_classes"] for r in out),
        results=sorted(out, key=lambda r: (r["kind"], r["octant"])))
    (Path(__file__).parent / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print("SUMMARY n", len(out), "all_psd_eta0", res["all_psd_eta0"],
          "blocks_are_parity_classes", res["all_blocks_parity"])
    return 0 if res["all_psd_eta0"] and res["all_blocks_parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
