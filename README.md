# GF-Lyapunov

Generalized-form Lyapunov functions for homogeneous sliding-mode algorithms: the code that finds
them, the exact certificates of every function in the paper, and an independent verifier.

> M. Zapf, J. A. Moreno, M. Horn, *Automatic Discovery and Exact Certification of Lyapunov
> Functions for Homogeneous Sliding-Mode Algorithms*, manuscript prepared for the International
> Journal of Robust and Nonlinear Control (not yet published).

Repository: https://github.com/MartinZapf/GF-Lyapunov · Licence: MIT (`LICENSE`). All
certificates, including their Gram files, are in this repository.

The repository has two parts:

1. **The certificates and their verifier** (no solver needed). Every Lyapunov function of the
   paper is a *certificate*: a record with the function's exact rational coefficients, the
   system, the gains (irrational gains as rational brackets), the disturbance bound delta0 and
   the margin eps, plus a Gram file with one rational Gram matrix per orthant for V and for each
   decrease rate. The independent verifier `verification/standalone/verify_certificate.py`
   (Python standard library only) rebuilds the decrease rates from the equations of the system,
   which it writes down itself, and decides paper Lemma 2 on every orthant in exact rational
   arithmetic.
2. **The GF framework** (`src/gf_framework`, `scripts/`), which designs the functions with the
   semidefinite solver MOSEK, sparsifies and rounds them, checks them exactly and writes the
   records. It is needed only to re-run a design; nothing it computes is trusted by the verifier.

## What is proved, and what is not

For each certificate the verifier proves, in exact arithmetic: V is an admissible C^1
generalized form of the stated degree; every rational bracket encloses its irrational gain; on
every closed orthant, for V and for the decrease rate at every corner of the gain box and both
disturbance extremes, the associated form is at least (eps - charge) sum_i y_i^d with
charge < eps; and the coverage of orthants, corners and extremes is complete. Theorem 1 of the
paper turns this into a strict Lyapunov function and uniform finite-time stability for every
gain in the box and every measurable disturbance with |delta(t)| <= delta0.

The scaling results (Proposition 1, Corollary 1: every Lipschitz constant L, every Delta), the
even-part result (Proposition 2), the twisting extension below the ceiling (Proposition 4) and
the necessary condition (Proposition 3) are proved in the paper; `tools/check_claims.py` checks
their computational inputs exactly. Numerical statements of the paper, such as "the design
program is infeasible at m = 3, 4, 5", are floating-point evidence, not proofs (`MANIFEST.md`,
Section 3). How each certificate was *found* is documented in `CONFIGURATION.md`; it is not part
of the proof.

`MANIFEST.md` maps every computer-assisted result of the paper (each row of Table 2, Theorem 2
including the gain box of order two, Proposition 4, the worked example of Section 5 and the
Appendix B listings) to its certificate, with gains, brackets, corners, delta0, scale, margins,
sha256 and the verifier command. `MARGINS_AND_TIMINGS.md` gives margins, Gram sizes, file sizes
and verification times with the machines they were measured on.

## Layout

```
README.md, LICENSE, .gitignore, SHA256SUMS
MANIFEST.md, MANIFEST.json            results -> certificates, gains, margins, hashes, commands
CONFIGURATION.md, config/<id>.json    the discovery procedure as implemented, per certificate
MARGINS_AND_TIMINGS.md, margins_and_timings.json
certificates/<id>/                    record (*.record.json[.gz]) and Gram file of each certificate
listings/<id>.txt                     every function: exact coefficients and terms, one per line
logs/                                 verifier logs (this package's runs and earlier runs)
checks/                               outputs of the side checks
verification/standalone/verify_certificate.py   the independent verifier (stdlib only)
tools/verify_all.py                   run the verifier on every certificate
tools/check_claims.py                 exact side checks (libraries, gains, evenness, Sturm, ...)
tools/make_listing.py                 the listing of any record
scripts/                              certify_gain_point.py (design D, exact check X, record),
                                      round_certificate.py (short coefficients),
                                      emit_certificate_record.py (portable records),
                                      even_part_certificate.py (Proposition 2),
                                      twisting_below_ceiling.py (Proposition 4, Sturm over Q),
                                      exact_gate_standard_gains.py, discover_gf.py, helpers
review/reverify_exact.py              exact re-verification driver (fixed-V route, nominal rows)
artifacts/.../p4_0*_design.py, p4_0*_exact.py, emit_x10.py
                                      the design drivers of the order-three and order-four
                                      certificates, at their development paths (they read
                                      inputs of the development repository; kept as provenance)
src/gf_framework/                     the GF framework package
tests/                                the test suite of the code above
pyproject.toml, requirements-lock.txt
```

Certificate ids (Table 2 order): `levant-order1` (eq. (6)), `levant-order2-a`, `levant-order2-b`
(29 terms), `levant-order2-box` (96 terms; `-a`, `-b`: the same function on two edges of the
box), `levant-order3-a`, `levant-order3-b` (83 terms), `levant-order4` (507 terms, nominal),
`twisting` (eq. (13)), `continuous-twisting` (9 terms), `continuous-twisting-frontier` (52
terms), `pid-like` (21 terms). The `*-dense` certificates are the dense functions whose even
parts the paper reports for the orders two and three and continuous twisting.

## Verify one certificate

Requirements: Python 3.9 or newer, nothing else. From the root of the repository:

```
python3 verification/standalone/verify_certificate.py \
    certificates/twisting/twisting_x9b_b04_delta2047_4096.printed_b04.record.json
```

The Gram file is found next to the record (or pass `--grams FILE`); `--json OUT.json` writes a
summary with the exact certified margin and the sha256 of the bytes that were read. Output:

```
[1/5] V admissible (C^1, homogeneous of degree 5, 5 terms)
[2/5] no irrational gains (all rational, injected exactly)
[3/5] V positivity: all 4 octants PASS (worst charge/eps 2.25e-11; 6 basis -> 2 parity blocks, largest 3)
[4/5] W: 2 endpoint x corner combinations, all 4 octants each, PASS (worst charge/eps 2.65e-08; 5 basis -> 2 parity blocks, largest 3)
[5/5] twisting: certificate VERIFIED — ... delta0 = 2047/4096, ...
      accepted under: charge < eps (strict, exact rational); WP P2-14 / decision D-l
      eps_certified = eps - charge = 259614836037198660742765353747/2596148429267413814265248164610048 (~1.000000e-04); worst charge/eps 2.65e-08
      note_half_margin_holds (...): yes [12/12 octants]
      psd_shift: eta = 0 on all 12 octants (the Gram itself is PSD; charge = ||E||_1)
RESULT: VERIFIED (exit code 0): every check passed
```

The `eps_certified` fraction expected for every certificate is in `MANIFEST.md`.

## Verify everything

```
python3 tools/verify_all.py            # every certificate except order four (about 10 minutes)
python3 tools/check_claims.py          # exact side checks (seconds)
sha256sum -c SHA256SUMS
```

`verify_all.py` checks every file's sha256 against `MANIFEST.json` first (a changed file is
reported as HASH MISMATCH and not run), then runs the verifier once per certificate and writes the
logs, `summary.json` and `summary.tsv` to `logs/verify_all_<timestamp>/`. It prints `ALL OK` and
exits 0 iff every certificate it ran is VERIFIED with the expected margin. On a 4-core x86_64
cloud machine the 18 certificates took 9.4 minutes in all, the four order-three certificates about
2 minutes each (`MARGINS_AND_TIMINGS.md`). `levant-order4` is marked `long_run` in
`MANIFEST.json` and skipped unless you pass `--include-long`: it took 10 h 17 min
single-threaded and about 9 GB RAM on the authors' workstation (log in `logs/workstation/`).

## Outcome classes of the verifier

Every run ends with one `RESULT:` line; the exit code says the same:

| Exit | Class | Meaning |
|---|---|---|
| 0 | VERIFIED | every check passed; the only accepting outcome |
| 1 | REJECTED | the files were read as a certificate and a **mathematical** check failed: an orthant inequality (charge >= eps or no PSD shift), a margin claim exceeding the proof ("MARGIN CLAIM EXCEEDS PROOF"), an invalid bracket, incomplete coverage (orthants, corners, disturbance extremes, a form without witness), V not admissible, or a declared quantity (kappa, degree, parity blocks) that contradicts the derived one. stderr: `VERIFICATION FAILED: <reason>` |
| 2 | INVALID INPUT | a file is missing or unreadable, a gzip stream is truncated, the JSON is malformed, or the record violates the schema (unknown version or system, missing field, wrong type, shape or number encoding, inconsistent declarations); also command-line usage errors. stderr: `INVALID INPUT: <reason>`. **Not** a mathematical rejection |
| 3 | ERROR | an unexpected condition inside the verifier (e.g. out of memory); no verdict |

A REJECTED verdict is a statement about the bytes that were read: compare their sha256 with
`SHA256SUMS` before reading it as a statement about a published certificate. The classes decide
only the message and the exit code; what is accepted does not depend on them.

## Large files

Every file is in the repository. One Gram file is larger than 50 MiB, the size above which
GitHub warns (it refuses files only above 100 MiB):

| File | sha256 | Size |
|---|---|---|
| `certificates/levant-order4/levant5d_m11_nominal.grams.json.gz` | `15fa28be0cec34636d1e0be2dd37477f495a58707dab13400ea7f01b916306bd` | 73,761,710 bytes |

## Re-running the designs and the tests

```
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"            # numpy, scipy, sympy, cvxpy (Clarabel), pytest, psutil
pip install -e ".[design]"          # MOSEK >= 11, needs a licence
python -m pytest -q                 # tests needing MOSEK or development artifacts are skipped
```

MOSEK needs a licence file (`~/mosek/mosek.lic`, or the environment variable
`MOSEKLM_LICENSE_FILE`); never commit it (`.gitignore` excludes `*.lic`). The design of record of
the twisting certificate, for example, was

```
python scripts/certify_gain_point.py --system twisting --gains r1=2,r2=3/2 \
    --delta0 2047/4096 --degrees 5 6 7 8 --sparsify rl1 --out-dir runs/twisting
```

(design margin 1e-4, the default; `config/twisting.json` has the recorded arguments of every
certificate). Its coefficients were then rounded with `scripts/round_certificate.py` and the
fixed-V route (`CONFIGURATION.md`, Section 7). An even part is rebuilt from its dense certificate
with

```
python3 scripts/even_part_certificate.py \
    --record certificates/continuous-twisting-frontier-dense/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.record.json \
    --out-dir runs/even
```

Floating-point reruns are not expected to be bit identical; whatever they produce is checked by
the same verifier.

## Licence

MIT, see `LICENSE`.
