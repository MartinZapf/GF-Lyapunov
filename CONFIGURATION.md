# Discovery configuration, as implemented

This file states the floating-point **design** procedure that produced every certificate of the
paper, as the code ran it (paper Section 3, "Construction of Lyapunov function candidates"). It
was extracted from the code of the GF framework and from the run records, not paraphrased from
the paper:

- `src/gf_framework/verification/sparse_sos.py` (reweighted l1, support ladder),
- `src/gf_framework/verification/joint_sos.py` and `mosek_direct.py` (the design program),
- `scripts/certify_gain_point.py` (degree loop, design D, exact check X, emission),
- `scripts/round_certificate.py` and `scripts/emit_certificate_record.py` (rounding, records),
- `scripts/even_part_certificate.py` (even parts),
- the run records `run.json`, `m<m>/design.json` and the records' `provenance` and `witness`
  blocks, named per certificate in `config/<id>.json`.

Paths under `src/`, `scripts/`, `review/`, `tests/` and the five design drivers under `artifacts/` are in this repository; every other `artifacts/...` or `docs/...` path names a run record of the development repository, cited for provenance.

**Nothing in this file is a proof input.** The verifier reads only the record and its Gram file,
and accepts or rejects them whatever procedure produced them (README, "What is proved"). A
reader who wants to re-run a design needs MOSEK and the GF framework; bit-identical
floating-point reruns are not claimed (solver versions, thread counts and platforms change the
last digits). What is claimed is that the procedure below is unambiguous and that its outputs,
whatever their last digits, are checked exactly.

## 1. Per-certificate summary

Read from `config/<id>.json`. "Degrees tried" are the degrees of the run of record, in order;
"eps" is the margin of the design program and of the exact check of that run. All designs ran
the MOSEK Task API encoding (`MOSEK_DIRECT`, MOSEK 11.1.11), strict, with no solver fallback,
on host spark-91bc.

| Certificate (paper) | Degrees tried (status) | Design eps | Library -> support | Rounding (tier, bits, eps, witness) | Scale |
|---|---|---|---|---|---|
| `levant-order1` (eq. (6)) | 3, 4, 5 infeasible; 6 designed (`loop_mosek_2026-10-12`) | 1e-6 | 14 -> 5 (rl1, 4 solves) | B, 7 bits, 1e-6, CLARABEL | L0 = 1 |
| `levant-order2-a`, `-b` (App. B) | 8 (one run, both tunings) | 1e-4 | 69 -> 29 (rl1, 4 solves) | B, 9 bits, 1e-7, CLARABEL | L0 = 1 |
| `levant-order2-box` (and `-a`, `-b`) | 11 | 1e-4 | 189, dense; even part 96 | B, 18 bits, 1e-4, MOSEK; then even part | L0 = 1 |
| `levant-order3-a` | 9 (single-degree design) | 1e-4 | 163, dense; even part 83 | A, 30 bits, 1e-4, design Gram; then even part | p = 1/2 (L0 = 1/16) |
| `levant-order3-b` | 9 (single-degree design) | 1e-4 | 163, dense; even part 83 | B, 15 bits, 1e-4, MOSEK; then even part | p = 1/2 (L0 = 1/16) |
| `levant-order4` | 11 (single-degree design), nominal | 1e-4 | 507, dense (reported with every term) | A, 26 bits, 1e-4, design Gram | L0 = 1 |
| `twisting` (eq. (13)) | 5 designed (ladder 5, 6, 7, 8; stopped at 5) | 1e-4 | 10 -> 5 (rl1, 4 solves) | B, 4 bits, 1e-4, MOSEK | none |
| `continuous-twisting` (App. B) | 5 | 1e-6 | 16 -> 9 (rl1, 3 solves) | B, 5 bits, 1e-9, CLARABEL | none |
| `continuous-twisting-frontier` | 8 `unknown:MOSEK_DIRECT`; 9 designed (ladder 8, 9, 10) | 1e-6 | 101, dense; even part 52 | B, 10 bits, 1e-6, MOSEK; then even part | none |
| `pid-like` (App. B) | 7 designed (ladder 7, 8; stopped at 7) | 1e-6 | 45 -> 21 (rl1, 5 solves) | B, 14 bits, 1e-6, MOSEK | Delta = 1 |

Degrees. The paper (Section 3.3) describes the loop m = max r_i + 1, max r_i + 2, ... For the
worked example the run of record starts there (m = 3). The other runs of record were started at
a degree chosen from earlier exploratory runs of the project (not part of this package), and the
order-three and order-four designs were single-degree runs. No degree or support is claimed to be
minimal (paper, Limitations), and none of the paper's statements depends on the lower degrees.

## 2. Inputs

A run is fully determined by: the system (one of the hand-written fields of the verifier, see
`MANIFEST.md`), its weights r, the gains (exact rationals), the irrational gains as rational
brackets `lo^k < q < hi^k` with q = gain^k rational, a gain box (pairs of rational endpoints of
affine gains), the bound delta0 (exact rational; 0 = nominal), the margin eps, and the list of
degrees. `certify_gain_point.py` checks every bracket in Q and the affinity of every varying gain
(guard R6) before any solve.

## 3. Lift and library (paper Definition 2)

k* is the smallest even positive integer such that k* r_i a is an even integer for every exponent
a of x_i in the field; mu* = k* r. At degree m the library holds the n diagonal terms
|x_i|^(m/r_i) and every term of weighted degree m with at least two positive exponents, exponent
vector on the lattice mu*_i a_i in 2Z, and admissible for C^1: an exponent in (0, 1) is excluded,
an exponent equal to 1 admits only the signed factor [x_i]^1 = x_i, an exponent above 1 admits
both |x_i|^a and [x_i]^a. Implementation: `enumerate_diagonal_terms(m, r, c1_only=True)` plus
`mu_filtered_cross` (`discovery/gf_discovery.py`), canonical library order `grlex-v1`.
`tools/check_claims.py` rebuilds the library from the field the verifier writes down and
reproduces the sizes of every design (14, 69, 189, 163, 507, 10, 16, 101, 45 terms; even terms
8, 36, 96, 83, 256, 6, 10, 52, 24).

## 4. The design program D (paper eq. (4))

Variables: the coefficients alpha of the library terms and one PSD Gram matrix per orthant s
for V and per orthant, vertex and disturbance extreme for W. Constraints, matched coefficient by
coefficient in the lifted variables y (degree d_V = k* m for V, d_W = k* (m + kappa) for W):

    Phi_s[V_alpha](y) - eps * sum_i y_i^d_V          = z^T Q_s z,            Q_s >= 0
    Phi_s[W_alpha^(w,sigma)](y) - eps * sum_i y_i^d_W = z^T Q_s^(w,sigma) z,  Q_s^(w,sigma) >= 0

for every orthant s, every vertex w of the gain box (every bracket corner) and sigma in {-1, +1}
(both disturbance extremes; one dynamics per corner when delta0 = 0). Scale: the coefficient of
|x_1|^(m/r_1) is fixed to 1; every other diagonal coefficient is bounded below by eps; cross
coefficients are free (MOSEK Task API encoding, `mosek_direct.py`). The basis z is the
degree-exact homogeneous monomial basis of degree d/2. The Gram matrices are parametrized per
y-parity block (design default). Objective: none (feasibility), except in the l1 stage below.

Status handling of one design solve (`certify_gain_point.py::run_design`):

| Solver outcome | Recorded as | Loop action |
|---|---|---|
| optimal / feasible (`standard_sos:MOSEK_DIRECT`) and the design screen passes | `designed` | go to X |
| primal infeasibility certificate (`infeasible:MOSEK_DIRECT`) | `design_infeasible_or_error` | next degree |
| no certificate either way (`unknown:MOSEK_DIRECT`) or a solver error | `design_infeasible_or_error` | next degree |
| feasible, but the float design screen refuses it (min V on the sphere, Gram residual vs eps) | `design_screen_refused` | next degree |

The design screen consults no solver status and can only reject. `infeasible` is a statement of
the floating-point solver about this library, this degree and this eps; it is never read as a
proof that no Lyapunov function of that degree exists (paper, Section 5 and Limitations).

## 5. Sparsification: reweighted l1 (paper Section 3.3)

Implemented in `design_sparse_lyapunov` (`sparse_sos.py`) with `SparseDesignConfig` at its
defaults, except `design_epsilon` = the run's eps (`certify_gain_point.py --sparsify rl1`). It
only SELECTS a support; the function of record is then designed by D on that support and proved
by X.

1. Initial weights ("normalized" arm): w_j = n_j, the Euclidean norm of term j's stacked
   associated-form coefficient column over all orthants, V and W columns (all vertices).
2. Solve the design program with objective min sum over CROSS terms j of w_j |alpha_j|
   (diagonals unpenalized), same constraints, same eps.
3. Support: diagonals always; a cross term is active iff |alpha_j| >= 1e-5 * max_cross |alpha|.
4. Update: w_j <- min( n_j / (|alpha_j| + delta), 1e6 * n_j ), delta = 1e-3 * max_cross |alpha|.
5. Termination: the support is unchanged over three consecutive solves (stable window 2), or 8
   solves, or a solve is infeasible (then the solves so far are used).
6. Support ladder: the distinct supports visited, smallest first. For each: (a) re-solve on that
   support WITHOUT the l1 objective at eps ("remargin"; infeasible -> skip); (b) mu guard: the
   V-only lift of the support must equal the dense library's (else skip, typed); (c) the float
   gate `certify_discovered_gf` (strict, solver MOSEK, all vertices) must pass (else skip). The
   first support passing (a)-(c) is selected. Backward elimination is OFF.

Supports observed (design.json): 14 -> 5 (4 solves), 69 -> 29 (4), 10 -> 5 (4), 16 -> 9 (3),
45 -> 21 (5), each "support stable". Dense designs (no sparsification): order-two box (189),
order three (163), order four (507), continuous twisting at 1.199 (101).

## 6. The exact check X and the verifier P

Both apply paper Lemma 2 in exact rational arithmetic with one acceptance rule (`margin_policy`
"achieved"): for every orthant of every form, charge = eta * C1 + ||E||_1 < eps with
Q_hat + eta I >= 0 decided by exact LDL^T per parity block, eta from the fixed ladder
0, 1e-15, ..., 1e-9, then the exact Gershgorin bound; the residual is charged densely. Every
certificate of the paper passes with eta = 0 on every orthant where the current verifier reports
it (the `psd_shift` line; for `levant-order4`, whose run predates that line, see
`MARGINS_AND_TIMINGS.md`). The certified margin is eps_certified = eps - worst charge. Older
exact stages of the order-three-a source and of order four ran the stricter `half` policy
(charge <= eps/2) in X; the verifier always uses charge < eps and states the margin it derives.

Witness chain of X (`exact_phase.run_exact_phase`): the design's persisted Gram bundle
(`witness_source = design_bundle`); for a block the bundle fails, a MOSEK re-solve of the Gram
matrices with V fixed (`resolve`), then exactly one CLARABEL retry. The witness actually used is
recorded per certificate in `config/<id>.json` (`witness`).

## 7. Rounding to short dyadic coefficients (paper Section 3.3)

A coefficient c != 0 is rounded to n significant binary digits: |c| = p * 2^e with
2^(n-1) <= p < 2^n, round half to even, exactly (`round_certificate.py --unit bits`). The first
diagonal coefficient is 1 and stays 1. n = 1, 2, 3, ... are tried in turn and the first n that
passes is kept (the stopping rule); "passes" means:

- **Tier A** (`round_certificate.py tier-a`): the design's Gram bundle is kept byte for byte and
  the verifier VERIFIES the rounded record. An exact residual screen first skips every n with
  max ||E||_1 / eps >= 1, which proves that the verifier would reject it. Used for
  `levant-order3-a` (30 bits) and `levant-order4` (26 bits).
- **Tier B** (fixed-V route): for each n, all Gram matrices are recomputed with V fixed by small
  SDPs at one fixed eps (`certify_gain_point.py --v-row` or `exact_gate_standard_gains.py --row`,
  stage label `fixed_v_row_not_D`), then X, emission and the verifier must all pass. A rounding
  for which the Gram re-solve is infeasible has no witness and is rejected; this is a float
  verdict, not an exact refutation. Used for every other certificate.

Which record is printed: tier B where one exists, otherwise tier A. Margin
policy: one eps per certificate from start to finish; the record writer refuses a capture
certified at a different eps (`epsilon_ladder_mismatch`). "Possibly at a smaller margin" in the
paper means a separate tier-B run at a smaller eps: 1e-7 for the order-two function (9 bits; at
1e-4 the shortest was 14 bits) and 1e-9 for continuous twisting (5 bits; at 1e-6 the design's
23-bit coefficients were kept). The shorter roundings that were tried and rejected are listed in
`config/<id>.json` (`rounding.shorter_rejected`).

## 8. Even parts (paper Proposition 2)

`scripts/even_part_certificate.py --record R --out-dir D` keeps the terms of V with an even
number of signed factors, with unchanged coefficients, and averages the witnesses:
V_grams[s] := (V_grams[s] + V_grams[-s]) / 2; for a perturbed record and each corner,
W+[s] := (W+[s] + W-[-s]) / 2 and W-[s] := (W-[s] + W+[-s]) / 2; for a nominal record
W[s] := (W[s] + W[-s]) / 2 (paired captures must share mu, degree, eps and basis). It writes a
new record and Gram file and lets the verifier decide them. Every even part the paper reports
(order-two box, order three a and b, continuous twisting at 1.199) was VERIFIED this way (logs in
`logs/`). The order-four function is nominal and is reported with all 507 terms.

## 9. Scale and normalization of each design

- Normalization of V: the coefficient of |x_1|^(m/r_1) (resp. |e_1|^(m/r_1)) is 1 in every design
  and every record.
- Levant's differentiator, orders one, two and four: records at L = 1 (L0 = 1), gains
  lambda_i of Table 1, delta0 = 1 (orders one and two) or 0 (order four).
- Order three: representative p = 1/2 of the covariant family lambda_i -> p^i lambda_i,
  delta0 -> p^4 delta0, i.e. L0 = p^4 = 1/16: rational gains lambda_1 / 2 and 11/160, bracket
  power targets (lambda_i p^i)^k (checked by `tools/check_claims.py`), delta0 = 1/16. This was a
  numerical design choice: at p = 1 the m = 9 design returned `solver_unknown` at eps 1e-4 for both
  tunings (and `solver_infeasible` at eps 1e-3 for tuning a), at p = 3/4 and p = 1/2 it designed;
  p = 1/2 was adopted (`artifacts/campaign_2026-08/E4_levant4d/ratio1_2026-09-04/README.md`,
  `artifacts/campaign_2026-09/X10/README.md`). Corollary 1 transfers the result to every L.
- Twisting and continuous twisting: at the gains themselves. PID-like: at Delta = 1 (the record's
  fixed parameter `delta = 1`), so the gains are D1 itself.

## 10. Software of the design runs

spark-91bc (Linux aarch64, 20 cores, 121 GiB RAM; CPU model not recorded), CPython
3.12.3 venv, MOSEK 11.1.11, cvxpy 1.8.1, CLARABEL 0.11.1, numpy 2.4.2, scipy 1.17.0, sympy 1.14.0
(recorded in the run records of the development repository). The cloud tier-B runs of 2026-10-12/17
(`levant-order1`, `levant-order2-a/-b`, `continuous-twisting`): Linux x86_64, CPython 3.13.16,
CLARABEL 0.11.1, cvxpy 1.9.3, numpy 2.5.3, scipy 1.18.1, no MOSEK (their `env.json`). The git
commit of each run is in its `run.json` / `env.json` and in the record (`git_sha`).
