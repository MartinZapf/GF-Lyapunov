# Manifest of the computer-assisted results

Generated from the certificate records; `MANIFEST.json` carries the same content in machine-readable form. Paper: *Automatic Discovery and Exact Certification of Lyapunov Functions for Homogeneous Sliding-Mode Algorithms* (Zapf, Moreno, Horn), submitted version of 2026-11-09. Section, equation, table and proposition numbers follow that version.

Verifier: `verification/standalone/verify_certificate.py`, sha256 `1a1908df5bd34b160d269f4cdbda4b356ed769859b181ef7af64c3bbd0113aa5`. Acceptance: for every orthant of V and of every decrease rate, charge = eta C1 + ||E||_1 < eps in exact rational arithmetic; the certified margin is eps_certified = eps - (worst charge). Exit codes: 0 VERIFIED, 1 REJECTED, 2 INVALID INPUT, 3 ERROR (README.md).

## 1. Results of the paper and the certificates behind them

| # | Where in the paper | Claim | Certificates | How the claim follows | Status |
|---|---|---|---|---|---|
| R1 | Section 5 (worked example), eq. (6) | eq. (6) is a strict Lyapunov function of super-twisting at (3/2, 11/10) for every measurable |delta| <= 1; the exact check accepts one V witness and two W witnesses per orthant, largest residual ||E||_1 below 4e-7 eps; Gram sizes 7 (V) and 6 (W), parity blocks 4+3 and 3+3 | `levant-order1` | certificate | `levant-order1`: VERIFIED |
| R2 | Theorem 2, k = 1; Table 2 row k = 1 | order one exact in finite time at (1.1, 1.5) for every L > 0 | `levant-order1` | certificate + Corollary 1 (L0 = 1) | `levant-order1`: VERIFIED |
| R3 | Theorem 2, k = 2 at tunings a and b; Table 2 row k = 2 a and b; Appendix B | the 29-term function (m = 8) is certified at tuning a with lambda2 in [2.121, 2.122] and at tuning b with lambda2 in [2.598, 2.599], delta0 = 1 | `levant-order2-a`, `levant-order2-b` | certificate + Corollary 1 (L0 = 1) | `levant-order2-a`: VERIFIED; `levant-order2-b`: VERIFIED |
| R4 | Theorem 2, k = 2, last sentence; Table 2 row k = 2 all tunings between; Figure 3 | the 96-term function (m = 11, all even terms of the library) is certified on the box [2,3] x [2.121,2.599] x {1.1}, which contains (c, 3/2 sqrt(c), 1.1) for every 2 <= c <= 3 | `levant-order2-box` | certificate (even part, verified directly) + Corollary 1; side checks `box_contains_tunings`, `library_even_support:levant-order2-box` (tools/check_claims.py) | `levant-order2-box`: VERIFIED |
| R5 | Theorem 2, k = 3, tunings a and b; Table 2 rows k = 3 | for each tuning, an 83-term function (m = 9, all even terms of the library) is certified at L0 = 1/16 (gains lambda_i / 2^i, delta0 = 1/16), irrational gains in rational brackets | `levant-order3-a`, `levant-order3-b` | certificate (even part, verified directly) + Corollary 1 (L0 = 1/16); side checks `scaled_gains:levant-order3-a`, `scaled_gains:levant-order3-b`, `library_even_support:levant-order3-a`, `library_even_support:levant-order3-b` (tools/check_claims.py) | `levant-order3-a`: VERIFIED; `levant-order3-b`: VERIFIED |
| R6 | Section 6.2, Order four; Table 2 row k = 4 | without disturbance, a function of degree 11 that uses all 507 terms of the library is certified at tuning a | `levant-order4` | certificate (VERIFIED on the authors' workstation, 10 h 17 min; not re-run here) + Lemma 3 and Corollary 1 (every L, polynomial signals of degree at most four); side checks `library_full_support:levant-order4`, `scaled_gains:levant-order4` (tools/check_claims.py) | `levant-order4`: VERIFIED on the workstation (not re-run here) |
| R7 | Section 7.1, eq. (13); Proposition 4; Table 2 row Twisting | eq. (13) is certified at delta0 = 2047/4096, and by exact Sturm root counting a strict Lyapunov function for every delta0 < 1/2 at (c1, c2) = (2, 3/2) | `twisting` | certificate + exact Sturm check (scripts/twisting_below_ceiling.py); side checks `twisting_sturm`, `twisting_V_matches_record` (tools/check_claims.py) | `twisting`: VERIFIED |
| R8 | Section 7.2; Table 2 row Continuous twisting delta0 = 1; Appendix B | a 9-term function (m = 5) certifies G1 at delta0 = 1 | `continuous-twisting` | certificate | `continuous-twisting`: VERIFIED |
| R9 | Section 7.2; Table 2 row Continuous twisting delta0 = 1.199; Conclusion | a 52-term function (m = 9, all even terms of the library) certifies G1 at delta0 = 1.199 = 1199/1200 of the ceiling k3 - k4 = 1.2 | `continuous-twisting-frontier` | certificate (even part, verified directly); side checks `library_even_support:continuous-twisting-frontier` (tools/check_claims.py) | `continuous-twisting-frontier`: VERIFIED |
| R10 | Section 7.3; Table 2 row PID-like; Appendix B | a 21-term function (m = 7) certifies the PID-like controller at D1 and Delta = delta0 = 1 | `pid-like` | certificate + Proposition 1 (every Delta) | `pid-like`: VERIFIED |
| R11 | Appendix B (three listings) and eqs. (6), (13) | the printed coefficients and terms are those of the certified records | `levant-order1`, `levant-order2-a`, `levant-order2-b`, `twisting`, `continuous-twisting`, `pid-like` | transcription check: the printed coefficients and terms were compared with the records term by term (checks/paper_vs_records.txt, no mismatch) | `levant-order1`: VERIFIED; `levant-order2-a`: VERIFIED; `levant-order2-b`: VERIFIED; `twisting`: VERIFIED; `continuous-twisting`: VERIFIED; `pid-like`: VERIFIED |
| R12 | Section 4.3 (Symmetry) and Section 8, library sizes in source comments | the printed functions are even; the even parts reported for the dense functions with a disturbance use all 52, 83 and 96 even terms of libraries of 101, 163 and 189 terms; the order-four function uses all 507 terms of its library | - | combinatorial check (tools/check_claims.py); side checks `all_reported_even`, `library_sizes` (tools/check_claims.py) | checked by tools/check_claims.py: PASS |

Side checks (`tools/check_claims.py`, exact, all PASS on 2026-11-05, output in `checks/check_claims.out`): library sizes and even terms of Definition 2 from the verifier's own fields; every printed function and every even part even; the dense functions use the whole library (the order-four function all 507 terms) and the even parts every even term; Levant gains = lambda_i p^i of Table 1 through eq. (9), brackets enclosing them in Q; the order-two box contains every tuning (1.1, 1.5, c), 2 <= c <= 3; the controller bounds; the Gram sizes of Section 5; Proposition 4's Sturm check and its V = the record's V. The printed functions of eqs. (6), (13) and Appendix B were compared with the records term by term, and Table 2 and the stated numbers with the manifest (`checks/paper_vs_records.txt`, no mismatch).

## 2. Certificates

Paths are relative to the root of the release. *reported*: the function the paper reports. *source*: the dense certificate a reported even part was derived from (Proposition 2), included so that the derivation can be re-run.

| id | role | system | n | m | terms | delta0 | scale | eps | eps_certified | record sha256 (12) | verified here |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `levant-order1` | reported | sta | 2 | 6 | 5 | 1 | L0 = 1 | 1e-06 | 1.000e-06 | `9e0b9f12d695` | VERIFIED, 0.08 s |
| `levant-order2-a` | reported | levant3d | 3 | 8 | 29 | 1 | L0 = 1 | 1e-07 | 9.997e-08 | `6af6214320a7` | VERIFIED, 1.02 s |
| `levant-order2-b` | reported | levant3d | 3 | 8 | 29 | 1 | L0 = 1 | 1e-07 | 1.000e-07 | `4faebe05a7f6` | VERIFIED, 1.11 s |
| `levant-order2-box` | reported | levant3d | 3 | 11 | 96 | 1 | L0 = 1 | 1e-04 | 1.000e-04 | `0e0717dd5fbb` | VERIFIED, 9.98 s |
| `levant-order3-a` | reported | levant4d | 4 | 9 | 83 | 1/16 | p = 1/2 (L0 = 1/16) | 1e-04 | 2.700e-05 | `4636bf676eba` | VERIFIED, 1 min 59 s |
| `levant-order3-b` | reported | levant4d_2003print | 4 | 9 | 83 | 1/16 | p = 1/2 (L0 = 1/16) | 1e-04 | 1.000e-04 | `1d0d37819d73` | VERIFIED, 2 min 29 s |
| `levant-order4` | reported | levant5d | 5 | 11 | 507 | 0 | L0 = 1 | 1e-04 | 7.272e-06 | `3386556ae24b` | not re-run (workstation: VERIFIED, 10 h 17 min) |
| `twisting` | reported | twisting | 2 | 5 | 5 | 2047/4096 | - | 1e-04 | 1.000e-04 | `be4a8ec13591` | VERIFIED, 0.08 s |
| `continuous-twisting` | reported | continuous_twisting | 3 | 5 | 9 | 1 | - | 1e-09 | 9.998e-10 | `b847924c4720` | VERIFIED, 0.16 s |
| `continuous-twisting-frontier` | reported | continuous_twisting | 3 | 9 | 52 | 1199/1000 | - | 1e-06 | 9.608e-07 | `6920c520fdd0` | VERIFIED, 1.20 s |
| `pid-like` | reported | pidsmc | 3 | 7 | 21 | 1 | Delta = 1 | 1e-06 | 9.924e-07 | `b08dc00562e8` | VERIFIED, 0.41 s |
| `levant-order2-box-a` | reported | levant3d | 3 | 11 | 96 | 1 | L0 = 1 | 1e-04 | 1.000e-04 | `a093592552bc` | VERIFIED, 6.22 s |
| `levant-order2-box-b` | reported | levant3d | 3 | 11 | 96 | 1 | L0 = 1 | 1e-04 | 1.000e-04 | `e0bc41997c62` | VERIFIED, 5.82 s |
| `levant-order2-box-dense` | source | levant3d | 3 | 11 | 189 | 1 | L0 = 1 | 1e-04 | 1.000e-04 | `34d1779a6414` | VERIFIED, 9.25 s |
| `levant-order2-box-a-dense` | source | levant3d | 3 | 11 | 189 | 1 | L0 = 1 | 1e-04 | 1.000e-04 | `f3827dbfa59c` | VERIFIED, 5.31 s |
| `levant-order2-box-b-dense` | source | levant3d | 3 | 11 | 189 | 1 | L0 = 1 | 1e-04 | 1.000e-04 | `4dc56c1bb9e2` | VERIFIED, 5.47 s |
| `levant-order3-a-dense` | source | levant4d | 4 | 9 | 163 | 1/16 | p = 1/2 (L0 = 1/16) | 1e-04 | 2.700e-05 | `99d5bf889772` | VERIFIED, 1 min 55 s |
| `levant-order3-b-dense` | source | levant4d_2003print | 4 | 9 | 163 | 1/16 | p = 1/2 (L0 = 1/16) | 1e-04 | 1.000e-04 | `e8aaddc01273` | VERIFIED, 2 min 15 s |
| `continuous-twisting-frontier-dense` | source | continuous_twisting | 3 | 9 | 101 | 1199/1000 | - | 1e-06 | 9.392e-07 | `15419d39decb` | VERIFIED, 1.16 s |

### `levant-order1`: Super-twisting / Levant's differentiator of order one, standard tuning

- **Role:** reported. **In the paper:** Section 5, eq. (6); Table 2, row k = 1; Theorem 2 (k = 1); Figure 2.
- **System:** `sta`, super-twisting = Levant's differentiator of order one, error coordinates: `x1' = -k1 [x1]^(1/2) + x2,  x2' = -k2 [x1]^0 + delta`; n = 2 states, weights r = (2, 1), kappa = -1.
- **Function:** 5 terms (0 odd), homogeneity degree m = 6, dyadic coefficients with at most 7 significant bits; listing `listings/levant-order1.txt`.
- **Rational gains:** k1 = 3/2, k2 = 11/10.
- **delta0:** 1. **Scale:** certified at L = 1; for every L > 0 the function V(e/L) (Corollary 1 with L0 = 1).
- **Margin:** requested eps = 4722366482869645/4722366482869645213696 (~1e-06); certified eps_certified = 356811784131064741581864406086449449143/356811923176489970264571492362373784095686656 (~9.999996e-07).
- **Record:** `certificates/levant-order1/sta_exactgains_20261012T072209Z.printed_b07.record.json`, sha256 `9e0b9f12d69508ed0f1fdfac3a27c5296568af6d7d05aa8b22dfeb4c958e9399`, 4.7 kB; from `artifacts/campaign_2026-10/WORKED_STA/tierB/printed/sta_exactgains_20261012T072209Z.printed_b07.record.json`.
- **Gram file:** `certificates/levant-order1/sta_exactgains_grams_20261012T072209Z.json`, sha256 `c980f14cb65ab4e741126341ad66c03a6e57669b1ce8d946f282a6a8a91303ee`, 12.8 kB; from `artifacts/campaign_2026-10/WORKED_STA/tierB/printed/sta_exactgains_grams_20261012T072209Z.json`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order1/sta_exactgains_20261012T072209Z.printed_b07.record.json --grams certificates/levant-order1/sta_exactgains_grams_20261012T072209Z.json --json logs/levant-order1.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 356811784131064741581864406086449449143/356811923176489970264571492362373784095686656 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 0.08 s, peak RSS 21 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order1.log`.
- **Configuration:** `config/levant-order1.json` and CONFIGURATION.md.

### `levant-order2-a`: Levant's differentiator of order two, tuning a, 29-term function

- **Role:** reported. **In the paper:** Section 6.2, Order two; Table 2, row k = 2 tunings a and b; Theorem 2 (k = 2, tuning a); Appendix B, first listing; Figure 3.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 29 terms (0 odd), homogeneity degree m = 8, dyadic coefficients with at most 9 significant bits; listing `listings/levant-order2-a.txt`.
- **Rational gains:** lambda1 = 2, lambda3 = 11/10.
- **Bracket lambda2:** [2121/1000, 1061/500] = [2.121, 2.122], enclosing the positive root of lambda2^2 = 9/2 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 2 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1 (Corollary 1 with L0 = 1).
- **Margin:** requested eps = 944473296573929/9444732965739290427392 (~1e-07); certified eps_certified = 71341998424075849889387670337470552789547/713623846352979940529142984724747568191373312000 (~9.997143e-08).
- **Record:** `certificates/levant-order2-a/levant3d_lev2_b09_e7_point0_delta1.record.json.gz`, sha256 `6af6214320a7208f261172a5cbc918953c61c30460f608f23e5b99af1c2e586d`, 3.0 kB; from `artifacts/campaign_2026-10/SHORT_COEFFS/levant2_b09/point0/m8/portable/levant3d_lev2_b09_e7_point0_delta1.record.json.gz`.
- **Gram file:** `certificates/levant-order2-a/levant3d_lev2_b09_e7_point0_delta1.grams.json.gz`, sha256 `7dfb155c788f73bea33cb88581a56b0bdb6eba6850b5ff46a5cfd16fb4c1a0dd`, 365.5 kB; from `artifacts/campaign_2026-10/SHORT_COEFFS/levant2_b09/point0/m8/portable/levant3d_lev2_b09_e7_point0_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-a/levant3d_lev2_b09_e7_point0_delta1.record.json.gz --grams certificates/levant-order2-a/levant3d_lev2_b09_e7_point0_delta1.grams.json.gz --json logs/levant-order2-a.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 71341998424075849889387670337470552789547/713623846352979940529142984724747568191373312000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 1.02 s, peak RSS 28 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-a.log`.
- **Configuration:** `config/levant-order2-a.json` and CONFIGURATION.md.

### `levant-order2-b`: Levant's differentiator of order two, tuning b, 29-term function

- **Role:** reported. **In the paper:** Section 6.2, Order two; Table 2, row k = 2 tunings a and b; Theorem 2 (k = 2, tuning b); Appendix B, first listing; Figure 3.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 29 terms (0 odd), homogeneity degree m = 8, dyadic coefficients with at most 9 significant bits; listing `listings/levant-order2-b.txt`.
- **Rational gains:** lambda1 = 3, lambda3 = 11/10.
- **Bracket lambda2:** [1299/500, 2599/1000] = [2.598, 2.599], enclosing the positive root of lambda2^2 = 27/4 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 2 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1 (Corollary 1 with L0 = 1).
- **Margin:** requested eps = 944473296573929/9444732965739290427392 (~1e-07); certified eps_certified = 3345032005670454423053024982060415234679/33451117797795934712303577408972542258970624000 (~9.999762e-08).
- **Record:** `certificates/levant-order2-b/levant3d_lev2_b09_e7_point1_delta1.record.json.gz`, sha256 `4faebe05a7f6764bb031164223163d43da7db97f989ed01cf74b9895f65176b2`, 3.0 kB; from `artifacts/campaign_2026-10/SHORT_COEFFS/levant2_b09/point1/m8/portable/levant3d_lev2_b09_e7_point1_delta1.record.json.gz`.
- **Gram file:** `certificates/levant-order2-b/levant3d_lev2_b09_e7_point1_delta1.grams.json.gz`, sha256 `cfb06e78f08112d174990a0564987d173ab7599ce68ee8e2e865be6cd19869f8`, 365.4 kB; from `artifacts/campaign_2026-10/SHORT_COEFFS/levant2_b09/point1/m8/portable/levant3d_lev2_b09_e7_point1_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-b/levant3d_lev2_b09_e7_point1_delta1.record.json.gz --grams certificates/levant-order2-b/levant3d_lev2_b09_e7_point1_delta1.grams.json.gz --json logs/levant-order2-b.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 3345032005670454423053024982060415234679/33451117797795934712303577408972542258970624000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 1.11 s, peak RSS 28 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-b.log`.
- **Configuration:** `config/levant-order2-b.json` and CONFIGURATION.md.

### `levant-order2-box`: Levant's differentiator of order two, 96-term function on the whole box [2,3] x [2.121,2.599] x {1.1}

- **Role:** reported. **In the paper:** Section 6.2, Order two; Table 2, row k = 2 all tunings between; Theorem 2 (k = 2, every recursive tuning (1.1, 1.5, c), 2 <= c <= 3); Figure 3.
- **Derivation:** even part (Proposition 2) of the 189-term certificate levant-order2-box-dense: the 93 odd terms are dropped and the witnesses averaged over the orthants s and -s (scripts/even_part_certificate.py); the even-part record is itself verified, it does not rest on the proposition.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 96 terms (0 odd), homogeneity degree m = 11, dyadic coefficients with at most 18 significant bits; listing `listings/levant-order2-box.txt`.
- **Rational gains:** lambda3 = 11/10.
- **Bracket lambda1:** [2, 3] = [2, 3], enclosing the positive root of lambda1^1 = 5/2 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda2:** [2121/1000, 2599/1000] = [2.121, 2.599], enclosing the positive root of lambda2^1 = 59/25 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 4 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1 (Corollary 1 with L0 = 1).
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 194709999609081582874636760368487/1947111321950560360698936123457536000 (~9.999942e-05).
- **Record:** `certificates/levant-order2-box/levant3d_x9b_b18_delta1.printed_b18.even.record.json`, sha256 `0e0717dd5fbb71c69ff96cb299897168df61e369dbc37348674c913785e473f5`, 27.1 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order2-box/levant3d_x9b_b18_delta1.printed_b18.even.record.json`.
- **Gram file:** `certificates/levant-order2-box/levant3d_x9b_b18_delta1.printed_b18.even.grams.json.gz`, sha256 `9316aefd8976f66c3bba67d58e65e0c031243e1b653528965e87a716076f770c`, 3.3 MB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order2-box/levant3d_x9b_b18_delta1.printed_b18.even.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-box/levant3d_x9b_b18_delta1.printed_b18.even.record.json --grams certificates/levant-order2-box/levant3d_x9b_b18_delta1.printed_b18.even.grams.json.gz --json logs/levant-order2-box.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 194709999609081582874636760368487/1947111321950560360698936123457536000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 9.98 s, peak RSS 73 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-box.log`.
- **Configuration:** `config/levant-order2-box.json` and CONFIGURATION.md.

### `levant-order3-a`: Levant's differentiator of order three, tuning a, 83-term function

- **Role:** reported. **In the paper:** Section 6.2, Order three; Table 2, row k = 3 tuning a; Theorem 2 (k = 3, tuning a, L0 = 1/16).
- **Derivation:** even part (Proposition 2) of the 163-term certificate levant-order3-a-dense (80 odd terms dropped, witnesses averaged over s and -s); the even-part record is itself verified.
- **System:** `levant4d`, Levant's differentiator of order three, error inclusion: `e_i' = -l_i [e1]^((4-i)/4) + e_(i+1) (i < 4),  e4' = -l4 [e1]^0 + delta`; n = 4 states, weights r = (4, 3, 2, 1), kappa = -1.
- **Function:** 83 terms (0 odd), homogeneity degree m = 9, dyadic coefficients with at most 30 significant bits; listing `listings/levant-order3-a.txt`.
- **Rational gains:** lambda1 = 3/2, lambda4 = 11/160.
- **Bracket lambda2:** [16250654867593/15625000000000, 4160167646103809/4000000000000000] = [1.040041911525952, 1.04004191152595225], enclosing the positive root of lambda2^3 = 9/8 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda3:** [1529736676991629/4000000000000000, 3059473353983259/8000000000000000] = [0.38243416924790725, 0.382434169247907375], enclosing the positive root of lambda3^6 = 6561/2097152 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 4 (all {lo, hi} combinations).
- **delta0:** 1/16. **Scale:** the record's gains are lambda_i p^i and delta0 = p^4 with p = 1/2, i.e. lambda_i L0^(i/4) and delta0 = L0 with L0 = p^4 = 1/16; Corollary 1 transfers the conclusions to every L > 0.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 12157827698762243865517941/450359962737049600000000000000 (~2.699580e-05).
- **Record:** `certificates/levant-order3-a/levant4d_p1_2_m9_delta1_16.printed_b30.even.record.json`, sha256 `4636bf676eba661dcd9699abe5eebdc5b99bc57384b3b54629ad13da950cdeee`, 30.7 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order3-a/levant4d_p1_2_m9_delta1_16.printed_b30.even.record.json`.
- **Gram file:** `certificates/levant-order3-a/levant4d_p1_2_m9_delta1_16.printed_b30.even.grams.json.gz`, sha256 `a4f85e65ab474a6bb0b777f0db2ca20b2ae91a0c55bddd69a8ab1d8a7709ba14`, 7.0 MB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order3-a/levant4d_p1_2_m9_delta1_16.printed_b30.even.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order3-a/levant4d_p1_2_m9_delta1_16.printed_b30.even.record.json --grams certificates/levant-order3-a/levant4d_p1_2_m9_delta1_16.printed_b30.even.grams.json.gz --json logs/levant-order3-a.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 12157827698762243865517941/450359962737049600000000000000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 1 min 59 s, peak RSS 436 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order3-a.log`.
- **Configuration:** `config/levant-order3-a.json` and CONFIGURATION.md.

### `levant-order3-b`: Levant's differentiator of order three, tuning b, 83-term function

- **Role:** reported. **In the paper:** Section 6.2, Order three; Table 2, row k = 3 tuning b; Theorem 2 (k = 3, tuning b, L0 = 1/16).
- **Derivation:** even part (Proposition 2) of the 163-term certificate levant-order3-b-dense (80 odd terms dropped, witnesses averaged over s and -s); the even-part record is itself verified.
- **System:** `levant4d_2003print`, Levant's differentiator of order three, error inclusion (same equations as levant4d; the system key names the gain point): `e_i' = -l_i [e1]^((4-i)/4) + e_(i+1) (i < 4),  e4' = -l4 [e1]^0 + delta`; n = 4 states, weights r = (4, 3, 2, 1), kappa = -1.
- **Function:** 83 terms (0 odd), homogeneity degree m = 9, dyadic coefficients with at most 15 significant bits; listing `listings/levant-order3-b.txt`.
- **Rational gains:** lambda1 = 5/2, lambda4 = 11/160.
- **Bracket lambda2:** [4386026607319299/2000000000000000, 8772053214638599/4000000000000000] = [2.1930133036596495, 2.19301330365964975], enclosing the positive root of lambda2^3 = 675/64 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda3:** [1110661957261773/2000000000000000, 4442647829047093/8000000000000000] = [0.5553309786308865, 0.555330978630886625], enclosing the positive root of lambda3^6 = 492075/16777216 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 4 (all {lo, hi} combinations).
- **delta0:** 1/16. **Scale:** the record's gains are lambda_i p^i and delta0 = p^4 with p = 1/2, i.e. lambda_i L0^(i/4) and delta0 = L0 with L0 = p^4 = 1/16; Corollary 1 transfers the conclusions to every L > 0.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 11883945951582837340004524119364845695329/118842243771396506390315925504000000000000000 (~9.999766e-05).
- **Record:** `certificates/levant-order3-b/levant4d_2003print_x9b_b15_delta1_16.printed_b15.even.record.json`, sha256 `1d0d37819d73045fa9fe546e0552515cd1150448f66ce373cd7ccba6bd39824a`, 27.4 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order3-b/levant4d_2003print_x9b_b15_delta1_16.printed_b15.even.record.json`.
- **Gram file:** `certificates/levant-order3-b/levant4d_2003print_x9b_b15_delta1_16.printed_b15.even.grams.json.gz`, sha256 `db7eea134d6b0373662d06fb77f1a875cd3c2ab18d98779c6cc8d7eed58e080c`, 46.9 MB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order3-b/levant4d_2003print_x9b_b15_delta1_16.printed_b15.even.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order3-b/levant4d_2003print_x9b_b15_delta1_16.printed_b15.even.record.json --grams certificates/levant-order3-b/levant4d_2003print_x9b_b15_delta1_16.printed_b15.even.grams.json.gz --json logs/levant-order3-b.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 11883945951582837340004524119364845695329/118842243771396506390315925504000000000000000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 2 min 29 s, peak RSS 703 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order3-b.log`.
- **Configuration:** `config/levant-order3-b.json` and CONFIGURATION.md.

### `levant-order4`: Levant's differentiator of order four, tuning a, nominal, 507-term function

- **Role:** reported. **In the paper:** Section 6.2, Order four; Table 2, row k = 4 tuning a; Section 8, Limitations (about ten hours).
- **System:** `levant5d`, Levant's differentiator of order four, error dynamics, nominal: `e_i' = -l_i [e1]^((5-i)/5) + e_(i+1) (i < 5),  e5' = -l5 [e1]^0`; n = 5 states, weights r = (5, 4, 3, 2, 1), kappa = -1.
- **Function:** 507 terms (251 odd), homogeneity degree m = 11, dyadic coefficients with at most 26 significant bits; listing `listings/levant-order4.txt`.
- **Rational gains:** lambda1 = 5, lambda5 = 11/10.
- **Bracket lambda2:** [1003110457464633/100000000000000, 10031104574646331/1000000000000000] = [10.03110457464633, 10.031104574646331], enclosing the positive root of lambda2^4 = 10125 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda3:** [4651208827241701/500000000000000, 2325604413620851/250000000000000] = [9.302417654483402, 9.302417654483404], enclosing the positive root of lambda3^6 = 648000 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda4:** [914995950211533/200000000000000, 2287489875528833/500000000000000] = [4.574979751057665, 4.574979751057666], enclosing the positive root of lambda4^12 = 5380840125/64 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 8 (all {lo, hi} combinations).
- **delta0:** 0 (nominal). **Scale:** nominal (delta0 = 0), certified at L = 1.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 65503281876540153956895377/9007199254740992000000000000000 (~7.272325e-06) (as derived by the workstation run).
- **Record:** `certificates/levant-order4/levant5d_m11_nominal.printed_b26.record.json`, sha256 `3386556ae24bea092195d94ae7fd65f40165b01489d7daff96f116e6007977c6`, 124.8 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierA/c9.4_e5_levant5d_m11/levant5d_m11_nominal.printed_b26.record.json`.
- **Gram file:** `certificates/levant-order4/levant5d_m11_nominal.grams.json.gz`, sha256 `15fa28be0cec34636d1e0be2dd37477f495a58707dab13400ea7f01b916306bd`, 73.8 MB; over 50 MiB, in the repository (README, 'Large files'); from `artifacts/campaign_2026-08/E5_levant5d/exact_2026-09-04/portable_2026-09-05/levant5d_m11_nominal.grams.json.gz`.
- **Note:** records.json names tierA/c9.4_e5_levant5d_m11/levant5d_m11_nominal.grams.json.gz, which is not in the checkout; this byte-identical copy (same sha256) is used.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order4/levant5d_m11_nominal.printed_b26.record.json --grams certificates/levant-order4/levant5d_m11_nominal.grams.json.gz --json logs/levant-order4.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 65503281876540153956895377/9007199254740992000000000000000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Not re-run here** (marked `long_run`: about 10 h single-threaded, about 9 GB RAM; `tools/verify_all.py --include-long` runs it). Recorded run: VERIFIED, exit 0, wall 10:16:31, peak RSS 8471 MiB, spark-91bc (Linux aarch64, 20 cores, 121 GiB RAM, CPython 3.12.3 /usr/bin/python3; CPU model not recorded), 2026-09-25/26, verifier sha256 a90047176958 (the arithmetic is that of the current verifier; only messages and exit codes changed since); log `logs/workstation/levant-order4.insurance_P.log`.
- **Configuration:** `config/levant-order4.json` and CONFIGURATION.md.

### `twisting`: Twisting at (c1, c2) = (2, 3/2), delta0 = 2047/4096

- **Role:** reported. **In the paper:** Section 7.1, eq. (13); Proposition 4 (Twisting below the ceiling); Table 2, row Twisting.
- **System:** `twisting`, twisting algorithm (paper eq. (12)): `x1' = x2,  x2' = -c1 [x1]^0 - c2 [x2]^0 + delta   (record gains r1 = c1, r2 = c2)`; n = 2 states, weights r = (2, 1), kappa = -1.
- **Function:** 5 terms (0 odd), homogeneity degree m = 5, dyadic coefficients with at most 4 significant bits; listing `listings/twisting.txt`.
- **Rational gains:** r1 = 2, r2 = 3/2.
- **delta0:** 2047/4096. **Scale:** at the gains themselves; Proposition 1 gives (c1, c2) = q (2, 3/2) for bounds below q/2.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 259614836037198660742765353747/2596148429267413814265248164610048 (~1.000000e-04).
- **Record:** `certificates/twisting/twisting_x9b_b04_delta2047_4096.printed_b04.record.json`, sha256 `be4a8ec13591cee6c90cb9ac6f1b11121e7d7a1897b813603ec807e194d8c464`, 8.0 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c7.2_x4_rl1_2047_4096/printed/twisting_x9b_b04_delta2047_4096.printed_b04.record.json`.
- **Gram file:** `certificates/twisting/twisting_x9b_b04_delta2047_4096.grams.json.gz`, sha256 `75fe6a97944fe5fc85f84a158ba6a058faa797a6996b2ebce5b03503da4ce745`, 1.7 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c7.2_x4_rl1_2047_4096/printed/twisting_x9b_b04_delta2047_4096.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/twisting/twisting_x9b_b04_delta2047_4096.printed_b04.record.json --grams certificates/twisting/twisting_x9b_b04_delta2047_4096.grams.json.gz --json logs/twisting.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 259614836037198660742765353747/2596148429267413814265248164610048 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 0.08 s, peak RSS 21 MiB; log `logs/verify_all_cloud_2026-11-05/twisting.log`.
- **Configuration:** `config/twisting.json` and CONFIGURATION.md.

### `continuous-twisting`: Continuous twisting at G1 = (25, 15, 2.3, 1.1), delta0 = 1, 9-term function

- **Role:** reported. **In the paper:** Section 7.2; Table 2, row Continuous twisting, delta0 = 1; Appendix B, second listing.
- **System:** `continuous_twisting`, continuous twisting algorithm (paper eq. (14)): `x1' = x2,  x2' = -k1 [x1]^(1/3) - k2 [x2]^(1/2) + x3,  x3' = -k3 [x1]^0 - k4 [x2]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 9 terms (0 odd), homogeneity degree m = 5, dyadic coefficients with at most 5 significant bits; listing `listings/continuous-twisting.txt`.
- **Rational gains:** k1 = 25, k2 = 15, k3 = 23/10, k4 = 11/10.
- **delta0:** 1. **Scale:** at the published gains and bound.
- **Margin:** requested eps = 4835703278458517/4835703278458516698824704 (~1e-09); certified eps_certified = 17534966849530619856389307593822947314349/17538019647970835018444217992595396235871190515712 (~9.998259e-10).
- **Record:** `certificates/continuous-twisting/continuous_twisting_cta_b05_e9_delta1.record.json.gz`, sha256 `b847924c4720898e166a8ba19a64325f61614d812d14167b1b26865587818c5d`, 2.7 kB; from `artifacts/campaign_2026-10/SHORT_COEFFS/cta_g1_b05/run/m5/portable/continuous_twisting_cta_b05_e9_delta1.record.json.gz`.
- **Gram file:** `certificates/continuous-twisting/continuous_twisting_cta_b05_e9_delta1.grams.json.gz`, sha256 `897a79b399a17cc900217ea60445f08d9f96e4574a6629b019bae3e233c78d43`, 39.0 kB; from `artifacts/campaign_2026-10/SHORT_COEFFS/cta_g1_b05/run/m5/portable/continuous_twisting_cta_b05_e9_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/continuous-twisting/continuous_twisting_cta_b05_e9_delta1.record.json.gz --grams certificates/continuous-twisting/continuous_twisting_cta_b05_e9_delta1.grams.json.gz --json logs/continuous-twisting.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 17534966849530619856389307593822947314349/17538019647970835018444217992595396235871190515712 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 0.16 s, peak RSS 22 MiB; log `logs/verify_all_cloud_2026-11-05/continuous-twisting.log`.
- **Configuration:** `config/continuous-twisting.json` and CONFIGURATION.md.

### `continuous-twisting-frontier`: Continuous twisting at G1, delta0 = 1199/1000, 52-term function

- **Role:** reported. **In the paper:** Section 7.2; Table 2, row Continuous twisting, delta0 = 1.199; Conclusion (1199/1200 of the ceiling).
- **Derivation:** even part (Proposition 2) of the 101-term certificate continuous-twisting-frontier-dense (49 odd terms dropped, witnesses averaged over s and -s); the even-part record is itself verified.
- **System:** `continuous_twisting`, continuous twisting algorithm (paper eq. (14)): `x1' = x2,  x2' = -k1 [x1]^(1/3) - k2 [x2]^(1/2) + x3,  x3' = -k3 [x1]^0 - k4 [x2]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 52 terms (0 odd), homogeneity degree m = 9, dyadic coefficients with at most 10 significant bits; listing `listings/continuous-twisting-frontier.txt`.
- **Rational gains:** k1 = 25, k2 = 15, k3 = 23/10, k4 = 11/10.
- **delta0:** 1199/1000. **Scale:** at the published gains.
- **Margin:** requested eps = 4722366482869645/4722366482869645213696 (~1e-06); certified eps_certified = 31386725221269236366991072746355293563/32667107224410092492483962313449748299776000 (~9.608052e-07).
- **Record:** `certificates/continuous-twisting-frontier/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.even.record.json`, sha256 `6920c520fdd0437beeaa0c71b7aace4c679e79ac6ee0c1db1310b20da5655c6e`, 18.5 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/continuous-twisting-frontier/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.even.record.json`.
- **Gram file:** `certificates/continuous-twisting-frontier/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.even.grams.json.gz`, sha256 `91b68046430a0f7e2a601bf6e00b31b2653362ee400ab5c9d64b14d107728f0f`, 522.1 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/continuous-twisting-frontier/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.even.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/continuous-twisting-frontier/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.even.record.json --grams certificates/continuous-twisting-frontier/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.even.grams.json.gz --json logs/continuous-twisting-frontier.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 31386725221269236366991072746355293563/32667107224410092492483962313449748299776000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 1.20 s, peak RSS 30 MiB; log `logs/verify_all_cloud_2026-11-05/continuous-twisting-frontier.log`.
- **Configuration:** `config/continuous-twisting-frontier.json` and CONFIGURATION.md.

### `pid-like`: PID-like controller at D1 = (2.7, 5.345, 1.1), Delta = delta0 = 1, 21 terms

- **Role:** reported. **In the paper:** Section 7.3; Table 2, row PID-like; Appendix B, third listing.
- **System:** `pidsmc`, PID-like controller = eq. (14) with k4 = 0: `x1' = x2,  x2' = -l1 [x1]^(1/3) - l2 [x2]^(1/2) + x3,  x3' = -l3 [x1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 21 terms (0 odd), homogeneity degree m = 7, dyadic coefficients with at most 14 significant bits; listing `listings/pid-like.txt`.
- **Rational gains:** delta = 1, lambda1 = 27/10, lambda2 = 1069/200, lambda3 = 11/10.
- **delta0:** 1. **Scale:** at Delta = 1; Proposition 1 gives V(x/Delta) for every Delta > 0.
- **Margin:** requested eps = 4722366482869645/4722366482869645213696 (~1e-06); certified eps_certified = 16102533130475393163343698871/16225927682921336339157801028812800 (~9.923952e-07).
- **Record:** `certificates/pid-like/pidsmc_x9b_b14_delta1.printed_b14.record.json`, sha256 `b08dc00562e8d0af319ee503c591a4c595f13f9734126219f22cca3a9f40cd31`, 11.3 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c8.2_x5_rl1_delta1_m7_xfix/printed/pidsmc_x9b_b14_delta1.printed_b14.record.json`.
- **Gram file:** `certificates/pid-like/pidsmc_x9b_b14_delta1.grams.json.gz`, sha256 `d4fac96113104e11817abb902610650d8d637dfc2426d0ecba14e2d58f645b67`, 138.7 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c8.2_x5_rl1_delta1_m7_xfix/printed/pidsmc_x9b_b14_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/pid-like/pidsmc_x9b_b14_delta1.printed_b14.record.json --grams certificates/pid-like/pidsmc_x9b_b14_delta1.grams.json.gz --json logs/pid-like.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 16102533130475393163343698871/16225927682921336339157801028812800 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 0.41 s, peak RSS 24 MiB; log `logs/verify_all_cloud_2026-11-05/pid-like.log`.
- **Configuration:** `config/pid-like.json` and CONFIGURATION.md.

### `levant-order2-box-a`: Levant's differentiator of order two, 96-term function on the box function at lambda1 = 2, lambda2 in [2.121, 2.599] (contains tuning a)

- **Role:** reported. **In the paper:** Section 6.2, Order two (same 96-term function as levant-order2-box, restricted to one edge of the box; supplementary).
- **Derivation:** even part (Proposition 2) of the 189-term certificate levant-order2-box-a-dense: the 93 odd terms are dropped and the witnesses averaged over the orthants s and -s (scripts/even_part_certificate.py); the even-part record is itself verified, it does not rest on the proposition.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 96 terms (0 odd), homogeneity degree m = 11, dyadic coefficients with at most 18 significant bits; listing `listings/levant-order2-box-a.txt`.
- **Rational gains:** lambda1 = 2, lambda3 = 11/10.
- **Bracket lambda2:** [2121/1000, 2599/1000] = [2.121, 2.599], enclosing the positive root of lambda2^2 = 9/2 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 2 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1 (Corollary 1 with L0 = 1).
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 194709999609081582874636760368487/1947111321950560360698936123457536000 (~9.999942e-05).
- **Record:** `certificates/levant-order2-box-a/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.even.record.json`, sha256 `a093592552bcd0cc8b7690b189dd12b20731a9e9b1ef5511fd03248a0503f4ee`, 26.4 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order2-box-a/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.even.record.json`.
- **Gram file:** `certificates/levant-order2-box-a/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.even.grams.json.gz`, sha256 `4d0a68530948661c22f94fd6d4624cd2704a98ab474d77b8a0179f0100278bcb`, 1.9 MB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order2-box-a/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.even.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-box-a/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.even.record.json --grams certificates/levant-order2-box-a/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.even.grams.json.gz --json logs/levant-order2-box-a.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 194709999609081582874636760368487/1947111321950560360698936123457536000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 6.22 s, peak RSS 51 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-box-a.log`.
- **Configuration:** `config/levant-order2-box-a.json` and CONFIGURATION.md.

### `levant-order2-box-b`: Levant's differentiator of order two, 96-term function on the box function at lambda1 = 3, lambda2 in [2.121, 2.599] (contains tuning b)

- **Role:** reported. **In the paper:** Section 6.2, Order two (same 96-term function as levant-order2-box, restricted to one edge of the box; supplementary).
- **Derivation:** even part (Proposition 2) of the 189-term certificate levant-order2-box-b-dense: the 93 odd terms are dropped and the witnesses averaged over the orthants s and -s (scripts/even_part_certificate.py); the even-part record is itself verified, it does not rest on the proposition.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 96 terms (0 odd), homogeneity degree m = 11, dyadic coefficients with at most 18 significant bits; listing `listings/levant-order2-box-b.txt`.
- **Rational gains:** lambda1 = 3, lambda3 = 11/10.
- **Bracket lambda2:** [2121/1000, 2599/1000] = [2.121, 2.599], enclosing the positive root of lambda2^2 = 27/4 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 2 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1 (Corollary 1 with L0 = 1).
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 194710188163337560604196390911417/1947111321950560360698936123457536000 (~9.999952e-05).
- **Record:** `certificates/levant-order2-box-b/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.even.record.json`, sha256 `e0bc41997c6222a21d36d6381ced024965d40d65e83719a7304a53b0e418442b`, 26.4 kB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order2-box-b/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.even.record.json`.
- **Gram file:** `certificates/levant-order2-box-b/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.even.grams.json.gz`, sha256 `1f4c3bedd22fa0914ccb5032fd26d01fde311c1785592ee77912c58cb81b39a7`, 1.9 MB; from `artifacts/campaign_2026-11/EVEN_PARTS/levant-order2-box-b/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.even.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-box-b/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.even.record.json --grams certificates/levant-order2-box-b/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.even.grams.json.gz --json logs/levant-order2-box-b.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 194710188163337560604196390911417/1947111321950560360698936123457536000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 5.82 s, peak RSS 51 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-box-b.log`.
- **Configuration:** `config/levant-order2-box-b.json` and CONFIGURATION.md.

### `levant-order2-box-dense`: 189-term source certificate of levant-order2-box

- **Role:** source. **In the paper:** source of the even part levant-order2-box (Proposition 2); not printed in the paper.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 189 terms (93 odd), homogeneity degree m = 11, dyadic coefficients with at most 18 significant bits; listing `listings/levant-order2-box-dense.txt`.
- **Rational gains:** lambda3 = 11/10.
- **Bracket lambda1:** [2, 3] = [2, 3], enclosing the positive root of lambda1^1 = 5/2 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda2:** [2121/1000, 2599/1000] = [2.121, 2.599], enclosing the positive root of lambda2^1 = 59/25 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 4 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 8112896895460404282478337327819/81129638414606681695789005144064000 (~9.999917e-05).
- **Record:** `certificates/levant-order2-box-dense/levant3d_x9b_b18_delta1.printed_b18.record.json`, sha256 `34d1779a64142ee43311dfb51748d4438908fa1600ce1b6bccb53fa1638e8cc4`, 41.6 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c9.2_x2_box_m11/printed/levant3d_x9b_b18_delta1.printed_b18.record.json`.
- **Gram file:** `certificates/levant-order2-box-dense/levant3d_x9b_b18_delta1.grams.json.gz`, sha256 `f66216b6bfe4ca5fa72661f54ba26d2ece16e4fbc7fa9ef56acfcc9f9d1f581e`, 2.7 MB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c9.2_x2_box_m11/printed/levant3d_x9b_b18_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-box-dense/levant3d_x9b_b18_delta1.printed_b18.record.json --grams certificates/levant-order2-box-dense/levant3d_x9b_b18_delta1.grams.json.gz --json logs/levant-order2-box-dense.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 8112896895460404282478337327819/81129638414606681695789005144064000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 9.25 s, peak RSS 60 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-box-dense.log`.
- **Configuration:** `config/levant-order2-box-dense.json` and CONFIGURATION.md.

### `levant-order2-box-a-dense`: 189-term source certificate of levant-order2-box-a

- **Role:** source. **In the paper:** source of the even part levant-order2-box-a (Proposition 2); not printed in the paper.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 189 terms (93 odd), homogeneity degree m = 11, dyadic coefficients with at most 18 significant bits; listing `listings/levant-order2-box-a-dense.txt`.
- **Rational gains:** lambda1 = 2, lambda3 = 11/10.
- **Bracket lambda2:** [2121/1000, 2599/1000] = [2.121, 2.599], enclosing the positive root of lambda2^2 = 9/2 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 2 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 778839374969096049020686142230861/7788445287802241442795744493830144000 (~9.999934e-05).
- **Record:** `certificates/levant-order2-box-a-dense/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.record.json`, sha256 `f3827dbfa59c5abfa81ef6821eceff9701d94086fc280383e5559ce1562ca6a7`, 40.8 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c9.2_x2_box_m11/printed/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.record.json`.
- **Gram file:** `certificates/levant-order2-box-a-dense/levant3d_x9b_b18_sub_lambda1_2_delta1.grams.json.gz`, sha256 `c3982b438d7ad26dd2d9f0b9fedd028ac55e98cdb4b9945cb567941f81dd577c`, 1.6 MB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c9.2_x2_box_m11/printed/levant3d_x9b_b18_sub_lambda1_2_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-box-a-dense/levant3d_x9b_b18_sub_lambda1_2_delta1.printed_b18.record.json --grams certificates/levant-order2-box-a-dense/levant3d_x9b_b18_sub_lambda1_2_delta1.grams.json.gz --json logs/levant-order2-box-a-dense.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 778839374969096049020686142230861/7788445287802241442795744493830144000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 5.31 s, peak RSS 43 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-box-a-dense.log`.
- **Configuration:** `config/levant-order2-box-a-dense.json` and CONFIGURATION.md.

### `levant-order2-box-b-dense`: 189-term source certificate of levant-order2-box-b

- **Role:** source. **In the paper:** source of the even part levant-order2-box-b (Proposition 2); not printed in the paper.
- **System:** `levant3d`, Levant's differentiator of order two, error inclusion (paper eq. (10)): `e1' = -l1 [e1]^(2/3) + e2,  e2' = -l2 [e1]^(1/3) + e3,  e3' = -l3 [e1]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 189 terms (93 odd), homogeneity degree m = 11, dyadic coefficients with at most 18 significant bits; listing `listings/levant-order2-box-b-dense.txt`.
- **Rational gains:** lambda1 = 3, lambda3 = 11/10.
- **Bracket lambda2:** [2121/1000, 2599/1000] = [2.121, 2.599], enclosing the positive root of lambda2^2 = 27/4 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 2 (all {lo, hi} combinations).
- **delta0:** 1. **Scale:** certified at L = 1.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 8112896895460404282478337327819/81129638414606681695789005144064000 (~9.999917e-05).
- **Record:** `certificates/levant-order2-box-b-dense/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.record.json`, sha256 `4dc56c1bb9e25516620818adfb26df1bb8c793d8e4fee24903eb2aac9e22b3be`, 40.8 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c9.2_x2_box_m11/printed/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.record.json`.
- **Gram file:** `certificates/levant-order2-box-b-dense/levant3d_x9b_b18_sub_lambda1_3_delta1.grams.json.gz`, sha256 `30f031b54db7bee53774103e1056746434acbdd5ac15050aef7973f793b2ae82`, 1.6 MB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierB/c9.2_x2_box_m11/printed/levant3d_x9b_b18_sub_lambda1_3_delta1.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order2-box-b-dense/levant3d_x9b_b18_sub_lambda1_3_delta1.printed_b18.record.json --grams certificates/levant-order2-box-b-dense/levant3d_x9b_b18_sub_lambda1_3_delta1.grams.json.gz --json logs/levant-order2-box-b-dense.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 8112896895460404282478337327819/81129638414606681695789005144064000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 5.47 s, peak RSS 43 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order2-box-b-dense.log`.
- **Configuration:** `config/levant-order2-box-b-dense.json` and CONFIGURATION.md.

### `levant-order3-a-dense`: 163-term source certificate of levant-order3-a

- **Role:** source. **In the paper:** source of the even part levant-order3-a (Proposition 2); not printed.
- **System:** `levant4d`, Levant's differentiator of order three, error inclusion: `e_i' = -l_i [e1]^((4-i)/4) + e_(i+1) (i < 4),  e4' = -l4 [e1]^0 + delta`; n = 4 states, weights r = (4, 3, 2, 1), kappa = -1.
- **Function:** 163 terms (80 odd), homogeneity degree m = 9, dyadic coefficients with at most 30 significant bits; listing `listings/levant-order3-a-dense.txt`.
- **Rational gains:** lambda1 = 3/2, lambda4 = 11/160.
- **Bracket lambda2:** [16250654867593/15625000000000, 4160167646103809/4000000000000000] = [1.040041911525952, 1.04004191152595225], enclosing the positive root of lambda2^3 = 9/8 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda3:** [1529736676991629/4000000000000000, 3059473353983259/8000000000000000] = [0.38243416924790725, 0.382434169247907375], enclosing the positive root of lambda3^6 = 6561/2097152 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 4 (all {lo, hi} combinations).
- **delta0:** 1/16. **Scale:** the record's gains are lambda_i p^i and delta0 = p^4 with p = 1/2, i.e. lambda_i L0^(i/4) and delta0 = L0 with L0 = p^4 = 1/16; Corollary 1 transfers the conclusions to every L > 0.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 36473225356696126285318031/1351079888211148800000000000000 (~2.699561e-05).
- **Record:** `certificates/levant-order3-a-dense/levant4d_p1_2_m9_delta1_16.printed_b30.record.json`, sha256 `99d5bf8897726c2043acbfdaf9e73f1385a8583d779af568dbf9e2dfc26660e1`, 45.0 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierA/c9.3_x1_levant4d_p1_2/levant4d_p1_2_m9_delta1_16.printed_b30.record.json`.
- **Gram file:** `certificates/levant-order3-a-dense/levant4d_p1_2_m9_delta1_16.grams.json.gz`, sha256 `323abcf345daa952498e1d867e7e6b8e01397465477c5037310f6a9917715f9c`, 6.2 MB; from `artifacts/campaign_2026-09/X9/printed_2026-09-25/tierA/c9.3_x1_levant4d_p1_2/levant4d_p1_2_m9_delta1_16.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order3-a-dense/levant4d_p1_2_m9_delta1_16.printed_b30.record.json --grams certificates/levant-order3-a-dense/levant4d_p1_2_m9_delta1_16.grams.json.gz --json logs/levant-order3-a-dense.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 36473225356696126285318031/1351079888211148800000000000000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 1 min 55 s, peak RSS 424 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order3-a-dense.log`.
- **Configuration:** `config/levant-order3-a-dense.json` and CONFIGURATION.md.

### `levant-order3-b-dense`: 163-term source certificate of levant-order3-b

- **Role:** source. **In the paper:** source of the even part levant-order3-b (Proposition 2); not printed.
- **System:** `levant4d_2003print`, Levant's differentiator of order three, error inclusion (same equations as levant4d; the system key names the gain point): `e_i' = -l_i [e1]^((4-i)/4) + e_(i+1) (i < 4),  e4' = -l4 [e1]^0 + delta`; n = 4 states, weights r = (4, 3, 2, 1), kappa = -1.
- **Function:** 163 terms (80 odd), homogeneity degree m = 9, dyadic coefficients with at most 15 significant bits; listing `listings/levant-order3-b-dense.txt`.
- **Rational gains:** lambda1 = 5/2, lambda4 = 11/160.
- **Bracket lambda2:** [4386026607319299/2000000000000000, 8772053214638599/4000000000000000] = [2.1930133036596495, 2.19301330365964975], enclosing the positive root of lambda2^3 = 675/64 (lo^k < q < hi^k checked in Q by the verifier).
- **Bracket lambda3:** [1110661957261773/2000000000000000, 4442647829047093/8000000000000000] = [0.5553309786308865, 0.555330978630886625], enclosing the positive root of lambda3^6 = 492075/16777216 (lo^k < q < hi^k checked in Q by the verifier).
- **Corners:** 4 (all {lo, hi} combinations).
- **delta0:** 1/16. **Scale:** the record's gains are lambda_i p^i and delta0 = p^4 with p = 1/2, i.e. lambda_i L0^(i/4) and delta0 = L0 with L0 = p^4 = 1/16; Corollary 1 transfers the conclusions to every L > 0.
- **Margin:** requested eps = 7378697629483821/73786976294838206464 (~0.0001); certified eps_certified = 95069381376918928115833401705123313804019/950737950171172051122527404032000000000000000 (~9.999536e-05).
- **Record:** `certificates/levant-order3-b-dense/levant4d_2003print_x9b_b15_delta1_16.printed_b15.record.json`, sha256 `e8aaddc0127363bf69286ce045f7a7a141eeb2518bd8d8e717705876ff089d6e`, 41.1 kB; from `artifacts/campaign_2026-09/X9/printed_2026-09-26/tierB/c9.3_x10_levant4d_2003print_p1_2/printed/levant4d_2003print_x9b_b15_delta1_16.printed_b15.record.json`.
- **Gram file:** `certificates/levant-order3-b-dense/levant4d_2003print_x9b_b15_delta1_16.grams.json.gz`, sha256 `4a4d78c8aedbf65cd51821cf1d2dde9f58619bad452c5d9189d9a62d61b5cbe6`, 40.2 MB; from `artifacts/campaign_2026-09/X9/printed_2026-09-26/tierB/c9.3_x10_levant4d_2003print_p1_2/printed/levant4d_2003print_x9b_b15_delta1_16.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/levant-order3-b-dense/levant4d_2003print_x9b_b15_delta1_16.printed_b15.record.json --grams certificates/levant-order3-b-dense/levant4d_2003print_x9b_b15_delta1_16.grams.json.gz --json logs/levant-order3-b-dense.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 95069381376918928115833401705123313804019/950737950171172051122527404032000000000000000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 2 min 15 s, peak RSS 519 MiB; log `logs/verify_all_cloud_2026-11-05/levant-order3-b-dense.log`.
- **Configuration:** `config/levant-order3-b-dense.json` and CONFIGURATION.md.

### `continuous-twisting-frontier-dense`: 101-term source certificate of continuous-twisting-frontier

- **Role:** source. **In the paper:** source of the even part continuous-twisting-frontier (Proposition 2).
- **System:** `continuous_twisting`, continuous twisting algorithm (paper eq. (14)): `x1' = x2,  x2' = -k1 [x1]^(1/3) - k2 [x2]^(1/2) + x3,  x3' = -k3 [x1]^0 - k4 [x2]^0 + delta`; n = 3 states, weights r = (3, 2, 1), kappa = -1.
- **Function:** 101 terms (49 odd), homogeneity degree m = 9, dyadic coefficients with at most 10 significant bits; listing `listings/continuous-twisting-frontier-dense.txt`.
- **Rational gains:** k1 = 25, k2 = 15, k3 = 23/10, k4 = 11/10.
- **delta0:** 1199/1000. **Scale:** at the published gains.
- **Margin:** requested eps = 4722366482869645/4722366482869645213696 (~1e-06); certified eps_certified = 5113675704153982176939661276083292661/5444517870735015415413993718908291383296000 (~9.392339e-07).
- **Record:** `certificates/continuous-twisting-frontier-dense/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.record.json`, sha256 `15419d39decb6e5380827396da7a2ba6a81652912af80dd67721d24f8e8fa54b`, 24.9 kB; from `artifacts/campaign_2026-09/CTA_TG2017/printed/tierB/c8.1_tg_g1_delta1199_1000_m9/printed/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.record.json`.
- **Gram file:** `certificates/continuous-twisting-frontier-dense/continuous_twisting_tgb_b10_delta1199_1000.grams.json.gz`, sha256 `3b32b1b0fb3fea5e6e70ca4243dc51d401f811a019c77ef7f36402592874067d`, 419.6 kB; from `artifacts/campaign_2026-09/CTA_TG2017/printed/tierB/c8.1_tg_g1_delta1199_1000_m9/printed/continuous_twisting_tgb_b10_delta1199_1000.grams.json.gz`.
- **Verify:** `python3 verification/standalone/verify_certificate.py certificates/continuous-twisting-frontier-dense/continuous_twisting_tgb_b10_delta1199_1000.printed_b10.record.json --grams certificates/continuous-twisting-frontier-dense/continuous_twisting_tgb_b10_delta1199_1000.grams.json.gz --json logs/continuous-twisting-frontier-dense.verification.json`
- **Expected output:** a line `eps_certified = eps - charge = 5113675704153982176939661276083292661/5444517870735015415413993718908291383296000 (...)` and the last line `RESULT: VERIFIED (exit code 0): every check passed`; exit code 0.
- **Run here (2026-11-05, cloud container, see MARGINS_AND_TIMINGS.md):** VERIFIED, exit 0, 1.16 s, peak RSS 28 MiB; log `logs/verify_all_cloud_2026-11-05/continuous-twisting-frontier-dense.log`.
- **Configuration:** `config/continuous-twisting-frontier-dense.json` and CONFIGURATION.md.

## 3. Statements that rest on floating-point evidence (not proofs)

| Where | Statement | Evidence | Nature |
|---|---|---|---|
| Section 5, Design | the design program is infeasible at m = 3, 4 and 5 and feasible at m = 6 (eps = 1e-6) | artifacts/campaign_2026-10/WORKED_STA/loop_mosek_2026-10-12/run.json (solver_status infeasible:MOSEK_DIRECT at m = 3, 4, 5; designed at m = 6); Clarabel agrees: artifacts/campaign_2026-10/WORKED_STA/loop_clarabel_2026-10-12/run.json | solver verdict for this library (paper: a numerical verdict, not a proof of non-existence) |
| Section 5, Design and eq. (6) | sparsification reduces the design to five terms; 7 bits is the shortest rounding the check accepts at eps = 1e-6 | artifacts/campaign_2026-10/WORKED_STA/trace_2026-10-12/tb_chain/m6/design.json (rl1 14 -> 5, 4 solves); artifacts/campaign_2026-10/WORKED_STA/tierB/NOTE.md (1 to 6 bits: W re-solve infeasible) | a shorter rounding was rejected because the float re-solve found no Gram witness, not because an exact check refuted it |
| Appendix B, introduction | each listed function is the shortest rounding of its design that the exact check accepted | artifacts/campaign_2026-10/SHORT_COEFFS/README.md and summary.tsv (order two 1-8 bits, continuous twisting 1-4 bits, PID-like 1-13 bits: W+/W- Gram re-solve infeasible) | as above: no witness found for the shorter roundings |

## 4. Files over 50 MiB

They are in the repository like every other file (GitHub warns above 50 MiB and refuses only above 100 MiB).

| File | sha256 | size |
|---|---|---|
| `certificates/levant-order4/levant5d_m11_nominal.grams.json.gz` | `15fa28be0cec34636d1e0be2dd37477f495a58707dab13400ea7f01b916306bd` | 73,761,710 bytes |

