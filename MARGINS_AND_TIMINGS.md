# Margins, Gram sizes and verification times

Margins, problem sizes and verification times of every certificate, with the machines and software they were measured on. Every number is read from the verifier's own output (`--json` summary and log) or from the Gram files; `margins_and_timings.json` has the same rows.

**Columns.** eps: the margin the record requests. eps_V, eps_W: eps minus the worst charge over the orthants of V, resp. of every decrease rate (all corners and disturbance extremes); eps_certified = min(eps_V, eps_W) is what the certificate proves. ||E||_1/eps: the worst charge over eps, V and W; it equals the worst residual ratio because the accepted PSD shift is eta = 0 on every orthant (column eta; Lemma 2 of the paper is the eta = 0 case). Gram: the largest Gram matrix (number of basis monomials) for V and W; block: the largest parity block of that Gram (the verifier factors each block separately), both from the Gram file's bases and equal to the verifier's log line. Files: record and Gram file sizes. Time: wall time of one verifier process and its peak RSS.

| id | n | m | terms | eps | eps_V | eps_W | worst ||E||_1/eps (V; W) | eta = 0 | Gram V / W | block V / W | record / Gram file | verifier wall, peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `levant-order1` | 2 | 6 | 5 | 1e-06 | 1.000e-06 | 1.000e-06 | 3.90e-07; 2.33e-08 | yes, all 12 orthants | 7 / 6 | 4 / 3 | 4.7 kB / 12.8 kB | 0.08 s, 21 MiB (here) |
| `levant-order2-a` | 3 | 8 | 29 | 1e-07 | 1.000e-07 | 9.997e-08 | 1.65e-06; 2.86e-04 | yes, all 40 orthants | 45 / 36 | 15 / 10 | 3.0 kB / 365.5 kB | 1.02 s, 28 MiB (here) |
| `levant-order2-b` | 3 | 8 | 29 | 1e-07 | 1.000e-07 | 1.000e-07 | 1.65e-06; 2.38e-05 | yes, all 40 orthants | 45 / 36 | 15 / 10 | 3.0 kB / 365.4 kB | 1.11 s, 28 MiB (here) |
| `levant-order2-box` | 3 | 11 | 96 | 1e-04 | 1.000e-04 | 1.000e-04 | 6.48e-09; 5.82e-06 | yes, all 72 orthants | 78 / 66 | 21 / 21 | 27.1 kB / 3.3 MB | 9.98 s, 73 MiB (here) |
| `levant-order3-a` | 4 | 9 | 83 | 1e-04 | 7.523e-05 | 2.700e-05 | 2.48e-01; 7.30e-01 | yes, all 144 orthants | 220 / 165 | 35 / 35 | 30.7 kB / 7.0 MB | 1 min 59 s, 436 MiB (here) |
| `levant-order3-b` | 4 | 9 | 83 | 1e-04 | 1.000e-04 | 1.000e-04 | 8.06e-06; 2.34e-05 | yes, all 144 orthants | 220 / 165 | 35 / 35 | 27.4 kB / 46.9 MB | 2 min 29 s, 703 MiB (here) |
| `levant-order4` | 5 | 11 | 507 | 1e-04 | 9.439e-05 | 7.272e-06 | 5.61e-02; 9.27e-01 | yes, all 64 Gram matrices (separate exact check, `logs/order4_psd_2026-11-09/`) | 1365 / 1001 | 126 / 126 | 124.8 kB / 73.8 MB | 10:16:31, 8471 MiB (spark-91bc) |
| `twisting` | 2 | 5 | 5 | 1e-04 | 1.000e-04 | 1.000e-04 | 2.25e-11; 2.65e-08 | yes, all 12 orthants | 6 / 5 | 3 / 3 | 8.0 kB / 1.7 kB | 0.08 s, 21 MiB (here) |
| `continuous-twisting` | 3 | 5 | 9 | 1e-09 | 1.000e-09 | 9.998e-10 | 2.57e-06; 1.74e-04 | yes, all 24 orthants | 21 / 15 | 6 / 6 | 2.7 kB / 39.0 kB | 0.16 s, 22 MiB (here) |
| `continuous-twisting-frontier` | 3 | 9 | 52 | 1e-06 | 9.998e-07 | 9.608e-07 | 2.35e-04; 3.92e-02 | yes, all 24 orthants | 55 / 45 | 15 / 15 | 18.5 kB / 522.1 kB | 1.20 s, 30 MiB (here) |
| `pid-like` | 3 | 7 | 21 | 1e-06 | 1.000e-06 | 9.924e-07 | 1.11e-05; 7.60e-03 | yes, all 24 orthants | 36 / 28 | 10 / 10 | 11.3 kB / 138.7 kB | 0.41 s, 24 MiB (here) |
| `levant-order2-box-a` | 3 | 11 | 96 | 1e-04 | 1.000e-04 | 1.000e-04 | 6.48e-09; 5.82e-06 | yes, all 40 orthants | 78 / 66 | 21 / 21 | 26.4 kB / 1.9 MB | 6.22 s, 51 MiB (here) |
| `levant-order2-box-b` | 3 | 11 | 96 | 1e-04 | 1.000e-04 | 1.000e-04 | 6.48e-09; 4.85e-06 | yes, all 40 orthants | 78 / 66 | 21 / 21 | 26.4 kB / 1.9 MB | 5.82 s, 51 MiB (here) |
| `levant-order2-box-dense` | 3 | 11 | 189 | 1e-04 | 1.000e-04 | 1.000e-04 | 1.00e-08; 8.25e-06 | yes, all 72 orthants | 78 / 66 | 21 / 21 | 41.6 kB / 2.7 MB | 9.25 s, 60 MiB (here) |
| `levant-order2-box-a-dense` | 3 | 11 | 189 | 1e-04 | 1.000e-04 | 1.000e-04 | 1.00e-08; 6.62e-06 | yes, all 40 orthants | 78 / 66 | 21 / 21 | 40.8 kB / 1.6 MB | 5.31 s, 43 MiB (here) |
| `levant-order2-box-b-dense` | 3 | 11 | 189 | 1e-04 | 1.000e-04 | 1.000e-04 | 1.00e-08; 8.25e-06 | yes, all 40 orthants | 78 / 66 | 21 / 21 | 40.8 kB / 1.6 MB | 5.47 s, 43 MiB (here) |
| `levant-order3-a-dense` | 4 | 9 | 163 | 1e-04 | 7.523e-05 | 2.700e-05 | 2.48e-01; 7.30e-01 | yes, all 144 orthants | 220 / 165 | 35 / 35 | 45.0 kB / 6.2 MB | 1 min 55 s, 424 MiB (here) |
| `levant-order3-b-dense` | 4 | 9 | 163 | 1e-04 | 1.000e-04 | 1.000e-04 | 1.26e-05; 4.64e-05 | yes, all 144 orthants | 220 / 165 | 35 / 35 | 41.1 kB / 40.2 MB | 2 min 15 s, 519 MiB (here) |
| `continuous-twisting-frontier-dense` | 3 | 9 | 101 | 1e-06 | 9.997e-07 | 9.392e-07 | 3.43e-04; 6.08e-02 | yes, all 24 orthants | 55 / 45 | 15 / 15 | 24.9 kB / 419.6 kB | 1.16 s, 28 MiB (here) |

`levant-order4` is marked `long_run` and was not re-run here; its row comes from the workstation run (`logs/workstation/`), and its Gram sizes are those of the full homogeneous bases of degree m and m - 1 in five variables, equal to that run's log (1365 basis -> 16 parity blocks, largest 126; 1001 -> 16, largest 126).

## Machines

**Runs labelled 'here'** (this package, 2026-11-05): the verifier of this release (sha256 `1a1908df5bd34b160d269f4cdbda4b356ed769859b181ef7af64c3bbd0113aa5`), one process per certificate, one after the other, by `tools/verify_all.py` on a cloud container: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 cores, 15.7 GiB RAM, Linux-6.18.44-fc-v236-x86_64-with-glibc2.39, CPython 3.13.16 (`/usr/bin/python3`); apart from light activity (file edits, a check script of a few seconds) nothing else ran. Logs: `logs/verify_all_cloud_2026-11-05/`.

**spark-91bc** (the author's workstation): Linux aarch64, 20 cores, 121 GiB RAM, CPython 3.12.3 (`/usr/bin/python3`, stdlib only for the verifier), MOSEK 11.1.11 for the designs. **CPU model not recorded.** The order-four run (10:16:31, 2026-09-25/26) shared the host with other campaign jobs.

The cloud machine is slower than the workstation: here the order-three certificates took 115 to 149 s, while earlier workstation runs took 61 to 89 s for the dense order-three records (the authors estimate a factor of about 1.8). The paper's Conclusion quotes the workstation (less than a minute with two or three states, one to two minutes with four, about ten hours with five).

## Earlier recorded verifier runs on the same bytes

| id | wall | machine | source |
|---|---|---|---|
| `levant-order2-box` | 9.71 s | cloud container, when the record was made | the record's verification log (copied to `logs/earlier/`) |
| `levant-order3-a` | 1 min 56 s | cloud container, when the record was made | the record's verification log (copied to `logs/earlier/`) |
| `levant-order3-b` | 2 min 28 s | cloud container, when the record was made | the record's verification log (copied to `logs/earlier/`) |
| `levant-order4` | 10:16:31 | spark-91bc | insurance P (`logs/workstation/`) |
| `continuous-twisting-frontier` | 1.27 s | cloud container, when the record was made | the record's verification log (copied to `logs/earlier/`) |
| `levant-order2-box-a` | 5.78 s | cloud container, when the record was made | the record's verification log (copied to `logs/earlier/`) |
| `levant-order2-box-b` | 5.67 s | cloud container, when the record was made | the record's verification log (copied to `logs/earlier/`) |

## Software

- Verifier: Python standard library only (fractions, gzip, json, hashlib, argparse, itertools, pathlib, traceback, zlib); any CPython >= 3.9.
- Designs (not needed to verify): MOSEK 11.1.11 through its Task API, CPython 3.12.3, cvxpy 1.8.1, CLARABEL 0.11.1, numpy 2.4.2, scipy 1.17.0 on spark-91bc; CLARABEL 0.11.1, cvxpy 1.9.3, numpy 2.5.3, scipy 1.18.1 on CPython 3.13.16 for the cloud tier-B roundings (CONFIGURATION.md, Section 10).

