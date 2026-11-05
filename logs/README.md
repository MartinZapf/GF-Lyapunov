# Verifier logs

- `verify_all_cloud_2026-11-05/`: the runs of this package (`tools/verify_all.py`, 2026-11-05),
  one verifier process per certificate on a 4-core x86_64 cloud container (environment in
  `summary.json`). `<id>.log` is the verifier's output, `<id>.verification.json` its `--json`
  summary, which binds the verdict to the sha256 of the verifier, record and Gram file it read.
- `workstation/`: the earlier run of `levant-order4` on the authors' workstation
  (2026-09-25/26, 10:16:31 wall, `/usr/bin/time -v` output beside it). It used the verifier
  of that date (sha256 a9004717...), whose arithmetic is that of the current one; only the
  messages and exit codes changed since (outcome classes, 2026-11-05).
- `earlier/`: the verifier logs written when the records were made (cloud, 2026-10-12 to
  2026-11-05), kept for comparison.

Absolute path prefixes of the machines that ran these logs were replaced by `<release root>`
or `<GF Framework checkout>`. Nothing else was changed; the sha256 values are untouched.
