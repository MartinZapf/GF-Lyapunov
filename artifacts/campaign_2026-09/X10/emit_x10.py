#!/usr/bin/env python3
"""X10 emission shim: run scripts/emit_certificate_record.py (EM) unchanged,
with one additive entry in its COVARIANT_SCALING table.

Why. EM's ``--representative-p`` (WP TA) looks the covariant map up by the
base point's system key, and the table has only ``levant4d``. The X10 cells
are proved at ``FLAGSHIP_GAIN_POINTS["levant4d_2003print"]`` scaled by p, a
point with the SAME equations and therefore the SAME covariant map
(lambda_i -> p^i lambda_i, delta0 -> p^4 delta0). EM is not X10's file, so the
entry is proposed as a one-line diff (docs/papers/big_draft/reports/X10.md,
"Proposed code change outside X10's files") and applied here at run time,
nothing else. Every EM pin, refusal and the self-verification run exactly as
in EM: this shim changes no function, only adds one dict entry, and only when
EM does not already carry it. If EM carries it (the diff has landed), the
shim checks it is the levant4d map and changes nothing.

The record's content does not name the shim: EM writes its constant
``provenance.emitter`` and hashes only its inputs, so the same command through
EM itself (after the diff lands) reproduces the record except its
``timestamp_utc``.

Usage: identical to EM, e.g.
  .venv/bin/python artifacts/campaign_2026-09/X10/emit_x10.py \
      --exact-cell <cell>/exact --design-record <rec> --design-bundle-npz <npz> \
      --gain-point flagship --representative-p 1/2 --stem-tag ... --out-dir ...
"""
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import emit_certificate_record as em  # noqa: E402

KEY = "levant4d_2003print"
BASE = "levant4d"

if KEY in em.COVARIANT_SCALING:
    if em.COVARIANT_SCALING[KEY] != em.COVARIANT_SCALING[BASE]:
        raise SystemExit(f"EM's COVARIANT_SCALING[{KEY!r}] differs from the "
                         f"{BASE} map; refusing to guess")
    print(f"[emit_x10] EM already registers {KEY}; shim inactive",
          file=sys.stderr)
else:
    em.COVARIANT_SCALING[KEY] = copy.deepcopy(em.COVARIANT_SCALING[BASE])
    print(f"[emit_x10] COVARIANT_SCALING[{KEY!r}] = "
          f"{em.COVARIANT_SCALING[KEY]!r} (copied from {BASE!r})",
          file=sys.stderr)

sys.argv[0] = str(REPO / "scripts" / "emit_certificate_record.py")
raise SystemExit(em.main())
