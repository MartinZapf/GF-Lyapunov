"""Write-side guard for the frozen artifact roots (WP P2-12).

``artifacts/ablation/**`` holds the note's frozen wording root
(``artifacts/ablation/levant_2003_perturbed/README.md``, last in the repository
precedence order) and the records every published claim is worded from. Several
drivers used to default their output directory *into* that tree, so an
exploratory re-run silently landed next to a published record.

Since 2026-09-03 those drivers default to a campaign directory instead. This
module is the single place that says where "outside the frozen root" is and
refuses a write back into it unless the caller opts in explicitly.

Nothing here reads or edits an artifact; it only resolves a path.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Trees no driver may write into by default.
FROZEN_ROOTS = (REPO_ROOT / "artifacts" / "ablation",)

#: Where the retargeted standard-tuning drivers write instead.
CAMPAIGN_STANDARD_GAINS = (
    REPO_ROOT / "artifacts" / "campaign_2026-08" / "E1_matrix"
    / "standard_gains_2026-09"
)

OPT_IN_FLAG = "--allow-frozen-root"


def is_frozen(path: Path) -> bool:
    """True when ``path`` lies inside one of :data:`FROZEN_ROOTS`."""
    resolved = Path(path).resolve()
    for root in FROZEN_ROOTS:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def resolve_out_dir(requested, default: Path, *, allow_frozen: bool = False,
                    create: bool = True) -> Path:
    """Resolve a driver's output directory, refusing the frozen root.

    ``requested`` is the CLI value (``None`` selects ``default``). A path
    inside :data:`FROZEN_ROOTS` raises unless ``allow_frozen`` is set, which is
    what the ``--allow-frozen-root`` flag is for: regenerating a published
    record in place is a deliberate act, never a default.
    """
    out = Path(requested) if requested else Path(default)
    if is_frozen(out) and not allow_frozen:
        raise SystemExit(
            f"refusing to write into the frozen artifact root: {out}\n"
            f"  {FROZEN_ROOTS[0]} holds the published records and the note's\n"
            f"  wording root; drivers default to {default} instead.\n"
            f"  Pass {OPT_IN_FLAG} to override deliberately."
        )
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def add_out_dir_args(ap, default: Path, *, flags=("--out-dir", "--output-dir"),
                     what: str = "all outputs") -> None:
    """Register the standard ``--out-dir`` / ``--allow-frozen-root`` pair."""
    ap.add_argument(*flags, dest="out_dir", default=None,
                    help=f"{what} are written under this directory only "
                         f"(default: {default.relative_to(REPO_ROOT)}). The "
                         f"historical artifacts/ablation/ path is NO LONGER "
                         f"the default and is refused unless "
                         f"{OPT_IN_FLAG} is given.")
    ap.add_argument(OPT_IN_FLAG, action="store_true",
                    help="permit writes into artifacts/ablation/** (the frozen "
                         "wording root). Off by default; only for a deliberate "
                         "regeneration of a published record.")
