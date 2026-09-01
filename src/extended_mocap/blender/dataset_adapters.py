"""Dataset adapters: convert external motion data into a format the pipeline accepts.

Each adapter yields a sequence of ``(source_fbx_or_npz, output_csv, metadata)``
tuples that the headless Blender pipeline (or equivalent) can process.

Currently supported:

* ``mixamo_fbx`` – directory of Mixamo FBX files (direct input for
  :mod:`extended_mocap.blender.headless_process`).
* ``amass_smpl`` – AMASS SMPL-X ``*.npz`` files. Requires ``smplx`` and
  ``torch``; extracts per-frame 6D rotations from SMPL-X parameters and
  writes them in a format compatible with the CC_Base extraction layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class AnimEntry:
    """One animation ready for the pipeline."""

    anim_id: str
    source_path: str
    format: str  # "fbx" | "npz"
    metadata: dict | None = None


# ── Mixamo FBX adapter ───────────────────────────────────────────────────────


def scan_mixamo_fbx(anim_dir: Path) -> list[AnimEntry]:
    """Return ``AnimEntry`` objects for every ``*.fbx`` in *anim_dir*.

    These are consumed directly by ``blender --python headless_process.py``.
    """
    entries: list[AnimEntry] = []
    for f in sorted(anim_dir.glob("*.fbx")) + sorted(anim_dir.glob("*.FBX")):
        entries.append(AnimEntry(anim_id=f.stem, source_path=str(f), format="fbx"))
    log.info("scan_mixamo_fbx: found %d FBX files in %s", len(entries), anim_dir)
    return entries


# ── AMASS SMPL-X adapter (skeleton) ─────────────────────────────────────────

def _smplx_to_ccbase_not_implemented() -> None:
    """TODO: implement per-frame SMPL-X → CC_Base joint rotation mapping.

    SMPL-X ``poses`` tensor (55 × 3 × 3 or 55 × 6) encodes per-joint local
    rotations in the SMPL-X skeleton order.  CC_Base uses a different bone
    hierarchy and rest-pose convention.  A minimal approach:

    1. Convert SMPL-X ``poses`` to 6D rotation per joint.
    2. Apply the SMPL-X forward kinematics to recover global rotations.
    3. Map SMPL-X joint IDs to CC_Base bone names via an explicit lookup.
    4. Re-express global rotations in CC_Base rest-pose-local convention.

    References:
    * ``smplx`` library: ``forwardKinematics`` method.
    * ``MixamoRetarget`` in ``retargeting.py`` for bone-name mapping.
    """
    raise NotImplementedError


def scan_amass_smpl(npz_dir: Path) -> list[AnimEntry]:
    """Stub: list AMASS SMPL-X ``*.npz`` files.

    Each file contains keys ``poses`` ``(T, 165)`` or ``(T, 55, 3, 3)``,
    ``betas``, ``trans``, etc.  Actual conversion requires ``smplx`` and
    the ``configs/amass_to_ccbase.json`` joint mapping (to be built).
    """
    entries: list[AnimEntry] = []
    for f in sorted(npz_dir.glob("*.npz")):
        entries.append(AnimEntry(anim_id=f.stem, source_path=str(f), format="npz"))
    log.info("scan_amass_smpl: found %d npz files in %s", len(entries), npz_dir)
    return entries
