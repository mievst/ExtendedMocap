"""Headless Blender: extract quaternion CSVs from CC_Base armatures in blend files.

Opens a .blend (which already contains fully retargeted CC_Base armatures), iterates
every armature with 101 bones, and writes one per-frame quaternion CSV per armature.

The per-armature CSV follows the original exporter naming ``{armature}_{motion}.csv``
and a JSON manifest records provenance (source blend, armature, motion_id, frames).

Run with Blender headless::

    blender --background --python extract_from_blends.py -- \
        --blend path/to/file.blend --csv-dir path/out --manifest path/manifest.json

Options after ``--`` are forwarded to this script; Blender ignores them.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_from_blends")

BONE_NAMES = [
    "CC_Base_Hip", "CC_Base_FacialBone", "CC_Base_NeckTwist02",
    "CC_Base_NeckTwist01", "CC_Base_Pelvis", "CC_Base_Waist",
    "CC_Base_Spine01", "CC_Base_Spine02", "CC_Base_R_Clavicle",
    "CC_Base_L_Clavicle", "CC_Base_R_Upperarm", "CC_Base_L_Upperarm",
    "CC_Base_R_Forearm", "CC_Base_L_Forearm", "CC_Base_R_Hand",
    "CC_Base_L_Hand", "CC_Base_L_Mid1", "CC_Base_L_Mid2", "CC_Base_L_Mid3",
    "CC_Base_L_Index1", "CC_Base_L_Index2", "CC_Base_L_Index3",
    "CC_Base_L_Ring1", "CC_Base_L_Ring2", "CC_Base_L_Ring3",
    "CC_Base_L_Pinky1", "CC_Base_L_Pinky2", "CC_Base_L_Pinky3",
    "CC_Base_L_Thumb1", "CC_Base_L_Thumb2", "CC_Base_L_Thumb3",
    "CC_Base_R_Mid1", "CC_Base_R_Mid2", "CC_Base_R_Mid3",
    "CC_Base_R_Ring1", "CC_Base_R_Ring2", "CC_Base_R_Ring3",
    "CC_Base_R_Thumb1", "CC_Base_R_Thumb2", "CC_Base_R_Thumb3",
    "CC_Base_R_Index1", "CC_Base_R_Index2", "CC_Base_R_Index3",
    "CC_Base_R_Pinky1", "CC_Base_R_Pinky2", "CC_Base_R_Pinky3",
    "CC_Base_R_Thigh", "CC_Base_R_Calf", "CC_Base_R_Foot",
    "CC_Base_L_Thigh", "CC_Base_L_Calf", "CC_Base_L_Foot", "CC_Base_BoneRoot",
]


def _clamp_loc(loc, lo: float = -0.5, hi: float = 0.5):
    return tuple(max(min(v, hi), lo) for v in loc)


def _motion_id_from_action(action: str | None) -> str | None:
    """Normalized motion name from an action label like ``Armature.001|elderly-idle``.

    The motion is the part after the last ``|`` where the earlier part repeats the
    armature name; for ``motion|A|Layer0`` style the motion is the first part.
    """
    if not action:
        return None
    parts = action.split("|")
    if len(parts) >= 3 and parts[1] == "A":
        # 'MotionName|A|Layer0 Retarget' -> take the first (or the \(N\)-stripped first)
        motion = parts[0]
    else:
        # 'Armature.NNN|motion-name' -> take the last
        motion = parts[-1]
    motion = re.sub(r"\.\d+$", "", motion)
    motion = re.sub(r"\s*\(\d+\)\s*$", "", motion).strip()
    return motion or None


def _extract_armature(armature_obj, out_path: str) -> int:
    """Write per-frame quaternion CSV. Returns frame count."""
    import pandas as pd

    pose_bones = armature_obj.pose
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    action = armature_obj.animation_data.action
    end = int(action.frame_range[-1]) + 1 if action else 1
    hip_initial = pose_bones["CC_Base_Hip"].location
    hip_off = tuple(-v for v in hip_initial)

    data: dict[str, list] = {}
    for name in BONE_NAMES:
        for q in range(4):
            data[f"{name}.Q{q}"] = []
    for t in range(3):
        data["CC_Base_Hip.T" + str(t)] = []

    scene = bpy.context.scene
    for frame in range(end):
        scene.frame_set(frame)
        for name in BONE_NAMES:
            bone = pose_bones[name]
            bone.rotation_mode = "QUATERNION"
            for q in range(4):
                data[f"{name}.Q{q}"].append(bone.rotation_quaternion[q])
        loc = pose_bones["CC_Base_Hip"].location
        clamped = _clamp_loc(tuple(loc[i] + hip_off[i] for i in range(3)))
        for t in range(3):
            data[f"CC_Base_Hip.T{t}"].append(clamped[t])

    bpy.ops.object.mode_set(mode="OBJECT")
    pd.DataFrame(data).to_csv(out_path, index=False)
    return end


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blend", required=True, help="Path to source .blend file")
    p.add_argument("--csv-dir", required=True, help="Output directory for quaternion CSVs")
    p.add_argument("--manifest", default=None, help="Optional JSON manifest path (write/append)")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    blend_path = Path(args.blend)
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    if not blend_path.is_file():
        log.error("Blend file not found: %s", blend_path)
        sys.exit(1)

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    entries = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        if len(obj.data.bones) != 101:
            continue
        action = obj.animation_data.action if obj.animation_data else None
        action_name = action.name if action else None
        motion_id = _motion_id_from_action(action_name)
        if motion_id is None:
            log.warning("  no motion name for armature %s (action %s), skipping", obj.name, action_name)
            continue

        safe_motion = re.sub(r"[^A-Za-z0-9._-]+", "_", motion_id)
        safe_arm = re.sub(r"[^A-Za-z0-9._-]+", "_", obj.name)
        out_name = f"{safe_arm}.{safe_motion}.csv"
        out_path = csv_dir / out_name
        if out_path.exists():
            log.info("  exists, skip: %s", out_name)
        else:
            log.info("extracting %s (action %s) ...", obj.name, action_name)
            n = _extract_armature(obj, str(out_path))
            log.info("  -> %s (%d frames)", out_name, n)
        entries.append(
            {
                "source_blend": blend_path.name,
                "armature": obj.name,
                "action": action_name,
                "motion_id": motion_id,
                "frames": int(action.frame_range[-1]) + 1 if action else 0,
                "csv": out_name,
            }
        )

    log.info("done: %d CC_Base armatures in %s", len(entries), blend_path.name)

    if args.manifest:
        manifest_path = Path(args.manifest)
        data = []
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data.append({"blend": blend_path.name, "entries": entries})
        manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("manifest updated: %s", manifest_path)


if __name__ == "__main__":
    main()