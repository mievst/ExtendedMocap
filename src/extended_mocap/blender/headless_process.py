"""Headless Blender pipeline: process one animation at a time.

Load a base scene (CC_Base rig + character, no animations), import a source
animation (Mixamo FBX), retarget, extract quaternion CSV, optionally render,
then clean up before the next animation. Run with::

    blender --background --python blender/headless_process.py -- \
        --anim-dir PATH/TO/fbx --csv-dir PATH/OUT/csv [--render-dir PATH/OUT/mp4] \
        [--base data/base.blend] [--model-name party-m-0001]

Options after ``--`` are forwarded to this script; Blender ignores them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("headless")

# ── helpers ──────────────────────────────────────────────────────────────────


def _clear_animation(armature_obj) -> None:
    """Remove all keyframe animation data from *armature_obj*."""
    if armature_obj.animation_data is not None:
        armature_obj.animation_data_clear()


def _collect_new_objects(before: set) -> list:
    """Return objects added to the scene since *before* was captured."""
    return [o for o in bpy.data.objects if o.name not in before]


def _clamp_loc(loc, lo: float = -0.5, hi: float = 0.5):
    return tuple(max(min(v, hi), lo) for v in loc)


# ── quat extraction ──────────────────────────────────────────────────────────

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


def _extract_quat_csv(armature_obj, out_path: str) -> int:
    """Write per-frame quaternion CSV for all CC_Base bones. Returns frame count."""
    import pandas as pd

    pose_bones = armature_obj.pose
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")

    start = 0
    end = int(armature_obj.animation_data.action.frame_range[-1]) + 1
    hip_initial = pose_bones["CC_Base_Hip"].location
    hip_off = tuple(-v for v in hip_initial)

    data: dict[str, list] = {}
    for name in BONE_NAMES:
        for q in range(4):
            data[f"{name}.Q{q}"] = []
    for t in range(3):
        data["CC_Base_Hip.T" + str(t)] = []

    scene = bpy.context.scene
    for frame in range(start, end):
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
    return end - start


# ── render ───────────────────────────────────────────────────────────────────


def _render(armature_obj, model_obj, out_path: str) -> int:
    """Copy animation from *armature_obj* onto *model_obj* and render to MP4."""
    # link model to scene and copy animation
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.make_links_data(type="ANIMATION")

    frames = int(armature_obj.animation_data.action.frame_range[-1]) + 1
    scene = bpy.context.scene
    scene.frame_set(0)
    hip_off = tuple(
        -v for v in armature_obj.pose.bones["CC_Base_Hip"].location
    )
    model_hip = model_obj.pose.bones["CC_Base_Hip"]
    for frame in range(frames):
        scene.frame_set(frame)
        new_loc = _clamp_loc(
            tuple(armature_obj.pose.bones["CC_Base_Hip"].location[i] + hip_off[i] for i in range(3))
        )
        model_hip.location = new_loc
        bpy.context.view_layer.update()
        model_hip.keyframe_insert(data_path="location", frame=frame)

    # render
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 8
    scene.eevee.sss_samples = 1
    scene.eevee.use_volumetric_lights = False
    scene.frame_start = 0
    scene.frame_end = frames
    scene.frame_step = 1
    scene.render.use_sequencer = False
    scene.render.use_compositing = False
    scene.render.fps = 60
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "MPEG4"
    scene.render.ffmpeg.video_bitrate = 10000
    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.ffmpeg.audio_bitrate = 192
    scene.render.filepath = out_path
    bpy.ops.render.render(animation=True)
    return frames


# ── main loop ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse args after the Blender ``--`` separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anim-dir", required=True, help="Directory of FBX animation files")
    p.add_argument("--csv-dir", required=True, help="Output directory for quaternion CSVs")
    p.add_argument("--render-dir", default=None, help="Output directory for rendered MP4s (omit to skip rendering)")
    p.add_argument("--base", default=None, help="Base .blend file with CC_Base rig + character")
    p.add_argument("--model-name", default="party-m-0001", help="Name of CC_Base mesh/armature object in base scene")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    anim_dir = Path(args.anim_dir)
    csv_dir = Path(args.csv_dir)
    render_dir = Path(args.render_dir) if args.render_dir else None
    csv_dir.mkdir(parents=True, exist_ok=True)
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    # load base scene
    if args.base and os.path.isfile(args.base):
        bpy.ops.wm.open_mainfile(filepath=args.base)
    target_obj = bpy.context.scene.objects.get(args.model_name)
    if target_obj is None:
        log.error("Model object '%s' not found in scene", args.model_name)
        sys.exit(1)

    # bone mapping
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(here, "configs", "mixamo_to_ccbase.json")
    with open(mapping_path) as f:
        bone_mapping = json.load(f)

    fbx_files = sorted(anim_dir.glob("*.fbx")) + sorted(anim_dir.glob("*.FBX"))
    log.info("found %d FBX files in %s", len(fbx_files), anim_dir)

    for fbx in fbx_files:
        anim_name = fbx.stem
        log.info("processing %s", anim_name)
        before = set(bpy.data.objects)
        bpy.ops.import_scene.fbx(filepath=str(fbx))
        imported = _collect_new_objects(before)
        source_obj = next((o for o in imported if o.type == "ARMATURE"), None)
        if source_obj is None:
            log.warning("no armature found in %s, skipping", fbx.name)
            continue

        # retarget source -> CC_Base (absolute import for blender --python)
        from extended_mocap.blender.retargeting import ArmatureRetargeting
        _clear_animation(target_obj)
        bpy.context.view_layer.objects.active = target_obj
        retargeter = ArmatureRetargeting(source_obj, target_obj, bone_mapping)
        retargeter.retarget_animation()

        # extract quats
        csv_path = csv_dir / f"{anim_name}.csv"
        n_frames = _extract_quat_csv(target_obj, str(csv_path))
        log.info("  -> %s (%d frames)", csv_path.name, n_frames)

        # optional render
        if render_dir is not None:
            mp4_path = render_dir / f"{anim_name}.mp4"
            _render(target_obj, target_obj, str(mp4_path))
            log.info("  -> %s", mp4_path.name)

        # cleanup: remove imported armature, clear target animation
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
        _clear_animation(target_obj)
        log.info("  cleaned up %s", anim_name)


if __name__ == "__main__":
    main()