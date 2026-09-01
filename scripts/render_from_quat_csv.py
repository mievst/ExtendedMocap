"""Headless Blender: render an animation from a quaternion CSV, not from in-blend actions.

Reads the per-armature quaternion CSV produced by ``extract_from_blends.py`` (columns
``CC_Base_*.Q0..Q3`` plus ``CC_Base_Hip.T0..T2``), keys the pose onto a CC_Base model
object, and renders an MP4. This decouples rendering from .blend files: anything with a
quat CSV can be visualized.

Run inside Blender headless::

    blender --background --python render_from_quat_csv.py -- \
        --csv path/anim.csv --out path/anim.mp4 [--base scene.blend] [--model party-m-0001]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render_from_quat_csv")


def _key_csv_animation(armature_obj, csv_path: str) -> int:
    """Apply quaternion CSV to *armature_obj* pose, keying each frame. Returns frame count."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    pose = armature_obj.pose

    # map CSV column -> pose bone
    bone_cols: dict[str, list[tuple[str, int]]] = {}
    for col in df.columns:
        # format: BoneName.Q{0..3} or CC_Base_Hip.T{0..2}
        parts = col.rsplit(".", 1)
        if len(parts) != 2:
            continue
        bone, comp = parts
        if comp[0] in ("Q", "T"):
            idx = int(comp[1:])
        else:
            continue
        bone_cols.setdefault(bone, []).append((comp[0], idx))

    scene = bpy.context.scene
    n_frames = len(df)

    for frame in range(n_frames):
        scene.frame_set(frame)
        row = df.iloc[frame]
        for bone_name, comps in bone_cols.items():
            if bone_name not in pose.bones:
                continue
            bone = pose.bones[bone_name]
            quat = mathutils.Quaternion(
                [float(row[f"{bone_name}.Q{i}"]) for i in range(4)]
            )
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = quat
            if any(c == "T" for c, _ in comps):
                loc = mathutils.Vector(
                    [float(row[f"{bone_name}.T{i}"]) for i in range(3)]
                )
                bone.location = loc
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            if any(c == "T" for c, _ in comps):
                bone.keyframe_insert(data_path="location", frame=frame)
        bpy.context.view_layer.update()

    return n_frames


def _render(scene, n_frames: int, out_path: str) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 8
    scene.eevee.sss_samples = 1
    scene.eevee.use_volumetric_lights = False
    scene.frame_start = 0
    scene.frame_end = n_frames - 1
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


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Quaternion CSV to animate from")
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument("--base", default=None, help="Optional base .blend scene (character+camera+lights)")
    p.add_argument("--model", default="party-m-0001", help="CC_Base model object name in scene")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    if not csv_path.is_file():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.base and os.path.isfile(args.base):
        bpy.ops.wm.open_mainfile(filepath=args.base)
    armature_obj = bpy.context.scene.objects.get(args.model)
    if armature_obj is None or armature_obj.type != "ARMATURE":
        log.error("Model armature '%s' not found", args.model)
        sys.exit(1)

    n_frames = _key_csv_animation(armature_obj, str(csv_path))
    log.info("keyed %d frames from %s", n_frames, csv_path.name)
    _render(bpy.context.scene, n_frames, str(out_path))
    log.info("rendered %s", out_path)


if __name__ == "__main__":
    main()