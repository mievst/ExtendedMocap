"""Convert/retarget Mixamo FBX armatures to a CC_Base (AccuRig) armature.

Runs inside Blender (bpy). Reads the bone mapping from a JSON file and
copies per-frame pose rotations from each armature in a source collection to
a fresh CC_Base target armature.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import bpy
import mathutils
from tqdm import tqdm


def _default_mapping_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "configs", "mixamo_to_ccbase.json")


def load_bone_mapping(path: str | None = None) -> Mapping[str, str]:
    path = path or _default_mapping_path()
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Bone mapping {path} must be a JSON dict")
    return raw


def calculate_scale_factor(source_armature, target_armature) -> float:
    """Ratio of target bounding-box size to source bounding-box size."""
    source_bbox = [
        source_armature.matrix_world @ mathutils.Vector(c) for c in source_armature.bound_box
    ]
    target_bbox = [
        target_armature.matrix_world @ mathutils.Vector(c) for c in target_armature.bound_box
    ]
    source_size = max(
        max(c[i] for c in source_bbox) - min(c[i] for c in source_bbox) for i in range(3)
    )
    target_size = max(
        max(c[i] for c in target_bbox) - min(c[i] for c in target_bbox) for i in range(3)
    )
    return target_size / source_size


def retarget_animation(source_armature, target_armature, bone_mapping: Mapping[str, str]) -> None:
    bpy.context.view_layer.objects.active = source_armature
    source_armature.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.ops.object.mode_set(mode="POSE")

    start_frame = 0
    end_frame = int(bpy.context.object.animation_data.action.frame_range[-1]) + 1

    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.object.mode_set(mode="POSE")

    scene = bpy.context.scene
    for frame in tqdm(range(start_frame, end_frame + 1)):
        scene.frame_set(frame)
        for src_bone_name, tgt_bone_name in bone_mapping.items():
            if src_bone_name not in source_armature.pose.bones:
                continue
            if tgt_bone_name not in target_armature.pose.bones:
                continue

            src_bone = source_armature.pose.bones[src_bone_name]
            tgt_bone = target_armature.pose.bones[tgt_bone_name]

            rot = (source_armature.matrix_world @ src_bone.matrix).decompose()[1]
            tgt_loc = (target_armature.matrix_world @ tgt_bone.matrix).decompose()[0]
            tgt_sca = (target_armature.matrix_world @ tgt_bone.matrix).decompose()[2]

            mat_out = mathutils.Matrix.LocRotScale(tgt_loc, rot, tgt_sca)
            tgt_bone.matrix = target_armature.matrix_world.inverted() @ mat_out
            bpy.context.view_layer.update()
            tgt_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            if tgt_bone_name == "CC_Base_Hip":
                tgt_bone.location = src_bone.location
                bpy.context.view_layer.update()
                tgt_bone.keyframe_insert(data_path="location", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


if __name__ == "__main__":
    example_armature = bpy.context.scene.objects.get("idle_251087")
    accurig_collection = bpy.data.collections["AccuRig"]
    mapping = load_bone_mapping()

    for source_armature in bpy.data.collections["mixamo"].objects:
        target = example_armature.copy()
        target.name = "AccuRig_" + source_armature.name
        target.animation_data_clear()
        accurig_collection.objects.link(target)
        retarget_animation(source_armature, target, mapping)
