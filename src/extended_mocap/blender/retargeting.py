"""Blender armature retargeting (Mixamo -> CC_Base / AccuRig).

Runs inside Blender's Python (bpy). Copies pose rotations from a source
armature to a target armature per-frame using an explicit bone mapping.
"""

from __future__ import annotations

import bpy
import mathutils
from tqdm import tqdm


class ArmatureRetargeting:
    def __init__(self, source_armature, target_armature, bone_mapping, finger_bones=None):
        self.source_armature = source_armature
        self.target_armature = target_armature
        self.bone_mapping = bone_mapping
        # Bones that keep their location (in addition to rotation).
        self.finger_bones = set(finger_bones or [])

    def calculate_scale_factor(self):
        source_bbox = [
            self.source_armature.matrix_world @ mathutils.Vector(corner)
            for corner in self.source_armature.bound_box
        ]
        target_bbox = [
            self.target_armature.matrix_world @ mathutils.Vector(corner)
            for corner in self.target_armature.bound_box
        ]
        source_size = max(
            [
                max(corner[i] for corner in source_bbox) - min(corner[i] for corner in source_bbox)
                for i in range(3)
            ]
        )
        target_size = max(
            [
                max(corner[i] for corner in target_bbox) - min(corner[i] for corner in target_bbox)
                for i in range(3)
            ]
        )
        return target_size / source_size

    def get_global_rotation_correction(self, source_bone, target_bone):
        src_global_orientation = source_bone.matrix.to_quaternion()
        tgt_global_orientation = target_bone.matrix.to_quaternion()
        correction_quat = tgt_global_orientation * src_global_orientation.inverted()
        return correction_quat

    def retarget_animation(self):
        bpy.context.view_layer.objects.active = self.source_armature
        self.source_armature.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.ops.object.mode_set(mode="POSE")

        start_frame, end_frame = self.read_anim_start_end(self.source_armature)

        bpy.context.view_layer.objects.active = self.target_armature
        bpy.ops.object.mode_set(mode="POSE")
        scene = bpy.context.scene

        for frame in tqdm(range(start_frame, end_frame + 1)):
            scene.frame_set(frame)
            for src_bone_name, tgt_bone_name in self.bone_mapping.items():
                if (
                    src_bone_name in self.source_armature.pose.bones
                    and tgt_bone_name in self.target_armature.pose.bones
                ):
                    src_bone = self.source_armature.pose.bones[src_bone_name]
                    tgt_bone = self.target_armature.pose.bones[tgt_bone_name]
                    loc, rot, _sca = (
                        self.source_armature.matrix_world @ src_bone.matrix
                    ).decompose()
                    mat_out = mathutils.Matrix.LocRotScale(
                        (self.target_armature.matrix_world @ tgt_bone.matrix).decompose()[0],
                        rot,
                        (self.target_armature.matrix_world @ tgt_bone.matrix).decompose()[2],
                    )
                    tgt_bone.matrix = self.target_armature.matrix_world.inverted() @ mat_out
                    bpy.context.view_layer.update()
                    tgt_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                    if tgt_bone_name in self.finger_bones or tgt_bone_name == "CC_Base_Hip":
                        tgt_bone.location = loc
                        bpy.context.view_layer.update()
                        tgt_bone.keyframe_insert(data_path="location", frame=frame)
        # self.apply_constraints()
        # self.bake_animation(start_frame, end_frame)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def apply_constraints(self):
        # Applying constraints to maintain correct bone orientations
        for src_bone_name, tgt_bone_name in self.bone_mapping.items():
            if (
                src_bone_name in self.source_armature.pose.bones
                and tgt_bone_name in self.target_armature.pose.bones
            ):
                tgt_bone = self.target_armature.pose.bones[tgt_bone_name]
                constraint = tgt_bone.constraints.new("COPY_TRANSFORMS")
                constraint.target = self.source_armature
                constraint.subtarget = src_bone_name

    def bake_animation(self, start_frame, end_frame):
        # Baking the animation to finalize all transformations
        bpy.ops.nla.bake(
            frame_start=start_frame,
            frame_end=end_frame,
            only_selected=True,
            visual_keying=True,
            clear_constraints=True,
            bake_types={"POSE"},
        )

    def read_anim_start_end(self, armature):
        action = armature.animation_data.action
        return int(action.frame_range[0]), int(action.frame_range[1])
