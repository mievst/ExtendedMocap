import copy
import math

import bpy
from mathutils import Matrix, Vector

RETARGET_ID = "_MIEVST_RETARGET"


class ArmatureRetargeting:
    def __init__(self, source_armature, target_armature, bone_mapping):
        self.source_armature = source_armature
        self.target_armature = target_armature
        self.bone_mapping = bone_mapping

    def retarget_animation(self):
        root_bones = self.find_root_bones()
        self.set_active(self.target_armature)
        bpy.ops.object.mode_set(mode="OBJECT")
        self.set_active(self.source_armature)
        bpy.ops.object.mode_set(mode="OBJECT")
        if self.target_armature.data.users > 1:
            self.target_armature.data = self.target_armature.data.copy()

        if self.source_armature.data.users > 1:
            self.source_armature.data = self.source_armature.data.copy()

        self.source_armature.data.pose_position = "POSE"
        self.target_armature.data.pose_position = "POSE"

        # Save and reset the current pose position of both armatures if rest position should be used
        pose_source, pose_target = {}, {}
        pose_source = self.get_and_reset_pose_rotations(self.source_armature)
        pose_target = self.get_and_reset_pose_rotations(self.target_armature)

        # Auto scaling
        source_scale = None
        # Clean source animation
        # TODO: This causes issues when all Hip bone data is on the armature itself
        self.clean_animation(self.source_armature)

        # Scale the source armature to fit the target armature
        source_scale = copy.deepcopy(self.source_armature.scale)
        self.scale_armature(self.source_armature, self.target_armature, root_bones)

        # Duplicate source armature to apply transforms to the animation
        armature_source_original = self.source_armature
        armature_source = self.copy_rest_pose(bpy.context, self.source_armature)

        # Save transforms of target armature
        rotation_mode = self.target_armature.rotation_mode
        self.target_armature.rotation_mode = "QUATERNION"
        rotation = copy.deepcopy(self.target_armature.rotation_quaternion)
        location = copy.deepcopy(self.target_armature.location)

        # Apply transforms of the target armature
        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(self.target_armature)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        bpy.ops.object.mode_set(mode="EDIT")

        # Create a transformation dict of all bones of the target armature and unselect all bones
        bone_transforms = {}
        for bone in bpy.context.object.data.edit_bones:
            bone.select = False
            bone_transforms[bone.name] = (
                self.source_armature.matrix_world.inverted() @ bone.head.copy(),
                self.source_armature.matrix_world.inverted() @ bone.tail.copy(),
                self.mat3_to_vec_roll(
                    self.source_armature.matrix_world.inverted().to_3x3() @ bone.matrix.to_3x3()
                ),
            )  # Head loc, tail loc, bone roll

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(armature_source)
        bpy.ops.object.mode_set(mode="EDIT")

        # Recreate bones from target armature in source armature
        for key, val in self.bone_mapping.items():
            bone_source = armature_source.data.edit_bones.get(key)

            # Recreate target bone
            bone_new = armature_source.data.edit_bones.new(val + RETARGET_ID)
            bone_new.head, bone_new.tail, bone_new.roll = bone_transforms[val]
            bone_new.parent = bone_source

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        # Add constraints to target armature and select the bones for animation
        for key, val in self.bone_mapping.items():
            bone_target = self.target_armature.pose.bones.get(val)

            # Add constraints
            constraint = bone_target.constraints.new("COPY_ROTATION")
            constraint.name += RETARGET_ID
            constraint.target = armature_source
            constraint.subtarget = val + RETARGET_ID

            if bone_target.name in root_bones:
                constraint = bone_target.constraints.new("COPY_LOCATION")
                constraint.name += RETARGET_ID
                constraint.target = armature_source
                constraint.subtarget = key

            # Select the bone for animation
            self.target_armature.data.bones.get(val).select = True

        # Bake the animation to the target armature
        self.bake_animation(armature_source, self.target_armature, root_bones)

        # Delete the duplicate helper armature
        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(armature_source)
        bpy.data.actions.remove(armature_source.animation_data.action)
        bpy.ops.object.delete()

        # Change armature source back to original
        armature_source = armature_source_original

        # Change action name
        if armature_source.animation_data.action is not None:
            self.target_armature.animation_data.action.name = (
                armature_source.animation_data.action.name + " Retarget"
            )

        # Remove constraints from target armature
        for bone in self.target_armature.pose.bones:
            for constraint in bone.constraints:
                if RETARGET_ID in constraint.name:
                    bone.constraints.remove(constraint)

        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(self.target_armature)

        # Reset target armature transforms to old state
        self.target_armature.rotation_quaternion = rotation
        self.target_armature.location = location

        self.target_armature.rotation_quaternion.w = -self.target_armature.rotation_quaternion.w
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        self.target_armature.rotation_quaternion = rotation
        self.target_armature.rotation_mode = rotation_mode

        # Reset source armature scale
        if source_scale:
            armature_source.scale = source_scale

        # Reset pose positions to old state
        # self.load_pose_rotations(armature_source, pose_source)
        # self.load_pose_rotations(armature_target, pose_target)

        bpy.ops.object.select_all(action="DESELECT")

    def find_root_bones(self):
        # Find all root bones
        root_bones = []
        for bone in self.target_armature.pose.bones:
            if not bone.parent:
                root_bones.append(bone)

        # Find animated root bones
        root_bones_animated = []
        target_bones = [val for key, val in self.bone_mapping.items()]
        while root_bones:
            for bone in copy.copy(root_bones):
                root_bones.remove(bone)
                if bone.name in target_bones:
                    root_bones_animated.append(bone.name)
                else:
                    for bone_child in bone.children:
                        root_bones.append(bone_child)
        return root_bones_animated

    def set_active(self, obj):
        obj.select_set(True)
        obj.hide_set(False)
        bpy.context.view_layer.objects.active = obj

    def get_and_reset_pose_rotations(self, armature):
        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(armature)
        bpy.ops.object.mode_set(mode="POSE")

        # Save rotations
        pose_rotations = {}
        for bone in armature.pose.bones:
            if bone.rotation_mode == "QUATERNION":
                pose_rotations[bone.name] = copy.deepcopy(bone.rotation_quaternion)
                bone.rotation_quaternion = (1, 0, 0, 0)
            else:
                pose_rotations[bone.name] = copy.deepcopy(bone.rotation_euler)
                bone.rotation_euler = (0, 0, 0)

        # Reset rotations
        # bpy.ops.pose.rot_clear()
        bpy.ops.object.mode_set(mode="OBJECT")

        return pose_rotations

    def clean_animation(self, armature_source):
        if armature_source.animation_data is None or armature_source.animation_data.action is None:
            print("No animation data found.")
            return

        deletable_fcurves = ["location", "rotation_euler", "rotation_quaternion", "scale"]
        for fcurve in armature_source.animation_data.action.fcurves:
            if fcurve.data_path in deletable_fcurves:
                armature_source.animation_data.action.fcurves.remove(fcurve)

    def scale_armature(self, armature_source, armature_target, root_bones):
        source_min = None
        source_min_root = None
        target_min = None
        target_min_root = None

        for key, val in self.bone_mapping.items():
            bone_source = armature_source.pose.bones.get(key)
            bone_target = armature_target.pose.bones.get(val)

            bone_source_z = (armature_source.matrix_world @ bone_source.head)[2]
            bone_target_z = (armature_target.matrix_world @ bone_target.head)[2]

            if val in root_bones:
                if source_min_root is None or source_min_root > bone_source_z:
                    source_min_root = bone_source_z
                if target_min_root is None or target_min_root > bone_target_z:
                    target_min_root = bone_target_z

            if source_min is None or source_min > bone_source_z:
                source_min = bone_source_z
            if target_min is None or target_min > bone_target_z:
                target_min = bone_target_z

        source_height = source_min_root - source_min
        target_height = target_min_root - target_min

        if not source_height or not target_height:
            print("No scaling needed")
            return

        scale_factor = target_height / source_height
        armature_source.scale *= scale_factor

    def copy_rest_pose(self, context, armature_source):
        # make sure auto keyframe is disabled, leads to issues
        context.scene.tool_settings.use_keyframe_insert_auto = False

        temp_collection = bpy.data.collections["temp"]

        # ensure the source armature selection
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(armature_source)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Duplicate the source armature
        bpy.ops.object.duplicate_move(
            OBJECT_OT_duplicate={"linked": False, "mode": "TRANSLATION"},
            TRANSFORM_OT_translate={
                "value": (0, 0, 0),
                "constraint_axis": (False, True, False),
                "mirror": False,
                "snap": False,
                "remove_on_cancel": False,
                "release_confirm": False,
            },
        )

        # Set name of the copied source armature
        source_armature_copy = armature_source.copy()
        source_armature_copy.name = armature_source.name + "_copy"
        temp_collection.objects.link(source_armature_copy)

        bpy.ops.object.select_all(action="DESELECT")
        self.set_active(source_armature_copy)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.mode_set(mode="POSE")

        # Apply transforms of the new source armature. Unlink action temporarily to prevent warning in console
        action_tmp = armature_source.animation_data.action
        source_armature_copy.animation_data.action = None
        bpy.ops.pose.armature_apply()
        source_armature_copy.animation_data.action = action_tmp

        # Mimic the animation of the original source armature by adding constraints to the bones.
        # -> the new armature has the exact same animation but with applied transforms
        for bone in source_armature_copy.pose.bones:
            constraint = bone.constraints.new("COPY_TRANSFORMS")
            constraint.name = bone.name
            constraint.target = armature_source
            constraint.subtarget = bone.name

        bpy.ops.object.mode_set(mode="OBJECT")

        return source_armature_copy

    def mat3_to_vec_roll(self, mat):
        vecmat = self.vec_roll_to_mat3(mat.col[1], 0)
        vecmatinv = vecmat.inverted()
        rollmat = vecmatinv @ mat
        roll = math.atan2(rollmat[0][2], rollmat[2][2])
        return roll

    def vec_roll_to_mat3(self, vec, roll):
        target = Vector((0, 0.1, 0))
        nor = vec.normalized()
        axis = target.cross(nor)
        if axis.dot(axis) > 0.0000000001:
            axis.normalize()
            theta = target.angle(nor)
            bMatrix = Matrix.Rotation(theta, 3, axis)
        else:
            updown = 1 if target.dot(nor) > 0 else -1
            bMatrix = Matrix.Scale(updown, 3)
            bMatrix[2][2] = 1.0

        rMatrix = Matrix.Rotation(roll, 3, nor)
        mat = rMatrix @ bMatrix
        return mat

    def bake_animation(self, armature_source, armature_target, root_bones):
        frame_split = 25
        frame_start, frame_end = self.read_anim_start_end(armature_source)
        frame_start, frame_end = int(frame_start), int(frame_end)
        self.set_active(armature_target)

        actions_all = []

        # Setup loading bar
        current_step = 0
        steps = int((frame_end - frame_start) / frame_split) + 1
        wm = bpy.context.window_manager
        wm.progress_begin(current_step, steps)

        import time

        start_time = time.time()

        # Bake the animation in parts because multiple short parts are processed much faster than one long animation
        for frame in range(frame_start, frame_end + 2, frame_split):
            start = frame
            end = frame + frame_split - 1
            end = min(end, frame_end)
            if start > end:
                continue

            # Bake animation part
            bpy.ops.nla.bake(
                frame_start=start,
                frame_end=end,
                visual_keying=True,
                only_selected=False,
                use_current_action=False,
                bake_types={"POSE"},
            )

            # Rename animation part
            armature_target.animation_data.action.name = "RSL_RETARGETING_" + str(frame)

            actions_all.append(armature_target.animation_data.action)

            current_step += 1
            if steps != current_step:
                wm.progress_update(current_step)

        if not actions_all:
            return

        # Count all keys for all data_paths
        key_counts = {}
        for action in actions_all:
            for fcurve in action.fcurves:
                key = fcurve.data_path + str(fcurve.array_index)
                if not key_counts.get(key):
                    key_counts[key] = 0
                key_counts[key] += len(fcurve.keyframe_points)

        # Create new action
        action_final = bpy.data.actions.new(name="RSL_RETARGETING_FINAL")
        action_final.use_fake_user = True
        armature_target.animation_data_create().action = action_final

        # Put all baked animations parts back together into one
        print_i = 0
        for fcurve in actions_all[0].fcurves:
            if fcurve.data_path.endswith("scale"):
                continue
            if fcurve.data_path.endswith("location"):
                bone_name = fcurve.data_path.split('"')
                if len(bone_name) != 3:
                    continue
                if bone_name[1] not in root_bones:
                    continue

            curve_final = action_final.fcurves.new(
                data_path=fcurve.data_path, index=fcurve.array_index, action_group=fcurve.group.name
            )
            keyframe_points = curve_final.keyframe_points
            keyframe_points.add(key_counts[fcurve.data_path + str(fcurve.array_index)])

            index = 0
            for action in actions_all:
                fcruve_to_add = action.fcurves.find(
                    data_path=fcurve.data_path, index=fcurve.array_index
                )

                for kp in fcruve_to_add.keyframe_points:
                    keyframe_points[index].co.x = kp.co.x
                    keyframe_points[index].co.y = kp.co.y
                    keyframe_points[index].interpolation = "LINEAR"
                    index += 1

            print_i += 1

        # Clean up animation. Delete all keyframes the use the same value as the previous and next one
        for fcurve in action_final.fcurves:
            if len(fcurve.keyframe_points) <= 2:
                continue

            kp_pre_pre = fcurve.keyframe_points[0]
            kp_pre = fcurve.keyframe_points[1]

            kp_to_delete = []
            for kp in fcurve.keyframe_points[2:]:
                if round(kp_pre_pre.co.y, 5) == round(kp_pre.co.y, 5) == round(kp.co.y, 5):
                    kp_to_delete.append(kp_pre)
                kp_pre_pre = kp_pre
                kp_pre = kp

            for kp in reversed(kp_to_delete):
                fcurve.keyframe_points.remove(kp)

        # Delete all baked animation parts, only the combined one is needed
        for action in actions_all:
            bpy.data.actions.remove(action)

        print("Retargeting Time:", round(time.time() - start_time, 2), "seconds")
        wm.progress_end()

    def read_anim_start_end(self, armature):
        action = armature.animation_data.action
        return int(action.frame_range[0]), int(action.frame_range[1])
