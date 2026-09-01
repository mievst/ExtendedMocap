import logging
import os

import bpy
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class ArmatureDataExtractor:
    def __init__(self, collection_name):
        self.collection = bpy.data.collections[collection_name]
        self.data = {}
        logging.info(f"Initialized ArmatureDataExtractor for collection: {collection_name}")

    def clamp_location(self, location):
        """Clamp the location coordinates to be within the unit cube bounds."""
        clamped_location = tuple(max(min(coord, 0.5), -0.5) for coord in location)
        return clamped_location

    def extract_data(self):
        for obj in self.collection.objects:
            armature_name = obj.name
            file_name = os.path.basename(bpy.data.filepath)
            export_name = f"{file_name}_{obj.name}.csv"
            logging.info(f"Processing armature: {armature_name}, export name: {export_name}")
            bpy.ops.object.mode_set(mode="OBJECT")
            armature = bpy.context.scene.objects.get(armature_name)
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature
            frames = int(bpy.context.object.animation_data.action.frame_range[-1]) + 1
            scene = bpy.context.scene
            armature = armature.pose
            bpy.ops.object.mode_set(mode="POSE")

            logging.info(f"Number of frames: {frames}")

            self.initialize_data_storage()
            bpy.context.scene.frame_set(0)
            initial_location = armature.bones["CC_Base_Hip"].location
            offset = tuple(-coord for coord in initial_location)
            for frame in tqdm(range(frames)):
                scene.frame_set(frame)
                self.process_frame(armature, offset)

            self.save_data(export_name)

    def initialize_data_storage(self):
        logging.debug("Initialized data storage for bones.")
        self.data = {}
        self.loc_bones = ["CC_Base_Hip"]
        self.rot_bones = set(
            [
                "CC_Base_Hip",
                "CC_Base_FacialBone",
                "CC_Base_NeckTwist02",
                "CC_Base_NeckTwist01",
                "CC_Base_Pelvis",
                "CC_Base_Waist",
                "CC_Base_Spine01",
                "CC_Base_Spine02",
                "CC_Base_R_Clavicle",
                "CC_Base_L_Clavicle",
                "CC_Base_R_Upperarm",
                "CC_Base_L_Upperarm",
                "CC_Base_R_Forearm",
                "CC_Base_L_Forearm",
                "CC_Base_R_Hand",
                "CC_Base_L_Hand",
                "CC_Base_L_Mid1",
                "CC_Base_L_Mid2",
                "CC_Base_L_Mid3",
                "CC_Base_L_Index1",
                "CC_Base_L_Index2",
                "CC_Base_L_Index3",
                "CC_Base_L_Ring1",
                "CC_Base_L_Ring2",
                "CC_Base_L_Ring3",
                "CC_Base_L_Pinky1",
                "CC_Base_L_Pinky2",
                "CC_Base_L_Pinky3",
                "CC_Base_L_Thumb1",
                "CC_Base_L_Thumb2",
                "CC_Base_L_Thumb3",
                "CC_Base_R_Mid1",
                "CC_Base_R_Mid2",
                "CC_Base_R_Mid3",
                "CC_Base_R_Ring1",
                "CC_Base_R_Ring2",
                "CC_Base_R_Ring3",
                "CC_Base_R_Thumb1",
                "CC_Base_R_Thumb2",
                "CC_Base_R_Thumb3",
                "CC_Base_R_Index1",
                "CC_Base_R_Index2",
                "CC_Base_R_Index3",
                "CC_Base_R_Pinky1",
                "CC_Base_R_Pinky2",
                "CC_Base_R_Pinky3",
                "CC_Base_R_Thigh",
                "CC_Base_R_Calf",
                "CC_Base_R_Foot",
                "CC_Base_L_Thigh",
                "CC_Base_L_Calf",
                "CC_Base_L_Foot",
                "CC_Base_BoneRoot",
                # Add all other bones as needed
            ]
        )

        for name in self.rot_bones:
            for q in range(4):
                self.data[f"{name}.Q{q}"] = []

        for name in self.loc_bones:
            for t in range(3):
                self.data[f"{name}.T{t}"] = []

    def process_frame(self, armature, offset):
        for name in self.rot_bones:
            bone = armature.bones[name]
            bone.rotation_mode = "QUATERNION"

            for q in range(4):
                # print(i, bone.rotation_quaternion[q], len(data[f"{name}.Q{q}"]))

                self.data[f"{name}.Q{q}"].append(bone.rotation_quaternion[q])

        for name in self.loc_bones:
            location = armature.bones[name].location
            new_location = tuple(coord + offset[i] for i, coord in enumerate(location))
            new_location = self.clamp_location(new_location)
            for t in range(3):
                self.data[f"{name}.T{t}"].append(new_location[t])

    def save_data(self, export_name):
        df = pd.DataFrame(self.data)
        df.to_csv(
            os.path.join(os.path.dirname(bpy.data.filepath), "data", "mocap", "csv", export_name),
            index=False,
        )
        logging.info(f"Data saved to {export_name}")


# Usage
# extractor = ArmatureDataExtractor('AccuRig')
# extractor.extract_data()
