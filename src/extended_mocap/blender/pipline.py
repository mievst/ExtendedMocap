"""End-to-end Blender pipeline: retarget Mixamo -> AccuRig, render, extract.

Runs inside Blender. Orchestrates the three class-based stages:
1. ``convert_mixamo_to_blend`` retargets source armatures to CC_Base.
2. ``render.BlenderAnimationProcessor`` renders each animation to MP4.
3. ``extract.ArmatureDataExtractor`` exports per-frame quaternion CSVs.
"""

from __future__ import annotations

import logging

import bpy

from .extract import ArmatureDataExtractor
from .render import BlenderAnimationProcessor
from .retargeting import ArmatureRetargeting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    source_collection_name = "mixamo"
    target_collection_name = "AccuRig"
    model_name = "party-m-0001"
    example_armature_name = "idle_251087"

    # Complete Mixamo -> CC_Base mapping (24 major bones).
    bone_mapping = {
        "mixamorig:Hips": "CC_Base_Hip",
        "mixamorig:Spine": "CC_Base_Spine01",
        "mixamorig:Spine1": "CC_Base_Spine02",
        "mixamorig:Neck": "CC_Base_NeckTwist01",
        "mixamorig:Head": "CC_Base_Head",
        "mixamorig:LeftArm": "CC_Base_L_Upperarm",
        "mixamorig:LeftForeArm": "CC_Base_L_Forearm",
        "mixamorig:LeftHand": "CC_Base_L_Hand",
        "mixamorig:RightArm": "CC_Base_R_Upperarm",
        "mixamorig:RightForeArm": "CC_Base_R_Forearm",
        "mixamorig:RightHand": "CC_Base_R_Hand",
        "mixamorig:LeftUpLeg": "CC_Base_L_Thigh",
        "mixamorig:LeftLeg": "CC_Base_L_Calf",
        "mixamorig:RightUpLeg": "CC_Base_R_Thigh",
        "mixamorig:RightLeg": "CC_Base_R_Calf",
    }

    target_collection = _ensure_collection(target_collection_name)
    temp_collection = _ensure_collection("temp")
    source_collection = bpy.data.collections[source_collection_name]

    example_armature = bpy.context.scene.objects.get(example_armature_name)
    if example_armature is None:
        raise ValueError(f"Example armature '{example_armature_name}' not found in scene")

    for source_armature in source_collection.objects:
        copied = source_armature.copy()
        copied.animation_data = source_armature.animation_data
        temp_collection.objects.link(copied)

        target = example_armature.copy()
        target.animation_data = example_armature.animation_data
        target.name = "AccuRig_" + source_armature.name
        target_collection.objects.link(target)

        logger.debug("Retargeting %s -> %s", source_armature.name, target.name)
        retargeter = ArmatureRetargeting(copied, target, bone_mapping)
        retargeter.retarget_animation()

    processor = BlenderAnimationProcessor(model_name, target_collection_name)
    processor.process_animations()

    extractor = ArmatureDataExtractor(target_collection_name)
    extractor.extract_data()
    logger.info("Pipeline completed")


def _ensure_collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        logger.info("Created collection: %s", name)
    return collection


if __name__ == "__main__":
    main()
