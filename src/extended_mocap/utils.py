"""Shared constants and helpers for the MediaPipe -> quaternion pipeline."""

from __future__ import annotations


def get_columns_names(segments: set[str], quat_columns: list[str]) -> list[str]:
    """Return all quaternion columns whose bone name starts with any of the
    segment prefixes.

    Segments are sets of CC_Base_* bone names. The returned list preserves the
    order of the target ``quat_columns``.
    """
    columns: list[str] = []
    for seg in segments:
        for item in quat_columns:
            if str(item).startswith(seg):
                columns.append(item)
    return columns


# Convenience groupings of CC_Base_* bones used to train/infer by body part.
BODY_BONES = {
    "CC_Base_L_Clavicle",
    "CC_Base_L_Upperarm",
    "CC_Base_L_Forearm",
    "CC_Base_R_Clavicle",
    "CC_Base_R_Upperarm",
    "CC_Base_R_Forearm",
    "CC_Base_NeckTwist02",
    "CC_Base_NeckTwist01",
    "CC_Base_Waist",
    "CC_Base_Spine01",
    "CC_Base_Spine02",
    "CC_Base_R_Thigh",
    "CC_Base_R_Calf",
    "CC_Base_R_Foot",
    "CC_Base_L_Thigh",
    "CC_Base_L_Calf",
    "CC_Base_L_Foot",
}

LEFT_HAND_BONES = {
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
    "CC_Base_L_Hand",
}

RIGHT_HAND_BONES = {
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
    "CC_Base_R_Hand",
}
