"""MediaPipe video landmark extraction.

``MediapipeExtractor`` reads a video frame-by-frame and produces a pandas
DataFrame of engineered features (pairwise distances, joint angles, and
hand-body interaction distances/angles) that downstream models consume to
predict bone-rotation quaternions.

Uses MediaPipe's current ``tasks`` API (``PoseLandmarker`` /
``HandLandmarker``) which requires the ``.task`` model files. If a model is
missing, it is downloaded automatically from Google's MediaPipe storage.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python.vision import HandLandmarksConnections, PoseLandmarksConnections
from tqdm import tqdm

POSE_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]

_MODEL_URLS: dict[str, str] = {
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def _ensure_model(path: str) -> str:
    """Return ``path``, downloading the model into the repo root if missing."""
    if os.path.isfile(path):
        return path
    filename = os.path.basename(path)
    url = _MODEL_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, path)
    return path


class MediapipeExtractor:
    def __init__(
        self, pose_task_path: str | None = None, hand_task_path: str | None = None
    ) -> None:
        self.pose_task_path = _ensure_model(
            pose_task_path
            or os.path.join(os.path.curdir, "pose_landmarker_heavy.task")
        )
        self.hand_task_path = _ensure_model(
            hand_task_path or os.path.join(os.path.curdir, "hand_landmarker.task")
        )

        base_options = mp.tasks.BaseOptions
        running_mode = mp.tasks.vision.RunningMode

        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.HandLandmarker = mp.tasks.vision.HandLandmarker

        self.pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=self.pose_task_path),
            output_segmentation_masks=True,
            running_mode=running_mode.IMAGE,
        )
        self.hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=self.hand_task_path),
            running_mode=running_mode.IMAGE,
            num_hands=2,
        )

    def run(
        self,
        video_path: str,
        output_csv_path: str | None = None,
        output_video_path: str | None = None,
        frame_count: int | None = None,
    ) -> list[dict[str, float]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = None
        if output_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height)
            )

        if frame_count is None:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        poses: list[dict[str, float]] = []

        with (
            self.PoseLandmarker.create_from_options(self.pose_options) as landmarker,
            self.HandLandmarker.create_from_options(self.hand_options) as hand_landmarker,
        ):
            for _ in tqdm(range(frame_count), desc="Extracting landmarks"):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                results = landmarker.detect(mp_image)
                hand_results = hand_landmarker.detect(mp_image)

                if not results.pose_world_landmarks:
                    continue

                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks[0]])

                detected_hands = len(hand_results.hand_world_landmarks)
                right_hand_landmarks = None
                left_hand_landmarks = None
                if detected_hands > 0:
                    right_hand_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_results.hand_world_landmarks[0]]
                    )
                if detected_hands > 1:
                    left_hand_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_results.hand_world_landmarks[1]]
                    )

                pose: dict[str, float] = {}
                if results.pose_landmarks:
                    pose.update(self._get_distance_features(landmarks, "body"))
                    pose.update(self._get_angle_connected_points(landmarks))

                    if detected_hands > 1:
                        self._add_hand_features(
                            pose, landmarks, right_hand_landmarks, left_hand_landmarks
                        )
                    elif detected_hands > 0:
                        self._add_single_hand_features(pose, landmarks, right_hand_landmarks)
                poses.append(pose)

                if video_writer is not None:
                    annotated = self._draw_landmarks_on_image(frame.copy(), results)
                    annotated = self._draw_hand_landmarks_on_image(annotated, hand_results)
                    video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        cap.release()
        if video_writer is not None:
            video_writer.release()

        if output_csv_path is not None:
            frame_df = pd.DataFrame(poses)
            frame_df = frame_df.ffill().bfill()
            frame_df.to_csv(output_csv_path, index=False)

        return poses

    def extract_pose_landmarks(
        self,
        video_path: str,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Extract raw 3D world-space pose landmarks for visualisation.

        Returns ``(landmarks, connections)`` where ``landmarks`` has shape
        ``(T, 33, 3)`` (one per detected frame, downsampled by ``stride``)
        and ``connections`` is the MediaPipe POSE_CONNECTIONS edge list.

        This is lighter than :meth:`run` (no feature math, no hand tracking)
        and is used by the Gradio app for the animated 3D skeleton.
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total = min(total, max_frames * stride)

        frames: list[np.ndarray] = []

        with self.PoseLandmarker.create_from_options(self.pose_options) as landmarker:
            for idx in tqdm(range(total)):
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride != 0:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                results = landmarker.detect(mp_image)
                if not results.pose_world_landmarks:
                    continue
                pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks[0]])
                frames.append(pts)

        cap.release()

        if not frames:
            return np.zeros((0, 33, 3)), POSE_CONNECTIONS

        return np.stack(frames), POSE_CONNECTIONS

    # ------------------------------------------------------------------ #
    # Hands
    # ------------------------------------------------------------------ #
    def _add_single_hand_features(self, pose, landmarks, hand_landmarks):
        # Decide left vs right by proximity to wrist landmarks 15 (left) / 16 (right).
        if np.linalg.norm(hand_landmarks[0] - landmarks[16]) < np.linalg.norm(
            hand_landmarks[0] - landmarks[15]
        ):
            mark = "left"
            body_point = 13
        else:
            mark = "right"
            body_point = 14
        self._apply_hand_features(pose, landmarks, hand_landmarks, mark, body_point)

    def _add_hand_features(self, pose, landmarks, right_hand_landmarks, left_hand_landmarks):
        if np.linalg.norm(right_hand_landmarks[0] - landmarks[16]) < np.linalg.norm(
            right_hand_landmarks[0] - landmarks[15]
        ):
            self._apply_hand_features(pose, landmarks, right_hand_landmarks, "left", 13)
            self._apply_hand_features(pose, landmarks, left_hand_landmarks, "right", 14)
        else:
            self._apply_hand_features(pose, landmarks, left_hand_landmarks, "left", 13)
            self._apply_hand_features(pose, landmarks, right_hand_landmarks, "right", 14)

    def _apply_hand_features(self, pose, body_landmarks, hand_landmarks, mark, body_point):
        pose.update(self._get_distance_features(hand_landmarks, mark))
        pose.update(self._get_hand_angle_connected_points(hand_landmarks, mark))
        pose.update(self._calculate_hand_body_distances(body_landmarks, hand_landmarks, mark))
        pose.update(
            self._calculate_body_hand_angles(body_landmarks, hand_landmarks, body_point, mark)
        )

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calculate_angle(a, b, c) -> float:
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return float(np.degrees(angle))

    @staticmethod
    def _get_distance_features(landmarks: np.ndarray, mark: str) -> dict[str, float]:
        pose: dict[str, float] = {}
        for i in range(len(landmarks)):
            for j in range(i + 1, len(landmarks)):
                pose[f"{mark}_distance_{i}_{j}"] = float(
                    np.linalg.norm(landmarks[i] - landmarks[j])
                )
        return pose

    def _get_angle_connected_points(self, landmarks: np.ndarray) -> dict[str, float]:
        pose: dict[str, float] = {}
        for connection in POSE_CONNECTIONS:
            a = landmarks[connection[0]]
            b = landmarks[connection[1]]
            for connection2 in POSE_CONNECTIONS:
                if connection2[0] == connection[1] and connection2[1] != connection[0]:
                    c = landmarks[connection2[1]]
                    pose[f"body_angle_{connection[0]}_{connection[1]}_{connection2[1]}"] = (
                        self._calculate_angle(a, b, c)
                    )
                elif connection2[1] == connection[1] and connection2[0] != connection[0]:
                    c = landmarks[connection2[0]]
                    pose[f"body_angle_{connection[0]}_{connection[1]}_{connection2[0]}"] = (
                        self._calculate_angle(a, b, c)
                    )
        return pose

    def _get_hand_angle_connected_points(
        self, hand_landmarks: np.ndarray, mark: str
    ) -> dict[str, float]:
        hand_pose: dict[str, float] = {}
        for connection in HAND_CONNECTIONS:
            a = hand_landmarks[connection[0]]
            b = hand_landmarks[connection[1]]
            for connection2 in HAND_CONNECTIONS:
                if connection2[0] == connection[1] and connection2[1] != connection[0]:
                    c = hand_landmarks[connection2[1]]
                    hand_pose[
                        f"{mark}_hand_angle_{connection[0]}_{connection[1]}_{connection2[1]}"
                    ] = self._calculate_angle(a, b, c)
                elif connection2[1] == connection[1] and connection2[0] != connection[0]:
                    c = hand_landmarks[connection2[0]]
                    hand_pose[
                        f"{mark}_hand_angle_{connection[0]}_{connection[1]}_{connection2[0]}"
                    ] = self._calculate_angle(a, b, c)
        return hand_pose

    @staticmethod
    def _calculate_hand_body_distances(
        body_landmarks, hand_landmarks, mark: str
    ) -> dict[str, float]:
        distances: dict[str, float] = {}
        for i, body_landmark in enumerate(body_landmarks):
            for j, hand_landmark in enumerate(hand_landmarks):
                distance = np.linalg.norm(body_landmark - hand_landmark)
                distances[f"distance_body_{i}_{mark}_hand_{j}"] = float(distance)
        return distances

    def _calculate_body_hand_angles(
        self, body_landmarks, hand_landmarks, body_point: int, mark: str
    ) -> dict[str, float]:
        angles: dict[str, float] = {}
        body_anchor = body_landmarks[body_point]
        hand_anchor = hand_landmarks[0]

        connected_points = [c[1] for c in HAND_CONNECTIONS if c[0] == 0]
        connected_points += [c[0] for c in HAND_CONNECTIONS if c[1] == 0]
        connected_points = sorted(set(connected_points))

        for i in connected_points:
            hand_target = hand_landmarks[i]
            angle = self._calculate_angle(body_anchor, hand_anchor, hand_target)
            angles[f"angle_body_{body_point}_{mark}_hand_0_{i}"] = angle
        return angles

    # ------------------------------------------------------------------ #
    # Visualisation (pure cv2, MediaPipe tasks API has no drawing utils)
    # ------------------------------------------------------------------ #
    def _draw_landmarks_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)
        h, w, _ = annotated_image.shape
        for pose_landmarks in detection_result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
            for a, b in POSE_CONNECTIONS:
                cv2.line(annotated_image, pts[a], pts[b], (0, 200, 0), 2)
            for p in pts:
                cv2.circle(annotated_image, p, 3, (0, 0, 255), -1)
        return annotated_image

    def _draw_hand_landmarks_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)
        h, w, _ = annotated_image.shape
        for hand_landmarks in detection_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(annotated_image, pts[a], pts[b], (200, 0, 0), 2)
            for p in pts:
                cv2.circle(annotated_image, p, 3, (255, 0, 0), -1)
        return annotated_image
