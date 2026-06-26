"""
Face detection, landmarking, and alignment.

We use Google's MediaPipe (BlazeFace for detection + FaceMesh for
landmarks) here. The more common choice for this kind of work is
RetinaFace, and honestly it's a great detector, but it has one practical
problem for us: its weights live as a separate download, and our build
and CI machines can't always reach out to grab files like that. MediaPipe
ships its weights inside the pip package, so once you've pip-installed it,
detection just works, fully offline, with nothing extra to fetch. It's
also Apache-2.0, so licensing is a non-issue.

If we ever move to an environment where pulling RetinaFace's weights is
easy, swapping it in is straightforward: implement the same FaceDetector
interface and return the same FaceObservation, and nothing else in the
pipeline has to change.

Whatever detector sits behind it, the output stays the same: a bounding
box, the landmarks, and a 112x112 aligned face crop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

ALIGN_SIZE = 112

# Canonical (template) eye/nose/mouth positions for 112x112 alignment,
# the same target geometry ArcFace's preprocessing uses, so a swapped-in
# ArcFace embedder downstream needs no changes to this module.
_TEMPLATE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)

# MediaPipe FaceMesh landmark indices for the 5 ArcFace-style points
_LEFT_EYE_IDX = 33
_RIGHT_EYE_IDX = 263
_NOSE_IDX = 1
_LEFT_MOUTH_IDX = 61
_RIGHT_MOUTH_IDX = 291

# Indices used for eye-aspect-ratio (blink) computation, liveness module
_LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_EAR_IDX = [263, 387, 385, 362, 380, 373]

# Indices roughly spanning face outline, used for yaw/pitch estimate
_FACE_OUTLINE_IDX = [10, 152, 234, 454]  # top, chin, left cheek, right cheek


@dataclass
class FaceObservation:
    """Everything the rest of the pipeline needs from one detected face."""
    bbox_xywh_rel: tuple        # (x, y, w, h) relative to frame, 0..1
    detection_score: float
    landmarks_px: np.ndarray    # (478, 2) pixel coords, FaceMesh order
    aligned_crop: np.ndarray    # (112, 112, 3) BGR uint8
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    left_ear: float              # eye-aspect-ratio, left eye
    right_ear: float             # eye-aspect-ratio, right eye
    frame_shape: tuple           # (h, w) of source frame
    source_frame_bgr: np.ndarray  # the full source frame (needed by the anti-spoof model's context crop)

    @property
    def bbox_xywh_px(self) -> tuple:
        """Absolute pixel bbox [x, y, w, h] derived from the relative bbox."""
        h, w = self.frame_shape
        x, y, bw, bh = self.bbox_xywh_rel
        return [int(x * w), int(y * h), int(bw * w), int(bh * h)]


class FaceDetector:
    """Thin wrapper around MediaPipe FaceDetection + FaceMesh.

    Designed so it can be dropped for a RetinaFace-backed implementation
    later: callers only depend on `detect_largest_face()` and the
    `FaceObservation` dataclass it returns.
    """

    def __init__(self, min_detection_confidence: float = 0.6):
        self._mp_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # "full range" model — better for varied capture distance
            min_detection_confidence=min_detection_confidence,
        )
        self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def close(self):
        self._mp_detector.close()
        self._mp_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def detect_largest_face(self, frame_bgr: np.ndarray) -> Optional[FaceObservation]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        det_result = self._mp_detector.process(rgb)
        if not det_result.detections:
            return None

        # Pick the largest detection (closest/most prominent face) — guards
        # against capturing a bystander in the background (spec FR-3 implies
        # single-subject capture at an enrollment/identify station).
        best = max(
            det_result.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        rbb = best.location_data.relative_bounding_box
        bbox = (rbb.xmin, rbb.ymin, rbb.width, rbb.height)
        score = float(best.score[0]) if best.score else 0.0

        mesh_result = self._mp_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            return None
        lm = mesh_result.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        aligned = self._align(frame_bgr, pts)
        if aligned is None:
            return None

        yaw, pitch, roll = self._estimate_pose(pts, w, h)
        left_ear = self._eye_aspect_ratio(pts, _LEFT_EYE_EAR_IDX)
        right_ear = self._eye_aspect_ratio(pts, _RIGHT_EYE_EAR_IDX)

        return FaceObservation(
            bbox_xywh_rel=bbox,
            detection_score=score,
            landmarks_px=pts,
            aligned_crop=aligned,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            left_ear=left_ear,
            right_ear=right_ear,
            frame_shape=(h, w),
            source_frame_bgr=frame_bgr,
        )

    @staticmethod
    def _align(frame_bgr: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
        src = np.array(
            [
                pts[_LEFT_EYE_IDX],
                pts[_RIGHT_EYE_IDX],
                pts[_NOSE_IDX],
                pts[_LEFT_MOUTH_IDX],
                pts[_RIGHT_MOUTH_IDX],
            ],
            dtype=np.float32,
        )
        # Similarity transform (umeyama) from source 5-pt to canonical template
        transform = _umeyama(src, _TEMPLATE_LANDMARKS)
        if transform is None:
            return None
        aligned = cv2.warpAffine(
            frame_bgr, transform, (ALIGN_SIZE, ALIGN_SIZE), flags=cv2.INTER_LINEAR
        )
        return aligned

    @staticmethod
    def _estimate_pose(pts: np.ndarray, w: int, h: int):
        top = pts[_FACE_OUTLINE_IDX[0]]
        chin = pts[_FACE_OUTLINE_IDX[1]]
        left_cheek = pts[_FACE_OUTLINE_IDX[2]]
        right_cheek = pts[_FACE_OUTLINE_IDX[3]]
        nose = pts[_NOSE_IDX]

        face_width = np.linalg.norm(right_cheek - left_cheek) + 1e-6
        face_height = np.linalg.norm(chin - top) + 1e-6

        # Yaw: nose horizontal offset from the midpoint of the cheeks
        mid_x = (left_cheek[0] + right_cheek[0]) / 2.0
        yaw = float(np.degrees(np.arctan2(nose[0] - mid_x, face_width / 2.0)))

        # Pitch: nose vertical offset from midpoint of top/chin
        mid_y = (top[1] + chin[1]) / 2.0
        pitch = float(np.degrees(np.arctan2(nose[1] - mid_y, face_height / 2.0)))

        # Roll: angle of the eye line
        left_eye = pts[_LEFT_EYE_IDX]
        right_eye = pts[_RIGHT_EYE_IDX]
        roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))

        return yaw, pitch, roll

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray, idx) -> float:
        p = pts[idx]
        # idx layout: [outer_corner, top1, top2, inner_corner, bottom1, bottom2]
        vertical_1 = np.linalg.norm(p[1] - p[5])
        vertical_2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3]) + 1e-6
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Estimate a similarity transform (rotation+scale+translation) mapping
    src points to dst points. Standard Umeyama algorithm, same approach
    InsightFace/ArcFace preprocessing uses for 5-point alignment."""
    assert src.shape == dst.shape
    n, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / n
    U, S, Vt = np.linalg.svd(A)

    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[-1] = -1

    R = U @ np.diag(d) @ Vt
    var_src = (src_demean ** 2).sum() / n
    if var_src < 1e-8:
        return None
    scale = (S * d).sum() / var_src

    t = dst_mean - scale * R @ src_mean
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M
