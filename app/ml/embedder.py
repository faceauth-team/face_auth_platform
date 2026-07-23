"""
Face embedding: aligned 112x112 crop -> fixed-length vector used for face
matching (1:N identification via /identify and 1:1 verification via the
/authorize step-up).

ACTIVE PATH: ArcFace (InsightFace `buffalo_l`, w600k_r50.onnx) is present
under models/buffalo_l/ and is the embedder this service runs today — a
deep network trained on millions of labeled faces that produces a
*universal* embedding: cosine similarity between two ArcFace vectors is
meaningful for identities it never saw during training, and the model is
frozen (spec Section 2/6). `get_embedder()` selects it automatically
whenever the ONNX file exists at ARCFACE_MODEL_PATH.

FALLBACK PATH: if the ArcFace weights are absent (e.g. a fresh checkout
on a machine that hasn't pulled the model pack yet), the service falls
back to a classical descriptor (LBP + HOG on the aligned crop) combined
with a PCA+LDA discriminant ("Fisherfaces"-style) projection *fit on the
currently-enrolled population*. That keeps the full pipeline (capture ->
enroll -> store -> identify -> token -> audit log) runnable fully
offline, but it is **not validated to any FAR/FRR target and must not be
treated as production-grade** — it is a stand-in for environments without
the weights, not a substitute for ArcFace. Either way, thresholds must be
calibrated against real pilot data per spec Section 17.

Two embedders, one interface:

  - `ArcFaceEmbedder` (active when weights present): 512-d universal
    embedding via ONNX Runtime, no per-tenant retraining step.
    `requires_discriminant = False`.

  - `ClassicalFeatureExtractor` (offline fallback): LBP+HOG raw
    descriptor. Must be paired with
    `app.ml.discriminant.DiscriminantProjector` (fit on enrolled
    templates) before its output is meaningfully comparable across
    identities — see `app/ml/matcher.py`, which wires this up.
    `requires_discriminant = True`.

`get_embedder()` is the single place that decides which is used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

from app.core import config

# Default to the project-local weights dir (models/buffalo_l/w600k_r50.onnx)
# so the production ArcFace path activates automatically once the buffalo_l
# pack is downloaded there; override with ARCFACE_MODEL_PATH if hosted
# elsewhere. BASE_DIR is the repo root (see app/core/config.py).
_DEFAULT_ARCFACE_PATH = config.BASE_DIR / "models" / "buffalo_l" / "w600k_r50.onnx"
ARCFACE_MODEL_PATH = Path(os.getenv("ARCFACE_MODEL_PATH", str(_DEFAULT_ARCFACE_PATH)))


class Embedder(Protocol):
    dim: int
    requires_discriminant: bool

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 raw feature vector of length `self.dim`."""
        ...


class ClassicalFeatureExtractor:
    """LBP-histogram + HOG grid descriptor. This is a *raw feature
    extractor*, not by itself a discriminative embedding — its output
    must be passed through a fitted DiscriminantProjector (see
    app/ml/discriminant.py) before identity comparison. Deterministic,
    CPU-only, zero external weights."""

    requires_discriminant = True

    def __init__(self, grid: int = 8):
        self.grid = grid
        self.dim = self._compute_dim()  # 6724 for grid=8 on a 112x112 input

    def _compute_dim(self) -> int:
        lbp_dim = self.grid * self.grid * 10
        cells_per_side = 112 // 8
        blocks_per_side = cells_per_side - 1
        hog_dim = blocks_per_side * blocks_per_side * 9 * 2 * 2
        return lbp_dim + hog_dim

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # normalize lighting before descriptor extraction

        lbp_feats = self._grid_lbp(gray)
        hog_feats = self._hog(gray)
        return np.concatenate([lbp_feats, hog_feats]).astype(np.float32)

    def _grid_lbp(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        gh, gw = h // self.grid, w // self.grid
        feats = []
        for i in range(self.grid):
            for j in range(self.grid):
                cell = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                lbp = local_binary_pattern(cell, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
                feats.append(hist)
        return np.concatenate(feats).astype(np.float32)

    def _hog(self, gray: np.ndarray) -> np.ndarray:
        feats = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        return feats.astype(np.float32)


# Backwards-compatible alias used by a couple of early call sites.
ClassicalCVEmbedder = ClassicalFeatureExtractor


class ArcFaceEmbedder:
    """Production embedder: ArcFace (InsightFace buffalo_l) via ONNX
    Runtime. Active whenever the weight file is present at
    ARCFACE_MODEL_PATH (default models/buffalo_l/w600k_r50.onnx)."""

    requires_discriminant = False

    def __init__(self, model_path: Path = ARCFACE_MODEL_PATH):
        import onnxruntime as ort  # local import: optional dependency until weights exist

        self.dim = 512
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        out = self._session.run(None, {self._input_name: blob})[0][0]
        norm = np.linalg.norm(out)
        return (out / norm) if norm > 1e-8 else out


_embedder_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    """Returns the ArcFace embedder if weights are available at
    ARCFACE_MODEL_PATH, otherwise falls back to the classical CV embedder.
    This is the single place to change when you swap embedders."""
    global _embedder_singleton
    if _embedder_singleton is not None:
        return _embedder_singleton

    if ARCFACE_MODEL_PATH.exists():
        _embedder_singleton = ArcFaceEmbedder()
    else:
        _embedder_singleton = ClassicalCVEmbedder()
    return _embedder_singleton
