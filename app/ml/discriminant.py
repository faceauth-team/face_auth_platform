"""
Discriminant projector for the classical (non-ArcFace) feature path.

ArcFace embeddings are directly comparable by cosine similarity for
identities the model never saw — that's the entire point of training a
deep metric-learning model on millions of labeled faces (spec Section 2).
The classical LBP+HOG descriptor in `app/ml/embedder.py` has no such
property: two raw descriptors from the same person are not reliably more
similar than two descriptors from different people (this was verified
empirically — generic texture statistics mostly capture lighting/pose,
not identity).

What *does* help, when no pretrained universal embedding is available:
fit a supervised projection (PCA for whitening/denoising, then Linear
Discriminant Analysis) on the descriptors of the people who are actually
enrolled, so the projection is optimized to separate exactly the
identities you have. This is the classical "Fisherfaces" approach. Its
defining trade-off vs. ArcFace: it must be refit whenever the enrolled
population changes meaningfully, and its quality depends on having
enough enrolled identities and enough genuine per-identity variation —
it is not a substitute for the spec's accuracy targets and should be
treated as a development/demo aid, not a production matcher.

This module only activates when `Embedder.requires_discriminant` is
True (see app/ml/embedder.py). With an ArcFace embedder, `transform()`
is a pure passthrough.
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from app.core import config

PROJECTOR_PATH = Path(config.DATA_DIR) / "discriminant_projector.pkl"

# Need at least this many distinct enrolled identities, each with at
# least MIN_SAMPLES_PER_CLASS templates, before a discriminant fit is
# attempted. Below this, matching falls back to raw cosine
# similarity on the unprojected descriptor (weak, but better than
# nothing while the population is still tiny).
MIN_CLASSES_FOR_FIT = 2
MIN_SAMPLES_PER_CLASS = 3


class DiscriminantProjector:
    """Fits PCA(whiten) -> LDA on raw classical descriptors, keyed by
    employee_id. Thread-safe singleton; persisted to disk so a restart
    doesn't lose the fit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pca: Optional[PCA] = None
        self._lda: Optional[LinearDiscriminantAnalysis] = None
        self._fitted_classes: list[str] = []
        self._load()

    @property
    def is_fitted(self) -> bool:
        return self._pca is not None and self._lda is not None

    def output_dim(self) -> int:
        if self._lda is not None:
            return self._lda.transform(np.zeros((1, self._pca.n_components_), dtype=np.float32)).shape[1]
        return 0

    def fit(self, raw_vectors: np.ndarray, employee_ids: list[str]) -> bool:
        """Refit on the full current set of (raw_descriptor, employee_id)
        pairs across all active employees. Call this after enrollment
        completes or an employee is removed. Returns False (no-op) if
        there isn't enough data yet to fit safely."""
        labels = np.array(employee_ids)
        unique, counts = np.unique(labels, return_counts=True)
        eligible = unique[counts >= MIN_SAMPLES_PER_CLASS]
        if len(eligible) < MIN_CLASSES_FOR_FIT:
            return False

        mask = np.isin(labels, eligible)
        X = raw_vectors[mask]
        y = labels[mask]

        n_samples, n_features = X.shape
        n_classes = len(eligible)
        n_pca = max(2, min(150, n_samples - 1, n_features))
        n_lda = max(1, min(n_classes - 1, n_pca))

        with self._lock:
            pca = PCA(n_components=n_pca, whiten=True, random_state=42).fit(X)
            lda = LinearDiscriminantAnalysis(n_components=n_lda).fit(pca.transform(X), y)
            self._pca, self._lda = pca, lda
            self._fitted_classes = list(eligible)
            self._save()
        return True

    def transform(self, raw_vector: np.ndarray) -> np.ndarray:
        """Project one raw descriptor into discriminant space. If not yet
        fitted, returns the L2-normalized raw vector unchanged (cosine
        similarity fallback)."""
        if not self.is_fitted:
            norm = np.linalg.norm(raw_vector)
            return raw_vector / norm if norm > 1e-8 else raw_vector
        projected = self._lda.transform(self._pca.transform(raw_vector[None, :]))[0]
        norm = np.linalg.norm(projected)
        return (projected / norm).astype(np.float32) if norm > 1e-8 else projected.astype(np.float32)

    def transform_batch(self, raw_vectors: np.ndarray) -> np.ndarray:
        if raw_vectors.shape[0] == 0:
            return raw_vectors
        if not self.is_fitted:
            norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            return raw_vectors / norms
        projected = self._lda.transform(self._pca.transform(raw_vectors))
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return (projected / norms).astype(np.float32)

    def _save(self):
        PROJECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROJECTOR_PATH, "wb") as f:
            pickle.dump(
                {"pca": self._pca, "lda": self._lda, "fitted_classes": self._fitted_classes}, f
            )

    def _load(self):
        if not PROJECTOR_PATH.exists():
            return
        try:
            with open(PROJECTOR_PATH, "rb") as f:
                state = pickle.load(f)
            self._pca = state.get("pca")
            self._lda = state.get("lda")
            self._fitted_classes = state.get("fitted_classes", [])
        except Exception:
            # Corrupt/incompatible pickle (e.g. sklearn version mismatch) —
            # fail safe to unfit rather than crash the service.
            self._pca = self._lda = None
            self._fitted_classes = []


_projector_singleton: Optional[DiscriminantProjector] = None


def get_projector() -> DiscriminantProjector:
    global _projector_singleton
    if _projector_singleton is None:
        _projector_singleton = DiscriminantProjector()
    return _projector_singleton
