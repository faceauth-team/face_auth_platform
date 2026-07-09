"""
Reduce the (couple-hundred) quality-passed raw descriptors captured
during enrollment down to a small set of representative templates
(spec Section 7.1 step 7: "10-20 representative templates... via
clustering, e.g. k-means on the embeddings, to capture pose/lighting
variation while reducing storage").

Works on whatever vector space the active embedder produces (raw
classical descriptors, or ArcFace embeddings if that's swapped in) —
this module doesn't care which.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from app.core import config


def cluster_to_templates(
    vectors: np.ndarray,
    min_templates: int = config.ENROLL_TEMPLATES_MIN,
    max_templates: int = config.ENROLL_TEMPLATES_MAX,
) -> np.ndarray:
    """K-means cluster `vectors` (n_frames x dim) down to between
    `min_templates` and `max_templates` representative vectors (cluster
    centroids). Picks k via a simple elbow heuristic, bounded by the
    spec's 10-20 range. Falls back gracefully if fewer frames than
    `min_templates` passed quality filtering."""
    n = vectors.shape[0]
    if n == 0:
        return vectors

    if n <= min_templates:
        # Not enough frames to cluster meaningfully — keep them all as
        # individual templates (capped by max for safety).
        return vectors[:max_templates]

    k = min(max_templates, max(min_templates, _elbow_k(vectors, min_templates, max_templates)))
    k = min(k, n)

    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(vectors)
    return km.cluster_centers_.astype(np.float32)


def _elbow_k(vectors: np.ndarray, lo: int, hi: int) -> int:
    """Cheap elbow-method k selection: try a few k values within
    [lo, hi], pick the smallest k where adding more clusters stops
    meaningfully reducing inertia. Bounded to a handful of fits so
    enrollment finalization stays fast."""
    candidates = sorted(set([lo, (lo + hi) // 2, hi]))
    inertias = []
    for k in candidates:
        k = min(k, vectors.shape[0])
        km = KMeans(n_clusters=k, n_init=2, random_state=42).fit(vectors)
        inertias.append((k, km.inertia_))

    if len(inertias) < 2:
        return lo

    # Pick the k after which inertia reduction per added cluster drops
    # below 15% of the first-step reduction.
    first_drop = max(inertias[0][1] - inertias[1][1], 1e-6)
    chosen = inertias[0][0]
    for i in range(1, len(inertias)):
        drop = inertias[i - 1][1] - inertias[i][1]
        if drop < 0.15 * first_drop:
            break
        chosen = inertias[i][0]
    return chosen
