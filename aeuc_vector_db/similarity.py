"""Similarity and distance metrics for aeuc-vector-db.

All functions operate on 1-D numpy float32 arrays.

Metrics
-------
cosine        — cosine similarity in [−1, 1] (higher = more similar)
euclidean     — 1 / (1 + L2 distance), converted to similarity in (0, 1]
dot           — raw dot product
phi_weighted  — cosine similarity with per-dimension φ-harmonic weights
                (AEUC §1.2: dimensions near integer powers of φ are up-weighted)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SimilarityMetric = Literal["cosine", "euclidean", "dot", "phi_weighted"]

_PHI: float = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio ≈ 1.618...
_PHI_POWERS = np.array([_PHI ** n for n in range(-3, 7)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, in [−1, 1]."""
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b)) / denom


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 (Euclidean) distance between two vectors."""
    return float(np.linalg.norm(a - b))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance converted to a similarity score in (0, 1]."""
    return 1.0 / (1.0 + euclidean_distance(a, b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot product of two vectors."""
    return float(np.dot(a, b))


def phi_weighted_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Phi-harmonic weighted cosine similarity.

    Each dimension i receives a weight inversely proportional to its
    distance from the nearest integer power of φ.  Dimensions whose
    values fall close to φ^n (n ∈ {−3, …, 6}) are boosted; far-from-φ
    dimensions are down-weighted.

    This reflects the AEUC harmonic model (§1.2): information encoded
    near golden-ratio scales carries higher semantic salience.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    avg_mag = (np.abs(a) + np.abs(b)) / 2.0 + 1e-12          # (N,)
    # distance of each dim's avg magnitude to every φ^n              (N, P)
    dist_matrix = np.abs(avg_mag[:, np.newaxis] - _PHI_POWERS)
    min_dists = dist_matrix.min(axis=1)                        # (N,)
    weights = 1.0 / (min_dists + 1e-8)
    weights /= weights.sum() + 1e-12
    wa = a * weights
    wb = b * weights
    denom = float(np.linalg.norm(wa)) * float(np.linalg.norm(wb)) + 1e-12
    return float(np.dot(wa, wb)) / denom


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def similarity(
    a: np.ndarray,
    b: np.ndarray,
    metric: SimilarityMetric = "cosine",
) -> float:
    """Compute similarity between two vectors using the specified metric.

    Parameters
    ----------
    a, b   : 1-D numpy arrays of the same length.
    metric : one of 'cosine', 'euclidean', 'dot', 'phi_weighted'.

    Returns
    -------
    float — similarity score (higher = more similar for all metrics).
    """
    if metric == "cosine":
        return cosine_similarity(a, b)
    elif metric == "euclidean":
        return euclidean_similarity(a, b)
    elif metric == "dot":
        return dot_product(a, b)
    elif metric == "phi_weighted":
        return phi_weighted_similarity(a, b)
    else:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            "Valid options: 'cosine', 'euclidean', 'dot', 'phi_weighted'."
        )
