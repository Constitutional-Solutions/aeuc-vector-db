"""Clustering helpers for aeuc-vector-db.

form_pglyph    — build a PGlyph centroid from a list of IGlyphs
phi_partition  — split IGlyphs into φ-scaled magnitude bands
compute_centroid / compute_inertia — low-level helpers
"""

from __future__ import annotations

import uuid
from typing import List, Optional

import numpy as np

from .types import IGlyph, PGlyph

_PHI: float = (1.0 + 5.0 ** 0.5) / 2.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean centroid of a list of embedding arrays."""
    if not embeddings:
        raise ValueError("Cannot compute centroid of an empty list.")
    return np.vstack(embeddings).mean(axis=0)


def compute_inertia(centroid: np.ndarray, embeddings: List[np.ndarray]) -> float:
    """Sum of squared L2 distances from each embedding to the centroid."""
    return float(sum(np.sum((e - centroid) ** 2) for e in embeddings))


# ---------------------------------------------------------------------------
# PGlyph formation
# ---------------------------------------------------------------------------


def form_pglyph(
    iglyphs: List[IGlyph],
    anchor_glyph_id: int,
    outer_context_id: int,
    cluster_tag: str = "",
    meta: Optional[dict] = None,
) -> PGlyph:
    """Build a PGlyph from a list of IGlyphs.

    Parameters
    ----------
    iglyphs          : member IGlyphs for this prototype cluster.
    anchor_glyph_id  : Glyph144k.id that anchors this prototype in 144k space.
    outer_context_id : 0–9 outer dimension context.
    cluster_tag      : human-readable cluster label.
    meta             : optional metadata dict.

    Returns
    -------
    PGlyph with centroid, inertia, and member_ids populated.
    """
    if not iglyphs:
        raise ValueError("iglyphs list is empty — cannot form PGlyph.")

    embeddings = [ig.np_embedding for ig in iglyphs]
    centroid = compute_centroid(embeddings)
    inertia = compute_inertia(centroid, embeddings)

    return PGlyph(
        pglyph_id=str(uuid.uuid4()),
        glyph_id=anchor_glyph_id,
        outer_context_id=outer_context_id,
        centroid=centroid.tolist(),
        member_ids=[ig.iglyph_id for ig in iglyphs],
        cluster_tag=cluster_tag,
        inertia=inertia,
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# Phi-partition
# ---------------------------------------------------------------------------


def phi_partition(
    iglyphs: List[IGlyph],
    levels: int = 3,
) -> List[List[IGlyph]]:
    """Partition IGlyphs into φ-scaled magnitude bands.

    The norm range [min_norm, max_norm] is divided into ``levels`` bands
    whose boundaries are placed at φ-scaled fractions of the total span,
    consistent with the AEUC golden-ratio harmonic model.

    Parameters
    ----------
    iglyphs : IGlyphs to partition.
    levels  : number of bands (default 3).

    Returns
    -------
    List of ``levels`` lists; each sub-list holds the IGlyphs whose
    embedding L2 norm falls within that band.
    """
    if not iglyphs:
        return [[] for _ in range(levels)]

    norms = np.array([np.linalg.norm(ig.np_embedding) for ig in iglyphs])
    min_n, max_n = float(norms.min()), float(norms.max())
    span = max_n - min_n + 1e-12

    # φ-scaled interior breakpoints
    breaks: List[float] = [min_n]
    for k in range(1, levels):
        breaks.append(min_n + span * (1.0 - 1.0 / (_PHI ** k)))
    breaks.append(max_n + 1e-9)  # inclusive upper bound

    partitions: List[List[IGlyph]] = [[] for _ in range(levels)]
    for ig, n in zip(iglyphs, norms):
        for i in range(levels):
            if breaks[i] <= n < breaks[i + 1]:
                partitions[i].append(ig)
                break

    return partitions
