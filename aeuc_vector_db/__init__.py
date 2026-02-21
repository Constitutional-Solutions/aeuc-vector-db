"""aeuc-vector-db — AEUC Vector Field Database.

Semantic embedding layer anchored to the glyph-registry base-144k address space.
FSOU Blake2b-256 hash-chained audit compliance.

Public API::

    from aeuc_vector_db import VectorFieldDB, IGlyph, PGlyph, VectorEntry
"""

from .types import IGlyph, PGlyph, VectorEntry
from .similarity import similarity, SimilarityMetric
from .clustering import form_pglyph, phi_partition, compute_centroid, compute_inertia
from .vector_field import VectorFieldDB

__version__ = "0.1.0"
__all__ = [
    "IGlyph",
    "PGlyph",
    "VectorEntry",
    "VectorFieldDB",
    "similarity",
    "SimilarityMetric",
    "form_pglyph",
    "phi_partition",
    "compute_centroid",
    "compute_inertia",
]
