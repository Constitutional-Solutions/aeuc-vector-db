"""Core record types for aeuc-vector-db.

IGlyph  — Instance Glyph: one concrete semantic observation anchored to a
           Glyph144k address and an OuterContext in the base-1.44M space.
PGlyph  — Proto Glyph: centroid prototype synthesised from a cluster of
           IGlyphs; acts as a compressed symbolic representative.
VectorEntry — Lightweight raw embedding record for streaming / bulk use.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# VectorEntry
# ---------------------------------------------------------------------------


@dataclass
class VectorEntry:
    """Lightweight raw embedding anchored to a glyph address.

    Schema
    ------
    entry_id        : UUID string
    glyph_id        : int  in [0, 143_999]  — Glyph144k.id
    outer_context_id: int  in [0, 9]        — OuterContext.outer_id
    embedding       : List[float]           — dense float32 vector
    source_tag      : human label (e.g. "harmonic", "geometry")
    meta            : arbitrary key/value store
    """

    entry_id: str
    glyph_id: int
    outer_context_id: int
    embedding: List[float]
    source_tag: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    timestamp: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if not (0 <= self.glyph_id <= 143_999):
            raise ValueError(
                f"glyph_id {self.glyph_id} out of range [0, 143999]"
            )
        if not (0 <= self.outer_context_id <= 9):
            raise ValueError(
                f"outer_context_id {self.outer_context_id} out of range [0, 9]"
            )
        if not self.embedding:
            raise ValueError("embedding must not be empty")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Blake2b-256 hash of the raw embedding bytes."""
        raw = json.dumps(self.embedding, separators=(",", ":")).encode()
        return hashlib.blake2b(raw, digest_size=32).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorEntry":
        return cls(**data)


# ---------------------------------------------------------------------------
# IGlyph
# ---------------------------------------------------------------------------


@dataclass
class IGlyph:
    """Instance Glyph — one concrete semantic observation.

    Maps to AEUC Formal Spec Page 2, Section 10 'instance record'.

    A single real event, measurement, memory, or data point anchored
    at ``glyph_id`` in the base-144k space, with a dense embedding
    vector capturing its position in semantic space.

    Schema
    ------
    iglyph_id       : UUID string (primary key)
    glyph_id        : int  in [0, 143_999]  — Glyph144k anchor
    outer_context_id: int  in [0, 9]        — OuterContext dimension
    embedding       : List[float]           — semantic embedding
    label           : human-readable name
    proto_id        : UUID of parent PGlyph (set by VectorFieldDB)
    meta            : arbitrary key/value store
    """

    iglyph_id: str
    glyph_id: int
    outer_context_id: int
    embedding: List[float]
    label: str = ""
    proto_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    timestamp: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if not (0 <= self.glyph_id <= 143_999):
            raise ValueError(
                f"glyph_id {self.glyph_id} out of range [0, 143999]"
            )
        if not (0 <= self.outer_context_id <= 9):
            raise ValueError(
                f"outer_context_id {self.outer_context_id} out of range [0, 9]"
            )

    # ------------------------------------------------------------------

    @property
    def np_embedding(self):
        """Return embedding as a numpy float32 array."""
        import numpy as np
        return np.array(self.embedding, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IGlyph":
        return cls(**data)

    @classmethod
    def new(
        cls,
        glyph_id: int,
        outer_context_id: int,
        embedding: List[float],
        label: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> "IGlyph":
        """Factory: create a new IGlyph with auto-generated UUID."""
        return cls(
            iglyph_id=str(uuid.uuid4()),
            glyph_id=glyph_id,
            outer_context_id=outer_context_id,
            embedding=embedding,
            label=label,
            meta=meta or {},
        )


# ---------------------------------------------------------------------------
# PGlyph
# ---------------------------------------------------------------------------


@dataclass
class PGlyph:
    """Proto Glyph — centroid prototype for a cluster of IGlyphs.

    Maps to AEUC Formal Spec Page 3, Section 15 'proto record'.

    Represents the geometric centre of multiple IGlyphs in embedding
    space, providing a compressed semantic prototype that acts as a
    symbolic representative for the cluster.

    Schema
    ------
    pglyph_id       : UUID string (primary key)
    glyph_id        : int  in [0, 143_999]  — anchor in 144k space
    outer_context_id: int  in [0, 9]
    centroid        : List[float]           — mean of all member embeddings
    member_ids      : List[str]             — IGlyph.iglyph_ids in cluster
    cluster_tag     : human-readable cluster name
    inertia         : sum of squared distances from members to centroid
    meta            : arbitrary key/value store
    """

    pglyph_id: str
    glyph_id: int
    outer_context_id: int
    centroid: List[float]
    member_ids: List[str]
    cluster_tag: str = ""
    inertia: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    timestamp: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if not (0 <= self.glyph_id <= 143_999):
            raise ValueError(
                f"glyph_id {self.glyph_id} out of range [0, 143999]"
            )
        if not (0 <= self.outer_context_id <= 9):
            raise ValueError(
                f"outer_context_id {self.outer_context_id} out of range [0, 9]"
            )

    # ------------------------------------------------------------------

    @property
    def np_centroid(self):
        """Return centroid as a numpy float32 array."""
        import numpy as np
        return np.array(self.centroid, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PGlyph":
        return cls(**data)
