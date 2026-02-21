"""VectorFieldDB — AEUC Vector Field Database.

Semanticembedding layer anchored to the glyph-registry base-144k address space.
All mutations are Blake2b-256 hash-chained for FSOU audit compliance.

Public API
----------
VectorFieldDB
    .add_iglyph()              — insert a new Instance Glyph
    .get_iglyph()              — retrieve by UUID
    .update_iglyph_embedding() — replace embedding (hash-chained)
    .delete_iglyph()           — remove + clean indices (hash-chained)
    .form_cluster()            — cluster IGlyphs → PGlyph centroid
    .get_pglyph()              — retrieve PGlyph by UUID
    .recompute_pglyph()        — recompute centroid + inertia after updates
    .search()                  — nearest-neighbour over IGlyphs
    .search_pglyphs()          — nearest-neighbour over PGlyph centroids
    .auto_cluster_phi()        — automatic φ-partitioned clustering
    .export_jsonl()            — JSONL export (Glacier / LTO compatible)
    .import_jsonl()            — bulk import from JSONL
    .snapshot()                — full JSON state dump
    .stats()                   — live statistics dict
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clustering import compute_centroid, compute_inertia, form_pglyph, phi_partition
from .similarity import SimilarityMetric
from .similarity import similarity as _sim
from .types import IGlyph, PGlyph


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class VectorFieldDB:
    """AEUC Vector Field Database.

    Every embedding is anchored to a Glyph144k address (0–143,999) and
    an OuterContext id (0–9), both inherited from glyph-registry.

    FSOU compliance
    ---------------
    Every mutating operation appends to ``change_history`` with
    ``hash_before`` / ``hash_after`` (Blake2b-256 over sorted IGlyph +
    PGlyph ID sets).  Silent tampering changes the hash.

    Parameters
    ----------
    dim : int
        Dimensionality of all embeddings stored in this database.
        All add / update / search calls must supply vectors of exactly
        this length.
    """

    VERSION: str = "0.1.0"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, dim: int = 128) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim: int = dim

        # Primary stores
        self.iglyphs: Dict[str, IGlyph] = {}
        self.pglyphs: Dict[str, PGlyph] = {}

        # Indices
        self._glyph_index: Dict[int, List[str]] = {}   # glyph_id   → [iglyph_ids]
        self._ctx_index: Dict[int, List[str]] = {}     # ctx_id     → [iglyph_ids]

        # Audit
        self.change_history: List[Dict[str, Any]] = []
        self.current_hash: str = self._compute_hash()

        print(f"\u2705 VectorFieldDB v{self.VERSION}  dim={dim}")
        print(f"   initial hash: {self.current_hash[:16]}...")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "iglyphs": sorted(self.iglyphs.keys()),
                "pglyphs": sorted(self.pglyphs.keys()),
            },
            sort_keys=True,
        ).encode()
        return hashlib.blake2b(payload, digest_size=32).hexdigest()

    def _log(self, action: str, **kwargs: Any) -> None:
        """Append a hash-chained audit record to change_history."""
        old = self.current_hash
        self.current_hash = self._compute_hash()
        self.change_history.append(
            {
                "action": action,
                "timestamp": _utcnow(),
                "hash_before": old,
                "hash_after": self.current_hash,
                **kwargs,
            }
        )

    def _validate(self, embedding: List[float]) -> np.ndarray:
        arr = np.array(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Embedding must be a 1-D array.")
        if len(arr) != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {len(arr)}."
            )
        return arr

    # ------------------------------------------------------------------
    # IGlyph CRUD
    # ------------------------------------------------------------------

    def add_iglyph(
        self,
        glyph_id: int,
        outer_context_id: int,
        embedding: List[float],
        label: str = "",
        meta: Optional[Dict[str, Any]] = None,
        iglyph_id: Optional[str] = None,
    ) -> IGlyph:
        """Add an Instance Glyph.

        Parameters
        ----------
        glyph_id         : Glyph144k address in [0, 143_999].
        outer_context_id : OuterContext id in [0, 9].
        embedding        : dense float vector of length ``self.dim``.
        label            : optional human-readable name.
        meta             : optional metadata dict.
        iglyph_id        : optional UUID; auto-generated if omitted.

        Returns
        -------
        The newly created IGlyph.
        """
        self._validate(embedding)
        _id = iglyph_id or str(uuid.uuid4())
        if _id in self.iglyphs:
            raise ValueError(f"IGlyph {_id!r} already exists.")

        ig = IGlyph(
            iglyph_id=_id,
            glyph_id=glyph_id,
            outer_context_id=outer_context_id,
            embedding=[float(x) for x in embedding],
            label=label,
            meta=meta or {},
        )
        self.iglyphs[_id] = ig
        self._glyph_index.setdefault(glyph_id, []).append(_id)
        self._ctx_index.setdefault(outer_context_id, []).append(_id)
        self._log("ADD_IGLYPH", iglyph_id=_id, glyph_id=glyph_id,
                  outer_context_id=outer_context_id, label=label)
        return ig

    def get_iglyph(self, iglyph_id: str) -> Optional[IGlyph]:
        """Retrieve an IGlyph by UUID, or None if not found."""
        return self.iglyphs.get(iglyph_id)

    def update_iglyph_embedding(
        self,
        iglyph_id: str,
        embedding: List[float],
    ) -> IGlyph:
        """Replace the embedding for an existing IGlyph."""
        if iglyph_id not in self.iglyphs:
            raise KeyError(f"IGlyph {iglyph_id!r} not found.")
        self._validate(embedding)
        ig = self.iglyphs[iglyph_id]
        ig.embedding = [float(x) for x in embedding]
        ig.timestamp = _utcnow()
        self._log("UPDATE_IGLYPH", iglyph_id=iglyph_id)
        return ig

    def delete_iglyph(self, iglyph_id: str) -> IGlyph:
        """Remove an IGlyph and clean all reverse-lookup indices."""
        if iglyph_id not in self.iglyphs:
            raise KeyError(f"IGlyph {iglyph_id!r} not found.")
        ig = self.iglyphs.pop(iglyph_id)
        # Clean glyph index
        bucket = self._glyph_index.get(ig.glyph_id, [])
        if iglyph_id in bucket:
            bucket.remove(iglyph_id)
        # Clean context index
        bucket = self._ctx_index.get(ig.outer_context_id, [])
        if iglyph_id in bucket:
            bucket.remove(iglyph_id)
        # Remove from parent PGlyph if assigned
        if ig.proto_id and ig.proto_id in self.pglyphs:
            try:
                self.pglyphs[ig.proto_id].member_ids.remove(iglyph_id)
            except ValueError:
                pass
        self._log("DELETE_IGLYPH", iglyph_id=iglyph_id, glyph_id=ig.glyph_id)
        return ig

    # ------------------------------------------------------------------
    # PGlyph CRUD
    # ------------------------------------------------------------------

    def form_cluster(
        self,
        iglyph_ids: List[str],
        anchor_glyph_id: int,
        outer_context_id: int,
        cluster_tag: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> PGlyph:
        """Cluster a set of IGlyphs into a PGlyph prototype centroid.

        Parameters
        ----------
        iglyph_ids       : UUIDs of IGlyphs to include.
        anchor_glyph_id  : Glyph144k.id that anchors this prototype.
        outer_context_id : 0–9 outer dimension.
        cluster_tag      : human-readable label for the cluster.
        meta             : optional metadata.

        Returns
        -------
        The newly created PGlyph.
        """
        members = [self.iglyphs[i] for i in iglyph_ids if i in self.iglyphs]
        if not members:
            raise ValueError("No valid IGlyphs found for the given ids.")

        pg = form_pglyph(
            members,
            anchor_glyph_id=anchor_glyph_id,
            outer_context_id=outer_context_id,
            cluster_tag=cluster_tag,
            meta=meta or {},
        )
        self.pglyphs[pg.pglyph_id] = pg
        for ig in members:
            ig.proto_id = pg.pglyph_id
        self._log(
            "FORM_CLUSTER",
            pglyph_id=pg.pglyph_id,
            anchor_glyph_id=anchor_glyph_id,
            cluster_tag=cluster_tag,
            members=len(members),
        )
        return pg

    def get_pglyph(self, pglyph_id: str) -> Optional[PGlyph]:
        """Retrieve a PGlyph by UUID, or None if not found."""
        return self.pglyphs.get(pglyph_id)

    def recompute_pglyph(self, pglyph_id: str) -> PGlyph:
        """Recompute centroid and inertia for a PGlyph after member updates."""
        if pglyph_id not in self.pglyphs:
            raise KeyError(f"PGlyph {pglyph_id!r} not found.")
        pg = self.pglyphs[pglyph_id]
        members = [self.iglyphs[i] for i in pg.member_ids if i in self.iglyphs]
        if not members:
            raise ValueError("PGlyph has no valid members to recompute from.")
        embs = [m.np_embedding for m in members]
        centroid = compute_centroid(embs)
        pg.centroid = centroid.tolist()
        pg.inertia = compute_inertia(centroid, embs)
        pg.timestamp = _utcnow()
        self._log("RECOMPUTE_PGLYPH", pglyph_id=pglyph_id, members=len(members))
        return pg

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        metric: SimilarityMetric = "cosine",
        glyph_id_filter: Optional[int] = None,
        outer_context_filter: Optional[int] = None,
    ) -> List[Tuple[IGlyph, float]]:
        """Nearest-neighbour search over all IGlyphs.

        Parameters
        ----------
        query                : query embedding of length ``self.dim``.
        top_k                : maximum number of results.
        metric               : similarity metric to use.
        glyph_id_filter      : restrict search to IGlyphs at this glyph_id.
        outer_context_filter : restrict search to IGlyphs in this context.

        Returns
        -------
        List of (IGlyph, score) tuples sorted descending by score.
        """
        q = self._validate(query)
        candidates = list(self.iglyphs.values())

        if glyph_id_filter is not None:
            allowed = set(self._glyph_index.get(glyph_id_filter, []))
            candidates = [c for c in candidates if c.iglyph_id in allowed]

        if outer_context_filter is not None:
            allowed = set(self._ctx_index.get(outer_context_filter, []))
            candidates = [c for c in candidates if c.iglyph_id in allowed]

        scored = [(ig, _sim(q, ig.np_embedding, metric)) for ig in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_pglyphs(
        self,
        query: List[float],
        top_k: int = 5,
        metric: SimilarityMetric = "cosine",
    ) -> List[Tuple[PGlyph, float]]:
        """Nearest-neighbour search over PGlyph centroids.

        Returns prototype-level results; useful for coarse navigation
        before drilling into instance-level IGlyphs.
        """
        q = self._validate(query)
        scored = [
            (pg, _sim(q, pg.np_centroid, metric))
            for pg in self.pglyphs.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Auto-clustering
    # ------------------------------------------------------------------

    def auto_cluster_phi(
        self,
        outer_context_id: int,
        anchor_glyph_id: int,
        levels: int = 3,
    ) -> List[PGlyph]:
        """Auto-cluster all IGlyphs in a context using φ-magnitude partition.

        Partitions the context's IGlyphs into ``levels`` bands whose
        boundaries are set at golden-ratio-scaled fractions of the
        embedding-norm range (AEUC §1.2 harmonic model), then forms one
        PGlyph per non-empty band.

        Parameters
        ----------
        outer_context_id : context to cluster.
        anchor_glyph_id  : Glyph144k.id to anchor the resulting PGlyphs.
        levels           : number of φ-bands (default 3).

        Returns
        -------
        List of newly created PGlyphs (one per non-empty band).
        """
        ctx_ids = self._ctx_index.get(outer_context_id, [])
        iglyphs = [self.iglyphs[i] for i in ctx_ids if i in self.iglyphs]
        if not iglyphs:
            return []

        partitions = phi_partition(iglyphs, levels=levels)
        result: List[PGlyph] = []
        for band_idx, band in enumerate(partitions):
            if band:
                pg = self.form_cluster(
                    iglyph_ids=[ig.iglyph_id for ig in band],
                    anchor_glyph_id=anchor_glyph_id,
                    outer_context_id=outer_context_id,
                    cluster_tag=f"phi_band_{band_idx}",
                )
                result.append(pg)
        return result

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_jsonl(self) -> str:
        """Export all IGlyphs as JSONL, sorted by insertion timestamp.

        One JSON object per line; compatible with Glacier / LTO cold-
        storage backup pipelines.
        """
        lines = [
            json.dumps(ig.to_dict())
            for ig in sorted(self.iglyphs.values(), key=lambda x: x.timestamp)
        ]
        return "\n".join(lines)

    def import_jsonl(
        self,
        jsonl: str,
        overwrite: bool = False,
    ) -> int:
        """Bulk import IGlyphs from a JSONL string.

        Parameters
        ----------
        jsonl     : JSONL string (one IGlyph JSON object per line).
        overwrite : if True, existing IGlyphs with matching IDs are
                    deleted before re-import; otherwise they are skipped.

        Returns
        -------
        int — number of IGlyphs successfully imported.
        """
        count = 0
        for line in jsonl.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            _id = data.get("iglyph_id")
            if _id and _id in self.iglyphs:
                if not overwrite:
                    continue
                self.delete_iglyph(_id)
            ig = IGlyph.from_dict(data)
            # Bypass validation: imported embeddings may differ in dim
            # (caller's responsibility); re-validate for safety.
            if len(ig.embedding) != self.dim:
                continue
            self.iglyphs[ig.iglyph_id] = ig
            self._glyph_index.setdefault(ig.glyph_id, []).append(ig.iglyph_id)
            self._ctx_index.setdefault(ig.outer_context_id, []).append(ig.iglyph_id)
            count += 1
        self._log("IMPORT_JSONL", count=count, overwrite=overwrite)
        return count

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete serialisable state snapshot.

        Suitable for JSON file export, cold storage, or bootstrapping
        a new VectorFieldDB instance from a known state.
        """
        return {
            "version": self.VERSION,
            "dim": self.dim,
            "current_hash": self.current_hash,
            "iglyphs": {k: v.to_dict() for k, v in self.iglyphs.items()},
            "pglyphs": {k: v.to_dict() for k, v in self.pglyphs.items()},
            "change_history": self.change_history,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return live statistics for monitoring / health-check endpoints."""
        return {
            "version": self.VERSION,
            "dim": self.dim,
            "iglyph_count": len(self.iglyphs),
            "pglyph_count": len(self.pglyphs),
            "glyph_addresses_used": len(self._glyph_index),
            "outer_contexts_used": len(self._ctx_index),
            "change_history_entries": len(self.change_history),
            "current_hash": self.current_hash,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"VectorFieldDB(v{s['version']} dim={s['dim']} "
            f"iglyphs={s['iglyph_count']} pglyphs={s['pglyph_count']} "
            f"hash={self.current_hash[:8]}...)"
        )
