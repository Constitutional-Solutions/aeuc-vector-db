# aeuc-vector-db

**AEUC Vector Field Database** — semantic embedding layer anchored to the
[glyph-registry](https://github.com/Constitutional-Solutions/glyph-registry)
base-144k address space.

Part of the [AEUC open-source stack](https://github.com/Constitutional-Solutions).

[![CI](https://github.com/Constitutional-Solutions/aeuc-vector-db/actions/workflows/ci.yml/badge.svg)](https://github.com/Constitutional-Solutions/aeuc-vector-db/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aeuc-vector-db)](https://pypi.org/project/aeuc-vector-db/)

---

## What this is

`aeuc-vector-db` provides the **semantic memory substrate** for the AEUC architecture.
Every embedding is anchored to a `Glyph144k` address (0–143,999) from `glyph-registry`,
giving each vector a deterministic coordinate in the base-144k / base-1.44M address space.

| Type | Role |
|------|------|
| `IGlyph` | **Instance Glyph** — one concrete semantic observation, memory, or measurement |
| `PGlyph` | **Proto Glyph** — centroid prototype for a cluster of IGlyphs |

All mutations are **Blake2b-256 hash-chained** (FSOU audit compliance),
matching the integrity model in `glyph-registry`.

---

## Installation

```bash
pip install aeuc-vector-db
```

Requires Python ≥ 3.10 and `numpy ≥ 1.24`.

---

## Quick start

```python
from aeuc_vector_db import VectorFieldDB

db = VectorFieldDB(dim=128)

# Add instance glyphs
for i in range(10):
    db.add_iglyph(
        glyph_id=i,
        outer_context_id=1,          # CTX_GEOMETRY
        embedding=[float(i) * 0.1] * 128,
        label=f"observation_{i}",
    )

# Nearest-neighbour search (cosine by default)
results = db.search(query=[0.45] * 128, top_k=3)
for ig, score in results:
    print(f"  [{ig.glyph_id}] {ig.label}  score={score:.4f}")

# Auto-cluster with phi-partition
pglyphs = db.auto_cluster_phi(outer_context_id=1, anchor_glyph_id=0)
print(f"Formed {len(pglyphs)} phi-band clusters")

# Export JSONL for cold storage / Glacier LTO backup
jsonl = db.export_jsonl()

# FSOU integrity
print(db.stats())
```

---

## Architecture

```
glyph-registry                  aeuc-vector-db
────────────────────────────    ──────────────────────────────────
Glyph144k  (id 0–143,999)  ◄── IGlyph.glyph_id      (anchor)
OuterContext (id 0–9)      ◄── IGlyph.outer_context_id
GlyphRegistry              ──► VectorFieldDB  (composes, not inherits)
```

---

## Similarity metrics

| Metric | Description |
|--------|-------------|
| `cosine` | Standard cosine similarity (default) |
| `euclidean` | 1 / (1 + L2 distance) — converted to similarity score |
| `dot` | Raw dot product |
| `phi_weighted` | Cosine similarity with φ-harmonic per-dimension weights |

The `phi_weighted` metric assigns higher weight to embedding dimensions
whose magnitude falls near an integer power of φ (golden ratio), reflecting
the AEUC harmonic model (AEUC Formal Spec §1.2).

---

## FSOU audit compliance

Every mutation (`add_iglyph`, `update_iglyph_embedding`, `delete_iglyph`,
`form_cluster`, `recompute_pglyph`, `import_jsonl`) appends a signed record
to `change_history`:

```json
{
  "action": "ADD_IGLYPH",
  "timestamp": "2026-02-21T05:16:00Z",
  "iglyph_id": "3fa85f64-...",
  "glyph_id": 1,
  "hash_before": "a3f7c2d1...",
  "hash_after":  "9c12e4b8..."
}
```

The `current_hash` (Blake2b-256) covers the sorted set of all IGlyph and
PGlyph IDs — any silent insertion, deletion, or re-ordering changes it.

---

## Package layout

```
aeuc_vector_db/
├── __init__.py       re-exports VectorFieldDB, IGlyph, PGlyph
├── types.py          IGlyph, PGlyph, VectorEntry dataclasses
├── similarity.py     cosine / euclidean / dot / phi_weighted
├── clustering.py     centroid, inertia, form_pglyph, phi_partition
└── vector_field.py   VectorFieldDB — full CRUD + search + export
```

---

## Related packages

| Package | Role |
|---------|------|
| [`glyph-registry`](https://github.com/Constitutional-Solutions/glyph-registry) | Base-144k glyph + outer-context store (Phase 1) |
| [`aeuc-api`](https://github.com/Constitutional-Solutions/aeuc-api) | FastAPI REST layer over glyph-registry (Phase 2) |
| `aeuc-vector-db` | **This package** — vector field / semantic memory (Phase 2.5) |
| `harmonic-engine` | Frequency-ratio algebra over harmonic payloads (Phase 3) |
| `geometry-engine` | Sacred geometry primitives over geometry payloads (Phase 3) |

Full roadmap: [ROADMAP.md in glyph-registry](https://github.com/Constitutional-Solutions/glyph-registry/blob/main/ROADMAP.md)

---

## License

MIT
