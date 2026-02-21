"""Unit tests for aeuc_vector_db.vector_field.VectorFieldDB."""

import pytest

from aeuc_vector_db.vector_field import VectorFieldDB

DIM = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    return VectorFieldDB(dim=DIM)


def emb(val: float = 0.1):
    return [val] * DIM


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction(db):
    assert db.dim == DIM
    assert len(db.iglyphs) == 0
    assert len(db.pglyphs) == 0
    assert db.current_hash  # non-empty


def test_invalid_dim():
    with pytest.raises(ValueError):
        VectorFieldDB(dim=0)


# ---------------------------------------------------------------------------
# IGlyph CRUD
# ---------------------------------------------------------------------------


def test_add_iglyph(db):
    ig = db.add_iglyph(glyph_id=1, outer_context_id=0, embedding=emb())
    assert ig.iglyph_id in db.iglyphs
    assert db.iglyphs[ig.iglyph_id].glyph_id == 1


def test_add_iglyph_wrong_dim(db):
    with pytest.raises(ValueError, match="dim"):
        db.add_iglyph(glyph_id=1, outer_context_id=0, embedding=[0.1] * (DIM + 1))


def test_add_iglyph_duplicate_id(db):
    ig = db.add_iglyph(1, 0, emb(), iglyph_id="fixed-id")
    with pytest.raises(ValueError, match="already exists"):
        db.add_iglyph(1, 0, emb(), iglyph_id="fixed-id")


def test_get_iglyph(db):
    ig = db.add_iglyph(1, 0, emb())
    assert db.get_iglyph(ig.iglyph_id) is ig
    assert db.get_iglyph("nonexistent") is None


def test_update_iglyph(db):
    ig = db.add_iglyph(1, 0, emb(0.1))
    db.update_iglyph_embedding(ig.iglyph_id, emb(0.9))
    assert db.iglyphs[ig.iglyph_id].embedding[0] == pytest.approx(0.9)


def test_update_iglyph_not_found(db):
    with pytest.raises(KeyError):
        db.update_iglyph_embedding("ghost", emb())


def test_delete_iglyph(db):
    ig = db.add_iglyph(2, 1, emb())
    db.delete_iglyph(ig.iglyph_id)
    assert ig.iglyph_id not in db.iglyphs
    assert ig.iglyph_id not in db._glyph_index.get(2, [])
    assert ig.iglyph_id not in db._ctx_index.get(1, [])


def test_delete_iglyph_not_found(db):
    with pytest.raises(KeyError):
        db.delete_iglyph("ghost")


# ---------------------------------------------------------------------------
# PGlyph / clustering
# ---------------------------------------------------------------------------


def test_form_cluster(db):
    ids = [db.add_iglyph(1, 0, emb()).iglyph_id for _ in range(4)]
    pg = db.form_cluster(ids, anchor_glyph_id=1, outer_context_id=0,
                         cluster_tag="test_cluster")
    assert pg.pglyph_id in db.pglyphs
    assert len(pg.member_ids) == 4
    # Members point back to PGlyph
    for _id in ids:
        assert db.iglyphs[_id].proto_id == pg.pglyph_id


def test_form_cluster_no_valid_members(db):
    with pytest.raises(ValueError, match="No valid IGlyphs"):
        db.form_cluster(["nonexistent"], 1, 0)


def test_recompute_pglyph(db):
    ids = [db.add_iglyph(1, 0, emb(float(i))).iglyph_id for i in range(3)]
    pg = db.form_cluster(ids, 1, 0)
    # Update one member and recompute
    db.update_iglyph_embedding(ids[0], emb(9.0))
    pg_updated = db.recompute_pglyph(pg.pglyph_id)
    # Centroid should now be higher than original
    assert pg_updated.centroid[0] > 3.0


def test_get_pglyph(db):
    ids = [db.add_iglyph(1, 0, emb()).iglyph_id for _ in range(2)]
    pg = db.form_cluster(ids, 1, 0)
    assert db.get_pglyph(pg.pglyph_id) is pg
    assert db.get_pglyph("ghost") is None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_returns_top_k(db):
    for i in range(10):
        db.add_iglyph(1, 0, emb(i * 0.1))
    results = db.search(query=emb(0.4), top_k=3)
    assert len(results) == 3
    assert all(isinstance(score, float) for _, score in results)


def test_search_sorted_descending(db):
    for i in range(5):
        db.add_iglyph(1, 0, emb(i * 0.1))
    results = db.search(emb(0.4), top_k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_glyph_filter(db):
    db.add_iglyph(glyph_id=1, outer_context_id=0, embedding=emb(0.1))
    db.add_iglyph(glyph_id=2, outer_context_id=0, embedding=emb(0.2))
    results = db.search(emb(0.1), top_k=10, glyph_id_filter=1)
    assert all(ig.glyph_id == 1 for ig, _ in results)


def test_search_context_filter(db):
    db.add_iglyph(1, outer_context_id=0, embedding=emb())
    db.add_iglyph(1, outer_context_id=3, embedding=emb())
    results = db.search(emb(), top_k=10, outer_context_filter=3)
    assert all(ig.outer_context_id == 3 for ig, _ in results)


def test_search_pglyphs(db):
    ids = [db.add_iglyph(1, 0, emb(i * 0.1)).iglyph_id for i in range(4)]
    db.form_cluster(ids, 1, 0, "cluster_a")
    results = db.search_pglyphs(emb(0.2), top_k=1)
    assert len(results) == 1
    pg, score = results[0]
    assert isinstance(score, float)


def test_search_phi_weighted_metric(db):
    db.add_iglyph(1, 0, [1.618] * DIM)
    db.add_iglyph(1, 0, [0.001] * DIM)
    results = db.search([1.618] * DIM, top_k=2, metric="phi_weighted")
    assert len(results) == 2
    # The phi-valued vector should rank first
    assert results[0][0].embedding[0] == pytest.approx(1.618, abs=1e-4)


# ---------------------------------------------------------------------------
# Auto-cluster phi
# ---------------------------------------------------------------------------


def test_auto_cluster_phi(db):
    for i in range(9):
        db.add_iglyph(1, outer_context_id=2,
                      embedding=[float(i) * 0.15 + 0.05] * DIM)
    pglyphs = db.auto_cluster_phi(outer_context_id=2, anchor_glyph_id=1)
    assert len(pglyphs) >= 1
    assert all(pg.pglyph_id in db.pglyphs for pg in pglyphs)


def test_auto_cluster_phi_empty_context(db):
    pglyphs = db.auto_cluster_phi(outer_context_id=9, anchor_glyph_id=0)
    assert pglyphs == []


# ---------------------------------------------------------------------------
# Import / Export
# ---------------------------------------------------------------------------


def test_export_import_jsonl_round_trip(db):
    for _ in range(3):
        db.add_iglyph(1, 0, emb())
    jsonl = db.export_jsonl()
    db2 = VectorFieldDB(dim=DIM)
    count = db2.import_jsonl(jsonl)
    assert count == 3
    assert len(db2.iglyphs) == 3


def test_import_jsonl_skip_duplicates(db):
    ig = db.add_iglyph(1, 0, emb())
    jsonl = db.export_jsonl()
    # Re-import into same db without overwrite — should skip
    count = db.import_jsonl(jsonl, overwrite=False)
    assert count == 0


def test_import_jsonl_overwrite(db):
    ig = db.add_iglyph(1, 0, emb(0.1))
    # Manually alter the JSONL to change the embedding
    import json
    data = ig.to_dict()
    data["embedding"] = emb(0.9)
    count = db.import_jsonl(json.dumps(data), overwrite=True)
    assert count == 1
    assert db.iglyphs[ig.iglyph_id].embedding[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Audit / integrity
# ---------------------------------------------------------------------------


def test_hash_changes_on_add(db):
    h0 = db.current_hash
    db.add_iglyph(1, 0, emb())
    assert db.current_hash != h0


def test_hash_changes_on_delete(db):
    ig = db.add_iglyph(1, 0, emb())
    h1 = db.current_hash
    db.delete_iglyph(ig.iglyph_id)
    assert db.current_hash != h1


def test_change_history_populated(db):
    db.add_iglyph(1, 0, emb())
    assert len(db.change_history) >= 1
    entry = db.change_history[-1]
    assert "action" in entry
    assert "hash_before" in entry
    assert "hash_after" in entry
    assert "timestamp" in entry


# ---------------------------------------------------------------------------
# Snapshot / stats
# ---------------------------------------------------------------------------


def test_snapshot(db):
    db.add_iglyph(1, 0, emb())
    snap = db.snapshot()
    assert snap["version"] == db.VERSION
    assert snap["dim"] == DIM
    assert len(snap["iglyphs"]) == 1
    assert "change_history" in snap


def test_stats(db):
    db.add_iglyph(1, 0, emb())
    s = db.stats()
    assert s["iglyph_count"] == 1
    assert s["pglyph_count"] == 0
    assert s["dim"] == DIM
    assert len(s["current_hash"]) == 64


def test_repr(db):
    r = repr(db)
    assert "VectorFieldDB" in r
    assert "dim=8" in r
