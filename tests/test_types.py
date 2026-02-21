"""Unit tests for aeuc_vector_db.types."""

import uuid

import pytest

from aeuc_vector_db.types import IGlyph, PGlyph, VectorEntry

DIM = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ig(glyph_id: int = 1, ctx: int = 0, dim: int = DIM) -> IGlyph:
    return IGlyph(
        iglyph_id=str(uuid.uuid4()),
        glyph_id=glyph_id,
        outer_context_id=ctx,
        embedding=[0.1] * dim,
        label="test",
    )


# ---------------------------------------------------------------------------
# VectorEntry
# ---------------------------------------------------------------------------


def test_vector_entry_valid():
    e = VectorEntry(
        entry_id=str(uuid.uuid4()),
        glyph_id=100,
        outer_context_id=2,
        embedding=[0.5] * DIM,
    )
    assert e.glyph_id == 100
    assert len(e.fingerprint()) == 64  # Blake2b-256 hex


def test_vector_entry_empty_embedding():
    with pytest.raises(ValueError, match="empty"):
        VectorEntry(
            entry_id=str(uuid.uuid4()),
            glyph_id=1,
            outer_context_id=0,
            embedding=[],
        )


def test_vector_entry_bad_glyph_id():
    with pytest.raises(ValueError, match="glyph_id"):
        VectorEntry(
            entry_id=str(uuid.uuid4()),
            glyph_id=144_000,  # out of range
            outer_context_id=0,
            embedding=[0.1],
        )


# ---------------------------------------------------------------------------
# IGlyph
# ---------------------------------------------------------------------------


def test_iglyph_valid():
    ig = _ig()
    assert ig.glyph_id == 1
    assert len(ig.embedding) == DIM
    assert ig.proto_id is None


def test_iglyph_boundary_glyph_id():
    ig_lo = _ig(glyph_id=0)
    ig_hi = _ig(glyph_id=143_999)
    assert ig_lo.glyph_id == 0
    assert ig_hi.glyph_id == 143_999


def test_iglyph_bad_glyph_id():
    with pytest.raises(ValueError, match="glyph_id"):
        _ig(glyph_id=144_000)


def test_iglyph_bad_ctx():
    with pytest.raises(ValueError, match="outer_context_id"):
        _ig(ctx=10)


def test_iglyph_boundary_ctx():
    ig = _ig(ctx=9)
    assert ig.outer_context_id == 9


def test_iglyph_np_embedding_dtype():
    import numpy as np
    ig = _ig()
    arr = ig.np_embedding
    assert arr.dtype == np.float32
    assert arr.shape == (DIM,)


def test_iglyph_round_trip():
    ig = _ig()
    d = ig.to_dict()
    ig2 = IGlyph.from_dict(d)
    assert ig2.iglyph_id == ig.iglyph_id
    assert ig2.embedding == ig.embedding
    assert ig2.label == ig.label


def test_iglyph_factory():
    ig = IGlyph.new(glyph_id=5, outer_context_id=3, embedding=[0.2] * DIM)
    assert ig.glyph_id == 5
    assert len(ig.iglyph_id) == 36  # UUID string length


# ---------------------------------------------------------------------------
# PGlyph
# ---------------------------------------------------------------------------


def test_pglyph_valid():
    pg = PGlyph(
        pglyph_id=str(uuid.uuid4()),
        glyph_id=10,
        outer_context_id=1,
        centroid=[0.5] * DIM,
        member_ids=[str(uuid.uuid4())],
        cluster_tag="geo_band_0",
    )
    assert pg.glyph_id == 10
    assert pg.inertia == 0.0


def test_pglyph_bad_glyph_id():
    with pytest.raises(ValueError, match="glyph_id"):
        PGlyph(
            pglyph_id=str(uuid.uuid4()),
            glyph_id=200_000,
            outer_context_id=0,
            centroid=[0.1] * DIM,
            member_ids=[],
        )


def test_pglyph_np_centroid():
    import numpy as np
    pg = PGlyph(
        pglyph_id=str(uuid.uuid4()),
        glyph_id=1,
        outer_context_id=0,
        centroid=[1.0, 2.0, 3.0, 4.0],
        member_ids=[],
    )
    arr = pg.np_centroid
    assert arr.dtype == np.float32
    assert float(arr[2]) == pytest.approx(3.0)
