"""Unit tests for aeuc_vector_db.similarity."""

import numpy as np
import pytest

from aeuc_vector_db.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    euclidean_similarity,
    phi_weighted_similarity,
    similarity,
)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    # Should not raise; denominator guarded by 1e-12
    score = cosine_similarity(a, b)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# euclidean
# ---------------------------------------------------------------------------


def test_euclidean_distance_known():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0, abs=1e-6)


def test_euclidean_similarity_identical():
    a = np.array([1.0, 2.0])
    assert euclidean_similarity(a, a) == pytest.approx(1.0, abs=1e-6)


def test_euclidean_similarity_decreases_with_distance():
    origin = np.zeros(4)
    near = np.array([0.1, 0.0, 0.0, 0.0])
    far = np.array([10.0, 0.0, 0.0, 0.0])
    assert euclidean_similarity(origin, near) > euclidean_similarity(origin, far)


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------


def test_dot_product_known():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert dot_product(a, b) == pytest.approx(32.0, abs=1e-6)


# ---------------------------------------------------------------------------
# phi_weighted_similarity
# ---------------------------------------------------------------------------


def test_phi_weighted_identical():
    a = np.array([1.618, 2.618, 0.618])
    assert phi_weighted_similarity(a, a) == pytest.approx(1.0, abs=1e-5)


def test_phi_weighted_returns_float():
    a = np.random.rand(16).astype(np.float32)
    b = np.random.rand(16).astype(np.float32)
    score = phi_weighted_similarity(a, b)
    assert isinstance(score, float)
    assert -1.1 < score < 1.1


# ---------------------------------------------------------------------------
# similarity dispatch
# ---------------------------------------------------------------------------


def test_similarity_cosine():
    a = np.array([1.0, 0.0])
    assert similarity(a, a, "cosine") == pytest.approx(1.0, abs=1e-6)


def test_similarity_euclidean():
    a = np.array([1.0, 0.0])
    assert similarity(a, a, "euclidean") == pytest.approx(1.0, abs=1e-6)


def test_similarity_dot():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert similarity(a, b, "dot") == pytest.approx(11.0, abs=1e-6)


def test_similarity_phi_weighted():
    a = np.ones(8, dtype=np.float32)
    score = similarity(a, a, "phi_weighted")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_similarity_invalid_metric():
    a = np.array([1.0])
    with pytest.raises(ValueError, match="Unknown metric"):
        similarity(a, a, "bogus")  # type: ignore
