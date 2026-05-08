"""
Unit tests for the Document Similarity Engine.

Tests TF-IDF vectorization, cosine similarity, Jaccard similarity,
similarity matrix computation, and duplicate detection.
"""

from __future__ import annotations

import math

import pytest

from src.similarity import (
    SimilarityEngine,
    SimilarityError,
    SimilarityMatrix,
    SimilarityResult,
    TFIDFVectorizer,
)


class TestTFIDFVectorizer:
    """Tests for the TFIDFVectorizer."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        vectorizer = TFIDFVectorizer()
        assert vectorizer.sublinear_tf is True
        assert vectorizer.normalize is True
        assert len(vectorizer.stop_words) > 0

    def test_tokenize(self) -> None:
        """Test text tokenization."""
        vectorizer = TFIDFVectorizer()
        tokens = vectorizer._tokenize("The quick brown fox jumps")
        assert "the" not in tokens  # stop word removed
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_tokenize_empty(self) -> None:
        """Test tokenization of empty string."""
        vectorizer = TFIDFVectorizer()
        tokens = vectorizer._tokenize("")
        assert tokens == []

    def test_tokenize_removes_short_words(self) -> None:
        """Test that short words are removed."""
        vectorizer = TFIDFVectorizer()
        tokens = vectorizer._tokenize("a an the cat")
        assert "a" not in tokens
        assert "an" not in tokens
        assert "cat" in tokens

    def test_fit(self) -> None:
        """Test fitting the vectorizer."""
        vectorizer = TFIDFVectorizer()
        documents = [
            "the quick brown fox",
            "the lazy dog sleeps",
            "quick dog jumps high",
        ]
        result = vectorizer.fit(documents)
        assert result is vectorizer
        assert len(vectorizer._vocabulary) > 0
        assert len(vectorizer._idf) > 0

    def test_fit_empty_raises(self) -> None:
        """Test that fitting on empty list raises error."""
        vectorizer = TFIDFVectorizer()
        with pytest.raises(SimilarityError):
            vectorizer.fit([])

    def test_transform(self) -> None:
        """Test transforming a document."""
        vectorizer = TFIDFVectorizer()
        vectorizer.fit(["the quick brown fox jumps over the lazy dog"])
        vector = vectorizer.transform("quick brown fox")

        assert isinstance(vector, dict)
        assert len(vector) > 0
        # All values should be non-negative
        assert all(v >= 0 for v in vector.values())

    def test_transform_before_fit_raises(self) -> None:
        """Test that transforming before fitting raises error."""
        vectorizer = TFIDFVectorizer()
        with pytest.raises(SimilarityError):
            vectorizer.transform("some text")

    def test_fit_transform(self) -> None:
        """Test fit and transform together."""
        vectorizer = TFIDFVectorizer()
        documents = [
            "machine learning is fascinating",
            "deep learning and neural networks",
            "natural language processing with machine learning",
        ]
        vectors = vectorizer.fit_transform(documents)

        assert len(vectors) == 3
        for vec in vectors:
            assert isinstance(vec, dict)
            assert len(vec) > 0

    def test_normalization(self) -> None:
        """Test that vectors are L2-normalized."""
        vectorizer = TFIDFVectorizer()
        vectorizer.fit(["the quick brown fox jumps over the lazy dog"])
        vector = vectorizer.transform("quick brown fox")

        # L2 norm should be ~1
        norm = math.sqrt(sum(v * v for v in vector.values()))
        assert abs(norm - 1.0) < 0.01

    def test_ngram_extraction(self) -> None:
        """Test n-gram extraction."""
        vectorizer = TFIDFVectorizer(ngram_range=(1, 2))
        tokens = ["machine", "learning", "is", "great"]
        ngrams = vectorizer._extract_ngrams(tokens)

        # Should have unigrams and bigrams
        assert "machine" in ngrams
        assert "machine learning" in ngrams
        assert "learning is" in ngrams
        assert "is great" in ngrams


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = SimilarityResult(
            doc_a_id="doc1",
            doc_b_id="doc2",
            similarity_score=0.85,
            method="cosine",
            shared_terms=["term1", "term2"],
            processing_time_ms=12.5,
        )
        d = result.to_dict()
        assert d["doc_a"] == "doc1"
        assert d["doc_b"] == "doc2"
        assert d["similarity"] == 0.85
        assert d["method"] == "cosine"


class TestSimilarityMatrix:
    """Tests for SimilarityMatrix dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        matrix = SimilarityMatrix(
            document_ids=["a", "b"],
            scores=[[1.0, 0.5], [0.5, 1.0]],
        )
        d = matrix.to_dict()
        assert d["document_ids"] == ["a", "b"]
        assert len(d["matrix"]) == 2

    def test_get_pairwise(self) -> None:
        """Test extracting pairwise results."""
        matrix = SimilarityMatrix(
            document_ids=["a", "b", "c"],
            scores=[[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]],
        )
        pairs = matrix.get_pairwise()
        assert len(pairs) == 3  # C(3,2) = 3
        assert pairs[0].doc_a_id == "a"
        assert pairs[0].doc_b_id == "b"
        assert pairs[0].similarity_score == 0.5


class TestSimilarityEngine:
    """Tests for SimilarityEngine."""

    def test_init(self) -> None:
        """Test default initialization."""
        engine = SimilarityEngine()
        assert engine.duplicate_threshold == 0.85

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        engine = SimilarityEngine(duplicate_threshold=0.95)
        assert engine.duplicate_threshold == 0.95

    def test_cosine_similarity_identical(self) -> None:
        """Test cosine similarity of identical vectors."""
        engine = SimilarityEngine()
        vec_a = {"term1": 0.6, "term2": 0.8}
        vec_b = {"term1": 0.6, "term2": 0.8}
        score = engine._cosine_similarity(vec_a, vec_b)
        assert abs(score - 1.0) < 0.01

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity of orthogonal vectors."""
        engine = SimilarityEngine()
        vec_a = {"term1": 1.0}
        vec_b = {"term2": 1.0}
        score = engine._cosine_similarity(vec_a, vec_b)
        assert abs(score - 0.0) < 0.01

    def test_cosine_similarity_empty(self) -> None:
        """Test cosine similarity with empty vectors."""
        engine = SimilarityEngine()
        score = engine._cosine_similarity({}, {})
        assert score == 0.0

    def test_jaccard_similarity_identical(self) -> None:
        """Test Jaccard similarity of identical texts."""
        engine = SimilarityEngine()
        score = engine._jaccard_similarity("hello world", "hello world")
        assert abs(score - 1.0) < 0.01

    def test_jaccard_similarity_different(self) -> None:
        """Test Jaccard similarity of completely different texts."""
        engine = SimilarityEngine()
        score = engine._jaccard_similarity(
            "aaa bbb ccc", "xxx yyy zzz"
        )
        assert score == 0.0

    def test_jaccard_similarity_empty(self) -> None:
        """Test Jaccard similarity with empty texts."""
        engine = SimilarityEngine()
        score = engine._jaccard_similarity("", "")
        assert score == 1.0

    def test_compare_cosine(self) -> None:
        """Test pairwise comparison with cosine method."""
        engine = SimilarityEngine()
        result = engine.compare(
            "doc_a", "the quick brown fox",
            "doc_b", "the quick brown fox jumps",
            method="cosine",
        )
        assert isinstance(result, SimilarityResult)
        assert result.doc_a_id == "doc_a"
        assert result.doc_b_id == "doc_b"
        assert result.method == "cosine"
        assert 0.0 <= result.similarity_score <= 1.0

    def test_compare_jaccard(self) -> None:
        """Test pairwise comparison with Jaccard method."""
        engine = SimilarityEngine()
        result = engine.compare(
            "doc_a", "hello world test",
            "doc_b", "hello world different",
            method="jaccard",
        )
        assert result.method == "jaccard"
        assert 0.0 <= result.similarity_score <= 1.0

    def test_compare_invalid_method(self) -> None:
        """Test that invalid method raises error."""
        engine = SimilarityEngine()
        with pytest.raises(SimilarityError):
            engine.compare("a", "text", "b", "text", method="invalid")

    def test_compare_batch_cosine(self) -> None:
        """Test batch comparison with cosine method."""
        engine = SimilarityEngine()
        documents = [
            ("doc1", "the quick brown fox"),
            ("doc2", "the quick brown fox jumps"),
            ("doc3", "completely different content here"),
        ]
        matrix = engine.compare_batch(documents, method="cosine")

        assert isinstance(matrix, SimilarityMatrix)
        assert len(matrix.document_ids) == 3
        assert len(matrix.scores) == 3
        # Diagonal should be 1.0
        for i in range(3):
            assert abs(matrix.scores[i][i] - 1.0) < 0.01

    def test_compare_batch_jaccard(self) -> None:
        """Test batch comparison with Jaccard method."""
        engine = SimilarityEngine()
        documents = [
            ("doc1", "hello world"),
            ("doc2", "hello world"),
        ]
        matrix = engine.compare_batch(documents, method="jaccard")
        assert abs(matrix.scores[0][1] - 1.0) < 0.01

    def test_compare_batch_empty_raises(self) -> None:
        """Test that empty document list raises error."""
        engine = SimilarityEngine()
        with pytest.raises(SimilarityError):
            engine.compare_batch([])

    def test_find_duplicates(self) -> None:
        """Test duplicate detection."""
        engine = SimilarityEngine(duplicate_threshold=0.8)
        documents = [
            ("doc1", "the quick brown fox jumps over the lazy dog"),
            ("doc2", "the quick brown fox jumps over the lazy dog"),
            ("doc3", "completely different document content here"),
        ]
        duplicates = engine.find_duplicates(documents)

        assert len(duplicates) >= 1
        # doc1 and doc2 should be detected as duplicates
        pair = duplicates[0]
        assert pair[0] in ("doc1", "doc2")
        assert pair[1] in ("doc1", "doc2")
        assert pair[2] >= 0.8

    def test_find_no_duplicates(self) -> None:
        """Test with documents that are not duplicates."""
        engine = SimilarityEngine(duplicate_threshold=0.95)
        documents = [
            ("doc1", "machine learning artificial intelligence"),
            ("doc2", "deep learning neural networks"),
            ("doc3", "natural language processing"),
        ]
        duplicates = engine.find_duplicates(documents)
        # At high threshold, should find no duplicates
        assert len(duplicates) == 0

    def test_find_most_similar(self) -> None:
        """Test finding most similar documents."""
        engine = SimilarityEngine()
        query = "the quick brown fox"
        candidates = [
            ("doc1", "the quick brown fox jumps"),
            ("doc2", "lazy dog sleeps all day"),
            ("doc3", "brown fox is quick"),
        ]
        results = engine.find_most_similar(query, candidates, top_k=2)

        assert len(results) == 2
        # First result should be most similar
        assert results[0].similarity_score >= results[1].similarity_score
        assert 0.0 <= results[0].similarity_score <= 1.0

    def test_find_most_similar_empty_candidates(self) -> None:
        """Test with empty candidate list."""
        engine = SimilarityEngine()
        results = engine.find_most_similar("query", [], top_k=5)
        assert results == []

    def test_similar_high_threshold_no_duplicates(self) -> None:
        """Test that very high threshold finds no duplicates."""
        engine = SimilarityEngine(duplicate_threshold=1.0)
        documents = [
            ("doc1", "the quick brown fox"),
            ("doc2", "the quick brown fox jumps"),
        ]
        duplicates = engine.find_duplicates(documents)
        # Should not find exact duplicates
        assert len(duplicates) == 0

    def test_similarity_score_range(self) -> None:
        """Test that all similarity scores are in [0, 1]."""
        engine = SimilarityEngine()
        documents = [
            ("a", "hello world foo bar"),
            ("b", "world foo bar baz"),
            ("c", "baz qux quux corge"),
        ]
        matrix = engine.compare_batch(documents)
        for row in matrix.scores:
            for score in row:
                assert 0.0 <= score <= 1.0

    def test_symmetric_scores(self) -> None:
        """Test that similarity matrix is symmetric."""
        engine = SimilarityEngine()
        documents = [
            ("a", "machine learning"),
            ("b", "deep learning"),
            ("c", "data science"),
        ]
        matrix = engine.compare_batch(documents)
        n = len(documents)
        for i in range(n):
            for j in range(n):
                assert abs(matrix.scores[i][j] - matrix.scores[j][i]) < 0.001
