"""
Document Similarity Engine.

Computes similarity between documents using TF-IDF vectorization
and cosine similarity. Supports pairwise comparisons, similarity
matrix generation, and near-duplicate detection.

The module implements efficient sparse vector representations and
provides both exact and approximate similarity computation methods.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SimilarityError(Exception):
    """Raised when similarity computation fails."""

    pass


@dataclass
class SimilarityResult:
    """Result of a similarity computation between two documents."""

    doc_a_id: str
    doc_b_id: str
    similarity_score: float
    method: str = "cosine"
    shared_terms: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "doc_a": self.doc_a_id,
            "doc_b": self.doc_b_id,
            "similarity": round(self.similarity_score, 6),
            "method": self.method,
            "shared_terms": self.shared_terms[:20],
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class SimilarityMatrix:
    """Similarity matrix for a collection of documents."""

    document_ids: List[str]
    scores: List[List[float]]
    method: str = "cosine"

    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to serializable dictionary."""
        return {
            "document_ids": self.document_ids,
            "matrix": [
                [round(s, 4) for s in row] for row in self.scores
            ],
            "method": self.method,
        }

    def get_pairwise(self) -> List[SimilarityResult]:
        """
        Get all pairwise similarity results from the matrix.

        Returns:
            List of SimilarityResult for each unique pair.
        """
        results = []
        n = len(self.document_ids)
        for i in range(n):
            for j in range(i + 1, n):
                results.append(
                    SimilarityResult(
                        doc_a_id=self.document_ids[i],
                        doc_b_id=self.document_ids[j],
                        similarity_score=self.scores[i][j],
                        method=self.method,
                    )
                )
        return results


class TFIDFVectorizer:
    """
    Custom TF-IDF vectorizer with configurable parameters.

    Tokenizes text, computes term frequencies, and applies
    TF-IDF weighting for document vectorization.
    """

    # Common English stop words
    DEFAULT_STOP_WORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "until", "while", "this", "that", "these", "those",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "am",
    }

    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        min_df: int = 1,
        max_df_ratio: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 1),
        sublinear_tf: bool = True,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the TF-IDF vectorizer.

        Args:
            stop_words: Set of stop words to exclude.
            min_df: Minimum document frequency for a term.
            max_df_ratio: Maximum document frequency ratio (0-1).
            ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
            sublinear_tf: Whether to use sublinear TF scaling (1 + log(tf)).
            normalize: Whether to L2-normalize output vectors.
        """
        self.stop_words = stop_words or self.DEFAULT_STOP_WORDS
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.normalize = normalize
        self._idf: Dict[str, float] = {}
        self._vocabulary: List[str] = []
        self._is_fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Raw text.

        Returns:
            List of tokens.
        """
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        tokens = [t for t in text.split() if len(t) > 2 and t not in self.stop_words]
        return tokens

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """
        Extract n-grams from tokens.

        Args:
            tokens: List of tokens.

        Returns:
            List of n-gram strings.
        """
        result = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                result.append(ngram)
        return result

    def _compute_tf(self, terms: List[str]) -> Dict[str, float]:
        """
        Compute term frequency for a list of terms.

        Args:
            terms: List of terms.

        Returns:
            Dictionary of term -> frequency.
        """
        if not terms:
            return {}
        freq: Dict[str, int] = {}
        for term in terms:
            freq[term] = freq.get(term, 0) + 1

        if self.sublinear_tf:
            return {term: 1 + math.log(count) for term, count in freq.items()}
        else:
            total = len(terms)
            return {term: count / total for term, count in freq.items()}

    def fit(self, documents: List[str]) -> TFIDFVectorizer:
        """
        Fit the vectorizer on a corpus of documents.

        Args:
            documents: List of document texts.

        Returns:
            Self for method chaining.
        """
        if not documents:
            raise SimilarityError("Cannot fit on empty document list")

        # Compute document frequency for each term
        doc_freq: Dict[str, int] = {}
        doc_terms: List[List[str]] = []
        total_docs = len(documents)

        for doc in documents:
            tokens = self._tokenize(doc)
            terms = self._extract_ngrams(tokens)
            doc_terms.append(terms)
            unique_terms = set(terms)
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Filter by min/max document frequency
        max_df = max(1, int(total_docs * self.max_df_ratio))
        filtered_terms = [
            term
            for term, freq in doc_freq.items()
            if self.min_df <= freq <= max_df
        ]

        # Compute IDF
        self._vocabulary = sorted(filtered_terms)
        self._idf = {
            term: math.log((total_docs + 1) / (doc_freq.get(term, 1) + 1)) + 1
            for term in self._vocabulary
        }
        self._is_fitted = True

        logger.info(
            "TF-IDF fitted: vocabulary=%d, documents=%d",
            len(self._vocabulary),
            total_docs,
        )
        return self

    def transform(self, text: str) -> Dict[str, float]:
        """
        Transform a single document into a TF-IDF vector.

        Args:
            text: Document text.

        Returns:
            Dictionary of term -> TF-IDF weight.
        """
        if not self._is_fitted:
            raise SimilarityError("Vectorizer must be fitted before transform")

        tokens = self._tokenize(text)
        terms = self._extract_ngrams(tokens)
        tf = self._compute_tf(terms)

        # Apply IDF weighting
        vector = {
            term: tf.get(term, 0) * self._idf.get(term, 0)
            for term in self._vocabulary
        }

        # Remove zero-weight entries for sparsity
        vector = {t: w for t, w in vector.items() if w > 0}

        if self.normalize:
            norm = math.sqrt(sum(w * w for w in vector.values()))
            if norm > 0:
                vector = {t: w / norm for t, w in vector.items()}

        return vector

    def fit_transform(self, documents: List[str]) -> List[Dict[str, float]]:
        """
        Fit and transform a list of documents.

        Args:
            documents: List of document texts.

        Returns:
            List of TF-IDF vectors.
        """
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


class SimilarityEngine:
    """
    Document similarity engine with multiple comparison methods.

    Provides TF-IDF cosine similarity, Jaccard similarity, and
    near-duplicate detection capabilities.
    """

    def __init__(
        self,
        vectorizer: Optional[TFIDFVectorizer] = None,
        duplicate_threshold: float = 0.85,
    ) -> None:
        """
        Initialize the similarity engine.

        Args:
            vectorizer: Custom TFIDFVectorizer or None for default.
            duplicate_threshold: Score threshold for near-duplicate detection.
        """
        self.vectorizer = vectorizer or TFIDFVectorizer()
        self.duplicate_threshold = duplicate_threshold

    def _cosine_similarity(
        self, vec_a: Dict[str, float], vec_b: Dict[str, float]
    ) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score (0 to 1).
        """
        dot_product = 0.0
        for term, weight_a in vec_a.items():
            weight_b = vec_b.get(term, 0.0)
            if weight_b:
                dot_product += weight_a * weight_b

        # Vectors are already normalized, so dot product = cosine similarity
        return max(0.0, min(1.0, dot_product))

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute Jaccard similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Jaccard similarity score (0 to 1).
        """
        tokens_a = set(self.vectorizer._tokenize(text_a))
        tokens_b = set(self.vectorizer._tokenize(text_b))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        return intersection / union

    def compare(
        self,
        doc_a_id: str,
        doc_a_text: str,
        doc_b_id: str,
        doc_b_text: str,
        method: str = "cosine",
    ) -> SimilarityResult:
        """
        Compare two documents for similarity.

        Args:
            doc_a_id: Identifier for first document.
            doc_a_text: Text content of first document.
            doc_b_id: Identifier for second document.
            doc_b_text: Text content of second document.
            method: Similarity method ('cosine' or 'jaccard').

        Returns:
            SimilarityResult with computed score.
        """
        import time

        start = time.perf_counter()

        if method == "cosine":
            # Fit vectorizer on both documents
            self.vectorizer.fit([doc_a_text, doc_b_text])
            vec_a = self.vectorizer.transform(doc_a_text)
            vec_b = self.vectorizer.transform(doc_b_text)
            score = self._cosine_similarity(vec_a, vec_b)
            shared = [t for t in vec_a if t in vec_b]
        elif method == "jaccard":
            score = self._jaccard_similarity(doc_a_text, doc_b_text)
            shared = []
        else:
            raise SimilarityError(f"Unknown similarity method: {method}")

        elapsed = (time.perf_counter() - start) * 1000

        return SimilarityResult(
            doc_a_id=doc_a_id,
            doc_b_id=doc_b_id,
            similarity_score=score,
            method=method,
            shared_terms=shared,
            processing_time_ms=elapsed,
        )

    def compare_batch(
        self,
        documents: List[Tuple[str, str]],
        method: str = "cosine",
    ) -> SimilarityMatrix:
        """
        Compute similarity matrix for a collection of documents.

        Args:
            documents: List of (doc_id, doc_text) tuples.
            method: Similarity method.

        Returns:
            SimilarityMatrix with all pairwise scores.
        """
        import time

        start = time.perf_counter()

        if not documents:
            raise SimilarityError("Empty document list")

        doc_ids = [d[0] for d in documents]
        doc_texts = [d[1] for d in documents]
        n = len(documents)

        if method == "cosine":
            self.vectorizer.fit(doc_texts)
            vectors = [self.vectorizer.transform(t) for t in doc_texts]

            # Compute pairwise similarity matrix
            matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                matrix[i][i] = 1.0
                for j in range(i + 1, n):
                    score = self._cosine_similarity(vectors[i], vectors[j])
                    matrix[i][j] = score
                    matrix[j][i] = score

        elif method == "jaccard":
            matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                matrix[i][i] = 1.0
                for j in range(i + 1, n):
                    score = self._jaccard_similarity(doc_texts[i], doc_texts[j])
                    matrix[i][j] = score
                    matrix[j][i] = score
        else:
            raise SimilarityError(f"Unknown similarity method: {method}")

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Batch similarity computed for %d documents in %.2fms",
            n,
            elapsed,
        )

        return SimilarityMatrix(
            document_ids=doc_ids,
            scores=matrix,
            method=method,
        )

    def find_duplicates(
        self,
        documents: List[Tuple[str, str]],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find near-duplicate document pairs.

        Args:
            documents: List of (doc_id, doc_text) tuples.
            threshold: Similarity threshold (defaults to duplicate_threshold).

        Returns:
            List of (doc_a_id, doc_b_id, score) tuples for duplicates.
        """
        threshold = threshold or self.duplicate_threshold
        matrix = self.compare_batch(documents)

        duplicates = []
        n = len(documents)
        for i in range(n):
            for j in range(i + 1, n):
                score = matrix.scores[i][j]
                if score >= threshold:
                    duplicates.append((documents[i][0], documents[j][0], score))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        logger.info(
            "Found %d duplicate pairs with threshold %.2f",
            len(duplicates),
            threshold,
        )

        return duplicates

    def find_most_similar(
        self,
        query_text: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[SimilarityResult]:
        """
        Find the most similar documents to a query document.

        Args:
            query_text: Query document text.
            candidates: List of (doc_id, doc_text) candidate documents.
            top_k: Number of top results to return.

        Returns:
            List of SimilarityResult for top-k matches.
        """
        if not candidates:
            return []

        # Build combined corpus for fitting
        all_texts = [query_text] + [c[1] for c in candidates]
        self.vectorizer.fit(all_texts)

        query_vec = self.vectorizer.transform(query_text)
        candidate_vecs = [self.vectorizer.transform(c[1]) for c in candidates]

        results = []
        for (doc_id, _), cand_vec in zip(candidates, candidate_vecs):
            score = self._cosine_similarity(query_vec, cand_vec)
            results.append(
                SimilarityResult(
                    doc_a_id="query",
                    doc_b_id=doc_id,
                    similarity_score=score,
                    method="cosine",
                )
            )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]
