"""
Document Type Classifier Module.

Classifies documents into categories (invoice, resume, report, letter) using
a hybrid approach: keyword-based heuristic scoring combined with a simple
TF-IDF weighted machine learning classifier. Provides confidence scores and
explainable classification results.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document type classifications."""

    INVOICE = "invoice"
    RESUME = "resume"
    REPORT = "report"
    LETTER = "letter"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of document classification."""

    predicted_type: DocumentType
    confidence: float
    scores: Dict[DocumentType, float] = field(default_factory=dict)
    method: str = "unknown"
    features_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "predicted_type": self.predicted_type.value,
            "confidence": round(self.confidence, 4),
            "scores": {
                k.value if isinstance(k, DocumentType) else k: round(v, 6)
                for k, v in self.scores.items()
            },
            "method": self.method,
            "features_used": self.features_used,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class HeuristicClassifier:
    """
    Rule-based document classifier using keyword frequency analysis.

    Each document type has a set of weighted keywords. The classifier
    scores documents based on keyword matches and selects the highest
    scoring category.
    """

    # Keyword patterns with weights for each document type
    KEYWORD_PROFILES: Dict[DocumentType, Dict[str, float]] = {
        DocumentType.INVOICE: {
            # Financial terms
            "invoice": 3.0,
            "invoice number": 4.0,
            "inv #": 3.5,
            "invoice#": 3.5,
            "invoice date": 3.0,
            "due date": 2.5,
            "payment due": 3.0,
            "balance due": 3.0,
            "total amount": 2.5,
            "subtotal": 2.5,
            "tax": 1.5,
            "vat": 1.5,
            "gst": 1.5,
            "quantity": 1.5,
            "unit price": 2.5,
            "item description": 2.0,
            "bill to": 2.5,
            "ship to": 2.0,
            "purchase order": 2.0,
            "po number": 2.0,
            "line item": 2.0,
            "amount due": 3.0,
            "net total": 2.5,
            "gross total": 2.5,
            "discount": 1.5,
            "credit note": 2.0,
            "remittance": 2.0,
            "bank transfer": 1.5,
            "wire transfer": 1.5,
            "iban": 2.0,
            "swift": 2.0,
            "routing number": 1.5,
            "account number": 1.0,
            "payment terms": 2.5,
            "terms net": 2.0,
        },
        DocumentType.RESUME: {
            # Career and education terms
            "resume": 3.0,
            "curriculum vitae": 3.5,
            "cv": 2.0,
            "objective": 2.0,
            "summary": 1.0,
            "professional summary": 2.5,
            "experience": 1.5,
            "work experience": 2.5,
            "professional experience": 2.5,
            "employment history": 2.5,
            "education": 2.0,
            "academic background": 2.0,
            "skills": 1.5,
            "technical skills": 2.0,
            "core competencies": 2.0,
            "qualifications": 1.5,
            "certifications": 2.0,
            "languages": 1.0,
            "references": 1.5,
            "contact information": 1.5,
            "phone": 0.5,
            "email": 0.5,
            "linkedin": 1.5,
            "bachelor": 1.5,
            "master": 1.5,
            "phd": 1.5,
            "university": 1.0,
            "college": 1.0,
            "gpa": 2.0,
            "honors": 1.5,
            "scholarship": 1.0,
            "internship": 1.5,
            "project": 0.5,
            "publication": 1.0,
        },
        DocumentType.REPORT: {
            # Analytical and business terms
            "report": 3.0,
            "annual report": 4.0,
            "quarterly": 2.0,
            "executive summary": 3.0,
            "introduction": 1.0,
            "methodology": 2.5,
            "findings": 2.0,
            "results": 1.5,
            "analysis": 1.5,
            "data analysis": 2.5,
            "market analysis": 2.5,
            "conclusion": 1.5,
            "recommendations": 2.0,
            "appendix": 2.0,
            "table of contents": 2.0,
            "figures": 1.5,
            "tables": 1.5,
            "abstract": 2.0,
            "literature review": 2.5,
            "survey": 1.5,
            "forecast": 2.0,
            "trend": 1.5,
            "growth rate": 2.0,
            "revenue": 1.5,
            "profit": 1.5,
            "loss": 1.0,
            "shareholders": 1.5,
            "stakeholders": 1.5,
            "board of directors": 2.0,
            "ceo": 1.0,
            "cfo": 1.5,
            "performance": 1.0,
            "kpi": 2.0,
            "metrics": 1.5,
        },
        DocumentType.LETTER: {
            # Correspondence terms
            "dear": 3.0,
            "to whom it may concern": 3.5,
            "sincerely": 3.0,
            "yours sincerely": 3.5,
            "yours faithfully": 3.5,
            "best regards": 2.5,
            "kind regards": 2.5,
            "regards": 1.5,
            "respectfully": 2.0,
            "letter": 2.0,
            "correspondence": 2.0,
            "subject": 1.5,
            "re:": 1.5,
            "regarding": 1.0,
            "reference": 1.0,
            "enclosure": 2.5,
            "attachment": 1.5,
            "cc:": 2.0,
            "carbon copy": 2.0,
            "bcc": 1.5,
        },
    }

    def __init__(self) -> None:
        """Initialize the heuristic classifier with compiled regex patterns."""
        self._compiled: Dict[DocumentType, List[Tuple[str, float, re.Pattern]]] = {}
        for doc_type, keywords in self.KEYWORD_PROFILES.items():
            self._compiled[doc_type] = [
                (kw, weight, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
                for kw, weight in keywords.items()
            ]

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document based on keyword heuristic scoring.

        Args:
            text: Full document text content.

        Returns:
            ClassificationResult with predicted type and confidence scores.
        """
        import time

        start = time.perf_counter()
        text_lower = text.lower()
        scores: Dict[DocumentType, float] = {t: 0.0 for t in DocumentType if t != DocumentType.UNKNOWN}
        matched_features: List[str] = []

        for doc_type, patterns in self._compiled.items():
            for keyword, weight, pattern in patterns:
                matches = len(pattern.findall(text_lower))
                if matches > 0:
                    scores[doc_type] += matches * weight
                    matched_features.append(f"{keyword}({matches})")

        # Normalize scores by document length to avoid bias toward long documents
        word_count = len(text.split())
        if word_count > 0:
            for doc_type in scores:
                scores[doc_type] /= math.sqrt(word_count)

        # Apply softmax to get confidence-like probabilities
        score_values = list(scores.values())
        max_score = max(score_values) if score_values else 0

        if max_score == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                predicted_type=DocumentType.UNKNOWN,
                confidence=0.0,
                scores={**scores, DocumentType.UNKNOWN: 1.0},
                method="heuristic",
                features_used=[],
                processing_time_ms=elapsed,
            )

        # Softmax normalization
        exp_scores = {t: math.exp(s - max_score) for t, s in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {t: s / total for t, s in exp_scores.items()}

        predicted = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted]

        elapsed = (time.perf_counter() - start) * 1000

        return ClassificationResult(
            predicted_type=predicted,
            confidence=confidence,
            scores=probabilities,
            method="heuristic",
            features_used=matched_features[:20],
            processing_time_ms=elapsed,
        )


class TFIDFClassifier:
    """
    Simple TF-IDF based classifier for document type classification.

    Uses a small, curated training corpus to compute TF-IDF vectors
    for each document type and classifies new documents by cosine
    similarity to the prototype vectors.
    """

    def __init__(self) -> None:
        """Initialize the TF-IDF classifier with built-in training data."""
        self._prototypes: Dict[DocumentType, Dict[str, float]] = {}
        self._idf: Dict[str, float] = {}
        self._vocabulary: Set[str] = set()
        self._trained = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer: lowercase, extract alphabetic tokens.

        Args:
            text: Raw text string.

        Returns:
            List of token strings.
        """
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        tokens = [t for t in text.split() if len(t) > 2 and not t.isdigit()]
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Compute term frequency for a token list.

        Args:
            tokens: List of tokens.

        Returns:
            Dictionary of term -> frequency.
        """
        if not tokens:
            return {}
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(tokens)
        return {t: count / total for t, count in freq.items()}

    def train(self, training_samples: Dict[DocumentType, List[str]]) -> None:
        """
        Train the TF-IDF classifier on labeled document samples.

        Args:
            training_samples: Mapping from DocumentType to list of
                training document texts.
        """
        # Build document frequency counts
        doc_freq: Dict[str, int] = {}
        type_tf_vectors: Dict[DocumentType, List[Dict[str, float]]] = {}
        total_docs = 0

        for doc_type, samples in training_samples.items():
            type_tf_vectors[doc_type] = []
            for sample_text in samples:
                tokens = self._tokenize(sample_text)
                tf = self._compute_tf(tokens)
                type_tf_vectors[doc_type].append(tf)
                unique_terms = set(tf.keys())
                for term in unique_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1
                total_docs += 1
                self._vocabulary.update(unique_terms)

        # Compute IDF
        self._idf = {
            term: math.log((total_docs + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }

        # Compute prototype TF-IDF vectors (average for each type)
        for doc_type, tf_vectors in type_tf_vectors.items():
            if not tf_vectors:
                continue

            prototype: Dict[str, float] = {}
            for tf in tf_vectors:
                for term, tf_val in tf.items():
                    tfidf = tf_val * self._idf.get(term, 0)
                    prototype[term] = prototype.get(term, 0) + tfidf

            # Average
            n = len(tf_vectors)
            self._prototypes[doc_type] = {
                term: val / n for term, val in prototype.items()
            }

        self._trained = True
        logger.info(
            "TF-IDF classifier trained: vocabulary=%d, types=%d",
            len(self._vocabulary),
            len(self._prototypes),
        )

    def _cosine_similarity(
        self, vec_a: Dict[str, float], vec_b: Dict[str, float]
    ) -> float:
        """
        Compute cosine similarity between two sparse vectors.

        Args:
            vec_a: First vector as term -> value dict.
            vec_b: Second vector as term -> value dict.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        dot = 0.0
        for term, val_a in vec_a.items():
            val_b = vec_b.get(term, 0.0)
            if val_b:
                dot += val_a * val_b

        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document using TF-IDF cosine similarity.

        Args:
            text: Document text content.

        Returns:
            ClassificationResult with similarity-based scores.
        """
        import time

        start = time.perf_counter()

        if not self._trained:
            # Auto-train with built-in samples if not explicitly trained
            self._auto_train()

        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        doc_tfidf = {term: tf_val * self._idf.get(term, 0) for term, tf_val in tf.items()}

        similarities: Dict[DocumentType, float] = {}
        for doc_type, prototype in self._prototypes.items():
            sim = self._cosine_similarity(doc_tfidf, prototype)
            similarities[doc_type] = max(sim, 0.0)  # Clip negative values

        # Softmax on similarities
        values = list(similarities.values())
        if not values or max(values) == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                predicted_type=DocumentType.UNKNOWN,
                confidence=0.0,
                scores={**similarities, DocumentType.UNKNOWN: 1.0},
                method="tfidf",
                processing_time_ms=elapsed,
            )

        max_sim = max(values)
        exp_scores = {t: math.exp(s - max_sim) for t, s in similarities.items()}
        total = sum(exp_scores.values())
        probabilities = {t: s / total for t, s in exp_scores.items()}

        predicted = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted]

        elapsed = (time.perf_counter() - start) * 1000

        return ClassificationResult(
            predicted_type=predicted,
            confidence=confidence,
            scores=probabilities,
            method="tfidf",
            features_used=list(tf.keys())[:20],
            processing_time_ms=elapsed,
        )

    def _auto_train(self) -> None:
        """Train with built-in sample documents for each type."""
        from typing import Dict, List

        samples: Dict[DocumentType, List[str]] = {
            DocumentType.INVOICE: [
                "Invoice Number: INV-2024-001 Invoice Date: January 15, 2024 "
                "Bill To: Acme Corporation Ship To: 123 Main St "
                "Item Description: Consulting Services Quantity: 10 "
                "Unit Price: $150.00 Subtotal: $1,500.00 Tax: $105.00 "
                "Total Amount: $1,605.00 Due Date: February 15, 2024 "
                "Payment Terms: Net 30 Please remit payment within 30 days.",
                "INVOICE Invoice #: 4421 Date: 2024-03-10 Vendor: TechSupply Inc "
                "Purchase Order: PO-8921 Line Items: Software License x5 @ $299 "
                "Support Package x1 @ $1,200 Subtotal $2,695 VAT 20% $539 "
                "Grand Total $3,234 Balance Due Upon Receipt.",
            ],
            DocumentType.RESUME: [
                "John Smith Resume Professional Summary: Experienced software "
                "engineer with 8 years in full-stack development. Skills: Python, "
                "JavaScript, AWS, Docker. Work Experience: Senior Developer at "
                "TechCorp 2019-Present. Education: BS Computer Science, State "
                "University 2016. Certifications: AWS Solutions Architect.",
                "JANE DOE CURRICULUM VITAE Objective: Seeking a research position "
                "in machine learning. Education: PhD Computer Science, MIT 2023. "
                "Publications: 12 peer-reviewed papers. Languages: English, Mandarin. "
                "References available upon request. Core Competencies: Deep Learning, "
                "NLP, Computer Vision.",
            ],
            DocumentType.REPORT: [
                "Annual Report 2024 Executive Summary: This report details company "
                "performance for fiscal year 2024. Revenue increased 15% year-over-year "
                "to $45M. Methodology: Data collected from all business units. "
                "Findings: Q4 showed strongest growth. Recommendations: Expand "
                "into Asian markets. Appendix: Financial statements attached.",
                "Market Analysis Report Introduction: This analysis examines trends "
                "in renewable energy. Data Analysis: Solar adoption grew 28% in 2023. "
                "Forecast: Continued growth expected through 2030. Conclusion: "
                "Investment opportunity in distributed solar. Stakeholders should "
                "review the figures in Table 3.",
            ],
            DocumentType.LETTER: [
                "Dear Mr. Johnson, I am writing regarding the proposal submitted "
                "on March 1st. We are pleased to accept your terms. Please find "
                "the signed agreement enclosed. Sincerely, Mary Williams.",
                "To Whom It May Concern, This letter serves as formal notice "
                "of our intent to renew the lease agreement. Subject: Lease "
                "Renewal 2024-2025. Please contact me at your earliest convenience. "
                "Best regards, Robert Chen. CC: Legal Department.",
            ],
        }
        self.train(samples)


class DocumentClassifier:
    """
    Unified document classifier combining heuristic and ML-based methods.

    Uses ensemble scoring to combine predictions from multiple
    classification strategies for robust, explainable results.
    """

    def __init__(
        self,
        heuristic_weight: float = 0.5,
        tfidf_weight: float = 0.5,
    ) -> None:
        """
        Initialize the document classifier.

        Args:
            heuristic_weight: Weight for heuristic classifier (0-1).
            tfidf_weight: Weight for TF-IDF classifier (0-1).
        """
        self.heuristic = HeuristicClassifier()
        self.tfidf = TFIDFClassifier()
        self.heuristic_weight = heuristic_weight
        self.tfidf_weight = tfidf_weight

    def classify(
        self, text: str, method: str = "ensemble"
    ) -> ClassificationResult:
        """
        Classify a document text into a document type.

        Args:
            text: The full document text to classify.
            method: Classification method ('heuristic', 'tfidf', or 'ensemble').

        Returns:
            ClassificationResult with predicted type and confidence.

        Raises:
            ValueError: If method is not recognized or text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot classify empty text")

        if method == "heuristic":
            result = self.heuristic.classify(text)
            return result
        elif method == "tfidf":
            result = self.tfidf.classify(text)
            return result
        elif method == "ensemble":
            return self._ensemble_classify(text)
        else:
            raise ValueError(f"Unknown classification method: {method}")

    def _ensemble_classify(self, text: str) -> ClassificationResult:
        """
        Combine heuristic and TF-IDF predictions via weighted voting.

        Args:
            text: Document text.

        Returns:
            ClassificationResult with ensemble scores.
        """
        import time

        start = time.perf_counter()

        h_result = self.heuristic.classify(text)
        t_result = self.tfidf.classify(text)

        # Combine scores
        all_types = set(h_result.scores.keys()) | set(t_result.scores.keys())
        ensemble_scores: Dict[DocumentType, float] = {}

        for doc_type in all_types:
            h_score = h_result.scores.get(doc_type, 0.0)
            t_score = t_result.scores.get(doc_type, 0.0)
            ensemble_scores[doc_type] = (
                self.heuristic_weight * h_score
                + self.tfidf_weight * t_score
            )

        # Normalize to probabilities
        total = sum(ensemble_scores.values())
        if total > 0:
            ensemble_scores = {t: s / total for t, s in ensemble_scores.items()}

        predicted = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[predicted]

        elapsed = (time.perf_counter() - start) * 1000

        return ClassificationResult(
            predicted_type=predicted,
            confidence=confidence,
            scores=ensemble_scores,
            method="ensemble",
            features_used=h_result.features_used[:10],
            processing_time_ms=elapsed,
        )
