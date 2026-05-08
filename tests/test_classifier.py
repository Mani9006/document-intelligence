"""
Unit tests for the Document Type Classifier Module.

Tests heuristic classification, TF-IDF classification, ensemble
classification, and edge cases.
"""

from __future__ import annotations

import pytest

from src.classifier import (
    ClassificationResult,
    DocumentClassifier,
    DocumentType,
    HeuristicClassifier,
    TFIDFClassifier,
)


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = ClassificationResult(
            predicted_type=DocumentType.INVOICE,
            confidence=0.92,
            scores={DocumentType.INVOICE: 0.92, DocumentType.REPORT: 0.08},
            method="heuristic",
            features_used=["invoice(2)", "total(1)"],
            processing_time_ms=15.5,
        )
        d = result.to_dict()
        assert d["predicted_type"] == "invoice"
        assert d["confidence"] == 0.92
        assert d["method"] == "heuristic"
        assert len(d["features_used"]) == 2

    def test_empty_features(self) -> None:
        """Test result with no features."""
        result = ClassificationResult(
            predicted_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="test",
            processing_time_ms=0.0,
        )
        d = result.to_dict()
        assert d["predicted_type"] == "unknown"
        assert d["features_used"] == []


class TestHeuristicClassifier:
    """Tests for the HeuristicClassifier."""

    def test_invoice_detection(self) -> None:
        """Test invoice document classification."""
        text = (
            "Invoice Number: INV-2024-001\n"
            "Invoice Date: January 15, 2024\n"
            "Bill To: Acme Corporation\n"
            "Item: Consulting Services\n"
            "Quantity: 10  Unit Price: $150.00\n"
            "Subtotal: $1,500.00  Tax: $105.00\n"
            "Total Amount: $1,605.00\n"
            "Due Date: February 15, 2024"
        )
        classifier = HeuristicClassifier()
        result = classifier.classify(text)

        assert result.predicted_type == DocumentType.INVOICE
        assert result.confidence > 0.5
        assert result.method == "heuristic"
        assert len(result.features_used) > 0

    def test_resume_detection(self) -> None:
        """Test resume document classification."""
        text = (
            "John Smith - Resume\n\n"
            "Professional Summary: Experienced software engineer\n"
            "Skills: Python, JavaScript, AWS\n"
            "Work Experience: Senior Developer at TechCorp 2019-Present\n"
            "Education: BS Computer Science, State University\n"
            "Certifications: AWS Solutions Architect"
        )
        classifier = HeuristicClassifier()
        result = classifier.classify(text)

        assert result.predicted_type == DocumentType.RESUME
        assert result.confidence > 0.5
        assert result.method == "heuristic"

    def test_report_detection(self) -> None:
        """Test report document classification."""
        text = (
            "Annual Report 2024\n\n"
            "Executive Summary: This report details company performance\n"
            "Methodology: Data collected from all business units\n"
            "Findings: Q4 showed strongest growth\n"
            "Analysis: Revenue increased 15% year-over-year\n"
            "Recommendations: Expand into Asian markets\n"
            "Conclusion: Strong fiscal year performance"
        )
        classifier = HeuristicClassifier()
        result = classifier.classify(text)

        assert result.predicted_type == DocumentType.REPORT
        assert result.confidence > 0.5

    def test_letter_detection(self) -> None:
        """Test letter document classification."""
        text = (
            "Dear Mr. Johnson,\n\n"
            "I am writing regarding the proposal submitted on March 1st.\n"
            "We are pleased to accept your terms.\n\n"
            "Sincerely,\n"
            "Mary Williams"
        )
        classifier = HeuristicClassifier()
        result = classifier.classify(text)

        assert result.predicted_type == DocumentType.LETTER
        assert result.confidence > 0.5

    def test_empty_text_returns_unknown(self) -> None:
        """Test that empty text returns UNKNOWN."""
        classifier = HeuristicClassifier()
        result = classifier.classify("")
        assert result.predicted_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0

    def test_unknown_content(self) -> None:
        """Test text with no matching keywords."""
        classifier = HeuristicClassifier()
        result = classifier.classify(
            "Random text about nothing specific at all."
        )
        assert result.predicted_type == DocumentType.UNKNOWN

    def test_scores_sum_to_one(self) -> None:
        """Test that probability scores sum to approximately 1."""
        text = "Invoice #123 Total Amount $500 Due Date Jan 1"
        classifier = HeuristicClassifier()
        result = classifier.classify(text)
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01


class TestTFIDFClassifier:
    """Tests for the TFIDFClassifier."""

    def test_train_and_classify(self) -> None:
        """Test training and classifying a document."""
        classifier = TFIDFClassifier()
        classifier._auto_train()

        text = (
            "Invoice INV-001. Total: $500. Tax: $50. "
            "Balance Due: $550. Payment Terms: Net 30"
        )
        result = classifier.classify(text)

        assert result.method == "tfidf"
        assert result.predicted_type in DocumentType
        assert result.confidence >= 0.0

    def test_classify_before_train(self) -> None:
        """Test classification auto-trains if not trained."""
        classifier = TFIDFClassifier()
        text = "Invoice total amount $500 payment due"
        result = classifier.classify(text)
        assert result.method == "tfidf"
        assert result.predicted_type in DocumentType

    def test_train_custom_samples(self) -> None:
        """Test training with custom samples."""
        classifier = TFIDFClassifier()
        samples = {
            DocumentType.INVOICE: [
                "invoice number 123 total amount $500",
                "invoice payment due date balance",
            ],
            DocumentType.RESUME: [
                "resume skills education experience",
                "curriculum vitae university degree",
            ],
        }
        classifier.train(samples)

        result = classifier.classify("Invoice total payment due")
        assert result.method == "tfidf"
        # Scores should be meaningful
        assert len(result.scores) > 0

    def test_scores_are_probabilities(self) -> None:
        """Test that scores form valid probability distribution."""
        classifier = TFIDFClassifier()
        classifier._auto_train()

        text = "Dear Sir, please find the invoice attached. Total: $100."
        result = classifier.classify(text)

        # Scores should be non-negative and sum to ~1
        assert all(s >= 0 for s in result.scores.values())
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01


class TestDocumentClassifier:
    """Tests for the unified DocumentClassifier."""

    def test_ensemble_invoice(self) -> None:
        """Test ensemble classification of invoice."""
        classifier = DocumentClassifier()
        text = (
            "Invoice #123. Bill To: ABC Corp. "
            "Subtotal: $1,000. Tax: $70. Total Amount: $1,070. "
            "Due Date: 2024-01-15. Payment Terms: Net 30."
        )
        result = classifier.classify(text, method="ensemble")

        assert result.predicted_type == DocumentType.INVOICE
        assert result.method == "ensemble"
        assert result.confidence > 0.0
        assert len(result.scores) > 0

    def test_ensemble_resume(self) -> None:
        """Test ensemble classification of resume."""
        classifier = DocumentClassifier()
        text = (
            "Resume of Jane Doe. Skills: Python, ML, Data Science. "
            "Education: PhD, MIT. Experience: 5 years at Google. "
            "Certifications: AWS, GCP. References available."
        )
        result = classifier.classify(text, method="ensemble")

        assert result.predicted_type == DocumentType.RESUME
        assert result.confidence > 0.0

    def test_heuristic_method(self) -> None:
        """Test heuristic-only classification."""
        classifier = DocumentClassifier()
        text = "Invoice number 456 total $500 due date"
        result = classifier.classify(text, method="heuristic")
        assert result.method == "heuristic"

    def test_tfidf_method(self) -> None:
        """Test TF-IDF-only classification."""
        classifier = DocumentClassifier()
        text = "Resume education skills experience university degree"
        result = classifier.classify(text, method="tfidf")
        assert result.method == "tfidf"

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises ValueError."""
        classifier = DocumentClassifier()
        with pytest.raises(ValueError):
            classifier.classify("")

    def test_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only text raises ValueError."""
        classifier = DocumentClassifier()
        with pytest.raises(ValueError):
            classifier.classify("   \n\t  ")

    def test_invalid_method(self) -> None:
        """Test that invalid method raises ValueError."""
        classifier = DocumentClassifier()
        with pytest.raises(ValueError):
            classifier.classify("Some text", method="invalid")

    def test_scores_normalize_to_one(self) -> None:
        """Test that ensemble scores sum to ~1."""
        classifier = DocumentClassifier()
        text = (
            "Invoice #123. Bill To: ABC Corp. Total: $500.\n"
            "Dear Sir, please process this invoice."
        )
        result = classifier.classify(text, method="ensemble")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01

    def test_report_classification(self) -> None:
        """Test report classification."""
        classifier = DocumentClassifier()
        text = (
            "Annual Report 2024. Executive Summary. "
            "Revenue increased 15%. Methodology. Findings. "
            "Recommendations. Conclusion. Appendix A."
        )
        result = classifier.classify(text, method="ensemble")
        assert result.predicted_type == DocumentType.REPORT

    def test_letter_classification(self) -> None:
        """Test letter classification."""
        classifier = DocumentClassifier()
        text = (
            "Dear Mr. Smith, I am writing regarding the meeting. "
            "Please review the attached documents. "
            "Sincerely, John Doe."
        )
        result = classifier.classify(text, method="ensemble")
        assert result.predicted_type == DocumentType.LETTER
