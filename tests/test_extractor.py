"""
Unit tests for the PDF Text Extraction Module.

Tests all extraction backends, error handling, and the unified
DocumentExtractor interface.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.extractor import (
    DocumentExtractor,
    ExtractionError,
    ExtractionResult,
    ExtractedPage,
    PyPDF2Backend,
    TextFileBackend,
    UnsupportedFileError,
)


class TestExtractedPage:
    """Tests for the ExtractedPage dataclass."""

    def test_empty_text(self) -> None:
        """Test page with empty text."""
        page = ExtractedPage(page_number=1, text="")
        assert page.word_count == 0
        assert page.line_count == 0
        assert page.avg_word_length == 0.0
        assert not page.has_numeric_content

    def test_basic_text(self) -> None:
        """Test page with basic text."""
        page = ExtractedPage(page_number=1, text="Hello world test")
        assert page.word_count == 3
        assert page.line_count == 1
        assert page.avg_word_length == pytest.approx(4.667, 0.01)  # (5+5+4)/3
        assert not page.has_numeric_content

    def test_multiline_text(self) -> None:
        """Test page with multiple lines."""
        text = "Line one\nLine two\n\nLine three"
        page = ExtractedPage(page_number=1, text=text)
        assert page.line_count == 3
        assert page.word_count == 6

    def test_numeric_detection(self) -> None:
        """Test numeric content detection."""
        page = ExtractedPage(page_number=1, text="Total: $1,234.56")
        assert page.has_numeric_content

    def test_no_numeric_detection(self) -> None:
        """Test that plain text has no numeric content."""
        page = ExtractedPage(page_number=1, text="Just words here")
        assert not page.has_numeric_content

    def test_page_metadata(self) -> None:
        """Test page with metadata."""
        page = ExtractedPage(
            page_number=2, text="Test", metadata={"key": "value"}
        )
        assert page.metadata == {"key": "value"}
        assert page.page_number == 2


class TestExtractionResult:
    """Tests for the ExtractionResult dataclass."""

    def test_total_word_count(self) -> None:
        """Test total word count across pages."""
        result = ExtractionResult(
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            file_size_bytes=1000,
            total_pages=2,
            pages=[
                ExtractedPage(page_number=1, text="Hello world"),
                ExtractedPage(page_number=2, text="More text here"),
            ],
            full_text="Hello world\nMore text here",
            success=True,
        )
        assert result.total_word_count == 5  # 2 + 3

    def test_language_hint_english(self) -> None:
        """Test English language detection."""
        result = ExtractionResult(
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            file_size_bytes=100,
            total_pages=1,
            pages=[
                ExtractedPage(page_number=1, text="The quick brown fox jumps over the lazy dog and is for the")
            ],
            full_text="The quick brown fox jumps over the lazy dog and is for the",
            success=True,
        )
        assert result.language_hint == "en"

    def test_language_hint_unknown(self) -> None:
        """Test unknown language hint for empty text."""
        result = ExtractionResult(
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            file_size_bytes=0,
            total_pages=0,
            pages=[],
            full_text="",
            success=True,
        )
        assert result.language_hint == "unknown"


class TestTextFileBackend:
    """Tests for the TextFileBackend."""

    def test_extract_txt_file(self) -> None:
        """Test extracting from a plain text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("This is test content for extraction.\n")
            f.write("Multiple lines of sample text.\n")
            f.write("Contains numbers: 123 and $456.78\n")
            temp_path = f.name

        try:
            backend = TextFileBackend()
            result = backend.extract(temp_path)

            assert result.success
            assert result.filename == Path(temp_path).name
            assert "test content" in result.full_text.lower()
            assert result.total_pages >= 1
            assert result.extraction_method == "text_file"
            assert result.total_word_count > 0
        finally:
            Path(temp_path).unlink()

    def test_extract_missing_file(self) -> None:
        """Test extracting from non-existent file raises error."""
        backend = TextFileBackend()
        with pytest.raises(ExtractionError):
            backend.extract("/nonexistent/file.txt")

    def test_extract_empty_file(self) -> None:
        """Test extracting from empty file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            backend = TextFileBackend()
            result = backend.extract(temp_path)
            assert result.success
            assert result.full_text == ""
        finally:
            Path(temp_path).unlink()

    def test_extract_md_file(self) -> None:
        """Test extracting from markdown file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Heading\n\nSome markdown content here.\n")
            temp_path = f.name

        try:
            backend = TextFileBackend()
            result = backend.extract(temp_path)
            assert result.success
            assert "Heading" in result.full_text
        finally:
            Path(temp_path).unlink()


class TestDocumentExtractor:
    """Tests for the unified DocumentExtractor."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        extractor = DocumentExtractor()
        assert extractor.preferred_backend == "auto"
        assert extractor.enable_fallback is True

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        extractor = DocumentExtractor(
            preferred_backend="text_file", enable_fallback=False
        )
        assert extractor.preferred_backend == "text_file"
        assert not extractor.enable_fallback

    def test_unsupported_extension(self) -> None:
        """Test that unsupported extensions raise error."""
        extractor = DocumentExtractor()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            with pytest.raises(UnsupportedFileError):
                extractor.extract(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_extract_txt(self) -> None:
        """Test extracting from .txt file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Document content for testing the extractor.")
            temp_path = f.name

        try:
            extractor = DocumentExtractor()
            result = extractor.extract(temp_path)
            assert result.success
            assert "extractor" in result.full_text.lower()
        finally:
            Path(temp_path).unlink()

    def test_batch_extraction(self) -> None:
        """Test batch extraction of multiple files."""
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(f"Content of document number {i + 1}.")
                temp_files.append(f.name)

        try:
            extractor = DocumentExtractor()
            results = extractor.extract_batch(temp_files)

            assert len(results) == 3
            for i, (path, result) in enumerate(results):
                assert result.success
                assert f"document number {i + 1}" in result.full_text.lower()
        finally:
            for f in temp_files:
                Path(f).unlink()

    def test_batch_with_missing_file(self) -> None:
        """Test batch extraction handles missing files gracefully."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Valid file content")
            valid_path = f.name

        try:
            extractor = DocumentExtractor()
            results = extractor.extract_batch(
                [valid_path, "/nonexistent/file.txt"]
            )

            assert len(results) == 2
            assert results[0][1].success  # Valid file
            assert not results[1][1].success  # Missing file
        finally:
            Path(valid_path).unlink()

    def test_select_backend_for_pdf(self) -> None:
        """Test backend auto-selection for PDF files."""
        extractor = DocumentExtractor()
        path = Path("test.pdf")
        backend = extractor._select_backend(path)
        assert backend in ("pdfplumber", "pypdf2", "text_file")

    def test_select_backend_for_txt(self) -> None:
        """Test backend auto-selection for text files."""
        extractor = DocumentExtractor()
        path = Path("test.txt")
        backend = extractor._select_backend(path)
        assert backend == "text_file"

    def test_extract_nonexistent_file(self) -> None:
        """Test extraction from non-existent file."""
        extractor = DocumentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract("/nonexistent/document.pdf")

    def test_extract_empty_text_result(self) -> None:
        """Test handling of file with no extractable text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            extractor = DocumentExtractor(enable_fallback=False)
            result = extractor.extract(temp_path)
            assert result.success
            assert result.full_text == ""
        finally:
            Path(temp_path).unlink()
