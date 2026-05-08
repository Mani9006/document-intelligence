"""
PDF Text Extraction Module.

Extracts raw and structured text from PDF documents using multiple
backend libraries (PyPDF2, pdfplumber) with fallback support.
Supports metadata extraction, page-level segmentation, and error recovery.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

# Configure module-level logger
logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when text extraction fails irrecoverably."""

    pass


class UnsupportedFileError(ExtractionError):
    """Raised when the file format is not supported."""

    pass


@dataclass
class ExtractedPage:
    """Represents a single extracted page with metadata."""

    page_number: int
    text: str
    word_count: int = 0
    line_count: int = 0
    avg_word_length: float = 0.0
    has_numeric_content: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute derived statistics after initialization."""
        if self.text:
            words = self.text.split()
            self.word_count = len(words)
            self.line_count = len([l for l in self.text.splitlines() if l.strip()])
            self.avg_word_length = (
                sum(len(w) for w in words) / len(words) if words else 0.0
            )
            self.has_numeric_content = bool(
                re.search(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", self.text)
            )


@dataclass
class ExtractionResult:
    """Complete extraction result for a document."""

    source_path: str
    filename: str
    file_size_bytes: int
    total_pages: int
    pages: List[ExtractedPage]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "unknown"
    processing_time_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None

    @property
    def total_word_count(self) -> int:
        """Return total word count across all pages."""
        return sum(p.word_count for p in self.pages)

    @property
    def language_hint(self) -> str:
        """Infer document language from character distribution."""
        if not self.full_text:
            return "unknown"
        # Simple heuristic: check common English words
        text_lower = self.full_text.lower()
        english_markers = ["the", "and", "of", "to", "a", "in", "is", "for"]
        score = sum(1 for m in english_markers if m in text_lower)
        return "en" if score >= 3 else "unknown"


class ExtractorBackend(Protocol):
    """Protocol for pluggable extraction backends."""

    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract text from the given file path."""
        ...


class PyPDF2Backend:
    """PDF text extraction using PyPDF2."""

    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from PDF using PyPDF2.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ExtractionResult with all extracted content.

        Raises:
            ExtractionError: If extraction fails.
        """
        import time

        import PyPDF2

        start_time = time.perf_counter()
        path = Path(file_path)

        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise UnsupportedFileError(f"Expected .pdf file, got: {path.suffix}")

        try:
            file_size = path.stat().st_size
            pages: List[ExtractedPage] = []
            full_text_parts: List[str] = []
            pdf_metadata: Dict[str, Any] = {}

            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                num_pages = len(reader.pages)

                # Extract document metadata if available
                if reader.metadata:
                    pdf_metadata = {
                        k: str(v) if v else None
                        for k, v in reader.metadata.items()
                    }

                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception as page_err:
                        logger.warning(
                            "Page %d extraction failed: %s", i + 1, page_err
                        )
                        text = ""

                    extracted_page = ExtractedPage(
                        page_number=i + 1,
                        text=text,
                        metadata={"rotation": getattr(page, "get", lambda x: 0)("/Rotate")},
                    )
                    pages.append(extracted_page)
                    full_text_parts.append(text)

            full_text = "\n".join(full_text_parts)
            elapsed = (time.perf_counter() - start_time) * 1000

            return ExtractionResult(
                source_path=str(path.resolve()),
                filename=path.name,
                file_size_bytes=file_size,
                total_pages=num_pages,
                pages=pages,
                full_text=full_text,
                metadata=pdf_metadata,
                extraction_method="pypdf2",
                processing_time_ms=elapsed,
                success=True,
            )

        except ExtractionError:
            raise
        except Exception as e:
            logger.error("PyPDF2 extraction failed for %s: %s", file_path, e)
            raise ExtractionError(f"PyPDF2 extraction failed: {e}") from e


class PdfPlumberBackend:
    """PDF text extraction using pdfplumber for table-aware extraction."""

    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from PDF using pdfplumber.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ExtractionResult with all extracted content.

        Raises:
            ExtractionError: If extraction fails.
        """
        import time

        import pdfplumber

        start_time = time.perf_counter()
        path = Path(file_path)

        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise UnsupportedFileError(f"Expected .pdf file, got: {path.suffix}")

        try:
            file_size = path.stat().st_size
            pages: List[ExtractedPage] = []
            full_text_parts: List[str] = []

            with pdfplumber.open(str(path)) as pdf:
                num_pages = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception as page_err:
                        logger.warning(
                            "Page %d extraction failed: %s", i + 1, page_err
                        )
                        text = ""

                    # Attempt table extraction as supplementary metadata
                    table_data: List[List[str]] = []
                    try:
                        tables = page.extract_tables()
                        if tables:
                            table_data = [
                                [cell or "" for cell in row]
                                for table in tables
                                for row in table
                            ]
                    except Exception:
                        pass

                    extracted_page = ExtractedPage(
                        page_number=i + 1,
                        text=text,
                        metadata={"table_rows": len(table_data), "has_tables": len(table_data) > 0},
                    )
                    if table_data:
                        extracted_page.metadata["table_preview"] = table_data[:5]

                    pages.append(extracted_page)
                    full_text_parts.append(text)

            full_text = "\n".join(full_text_parts)
            elapsed = (time.perf_counter() - start_time) * 1000

            return ExtractionResult(
                source_path=str(path.resolve()),
                filename=path.name,
                file_size_bytes=file_size,
                total_pages=num_pages,
                pages=pages,
                full_text=full_text,
                extraction_method="pdfplumber",
                processing_time_ms=elapsed,
                success=True,
            )

        except ExtractionError:
            raise
        except Exception as e:
            logger.error("pdfplumber extraction failed for %s: %s", file_path, e)
            raise ExtractionError(f"pdfplumber extraction failed: {e}") from e


class TextFileBackend:
    """Backend for plain text files (for testing/demo purposes)."""

    def extract(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract text from a plain text file.

        Args:
            file_path: Path to the text file.

        Returns:
            ExtractionResult with all extracted content.
        """
        import time

        start_time = time.perf_counter()
        path = Path(file_path)

        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            file_size = path.stat().st_size
            text = path.read_text(encoding="utf-8", errors="replace")

            # Split into pseudo-pages (~3000 chars per page)
            page_size = 3000
            text_pages = []
            for i in range(0, len(text), page_size):
                page_text = text[i : i + page_size]
                extracted_page = ExtractedPage(
                    page_number=(i // page_size) + 1,
                    text=page_text,
                )
                text_pages.append(extracted_page)

            elapsed = (time.perf_counter() - start_time) * 1000

            return ExtractionResult(
                source_path=str(path.resolve()),
                filename=path.name,
                file_size_bytes=file_size,
                total_pages=len(text_pages),
                pages=text_pages,
                full_text=text,
                extraction_method="text_file",
                processing_time_ms=elapsed,
                success=True,
            )

        except Exception as e:
            raise ExtractionError(f"Text file extraction failed: {e}") from e


class DocumentExtractor:
    """
    Unified document extractor with automatic backend selection.

    Supports pluggable backends and fallback extraction strategies
    for maximum robustness across different PDF types.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst"}

    def __init__(
        self,
        preferred_backend: str = "auto",
        enable_fallback: bool = True,
    ) -> None:
        """
        Initialize the document extractor.

        Args:
            preferred_backend: Preferred backend name ('pypdf2', 'pdfplumber',
                'text_file', or 'auto').
            enable_fallback: Whether to try alternative backends on failure.
        """
        self.preferred_backend = preferred_backend
        self.enable_fallback = enable_fallback
        self._backends: Dict[str, ExtractorBackend] = {}
        self._register_default_backends()

    def _register_default_backends(self) -> None:
        """Register all available extraction backends."""
        self._backends["pypdf2"] = PyPDF2Backend()
        self._backends["pdfplumber"] = PdfPlumberBackend()
        self._backends["text_file"] = TextFileBackend()

    def _select_backend(self, file_path: Path) -> str:
        """Select the best backend for a given file."""
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md", ".rst"}:
            return "text_file"

        if self.preferred_backend != "auto":
            return self.preferred_backend

        # Auto-select: prefer pdfplumber for complex PDFs
        if suffix == ".pdf":
            try:
                import pdfplumber  # noqa: F401

                return "pdfplumber"
            except ImportError:
                logger.info("pdfplumber not available, falling back to PyPDF2")
                return "pypdf2"
        return "text_file"

    def extract(
        self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract text from a document file.

        Args:
            file_path: Path to the document file.
            options: Optional extraction parameters.

        Returns:
            ExtractionResult containing all extracted content.

        Raises:
            UnsupportedFileError: If the file format is not supported.
            ExtractionError: If all extraction backends fail.
        """
        path = Path(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        primary_backend = self._select_backend(path)
        backends_to_try = [primary_backend]

        if self.enable_fallback:
            for name in self._backends:
                if name not in backends_to_try:
                    backends_to_try.append(name)

        last_error: Optional[Exception] = None

        for backend_name in backends_to_try:
            backend = self._backends.get(backend_name)
            if not backend:
                continue

            try:
                logger.info(
                    "Extracting %s with backend '%s'", path.name, backend_name
                )
                result = backend.extract(path)
                if result.success and result.full_text.strip():
                    return result
                elif result.success:
                    logger.warning(
                        "Backend '%s' returned empty text for %s",
                        backend_name,
                        path.name,
                    )
                    # Continue to next backend for non-empty result
            except Exception as e:
                last_error = e
                logger.warning("Backend '%s' failed: %s", backend_name, e)
                continue

        # If we get here, all backends failed or returned empty
        if last_error:
            raise ExtractionError(
                f"All extraction backends failed for {path.name}"
            ) from last_error

        # Return empty but successful result
        return ExtractionResult(
            source_path=str(path.resolve()),
            filename=path.name,
            file_size_bytes=path.stat().st_size,
            total_pages=0,
            pages=[],
            full_text="",
            success=True,
            error_message="No text content found",
        )

    def extract_batch(
        self, file_paths: List[Union[str, Path]], options: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Union[str, Path], ExtractionResult]]:
        """
        Extract text from multiple files.

        Args:
            file_paths: List of file paths to process.
            options: Optional extraction parameters.

        Returns:
            List of (file_path, result) tuples.
        """
        results = []
        for fp in file_paths:
            try:
                result = self.extract(fp, options)
                results.append((fp, result))
            except ExtractionError as e:
                logger.error("Failed to extract %s: %s", fp, e)
                results.append(
                    (
                        fp,
                        ExtractionResult(
                            source_path=str(fp),
                            filename=Path(fp).name,
                            file_size_bytes=0,
                            total_pages=0,
                            pages=[],
                            full_text="",
                            success=False,
                            error_message=str(e),
                        ),
                    )
                )
        return results
