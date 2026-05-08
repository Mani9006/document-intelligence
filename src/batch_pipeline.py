"""
Batch Processing Pipeline.

Orchestrates the complete document processing workflow:
1. Scan input directory for documents
2. Extract text from each document
3. Classify document type
4. Extract named entities
5. Compute inter-document similarity
6. Format and save results as JSON

Supports configurable pipeline stages, progress tracking,
error recovery, and detailed logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.classifier import (
    ClassificationResult,
    DocumentClassifier,
    DocumentType,
)
from src.entity_extractor import EntityExtractionPipeline, ExtractedEntity
from src.extractor import DocumentExtractor, ExtractionResult
from src.output_formatter import OutputFormatter
from src.similarity import SimilarityEngine, SimilarityMatrix

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the pipeline encounters a fatal error."""

    pass


@dataclass
class PipelineConfig:
    """Configuration for the batch processing pipeline."""

    input_dir: str
    output_dir: str
    output_format: str = "json"
    enable_classification: bool = True
    enable_entity_extraction: bool = True
    enable_similarity: bool = True
    similarity_method: str = "cosine"
    similarity_threshold: float = 0.85
    classification_method: str = "ensemble"
    supported_extensions: Tuple[str, ...] = (".pdf", ".txt", ".md", ".rst")
    max_workers: int = 1
    continue_on_error: bool = True
    log_level: str = "INFO"
    save_intermediate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "enable_classification": self.enable_classification,
            "enable_entity_extraction": self.enable_entity_extraction,
            "enable_similarity": self.enable_similarity,
            "similarity_method": self.similarity_method,
            "similarity_threshold": self.similarity_threshold,
            "classification_method": self.classification_method,
            "supported_extensions": list(self.supported_extensions),
            "max_workers": self.max_workers,
            "continue_on_error": self.continue_on_error,
            "log_level": self.log_level,
            "save_intermediate": self.save_intermediate,
        }


@dataclass
class ProcessedDocument:
    """Result of processing a single document through the pipeline."""

    doc_id: str
    source_path: str
    filename: str
    extraction: ExtractionResult
    classification: Optional[ClassificationResult] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "filename": self.filename,
            "extraction": {
                "total_pages": self.extraction.total_pages,
                "total_words": self.extraction.total_word_count,
                "file_size_bytes": self.extraction.file_size_bytes,
                "extraction_method": self.extraction.extraction_method,
                "processing_time_ms": round(
                    self.extraction.processing_time_ms, 2
                ),
            },
            "processing_time_ms": round(self.processing_time_ms, 2),
            "success": self.success,
        }

        if self.error_message:
            result["error_message"] = self.error_message

        if self.classification:
            result["classification"] = self.classification.to_dict()

        if self.entities:
            result["entities"] = [e.to_dict() for e in self.entities]

        return result


@dataclass
class PipelineResult:
    """Complete result of a pipeline run."""

    config: PipelineConfig
    documents: List[ProcessedDocument]
    similarity_matrix: Optional[SimilarityMatrix] = None
    duplicates: List[Tuple[str, str, float]] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    documents_processed: int = 0
    documents_failed: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "summary": {
                "documents_processed": self.documents_processed,
                "documents_failed": self.documents_failed,
                "total_time_ms": round(self.total_processing_time_ms, 2),
                "timestamp": self.timestamp,
            },
            "documents": [d.to_dict() for d in self.documents],
            "similarity": (
                self.similarity_matrix.to_dict()
                if self.similarity_matrix
                else None
            ),
            "duplicates": [
                {"doc_a": a, "doc_b": b, "score": round(s, 4)}
                for a, b, s in self.duplicates
            ],
        }


class BatchPipeline:
    """
    Orchestrated batch processing pipeline for document intelligence.

    Processes a folder of documents through the complete pipeline:
    extraction -> classification -> entity extraction -> similarity analysis.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize the batch pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.extractor = DocumentExtractor()
        self.classifier = DocumentClassifier()
        self.entity_pipeline = EntityExtractionPipeline()
        self.similarity_engine = SimilarityEngine(
            duplicate_threshold=config.similarity_threshold
        )
        self.formatter = OutputFormatter()

        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def scan_input(self) -> List[Path]:
        """
        Scan the input directory for supported documents.

        Returns:
            List of Path objects for discovered documents.

        Raises:
            PipelineError: If input directory does not exist.
        """
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            raise PipelineError(f"Input directory not found: {input_path}")
        if not input_path.is_dir():
            raise PipelineError(f"Input path is not a directory: {input_path}")

        files = [
            f
            for f in input_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in self.config.supported_extensions
        ]

        files.sort()
        logger.info(
            "Scanned %s: found %d documents", input_path, len(files)
        )

        return files

    def _generate_doc_id(self, file_path: Path, index: int) -> str:
        """Generate a unique document ID from file path and index."""
        stem = file_path.stem.replace(" ", "_").replace("-", "_")
        return f"{index:04d}_{stem}"

    def process_document(
        self, file_path: Path, index: int
    ) -> ProcessedDocument:
        """
        Process a single document through the pipeline.

        Args:
            file_path: Path to the document file.
            index: Document index for ID generation.

        Returns:
            ProcessedDocument with all pipeline results.
        """
        doc_id = self._generate_doc_id(file_path, index)
        start_time = time.perf_counter()

        logger.info("Processing document %d: %s", index, file_path.name)

        try:
            # Step 1: Text extraction
            extraction = self.extractor.extract(str(file_path))

            if not extraction.success:
                return ProcessedDocument(
                    doc_id=doc_id,
                    source_path=str(file_path),
                    filename=file_path.name,
                    extraction=extraction,
                    success=False,
                    error_message=extraction.error_message,
                )

            text = extraction.full_text

            # Step 2: Classification
            classification = None
            if self.config.enable_classification and text.strip():
                try:
                    classification = self.classifier.classify(
                        text, method=self.config.classification_method
                    )
                    logger.debug(
                        "Classified %s as %s (confidence: %.2f)",
                        file_path.name,
                        classification.predicted_type.value,
                        classification.confidence,
                    )
                except Exception as e:
                    logger.error("Classification failed for %s: %s", file_path.name, e)

            # Step 3: Entity extraction
            entities = []
            if self.config.enable_entity_extraction and text.strip():
                try:
                    entities = self.entity_pipeline.extract_all(text)
                    logger.debug(
                        "Extracted %d entities from %s",
                        len(entities),
                        file_path.name,
                    )
                except Exception as e:
                    logger.error("Entity extraction failed for %s: %s", file_path.name, e)

            elapsed = (time.perf_counter() - start_time) * 1000

            return ProcessedDocument(
                doc_id=doc_id,
                source_path=str(file_path),
                filename=file_path.name,
                extraction=extraction,
                classification=classification,
                entities=entities,
                processing_time_ms=elapsed,
                success=True,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error("Pipeline failed for %s: %s", file_path.name, e)

            error_result = ExtractionResult(
                source_path=str(file_path),
                filename=file_path.name,
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                total_pages=0,
                pages=[],
                full_text="",
                success=False,
                error_message=str(e),
            )

            return ProcessedDocument(
                doc_id=doc_id,
                source_path=str(file_path),
                filename=file_path.name,
                extraction=error_result,
                success=False,
                error_message=str(e),
                processing_time_ms=elapsed,
            )

    def process_batch(self) -> PipelineResult:
        """
        Process all documents in the input directory.

        Returns:
            PipelineResult with all processing results.

        Raises:
            PipelineError: If processing fails catastrophically.
        """
        pipeline_start = time.perf_counter()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Scan for documents
        files = self.scan_input()
        if not files:
            logger.warning("No documents found in %s", self.config.input_dir)
            return PipelineResult(
                config=self.config,
                documents=[],
                total_processing_time_ms=0.0,
                documents_processed=0,
                documents_failed=0,
                timestamp=timestamp,
            )

        logger.info("Starting batch processing of %d documents", len(files))

        # Process each document
        processed_docs: List[ProcessedDocument] = []
        success_count = 0
        fail_count = 0

        for i, file_path in enumerate(files):
            try:
                result = self.process_document(file_path, i)
                processed_docs.append(result)
                if result.success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                logger.error("Critical error processing %s: %s", file_path.name, e)
                if not self.config.continue_on_error:
                    raise PipelineError(f"Pipeline halted: {e}") from e

        # Step 4: Similarity analysis (across all documents)
        similarity_matrix = None
        duplicates = []

        if self.config.enable_similarity and len(processed_docs) >= 2:
            try:
                valid_docs = [
                    (d.doc_id, d.extraction.full_text)
                    for d in processed_docs
                    if d.success and d.extraction.full_text.strip()
                ]

                if len(valid_docs) >= 2:
                    similarity_matrix = self.similarity_engine.compare_batch(
                        valid_docs, method=self.config.similarity_method
                    )
                    duplicates = self.similarity_engine.find_duplicates(
                        valid_docs,
                        threshold=self.config.similarity_threshold,
                    )
                    logger.info(
                        "Similarity analysis: found %d duplicate pairs",
                        len(duplicates),
                    )
            except Exception as e:
                logger.error("Similarity analysis failed: %s", e)

        total_elapsed = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            "Pipeline complete: %d processed, %d failed, %.2fms total",
            success_count,
            fail_count,
            total_elapsed,
        )

        return PipelineResult(
            config=self.config,
            documents=processed_docs,
            similarity_matrix=similarity_matrix,
            duplicates=duplicates,
            total_processing_time_ms=total_elapsed,
            documents_processed=success_count,
            documents_failed=fail_count,
            timestamp=timestamp,
        )

    def run(self) -> str:
        """
        Run the complete pipeline and save results.

        Returns:
            Path to the output JSON file.
        """
        logger.info("=" * 60)
        logger.info("Document Intelligence Pipeline Starting")
        logger.info("=" * 60)

        result = self.process_batch()

        # Format and save results
        output_path = self.formatter.save(result, self.config.output_dir)

        logger.info("=" * 60)
        logger.info("Pipeline Complete. Results saved to: %s", output_path)
        logger.info("=" * 60)

        return output_path
