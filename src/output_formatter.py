"""
Output Formatting Module.

Formats pipeline results as structured JSON output with support for
pretty-printing, schema validation, and multiple output formats.
Provides clean, well-organized output suitable for downstream processing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.batch_pipeline import PipelineResult

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Formats and saves pipeline results in structured output formats.

    Supports JSON output with configurable formatting options including
    pretty-printing, custom indentation, and schema versioning.
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        pretty_print: bool = True,
        indent: int = 2,
        include_full_text: bool = False,
        max_text_length: int = 10000,
    ) -> None:
        """
        Initialize the output formatter.

        Args:
            pretty_print: Whether to format JSON with indentation.
            indent: Number of spaces for indentation.
            include_full_text: Whether to include full document text in output.
            max_text_length: Maximum text length to include.
        """
        self.pretty_print = pretty_print
        self.indent = indent
        self.include_full_text = include_full_text
        self.max_text_length = max_text_length

    def format(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Format a PipelineResult as a structured dictionary.

        Args:
            result: Pipeline result to format.

        Returns:
            Structured dictionary ready for JSON serialization.
        """
        output: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": result.timestamp,
            "pipeline": {
                "config": result.config.to_dict(),
                "summary": {
                    "documents_processed": result.documents_processed,
                    "documents_failed": result.documents_failed,
                    "total_time_ms": round(result.total_processing_time_ms, 2),
                    "timestamp": result.timestamp,
                },
            },
            "documents": [],
        }

        # Format each document
        for doc in result.documents:
            doc_dict = doc.to_dict()

            # Optionally include truncated full text
            if self.include_full_text:
                text = doc.extraction.full_text
                if len(text) > self.max_text_length:
                    text = text[: self.max_text_length] + "... [truncated]"
                doc_dict["extraction"]["full_text"] = text

            output["documents"].append(doc_dict)

        # Add similarity results if available
        if result.similarity_matrix:
            output["similarity"] = result.similarity_matrix.to_dict()
        else:
            output["similarity"] = None

        # Add duplicates
        output["duplicates"] = [
            {"doc_a": a, "doc_b": b, "score": round(s, 4)}
            for a, b, s in result.duplicates
        ]

        # Add statistics
        output["statistics"] = self._compute_statistics(result)

        return output

    def _compute_statistics(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all processed documents.

        Args:
            result: Pipeline result.

        Returns:
            Dictionary of statistics.
        """
        if not result.documents:
            return {"document_count": 0}

        type_counts: Dict[str, int] = {}
        total_words = 0
        total_pages = 0
        total_entities = 0
        entity_type_counts: Dict[str, int] = {}
        extraction_times = []
        processing_times = []

        for doc in result.documents:
            if doc.classification:
                t = doc.classification.predicted_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

            total_words += doc.extraction.total_word_count
            total_pages += doc.extraction.total_pages
            extraction_times.append(doc.extraction.processing_time_ms)
            processing_times.append(doc.processing_time_ms)

            for entity in doc.entities:
                total_entities += 1
                et = entity.entity_type.value
                entity_type_counts[et] = entity_type_counts.get(et, 0) + 1

        successful_docs = [d for d in result.documents if d.success]
        avg_words = total_words / len(successful_docs) if successful_docs else 0

        return {
            "document_count": len(result.documents),
            "successful": len(successful_docs),
            "failed": result.documents_failed,
            "total_words": total_words,
            "total_pages": total_pages,
            "average_words_per_doc": round(avg_words, 1),
            "total_entities_extracted": total_entities,
            "document_type_distribution": type_counts,
            "entity_type_distribution": entity_type_counts,
            "avg_extraction_time_ms": (
                round(sum(extraction_times) / len(extraction_times), 2)
                if extraction_times
                else 0
            ),
            "avg_processing_time_ms": (
                round(sum(processing_times) / len(processing_times), 2)
                if processing_times
                else 0
            ),
        }

    def to_json(self, result: PipelineResult) -> str:
        """
        Convert PipelineResult to JSON string.

        Args:
            result: Pipeline result to format.

        Returns:
            JSON string representation.
        """
        data = self.format(result)
        kwargs = {
            "ensure_ascii": False,
        }
        if self.pretty_print:
            kwargs["indent"] = self.indent
        return json.dumps(data, **kwargs)

    def save(self, result: PipelineResult, output_dir: str) -> str:
        """
        Save pipeline result to a JSON file.

        Args:
            result: Pipeline result to save.
            output_dir: Directory for output file.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"doc_intelligence_results_{timestamp}.json"
        filepath = output_path / filename

        json_content = self.to_json(result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_content)

        # Also save a latest.json symlink reference
        latest_path = output_path / "latest.json"
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(filepath.name)
        except OSError:
            # Symlink may fail on some systems; copy instead
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(json_content)

        logger.info("Results saved to: %s", filepath)
        return str(filepath)

    def save_individual(
        self, result: PipelineResult, output_dir: str
    ) -> List[str]:
        """
        Save each document result as a separate JSON file.

        Args:
            result: Pipeline result.
            output_dir: Directory for output files.

        Returns:
            List of saved file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for doc in result.documents:
            doc_data = doc.to_dict()
            filename = f"{doc.doc_id}_result.json"
            filepath = output_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            saved_files.append(str(filepath))

        logger.info("Saved %d individual result files", len(saved_files))
        return saved_files
