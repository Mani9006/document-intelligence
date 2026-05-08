"""
Command-Line Interface for the Document Intelligence System.

Provides a comprehensive CLI for running the document processing pipeline,
extracting text, classifying documents, extracting entities, and computing
similarity between documents.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from src.batch_pipeline import BatchPipeline, PipelineConfig
from src.classifier import DocumentClassifier, DocumentType
from src.entity_extractor import EntityExtractionPipeline, EntityType
from src.extractor import DocumentExtractor
from src.similarity import SimilarityEngine

# Configure logging with a consistent format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """
    Configure logging level.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all CLI commands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="docintel",
        description="Automated Document Intelligence System - "
        "Extract, classify, and analyze documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a directory
  docintel pipeline -i ./documents/ -o ./outputs/

  # Extract text from a single PDF
  docintel extract -f document.pdf --backend pdfplumber

  # Classify a document
  docintel classify -f document.txt --method ensemble

  # Extract entities from text
  docintel entities -f document.txt --types date amount

  # Compare similarity between documents
  docintel similar -d ./documents/ --method cosine

  # Run with verbose logging
  docintel pipeline -i ./docs/ -o ./out/ --verbose
        """,
    )

    parser.add_argument(
        "--version", action="version", version="%(prog)s 1.0.0"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full processing pipeline on a directory",
    )
    pipeline_parser.add_argument(
        "-i", "--input", required=True, help="Input directory containing documents"
    )
    pipeline_parser.add_argument(
        "-o", "--output", required=True, help="Output directory for results"
    )
    pipeline_parser.add_argument(
        "--method",
        choices=["ensemble", "heuristic", "tfidf"],
        default="ensemble",
        help="Classification method (default: ensemble)",
    )
    pipeline_parser.add_argument(
        "--similarity-method",
        choices=["cosine", "jaccard"],
        default="cosine",
        help="Similarity computation method (default: cosine)",
    )
    pipeline_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Duplicate detection threshold (default: 0.85)",
    )
    pipeline_parser.add_argument(
        "--no-classify", action="store_true", help="Skip document classification"
    )
    pipeline_parser.add_argument(
        "--no-entities", action="store_true", help="Skip entity extraction"
    )
    pipeline_parser.add_argument(
        "--no-similarity", action="store_true", help="Skip similarity analysis"
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract text from a single document"
    )
    extract_parser.add_argument(
        "-f", "--file", required=True, help="Path to the document file"
    )
    extract_parser.add_argument(
        "--backend",
        choices=["auto", "pypdf2", "pdfplumber", "text_file"],
        default="auto",
        help="Extraction backend (default: auto)",
    )
    extract_parser.add_argument(
        "--output", "-o", help="Optional output file for extracted text"
    )
    extract_parser.add_argument(
        "--pages", action="store_true", help="Output per-page information"
    )

    # Classify command
    classify_parser = subparsers.add_parser(
        "classify", help="Classify a document by type"
    )
    classify_parser.add_argument(
        "-f", "--file", required=True, help="Path to the document file"
    )
    classify_parser.add_argument(
        "--method",
        choices=["ensemble", "heuristic", "tfidf"],
        default="ensemble",
        help="Classification method (default: ensemble)",
    )
    classify_parser.add_argument(
        "--text", help="Classify raw text instead of file"
    )

    # Entities command
    entities_parser = subparsers.add_parser(
        "entities", help="Extract named entities from a document"
    )
    entities_parser.add_argument(
        "-f", "--file", required=True, help="Path to the document file"
    )
    entities_parser.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in EntityType],
        help="Entity types to extract (default: all)",
    )
    entities_parser.add_argument(
        "--text", help="Extract from raw text instead of file"
    )

    # Similar command
    similar_parser = subparsers.add_parser(
        "similar", help="Compute similarity between documents"
    )
    similar_parser.add_argument(
        "-d", "--directory", required=True, help="Directory of documents to compare"
    )
    similar_parser.add_argument(
        "--method",
        choices=["cosine", "jaccard"],
        default="cosine",
        help="Similarity method (default: cosine)",
    )
    similar_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Duplicate detection threshold (default: 0.85)",
    )
    similar_parser.add_argument(
        "--query", help="Find most similar to this document"
    )
    similar_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top matches (default: 5)"
    )

    return parser


def handle_pipeline(args: argparse.Namespace) -> int:
    """
    Handle the 'pipeline' command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        classification_method=args.method,
        similarity_method=args.similarity_method,
        similarity_threshold=args.threshold,
        enable_classification=not args.no_classify,
        enable_entity_extraction=not args.no_entities,
        enable_similarity=not args.no_similarity,
    )

    pipeline = BatchPipeline(config)

    try:
        output_path = pipeline.run()
        print(f"\nPipeline complete. Results saved to: {output_path}")
        return 0
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        return 1


def handle_extract(args: argparse.Namespace) -> int:
    """
    Handle the 'extract' command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    extractor = DocumentExtractor(preferred_backend=args.backend)

    try:
        result = extractor.extract(args.file)

        if not result.success:
            print(f"Extraction failed: {result.error_message}", file=sys.stderr)
            return 1

        print(f"File: {result.filename}")
        print(f"Pages: {result.total_pages}")
        print(f"Words: {result.total_word_count}")
        print(f"Size: {result.file_size_bytes:,} bytes")
        print(f"Method: {result.extraction_method}")
        print(f"Time: {result.processing_time_ms:.2f}ms")
        print()

        if args.pages:
            for page in result.pages:
                print(f"--- Page {page.page_number} ---")
                print(f"  Words: {page.word_count}, Lines: {page.line_count}")
                print(f"  Avg word length: {page.avg_word_length:.1f}")
                print(f"  Has numeric content: {page.has_numeric_content}")
                if page.metadata:
                    print(f"  Metadata: {page.metadata}")
                print()

        # Output text
        text_preview = result.full_text[:2000]
        if len(result.full_text) > 2000:
            text_preview += "\n... [truncated]"
        print("Extracted Text:")
        print(text_preview)

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.full_text)
            print(f"\nFull text saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return 1


def handle_classify(args: argparse.Namespace) -> int:
    """
    Handle the 'classify' command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    classifier = DocumentClassifier()

    try:
        if args.text:
            text = args.text
        else:
            extractor = DocumentExtractor()
            result = extractor.extract(args.file)
            text = result.full_text

        if not text or not text.strip():
            print("Error: No text to classify", file=sys.stderr)
            return 1

        classification = classifier.classify(text, method=args.method)

        print(f"Predicted Type: {classification.predicted_type.value}")
        print(f"Confidence: {classification.confidence:.4f}")
        print(f"Method: {classification.method}")
        print(f"Time: {classification.processing_time_ms:.2f}ms")
        print("\nScores:")
        for doc_type, score in sorted(
            classification.scores.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            marker = " <--" if doc_type == classification.predicted_type else ""
            print(f"  {doc_type.value}: {score:.6f}{marker}")

        if classification.features_used:
            print(f"\nMatched Features: {len(classification.features_used)}")
            for feat in classification.features_used[:15]:
                print(f"  - {feat}")

        return 0

    except Exception as e:
        logger.error("Classification failed: %s", e)
        return 1


def handle_entities(args: argparse.Namespace) -> int:
    """
    Handle the 'entities' command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    try:
        if args.text:
            text = args.text
        else:
            extractor = DocumentExtractor()
            result = extractor.extract(args.file)
            text = result.full_text

        if not text or not text.strip():
            print("Error: No text to process", file=sys.stderr)
            return 1

        pipeline = EntityExtractionPipeline()
        entities = pipeline.extract_all(text)

        # Filter by type if specified
        if args.types:
            entities = [e for e in entities if e.entity_type.value in args.types]

        print(f"Found {len(entities)} entities\n")

        # Group by type
        from collections import defaultdict

        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.entity_type.value].append(entity)

        for etype, entity_list in sorted(by_type.items()):
            print(f"--- {etype.upper()} ({len(entity_list)}) ---")
            for entity in entity_list[:10]:
                norm = f" -> {entity.normalized_value}" if entity.normalized_value else ""
                conf = f" [{entity.confidence:.2f}]"
                print(f"  {entity.text}{norm}{conf}")
            if len(entity_list) > 10:
                print(f"  ... and {len(entity_list) - 10} more")
            print()

        return 0

    except Exception as e:
        logger.error("Entity extraction failed: %s", e)
        return 1


def handle_similar(args: argparse.Namespace) -> int:
    """
    Handle the 'similar' command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    engine = SimilarityEngine(duplicate_threshold=args.threshold)

    try:
        # Scan directory
        input_path = Path(args.directory)
        if not input_path.exists():
            print(f"Directory not found: {input_path}", file=sys.stderr)
            return 1

        files = [
            f
            for f in input_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in (".pdf", ".txt", ".md", ".rst")
        ]
        files.sort()

        if len(files) < 2:
            print(
                "Need at least 2 documents for similarity comparison",
                file=sys.stderr,
            )
            return 1

        # Extract text from all files
        extractor = DocumentExtractor()
        documents = []
        for f in files:
            try:
                result = extractor.extract(str(f))
                if result.success and result.full_text.strip():
                    documents.append((f.name, result.full_text))
            except Exception as e:
                logger.warning("Skipping %s: %s", f.name, e)

        if len(documents) < 2:
            print("Not enough valid documents for comparison", file=sys.stderr)
            return 1

        # Handle query-based search
        if args.query:
            query_file = Path(args.query)
            if query_file.exists():
                result = extractor.extract(str(query_file))
                query_text = result.full_text
            else:
                query_text = args.query

            results = engine.find_most_similar(
                query_text, documents, top_k=args.top_k
            )
            print(f"\nTop {len(results)} most similar documents:\n")
            for i, sim_result in enumerate(results, 1):
                bar = "=" * int(sim_result.similarity_score * 40)
                print(
                    f"{i}. {sim_result.doc_b_id}"
                )
                print(f"   Similarity: {sim_result.similarity_score:.4f}")
                print(f"   [{bar}]")
                print()
            return 0

        # Full pairwise comparison
        print(f"Computing similarity for {len(documents)} documents...\n")
        matrix = engine.compare_batch(documents, method=args.method)
        duplicates = engine.find_duplicates(documents, threshold=args.threshold)

        # Print matrix
        ids = matrix.document_ids
        print("Similarity Matrix:")
        print("  " + " ".join(f"{id[:12]:>12}" for id in ids))
        for i, row in enumerate(matrix.scores):
            scores_str = " ".join(f"{s:>12.4f}" for s in row)
            print(f"  {ids[i][:12]:>12} {scores_str}")

        # Print duplicates
        if duplicates:
            print(f"\nNear-Duplicates Found ({len(duplicates)}):")
            for doc_a, doc_b, score in duplicates:
                print(f"  {doc_a} <-> {doc_b}: {score:.4f}")
        else:
            print("\nNo near-duplicates found.")

        return 0

    except Exception as e:
        logger.error("Similarity analysis failed: %s", e)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Exit code (0 for success).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging
    if args.verbose:
        setup_logging("DEBUG")
    elif args.quiet:
        setup_logging("ERROR")
    else:
        setup_logging("INFO")

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to handler
    handlers = {
        "pipeline": handle_pipeline,
        "extract": handle_extract,
        "classify": handle_classify,
        "entities": handle_entities,
        "similar": handle_similar,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
