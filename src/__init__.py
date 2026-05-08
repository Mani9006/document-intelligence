"""
Automated Document Intelligence System.

A Python document processing pipeline that extracts text from PDFs,
analyzes document structure, classifies document types, extracts entities,
detects similarity between documents, and outputs structured JSON.

Modules:
    extractor: PDF text extraction with multiple backends.
    classifier: Document type classification using heuristic and ML-based methods.
    entity_extractor: Named entity extraction (dates, amounts, names, orgs).
    similarity: TF-IDF and cosine similarity for document comparison.
    batch_pipeline: Orchestrated batch processing of document folders.
    output_formatter: Structured JSON output formatting.
    cli: Command-line interface for the complete pipeline.
"""

__version__ = "1.0.0"
__author__ = "Document Intelligence Team"
__license__ = "MIT"
