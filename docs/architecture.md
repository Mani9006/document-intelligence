# Document Intelligence System - Architecture

## Overview

The Automated Document Intelligence System is a modular Python pipeline for processing documents. It extracts text from PDFs and text files, classifies document types, extracts named entities, computes document similarity, and outputs structured JSON results.

## System Architecture

```
                    +-------------------+
                    |   Input Sources   |
                    | (PDF, TXT, MD)    |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |  DocumentExtractor |
                    |  (Multi-Backend)   |
                    +--------+----------+
                             |
                +------------+------------+
                |                         |
                v                         v
    +-----------+---------+   +----------+-----------+
    | DocumentClassifier  |   | EntityExtractionPipeline |
    | (Heuristic + TFIDF) |   | (Rule-Based NER)         |
    +-----------+---------+   +----------+-----------+
                |                         |
                v                         v
    +-----------+---------+   +----------+-----------+
    | ClassificationResult|   | List[ExtractedEntity]|
    +---------------------+   +----------------------+
                |                         |
                +------------+------------+
                             |
                             v
                    +--------+----------+
                    | SimilarityEngine  |
                    | (TF-IDF + Cosine) |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    | OutputFormatter   |
                    | (JSON Export)     |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |  output/*.json    |
                    +-------------------+
```

## Module Descriptions

### 1. extractor.py - Text Extraction Module

**Purpose**: Extract text from PDF documents using multiple backend libraries.

**Components**:
- `ExtractorBackend` (Protocol): Interface for pluggable extraction backends
- `PyPDF2Backend`: Uses PyPDF2 for PDF text extraction
- `PdfPlumberBackend`: Uses pdfplumber for table-aware extraction
- `TextFileBackend`: Handles plain text, markdown, and reStructuredText files
- `DocumentExtractor`: Unified interface with automatic backend selection and fallback

**Key Features**:
- Multi-backend extraction with automatic fallback
- Per-page metadata (word count, line count, numeric content detection)
- PDF metadata extraction
- Table detection via pdfplumber
- Language hint detection

### 2. classifier.py - Document Type Classifier

**Purpose**: Classify documents into categories: invoice, resume, report, letter.

**Components**:
- `HeuristicClassifier`: Keyword-based classification with weighted scoring
- `TFIDFClassifier`: TF-IDF based classification with cosine similarity
- `DocumentClassifier`: Ensemble classifier combining both methods

**Classification Method**:
1. **Heuristic**: Matches weighted keyword profiles for each document type
2. **TF-IDF**: Computes TF-IDF vectors and compares to trained prototypes
3. **Ensemble**: Weighted combination of heuristic and TF-IDF predictions

**Document Types**:
- `invoice`: Financial documents with payment terms
- `resume`: CVs and professional profiles
- `report`: Analytical and business documents
- `letter`: Correspondence with greetings/closings
- `unknown`: Unclassifiable documents

### 3. entity_extractor.py - Named Entity Extraction

**Purpose**: Extract structured entities from document text using regex patterns.

**Entity Types**:
- **DATE**: Various date formats (ISO, US, EU, written)
- **AMOUNT**: Monetary values ($, EUR, GBP, percentages)
- **PERSON**: Names with honorifics, signatures
- **ORGANIZATION**: Company names with legal suffixes
- **EMAIL**: Email addresses
- **PHONE**: Phone numbers in multiple formats
- **INVOICE_NUMBER**: Invoice identifiers
- **PO_NUMBER**: Purchase order numbers

**Architecture**:
- `BaseExtractor`: Abstract base class for all extractors
- Individual extractors for each entity type
- `EntityExtractionPipeline`: Unified pipeline with deduplication

### 4. similarity.py - Document Similarity Engine

**Purpose**: Compute similarity between documents for duplicate detection.

**Components**:
- `TFIDFVectorizer`: Custom TF-IDF with configurable parameters
- `SimilarityEngine`: Cosine and Jaccard similarity computation
- `SimilarityMatrix`: Full pairwise similarity matrix

**Methods**:
- **Cosine Similarity**: Best for document-level similarity
- **Jaccard Similarity**: Token-based overlap measure
- **Duplicate Detection**: Near-duplicate detection with configurable threshold

### 5. batch_pipeline.py - Batch Processing Pipeline

**Purpose**: Orchestrate the complete processing workflow.

**Pipeline Stages**:
1. Scan input directory for supported files
2. Extract text from each document
3. Classify document type (optional)
4. Extract named entities (optional)
5. Compute inter-document similarity (optional)
6. Format and save JSON results

**Configuration**:
- `PipelineConfig`: Complete pipeline configuration
- `ProcessedDocument`: Result for a single document
- `PipelineResult`: Complete batch processing result

### 6. output_formatter.py - JSON Output Formatting

**Purpose**: Format results as structured, readable JSON.

**Features**:
- Schema versioning
- Document statistics aggregation
- Similarity matrix export
- Duplicate pair listing
- Pretty-printed output

### 7. cli.py - Command-Line Interface

**Commands**:
- `pipeline`: Run full processing pipeline
- `extract`: Extract text from a single document
- `classify`: Classify a document type
- `entities`: Extract entities from a document
- `similar`: Compute similarity between documents

## Data Flow

```
Raw Documents -> Extractor -> Classifier -> Entity Extractor -> Similarity -> JSON Output
                                    |              |                |
                                    v              v                v
                           Classification   Entity List    Similarity Matrix
                                  Score       per Doc       & Duplicates
```

## Error Handling Strategy

1. **Extraction Errors**: Log warning, try fallback backends
2. **Classification Errors**: Skip classification, continue pipeline
3. **Entity Extraction Errors**: Skip entities, continue pipeline
4. **Similarity Errors**: Skip similarity, save other results
5. **Pipeline Errors**: Configurable `continue_on_error` behavior

## Performance Considerations

- **Lazy Loading**: TF-IDF classifier auto-trains on first use
- **Sparse Vectors**: TF-IDF uses sparse dictionary representations
- **Deduplication**: Entity overlap detection prevents redundant matches
- **Configurable Backends**: Choose appropriate backend for file type
