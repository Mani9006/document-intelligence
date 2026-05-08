# Automated Document Intelligence System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PEP8](https://img.shields.io/badge/PEP8-compliant-brightgreen.svg)](https://peps.python.org/pep-0008/)

A production-grade Python document processing pipeline that extracts text from PDFs, classifies document types, extracts named entities, detects document similarity, and outputs structured JSON.

## Features

- **PDF Text Extraction** - Multi-backend extraction using PyPDF2 and pdfplumber with automatic fallback
- **Document Classification** - Classify documents as invoice, resume, report, or letter using ensemble ML
- **Named Entity Extraction** - Extract dates, amounts, names, organizations, emails, phone numbers, and more
- **Document Similarity** - TF-IDF and cosine similarity for duplicate detection
- **Batch Processing** - Process entire directories with configurable pipeline stages
- **Structured JSON Output** - Clean, schema-versioned output with statistics
- **Command-Line Interface** - Full CLI with subcommands for all operations
- **Error Recovery** - Graceful handling of extraction failures with fallback strategies

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| PyPDF2 | PDF text extraction |
| pdfplumber | Advanced PDF extraction with table support |
| pytest | Unit testing framework |
| black | Code formatting |
| flake8 | Linting |
| mypy | Static type checking |

## Architecture

```
+---------------+    +---------------+    +------------------+
|   Documents   |--->|  Extractor    |--->|   Classifier     |
+---------------+    +---------------+    +------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
               +----------------+                           +--------------------+
               | Entity Extractor|                           | Similarity Engine  |
               +----------------+                           +--------------------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                        +-------+-------+
                                        v               v
                               +----------------+  +----------+
                               | JSON Formatter |->| Output   |
                               +----------------+  +----------+
```

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/docintel/doc-intelligence.git
cd doc-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

#### Run the Full Pipeline

```bash
# Process all documents in a directory
python -m src.cli pipeline -i ./data/sample_pdfs/ -o ./outputs/

# With verbose logging
python -m src.cli pipeline -i ./data/sample_pdfs/ -o ./outputs/ -v

# Skip specific stages
python -m src.cli pipeline -i ./data/sample_pdfs/ -o ./outputs/ --no-similarity
```

#### Extract Text from a Single Document

```bash
python -m src.cli extract -f document.pdf --backend pdfplumber

# Extract and save to file
python -m src.cli extract -f document.pdf -o extracted.txt
```

#### Classify a Document

```bash
python -m src.cli classify -f document.txt --method ensemble

# Available methods: ensemble, heuristic, tfidf
```

#### Extract Entities

```bash
# Extract all entity types
python -m src.cli entities -f invoice.txt

# Filter by entity type
python -m src.cli entities -f invoice.txt --types date amount
```

#### Compare Document Similarity

```bash
# Full pairwise comparison
python -m src.cli similar -d ./data/sample_pdfs/

# Find similar to a query document
python -m src.cli similar -d ./data/sample_pdfs/ --query document.txt
```

### Python API

```python
from src.extractor import DocumentExtractor
from src.classifier import DocumentClassifier
from src.entity_extractor import EntityExtractionPipeline

# Extract text
extractor = DocumentExtractor()
result = extractor.extract("document.pdf")
print(f"Extracted {result.total_word_count} words from {result.total_pages} pages")

# Classify document
classifier = DocumentClassifier()
classification = classifier.classify(result.full_text)
print(f"Document type: {classification.predicted_type.value}")
print(f"Confidence: {classification.confidence:.2f}")

# Extract entities
pipeline = EntityExtractionPipeline()
entities = pipeline.extract_all(result.full_text)
for entity in entities:
    print(f"{entity.entity_type.value}: {entity.text}")
```

## Sample Output

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2024-01-15T10:30:00Z",
  "pipeline": {
    "summary": {
      "documents_processed": 10,
      "documents_failed": 0,
      "total_time_ms": 2540.5
    }
  },
  "documents": [
    {
      "doc_id": "0001_invoice_001",
      "filename": "invoice_001.txt",
      "extraction": {
        "total_pages": 1,
        "total_words": 245,
        "extraction_method": "text_file"
      },
      "classification": {
        "predicted_type": "invoice",
        "confidence": 0.9823,
        "method": "ensemble"
      },
      "entities": [
        {
          "text": "January 15, 2024",
          "type": "date",
          "start_pos": 45,
          "confidence": 0.97,
          "normalized_value": "2024-01-15"
        },
        {
          "text": "$1,605.00",
          "type": "amount",
          "start_pos": 320,
          "confidence": 0.98,
          "normalized_value": "1605.00"
        }
      ]
    }
  ],
  "similarity": {
    "document_ids": ["0001_invoice_001", "0002_invoice_002"],
    "matrix": [[1.0, 0.85], [0.85, 1.0]]
  },
  "duplicates": [
    {"doc_a": "0001_invoice_001", "doc_b": "0002_invoice_002", "score": 0.85}
  ],
  "statistics": {
    "document_count": 10,
    "total_entities_extracted": 147,
    "document_type_distribution": {
      "invoice": 4,
      "resume": 2,
      "report": 3,
      "letter": 1
    }
  }
}
```

## Project Structure

```
project_05_doc_intelligence/
├── src/
│   ├── __init__.py              # Package init
│   ├── extractor.py             # PDF text extraction
│   ├── classifier.py            # Document type classifier
│   ├── entity_extractor.py      # Named entity extraction
│   ├── similarity.py            # Document similarity engine
│   ├── batch_pipeline.py        # Batch processing pipeline
│   ├── output_formatter.py      # JSON output formatting
│   └── cli.py                   # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py        # Extraction tests
│   ├── test_classifier.py       # Classification tests
│   ├── test_entity_extractor.py # Entity extraction tests
│   └── test_similarity.py       # Similarity tests
├── data/
│   └── sample_pdfs/             # Sample documents
├── outputs/                     # Generated JSON results
├── docs/
│   └── architecture.md          # Architecture documentation
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project config
├── setup.py                    # Package setup
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

## Screenshots

```
+---------------------------------------------------------------+
| $ python -m src.cli pipeline -i ./data/sample_pdfs/ -o ./out/ |
|                                                               |
| 2024-01-15 10:30:01 | INFO    | Scan complete: 10 documents   |
| 2024-01-15 10:30:02 | INFO    | Processing document 1/10      |
| 2024-01-15 10:30:02 | INFO    | Classified as: invoice        |
| 2024-01-15 10:30:03 | INFO    | Found 12 entities             |
| ...                                                           |
| 2024-01-15 10:30:05 | INFO    | Pipeline complete             |
| Results: 10 processed, 0 failed, 2540ms total                 |
+---------------------------------------------------------------+
```

```
+---------------------------------------------------------------+
| $ python -m src.cli classify -f invoice.txt                   |
|                                                               |
| Predicted Type: invoice                                       |
| Confidence: 0.9823                                            |
| Method: ensemble                                              |
|                                                               |
| Scores:                                                       |
|   unknown: 0.001234  <--                                      |
|   invoice: 0.982341  <--                                      |
|   resume: 0.005678                                            |
|   report: 0.007891                                            |
|   letter: 0.002856                                            |
+---------------------------------------------------------------+
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_classifier.py -v

# Run specific test
pytest tests/test_classifier.py::TestHeuristicClassifier::test_invoice_detection -v
```

## Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Run all quality checks
black --check src/ tests/ && flake8 src/ tests/ && mypy src/
```

## Future Improvements

- **Deep Learning NER**: Replace regex-based entity extraction with transformer models (BERT, spaCy)
- **OCR Support**: Add Tesseract integration for scanned PDFs
- **Multi-language Support**: Extend classification and entity extraction to non-English languages
- **REST API**: Expose pipeline as a FastAPI web service
- **Async Processing**: Add asyncio support for concurrent document processing
- **Persistent Storage**: Add database backend for document indexing and retrieval
- **Web Dashboard**: React-based frontend for visualizing results
- **Document Clustering**: Hierarchical clustering for document organization
- **Custom Entity Types**: User-configurable entity patterns
- **Model Persistence**: Save and load trained classification models
- **Confidence Thresholds**: Configurable confidence thresholds per entity type
- **Redaction Support**: Automatic PII redaction based on extracted entities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. All tests pass (`pytest tests/ -v`)
3. Type hints are included for all functions
4. Docstrings follow Google style
5. Changes are documented in code comments

## Contact

- **Project**: Automated Document Intelligence System
- **Version**: 1.0.0
- **License**: MIT
