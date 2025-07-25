# Document Processing System (PMS)

A comprehensive, optimized document processing system that intelligently detects document types and routes them to the appropriate parsing/cleaning logic using the most optimal tools.

## Features

- **Intelligent Document Detection**: Automatically detects file types using MIME types, magic numbers, and file extensions
- **Multi-Format Support**: PDF, DOCX, emails (.eml, .msg), and images (.jpg, .png, .tiff, etc.)
- **OCR Integration**: Automatic OCR with PaddleOCR and Tesseract fallback
- **Batch Processing**: Process multiple documents efficiently
- **RESTful API**: FastAPI-based web API for integration
- **Comprehensive Metadata**: Detailed extraction metadata and confidence scoring
- **Error Handling**: Robust error handling with fallback mechanisms

## Supported Document Types

| Type | Extensions | Features |
|------|------------|----------|
| **PDF** | `.pdf` | Text extraction, OCR fallback, metadata extraction |
| **Word Documents** | `.docx`, `.doc` | Text/table extraction, embedded image OCR |
| **Email** | `.eml`, `.msg` | Header/body extraction, recursive attachment processing |
| **Images** | `.jpg`, `.png`, `.tiff`, `.bmp`, `.gif` | Multi-engine OCR, preprocessing |

## Quick Start

### Installation

1. **Clone and setup:**
```bash
cd c:\sambhranta\projects\pms
pip install -r requirements.txt
```

2. **Install system dependencies (Windows):**
```powershell
# For Tesseract OCR (optional)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH

# For python-magic (Windows)
pip install python-magic-bin
```

### Basic Usage

```python
from document_processor import process_document

# Process a single document
result = process_document("path/to/document.pdf")

print(f"Type: {result.detection_result.metadata.source_type}")
print(f"Success: {result.parsing_success}")
print(f"Text length: {len(result.extracted_text)} characters")
print(f"Confidence: {result.confidence_score:.2f}")
```

### Batch Processing

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
file_paths = ["doc1.pdf", "doc2.docx", "email.eml"]
results = processor.batch_process(file_paths)

for result in results:
    print(f"{result.detection_result.metadata.filename}: {result.parsing_success}")
```

## Web API

### Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

### API Examples

**Process Single Document:**
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "ocr_language=en"
```

**Batch Processing:**
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "ocr_language=en"
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input File    │───▶│  Type Detection  │───▶│  Route to Parser│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  MIME Detection  │    │   PDF Parser    │
                       │  Magic Numbers   │    │   DOCX Parser   │
                       │  File Extension  │    │   Email Parser  │
                       └──────────────────┘    │   OCR Parser    │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Structured      │
                                              │ Output + Meta   │
                                              └─────────────────┘
```

## Processing Pipeline

### 1. Document Type Detection
- **Primary**: MIME type detection using `python-magic`
- **Secondary**: Binary signature detection using `filetype`
- **Fallback**: File extension analysis
- **Confidence Scoring**: Multi-method validation

### 2. Intelligent Routing
- **PDF**: pdfplumber → PyMuPDF → OCR (if needed)
- **DOCX**: python-docx + embedded image OCR
- **Email**: mail-parser or standard email library + recursive attachment processing
- **Images**: PaddleOCR → Tesseract (with preprocessing)

### 3. Content Extraction
- **Text Extraction**: Native parsers with quality assessment
- **OCR Processing**: Automatic triggering based on content quality
- **Metadata Extraction**: Comprehensive document metadata
- **Structured Output**: Consistent response format

## Configuration

### OCR Languages
```python
# Support for multiple languages
processor = DocumentProcessor(ocr_lang='en')  # English
processor = DocumentProcessor(ocr_lang='chi_sim')  # Chinese Simplified
processor = DocumentProcessor(ocr_lang='fra')  # French
```

### Processing Options
```python
# Customize processing behavior
from document_detector import InputSource

result = detector.detect_document_type(
    file_path, 
    input_source=InputSource.WEB_UPLOAD
)
```

## Performance Optimization

### Speed Optimizations
- **Lazy Loading**: Parsers initialized only when needed
- **Efficient Detection**: Multi-method detection with early exit
- **Batch Processing**: Optimized for multiple documents
- **Memory Management**: Streaming for large files

### Quality Optimizations
- **Confidence Scoring**: Multi-factor quality assessment
- **Fallback Mechanisms**: Multiple parsers per format
- **Image Preprocessing**: Enhanced OCR accuracy
- **Error Recovery**: Graceful handling of parsing failures

## Example Outputs

### Detection Result
```json
{
  "filename": "contract.pdf",
  "source_type": "pdf",
  "mime_type": "application/pdf", 
  "confidence_score": 0.95,
  "is_ocr_required": false,
  "parser_used": "pdfplumber"
}
```

### Processing Result
```json
{
  "parsing_success": true,
  "extracted_text": "Contract Agreement...",
  "confidence_score": 0.88,
  "processing_time": 1.23,
  "metadata": {
    "page_count": 5,
    "author": "Legal Dept",
    "creation_date": "2024-01-15"
  }
}
```

## Dependencies

### Core Dependencies
- **pdfplumber**: PDF text extraction
- **PyMuPDF**: PDF processing and image extraction
- **python-docx**: Word document processing
- **python-magic**: File type detection
- **filetype**: Binary signature detection
- **mail-parser**: Email processing
- **Pillow**: Image processing

### OCR Dependencies
- **PaddleOCR**: Primary OCR engine
- **pytesseract**: Fallback OCR engine
- **opencv-python**: Image preprocessing

### API Dependencies
- **FastAPI**: Web API framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation

## Testing

```bash
# Run basic example
python example_usage.py

# Test with your files
python example_usage.py document1.pdf document2.docx email.eml image.jpg

# Start API server
python api.py

# Test API health
curl http://localhost:8000/health
```

## Logging

The system uses structured logging with **loguru**:

```python
from loguru import logger

# Logs include:
# - Document type detection results
# - Parser selection decisions  
# - Processing performance metrics
# - Error details and stack traces
# - OCR confidence scores
```

## Security Considerations

- **File Validation**: MIME type verification
- **Size Limits**: Configurable file size restrictions
- **Temporary Files**: Secure cleanup of processed files
- **Input Sanitization**: Safe filename handling
- **API Rate Limiting**: Prevent abuse (implement as needed)

## Future Enhancements

- **Cloud Storage Integration**: S3, Azure Blob, Google Cloud
- **Database Persistence**: PostgreSQL, MongoDB support
- **Advanced OCR**: Table detection, form processing
- **Machine Learning**: Document classification, content extraction
- **Microservices**: Containerized deployment with Docker
- **Real-time Processing**: WebSocket support for live updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

---

**Ready for Enterprise Use**: This system provides production-ready document processing capabilities with comprehensive error handling, logging, and API integration.
