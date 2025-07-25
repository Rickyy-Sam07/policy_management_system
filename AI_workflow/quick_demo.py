"""
Quick Start Demo
===============

Minimal demo script that works without heavy dependencies.
"""

import sys
import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class DocumentType(str, Enum):
    """Document types supported by the system."""
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class QuickDetectionResult:
    """Simplified detection result."""
    filename: str
    file_size: int
    mime_type: Optional[str]
    detected_type: DocumentType
    confidence: float
    next_steps: list


def quick_detect_document_type(file_path: str) -> QuickDetectionResult:
    """
    Quick document type detection using only standard library.
    
    This is a lightweight version that demonstrates the core logic
    without requiring heavy dependencies.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return QuickDetectionResult(
            filename=file_path.name,
            file_size=0,
            mime_type=None,
            detected_type=DocumentType.UNKNOWN,
            confidence=0.0,
            next_steps=["File not found"]
        )
    
    # Get file info
    file_size = file_path.stat().st_size
    
    # MIME type detection using standard library
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    # File extension
    extension = file_path.suffix.lower()
    
    # Determine document type
    confidence = 0.0
    detected_type = DocumentType.UNKNOWN
    
    if mime_type:
        if mime_type == 'application/pdf':
            detected_type = DocumentType.PDF
            confidence = 0.9
        elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                          'application/msword']:
            detected_type = DocumentType.DOCX
            confidence = 0.9
        elif mime_type in ['message/rfc822', 'application/vnd.ms-outlook']:
            detected_type = DocumentType.EMAIL
            confidence = 0.9
        elif mime_type.startswith('image/'):
            detected_type = DocumentType.IMAGE
            confidence = 0.9
        elif mime_type.startswith('text/'):
            detected_type = DocumentType.TEXT
            confidence = 0.8
    
    # Fallback to extension if MIME detection failed
    if detected_type == DocumentType.UNKNOWN:
        if extension == '.pdf':
            detected_type = DocumentType.PDF
            confidence = 0.7
        elif extension in ['.docx', '.doc']:
            detected_type = DocumentType.DOCX
            confidence = 0.7
        elif extension in ['.eml', '.msg']:
            detected_type = DocumentType.EMAIL
            confidence = 0.7
        elif extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
            detected_type = DocumentType.IMAGE
            confidence = 0.7
        elif extension in ['.txt', '.md']:
            detected_type = DocumentType.TEXT
            confidence = 0.7
    
    # Generate next steps
    next_steps = []
    if detected_type == DocumentType.PDF:
        next_steps = [
            "Initialize PDF parser (pdfplumber or PyMuPDF)",
            "Extract text content",
            "Check if OCR is needed for image-based PDFs",
            "Extract metadata"
        ]
    elif detected_type == DocumentType.DOCX:
        next_steps = [
            "Initialize python-docx parser",
            "Extract text and tables",
            "Process embedded images with OCR",
            "Extract document metadata"
        ]
    elif detected_type == DocumentType.EMAIL:
        next_steps = [
            "Initialize email parser",
            "Extract headers and body",
            "Process attachments recursively",
            "Extract contact information"
        ]
    elif detected_type == DocumentType.IMAGE:
        next_steps = [
            "Initialize OCR engine (PaddleOCR recommended)",
            "Apply image preprocessing",
            "Extract text using OCR",
            "Validate and clean OCR results"
        ]
    elif detected_type == DocumentType.TEXT:
        next_steps = [
            "Read text content directly",
            "Detect encoding if needed",
            "Parse structure (markdown, etc.)",
            "Extract metadata"
        ]
    else:
        next_steps = [
            "Manual inspection required",
            "Check file headers",
            "Try generic text extraction"
        ]
    
    return QuickDetectionResult(
        filename=file_path.name,
        file_size=file_size,
        mime_type=mime_type,
        detected_type=detected_type,
        confidence=confidence,
        next_steps=next_steps
    )


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def demonstrate_detection(file_paths: list):
    """Demonstrate document type detection."""
    print("Document Type Detection Demo")
    print("=" * 60)
    print("This demo shows the core document detection logic")
    print("without requiring heavy dependencies.\n")
    
    for file_path in file_paths:
        print(f"Analyzing: {file_path}")
        print("-" * 40)
        
        result = quick_detect_document_type(file_path)
        
        print(f"Filename: {result.filename}")
        print(f"File Size: {format_file_size(result.file_size)}")
        print(f"MIME Type: {result.mime_type or 'Unknown'}")
        print(f"Detected Type: {result.detected_type.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        
        print("\nRecommended Processing Steps:")
        for i, step in enumerate(result.next_steps, 1):
            print(f"   {i}. {step}")
        
        print("\n" + "=" * 60 + "\n")


def create_sample_files():
    """Create sample files for demonstration."""
    print("Creating sample files for demonstration...")
    
    sample_dir = Path("demo_files")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    sample_txt = sample_dir / "sample_document.txt"
    with open(sample_txt, 'w', encoding='utf-8') as f:
        f.write("""
Sample Document for Processing
=============================

This is a sample text document that demonstrates the document 
processing system's capabilities.

Key Features:
• Intelligent document type detection
• Multi-format support (PDF, DOCX, emails, images)
• OCR capabilities for image-based content
• Metadata extraction and confidence scoring
• Batch processing capabilities

The system can process various document types and extract
meaningful content for further analysis or indexing.

Processing Pipeline:
1. File type detection using MIME types and signatures
2. Route to appropriate parser based on detected type
3. Extract text content using specialized tools
4. Apply OCR if needed for image-based documents
5. Return structured results with metadata

This approach ensures optimal processing for each document type
while maintaining high accuracy and performance.
        """.strip())
    
    # Create sample markdown file
    sample_md = sample_dir / "README.md"
    with open(sample_md, 'w', encoding='utf-8') as f:
        f.write("""# Sample Markdown Document

This is a **markdown** document that would be processed as text.

## Features

- Structured content detection
- *Italic* and **bold** text recognition
- List processing

### Code Example

```python
from document_processor import process_document
result = process_document("document.pdf")
```

> This demonstrates how the system handles different text formats.
""")
    
    print(f"Created sample files in {sample_dir}")
    return [str(sample_txt), str(sample_md)]


def show_system_architecture():
    """Display system architecture overview."""
    print("\nSystem Architecture Overview")
    print("=" * 60)
    
    architecture = """
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input File    │───▶│  Type Detection  │───▶│  Route to Parser│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │                        │
                               ▼                        ▼
                    ┌──────────────────┐    ┌─────────────────────┐
                    │  Detection Logic │    │    Specialized      │
                    │  • MIME Types    │    │     Parsers         │
                    │  • Magic Numbers │    │  • PDF Parser       │
                    │  • Extensions    │    │  • DOCX Parser      │
                    │  • Confidence    │    │  • Email Parser     │
                    └──────────────────┘    │  • OCR Parser       │
                                           └─────────────────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────────┐
                                           │  Structured Output  │
                                           │  • Extracted Text   │
                                           │  • Metadata         │
                                           │  • Confidence Score │
                                           │  • Processing Time  │
                                           └─────────────────────┘

Processing Flow:
1. Detection: Multi-method file type identification
2. Routing: Intelligent parser selection
3. Extraction: Specialized content processing
4. OCR: Automatic image text recognition
5. Output: Structured results with metadata
"""
    print(architecture)


def main():
    """Main demo function."""
    if len(sys.argv) > 1:
        # Use provided files
        file_paths = sys.argv[1:]
        demonstrate_detection(file_paths)
    else:
        # Create and use sample files
        print("Document Processing System - Quick Demo")
        print("=" * 60)
        print("No files provided. Creating sample files for demonstration...\n")
        
        sample_files = create_sample_files()
        demonstrate_detection(sample_files)
        
        show_system_architecture()
        
        print("\nTo test with your own files:")
        print("   python quick_demo.py file1.pdf file2.docx file3.jpg")
        print("\nFor full functionality, run:")
        print("   python setup.py")
        print("   python example_usage.py")
        print("\nFull documentation in README.md")


if __name__ == "__main__":
    main()
