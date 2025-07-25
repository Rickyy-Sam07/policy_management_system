"""
Document Type Detection and Routing System
==========================================

An optimized system to intelligently detect document types and route them
to the appropriate parsing/cleaning logic.

Features:
- Multi-format support (PDF, DOCX, images, emails)
- Intelligent MIME type detection with fallbacks
- OCR detection and routing
- Structured metadata output
- Comprehensive logging
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import mimetypes
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None

import filetype
from pydantic import BaseModel, Field
from loguru import logger
import json
from datetime import datetime


class DocumentType(str, Enum):
    """Enumeration of supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


class ParserType(str, Enum):
    """Enumeration of available parsers."""
    PDFPLUMBER = "pdfplumber"
    PYMUPDF = "pymupdf"
    PYTHON_DOCX = "python_docx"
    MAIL_PARSER = "mail_parser"
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"


class InputSource(str, Enum):
    """Enumeration of input sources."""
    WEB_UPLOAD = "web_upload"
    FOLDER_DROP = "folder_drop"
    API_UPLOAD = "api_upload"


class DocumentMetadata(BaseModel):
    """Structured metadata for document processing."""
    filename: str
    file_path: str
    file_size: int
    mime_type: Optional[str] = None
    detected_extension: Optional[str] = None
    source_type: DocumentType
    input_source: InputSource = InputSource.API_UPLOAD
    is_ocr_required: bool = False
    ocr_triggered: bool = False
    parser_used: Optional[ParserType] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None
    extracted_text_preview: Optional[str] = None
    has_attachments: bool = False
    attachment_count: int = 0


class DetectionResult(BaseModel):
    """Result of document type detection."""
    metadata: DocumentMetadata
    routing_instructions: Dict[str, Any]
    next_steps: List[str]
    success: bool = True


class DocumentTypeDetector:
    """
    Optimized document type detector with intelligent routing.
    
    This class provides comprehensive document type detection using multiple
    methods with fallbacks, and routes documents to appropriate parsers.
    """
    
    # MIME type mappings for quick lookup
    MIME_TO_TYPE = {
        'application/pdf': DocumentType.PDF,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
        'application/msword': DocumentType.DOCX,  # Legacy .doc files
        'message/rfc822': DocumentType.EMAIL,
        'application/vnd.ms-outlook': DocumentType.EMAIL,  # .msg files
        'text/plain': DocumentType.TEXT,
        'image/jpeg': DocumentType.IMAGE,
        'image/jpg': DocumentType.IMAGE,
        'image/png': DocumentType.IMAGE,
        'image/tiff': DocumentType.IMAGE,
        'image/bmp': DocumentType.IMAGE,
        'image/gif': DocumentType.IMAGE,
        'image/webp': DocumentType.IMAGE,
    }
    
    # Extension mappings as fallback
    EXT_TO_TYPE = {
        '.pdf': DocumentType.PDF,
        '.docx': DocumentType.DOCX,
        '.doc': DocumentType.DOCX,
        '.eml': DocumentType.EMAIL,
        '.msg': DocumentType.EMAIL,
        '.txt': DocumentType.TEXT,
        '.log': DocumentType.TEXT,
        '.md': DocumentType.TEXT,
        '.json': DocumentType.TEXT,
        '.csv': DocumentType.TEXT,
        '.xml': DocumentType.TEXT,
        '.jpg': DocumentType.IMAGE,
        '.jpeg': DocumentType.IMAGE,
        '.png': DocumentType.IMAGE,
        '.tiff': DocumentType.IMAGE,
        '.tif': DocumentType.IMAGE,
        '.bmp': DocumentType.IMAGE,
        '.gif': DocumentType.IMAGE,
        '.webp': DocumentType.IMAGE,
    }
    
    # Preferred parsers for each document type
    TYPE_TO_PARSER = {
        DocumentType.PDF: [ParserType.PDFPLUMBER, ParserType.PYMUPDF],
        DocumentType.DOCX: [ParserType.PYTHON_DOCX],
        DocumentType.EMAIL: [ParserType.MAIL_PARSER],
        DocumentType.IMAGE: [ParserType.PADDLEOCR, ParserType.TESSERACT],
    }
    
    def __init__(self):
        """Initialize the detector with magic number detection."""
        try:
            if MAGIC_AVAILABLE:
                self.magic_detector = magic.Magic(mime=True)
                self.magic_available = True
                logger.info("python-magic initialized successfully")
            else:
                raise ImportError("python-magic not available")
        except Exception as e:
            logger.warning(f"python-magic not available: {e}")
            self.magic_available = False
            self.magic_detector = None
    
    def detect_mime_type(self, file_path: Path) -> Tuple[Optional[str], float]:
        """
        Detect MIME type using multiple methods with confidence scoring.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Tuple of (mime_type, confidence_score)
        """
        confidence = 0.0
        detected_mime = None
        
        # Method 1: python-magic (most reliable)
        if self.magic_available and self.magic_detector:
            try:
                detected_mime = self.magic_detector.from_file(str(file_path))
                confidence = 0.9
                logger.debug(f"Magic detection: {detected_mime}")
            except Exception as e:
                logger.warning(f"Magic detection failed: {e}")
        
        # Method 2: filetype library (binary signature)
        if not detected_mime or confidence < 0.8:
            try:
                kind = filetype.guess(str(file_path))
                if kind:
                    detected_mime = kind.mime
                    confidence = max(confidence, 0.8)
                    logger.debug(f"Filetype detection: {detected_mime}")
            except Exception as e:
                logger.warning(f"Filetype detection failed: {e}")
        
        # Method 3: mimetypes (extension-based fallback)
        if not detected_mime or confidence < 0.5:
            try:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type:
                    detected_mime = mime_type
                    confidence = max(confidence, 0.5)
                    logger.debug(f"Mimetypes detection: {detected_mime}")
            except Exception as e:
                logger.warning(f"Mimetypes detection failed: {e}")
        
        return detected_mime, confidence
    
    def detect_document_type(self, file_path: Union[str, Path], 
                           input_source: InputSource = InputSource.API_UPLOAD) -> DetectionResult:
        """
        Main method to detect document type and create routing instructions.
        
        Args:
            file_path: Path to the document file
            input_source: Source of the input (web, folder, API)
            
        Returns:
            DetectionResult with metadata and routing instructions
        """
        file_path = Path(file_path)
        
        # Initialize metadata
        try:
            file_size = file_path.stat().st_size
        except Exception as e:
            logger.error(f"Cannot access file {file_path}: {e}")
            return self._create_error_result(file_path, f"File access error: {e}")
        
        # Detect MIME type first
        mime_type, confidence = self.detect_mime_type(file_path)
        
        # Determine document type
        doc_type = self._determine_document_type(mime_type, file_path.suffix.lower())
        
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path.absolute()),
            file_size=file_size,
            input_source=input_source,
            source_type=doc_type,
            mime_type=mime_type,
            confidence_score=confidence,
            detected_extension=file_path.suffix.lower()
        )
        
        # Create routing instructions
        routing_instructions = self._create_routing_instructions(doc_type, metadata)
        next_steps = self._generate_next_steps(doc_type, metadata)
        
        logger.info(f"Detected {file_path.name} as {doc_type} with confidence {confidence:.2f}")
        
        return DetectionResult(
            metadata=metadata,
            routing_instructions=routing_instructions,
            next_steps=next_steps,
            success=True
        )
    
    def _determine_document_type(self, mime_type: Optional[str], 
                               extension: str) -> DocumentType:
        """Determine document type from MIME type and extension."""
        # Primary: MIME type lookup
        if mime_type and mime_type in self.MIME_TO_TYPE:
            return self.MIME_TO_TYPE[mime_type]
        
        # Secondary: Extension lookup
        if extension in self.EXT_TO_TYPE:
            return self.EXT_TO_TYPE[extension]
        
        # Special cases for email files
        if extension in ['.eml', '.msg'] or (mime_type and 'message' in mime_type):
            return DocumentType.EMAIL
        
        # Image MIME type prefix matching
        if mime_type and mime_type.startswith('image/'):
            return DocumentType.IMAGE
        
        logger.warning(f"Unknown document type for MIME: {mime_type}, ext: {extension}")
        return DocumentType.UNKNOWN
    
    def _create_routing_instructions(self, doc_type: DocumentType, 
                                   metadata: DocumentMetadata) -> Dict[str, Any]:
        """Create detailed routing instructions for the document."""
        instructions = {
            "document_type": doc_type,
            "recommended_parsers": self.TYPE_TO_PARSER.get(doc_type, []),
            "processing_priority": self._get_processing_priority(doc_type),
            "requires_preprocessing": False,
            "parallel_processing_safe": True,
        }
        
        # Type-specific instructions
        if doc_type == DocumentType.PDF:
            instructions.update({
                "try_text_extraction_first": True,
                "fallback_to_ocr": True,
                "ocr_trigger_conditions": ["empty_text", "low_confidence", "image_based_pdf"],
                "preprocessing_steps": ["check_text_layer", "analyze_image_content"]
            })
            metadata.is_ocr_required = False  # Will be determined during parsing
            
        elif doc_type == DocumentType.DOCX:
            instructions.update({
                "extract_text": True,
                "extract_images": True,
                "ocr_embedded_images": True,
                "preserve_formatting": True
            })
            metadata.is_ocr_required = True  # For embedded images
            
        elif doc_type == DocumentType.EMAIL:
            instructions.update({
                "extract_headers": True,
                "extract_body": True,
                "process_attachments": True,
                "recursive_attachment_processing": True,
                "parallel_processing_safe": False  # Due to attachment dependencies
            })
            metadata.has_attachments = True  # Assumption for emails
            
        elif doc_type == DocumentType.IMAGE:
            instructions.update({
                "direct_ocr": True,
                "preprocessing_steps": ["image_enhancement", "noise_reduction"],
                "ocr_engines": ["paddleocr", "tesseract"],
                "language_detection": True
            })
            metadata.is_ocr_required = True
            metadata.ocr_triggered = True
            
        return instructions
    
    def _generate_next_steps(self, doc_type: DocumentType, 
                           metadata: DocumentMetadata) -> List[str]:
        """Generate specific next steps for processing."""
        steps = []
        
        if doc_type == DocumentType.PDF:
            steps.extend([
                "Initialize PDF parser (pdfplumber/PyMuPDF)",
                "Attempt text extraction",
                "Evaluate text quality and completeness",
                "If text extraction fails or poor quality, route to OCR pipeline"
            ])
            
        elif doc_type == DocumentType.DOCX:
            steps.extend([
                "Initialize python-docx parser",
                "Extract document text and structure",
                "Identify and extract embedded images",
                "Route images to OCR pipeline",
                "Combine text and OCR results"
            ])
            
        elif doc_type == DocumentType.EMAIL:
            steps.extend([
                "Initialize mail-parser",
                "Extract email headers and metadata",
                "Extract email body (text/HTML)",
                "Identify and extract attachments",
                "Route each attachment through document detection pipeline",
                "Aggregate all extracted content"
            ])
            
        elif doc_type == DocumentType.IMAGE:
            steps.extend([
                "Initialize OCR engine (PaddleOCR recommended)",
                "Preprocess image (enhancement, noise reduction)",
                "Perform OCR text extraction",
                "Post-process OCR results (cleanup, validation)"
            ])
            
        else:
            steps.append("Document type unknown - manual review required")
        
        return steps
    
    def _get_processing_priority(self, doc_type: DocumentType) -> int:
        """Get processing priority (1=highest, 5=lowest)."""
        priority_map = {
            DocumentType.PDF: 2,
            DocumentType.DOCX: 1,
            DocumentType.EMAIL: 3,
            DocumentType.IMAGE: 2,
            DocumentType.UNKNOWN: 5
        }
        return priority_map.get(doc_type, 5)
    
    def _create_error_result(self, file_path: Path, error_message: str) -> DetectionResult:
        """Create an error result for failed detection."""
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path.absolute()),
            file_size=0,
            source_type=DocumentType.UNKNOWN,
            error_message=error_message
        )
        
        return DetectionResult(
            metadata=metadata,
            routing_instructions={"error": True},
            next_steps=["Fix file access issue", "Retry detection"],
            success=False
        )
    
    def batch_detect(self, file_paths: List[Union[str, Path]], 
                    input_source: InputSource = InputSource.FOLDER_DROP) -> List[DetectionResult]:
        """
        Batch process multiple files for type detection.
        
        Args:
            file_paths: List of file paths to process
            input_source: Source of the input files
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        logger.info(f"Starting batch detection for {len(file_paths)} files")
        
        for file_path in file_paths:
            try:
                result = self.detect_document_type(file_path, input_source)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch detection failed for {file_path}: {e}")
                error_result = self._create_error_result(Path(file_path), str(e))
                results.append(error_result)
        
        logger.info(f"Batch detection completed: {len(results)} results")
        return results
    
    def export_results(self, results: List[DetectionResult], 
                      output_path: Union[str, Path]) -> bool:
        """
        Export detection results to JSON file.
        
        Args:
            results: List of detection results
            output_path: Path to save the JSON file
            
        Returns:
            Success status
        """
        try:
            output_data = {
                "detection_summary": {
                    "total_files": len(results),
                    "successful_detections": sum(1 for r in results if r.success),
                    "failed_detections": sum(1 for r in results if not r.success),
                    "export_timestamp": datetime.now().isoformat()
                },
                "results": [result.model_dump() for result in results]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False


# Convenience function for quick detection
def detect_document_type(file_path: Union[str, Path], 
                        input_source: InputSource = InputSource.API_UPLOAD) -> DetectionResult:
    """
    Quick function to detect document type.
    
    Args:
        file_path: Path to the document
        input_source: Source of the input
        
    Returns:
        DetectionResult with metadata and routing instructions
    """
    detector = DocumentTypeDetector()
    return detector.detect_document_type(file_path, input_source)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = detect_document_type(file_path)
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print("Usage: python document_detector.py <file_path>")
