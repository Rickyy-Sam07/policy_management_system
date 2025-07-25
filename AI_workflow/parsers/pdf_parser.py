"""
PDF Parser Module
================

Optimized PDF parsing with intelligent text extraction and OCR fallback.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import io
from dataclasses import dataclass

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from loguru import logger


@dataclass
class PDFMetadata:
    """Metadata extracted from PDF."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    total_pages: int = 0
    is_encrypted: bool = False
    has_text_layer: bool = False
    has_images: bool = False


@dataclass
class PDFParseResult:
    """Result of PDF parsing operation."""
    text_content: str
    metadata: PDFMetadata
    page_texts: List[str]
    needs_ocr: bool = False
    confidence_score: float = 0.0
    extraction_method: str = ""
    error_message: Optional[str] = None
    image_count: int = 0


class PDFParser:
    """
    Optimized PDF parser with multiple extraction strategies.
    """
    
    def __init__(self):
        """Initialize PDF parser with available libraries."""
        self.pdfplumber_available = PDFPLUMBER_AVAILABLE
        self.pymupdf_available = PYMUPDF_AVAILABLE
        
        if not (self.pdfplumber_available or self.pymupdf_available):
            raise ImportError("Neither pdfplumber nor PyMuPDF is available")
        
        logger.info(f"PDF Parser initialized - pdfplumber: {self.pdfplumber_available}, PyMuPDF: {self.pymupdf_available}")
    
    def parse_pdf(self, file_path: Path) -> PDFParseResult:
        """
        Parse PDF with intelligent method selection.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDFParseResult with extracted content and metadata
        """
        logger.info(f"Starting PDF parsing for {file_path}")
        
        # Try pdfplumber first (better text extraction)
        if self.pdfplumber_available:
            try:
                result = self._parse_with_pdfplumber(file_path)
                if result.confidence_score > 0.7:
                    logger.info(f"pdfplumber extraction successful with confidence {result.confidence_score:.2f}")
                    return result
                else:
                    logger.info(f"pdfplumber extraction low confidence ({result.confidence_score:.2f}), trying PyMuPDF")
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        # Fallback to PyMuPDF
        if self.pymupdf_available:
            try:
                result = self._parse_with_pymupdf(file_path)
                logger.info(f"PyMuPDF extraction completed with confidence {result.confidence_score:.2f}")
                return result
            except Exception as e:
                logger.error(f"PyMuPDF also failed: {e}")
                return PDFParseResult(
                    text_content="",
                    metadata=PDFMetadata(),
                    page_texts=[],
                    needs_ocr=True,
                    error_message=f"All PDF parsing methods failed: {e}"
                )
        
        return PDFParseResult(
            text_content="",
            metadata=PDFMetadata(),
            page_texts=[],
            needs_ocr=True,
            error_message="No PDF parsing libraries available"
        )
    
    def _parse_with_pdfplumber(self, file_path: Path) -> PDFParseResult:
        """Parse PDF using pdfplumber."""
        page_texts = []
        all_text = ""
        image_count = 0
        
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            metadata = self._extract_metadata_pdfplumber(pdf)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text() or ""
                    page_texts.append(page_text)
                    all_text += page_text + "\n"
                    
                    # Count images
                    if hasattr(page, 'images'):
                        image_count += len(page.images)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    page_texts.append("")
        
        # Evaluate extraction quality
        confidence, needs_ocr = self._evaluate_extraction_quality(all_text, metadata.total_pages)
        
        return PDFParseResult(
            text_content=all_text.strip(),
            metadata=metadata,
            page_texts=page_texts,
            needs_ocr=needs_ocr,
            confidence_score=confidence,
            extraction_method="pdfplumber",
            image_count=image_count
        )
    
    def _parse_with_pymupdf(self, file_path: Path) -> PDFParseResult:
        """Parse PDF using PyMuPDF."""
        page_texts = []
        all_text = ""
        image_count = 0
        
        doc = fitz.open(file_path)
        
        try:
            # Extract metadata
            metadata = self._extract_metadata_pymupdf(doc)
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                try:
                    page_text = page.get_text()
                    page_texts.append(page_text)
                    all_text += page_text + "\n"
                    
                    # Count images
                    image_list = page.get_images()
                    image_count += len(image_list)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    page_texts.append("")
        
        finally:
            doc.close()
        
        # Evaluate extraction quality
        confidence, needs_ocr = self._evaluate_extraction_quality(all_text, metadata.total_pages)
        
        return PDFParseResult(
            text_content=all_text.strip(),
            metadata=metadata,
            page_texts=page_texts,
            needs_ocr=needs_ocr,
            confidence_score=confidence,
            extraction_method="pymupdf",
            image_count=image_count
        )
    
    def _extract_metadata_pdfplumber(self, pdf) -> PDFMetadata:
        """Extract metadata using pdfplumber."""
        metadata_dict = pdf.metadata or {}
        
        return PDFMetadata(
            title=metadata_dict.get('Title'),
            author=metadata_dict.get('Author'),
            subject=metadata_dict.get('Subject'),
            creator=metadata_dict.get('Creator'),
            producer=metadata_dict.get('Producer'),
            creation_date=str(metadata_dict.get('CreationDate', '')),
            modification_date=str(metadata_dict.get('ModDate', '')),
            total_pages=len(pdf.pages),
            is_encrypted=pdf.is_encrypted if hasattr(pdf, 'is_encrypted') else False
        )
    
    def _extract_metadata_pymupdf(self, doc) -> PDFMetadata:
        """Extract metadata using PyMuPDF."""
        metadata_dict = doc.metadata
        
        return PDFMetadata(
            title=metadata_dict.get('title'),
            author=metadata_dict.get('author'),
            subject=metadata_dict.get('subject'),
            creator=metadata_dict.get('creator'),
            producer=metadata_dict.get('producer'),
            creation_date=metadata_dict.get('creationDate', ''),
            modification_date=metadata_dict.get('modDate', ''),
            total_pages=len(doc),
            is_encrypted=doc.needs_pass,
            has_text_layer=self._check_text_layer_pymupdf(doc)
        )
    
    def _check_text_layer_pymupdf(self, doc) -> bool:
        """Check if PDF has searchable text layer."""
        # Sample first few pages to check for text
        sample_pages = min(3, len(doc))
        total_text_length = 0
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            text = page.get_text()
            total_text_length += len(text.strip())
        
        # If we have reasonable amount of text in first few pages, assume text layer exists
        return total_text_length > 50
    
    def _evaluate_extraction_quality(self, text: str, total_pages: int) -> Tuple[float, bool]:
        """
        Evaluate the quality of text extraction and determine if OCR is needed.
        
        Args:
            text: Extracted text content
            total_pages: Total number of pages in PDF
            
        Returns:
            Tuple of (confidence_score, needs_ocr)
        """
        if not text or not text.strip():
            return 0.0, True
        
        text_length = len(text.strip())
        
        # Basic heuristics for text quality
        words = text.split()
        word_count = len(words)
        
        # Calculate confidence based on various factors
        confidence_factors = []
        
        # Factor 1: Text length vs page count
        avg_chars_per_page = text_length / total_pages if total_pages > 0 else 0
        if avg_chars_per_page > 500:
            confidence_factors.append(0.9)
        elif avg_chars_per_page > 200:
            confidence_factors.append(0.7)
        elif avg_chars_per_page > 50:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Factor 2: Word density and coherence
        if word_count > 20:
            avg_word_length = text_length / word_count
            if 3 <= avg_word_length <= 8:  # Reasonable word lengths
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.3)
        
        # Factor 3: Character variety (not just symbols or garbled text)
        unique_chars = len(set(text.lower()))
        if unique_chars > 20:  # Good character variety
            confidence_factors.append(0.8)
        elif unique_chars > 10:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Factor 4: Presence of common English patterns
        common_words = ['the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at']
        text_lower = text.lower()
        common_word_count = sum(1 for word in common_words if word in text_lower)
        
        if common_word_count >= 5:
            confidence_factors.append(0.9)
        elif common_word_count >= 3:
            confidence_factors.append(0.7)
        elif common_word_count >= 1:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Factor 5: Check for mostly special characters or whitespace
        if text_length > 0:
            # Count actual alphanumeric characters
            alphanumeric_chars = sum(1 for c in text if c.isalnum())
            alphanumeric_ratio = alphanumeric_chars / text_length
            
            if alphanumeric_ratio > 0.7:  # Good content
                confidence_factors.append(0.8)
            elif alphanumeric_ratio > 0.4:  # Moderate content
                confidence_factors.append(0.6)
            else:  # Mostly special chars/whitespace
                confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.0)
        
        # Recalculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Enhanced OCR triggering conditions
        needs_ocr = (
            overall_confidence < 0.6 or          # Low confidence
            text_length < 50 or                  # Very little text
            (text_length > 0 and alphanumeric_chars / text_length < 0.3)  # Mostly special chars
        )
        
        logger.debug(f"Text extraction quality - Length: {text_length}, Words: {word_count}, "
                    f"Alphanumeric ratio: {alphanumeric_chars/text_length if text_length > 0 else 0:.2f}, "
                    f"Confidence: {overall_confidence:.2f}, Needs OCR: {needs_ocr}")
        
        return overall_confidence, needs_ocr
    

    
    def render_pages_as_images(self, file_path: Path, dpi: int = 200) -> List[Dict[str, Any]]:
        """
        Render PDF pages as images for OCR processing.
        
        Args:
            file_path: Path to the PDF file
            dpi: Resolution for rendering (higher = better quality, slower)
            
        Returns:
            List of page image data dictionaries
        """
        page_images = []
        
        if not self.pymupdf_available:
            logger.warning("PyMuPDF not available for page rendering")
            return page_images
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    
                    # Create transformation matrix for the desired DPI
                    zoom = dpi / 72  # 72 is the default DPI
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Render page as image
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to bytes
                    img_data = pix.tobytes("png")
                    
                    page_image = {
                        'page_number': page_num + 1,
                        'width': pix.width,
                        'height': pix.height,
                        'dpi': dpi,
                        'format': 'png',
                        'data': img_data,
                        'size_bytes': len(img_data)
                    }
                    
                    page_images.append(page_image)
                    logger.debug(f"Rendered page {page_num + 1}: {pix.width}x{pix.height} at {dpi} DPI")
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    logger.warning(f"Failed to render page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to render PDF pages: {e}")
        
        logger.info(f"Rendered {len(page_images)} pages as images")
        return page_images


def parse_pdf(file_path: Path) -> PDFParseResult:
    """
    Convenience function to parse a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PDFParseResult with extracted content
    """
    parser = PDFParser()
    return parser.parse_pdf(file_path)
