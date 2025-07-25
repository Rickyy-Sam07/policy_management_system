"""
Parser Factory and Coordinator
=============================

Central coordinator for all document parsers with intelligent routing.
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from document_detector import DocumentTypeDetector, DocumentType, DetectionResult
from parsers.pdf_parser import PDFParser, parse_pdf
from parsers.docx_parser import DOCXParser, parse_docx
from parsers.email_parser import EmailParser, parse_email
from parsers.ocr_parser import OCRParser, parse_image_ocr

from loguru import logger


@dataclass
class DocumentProcessingResult:
    """Comprehensive result of document processing."""
    detection_result: DetectionResult
    parsing_success: bool
    extracted_text: str
    parser_used: str
    confidence_score: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    attachments: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.attachments is None:
            self.attachments = []


class DocumentProcessor:
    """
    Central document processor that coordinates detection and parsing.
    """
    
    def __init__(self, ocr_lang: str = 'en', prefer_paddle_ocr: bool = True):
        """
        Initialize document processor.
        
        Args:
            ocr_lang: Language for OCR processing
            prefer_paddle_ocr: Whether to prefer PaddleOCR over Tesseract
        """
        self.ocr_lang = ocr_lang
        self.prefer_paddle_ocr = prefer_paddle_ocr
        
        # Initialize components
        self.detector = DocumentTypeDetector()
        
        # Initialize parsers lazily to avoid import errors
        self._pdf_parser = None
        self._docx_parser = None
        self._email_parser = None
        self._ocr_parser = None
        
        logger.info("Document Processor initialized")
    
    @property
    def pdf_parser(self) -> PDFParser:
        """Get PDF parser (lazy initialization)."""
        if self._pdf_parser is None:
            self._pdf_parser = PDFParser()
        return self._pdf_parser
    
    @property
    def docx_parser(self) -> DOCXParser:
        """Get DOCX parser (lazy initialization)."""
        if self._docx_parser is None:
            self._docx_parser = DOCXParser()
        return self._docx_parser
    
    @property
    def email_parser(self) -> EmailParser:
        """Get email parser (lazy initialization)."""
        if self._email_parser is None:
            self._email_parser = EmailParser()
        return self._email_parser
    
    @property
    def ocr_parser(self) -> OCRParser:
        """Get OCR parser (lazy initialization)."""
        if self._ocr_parser is None:
            self._ocr_parser = OCRParser(
                prefer_paddle=self.prefer_paddle_ocr,
                lang=self.ocr_lang
            )
        return self._ocr_parser
    
    def process_document(self, file_path: Union[str, Path]) -> DocumentProcessingResult:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentProcessingResult with comprehensive processing information
        """
        import time
        start_time = time.time()
        
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Step 1: Detect document type
            detection_result = self.detector.detect_document_type(file_path)
            
            if not detection_result.success:
                return DocumentProcessingResult(
                    detection_result=detection_result,
                    parsing_success=False,
                    extracted_text="",
                    parser_used="none",
                    processing_time=time.time() - start_time,
                    error_message=detection_result.metadata.error_message
                )
            
            # Step 2: Route to appropriate parser
            doc_type = detection_result.metadata.source_type
            
            if doc_type == DocumentType.PDF:
                result = self._process_pdf(file_path, detection_result)
            elif doc_type == DocumentType.DOCX:
                result = self._process_docx(file_path, detection_result)
            elif doc_type == DocumentType.EMAIL:
                result = self._process_email(file_path, detection_result)
            elif doc_type == DocumentType.IMAGE:
                result = self._process_image(file_path, detection_result)
            elif doc_type == DocumentType.TEXT:
                result = self._process_text(file_path, detection_result)
            else:
                result = DocumentProcessingResult(
                    detection_result=detection_result,
                    parsing_success=False,
                    extracted_text="",
                    parser_used="none",
                    error_message=f"Unsupported document type: {doc_type}"
                )
            
            result.processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            
            # Create default metadata if detection failed
            if 'detection_result' not in locals() or not detection_result or not detection_result.metadata:
                from document_detector import DocumentMetadata, InputSource
                default_metadata = DocumentMetadata(
                    filename=file_path.name,
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size if file_path.exists() else 0,
                    source_type=DocumentType.UNKNOWN,
                    input_source=InputSource.API_UPLOAD
                )
            else:
                default_metadata = detection_result.metadata
            
            from document_detector import DetectionResult
            return DocumentProcessingResult(
                detection_result=DetectionResult(
                    metadata=default_metadata,
                    routing_instructions={},
                    next_steps=[],
                    success=False
                ),
                parsing_success=False,
                extracted_text="",
                parser_used="none",
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _process_pdf(self, file_path: Path, detection_result: DetectionResult) -> DocumentProcessingResult:
        """Process PDF document."""
        try:
            pdf_result = self.pdf_parser.parse_pdf(file_path)
            
            # Check if OCR is needed
            if pdf_result.needs_ocr:
                logger.info("PDF needs OCR, rendering pages as images and processing")
                
                # Render PDF pages as images for OCR
                page_images = self.pdf_parser.render_pages_as_images(file_path, dpi=200)
                ocr_text = self._ocr_pdf_pages(page_images)
                
                # Combine original text (if any) with OCR results
                combined_text = pdf_result.text_content
                if ocr_text:
                    if combined_text.strip():
                        combined_text += "\n\n--- OCR TEXT FROM PAGES ---\n" + ocr_text
                    else:
                        combined_text = ocr_text
                
                return DocumentProcessingResult(
                    detection_result=detection_result,
                    parsing_success=True,
                    extracted_text=combined_text,
                    parser_used=f"{pdf_result.extraction_method}+page_ocr",
                    confidence_score=max(pdf_result.confidence_score, 0.7),  # OCR typically gives good results
                    metadata={
                        "pdf_metadata": pdf_result.metadata.__dict__,
                        "page_count": pdf_result.metadata.total_pages,
                        "image_count": pdf_result.image_count,
                        "ocr_applied": True,
                        "ocr_method": "page_rendering",
                        "pages_processed": len(page_images)
                    }
                )
            else:
                return DocumentProcessingResult(
                    detection_result=detection_result,
                    parsing_success=True,
                    extracted_text=pdf_result.text_content,
                    parser_used=pdf_result.extraction_method,
                    confidence_score=pdf_result.confidence_score,
                    metadata={
                        "pdf_metadata": pdf_result.metadata.__dict__,
                        "page_count": pdf_result.metadata.total_pages,
                        "image_count": pdf_result.image_count,
                        "ocr_applied": False
                    }
                )
                
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=False,
                extracted_text="",
                parser_used="pdf_parser",
                error_message=str(e)
            )
    
    def _process_docx(self, file_path: Path, detection_result: DetectionResult) -> DocumentProcessingResult:
        """Process DOCX document."""
        try:
            docx_result = self.docx_parser.parse_docx(file_path)
            
            # Check if OCR fallback is needed
            needs_ocr = self.docx_parser.needs_ocr_fallback(
                docx_result.text_content, 
                docx_result.embedded_images
            )
            
            # Process embedded images with OCR if any or if fallback needed
            ocr_text = ""
            if docx_result.embedded_images and (needs_ocr or len(docx_result.embedded_images) > 0):
                logger.info(f"Processing {len(docx_result.embedded_images)} embedded images with OCR "
                           f"(OCR fallback needed: {needs_ocr})")
                ocr_text = self._ocr_embedded_images(docx_result.embedded_images)
            
            # Combine text and OCR results
            combined_text = docx_result.text_content
            if ocr_text:
                if combined_text.strip():
                    combined_text += "\n\n--- EMBEDDED IMAGE OCR ---\n" + ocr_text
                else:
                    # If no text was extracted, use only OCR results
                    combined_text = ocr_text
            
            # Determine final confidence and parser used
            final_confidence = docx_result.confidence_score
            parser_used = docx_result.extraction_method
            
            if ocr_text:
                parser_used += "+ocr"
                if needs_ocr:
                    # OCR was critical, boost confidence
                    final_confidence = max(final_confidence, 0.7)
            
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=True,
                extracted_text=combined_text,
                parser_used=parser_used,
                confidence_score=final_confidence,
                metadata={
                    "docx_metadata": docx_result.metadata.__dict__,
                    "paragraph_count": len(docx_result.paragraph_texts),
                    "table_count": len(docx_result.table_texts),
                    "embedded_image_count": len(docx_result.embedded_images),
                    "structured_content": docx_result.structured_content,
                    "ocr_fallback_used": needs_ocr,
                    "text_quality_issues": needs_ocr
                }
            )
            
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}")
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=False,
                extracted_text="",
                parser_used="docx_parser",
                error_message=str(e)
            )
    
    def _process_email(self, file_path: Path, detection_result: DetectionResult) -> DocumentProcessingResult:
        """Process email document."""
        try:
            email_result = self.email_parser.parse_email(file_path)
            
            # Process attachments recursively
            attachment_texts = []
            processed_attachments = []
            
            if email_result.attachments:
                logger.info(f"Processing {len(email_result.attachments)} email attachments")
                
                for i, attachment in enumerate(email_result.attachments):
                    try:
                        # Save attachment temporarily and process it
                        temp_dir = Path("temp_attachments")
                        temp_dir.mkdir(exist_ok=True)
                        
                        temp_file = temp_dir / f"attachment_{i}_{attachment.filename}"
                        
                        if attachment.data:
                            with open(temp_file, 'wb') as f:
                                f.write(attachment.data)
                            
                            # Check if attachment is an image for direct OCR
                            if attachment.content_type.startswith('image/'):
                                logger.debug(f"Processing image attachment: {attachment.filename}")
                                try:
                                    ocr_result = self.ocr_parser.parse_image(temp_file, preprocess=True)
                                    if ocr_result.text_content.strip():
                                        attachment_texts.append(f"--- {attachment.filename} (OCR) ---\n{ocr_result.text_content}")
                                        
                                        processed_attachments.append({
                                            "filename": attachment.filename,
                                            "content_type": attachment.content_type,
                                            "size_bytes": attachment.size_bytes,
                                            "processing_success": True,
                                            "processing_method": "direct_ocr",
                                            "extracted_text_length": len(ocr_result.text_content),
                                            "ocr_confidence": ocr_result.confidence_score
                                        })
                                    else:
                                        processed_attachments.append({
                                            "filename": attachment.filename,
                                            "content_type": attachment.content_type,
                                            "size_bytes": attachment.size_bytes,
                                            "processing_success": False,
                                            "processing_method": "direct_ocr",
                                            "error": "No text extracted from image"
                                        })
                                except Exception as ocr_e:
                                    logger.warning(f"OCR failed for image attachment {attachment.filename}: {ocr_e}")
                                    processed_attachments.append({
                                        "filename": attachment.filename,
                                        "content_type": attachment.content_type,
                                        "size_bytes": attachment.size_bytes,
                                        "processing_success": False,
                                        "processing_method": "direct_ocr",
                                        "error": str(ocr_e)
                                    })
                            else:
                                # Recursively process non-image attachments
                                attachment_result = self.process_document(temp_file)
                                
                                if attachment_result.parsing_success:
                                    attachment_texts.append(f"--- {attachment.filename} ---\n{attachment_result.extracted_text}")
                                
                                processed_attachments.append({
                                    "filename": attachment.filename,
                                    "content_type": attachment.content_type,
                                    "size_bytes": attachment.size_bytes,
                                    "processing_success": attachment_result.parsing_success,
                                    "processing_method": "recursive_processing",
                                    "extracted_text_length": len(attachment_result.extracted_text),
                                    "parser_used": attachment_result.parser_used
                                })
                            
                            # Clean up
                            temp_file.unlink(missing_ok=True)
                    
                    except Exception as e:
                        logger.warning(f"Failed to process attachment {attachment.filename}: {e}")
                        continue
            
            # Combine email content and attachment texts
            combined_text = email_result.text_content
            if email_result.html_content and len(email_result.html_content) > len(email_result.text_content):
                combined_text = email_result.html_content
            
            if attachment_texts:
                combined_text += "\n\n--- ATTACHMENTS ---\n" + "\n\n".join(attachment_texts)
            
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=True,
                extracted_text=combined_text,
                parser_used=email_result.extraction_method,
                confidence_score=email_result.confidence_score,
                metadata={
                    "email_metadata": email_result.metadata.__dict__,
                    "has_html": bool(email_result.html_content),
                    "has_text": bool(email_result.text_content),
                    "structured_content": email_result.structured_content
                },
                attachments=processed_attachments
            )
            
        except Exception as e:
            logger.error(f"Email processing failed: {e}")
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=False,
                extracted_text="",
                parser_used="email_parser",
                error_message=str(e)
            )
    
    def _process_image(self, file_path: Path, detection_result: DetectionResult) -> DocumentProcessingResult:
        """Process image document with OCR."""
        try:
            ocr_result = self.ocr_parser.parse_image(file_path, preprocess=True)
            
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=True,
                extracted_text=ocr_result.text_content,
                parser_used=ocr_result.extraction_method,
                confidence_score=ocr_result.confidence_score,
                metadata={
                    "image_metadata": ocr_result.metadata.__dict__,
                    "text_blocks": len(ocr_result.text_blocks),
                    "preprocessing_applied": ocr_result.preprocessing_applied,
                    "detected_languages": ocr_result.detected_languages
                }
            )
            
        except Exception as e:
            logger.error(f"Image OCR processing failed: {e}")
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=False,
                extracted_text="",
                parser_used="ocr_parser",
                error_message=str(e)
            )
    
    def _ocr_pdf_pages(self, page_images: List[Dict[str, Any]]) -> str:
        """
        OCR rendered PDF pages.
        
        Args:
            page_images: List of page image data from PDF rendering
            
        Returns:
            Combined OCR text from all pages
        """
        if not page_images:
            return ""
        
        ocr_texts = []
        logger.info(f"Starting OCR processing for {len(page_images)} PDF pages")
        
        for page_data in page_images:
            try:
                page_num = page_data['page_number']
                image_data = page_data['data']
                
                # Create temporary file for OCR
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_file.write(image_data)
                    temp_path = Path(temp_file.name)
                
                try:
                    # Run OCR on the page image
                    ocr_result = self.ocr_parser.parse_image(temp_path, preprocess=True)
                    
                    if ocr_result.text_content.strip():
                        page_text = f"--- PAGE {page_num} ---\n{ocr_result.text_content}"
                        ocr_texts.append(page_text)
                        logger.debug(f"Page {page_num}: OCR extracted {len(ocr_result.text_content)} characters "
                                   f"(confidence: {ocr_result.confidence_score:.2f})")
                    else:
                        logger.debug(f"Page {page_num}: No text extracted via OCR")
                
                finally:
                    # Clean up temporary file
                    temp_path.unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning(f"OCR failed for page {page_data.get('page_number', '?')}: {e}")
                continue
        
        combined_text = "\n\n".join(ocr_texts)
        logger.info(f"PDF OCR completed: {len(combined_text)} total characters from {len(ocr_texts)} pages")
        return combined_text
    
    def _process_text(self, file_path: Path, detection_result: DetectionResult) -> DocumentProcessingResult:
        """Process plain text document."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=True,
                extracted_text=text_content,
                parser_used="text_reader",
                confidence_score=1.0,
                metadata={"encoding": "utf-8", "file_type": "text"}
            )
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return DocumentProcessingResult(
                detection_result=detection_result,
                parsing_success=False,
                extracted_text="",
                parser_used="text_reader",
                error_message=str(e)
            )
    
    def _ocr_pdf_images(self, pdf_path: Path, images: List[Dict[str, Any]]) -> str:
        """
        OCR extracted images from PDF (legacy method).
        
        Args:
            pdf_path: Path to the PDF file
            images: List of image metadata from PDF
            
        Returns:
            Combined OCR text from extracted images
        """
        # This method is kept for compatibility but page rendering is preferred
        logger.info(f"Using legacy image extraction OCR for {len(images)} images")
        return ""
    
    def _ocr_embedded_images(self, embedded_images: List[Any]) -> str:
        """OCR embedded images from DOCX."""
        ocr_texts = []
        
        for i, image in enumerate(embedded_images):
            try:
                if hasattr(image, 'data') and image.data:
                    # Create temporary file for OCR
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=f'.{image.format}', delete=False) as temp_file:
                        temp_file.write(image.data)
                        temp_path = Path(temp_file.name)
                    
                    # Run OCR
                    ocr_result = self.ocr_parser.parse_image(temp_path, preprocess=True)
                    
                    if ocr_result.text_content.strip():
                        ocr_texts.append(f"Image {i+1} ({image.name}): {ocr_result.text_content}")
                    
                    # Clean up
                    temp_path.unlink(missing_ok=True)
            
            except Exception as e:
                logger.warning(f"Failed to OCR embedded image {i}: {e}")
                continue
        
        return "\n\n".join(ocr_texts)
    
    def batch_process(self, file_paths: List[Union[str, Path]]) -> List[DocumentProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of DocumentProcessingResult objects
        """
        results = []
        logger.info(f"Starting batch processing for {len(file_paths)} documents")
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed for {file_path}: {e}")
                # Create error result
                error_result = DocumentProcessingResult(
                    detection_result=DetectionResult(
                        metadata=None,
                        routing_instructions={},
                        next_steps=[],
                        success=False
                    ),
                    parsing_success=False,
                    extracted_text="",
                    parser_used="none",
                    error_message=str(e)
                )
                results.append(error_result)
        
        successful = sum(1 for r in results if r.parsing_success)
        logger.info(f"Batch processing completed: {successful}/{len(results)} successful")
        return results


# Convenience function for quick processing
def process_document(file_path: Union[str, Path], ocr_lang: str = 'en') -> DocumentProcessingResult:
    """
    Quick function to process a document.
    
    Args:
        file_path: Path to the document
        ocr_lang: Language for OCR processing
        
    Returns:
        DocumentProcessingResult with processing information
    """
    processor = DocumentProcessor(ocr_lang=ocr_lang)
    return processor.process_document(file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    import json
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = process_document(file_path)
        
        # Print summary
        print(f"Document: {file_path}")
        print(f"Type: {result.detection_result.metadata.source_type if result.detection_result else 'Unknown'}")
        print(f"Success: {result.parsing_success}")
        print(f"Parser: {result.parser_used}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Text Length: {len(result.extracted_text)} characters")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        # Show first 200 characters of extracted text
        if result.extracted_text:
            print("\nExtracted Text (preview):")
            print(result.extracted_text[:200] + "..." if len(result.extracted_text) > 200 else result.extracted_text)
    else:
        print("Usage: python document_processor.py <file_path>")
