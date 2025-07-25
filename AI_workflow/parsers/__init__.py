"""
Parsers Package
==============

Package initialization for document parsers.
"""

from .pdf_parser import PDFParser, parse_pdf
from .docx_parser import DOCXParser, parse_docx
from .email_parser import EmailParser, parse_email
from .ocr_parser import OCRParser, parse_image_ocr

__all__ = [
    'PDFParser', 'parse_pdf',
    'DOCXParser', 'parse_docx',
    'EmailParser', 'parse_email',
    'OCRParser', 'parse_image_ocr'
]
