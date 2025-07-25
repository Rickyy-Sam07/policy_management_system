#!/usr/bin/env python3
"""Test OCR debug functionality"""

from pathlib import Path
from document_processor import DocumentProcessor

def test_ocr_debug():
    """Test that OCR debug files are created when needed."""
    processor = DocumentProcessor(save_ocr_debug=True)
    
    # Test the save_ocr_debug method directly
    test_ocr_text = """This is sample OCR extracted text.
It contains multiple lines and special characters: áéíóú
Numbers: 123456789
Symbols: @#$%^&*()"""
    
    source_file = Path("test_files/sample.txt")
    debug_file = processor._save_ocr_debug(test_ocr_text, source_file, "test_ocr")
    
    if debug_file:
        print(f"✅ OCR debug file created: {debug_file}")
        with open(debug_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📄 Debug file contains {len(content)} characters")
        print("🔍 First 200 characters:")
        print(content[:200])
    else:
        print("❌ OCR debug file was not created")

if __name__ == "__main__":
    test_ocr_debug()
