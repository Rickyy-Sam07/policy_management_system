#!/usr/bin/env python3
"""Simple test for document processor"""

from document_processor import DocumentProcessor
from pathlib import Path

def test_simple():
    processor = DocumentProcessor()
    result = processor.process_document(Path("test_files/sample.txt"))
    print(f"Success: {result.parsing_success}")
    print(f"Text length: {len(result.extracted_text)}")
    print(f"Type: {result.detection_result.metadata.source_type}")
    if result.error_message:
        print(f"Error: {result.error_message}")

if __name__ == "__main__":
    test_simple()
