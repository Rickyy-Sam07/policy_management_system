#!/usr/bin/env python3
"""
Test Multi-Format Document Processing
"""

from multi_format_processor import MultiFormatProcessor

def test_format_detection():
    """Test document format detection"""
    
    processor = MultiFormatProcessor()
    
    # Test cases
    test_cases = [
        ("https://example.com/doc.pdf", "application/pdf", ".pdf"),
        ("https://example.com/doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        ("https://example.com/email.eml", "message/rfc822", ".eml"),
        ("https://example.com/unknown", "text/plain", ".pdf")  # fallback
    ]
    
    for url, content_type, expected in test_cases:
        result = processor._detect_format(url, content_type)
        print(f"URL: {url} -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ Format detection tests passed!")

if __name__ == "__main__":
    test_format_detection()
    print("🚀 Multi-format processor ready!")