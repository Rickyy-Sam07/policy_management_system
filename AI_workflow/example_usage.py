"""
Example Usage and Testing Script
==============================

Demonstrates how to use the document processing system.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor, process_document
from document_detector import DocumentType


def save_extracted_text(result, source_file: str, output_dir: Path = None):
    if output_dir is None:
        output_dir = Path("extracted_texts")
    output_dir.mkdir(exist_ok=True)
    
    try:
        source_path = Path(source_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{source_path.stem}_{timestamp}_extracted.txt"
        output_path.write_text(result.extracted_text or "(No text extracted)", encoding='utf-8')
        print(f"Saved {len(result.extracted_text):,} characters to {output_path.name}")
        return output_path
    except Exception as e:
        print(f"Failed to save extracted text: {e}")
        return None





def demo_single_document(file_path: str):
    """Demonstrate processing a single document."""
    print(f"\n{'='*60}")
    print(f"PROCESSING SINGLE DOCUMENT: {file_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = process_document(file_path)
        
        print(f"Processing completed in {time.time() - start_time:.2f} seconds")
        
        # Save extracted text to file
        saved_file = save_extracted_text(result, file_path)
        
        print(f"\nDETECTION RESULTS:")
        print(f"   File: {result.detection_result.metadata.filename}")
        print(f"   Type: {result.detection_result.metadata.source_type}")
        print(f"   MIME: {result.detection_result.metadata.mime_type}")
        print(f"   Size: {result.detection_result.metadata.file_size:,} bytes")
        print(f"   Confidence: {result.detection_result.metadata.confidence_score:.2f}")
        
        print(f"\nPARSING RESULTS:")
        print(f"   Success: {result.parsing_success}")
        print(f"   Parser Used: {result.parser_used}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Text Length: {len(result.extracted_text):,} characters")
        
        if result.metadata:
            print(f"\nMETADATA:")
            for key, value in result.metadata.items():
                if isinstance(value, dict):
                    print(f"   {key}: {len(value)} items")
                elif isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {value}")
        
        if result.attachments:
            print(f"\nATTACHMENTS: {len(result.attachments)}")
            for i, att in enumerate(result.attachments):
                print(f"   {i+1}. {att.get('filename', 'Unknown')} ({att.get('size_bytes', 0):,} bytes)")
        
        if result.error_message:
            print(f"\nERROR: {result.error_message}")
        
        # Show text preview
        if result.extracted_text:
            print(f"\nTEXT PREVIEW (first 300 characters):")
            preview = result.extracted_text[:300]
            print(f"   {repr(preview)}")
            if len(result.extracted_text) > 300:
                print(f"   ... and {len(result.extracted_text) - 300:,} more characters")
        
    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()


def demo_batch_processing(file_paths: List[str]):
    """Demonstrate batch processing of multiple documents."""
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {len(file_paths)} documents")
    print(f"{'='*60}")
    
    processor = DocumentProcessor()
    start_time = time.time()
    
    # Create batch output directory
    batch_output_dir = Path("extracted_texts") / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        results = processor.batch_process(file_paths)
        total_time = time.time() - start_time
        
        print(f"Batch processing completed in {total_time:.2f} seconds")
        print(f"   Average: {total_time/len(results):.2f} seconds per document")
        
        # Save all extracted texts
        print(f"\nSaving extracted texts to: {batch_output_dir}")
        saved_files = []
        for i, (result, file_path) in enumerate(zip(results, file_paths)):
            saved_file = save_extracted_text(result, file_path, batch_output_dir)
            if saved_file:
                saved_files.append(saved_file)
        
        print(f"Saved {len(saved_files)} text files from batch processing")
        

        
        # Summary statistics
        successful = sum(1 for r in results if r.parsing_success)
        total_text = sum(len(r.extracted_text) for r in results)
        
        print(f"\nBATCH SUMMARY:")
        print(f"   Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"   Total text extracted: {total_text:,} characters")
        print(f"   Text files saved: {len(saved_files)}")
        
        # Group by document type
        type_counts = {}
        for result in results:
            if result.detection_result and result.detection_result.metadata:
                doc_type = result.detection_result.metadata.source_type
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        print(f"\nDOCUMENT TYPES:")
        for doc_type, count in type_counts.items():
            print(f"   {doc_type}: {count}")
        
        # Show individual results
        print(f"\nINDIVIDUAL RESULTS:")
        for i, result in enumerate(results):
            status = "SUCCESS" if result.parsing_success else "FAILED"
            filename = Path(file_paths[i]).name
            text_len = len(result.extracted_text)
            confidence = result.confidence_score
            print(f"   {status}: {filename}: {text_len:,} chars (confidence: {confidence:.2f})")
    
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()





def create_test_files():
    """Create sample test files for demonstration."""
    print(f"\n{'='*60}")
    print(f"CREATING TEST FILES")
    print(f"{'='*60}")
    
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple text file to simulate PDF content
    test_txt = test_dir / "sample.txt"
    with open(test_txt, 'w', encoding='utf-8') as f:
        f.write("""
        Sample Document Content
        =======================
        
        This is a sample document that demonstrates the document processing system.
        
        Key Features:
        • Intelligent document type detection
        • Multi-format support (PDF, DOCX, emails, images)
        • OCR capabilities for image-based content
        • Metadata extraction
        • Batch processing
        
        Processing Pipeline:
        1. File type detection using MIME types and magic numbers
        2. Route to appropriate parser
        3. Extract text content
        4. Apply OCR if needed
        5. Combine results with metadata
        
        This system can handle various document types and provides
        comprehensive extraction capabilities for enterprise use.
        """)
    
    print(f"Created test file: {test_txt}")
    return [str(test_txt)]


def export_results_to_json(results: List[Any], output_file: str):
    """Export processing results to JSON file."""
    print(f"\n📤 Exporting results to {output_file}")
    
    try:
        export_data = []
        for result in results:
            # Convert result to serializable format
            export_item = {
                "detection": {
                    "filename": result.detection_result.metadata.filename if result.detection_result else None,
                    "source_type": str(result.detection_result.metadata.source_type) if result.detection_result else None,
                    "mime_type": result.detection_result.metadata.mime_type if result.detection_result else None,
                    "confidence": result.detection_result.metadata.confidence_score if result.detection_result else 0.0
                },
                "parsing": {
                    "success": result.parsing_success,
                    "parser_used": result.parser_used,
                    "confidence": result.confidence_score,
                    "text_length": len(result.extracted_text),
                    "processing_time": result.processing_time,
                    "error": result.error_message
                },
                "metadata": result.metadata,
                "text_preview": result.extracted_text[:200] if result.extracted_text else ""
            }
            export_data.append(export_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Results exported successfully")
        
    except Exception as e:
        print(f"Export failed: {e}")


def main():
    """Main demonstration function."""
    print("Document Processing System Demo")
    print("=" * 60)
    
    # Ensure output directories exist
    Path("extracted_texts").mkdir(exist_ok=True)
    Path("ocr_debug").mkdir(exist_ok=True)
    print("Created output directories: extracted_texts/, ocr_debug/")
    
    # Check if files provided as arguments
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
        print(f"Processing {len(file_paths)} file(s) from command line")
        
        if len(file_paths) == 1:
            demo_single_document(file_paths[0])
        else:
            demo_batch_processing(file_paths)
    
    else:
        print("No files provided. Creating test files for demonstration...")
        test_files = create_test_files()
        

        
        # Process test files
        if test_files:
            demo_single_document(test_files[0])
    
    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"\nOutput locations:")
    print(f"   - Extracted texts: ./extracted_texts/")
    print(f"   - OCR debug files: ./ocr_debug/")
    print(f"   - All document content saved to txt files for analysis")
    print("\nTo test with your own files, run:")
    print("   python example_usage.py <file1> <file2> ...")
    print("\nSupported formats: PDF, DOCX, EML, MSG, JPG, PNG, TIFF, etc.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
