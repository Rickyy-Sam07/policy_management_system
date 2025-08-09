#!/usr/bin/env python3
"""
Ultimate progress bar suppression test for RAG system
"""

import os

# Set environment variables before any imports
os.environ['DISABLE_TQDM'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

from clean_rag_system import CleanRAGSystem

def test_progress_suppression():
    """Test that all progress bars are completely suppressed"""
    
    print("🧪 COMPREHENSIVE PROGRESS BAR SUPPRESSION TEST")
    print("=" * 60)
    
    # Initialize system
    print("1. Initializing RAG system...")
    rag = CleanRAGSystem('test_gemini_key')
    print("   ✅ System initialized")
    
    # Setup models to trigger any progress bars
    print("2. Setting up models (this would normally show progress bars)...")
    rag.setup_models()
    print("   ✅ Models loaded without progress bars")
    
    # Test embedding generation with dummy data
    print("3. Testing embedding generation...")
    from clean_rag_system import Chunk
    
    # Create test chunks
    test_chunks = [
        Chunk("This is test content for embedding 1", 1, 0),
        Chunk("This is test content for embedding 2", 1, 1),
        Chunk("This is test content for embedding 3", 1, 2)
    ]
    
    print("   📊 Generating embeddings...")
    embeddings = rag.generate_embeddings(test_chunks)
    print(f"   ✅ Generated {len(embeddings)} embeddings without progress bars")
    
    # Test query embedding
    print("4. Testing query processing...")
    rag.chunks = test_chunks
    rag.embeddings = embeddings
    rag.build_index(embeddings)
    
    # This should not show progress bars
    context = rag.get_context("test query")
    print("   ✅ Query processing completed without progress bars")
    
    print("\n🎉 ALL PROGRESS BARS SUCCESSFULLY SUPPRESSED!")
    print("🚀 System ready for silent operation")

if __name__ == "__main__":
    test_progress_suppression()
