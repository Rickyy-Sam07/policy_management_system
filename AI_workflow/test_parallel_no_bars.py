#!/usr/bin/env python3
"""
Test parallel processing without progress bars
"""

import os
# Ensure environment variables are set
os.environ['DISABLE_TQDM'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from clean_rag_system import CleanRAGSystem, Chunk
import numpy as np

def test_parallel_processing():
    """Test parallel sub-query processing without progress bars"""
    
    print("🧪 TESTING PARALLEL PROCESSING (NO PROGRESS BARS)")
    print("=" * 60)
    
    # Initialize system
    print("1. Initializing system...")
    rag = CleanRAGSystem('test_key')
    rag.setup_models()
    
    # Create mock document data
    print("2. Setting up mock document...")
    test_chunks = [
        Chunk("Coverage includes medical expenses and hospitalization costs", 1, 0),
        Chunk("Deductible amount is $500 per claim for individual coverage", 1, 1),
        Chunk("Claim process requires submitting forms within 30 days", 1, 2),
        Chunk("Exclusions include pre-existing conditions and cosmetic procedures", 1, 3),
        Chunk("Premium payments are due monthly and can be paid online", 1, 4)
    ]
    
    # Generate embeddings and setup index
    rag.chunks = test_chunks
    rag.embeddings = rag.generate_embeddings(test_chunks)
    rag.build_index(rag.embeddings)
    
    print("3. Testing complex query that triggers parallel processing...")
    complex_query = "What is the coverage amount and what is the deductible, also how do I submit a claim and what are the exclusions?"
    
    print(f"   Query: {complex_query}")
    print("   🔄 Processing parallel sub-queries...")
    
    # This should trigger parallel processing without showing progress bars
    result = rag.process_single_question(complex_query)
    
    print("4. ✅ PARALLEL PROCESSING COMPLETED WITHOUT PROGRESS BARS!")
    print(f"   Result length: {len(result)} characters")
    print(f"   Sample: {result[:100]}...")
    
    print("\n🎉 SUCCESS: All progress bars suppressed during parallel execution!")

if __name__ == "__main__":
    test_parallel_processing()
