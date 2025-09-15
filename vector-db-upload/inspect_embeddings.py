#!/usr/bin/env python3
import numpy as np
import faiss
import argparse
import sys
import os


def inspect_embedding_dtype(index_path: str, sample_size: int = 10):
    """
    Inspect the data type of embeddings in a FAISS index.
    
    Args:
        index_path: Path to the FAISS index file
        sample_size: Number of vectors to sample for inspection
    """
    if not os.path.exists(index_path):
        print(f"❌ Index file not found: {index_path}")
        return False
    
    try:
        # Load FAISS index
        print(f"🔍 Loading FAISS index: {index_path}")
        index = faiss.read_index(index_path)
        
        # Get base index (unwrap IDMap if present)
        base = index.index if hasattr(index, "index") else index
        ntotal, dim = index.ntotal, index.d
        
        print(f"📊 Index stats: {ntotal} vectors × {dim} dimensions")
        
        if ntotal == 0:
            print("⚠️  Index is empty")
            return True
        
        # Sample vectors to inspect their data type
        sample_count = min(sample_size, ntotal)
        print(f"🔬 Sampling {sample_count} vectors for dtype inspection...")
        
        # Reconstruct sample vectors
        V = base.reconstruct_n(0, sample_count)
        if not isinstance(V, np.ndarray):
            V = np.array(V)
        
        # Inspect data type
        print(f"📋 Vector array dtype: {V.dtype}")
        print(f"📋 Vector array shape: {V.shape}")
        
        # Check precision
        if V.dtype == np.float16:
            print("✅ Embeddings are stored as float16 (half precision)")
        elif V.dtype == np.float32:
            print("✅ Embeddings are stored as float32 (single precision)")  
        elif V.dtype == np.float64:
            print("✅ Embeddings are stored as float64 (double precision)")
        elif V.dtype == np.uint8:
            print("✅ Embeddings are stored as uint8 (quantized)")
        else:
            print(f"ℹ️  Embeddings are stored as: {V.dtype}")
        
        # Show sample values for first vector
        print(f"📊 Sample values from first vector (first 10 elements):")
        print(f"   {V[0][:10]}")
        
        # Memory usage estimation
        bytes_per_element = V.dtype.itemsize
        total_memory_mb = (ntotal * dim * bytes_per_element) / (1024 * 1024)
        print(f"💾 Estimated memory usage: {total_memory_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting embeddings: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect embedding data types in FAISS index")
    parser.add_argument("index_path", help="Path to FAISS index file")
    parser.add_argument("--sample-size", type=int, default=10, 
                       help="Number of vectors to sample (default: 10)")
    
    args = parser.parse_args()
    
    success = inspect_embedding_dtype(args.index_path, args.sample_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()