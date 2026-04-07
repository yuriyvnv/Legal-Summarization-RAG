import os
import json
import logging
from typing import Dict, List
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
import numpy as np

def check_collection_embeddings(collection_path: str):
    """Check embeddings dimensions for all collections in the database."""
    # Connect to the persistent client
    client = chromadb.PersistentClient(path=collection_path)
    
    # Get all collection names
    collection_names = client.list_collections()
    
    for name in collection_names:
        print(f"\nChecking collection: {name}")
        
        # Get the collection
        collection = client.get_collection(name=name)
        
        # Get a sample of items from the collection
        try:
            results = collection.get(
                limit=1,  # Get just one item to check dimensions
                include=['embeddings']
            )
            
            # Check if we have embeddings in the results
            if len(results['embeddings']) > 0:
                # Get the first embedding
                sample_embedding = results['embeddings'][0]
                
                # Print dimensions
                print(f"Embedding dimensions: {len(sample_embedding)}")
                
                # Print some basic stats
                print(f"Min value: {np.min(sample_embedding):.6f}")
                print(f"Max value: {np.max(sample_embedding):.6f}")
                print(f"Mean value: {np.mean(sample_embedding):.6f}")
                
                # Get total count of items
                total_count = collection.count()
                print(f"Total number of items in collection: {total_count}")
            else:
                print("No embeddings found in collection")
                
        except Exception as e:
            print(f"Error accessing collection: {str(e)}")

if __name__ == "__main__":
    # Use the same path as in your original script
    db_path = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/baseline_ours_bge_m3"
    check_collection_embeddings(db_path)