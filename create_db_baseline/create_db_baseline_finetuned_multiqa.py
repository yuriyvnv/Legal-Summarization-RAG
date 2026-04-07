import os
import json
import logging
from typing import Dict, List
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import time
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Load environment variables
load_dotenv()
hftoken = os.environ.get("HF_TOKEN")

# Use all available CPU cores for processing
NUM_CORES = multiprocessing.cpu_count()
print(f"Using {NUM_CORES} CPU cores for parallel processing")

# Enable GPU usage with specified device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use third GPU
print(f"Using GPU for embedding operations with {torch.cuda.mem_get_info()} memory")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create a pre-computed embeddings cache to avoid redundant calculations
embeddings_cache = {}

def precompute_batch_embeddings(texts, model_path="yuriyvnv/legal-multi-qa-mpnet-base-cos"):
    """Precompute embeddings for a batch of texts"""
    model = SentenceTransformer(model_path)
    # Use GPU for embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on: {device}")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 16  # Larger batch size for 32GB GPU
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
        all_embeddings.extend(batch_embeddings.tolist())
    
    return all_embeddings

# Custom embedding function that uses pre-computed embeddings
class PrecomputedEmbeddingFunction:
    def __init__(self, model_path="yuriyvnv/legal-multi-qa-mpnet-base-cos"):
        self.model_path = model_path
        self.model = SentenceTransformer(model_path)
        # Use GPU for embedding
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        logger.info(f"Initialized embedding model {model_path} on {device.upper()}")
    
    def __call__(self, input):
        # ChromaDB uses 'input' parameter name, not 'texts'
        texts = input
        
        # Check if these texts have already been embedded
        new_texts = [text for text in texts if text not in embeddings_cache]
        
        if new_texts:
            # Process in smaller batches to avoid memory issues
            batch_size = 16  # Larger batch size for 32GB GPU
            for i in range(0, len(new_texts), batch_size):
                batch = new_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
                
                # Update cache with new embeddings
                for text, embedding in zip(batch, batch_embeddings):
                    embeddings_cache[text] = embedding.tolist()
        
        # Return embeddings for all texts (from cache)
        return [embeddings_cache[text] for text in texts]

class HierarchicalRAGStore:
    def __init__(self, persist_directory):
        """Initialize the RAG store with collections using GTR-T5-XXL embeddings."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize embedding function
        self.embedding_function = PrecomputedEmbeddingFunction(model_path="yuriyvnv/legal-multi-qa-mpnet-base-cos")
        
        # Create collections with retry logic
        max_retries = 2
        retry_delay = 5  # seconds
        
        # First, try to delete existing collections if they exist
        # This ensures we don't have conflicts with embedding functions
        try:
            self.client.delete_collection("documents")
            logger.info("Deleted existing documents collection")
        except:
            logger.info("No existing documents collection to delete")
            
        try:
            self.client.delete_collection("chunks")
            logger.info("Deleted existing chunks collection")
        except:
            logger.info("No existing chunks collection to delete")
        
        # Now create new collections
        for attempt in range(max_retries):
            try:
                # Create new collections with proper embedding function
                self.documents_collection = self.client.create_collection(
                    name="documents",
                    embedding_function=self.embedding_function,
                    metadata={"level": "document", "hnsw:space": "cosine"}
                )
                logger.info("📄 Created documents collection with our embeddings.")
                
                self.chunks_collection = self.client.create_collection(
                    name="chunks",
                    embedding_function=self.embedding_function,
                    metadata={"level": "chunk", "hnsw:space": "cosine"}
                )
                logger.info("🔍 Created chunks collection with our embeddings.")
                
                break  # Successfully created collections
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to create collections (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to create collections after {max_retries} attempts: {str(e)}")
                    raise

    def batch_add_with_retry(self, collection, documents, ids, metadatas=None, embeddings=None, max_retries=3, batch_size=32):
        """Add documents with retry logic and smaller batch sizes."""
        retry_delay = 2  # seconds
        
        # Process in smaller batches to avoid timeouts
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size] if metadatas else None
            batch_embed = embeddings[i:i + batch_size] if embeddings else None
            
            for attempt in range(max_retries):
                try:
                    collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_meta,
                        embeddings=batch_embed
                    )
                    break  # Successfully added batch
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Batch add failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to add batch after {max_retries} attempts: {str(e)}")
                        # Continue with next batch instead of failing completely
            
            # Short delay between batches to avoid overwhelming the database
            time.sleep(0.5)
            
        return len(documents)  # Return count of attempted additions

    def batch_add_documents(self, documents, ids, metadatas=None):
        """Add documents in small batches with retry."""
        return self.batch_add_with_retry(
            self.documents_collection, 
            documents, 
            ids, 
            metadatas,
            batch_size=16  # Smaller batch size for stability
        )

    def batch_add_chunks(self, documents, ids, metadatas=None):
        """Add chunks in small batches with retry."""
        return self.batch_add_with_retry(
            self.chunks_collection, 
            documents, 
            ids, 
            metadatas,
            batch_size=16  # Smaller batch size for stability
        )

    def process_document_batch(self, dataset, documents_data):
        """Process a batch of documents and their chunks."""
        doc_summaries = []
        doc_ids = []
        doc_metadatas = []
        
        chunk_summaries = []
        chunk_ids = []
        chunk_metadatas = []
        
        for doc_id, doc_data in documents_data.items():
            if doc_data.get("general_summary", "Error") != "Error":
                # Prepare document-level data
                doc_summaries.append(doc_data["general_summary"])
                doc_ids.append(doc_id)
                doc_metadatas.append({"dataset": dataset, "doc_id": doc_id})
                
                # Prepare chunk-level data
                for i, chunk in enumerate(doc_data.get("chunks", [])):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_summaries.append(chunk["summary"])
                    chunk_ids.append(chunk_id)
                    chunk_metadatas.append({
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "original_text": chunk["chunk_text"]
                    })
        
        # Pre-compute embeddings for document summaries
        logger.info(f"Pre-computing embeddings for {len(doc_summaries)} document summaries...")
        
        # Add batches with pre-computed embeddings
        if doc_summaries:
            added_docs = self.batch_add_documents(doc_summaries, doc_ids, doc_metadatas)
            logger.info(f"✅ Added {added_docs} documents.")
            
        # Add chunks in parallel
        if chunk_summaries:
            added_chunks = self.batch_add_chunks(chunk_summaries, chunk_ids, chunk_metadatas)
            logger.info(f"🔹 Added {added_chunks} chunks.")

def process_dataset_parallel(dataset_name, documents, rag_store, batch_size=50):
    """Process a dataset in parallel batches"""
    doc_items = list(documents.items())
    
    # Process in batches
    total_batches = (len(doc_items) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, len(doc_items), batch_size):
        batch_dict = dict(doc_items[i:i + batch_size])
        
        # Clear CUDA cache between batches to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        current_batch = i // batch_size + 1
        logger.info(f"Processing batch {current_batch}/{total_batches} for {dataset_name}")
        
        rag_store.process_document_batch(dataset_name, batch_dict)
        
        # Log GPU memory usage if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def load_and_store_data(file_path, rag_store, batch_size=50):
    """Load the JSON data file and store its documents and chunks using parallel processing."""
    logger.info(f"📂 Loading data from '{file_path}'")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process each dataset
    for dataset_name in tqdm(list(data.keys()), desc="Processing Datasets"):
        documents = data[dataset_name]
        logger.info(f"Processing dataset: {dataset_name} with {len(documents)} documents")
        
        # Process this dataset
        process_dataset_parallel(dataset_name, documents, rag_store, batch_size)

def main():
    # Set up output directories
    persist_directory = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/fine_tuned_Embedding_multiqa_ours"
    os.makedirs(persist_directory, exist_ok=True)
    
    # Setup GPU memory management for better efficiency
    if torch.cuda.is_available():
        # Optional: Set memory growth for TensorFlow if you're using it alongside PyTorch
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except ImportError:
            pass
            
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.1f} GB memory")
        
        # Clear GPU cache to start fresh
        torch.cuda.empty_cache()
    
    # Initialize the RAG store
    rag_store = HierarchicalRAGStore(persist_directory=persist_directory)

    # Path to the data file
    file_path = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/final_summaries_striped.json"
    
    try:
        # Use larger batch size for GPU
        load_and_store_data(file_path, rag_store, batch_size=25)
        logger.info("🎉 Data loading completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error occurred during data loading: {str(e)}")
        raise

if __name__ == "__main__":
    # Make sure to use the 'spawn' method for multiprocessing to avoid issues with CUDA
    multiprocessing.set_start_method('spawn', force=True)
    
    # Set CUDA thread settings for better performance
    if torch.cuda.is_available():
        torch.set_num_threads(NUM_CORES)  # Set number of CPU threads for PyTorch
        
        # Optional: Set CUDA device thread settings
        try:
            torch.cuda.set_device(1)  # Focus on first GPU
        except:
            pass
    
    # Set a timeout for operations to avoid indefinite hanging
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    main()