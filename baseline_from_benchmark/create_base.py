import os
import json
import logging
from typing import Dict, List
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Dict, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
hftoken = os.getenv("HF_TOKEN")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BGEEmbeddingFunction:
    """Wrapper class to make LangChain's BGE embeddings compatible with ChromaDB."""
    def __init__(self, model_name="BAAI/bge-m3", device="cuda", token=None):
        encode_kwargs = {'normalize_embeddings': True}
        self.bge = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs=encode_kwargs,
        )
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB-compatible embedding function interface."""
        embeddings = self.bge.embed_documents(input)
        return embeddings


def batch_documents(documents: List[str], metadatas: List[Dict], ids: List[str], batch_size: int = 1000):
    """Batch documents to avoid request limits"""
    for i in range(0, len(documents), batch_size):
        yield (
            documents[i:i + batch_size],
            metadatas[i:i + batch_size],
            ids[i:i + batch_size]
        )

def embed_documents(json_file_path, collection_name="documents"):
    """
    Embeds document chunks into a Chroma collection with metadata.
    
    Args:
        json_file_path (str): Path to the JSON file containing documents
        collection_name (str): Name for the Chroma collection
    """
    logger.info("🚀 Starting embedding process...")

    # Initialize Chroma client
    client = PersistentClient(
        path="/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/baseline_benchmark_original_bge"
    )
    logger.info("✅ Connected to Chroma database")

    # Initialize OpenAI embedding function with proper configuration
    # embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key=openai_api_key,
    #     model_name="text-embedding-3-large",
    #     api_base="https://api.openai.com/v1",
    #     dimensions=3072  # Specify dimensions for text-embedding-3-large
    # )
    # embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     device="cuda",
    #     token=hftoken
    # )
    embedding_function = BGEEmbeddingFunction()
    logger.info("🤖 Initialized OpenAI embedding model")

    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"📚 Retrieved existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        logger.info(f"📚 Created new collection: {collection_name}")

    # Load and process JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    logger.info("📄 Loaded JSON data")

    total_embedded = 0
    for dataset_name, dataset_content in data.items():
        logger.info(f"📊 Processing dataset: {dataset_name}")
        
        for document_name, chunks in dataset_content.items():
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    chunk_id = f"{dataset_name}_{document_name}_{i}"
                    
                    metadata = {
                        "dataset": dataset_name,
                        "document": document_name.split('/')[-1] if '/' in document_name else document_name,
                        "chunk_index": i
                    }
                    
                    documents.append(chunk)
                    metadatas.append(metadata)
                    ids.append(chunk_id)
            
            # Process in batches
            if documents:
                total_chunks = len(documents)
                with tqdm(total=total_chunks, desc=f"Embedding chunks for {document_name}", unit="chunk") as pbar:
                    for batch_docs, batch_meta, batch_ids in batch_documents(documents, metadatas, ids):
                        try:
                            collection.add(
                                documents=batch_docs,
                                metadatas=batch_meta,
                                ids=batch_ids
                            )
                            total_embedded += len(batch_docs)
                            pbar.update(len(batch_docs))
                            
                            # Add a small delay between batches to avoid rate limits
                            time.sleep(0.5)
                            
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
                            logger.error(f"Batch size: {len(batch_docs)}")
                            continue
                
                logger.info(f"💫 Added {total_chunks} chunks from {document_name}")

    logger.info(f"✨ Embedding complete! Total chunks embedded: {total_embedded}")

def main():
    json_file_path = "/home/yperezhohin/Legal-Summarization-RAG/data/original_chunked/processed_documents_benchmark.json"
    
    try:
        embed_documents(json_file_path)
        logger.info("🎉 Process completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()