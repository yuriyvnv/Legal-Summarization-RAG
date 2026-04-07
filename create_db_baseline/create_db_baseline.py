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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
from sentence_transformers import SentenceTransformer


load_dotenv()
hftoken = os.environ["HF_TOKEN"]

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
#model = SentenceTransformer("/home/yperezhohin/legal_embeddings/models/legal_embedding_model/model")

# class BGEEmbeddingFunction:
#     """Wrapper class to make LangChain's BGE embeddings compatible with ChromaDB."""
#     def __init__(self, model_name="sentence-transformers/gtr-t5-xxl", device="cuda", token=None):
#         encode_kwargs = {'normalize_embeddings': True}
#         self.bge = HuggingFaceBgeEmbeddings(
#             model_name=model_name,
#             model_kwargs={'device': device},
#             encode_kwargs=encode_kwargs,
#         )
    
#     def __call__(self, input: List[str]) -> List[List[float]]:
#         """ChromaDB-compatible embedding function interface."""
#         embeddings = self.bge.embed_documents(input)
#         return embeddings

class HierarchicalRAGStore:
    def __init__(self, persist_directory: str = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/fine_tuned_Embedding_multiqa_ours"):
        """Initialize the RAG store with two collections using our embeddings."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize BGE embedding function with wrapper
        
        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=hftoken,
            model_name="yuriyvnv/legal-multi-qa-mpnet-base-cos"
                )
        
        # Create collections
        self.documents_collection = self.client.create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={"level": "document", "hnsw:space": "cosine"}
        )
        logger.info("📄 Created documents collection with our model embeddings.")

        self.chunks_collection = self.client.create_collection(
            name="chunks",
            embedding_function=self.embedding_function,
            metadata={"level": "chunk", "hnsw:space": "cosine"}
        )
        logger.info("🔍 Created chunks collection with OUR embeddings.")

    def batch_add_documents(self, documents: List[str], ids: List[str], metadatas: List[Dict], batch_size: int = 20):
        """Add documents in batches."""
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            try:
                self.documents_collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_meta
                )
                logger.info(f"✅ Added batch of {len(batch_docs)} documents.")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"❌ Error adding document batch: {str(e)}")
                continue

    def batch_add_chunks(self, documents: List[str], ids: List[str], metadatas: List[Dict], batch_size: int = 20):
        """Add chunks in batches."""
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            try:
                self.chunks_collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_meta
                )
                logger.info(f"🔹 Added batch of {len(batch_docs)} chunks.")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"❌ Error adding chunk batch: {str(e)}")
                continue

    def process_document_batch(self, dataset: str, documents_data: Dict[str, Dict]):
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
        
        # Add batches
        if doc_summaries:
            self.batch_add_documents(doc_summaries, doc_ids, doc_metadatas)
        if chunk_summaries:
            self.batch_add_chunks(chunk_summaries, chunk_ids, chunk_metadatas)

def load_and_store_data(file_path: str, rag_store: HierarchicalRAGStore, batch_size: int = 20):
    """Load the JSON data file and store its documents and chunks in batches."""
    logger.info(f"📂 Loading data from '{file_path}'")
    with open(file_path, 'r') as f:
        data = json.load(f)

    for dataset in tqdm(list(data.keys()), desc="Processing Datasets", unit="dataset"):
        documents = data[dataset]
        current_batch = {}
        
        for doc_id, doc_data in tqdm(documents.items(), desc=f"Processing {dataset}", unit="doc"):
            current_batch[doc_id] = doc_data
            
            if len(current_batch) >= batch_size:
                rag_store.process_document_batch(dataset, current_batch)
                current_batch = {}
        
        # Process remaining documents
        if current_batch:
            rag_store.process_document_batch(dataset, current_batch)

def main():
    # Initialize the RAG store with BGE embeddings
    rag_store = HierarchicalRAGStore(
        persist_directory="/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/fine_tuned_Embedding_multiqa_ours"
    )

    file_path = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/final_summaries_striped.json"
    
    try:
        load_and_store_data(file_path, rag_store)
        logger.info("🎉 Data loading completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error occurred during data loading: {str(e)}")
        raise

if __name__ == "__main__":
    main()