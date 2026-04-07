#!/usr/bin/env python3
import os
import json
import logging
import csv
from tqdm import tqdm
import sys
import pysqlite3
import pandas as pd
from typing import List, Dict, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sentence_transformers import SentenceTransformer

# Use pysqlite3 for sqlite3
sys.modules["sqlite3"] = pysqlite3

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables from .env (optional if you use HF token).
load_dotenv()
hftoken = os.getenv("HF_TOKEN")

# Enable GPU (adjust as necessary; can be "0" if you have a single GPU).
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the embedding model globally
model = None

def initialize_model():
    """
    Initialize and return a global model (SentenceTransformer).
    This will load onto GPU if available.
    """
    global model
    if model is None:
        # Adjust your chosen model here:
        model = SentenceTransformer("sentence-transformers/gtr-t5-xxl")
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Model loaded on CPU")
    return model

class OptimizedEmbeddingFunction:
    def __init__(self, batch_size=124, max_cache_size=10000):
        """
        Args:
            batch_size (int): The batch size for embedding.
            max_cache_size (int): Maximum number of unique texts to hold in cache.
        """
        self.model = initialize_model()
        self.batch_size = batch_size
        self.max_cache_size = max_cache_size
        self.cache = {}

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Chroma now mandates this exact signature: 
            __call__(self, input: List[str]) -> List[List[float]]
        """
        embeddings = [None] * len(input)
        new_texts = []
        new_indices = []

        # Identify which texts are new (not in cache)
        for i, text in enumerate(input):
            if text in self.cache:
                embeddings[i] = self.cache[text]
            else:
                new_texts.append(text)
                new_indices.append(i)

        # Process new texts in batches, if any
        if new_texts:
            for start_idx in range(0, len(new_texts), self.batch_size):
                batch = new_texts[start_idx:start_idx + self.batch_size]
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch,
                        normalize_embeddings=True,
                        batch_size=self.batch_size,
                        show_progress_bar=False
                    )
                # Update cache
                for text_item, embedding in zip(batch, batch_embeddings):
                    # Evict from cache if we exceed max size
                    if len(self.cache) >= self.max_cache_size:
                        self.cache.pop(next(iter(self.cache)))  # simple FIFO
                    self.cache[text_item] = embedding.tolist()

            # Insert new embeddings back into the results
            for idx, text_item in zip(new_indices, new_texts):
                embeddings[idx] = self.cache[text_item]

        return embeddings

def query_vector_db(query_text: str, client: PersistentClient, embedding_function, n_docs: int = 1, n_chunks: int = 1):
    try:
        # Split the query if it contains a separator
        parts = query_text.split(";", 1)
        doc_query = parts[0].strip()
        chunk_query = parts[1].strip() if len(parts) > 1 else doc_query
        
        # Get collections with the optimized embedding function
        documents_collection = client.get_collection(
            name="documents",
            embedding_function=embedding_function
        )
        chunks_collection = client.get_collection(
            name="chunks",
            embedding_function=embedding_function
        )
        
        # Query documents with retry logic
        max_retries = 3
        doc_results = None
        for attempt in range(max_retries):
            try:
                doc_results = documents_collection.query(
                    query_texts=[doc_query],
                    n_results=n_docs
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to query documents after {max_retries} attempts: {str(e)}")
                    return []
                logger.warning(f"Retrying document query (attempt {attempt + 1})")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache
        
        if not doc_results or not doc_results['ids'][0]:
            return []
        
        relevant_chunks = []
        for doc_id, doc_score in zip(doc_results['ids'][0], doc_results['distances'][0]):
            # Query chunks with retry logic
            chunk_results = None
            for attempt in range(max_retries):
                try:
                    chunk_results = chunks_collection.query(
                        query_texts=[chunk_query],
                        n_results=n_chunks,
                        where={"doc_id": doc_id}
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to query chunks after {max_retries} attempts: {str(e)}")
                        continue
                    logger.warning(f"Retrying chunk query (attempt {attempt + 1})")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear GPU cache
            
            if not chunk_results or not chunk_results['ids'][0]:
                continue
            
            for chunk_id, chunk_score, chunk_metadata in zip(
                chunk_results['ids'][0],
                chunk_results['distances'][0],
                chunk_results['metadatas'][0]
            ):
                relevant_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "doc_score": doc_score,
                    "chunk_score": chunk_score,
                    "text": chunk_metadata.get("original_text", ""),
                    "dataset": chunk_metadata.get("dataset", "")
                })
        
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error in query_vector_db: {str(e)}")
        return []

def check_answer_in_chunks(chunks: List[Dict], answers: List[str]) -> bool:
    """
    Simply check if any answer appears in any of the retrieved chunks.
    Args:
        chunks: List of retrieved chunks
        answers: List of possible correct answers
    Returns:
        bool: True if any answer is found in any chunk, False otherwise
    """
    if not chunks or not answers:
        return False
    
    for chunk in chunks:
        chunk_text = chunk["text"].lower()
        for answer in answers:
            if answer.lower() in chunk_text:
                return True
    return False

class MetricsCalculator:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_queries = 0
    
    def update(self, has_match: bool, num_retrieved: int, num_answers: int):
        """
        Update metrics for a single query.
        
        Args:
            has_match: Whether any of the retrieved chunks contained any of the correct answers
            num_retrieved: Number of chunks retrieved
            num_answers: Number of possible correct answers for this query
        """
        self.total_queries += 1
        
        if has_match:
            # We found at least one correct answer
            self.true_positives += 1
            # Count extra retrievals (beyond number of possible answers) as false positives
            if num_retrieved > num_answers:
                self.false_positives += (num_retrieved - num_answers)
        else:
            # No correct answers found
            self.false_positives += num_retrieved  # All retrievals were wrong
            self.false_negatives += num_answers    # Missed all possible correct answers
    
    def calculate_metrics(self) -> Dict[str, float]:
        denominator_precision = self.true_positives + self.false_positives
        denominator_recall = self.true_positives + self.false_negatives
        
        precision = self.true_positives / denominator_precision if denominator_precision > 0 else 0
        recall = self.true_positives / denominator_recall if denominator_recall > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = self.true_positives / self.total_queries if self.total_queries > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_queries": self.total_queries
        }

def process_batch(batch: List[Dict], client: PersistentClient, embedding_function, k_values: List[int]) -> List[Dict]:
    """Process a batch of queries."""
    batch_results = []
    
    query_pbar = tqdm(batch, desc="Queries in batch", position=2, leave=False)
    for test in query_pbar:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        query = test.get("query", "")
        snippets = test.get("snippets", [])
        gold_answers = [snippet.get("answer", "") for snippet in snippets]
        
        test_results = {"query": query, "gold_answers": gold_answers}
        
        for k in k_values:
            try:
                rag_output = query_vector_db(
                    query, 
                    client=client,
                    embedding_function=embedding_function,
                    n_docs=1, 
                    n_chunks=k
                )
                has_match = check_answer_in_chunks(rag_output, gold_answers)
                
                test_results[f"k_{k}_output"] = rag_output
                test_results[f"k_{k}_has_match"] = has_match
                test_results[f"k_{k}_num_retrieved"] = len(rag_output)
                
            except Exception as e:
                logger.error(f"Error processing k={k} for query: {str(e)}")
                test_results[f"k_{k}_has_match"] = False
                test_results[f"k_{k}_num_retrieved"] = 0
        
        batch_results.append(test_results)
    
    query_pbar.close()
    return batch_results

def safe_save_json(data: Dict, filepath: str, temp_suffix: str = ".tmp") -> bool:
    """Safely save JSON data with atomic write operation."""
    temp_path = filepath + temp_suffix
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, filepath)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def batch_queries(queries: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """Split queries into batches."""
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

def evaluate_dataset(eval_file_path: str, k_values: List[int], batch_size: int = 124, num_workers: int = 3, query_timeout: int = 30) -> Dict:
    """
    Evaluate a single dataset JSON file.

    Returns:
        Dict with "metrics" and "evaluation_results", or None if there's an error.
    """
    dataset_name = os.path.basename(eval_file_path).split('.')[0]
    logger.info(f"Evaluating dataset: {dataset_name}")
    
    # Read input file
    try:
        with open(eval_file_path, 'r') as f:
            eval_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading evaluation file {eval_file_path}: {str(e)}")
        return None
    
    # Initialize metrics
    metrics_by_k = {k: MetricsCalculator() for k in k_values}
    all_results = []
    
    # Initialize client and embedding function
    try:
        client = PersistentClient(
            path="/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/baseline_ours_gtr5_xxl"
        )
        # We now pass the max_cache_size param properly
        embedding_function = OptimizedEmbeddingFunction(batch_size=124, max_cache_size=10000)
    except Exception as e:
        logger.error(f"Error initializing client or embedding function: {str(e)}")
        return None
    
    tests = eval_data.get("tests", [])
    total_tests = len(tests)
    if total_tests == 0:
        logger.warning(f"No tests found in {eval_file_path}.")
        return None
    
    batches = list(batch_queries(tests, batch_size))
    
    dataset_pbar = tqdm(total=total_tests, desc=f"Dataset: {dataset_name}", position=0)
    batch_pbar = tqdm(total=len(batches), desc="Batches", position=1, leave=False)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_batch, 
                batch, 
                client, 
                embedding_function,
                k_values
            ): i for i, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result(timeout=query_timeout)
                all_results.extend(batch_results)
                
                # Update metrics
                for result in batch_results:
                    for k in k_values:
                        metrics_by_k[k].update(
                            has_match=result[f"k_{k}_has_match"],
                            num_retrieved=result[f"k_{k}_num_retrieved"],
                            num_answers=len(result["gold_answers"])
                        )
                
                batch_pbar.update(1)
                dataset_pbar.update(len(batch_results))
                
                # Save intermediate results periodically
                if batch_idx % 5 == 0:  # e.g., every 5 batches
                    intermediate_results = {
                        "metrics": {
                            f"k_{k}": calc.calculate_metrics() 
                            for k, calc in metrics_by_k.items()
                        },
                        "evaluation_results": all_results
                    }
                    safe_save_json(intermediate_results, f"intermediate_{dataset_name}.json")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue  # skip this batch
    
    batch_pbar.close()
    dataset_pbar.close()
    
    # Final results
    return {
        "metrics": {
            f"k_{k}": calc.calculate_metrics() for k, calc in metrics_by_k.items()
        },
        "evaluation_results": all_results
    }

def generate_metrics_table(results: Dict[str, Dict], k_values: List[int], output_folder: str):
    """
    Generates a CSV and pretty TXT table of metrics for all datasets.
    """
    metrics_data = {
        'Dataset': [],
        'Metric': []
    }
    for k in k_values:
        metrics_data[f'k={k}'] = []

    metrics = ['Precision', 'Recall', 'Accuracy', 'F1']
    datasets = list(results.keys())

    # Collect data
    for dataset in datasets:
        dataset_result = results[dataset]
        # If dataset_result is None, skip it
        if not dataset_result or 'metrics' not in dataset_result:
            continue

        for metric in metrics:
            metrics_data['Dataset'].append(dataset)
            metrics_data['Metric'].append(metric.lower())
            
            for k in k_values:
                # Map "F1" -> "f1_score"
                key = 'f1_score' if metric == 'F1' else metric.lower()
                value = dataset_result['metrics'][f'k_{k}'][key]
                metrics_data[f'k={k}'].append(round(value * 100, 2))

    # Compute overall average across valid datasets
    valid_datasets = [
        ds for ds in datasets 
        if results[ds] is not None and 'metrics' in results[ds]
    ]
    if valid_datasets:
        for metric in metrics:
            metrics_data['Dataset'].append('ALL')
            metrics_data['Metric'].append(metric.lower())
            for k in k_values:
                key = 'f1_score' if metric == 'F1' else metric.lower()
                # Only average over datasets that produced metrics
                values = [
                    results[ds]['metrics'][f'k_{k}'][key] 
                    for ds in valid_datasets
                ]
                if values:
                    avg = round(sum(values) / len(values) * 100, 2)
                else:
                    avg = 0
                metrics_data[f'k={k}'].append(avg)

    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(output_folder, 'metrics_table.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Metrics table saved to {csv_path}")

    pretty_path = os.path.join(output_folder, 'metrics_table_pretty.txt')
    with open(pretty_path, 'w') as f:
        f.write("LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain\n")
        f.write("=" * 80 + "\n\n")

        for metric in metrics:
            f.write(f"\n{metric} @ k\n")
            f.write("-" * 80 + "\n")
            
            f.write("Dataset".ljust(15))
            for k in k_values:
                f.write(f"k={k}".rjust(12))
            f.write("\n")
            f.write("-" * 80 + "\n")
            
            # Filter the dataframe rows for the current metric
            metric_data = df[df['Metric'] == metric.lower()]
            for _, row in metric_data.iterrows():
                dataset = row['Dataset']
                f.write(dataset.ljust(15))
                for k in k_values:
                    value = row[f'k={k}']
                    f.write(f"{value:>12.2f}")
                f.write("\n")
            f.write("\n")

    logger.info(f"Pretty metrics table saved to {pretty_path}")

def main():
    # List your evaluation files here
    eval_files = [
        "maud.json", 
        "contractnli.json", 
        "cuad.json", 
        "privacy_qa.json"
    ]
    eval_folder = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks"
    output_folder = "evaluation_results_gtr5_xxl_multiK"
    os.makedirs(output_folder, exist_ok=True)

    # You can tune these values
    k_values = [1, 2, 4, 8]
    batch_size = 124  # Increase to utilize GPU more (but watch for OOM)
    num_workers = 3 # Reduce if concurrency overhead is high; or set to 1
    
    logger.info(f"Using {num_workers} workers for parallel processing.")
    all_results = {}

    overall_pbar = tqdm(eval_files, desc="Overall Progress", position=0)
    
    # Extra spacing for multiple progress bars in console
    print("\n" * 2)
    
    for filename in overall_pbar:
        dataset_name = filename.split('.')[0]
        eval_file_path = os.path.join(eval_folder, filename)
        
        try:
            overall_pbar.set_description(f"Processing {dataset_name}")
            
            results = evaluate_dataset(
                eval_file_path, 
                k_values, 
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            # If results is None, skip
            if results is None:
                logger.error(f"Skipping dataset {dataset_name} due to initialization/reading error.")
                continue
            
            all_results[dataset_name] = results
            
            # Save individual dataset results
            output_file = os.path.join(output_folder, f"results_{filename}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            metrics = results['metrics']
            overall_pbar.write(f"\nResults for {dataset_name}:")
            for k in k_values:
                precision = metrics[f'k_{k}']['precision'] * 100
                recall = metrics[f'k_{k}']['recall'] * 100
                f1 = metrics[f'k_{k}']['f1_score'] * 100
                overall_pbar.write(f"k={k}: Precision={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%")
                
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {str(e)}")
            # You may optionally store None or skip
            all_results[dataset_name] = None
            continue
    
    overall_pbar.close()
    
    # Generate final metrics table
    generate_metrics_table(all_results, k_values, output_folder)
    logger.info("✨ Evaluation complete!")

if __name__ == "__main__":
    main()
