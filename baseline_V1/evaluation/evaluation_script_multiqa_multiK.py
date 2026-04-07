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
sys.modules["sqlite3"] = pysqlite3
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
hftoken = os.getenv("HF_TOKEN")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def batch_queries(queries: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """Split queries into batches."""
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

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

def query_vector_db(query_text: str, client: PersistentClient, embedding_function, n_docs: int = 1, n_chunks: int = 1):
    parts = query_text.split(";", 1)
    if len(parts) == 2:
        doc_query = parts[0].strip()
        chunk_query = parts[1].strip()
    else:
        doc_query = query_text.strip()
        chunk_query = query_text.strip()
    
    documents_collection = client.get_collection(
        name="documents",
        embedding_function=embedding_function
    )
    chunks_collection = client.get_collection(
        name="chunks",
        embedding_function=embedding_function
    )
    
    doc_results = documents_collection.query(
        query_texts=[doc_query],
        n_results=n_docs
    )
    
    if not doc_results['ids'][0]:
        return []
    
    relevant_chunks = []
    for doc_id, doc_score in zip(doc_results['ids'][0], doc_results['distances'][0]):
        chunk_results = chunks_collection.query(
            query_texts=[chunk_query],
            n_results=n_chunks,
            where={"doc_id": doc_id}
        )
        
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

def check_answer_in_chunks(chunks: List[Dict], answers: List[str]) -> bool:
    if not chunks or not answers:
        return False
    
    for chunk in chunks:
        chunk_text = chunk["text"].lower()
        for answer in answers:
            if answer.lower() in chunk_text:
                return True
    return False

def process_batch(batch: List[Dict], client: PersistentClient, embedding_function, k_values: List[int]) -> List[Dict]:
    """Process a batch of queries for all k values."""
    batch_results = []
    
    query_pbar = tqdm(batch, desc="Queries in batch", position=2, leave=False)
    for test in query_pbar:
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
                test_results[f"k_{k}_match"] = False
                test_results[f"k_{k}_num_retrieved"] = 0
        
        batch_results.append(test_results)
    
    query_pbar.close()
    return batch_results

def evaluate_dataset(eval_file_path: str, k_values: List[int], batch_size: int = 10000, num_workers: int = 24) -> Dict:
    """Evaluate a dataset using parallel batch processing with nested progress bars."""
    dataset_name = os.path.basename(eval_file_path).split('.')[0]
    logger.info(f"Evaluating dataset: {dataset_name}")
    
    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)
    
    metrics_by_k = {k: MetricsCalculator() for k in k_values}
    all_results = []
    
    client = PersistentClient(path="/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/baseline_ours_multiqa")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        device="cuda",
        token=hftoken
    )
    
    tests = eval_data.get("tests", [])
    total_tests = len(tests)
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
                batch_results = future.result()
                
                for result in batch_results:
                    for k in k_values:
                        metrics_by_k[k].update(
                            has_match=result[f"k_{k}_has_match"],
                            num_retrieved=result[f"k_{k}_num_retrieved"],
                            num_answers=len(result["gold_answers"])
                        )
                
                all_results.extend(batch_results)
                
                batch_pbar.update(1)
                dataset_pbar.update(len(batch_results))
                
                intermediate_results = {
                    "metrics": {f"k_{k}": calc.calculate_metrics() for k, calc in metrics_by_k.items()},
                    "evaluation_results": all_results
                }
                with open(f"intermediate_{dataset_name}.json", 'w') as f:
                    json.dump(intermediate_results, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
    
    batch_pbar.close()
    dataset_pbar.close()
    
    return {
        "metrics": {f"k_{k}": calc.calculate_metrics() for k, calc in metrics_by_k.items()},
        "evaluation_results": all_results
    }

def generate_metrics_table(results: Dict[str, Dict], k_values: List[int], output_folder: str):
    metrics_data = {
        'Dataset': [],
        'Metric': []
    }
    for k in k_values:
        metrics_data[f'k={k}'] = []

    metrics = ['Precision', 'Recall', 'Accuracy', 'F1']
    datasets = list(results.keys())

    for dataset in datasets:
        for metric in metrics:
            metrics_data['Dataset'].append(dataset)
            metrics_data['Metric'].append(metric.lower())
            
            for k in k_values:
                value = results[dataset]['metrics'][f'k_{k}'][f'{metric.lower()}_score' if metric == 'F1' else metric.lower()]
                metrics_data[f'k={k}'].append(round(value * 100, 2))

    for metric in metrics:
        metrics_data['Dataset'].append('ALL')
        metrics_data['Metric'].append(metric.lower())
        for k in k_values:
            values = [results[dataset]['metrics'][f'k_{k}'][f'{metric.lower()}_score' if metric == 'F1' else metric.lower()] 
                     for dataset in datasets]
            avg = round(sum(values) / len(values) * 100, 2)
            metrics_data[f'k={k}'].append(avg)

    df = pd.DataFrame(metrics_data)
    df.to_csv(os.path.join(output_folder, 'metrics_table.csv'), index=False)

    with open(os.path.join(output_folder, 'metrics_table_pretty.txt'), 'w') as f:
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
            
            metric_data = df[df['Metric'] == metric.lower()]
            for _, row in metric_data.iterrows():
                dataset = row['Dataset']
                f.write(dataset.ljust(15))
                for k in k_values:
                    value = row[f'k={k}']
                    f.write(f"{value:>12.2f}")
                f.write("\n")
            f.write("\n")

def main():
    eval_files = ["maud.json", "contractnli.json", "cuad.json", "privacy_qa.json"]
    eval_folder = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks"
    output_folder = "evaluation_results_multiqa_multiK"
    os.makedirs(output_folder, exist_ok=True)

    k_values = [1, 2, 4, 8]
    all_results = {}

    overall_pbar = tqdm(eval_files, desc="Overall Progress", position=0)
    
    print("\n" * 3)  # Add space for progress bars
    
    for filename in overall_pbar:
        dataset_name = filename.split('.')[0]
        eval_file_path = os.path.join(eval_folder, filename)
        
        try:
            overall_pbar.set_description(f"Processing {dataset_name}")
            
            results = evaluate_dataset(
                eval_file_path, 
                k_values, 
                batch_size=10000,
                num_workers=18
            )
            all_results[dataset_name] = results
            
            output_file = os.path.join(output_folder, f"results_{filename}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            metrics = results['metrics']
            overall_pbar.write(f"\nResults for {dataset_name}:")
            for k in k_values:
                precision = metrics[f'k_{k}']['precision'] * 100
                recall = metrics[f'k_{k}']['recall'] * 100
                f1 = metrics[f'k_{k}']['f1_score'] * 100
                overall_pbar.write(f"k={k}: Precision={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%")
                
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {str(e)}")
            continue
    
    overall_pbar.close()
    
    generate_metrics_table(all_results, k_values, output_folder)
    logger.info("✨ Evaluation complete!")

if __name__ == "__main__":
    main()