import json
import logging
from typing import List, Dict, Tuple, Iterator
import os
from tqdm import tqdm
import pysqlite3
import sys
import pandas as pd
sys.modules["sqlite3"] = pysqlite3
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
hftoken = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

def get_document_name(file_path: str) -> str:
    return file_path.split('/')[-1]

def extract_query(full_query: str) -> str:
    return full_query.split('; ')[-1]

def check_answer_in_chunks(chunks: List[str], answers: List[str]) -> bool:
    if not chunks or not answers:
        return False
    
    for chunk in chunks:
        chunk_text = chunk.lower()
        for answer in answers:
            if answer.lower() in chunk_text:
                return True
    return False

def process_batch(collection, batch: List[Dict], k_values: List[int]) -> List[Dict]:
    """Process a batch of queries for all k values."""
    batch_results = []
    
    for test in tqdm(batch, desc="Processing queries in batch", leave=False):
        query = extract_query(test['query'])
        doc_path = test['snippets'][0]['file_path']
        doc_name = get_document_name(doc_path)
        answers = [snippet['answer'] for snippet in test['snippets']]
        
        test_results = {"query": query, "gold_answers": answers}
        
        for k in k_values:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=k,
                    where={"document": doc_name}
                )
                
                has_match = check_answer_in_chunks(results['documents'][0], answers)
                test_results[f"k_{k}_match"] = has_match
                test_results[f"k_{k}_num_retrieved"] = len(results['documents'][0])
                
            except Exception as e:
                logger.error(f"Error processing k={k} for query: {str(e)}")
                test_results[f"k_{k}_match"] = False
                test_results[f"k_{k}_num_retrieved"] = 0
        
        batch_results.append(test_results)
    
    return batch_results

def evaluate_dataset(collection, test_file_path: str, k_values: List[int], batch_size: int = 8) -> Dict:
    """Evaluate a dataset in batches for multiple k values."""
    logger.info(f"\nEvaluating dataset: {test_file_path}")
    
    with open(test_file_path, 'r') as f:
        test_data = json.load(f)
    
    metrics_by_k = {k: MetricsCalculator() for k in k_values}
    all_results = []
    
    test_batches = list(batch_queries(test_data['tests'], batch_size))
    
    for batch_idx, batch in enumerate(tqdm(test_batches, desc="Processing batches")):
        logger.info(f"Processing batch {batch_idx + 1}/{len(test_batches)}")
        batch_results = process_batch(collection, batch, k_values)
        
        for result in batch_results:
            for k in k_values:
                metrics_by_k[k].update(
                    has_match=result[f"k_{k}_match"],
                    num_retrieved=result[f"k_{k}_num_retrieved"],
                    num_answers=len(result["gold_answers"])
                )
        
        all_results.extend(batch_results)
        
        # Save intermediate results
        intermediate_results = {
            "metrics": {f"k_{k}": calc.calculate_metrics() for k, calc in metrics_by_k.items()},
            "evaluation_results": all_results
        }
        intermediate_file = f"intermediate_{os.path.basename(test_file_path)}"
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_results, f, indent=2)
        logger.info(f"Saved intermediate results to {intermediate_file}")
    
    return {
        "metrics": {f"k_{k}": calc.calculate_metrics() for k, calc in metrics_by_k.items()},
        "evaluation_results": all_results
    }

def generate_metrics_table(results: Dict[str, Dict], k_values: List[int], output_folder: str):
    """Generate CSV and pretty-printed tables with metrics."""
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

    # Calculate ALL rows
    for metric in metrics:
        metrics_data['Dataset'].append('ALL')
        metrics_data['Metric'].append(metric.lower())
        for k in k_values:
            values = [results[dataset]['metrics'][f'k_{k}'][f'{metric.lower()}_score' if metric == 'F1' else metric.lower()] 
                     for dataset in datasets]
            avg = round(sum(values) / len(values) * 100, 2)
            metrics_data[f'k={k}'].append(avg)

    # Save to CSV
    df = pd.DataFrame(metrics_data)
    df.to_csv(os.path.join(output_folder, 'metrics_table.csv'), index=False)

    # Generate pretty-printed text file
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
    client = PersistentClient(
        path="/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/baseline_benchmark_original_openai"
    )
    
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )
    
    collection = client.get_collection(
        name="documents",
        embedding_function=embedding_function
    )
    
    collection_stats = collection.get()
    logger.info(f"\nCollection stats:")
    logger.info(f"Number of documents: {len(collection_stats['ids'])}")
    
    datasets = ['cuad', 'maud', 'contractnli', 'privacy_qa']
    base_path = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks"
    output_folder = "evaluation_results_benchmark_openai_multi_k"
    os.makedirs(output_folder, exist_ok=True)
    
    k_values = [1, 2, 4, 8]
    all_results = {}

    for dataset in datasets:
        test_file = f"{base_path}/{dataset}.json"
        logger.info(f"\n📊 Evaluating {dataset}...")
        
        try:
            results = evaluate_dataset(collection, test_file, k_values, batch_size=8)
            all_results[dataset] = results
            
            output_file = os.path.join(output_folder, f"results_{dataset}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results for {dataset} to {output_file}")
                
        except Exception as e:
            logger.error(f"❌ Error evaluating {dataset}: {str(e)}")
            continue
    
    generate_metrics_table(all_results, k_values, output_folder)
    logger.info("✅ Evaluation complete!")

if __name__ == "__main__":
    main()