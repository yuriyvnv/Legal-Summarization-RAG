"""
Multi-threaded Analysis Script for Evaluating Text Summaries

This script analyzes evaluation results for text summaries across multiple datasets.
It identifies summaries that received low scores (below 0.6) and collects detailed
information about these cases, including the original text, the summary, and the
evaluation reasoning.

The script uses multiprocessing to parallelize the analysis across CPU cores,
making it efficient for large datasets. It handles various formats of evaluation
responses and includes robust error handling.

Main components:
- SummaryAnalyzer: Main class that orchestrates the analysis
- Multiprocessing: Parallel processing of documents
- Score extraction: Handles multiple score formats
- Progress tracking: Visual feedback during processing
- Result aggregation: Combines results from parallel processes

"""

import json
import os
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

class SummaryAnalyzer:
    """
    Main class for analyzing summary evaluation results.
    
    This class handles the loading, processing, and analysis of summary evaluation
    data across multiple datasets. It supports parallel processing of documents
    and provides detailed reporting of low-scoring summaries.
    
    Attributes:
        base_path (Path): Root path of the project
        results_path (Path): Path to evaluation results
        summaries_path (Path): Path to original summaries
    """

    def __init__(self, base_path: str):
        """
        Initialize the analyzer with project paths.
        
        Args:
            base_path (str): Root directory of the project containing all required files
        """
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "utils" / "quality_evaluation" / "results"
        self.summaries_path = self.base_path

    def extract_score(self, text: str) -> float:
        """
        Extract a numerical score from evaluation text.
        
        Handles multiple score formats:
        - "score: 0.8"
        - "score = 0.8"
        - "score=0.8"
        
        Args:
            text (str): The evaluation text containing a score
            
        Returns:
            float or None: The extracted score if found and valid, None otherwise
        """
        all_scores = []
        
        def find_scores_with_pattern(text, pattern):
            """Helper function to find scores using a specific pattern."""
            scores = []
            for line in text.lower().split('\n'):
                if pattern in line:
                    try:
                        score_text = line.split(pattern)[1].strip()
                        if score_text:  # Only process non-empty strings
                            score = float(score_text)
                            if 0.0 <= score <= 1.0:  # Validate score range
                                scores.append(score)
                    except (ValueError, IndexError):
                        continue
            return scores
        
        # Try different patterns to find scores
        patterns = ['score:', 'score =', 'score=']
        for pattern in patterns:
            all_scores.extend(find_scores_with_pattern(text, pattern))
        
        # Return the last valid score found (most recent)
        return all_scores[-1] if all_scores else None

    def process_document(self, args: Tuple[str, str, Path, Path]) -> Dict:
        """
        Process a single document's evaluation results.
        
        This function is designed to run in parallel. It processes all evaluations
        for one document and identifies low-scoring summaries.
        
        Args:
            args: Tuple containing (dataset_name, doc_name, eval_file_path, summary_file_path)
            
        Returns:
            Dict containing processing results or error information
        """
        dataset_name, doc_name, eval_file, summary_file = args
        
        try:
            # Read evaluation data
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            low_scores = []
            total_summaries = 0
            
            doc_content = eval_data[dataset_name]['documents'].get(doc_name)
            if not doc_content:
                return {'error': f'Document {doc_name} not found in evaluation data'}
            
            # Process each evaluation in the document
            for eval_item in doc_content['evaluations']:
                total_summaries += 1
                chunk_id = eval_item['chunk_index']
                evaluation_text = eval_item['evaluation']
                
                # Extract score using different methods
                score = None
                if 'END_OF_RESPONSE' in evaluation_text:
                    # If END_OF_RESPONSE is present, try parts after it first
                    for part in reversed(evaluation_text.split('END_OF_RESPONSE')):
                        if part.strip():
                            score = self.extract_score(part)
                            if score is not None:
                                break
                
                # If no score found after END_OF_RESPONSE, try whole text
                if score is None:
                    score = self.extract_score(evaluation_text)
                
                # Process low scores
                if score is not None and score < 0.6:
                    try:
                        # Load and extract relevant summary data
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary_data = json.load(f)
                        
                        chunk_data = summary_data[dataset_name][doc_name][chunk_id]
                        summary = chunk_data.get('summary', 'Summary not found')
                        chunk_text = chunk_data.get('chunk_text', 'Original text not found')
                        
                        low_scores.append({
                            'dataset': dataset_name,
                            'document': doc_name,
                            'chunk_id': chunk_id,
                            'score': score,
                            'summary': summary,
                            'evaluation': evaluation_text,
                            'original_text': chunk_text,
                            'error': False
                        })
                    except Exception as e:
                        print(f"Error processing document {doc_name}, chunk {chunk_id}: {e}")
                        low_scores.append({
                            'dataset': dataset_name,
                            'document': doc_name,
                            'chunk_id': chunk_id,
                            'score': score,
                            'summary': 'Error retrieving summary',
                            'evaluation': evaluation_text,
                            'original_text': 'Error retrieving original text',
                            'error': True,
                            'error_message': str(e)
                        })
    
            return {
                'document': doc_name,
                'total_summaries': total_summaries,
                'low_scores': low_scores
            }
            
        except Exception as e:
            return {'error': f'Error processing document {doc_name}: {str(e)}'}

    def analyze_dataset(self, dataset: str) -> Dict:
        """
        Analyze an entire dataset using parallel processing.
        
        Orchestrates the parallel processing of all documents in a dataset
        and aggregates the results.
        
        Args:
            dataset (str): Name of the dataset to analyze
            
        Returns:
            Dict containing aggregated results for the dataset
        """
        eval_file = self.results_path / f"results_evaluation_{dataset}.json"
        summary_file = self.summaries_path / f"summarized_docs_{dataset}.json"
        
        try:
            # Read evaluation data to get document list
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # Prepare arguments for parallel processing
            doc_names = list(eval_data[dataset]['documents'].keys())
            process_args = [(dataset, doc_name, eval_file, summary_file) 
                          for doc_name in doc_names]
            
            # Process documents in parallel
            num_cores = mp.cpu_count() - 1  # Leave one core free
            with mp.Pool(num_cores) as pool:
                with tqdm(total=len(doc_names), desc=f"Processing {dataset}") as pbar:
                    results = []
                    for result in pool.imap_unordered(partial(self.process_document), process_args):
                        results.append(result)
                        pbar.update()
            
            # Aggregate results from all processes
            total_summaries = 0
            all_low_scores = []
            errors = []
            
            for result in results:
                if 'error' in result:
                    errors.append(result['error'])
                else:
                    total_summaries += result['total_summaries']
                    all_low_scores.extend(result['low_scores'])
            
            return {
                'dataset': dataset,
                'total_summaries': total_summaries,
                'low_scores': all_low_scores,
                'low_scores_count': len(all_low_scores),
                'errors': errors
            }
            
        except Exception as e:
            return {
                'dataset': dataset,
                'error': f'Error processing dataset {dataset}: {str(e)}'
            }

    def analyze_all_datasets(self):
        """
        Analyze all datasets and generate a comprehensive report.
        
        This is the main entry point for analysis. It:
        1. Processes each dataset
        2. Aggregates results
        3. Generates statistics
        4. Saves detailed results to a file
        """
        datasets = ['maud', 'contractnli', 'cuad', 'privacy_qa']
        
        all_results = {
            'total_summaries': 0,
            'total_low_scores': 0,
            'datasets': {}
        }
        
        # Process each dataset
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            result = self.analyze_dataset(dataset)
            
            if 'error' in result:
                print(f"Error processing dataset {dataset}: {result['error']}")
                continue
                
            all_results['datasets'][dataset] = result
            all_results['total_summaries'] += result['total_summaries']
            all_results['total_low_scores'] += result['low_scores_count']
            
            # Print dataset results
            print(f"\nResults for {dataset}:")
            print(f"Total summaries analyzed: {result['total_summaries']}")
            print(f"Number of low scores (<0.6): {result['low_scores_count']}")
            if result.get('errors'):
                print(f"Errors encountered: {len(result['errors'])}")
            
            # Print details of low-scoring chunks
            print("\nLow scoring chunks:")
            for item in result['low_scores']:
                print(f"\nDocument: {item['document']}")
                print(f"Chunk ID: {item['chunk_id']}")
                print(f"Score: {item['score']}")
                print(f"Original Text: {item['original_text'][:500]}...")
                print(f"Summary: {item['summary'][:500]}...")
                print(f"Evaluation: {item['evaluation'][:500]}...")
                print("---")
        
        # Print final statistics
        print("\nFinal Statistics:")
        print(f"Total summaries analyzed across all datasets: {all_results['total_summaries']}")
        print(f"Total number of low scores (<0.6): {all_results['total_low_scores']}")
        if all_results['total_summaries'] > 0:
            percentage = (all_results['total_low_scores'] / all_results['total_summaries']) * 100
            print(f"Percentage of low scores: {percentage:.2f}%")
        
        # Save results to file
        output_file = self.base_path / "utils/quality_evaluation/results/summary_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")

def main():
    """Entry point of the script."""
    base_path = "/home/yperezhohin/Legal-Summarization-RAG"
    analyzer = SummaryAnalyzer(base_path)
    analyzer.analyze_all_datasets()

if __name__ == "__main__":
    main()