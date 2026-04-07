import json
from collections import defaultdict
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

# Set up logging for analysis tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('json_analysis.log'),
        logging.StreamHandler()
    ]
)

class JsonAnalyzer:
    """
    Analyzes the structure and content of the final document summaries JSON file.
    Provides detailed statistics about datasets, documents, and chunks.
    """
    
    def __init__(self, file_path: str):
        """Initialize the analyzer with the path to the JSON file."""
        self.file_path = Path(file_path)
        self.data = self._load_json()
        
    def _load_json(self) -> Dict:
        """Load and validate the JSON file."""
        logging.info(f"📂 Loading JSON file from {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info("✅ JSON file loaded successfully")
            return data
        except Exception as e:
            logging.error(f"❌ Error loading JSON file: {str(e)}")
            raise

    def analyze_dataset_statistics(self) -> Dict:
        """Calculate comprehensive statistics for each dataset."""
        logging.info("📊 Analyzing dataset statistics...")
        
        stats = {}
        for dataset_name, documents in self.data.items():
            # Initialize dataset statistics
            dataset_stats = {
                "total_documents": len(documents),
                "total_chunks": 0,
                "chunks_per_document": [],
                "successful_summaries": 0,
                "failed_summaries": 0,
                "average_summary_length": 0,
                "summary_lengths": []
            }
            
            # Analyze each document in the dataset
            for doc_name, doc_data in documents.items():
                num_chunks = len(doc_data["chunks"])
                dataset_stats["total_chunks"] += num_chunks
                dataset_stats["chunks_per_document"].append(num_chunks)
                
                # Analyze summary success and length
                if doc_data["general_summary"] != "Error":
                    dataset_stats["successful_summaries"] += 1
                    summary_length = len(doc_data["general_summary"].split())
                    dataset_stats["summary_lengths"].append(summary_length)
                else:
                    dataset_stats["failed_summaries"] += 1
            
            # Calculate averages and additional statistics
            dataset_stats["average_chunks_per_document"] = np.mean(dataset_stats["chunks_per_document"])
            dataset_stats["median_chunks_per_document"] = np.median(dataset_stats["chunks_per_document"])
            dataset_stats["min_chunks"] = min(dataset_stats["chunks_per_document"])
            dataset_stats["max_chunks"] = max(dataset_stats["chunks_per_document"])
            
            if dataset_stats["summary_lengths"]:
                dataset_stats["average_summary_length"] = np.mean(dataset_stats["summary_lengths"])
                dataset_stats["median_summary_length"] = np.median(dataset_stats["summary_lengths"])
                dataset_stats["min_summary_length"] = min(dataset_stats["summary_lengths"])
                dataset_stats["max_summary_length"] = max(dataset_stats["summary_lengths"])
            
            dataset_stats["success_rate"] = (dataset_stats["successful_summaries"] / 
                                           dataset_stats["total_documents"] * 100)
            
            stats[dataset_name] = dataset_stats
        
        return stats

    def generate_analysis_report(self) -> str:
        """Generate a detailed analysis report in a human-readable format."""
        stats = self.analyze_dataset_statistics()
        
        report = "📑 Document Summary Analysis Report\n"
        report += "=" * 40 + "\n\n"
        
        # Overall statistics
        total_documents = sum(s["total_documents"] for s in stats.values())
        total_chunks = sum(s["total_chunks"] for s in stats.values())
        total_successful = sum(s["successful_summaries"] for s in stats.values())
        
        report += f"Overall Statistics:\n{'-' * 20}\n"
        report += f"Total Datasets: {len(stats)}\n"
        report += f"Total Documents: {total_documents}\n"
        report += f"Total Chunks: {total_chunks}\n"
        report += f"Overall Success Rate: {(total_successful/total_documents)*100:.2f}%\n\n"
        
        # Per-dataset statistics
        for dataset_name, dataset_stats in stats.items():
            report += f"Dataset: {dataset_name}\n{'-' * 20}\n"
            report += f"Documents: {dataset_stats['total_documents']}\n"
            report += f"Total Chunks: {dataset_stats['total_chunks']}\n"
            report += f"Average Chunks per Document: {dataset_stats['average_chunks_per_document']:.2f}\n"
            report += f"Median Chunks per Document: {dataset_stats['median_chunks_per_document']:.2f}\n"
            report += f"Chunk Range: {dataset_stats['min_chunks']} - {dataset_stats['max_chunks']}\n"
            
            if dataset_stats['summary_lengths']:
                report += f"Average Summary Length (words): {dataset_stats['average_summary_length']:.2f}\n"
                report += f"Median Summary Length (words): {dataset_stats['median_summary_length']:.2f}\n"
                report += f"Summary Length Range: {dataset_stats['min_summary_length']} - {dataset_stats['max_summary_length']}\n"
            
            report += f"Successful Summaries: {dataset_stats['successful_summaries']}\n"
            report += f"Failed Summaries: {dataset_stats['failed_summaries']}\n"
            report += f"Success Rate: {dataset_stats['success_rate']:.2f}%\n\n"
        
        return report

def main():
    """Main execution function for the JSON analysis."""
    try:
        # Initialize analyzer with the JSON file path
        file_path = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/final_document_summaries.json"
        analyzer = JsonAnalyzer(file_path)
        
        # Generate and save the analysis report
        report = analyzer.generate_analysis_report()
        
        # Save the report to a file
        output_path = "document_analysis_report.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print the report to console
        print(report)
        logging.info(f"✅ Analysis complete. Report saved to {output_path}")
        
    except Exception as e:
        logging.error(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()