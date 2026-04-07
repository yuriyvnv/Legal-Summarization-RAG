import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

# Configure logging to track both console output and keep a file record
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summary_analysis.log'),
        logging.StreamHandler()
    ]
)

class SummaryAnalyzer:
    """
    Analyzes and transforms document summaries based on the 'CONCISED-SUMMARY' keyword.
    This class handles loading the original JSON, processing summaries, and generating
    analysis statistics about the transformation process.
    """
    
    def __init__(self, input_file: str):
        """
        Initialize the analyzer with the input file path.
        
        Args:
            input_file (str): Path to the JSON file containing document summaries
        """
        self.input_path = Path(input_file)
        self.data = self._load_json_file()
        # Initialize counters for our analysis
        self.found_keyword = 0
        self.missing_keyword = 0
        self.total_documents = 0

    def _load_json_file(self) -> Dict:
        """
        Load and validate the JSON data containing document summaries.
        
        Returns:
            Dict: The loaded JSON data
        """
        logging.info(f"📂 Loading data from: {self.input_path}")
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info("✅ Data loaded successfully")
            return data
        except FileNotFoundError:
            logging.error(f"❌ File not found: {self.input_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"❌ Invalid JSON in file: {self.input_path}")
            raise

    def process_summary(self, summary: str) -> Tuple[str, bool]:
        """
        Process a single summary to extract the concise portion.
        
        Args:
            summary (str): The original summary text
            
        Returns:
            Tuple[str, bool]: The processed summary and whether the keyword was found
        """
        if "CONCISED-SUMMARY" in summary:
            # Extract the text after the keyword
            concise_part = summary.split("CONCISED-SUMMARY", 1)[1].strip()
            return concise_part, True
        return summary, False

    def transform_summaries(self) -> Dict:
        """
        Transform all summaries in the dataset to extract concise versions.
        Also collects statistics about the transformation process.
        
        Returns:
            Dict: The transformed data structure with statistics
        """
        transformed_data = {}
        dataset_stats = {}
        
        for dataset_name, documents in self.data.items():
            logging.info(f"📊 Processing dataset: {dataset_name}")
            transformed_data[dataset_name] = {}
            dataset_stats[dataset_name] = {
                "total_documents": len(documents),
                "found_keyword": 0,
                "missing_keyword": 0
            }
            
            for doc_name, doc_data in documents.items():
                # Process the general summary
                processed_summary, found_keyword = self.process_summary(doc_data["general_summary"])
                
                # Update statistics
                if found_keyword:
                    dataset_stats[dataset_name]["found_keyword"] += 1
                    self.found_keyword += 1
                else:
                    dataset_stats[dataset_name]["missing_keyword"] += 1
                    self.missing_keyword += 1
                
                # Create the transformed document structure
                transformed_data[dataset_name][doc_name] = {
                    "general_summary": processed_summary,
                    "chunks": doc_data["chunks"]  # Maintain original chunks
                }
                
            self.total_documents += dataset_stats[dataset_name]["total_documents"]
        
        return transformed_data, dataset_stats

    def generate_analysis_report(self, dataset_stats: Dict) -> str:
        """
        Generate a detailed analysis report of the transformation process.
        
        Args:
            dataset_stats (Dict): Statistics collected during transformation
            
        Returns:
            str: Formatted analysis report
        """
        report = "\n📊 Summary Transformation Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall statistics
        report += "Overall Statistics:\n"
        report += "-" * 20 + "\n"
        report += f"Total Documents Processed: {self.total_documents}\n"
        report += f"Total Found Keywords: {self.found_keyword}\n"
        report += f"Total Missing Keywords: {self.missing_keyword}\n"
        report += f"Overall Success Rate: {(self.found_keyword/self.total_documents)*100:.2f}%\n\n"
        
        # Per-dataset statistics
        report += "Dataset-specific Statistics:\n"
        report += "-" * 20 + "\n"
        for dataset_name, stats in dataset_stats.items():
            report += f"\nDataset: {dataset_name}\n"
            report += f"  Documents: {stats['total_documents']}\n"
            report += f"  Found Keywords: {stats['found_keyword']}\n"
            report += f"  Missing Keywords: {stats['missing_keyword']}\n"
            success_rate = (stats['found_keyword'] / stats['total_documents']) * 100
            report += f"  Success Rate: {success_rate:.2f}%\n"
        
        return report

def main():
    """Main execution function for summary transformation and analysis."""
    try:
        # Define input and output paths
        input_file = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/final_document_summaries.json"
        output_file = "final_summaries_striped.json"
        analysis_file = "summary_analysis_report.txt"
        
        # Initialize analyzer and process summaries
        analyzer = SummaryAnalyzer(input_file)
        transformed_data, dataset_stats = analyzer.transform_summaries()
        
        # Generate and save the analysis report
        analysis_report = analyzer.generate_analysis_report(dataset_stats)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis_report)
        
        # Save the transformed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)
        
        # Print the analysis report
        print(analysis_report)
        logging.info(f"✅ Transformation complete! Results saved to {output_file}")
        logging.info(f"📝 Analysis report saved to {analysis_file}")
        
    except Exception as e:
        logging.error(f"❌ Execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()