import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import logging
from typing import Dict, List, Tuple

# Configure logging to track both console output and maintain a log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summary_length_analysis.log'),
        logging.StreamHandler()
    ]
)

class SummaryLengthAnalyzer:
    """
    A specialized analyzer for visualizing and comparing token-based lengths of document summaries.
    This class handles both general and chunk-level summaries, creating clear visualizations
    that highlight the distribution patterns in the data.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the analyzer with required components for token counting and visualization.
        
        Args:
            file_path (str): Path to the JSON file containing document summaries
        """
        self.file_path = Path(file_path)
        logging.info("🔄 Loading tokenizer for length analysis...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.data = self._load_json_file()
        self._setup_plot_style()
        logging.info("✨ Analyzer initialization complete")

    def _load_json_file(self) -> Dict:
        """
        Load and validate the JSON data containing document summaries.
        
        Returns:
            Dict: The loaded JSON data containing document summaries
        """
        logging.info(f"📂 Loading data from: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"❌ Error loading file: {str(e)}")
            raise

    def _setup_plot_style(self):
        """
        Configure visualization parameters for optimal readability and professional appearance.
        Sets consistent font sizes and figure dimensions for clear presentation.
        """
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.figsize': (20, 10)
        })

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the LLaMA tokenizer.
        
        Args:
            text (str): Text to be tokenized
            
        Returns:
            int: Number of tokens in the text
        """
        return len(self.tokenizer.encode(text))

    def prepare_length_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the data to create separate DataFrames for general and chunk summaries.
        This separation enables appropriate scaling and comparison of the distributions.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for general and chunk summaries
        """
        general_summary_data = []
        chunk_summary_data = []
        
        for dataset_name, documents in self.data.items():
            logging.info(f"📊 Processing dataset: {dataset_name}")
            
            for doc_name, doc_data in documents.items():
                # Process general summary if valid
                if doc_data["general_summary"] != "Error":
                    general_length = self.count_tokens(doc_data["general_summary"])
                    general_summary_data.append({
                        "dataset": dataset_name,
                        "document": doc_name,
                        "length": general_length,
                        "type": "General Summary"
                    })
                
                # Process each chunk summary
                for chunk in doc_data["chunks"]:
                    chunk_length = self.count_tokens(chunk["summary"])
                    chunk_summary_data.append({
                        "dataset": dataset_name,
                        "document": doc_name,
                        "length": chunk_length,
                        "type": "Chunk Summary"
                    })
        
        return (pd.DataFrame(general_summary_data), 
                pd.DataFrame(chunk_summary_data))

    def create_violin_plots(self):
        """
        Create and save violin plots with optimized scaling and clear statistical indicators.
        The plots show distribution patterns while maintaining readability and proper scaling.
        """
        general_df, chunk_df = self.prepare_length_data()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Calculate appropriate axis limits with padding
        def get_axis_limits(data):
            min_val = data['length'].min()
            max_val = data['length'].max()
            range_val = max_val - min_val
            padding = range_val * 0.1
            return (min_val - padding, max_val + padding)
        
        # Plot general summary lengths
        sns.violinplot(data=general_df, x="dataset", y="length",
                      ax=ax1, inner="box", cut=0,
                      density_norm='width',  # Updated parameter name
                      color="lightblue")
        
        y_min_general, y_max_general = get_axis_limits(general_df)
        ax1.set_ylim(y_min_general, y_max_general)
        ax1.set_title("Distribution of General Summary Lengths", pad=20)
        ax1.set_ylabel("Number of Tokens")
        ax1.set_xlabel("Dataset")
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot chunk summary lengths
        sns.violinplot(data=chunk_df, x="dataset", y="length",
                      ax=ax2, inner="box", cut=0,
                      density_norm='width',  # Updated parameter name
                      color="lightgreen")
        
        y_min_chunk, y_max_chunk = get_axis_limits(chunk_df)
        ax2.set_ylim(y_min_chunk, y_max_chunk)
        ax2.set_title("Distribution of Chunk Summary Lengths", pad=20)
        ax2.set_ylabel("Number of Tokens")
        ax2.set_xlabel("Dataset")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add statistical annotations
        for ax, df in [(ax1, general_df), (ax2, chunk_df)]:
            for i, dataset in enumerate(df['dataset'].unique()):
                data = df[df['dataset'] == dataset]['length']
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                ax.text(i, ax.get_ylim()[1], 
                       f'Mean: {mean_val:.0f}\nMedian: {median_val:.0f}\nStd: {std_val:.0f}',
                       ha='center', va='bottom')
        
        plt.suptitle("Summary Length Distributions by Dataset (Token Counts)", 
                    fontsize=16, y=1.05)
        plt.tight_layout()
        
        # Save the visualization
        output_path = 'summary_length_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistical summary
        self._save_statistical_summary(general_df, chunk_df)
        logging.info(f"📊 Visualizations and statistics saved")

    def _save_statistical_summary(self, general_df: pd.DataFrame, chunk_df: pd.DataFrame):
        """
        Save comprehensive statistical analysis of summary lengths.
        
        Args:
            general_df (pd.DataFrame): DataFrame with general summary data
            chunk_df (pd.DataFrame): DataFrame with chunk summary data
        """
        stats = []
        
        for dataset in general_df['dataset'].unique():
            general_stats = general_df[general_df['dataset'] == dataset]['length']
            chunk_stats = chunk_df[chunk_df['dataset'] == dataset]['length']
            
            stats.append({
                "dataset": dataset,
                "general_summary_stats": {
                    "mean": float(general_stats.mean()),
                    "median": float(general_stats.median()),
                    "std": float(general_stats.std()),
                    "min": int(general_stats.min()),
                    "max": int(general_stats.max()),
                    "count": int(len(general_stats))
                },
                "chunk_summary_stats": {
                    "mean": float(chunk_stats.mean()),
                    "median": float(chunk_stats.median()),
                    "std": float(chunk_stats.std()),
                    "min": int(chunk_stats.min()),
                    "max": int(chunk_stats.max()),
                    "count": int(len(chunk_stats))
                }
            })
        
        with open('summary_length_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    """Main execution function for creating summary length visualizations."""
    try:
        file_path = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/final_summaries_striped.json"
        analyzer = SummaryLengthAnalyzer(file_path)
        analyzer.create_violin_plots()
        logging.info("✅ Analysis complete!")
        
    except Exception as e:
        logging.error(f"❌ Execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()