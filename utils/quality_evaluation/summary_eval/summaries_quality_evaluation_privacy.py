from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
from typing import Dict, List
from tqdm import tqdm
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set up logging configuration for tracking the evaluation process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SummaryEvaluator:
    """
    A class to evaluate text summaries using a language model.
    Implements a feedback loop to improve evaluation quality when formatting issues occur.
    """
    
    def __init__(self):
        """
        Initializes the evaluator with a specific model and necessary settings.
        Sets up GPU resources and tracking for formatting issues.
        """
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.token = "token"
        
        if not self.token:
            raise ValueError("HF_API_KEY environment variable not set")
        
        logger.info("Loading model and tokenizer...")
        
        # Initialize model with specific settings for stability
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            token=self.token
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token
        )
        
        # Initialize formatting history for feedback loop
        self.formatting_history = []
        
        # Create directory for saving evaluation results
        os.makedirs('evaluation_results', exist_ok=True)
        
        logger.info(f"Initialized {self.model_name}")

    def create_evaluation_prompt(self, original_text: str, summary: str) -> str:
        """
        Creates a detailed prompt for evaluation, incorporating feedback from previous attempts.
        
        Args:
            original_text: The source text to be summarized
            summary: The summary to evaluate
            
        Returns:
            A string containing the complete prompt with formatting instructions
        """
        base_prompt = f"""You are an expert at evaluating text summaries. Your task is to evaluate how well a summary captures the original text.

            Original text:
            {original_text}

            Summary to evaluate:
            {summary}

            Think through this step by step:
            1. First, identify the key points in the original text
            2. Check if these key points are accurately represented in the summary
            3. Look for any missing important information
            4. Check for any added or incorrect information
            5. Evaluate the clarity and conciseness"""

        # Add formatting history if there were previous issues
        if self.formatting_history:
            base_prompt += "\n\nPrevious attempts had these formatting issues:"
            for issue in self.formatting_history:
                base_prompt += f"\n- {issue}"
            base_prompt += "\nPlease avoid these issues in your response."

        base_prompt += """

            Your response MUST follow this exact format:
            score: [a single number between 0.0 and 1.0]
            reason: [your detailed explanation]

            Examples of correct formatting:
            score: 0.8
            reason: The summary effectively captures the main points while maintaining clarity...

            score: 0.4
            reason: The summary misses several key points and lacks proper context...

            Do not include any other text or formatting in your response."""

        return base_prompt

    def extract_evaluation(self, response: str, prompt_length: int) -> Dict[str, str]:
        """
        Extracts and validates the evaluation from the model's response.
        
        Args:
            response: The complete response from the model
            prompt_length: Length of the original prompt to remove
            
        Returns:
            Dictionary containing the evaluation and any formatting issues found
        """
        evaluation = response[prompt_length:].strip()
        formatting_issues = []
        
        if not evaluation:
            formatting_issues.append("Response was empty")
            return {"evaluation": "", "formatting_issues": formatting_issues}
        
        if "score:" not in evaluation.lower():
            formatting_issues.append("Missing 'score:' label")
        
        if "reason:" not in evaluation.lower():
            formatting_issues.append("Missing 'reason:' label")
        
        try:
            score_line = [line for line in evaluation.split('\n') 
                         if 'score:' in line.lower()][0]
            score_str = score_line.split('score:')[1].strip()
            score = float(score_str)
            
            if not (0.0 <= score <= 1.0):
                formatting_issues.append(f"Score {score} is not between 0.0 and 1.0")
        except (IndexError, ValueError) as e:
            formatting_issues.append(f"Invalid score format: {str(e)}")
        
        return {
            "evaluation": evaluation,
            "formatting_issues": formatting_issues
        }

    def evaluate_summary(self, original_text: str, summary: str, 
                        max_attempts: int = 3) -> Dict[str, str]:
        """
        Evaluates a summary with retry logic and formatting feedback.
        
        Args:
            original_text: The source text to be summarized
            summary: The summary to evaluate
            max_attempts: Maximum number of retry attempts
            
        Returns:
            Dictionary containing the evaluation results
        """
        for attempt in range(max_attempts):
            try:
                prompt = self.create_evaluation_prompt(original_text, summary)
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_length=1000,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = self.extract_evaluation(response, len(prompt))
                
                if not result["formatting_issues"]:
                    self.formatting_history = []
                    return {"raw_evaluation": result["evaluation"]}
                
                self.formatting_history.extend(result["formatting_issues"])
                logger.warning(
                    f"Attempt {attempt + 1} had formatting issues: "
                    f"{', '.join(result['formatting_issues'])}"
                )
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
                self.formatting_history.append(f"Technical error: {str(e)}")
                time.sleep(2)
        
        return {
            "raw_evaluation": (
                f"Evaluation failed after {max_attempts} attempts. "
                f"Formatting issues: {', '.join(self.formatting_history)}"
            )
        }

    def process_document(self, doc_name: str, chunks: List[Dict]) -> Dict:
        """
        Processes all chunks in a document and saves progress.
        
        Args:
            doc_name: Name of the document being processed
            chunks: List of text chunks to evaluate
            
        Returns:
            Dictionary containing evaluation results for all chunks
        """
        results = {
            "document_name": doc_name,
            "evaluations": [],
            "metadata": {
                "total_chunks": len(chunks),
                "successful_evaluations": 0,
                "failed_evaluations": 0
            }
        }
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_name}_chunk_{i}"
            logger.info(f"Processing {chunk_id}")
            
            evaluation = self.evaluate_summary(
                chunk["chunk_text"],
                chunk["summary"]
            )
            
            # Track success/failure
            if "Evaluation failed" not in evaluation["raw_evaluation"]:
                results["metadata"]["successful_evaluations"] += 1
            else:
                results["metadata"]["failed_evaluations"] += 1
            
            results["evaluations"].append({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "evaluation": evaluation["raw_evaluation"]
            })
            
            # Save progress after each chunk
            self._save_progress(doc_name, results)
        
        return results

    def _save_progress(self, doc_name: str, results: Dict):
        """
        Saves intermediate results to a file.
        
        Args:
            doc_name: Name of the document
            results: Current results to save
        """
        save_path = f'evaluation_results/{doc_name}_progress.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def cleanup(self):
        """Cleans up GPU resources and resets formatting history."""
        del self.model
        del self.tokenizer
        self.formatting_history = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up resources")

def evaluate_documents(json_file_path: str) -> Dict:
    """
    Main function to evaluate all documents in the input JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing documents to evaluate
        
    Returns:
        Dictionary containing all evaluation results
    """
    logger.info(f"Loading data from {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    evaluator = SummaryEvaluator()
    final_results = {}
    
    try:
        for dataset_name, documents in data.items():
            logger.info(f"Processing dataset: {dataset_name}")
            dataset_results = {"documents": {}}
            
            for doc_name, chunks in documents.items():
                doc_results = evaluator.process_document(doc_name, chunks)
                dataset_results["documents"][doc_name] = doc_results
                
                # Save intermediate results
                intermediate_path = (
                    f'evaluation_results/{dataset_name}_intermediate.json'
                )
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_results, f, ensure_ascii=False, indent=2)
            
            final_results[dataset_name] = dataset_results
            logger.info(f"Completed dataset: {dataset_name}")
    
    finally:
        evaluator.cleanup()
    
    # Save final results
    output_path = "results_evaluation_privacy_qa.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    return final_results

def main():
    """Main execution function."""
    try:
        json_file_path = "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_privacy_qa.json"
        results = evaluate_documents(json_file_path)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()