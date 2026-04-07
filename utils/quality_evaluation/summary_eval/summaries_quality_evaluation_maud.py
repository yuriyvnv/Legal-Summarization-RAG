#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
from typing import Dict, List
from tqdm import tqdm
import time

# Set the GPU (modify the device number as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    A class to evaluate text summaries using a language model with batched inference.
    It collects prompts from document chunks, runs them in batches, extracts the evaluation,
    and stores results along with indexing information.
    """
    
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.token = "token"
        if not self.token:
            raise ValueError("HF_API_KEY environment variable not set")
        
        logger.info("Loading model and tokenizer...")
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
         # Set the pad token to the eos token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # This history can be used to remind the model of formatting issues
        self.formatting_history = []
        
        # Create directory for saving evaluation results
        os.makedirs('evaluation_results', exist_ok=True)
        logger.info(f"Initialized {self.model_name}")

    def create_evaluation_prompt(self, original_text: str, summary: str) -> str:
        """
        Creates a detailed prompt for evaluation that instructs the model to respond in a fixed format.
        A special marker ("END_OF_RESPONSE") is added to signal the end of the output.
        """
        base_prompt = f"""You are an expert at evaluating text summaries. Your task is to evaluate how well a summary captures the original text.
        Your response MUST follow this exact format:
        score: [a single float number between 0.0 and 1.0]
        reason: [your detailed explanation]

        Original text:
        {original_text}

        Summary to evaluate:
        {summary}

        When done, output the marker: END_OF_RESPONSE
"""

        return base_prompt

    def extract_evaluation(self, response: str, prompt: str) -> Dict[str, str]:
        """
        Extracts the evaluation text from the model's response by removing the prompt portion.
        Also performs simple formatting checks.
        """
        # Remove the prompt text if it is present in the response
        if response.startswith(prompt):
            evaluation = response[len(prompt):].strip()
        else:
            evaluation = response.strip()
        formatting_issues = []
        print(evaluation)
        # Check that required labels exist
        if "score:" not in evaluation.lower():
            formatting_issues.append("Missing 'score:' label")
        if "reason:" not in evaluation.lower():
            formatting_issues.append("Missing 'reason:' label")
        try:
            score_line = [line for line in evaluation.split('\n') if 'score:' in line.lower()][0]
            score_str = score_line.split('score:')[1].strip()
            score = float(score_str)
            if not (0.0 <= score <= 1.0):
                formatting_issues.append(f"Score {score} is not between 0.0 and 1.0")
        except Exception as e:
            formatting_issues.append(f"Invalid score format: {str(e)}")
        
        return {
            "evaluation": evaluation,
            "formatting_issues": formatting_issues
        }
    
    def batch_evaluate_summaries(self, prompts: List[str], batch_size: int = 65, max_new_tokens: int = 128) -> List[str]:
        """
        Evaluates a list of prompts in batches. This method tokenizes a batch of prompts, generates responses,
        and decodes the outputs.
        """
        all_responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("END_OF_RESPONSE")
            )
            batch_responses = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            all_responses.extend(batch_responses)
        return all_responses

    def process_document(self, doc_name: str, chunks: List[Dict], batch_size: int = 65) -> Dict:
        """
        Processes all chunks in a document using batched inference. It generates a prompt for each chunk,
        runs batch evaluation, extracts the evaluations, and indexes them.
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
        
        prompts = []
        chunk_infos = []  # Store information to later associate each response with its chunk index
        for i, chunk in enumerate(chunks):
            prompt = self.create_evaluation_prompt(chunk["chunk_text"], chunk["summary"])
            prompts.append(prompt)
            chunk_infos.append({
                "chunk_id": f"{doc_name}_chunk_{i}",
                "chunk_index": i,
                "prompt": prompt
            })
        
        logger.info(f"Processing document '{doc_name}' with {len(prompts)} chunks (batch size: {batch_size})...")
        responses = self.batch_evaluate_summaries(prompts, batch_size=batch_size)
        
        # Iterate over each response, extract evaluation details, and index them properly.
        for info, response in zip(chunk_infos, responses):
            extracted = self.extract_evaluation(response, info["prompt"])
            evaluation_text = extracted["evaluation"]
            if extracted["formatting_issues"]:
                self.formatting_history.extend(extracted["formatting_issues"])
                results["metadata"]["failed_evaluations"] += 1
            else:
                results["metadata"]["successful_evaluations"] += 1
            
            results["evaluations"].append({
                "chunk_id": info["chunk_id"],
                "chunk_index": info["chunk_index"],
                "evaluation": evaluation_text
            })
            
            # Save intermediate progress after processing each chunk
            self._save_progress(doc_name, results)
        
        return results

    def _save_progress(self, doc_name: str, results: Dict):
        """
        Saves intermediate results to a file.
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

def evaluate_documents(json_file_path: str, batch_size: int = 65) -> Dict:
    """
    Main function to evaluate all documents contained in the input JSON file using batched inference.
    The final result follows the structure:
    
    {
      "maud": {
        "documents": {
          "<document_name>": {
            "document_name": "<document_name>",
            "evaluations": [ ... ]
          },
          ...
        }
      }
    }
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
                doc_results = evaluator.process_document(doc_name, chunks, batch_size=batch_size)
                dataset_results["documents"][doc_name] = doc_results
                
                # Save intermediate results for the dataset
                intermediate_path = f'evaluation_results/{dataset_name}_intermediate.json'
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_results, f, ensure_ascii=False, indent=2)
            
            final_results[dataset_name] = dataset_results
            logger.info(f"Completed dataset: {dataset_name}")
    
    finally:
        evaluator.cleanup()
    
    output_path = "results_evaluation_maud.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    return final_results

def main():
    """Main execution function."""
    try:
        json_file_path = "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_maud.json"
        results = evaluate_documents(json_file_path, batch_size=65)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
