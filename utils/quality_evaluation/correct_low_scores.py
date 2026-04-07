#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regeneration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SummaryRegenerator:
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        os.makedirs('regenerated_summaries', exist_ok=True)
        logger.info(f"Initialized {self.model_name}")

    def create_regeneration_prompt(self, original_text: str, old_summary: str, evaluation: str) -> str:
        base_prompt = f"""As an expert summarizer, create an improved summary of the text below.
        Consider the previous summary and its evaluation to create a better version.
        The new summary must be no longer than 100 words and should address the issues mentioned in the evaluation.

        Original text:
        {original_text}

        Previous summary:
        {old_summary}

        Evaluation feedback:
        {evaluation}

        Generate an improved summary that better captures the original text while staying within 100 words.
        New summary:"""

        return base_prompt

    def batch_regenerate_summaries(self, prompts: List[str], batch_size: int = 22, max_new_tokens: int = 200) -> List[str]:
        all_responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("END_OF_RESPONSE")

            )
            batch_responses = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            all_responses.extend(batch_responses)
        return all_responses

    def process_chunks(self, chunks_to_process: List[Dict], batch_size: int = 22) -> Dict:
        """
        Process a specific number of chunks across all documents.
        
        Args:
            chunks_to_process: List of chunks from any document that need regeneration
            batch_size: Number of chunks to process in parallel
        """
        results = {
            "regenerated_summaries": [],
            "metadata": {
                "total_chunks": len(chunks_to_process),
                "successful_regenerations": 0,
                "failed_regenerations": 0
            }
        }
        
        prompts = []
        chunk_infos = []
        
        # Prepare all chunks for processing
        for item in chunks_to_process:
            try:
                prompt = self.create_regeneration_prompt(
                    item['original_text'],
                    item['summary'],
                    item['evaluation']
                )
                prompts.append(prompt)
                chunk_infos.append({
                    "dataset": item['dataset'],
                    "document": item['document'],
                    "chunk_id": item['chunk_id'],
                    "original_text": item['original_text'],
                    "old_summary": item['summary'],
                    "evaluation": item['evaluation']
                })
                
            except Exception as e:
                logger.error(f"Error preparing prompt for chunk {item.get('chunk_id', 'unknown')}: {str(e)}")
                results["metadata"]["failed_regenerations"] += 1
                continue
        
        logger.info(f"Processing {len(prompts)} chunks (batch size: {batch_size})...")
        
        try:
            new_summaries = self.batch_regenerate_summaries(prompts, batch_size=batch_size)
            
            for info, new_summary in zip(chunk_infos, new_summaries):
                if "New summary:" in new_summary:
                    new_summary = new_summary.split("New summary:")[-1].strip()
                
                results["regenerated_summaries"].append({
                    "dataset": info["dataset"],
                    "document": info["document"],
                    "chunk_id": info["chunk_id"],
                    "original_text": info["original_text"],
                    "old_summary": info["old_summary"],
                    "new_summary": new_summary,
                    "evaluation": info["evaluation"]
                })
                results["metadata"]["successful_regenerations"] += 1
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
        
        return results

    def cleanup(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up resources")

def regenerate_summaries(analysis_results_path: str, batch_size: int = 22) -> Dict:
    """
    Main function to regenerate summaries for low-scoring chunks in batches.
    
    Args:
        analysis_results_path: Path to the analysis results JSON
        batch_size: Size of each batch to process at once
    """
    logger.info(f"Loading analysis results from {analysis_results_path}")
    with open(analysis_results_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # Collect all chunks that need regeneration
    all_chunks = []
    for dataset_name, dataset_info in analysis_data['datasets'].items():
        for item in dataset_info['low_scores']:
            all_chunks.append({
                'dataset': dataset_name,
                **item
            })
    
    total_chunks = len(all_chunks)
    logger.info(f"Found {total_chunks} chunks that need regeneration")
    
    # Sort chunks by score (optional)
    all_chunks.sort(key=lambda x: x.get('score', 1.0))
    
    regenerator = SummaryRegenerator()
    organized_results = {}
    
    try:
        # Process chunks in batches
        for i in range(0, total_chunks, batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
            
            # Process current batch
            results = regenerator.process_chunks(batch_chunks, batch_size=len(batch_chunks))
            
            # Organize results by dataset and document
            for summary in results['regenerated_summaries']:
                dataset = summary['dataset']
                document = summary['document']
                
                if dataset not in organized_results:
                    organized_results[dataset] = {}
                
                if document not in organized_results[dataset]:
                    organized_results[dataset][document] = {
                        'regenerated_summaries': [],
                        'metadata': {
                            'total_chunks': 0,
                            'successful_regenerations': 0
                        }
                    }
                
                organized_results[dataset][document]['regenerated_summaries'].append(summary)
                organized_results[dataset][document]['metadata']['total_chunks'] += 1
                organized_results[dataset][document]['metadata']['successful_regenerations'] += 1
            
            # Save intermediate results after each batch
            intermediate_path = "regenerated_summaries_intermediate.json"
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(organized_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Completed batch. Progress: {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
    
    finally:
        regenerator.cleanup()
    
    # Save final results
    output_path = "/home/yperezhohin/Legal-Summarization-RAG/utils/quality_evaluation/results/corrected_summaries.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(organized_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Regeneration complete. Results saved to {output_path}")
    return organized_results

def main():
    try:
        analysis_results_path = "/home/yperezhohin/Legal-Summarization-RAG/utils/quality_evaluation/results/summary_analysis_results.json"
        # Now batch_size parameter controls how many chunks to process at once
        results = regenerate_summaries(analysis_results_path, batch_size=22)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()