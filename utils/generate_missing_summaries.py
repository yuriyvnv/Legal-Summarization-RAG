from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
from typing import Dict, List
import time
from tqdm import tqdm
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def initialize_model_and_tokenizer():
    """Sets up the Llama 3 model and tokenizer"""
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    logger.info(f"Model initialized with {model_name}")
    logger.info(f"Model using dtype: {model.dtype}")
    
    return model, tokenizer

def create_instruct_prompt(chunk: str) -> str:
    """Creates a structured prompt for the Llama 3 model"""
    return f"""
    THE CHUNK: {chunk} END OF CHUNK.
    [INST] Summarise the chunk that is provided, with only 2 or 3 senteces maximums 100 words. Do not generate anything else beside the summary, neither your thoughts[/INST]"""

def generate_summary(
    model, 
    tokenizer, 
    chunk: str, 
    max_retries: int = 3
) -> str:
    """Generates summary with GPU optimization"""
    prompt = create_instruct_prompt(chunk)
    
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs.to("cuda")
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.split("[/INST]")[-1].strip()
            
            if not summary:  # Check if summary is empty
                raise ValueError("Generated empty summary")
                
            return summary
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return "Error generating summary"
            time.sleep(1)

def verify_file_access(file_paths: Dict[str, str]) -> None:
    """Verify read/write access to all necessary files"""
    for dataset, path in file_paths.items():
        # Check if file exists and has write permissions, or directory is writable
        file_path = Path(path)
        if file_path.exists():
            if not os.access(path, os.W_OK):
                raise PermissionError(f"No write permission for {path}")
        else:
            if not os.access(file_path.parent, os.W_OK):
                raise PermissionError(f"No write permission for directory {file_path.parent}")
        logger.info(f"Verified access for {dataset} at {path}")

def load_json_file(file_path: str) -> Dict:
    """Load and return JSON file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSON structure in {file_path}")
            return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def save_updated_summaries(file_path: str, updated_data: Dict):
    """Save updated summaries back to file with backup"""
    backup_path = f"{file_path}.bak"
    try:
        # Create backup of existing file if it exists
        if os.path.exists(file_path):
            os.replace(file_path, backup_path)
        
        # Save new data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        
        # Remove backup if save was successful
        if os.path.exists(backup_path):
            os.remove(backup_path)
            
        logger.info(f"Successfully updated summaries in {file_path}")
        
    except Exception as e:
        # Restore from backup if save failed
        if os.path.exists(backup_path):
            os.replace(backup_path, file_path)
        logger.error(f"Error saving to {file_path}: {str(e)}")
        raise

def process_missing_summaries(
    missing_summaries: Dict,
    processed_docs: Dict,
    model,
    tokenizer
) -> None:
    """Process missing summaries for each dataset"""
    summary_files = {
        "cuad": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_cuad.json",
        "contractnli": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_contractnli.json",
        "maud": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_maud.json",
        "privacy_qa": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_privacy_qa.json"
    }
    
    # Verify file access before starting
    verify_file_access(summary_files)
    
    # Verify dataset consistency
    for dataset_name in missing_summaries:
        if dataset_name not in processed_docs:
            raise ValueError(f"Dataset {dataset_name} not found in processed documents")
        if dataset_name not in summary_files:
            raise ValueError(f"No summary file path configured for dataset {dataset_name}")
    
    for dataset_name, documents in tqdm(missing_summaries.items(), desc="Processing datasets"):
        logger.info(f"\nProcessing missing summaries for dataset: {dataset_name}")
        
        # Load current summaries for this dataset
        try:
            current_summaries = load_json_file(summary_files[dataset_name])
        except FileNotFoundError:
            logger.info(f"No existing summary file for {dataset_name}. Creating new one.")
            current_summaries = {dataset_name: {}}
        
        # Ensure dataset exists in current_summaries
        if dataset_name not in current_summaries:
            current_summaries[dataset_name] = {}
        
        # Process each document with missing summaries
        for doc_name, missing_info in tqdm(documents.items(), desc=f"Processing {dataset_name} documents"):
            if doc_name not in processed_docs[dataset_name]:
                logger.error(f"Document {doc_name} not found in processed documents. Skipping.")
                continue
                
            logger.info(f"\nProcessing document: {doc_name}")
            
            # Check if document exists in current summaries
            doc_exists_in_summaries = (
                doc_name in current_summaries.get(dataset_name, {}) and 
                len(current_summaries[dataset_name][doc_name]) > 0
            )
            
            # Get total chunks from processed docs
            total_chunks = len(processed_docs[dataset_name][doc_name])
            
            # Initialize document in current summaries if it doesn't exist
            if not doc_exists_in_summaries:
                logger.info(f"Initializing new document: {doc_name}")
                current_summaries[dataset_name][doc_name] = [
                    {"chunk_text": chunk, "summary": ""} 
                    for chunk in processed_docs[dataset_name][doc_name]
                ]
            
            # Process each missing chunk
            for chunk_idx in tqdm(missing_info["missing_chunk_indices"], desc=f"Processing chunks for {doc_name}"):
                if chunk_idx >= total_chunks:
                    logger.error(f"Invalid chunk index {chunk_idx} for document {doc_name}")
                    continue
                    
                logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}")
                
                # Get chunk text from appropriate source
                if doc_exists_in_summaries:
                    chunk = current_summaries[dataset_name][doc_name][chunk_idx]["chunk_text"]
                else:
                    chunk = processed_docs[dataset_name][doc_name][chunk_idx]
                
                # Skip if chunk is empty or only whitespace
                if not chunk or not chunk.strip():
                    logger.warning(f"Empty chunk found at index {chunk_idx}. Skipping.")
                    continue
                
                # Generate new summary
                summary = generate_summary(model, tokenizer, chunk)
                
                # Skip if summary generation failed
                if summary == "Error generating summary":
                    logger.error(f"Failed to generate summary for chunk {chunk_idx}. Skipping.")
                    continue
                
                # Update the summary
                current_summaries[dataset_name][doc_name][chunk_idx]["summary"] = summary
                if not doc_exists_in_summaries:
                    current_summaries[dataset_name][doc_name][chunk_idx]["chunk_text"] = chunk
                
                print(f"\nSUMMARY for chunk {chunk_idx + 1}")
                print("-" * 50)
                print(summary)
                print("-" * 50)
                
                # Free up CUDA memory
                torch.cuda.empty_cache()
                time.sleep(0.5)
            
            # Save progress after each document
            save_updated_summaries(summary_files[dataset_name], current_summaries)
            logger.info(f"Saved updated summaries for {doc_name}")

def main():
    try:
        # Load missing summaries file
        missing_summaries_path = "/home/yperezhohin/Legal-Summarization-RAG/utils/missing_summaries.json"
        logger.info(f"Loading missing summaries from {missing_summaries_path}")
        missing_summaries = load_json_file(missing_summaries_path)
        
        # Load processed documents
        processed_docs_path = "/home/yperezhohin/Legal-Summarization-RAG/processed_documents.json"
        logger.info("Loading processed documents...")
        processed_docs = load_json_file(processed_docs_path)
        
        # Initialize model and tokenizer
        logger.info("Initializing model and tokenizer...")
        model, tokenizer = initialize_model_and_tokenizer()
        
        # Process missing summaries
        logger.info("Starting to process missing summaries...")
        process_missing_summaries(missing_summaries, processed_docs, model, tokenizer)
        logger.info("Successfully processed all missing summaries")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")
        raise
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()