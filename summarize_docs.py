from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
from typing import Dict, List
import time
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def initialize_model_and_tokenizer():
    """
    Sets up the Llama 3 model and tokenizer with optimized GPU settings
    """
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

def load_existing_progress(output_path: str) -> Dict:
    """Loads existing progress from output file"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_progress = json.load(f)
                logger.info(f"Loaded existing progress from {output_path}")
                return existing_progress
        except json.JSONDecodeError:
            logger.warning(f"Corrupted progress file found at {output_path}. Starting fresh.")
            return {}
    logger.info("No existing progress file found. Starting fresh.")
    return {}

def save_progress(
    summarized_structure: Dict, 
    output_path: str, 
    doc_name: str,
):
    """Saves progress after each document is processed"""
    try:
        # No need to load existing progress since we have the full structure
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summarized_structure, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Progress saved for document: {doc_name}")
        
    except Exception as e:
        logger.error(f"Error saving progress for {doc_name}: {str(e)}")
        raise

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
            
            return summary
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return "Error generating summary"
            time.sleep(1)

def process_maud_documents(chunks_dict: Dict, output_path: str) -> Dict:
    """
    Processes only the documents under the 'maud' key in the input JSON
    """
    model, tokenizer = initialize_model_and_tokenizer()
    summarized_structure = load_existing_progress(output_path)
    
    if "maud" not in chunks_dict:
        logger.error("'maud' key not found in input data")
        return summarized_structure

    if "maud" not in summarized_structure:
        summarized_structure["maud"] = {}
        
    total_docs = len(chunks_dict["maud"])
    logger.info(f"Found {total_docs} documents to process in CUAD")
    
    # Process each document in maud
    for doc_index, (doc_name, chunks) in enumerate(chunks_dict["maud"].items(), 1):
        # Skip already processed documents
        if doc_name in summarized_structure.get("maud", {}):
            logger.info(f"Skipping already processed document: {doc_name} ({doc_index}/{total_docs})")
            continue
            
        logger.info(f"Processing document: {doc_name} ({doc_index}/{total_docs})")
        document_summaries = []
        
        try:
            # Process chunks with progress bar
            total_chunks = len(chunks)
            for i, chunk in enumerate(tqdm(chunks, desc=f"Summarizing {doc_name}", ncols=100)):
                logger.info(f"Summarizing chunk {i+1}/{total_chunks} of {doc_name}")
                
                summary = generate_summary(model, tokenizer, chunk)
                
                summary_entry = {
                    "chunk_text": chunk,
                    "summary": summary
                }
                document_summaries.append(summary_entry)
                
                print(f"\nSUMMARY for chunk {i+1}/{total_chunks}")
                print("-" * 50)
                print(summary)
                print("-" * 50)
                
                # Reduced sleep time due to GPU optimization
                time.sleep(0.5)
            
            # Save progress after document completion
            summarized_structure["maud"][doc_name] = document_summaries
            save_progress(summarized_structure, output_path, doc_name)
            logger.info(f"Completed and saved summarization for {doc_name}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_name}: {str(e)}")
            # Save partial progress if available
            if document_summaries:
                summarized_structure["maud"][doc_name] = document_summaries
                save_progress(summarized_structure, output_path, doc_name, document_summaries)
                logger.info(f"Saved partial progress for {doc_name}")
            raise
    
    return summarized_structure


# Update these paths for your environment
input_path = "/home/yperezhohin/Legal-Summarization-RAG/processed_documents.json"
output_path = "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_maud.json"

logger.info("Starting the summarization pipeline...")

try:
    # Load input documents
    logger.info("Loading chunked documents...")
    with open(input_path, 'r', encoding='utf-8') as f:
        chunked_documents = json.load(f)

    # Process only maud documents with GPU optimization
    logger.info("Starting summarization process for CUAD documents...")
    process_maud_documents(chunked_documents, output_path)

    logger.info("Process completed successfully")
    
except Exception as e:
    logger.error(f"An error occurred in the main process: {str(e)}")
    raise