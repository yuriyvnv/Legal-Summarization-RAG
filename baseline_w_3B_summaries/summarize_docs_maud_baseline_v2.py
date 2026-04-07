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

class BatchSummarizer:
    def __init__(self, batch_size: int = 60):
        self.batch_size = batch_size
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.model = None
        self.tokenizer = None
        self.token= "token"
        
    def initialize_model(self):
        """Sets up the Llama 3 model and tokenizer with optimized GPU settings"""
        logger.info("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            token=self.token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        logger.info(f"Model initialized with {self.model_name}")
        logger.info(f"Model using dtype: {self.model.dtype}")

    def create_prompt(self, chunk: str) -> str:
        """Creates a structured prompt for the Llama 3 model"""
        return f"""
        THE CHUNK: {chunk} END OF CHUNK.
        [INST] Summarise the chunk that is provided, with only 2 or 3 senteces maximums 100 words. Do not generate anything else beside the summary, neither your thoughts[/INST]"""

    def batch_generate_summaries(self, chunks: List[str], max_retries: int = 3) -> List[str]:
        """Generates summaries for a batch of chunks"""
        prompts = [self.create_prompt(chunk) for chunk in chunks]
        
        for attempt in range(max_retries):
            try:
                # Tokenize all prompts in the batch
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to("cuda")
                
                # Generate all summaries at once
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=100,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.2
                )
                
                # Decode all outputs
                summaries = [
                    self.tokenizer.decode(output, skip_special_tokens=True).split("[/INST]")[-1].strip()
                    for output in outputs
                ]
                
                return summaries
                
            except Exception as e:
                logger.error(f"Batch attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return ["Error generating summary"] * len(chunks)
                time.sleep(1)

    def process_documents(self, chunks_dict: Dict, output_path: str) -> Dict:
        """Processes documents in batches"""
        if not self.model:
            self.initialize_model()
            
        summarized_structure = self.load_existing_progress(output_path)
        
        if "maud" not in chunks_dict:
            logger.error("'maud' key not found in input data")
            return summarized_structure

        if "maud" not in summarized_structure:
            summarized_structure["maud"] = {}
            
        # Collect all chunks that need processing
        chunks_to_process = []
        chunk_info = []  # Store metadata about each chunk
        
        for doc_name, chunks in chunks_dict["maud"].items():
            if doc_name in summarized_structure.get("maud", {}):
                logger.info(f"Skipping already processed document: {doc_name}")
                continue
                
            for i, chunk in enumerate(chunks):
                chunks_to_process.append(chunk)
                chunk_info.append({
                    "doc_name": doc_name,
                    "chunk_index": i
                })
        
        total_chunks = len(chunks_to_process)
        logger.info(f"Found {total_chunks} chunks to process")
        
        # Process chunks in batches
        for i in range(0, total_chunks, self.batch_size):
            batch_chunks = chunks_to_process[i:i + self.batch_size]
            batch_info = chunk_info[i:i + self.batch_size]
            
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_chunks + self.batch_size - 1)//self.batch_size} ({len(batch_chunks)} chunks)")
            
            # Generate summaries for the batch
            summaries = self.batch_generate_summaries(batch_chunks)
            
            # Organize results by document
            for chunk_text, info, summary in zip(batch_chunks, batch_info, summaries):
                doc_name = info["doc_name"]
                
                if doc_name not in summarized_structure["maud"]:
                    summarized_structure["maud"][doc_name] = []
                
                while len(summarized_structure["maud"][doc_name]) <= info["chunk_index"]:
                    summarized_structure["maud"][doc_name].append(None)
                
                summarized_structure["maud"][doc_name][info["chunk_index"]] = {
                    "chunk_text": chunk_text,
                    "summary": summary
                }
            
            # Save progress after each batch
            self.save_progress(summarized_structure, output_path)
            logger.info(f"Completed batch. Progress: {min(i + self.batch_size, total_chunks)}/{total_chunks} chunks")
            
            # Small delay between batches
            time.sleep(0.5)
        
        return summarized_structure

    @staticmethod
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

    @staticmethod
    def save_progress(summarized_structure: Dict, output_path: str):
        """Saves current progress"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summarized_structure, f, ensure_ascii=False, indent=2)
            logger.info(f"Progress saved successfully")
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
            raise

def main():
    # Update these paths for your environment
    input_path = "/home/yperezhohin/Legal-Summarization-RAG/processed_documents.json"
    output_path = "/home/yperezhohin/Legal-Summarization-RAG/baseline_v2/summarized_docs_maudV2.json"

    logger.info("Starting the batch summarization pipeline...")

    try:
        # Load input documents
        logger.info("Loading chunked documents...")
        with open(input_path, 'r', encoding='utf-8') as f:
            chunked_documents = json.load(f)

        # Initialize summarizer with batch size
        summarizer = BatchSummarizer(batch_size=60)
        
        # Process documents in batches
        logger.info("Starting batch summarization process...")
        summarizer.process_documents(chunked_documents, output_path)

        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()