import json
import logging
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s [%(levelname)s] %(message)s',
   handlers=[
       logging.FileHandler('summary_generation.log'),
       logging.StreamHandler()
   ]
)

def save_intermediate_summary(dataset_name: str, doc_name: str, summary: str, summary_type: str):
   """Save each generated summary to a backup file."""
   backup_dir = "/home/yperezhohin/Legal-Summarization-RAG/data/summary_backups"
   os.makedirs(backup_dir, exist_ok=True)
   
   timestamp = time.strftime("%Y%m%d_%H%M%S")
   filename = f"{dataset_name}_{doc_name.replace('/', '_')}_{summary_type}_{timestamp}.json"
   filepath = os.path.join(backup_dir, filename)
   
   backup_data = {
       "dataset": dataset_name,
       "document": doc_name,
       "summary_type": summary_type,
       "summary": summary,
       "timestamp": timestamp
   }
   
   try:
       with open(filepath, 'w', encoding='utf-8') as f:
           json.dump(backup_data, f, indent=2, ensure_ascii=False)
       logging.info(f"💾 Saved {summary_type} summary backup for {doc_name}")
   except Exception as e:
       logging.error(f"❌ Failed to save backup for {doc_name}: {str(e)}")

class SummaryGenerator:
   def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
       """Initialize the summary generator with LLaMA model."""
       logging.info(f"Initializing model:🫠 {model_name}🫠")
       try:
           self.tokenizer = AutoTokenizer.from_pretrained(model_name, token="token")
           self.model = AutoModelForCausalLM.from_pretrained(
               model_name,
               torch_dtype=torch.float16,
               device_map="cuda",
               token=hf_token
           )
           
           # Set the pad token to the eos token if not already set
           if self.tokenizer.pad_token is None:
               self.tokenizer.pad_token = self.tokenizer.eos_token
           self.tokenizer.padding_side = "left"
           
           logging.info("🌈Model loaded successfully🌈")
       except Exception as e:
           logging.error(f"Failed to load model: {str(e)}")
           raise

   def create_summary_prompt(self, summaries: List[str]) -> str:
    """Create a RAG-oriented chain-of-thought prompt for document summarization."""
    base_prompt = """You are a legal summarization and retrieval expert. You are creating a document summary for a RAG. 
            Create a comprehensive summary under 600 words that will help determine document relevance.
              Your task is to combine multiple chunk summaries from a legal document into one cohesive final summary. 
              This final summary must retain all key information—such as parties, legal issues, obligations, rights, clauses, and outcomes—and be structured in a way that is easily searchable for subsequent question–answering tasks. 
           
            Input Summaries:
            {summaries}

           
            Please provide the summary starting with "CONCISED-SUMMARY:" keyword
            CONCISED-SUMMARY:  """

    prompt = base_prompt.format(
        summaries="\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(summaries)]))
    return prompt

   def create_merged_summaries_prompt(self, summary1: str, summary2: str) -> str:
    """Create a RAG-oriented prompt for merging partial summaries."""
    base_prompt = """Merge these two summaries into a single comprehensive summary under 600 words for a RAG system.

        First Summary-->
        {summary1}

        Second Summary-->
        {summary2}

        Guidelines:
        - Combine key information from both summaries
        - Maintain important details and context
        - Ensure clear and searchable language
        - Preserve specific terms and entities

        Please provide the summary starting with 'SUMMARY:'  """

    prompt = base_prompt.format(summary1=summary1, summary2=summary2)
    return prompt

   def generate_summary(self, prompt: str) -> str:
        """Generate a summary using the LLaMA model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt",
             padding=True,
             return_attention_mask=True).to(self.model.device)
            
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=750,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Get the input length to skip the prompt in the output
            
            # Decode only the new tokens (skip the prompt)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            

            return generated_text

        except Exception as e:
            logging.error(f"🚨 Error generating summary: {str(e)}")
            raise

def process_document_summaries(data: Dict) -> Dict:
   """Process all documents and maintain original structure with added general summary."""
   generator = SummaryGenerator()
   processed_data = {}
   
   for dataset_name, documents in data.items():
       logging.info(f"⛓️Processing dataset: {dataset_name}⛓️")
       processed_data[dataset_name] = {}
       
       for doc_name, chunks in documents.items():
           logging.info(f"🔗Processing document: {doc_name}🔗")
           
           # Extract all summaries for this document
           summaries = [chunk['summary'] for chunk in chunks]
           logging.info(f"📚 Found {len(summaries)} summaries for {doc_name}📚")
           try:
               # First attempt: Try processing all summaries at once
               prompt = generator.create_summary_prompt(summaries)
               general_summary = generator.generate_summary(prompt)
               torch.cuda.empty_cache()
               logging.info(f"💯 Successfully generated summary for {doc_name}💯")
               
               # Save the complete summary
               save_intermediate_summary(
                   dataset_name, 
                   doc_name, 
                   general_summary, 
                   "complete"
               )
               
           except RuntimeError as e:
               logging.warning(f"⚠ Memory error for {doc_name}, trying split approach ⚠")
               
               # Split summaries in half and process separately
               try:
                   mid_point = len(summaries) // 2
                   # Process first half
                   first_half_prompt = generator.create_summary_prompt(summaries[:mid_point])
                   first_half_summary = generator.generate_summary(first_half_prompt)
                   
                   # Save first half summary
                   save_intermediate_summary(
                       dataset_name, 
                       doc_name, 
                       first_half_summary, 
                       "first_half"
                   )
                   
                   # Process second half
                   second_half_prompt = generator.create_summary_prompt(summaries[mid_point:])
                   second_half_summary = generator.generate_summary(second_half_prompt)
                   
                   # Save second half summary
                   save_intermediate_summary(
                       dataset_name, 
                       doc_name, 
                       second_half_summary, 
                       "second_half"
                   )
                   
                   # Merge the two summaries
                   merge_prompt = generator.create_merged_summaries_prompt(
                       first_half_summary, 
                       second_half_summary
                   )
                   general_summary = generator.generate_summary(merge_prompt)
                   
                   # Save merged summary
                   save_intermediate_summary(
                       dataset_name, 
                       doc_name, 
                       general_summary, 
                       "merged"
                   )
                   
                   logging.info(f"💯 Successfully generated split summary for {doc_name} 💯")
                   
               except Exception as e:
                   logging.error(f"✗ Failed to process {doc_name} even with splitting: {str(e)}✗")
                   general_summary = "Error"
                   
                   # Save error state
                   save_intermediate_summary(
                       dataset_name, 
                       doc_name, 
                       str(e), 
                       "error"
                   )
           
           # Create new document structure with general_summary
           processed_data[dataset_name][doc_name] = {
               "general_summary": general_summary,
               "chunks": chunks  # Maintain original chunks
           }
           
           # Add a small delay to prevent potential rate limiting
           time.sleep(1)
   
   return processed_data

def main():
   """Main function to run the summary generation pipeline."""
   try:
       # Load combined data
       input_file = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/combined_updated_summaries.json"
       logging.info(f"Loading combined JSON file: {input_file}")
       
       if not os.path.exists(input_file):
           logging.error(f"🔦Input file {input_file} not found!🔦")
           return
           
       with open(input_file, 'r', encoding='utf-8') as f:
           combined_data = json.load(f)
       
       # Process all documents
       processed_data = process_document_summaries(combined_data)
       
       # Save results
       output_file = "/home/yperezhohin/Legal-Summarization-RAG/data/combined_summarized_corrected/final_document_summaries.json"
       with open(output_file, 'w', encoding='utf-8') as f:
           json.dump(processed_data, f, indent=2, ensure_ascii=False)
       
       logging.info(f"🎆 💯 Successfully saved final summaries to {output_file}🎆 💯")
       
       # Print statistics
       total_docs = sum(len(docs) for docs in processed_data.values())
       total_successful = sum(
           1 for docs in processed_data.values() 
           for doc in docs.values() 
           if doc["general_summary"] != "Error"
       )
       
       logging.info(f"""
Summary Generation Statistics:
----------------------------
Total Documents Processed: {total_docs}
Successfully Summarized: {total_successful}
Failed Summaries: {total_docs - total_successful}
Success Rate: {(total_successful/total_docs)*100:.2f}%
""")
       
   except Exception as e:
       logging.error(f"✗ Error in main process: {str(e)}")

if __name__ == "__main__":
   main()