import json
import os

def load_json_file(filepath):
    """Load and return JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_summary_mapping(corrected_summaries):
    """Create a mapping of original texts to their new summaries."""
    mapping = {}
    
    # Iterate through all datasets in corrected summaries
    for dataset, documents in corrected_summaries.items():
        for doc_name, doc_data in documents.items():
            if 'regenerated_summaries' in doc_data:
                for summary_item in doc_data['regenerated_summaries']:
                    # Use the original text as key and new summary as value
                    if 'original_text' in summary_item and 'new_summary' in summary_item:
                        mapping[summary_item['original_text']] = summary_item['new_summary']
    
    return mapping

def update_summaries(original_data, summary_mapping):
    """Update summaries in the original data structure using the mapping."""
    updated_data = {}
    
    # Iterate through each dataset
    for dataset, documents in original_data.items():
        updated_data[dataset] = {}
        
        # Iterate through each document in the dataset
        for doc_name, chunks in documents.items():
            updated_chunks = []
            
            # Iterate through each chunk in the document
            for chunk in chunks:
                updated_chunk = chunk.copy()
                
                # Update summary if the chunk_text exists in our mapping
                if 'chunk_text' in chunk and chunk['chunk_text'] in summary_mapping:
                    updated_chunk['summary'] = summary_mapping[chunk['chunk_text']]
                
                updated_chunks.append(updated_chunk)
            
            updated_data[dataset][doc_name] = updated_chunks
    
    return updated_data

def main():
    # Define input file paths
    input_files = [
        "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_cuad.json",
        "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_maud.json",
        "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_privacy_qa.json",
        "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_contractnli.json"
    ]
    corrected_summaries_path = "/home/yperezhohin/Legal-Summarization-RAG/utils/quality_evaluation/results/corrected_summaries.json"
    
    # Load corrected summaries
    corrected_summaries = load_json_file(corrected_summaries_path)
    
    # Create mapping of original texts to new summaries
    summary_mapping = create_summary_mapping(corrected_summaries)
    
    # Load and combine all original datasets
    combined_data = {}
    for file in input_files:
        print(file)
        data = load_json_file(file)
        combined_data.update(data)
    
    # Update summaries in the combined data
    updated_data = update_summaries(combined_data, summary_mapping)
    
    # Write the result to a new file
    output_path = "/home/yperezhohin/Legal-Summarization-RAG/utils/quality_evaluation/combined_updated_summaries.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created combined and updated JSON file: {output_path}")

if __name__ == "__main__":
    main()