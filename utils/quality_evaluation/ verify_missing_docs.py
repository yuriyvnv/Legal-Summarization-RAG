import json
import os

def count_documents_and_chunks(data):
    """
    Count documents and chunks in a dataset.
    Returns a dictionary with counts for each dataset.
    """
    counts = {}
    for dataset_name, documents in data.items():
        doc_count = len(documents)
        chunk_count = sum(len(chunks) for chunks in documents.values())
        counts[dataset_name] = {
            'documents': doc_count,
            'chunks': chunk_count
        }
    return counts

def verify_json_files():
    """
    Verify the count of documents and chunks in original files and merged result.
    """
    # Define paths
    base_path = "/home/yperezhohin/Legal-Summarization-RAG"
    original_files = [
        "summarized_docs_cuad.json", 
        "summarized_docs_maud.json",
        "summarized_docs_privacy_qa.json",
        "summarized_docs_contractnli.json"
    ]
    merged_file = "/home/yperezhohin/Legal-Summarization-RAG/utils/quality_evaluation/combined_updated_summaries.json"
    
    # Store original counts
    original_counts = {}
    total_original_docs = 0
    total_original_chunks = 0
    
    print("Analyzing original files...")
    print("-" * 50)
    
    # Count documents and chunks in original files
    for file in original_files:
        filepath = os.path.join(base_path, file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                counts = count_documents_and_chunks(data)
                original_counts.update(counts)
                
                # Print counts for each dataset
                for dataset, count in counts.items():
                    print(f"\nDataset: {dataset}")
                    print(f"Documents: {count['documents']}")
                    print(f"Chunks: {count['chunks']}")
                    total_original_docs += count['documents']
                    total_original_chunks += count['chunks']
                    
        except FileNotFoundError:
            print(f"Warning: File {file} not found")
        except json.JSONDecodeError:
            print(f"Error: File {file} contains invalid JSON")
    
    print("\nTotal in original files:")
    print(f"Total Documents: {total_original_docs}")
    print(f"Total Chunks: {total_original_chunks}")
    
    print("\nAnalyzing merged file...")
    print("-" * 50)
    
    # Count documents and chunks in merged file
    try:
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
            merged_counts = count_documents_and_chunks(merged_data)
            
            total_merged_docs = 0
            total_merged_chunks = 0
            
            for dataset, count in merged_counts.items():
                print(f"\nDataset: {dataset}")
                print(f"Documents: {count['documents']}")
                print(f"Chunks: {count['chunks']}")
                total_merged_docs += count['documents']
                total_merged_chunks += count['chunks']
            
            print("\nTotal in merged file:")
            print(f"Total Documents: {total_merged_docs}")
            print(f"Total Chunks: {total_merged_chunks}")
            
            # Verify counts match
            print("\nVerification Results:")
            print("-" * 50)
            
            if total_original_docs == total_merged_docs and total_original_chunks == total_merged_chunks:
                print("✅ Success: Document and chunk counts match!")
            else:
                print("❌ Error: Counts don't match!")
                print("\nDiscrepancies by dataset:")
                for dataset in set(original_counts.keys()) | set(merged_counts.keys()):
                    if dataset in original_counts and dataset in merged_counts:
                        orig = original_counts[dataset]
                        merged = merged_counts[dataset]
                        if orig != merged:
                            print(f"\nDataset: {dataset}")
                            print(f"Original: {orig['documents']} docs, {orig['chunks']} chunks")
                            print(f"Merged: {merged['documents']} docs, {merged['chunks']} chunks")
                    else:
                        print(f"\nDataset: {dataset}")
                        print("Missing in", "original" if dataset not in original_counts else "merged")
                        
    except FileNotFoundError:
        print(f"Error: Merged file {merged_file} not found")
    except json.JSONDecodeError:
        print(f"Error: Merged file {merged_file} contains invalid JSON")

if __name__ == "__main__":
    verify_json_files()