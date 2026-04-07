import json
import logging
from typing import Dict, Set, List
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Dict:
    """Load and return JSON file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def save_json_file(file_path: str, data: Dict) -> None:
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {str(e)}")
        raise

def verify_summaries(
    processed_docs: Dict,
    summary_files: Dict[str, str]
) -> Dict:
    """
    Comprehensive verification of documents and summaries
    """
    results = {}
    
    # Process each dataset
    for dataset_name, summary_path in tqdm(summary_files.items(), desc="Verifying datasets"):
        logger.info(f"\nVerifying dataset: {dataset_name}")
        
        # Skip if dataset not in processed docs
        if dataset_name not in processed_docs:
            logger.warning(f"Dataset {dataset_name} not found in processed documents")
            continue
            
        # Load summary file if it exists
        try:
            summaries = load_json_file(summary_path)
        except FileNotFoundError:
            logger.warning(f"Summary file not found for {dataset_name}")
            summaries = {}
            
        # Debug logging for structure
        logger.info(f"Summaries structure for {dataset_name}: {list(summaries.keys()) if summaries else 'empty'}")
        
        dataset_results = {
            "completely_missing_documents": [],
            "documents_with_issues": {},
            "statistics": {
                "total_documents": len(processed_docs[dataset_name]),
                "completely_missing": 0,
                "partially_missing": 0,
                "total_chunks": 0,
                "missing_chunks": 0
            }
        }
        
        # Check each document in processed_docs
        for doc_name, chunks in tqdm(processed_docs[dataset_name].items(), 
                                   desc=f"Checking documents in {dataset_name}"):
            total_chunks = len(chunks)
            dataset_results["statistics"]["total_chunks"] += total_chunks
            
            # Debug log for document checking
            logger.debug(f"Checking document: {doc_name}")
            logger.debug(f"Expected chunks: {total_chunks}")
            
            # More detailed existence check
            doc_summaries = None
            if summaries:
                # Handle both possible structures
                if dataset_name in summaries and doc_name in summaries[dataset_name]:
                    doc_summaries = summaries[dataset_name][doc_name]
                elif doc_name in summaries:  # Direct document access
                    doc_summaries = summaries[doc_name]
            
            if not doc_summaries:
                # Document completely missing
                logger.debug(f"Document {doc_name} is completely missing")
                dataset_results["completely_missing_documents"].append({
                    "document": doc_name,
                    "total_chunks": total_chunks
                })
                dataset_results["statistics"]["completely_missing"] += 1
                dataset_results["statistics"]["missing_chunks"] += total_chunks
                continue
            
            # Verify each chunk for existing documents
            missing_indices = []
            for idx, chunk in enumerate(chunks):
                # Check if summary exists and is valid
                summary_missing = (
                    idx >= len(doc_summaries) or
                    not doc_summaries[idx].get("summary") or
                    not doc_summaries[idx]["summary"].strip() or
                    doc_summaries[idx].get("summary") == "Error generating summary"
                )
                
                if summary_missing:
                    missing_indices.append(idx)
                    dataset_results["statistics"]["missing_chunks"] += 1
            
            # Record documents with missing summaries
            if missing_indices:
                dataset_results["documents_with_issues"][doc_name] = {
                    "total_chunks": total_chunks,
                    "missing_summaries": len(missing_indices),
                    "missing_chunk_indices": missing_indices
                }
                dataset_results["statistics"]["partially_missing"] += 1
                logger.debug(f"Document {doc_name} has {len(missing_indices)} missing summaries")
            else:
                logger.debug(f"Document {doc_name} is complete")
        
        # Store results for this dataset
        results[dataset_name] = dataset_results
        
        # Log summary for this dataset
        logger.info(f"\nResults for {dataset_name}:")
        logger.info(f"Total documents: {dataset_results['statistics']['total_documents']}")
        logger.info(f"Completely missing documents: {dataset_results['statistics']['completely_missing']}")
        logger.info(f"Documents with partial missing summaries: {dataset_results['statistics']['partially_missing']}")
        logger.info(f"Total chunks: {dataset_results['statistics']['total_chunks']}")
        logger.info(f"Missing chunks: {dataset_results['statistics']['missing_chunks']}")
    
    return results

def generate_missing_summaries_file(results: Dict) -> Dict:
    """
    Generate a file containing all documents needing processing
    """
    missing_summaries = {}
    
    for dataset_name, dataset_results in results.items():
        if dataset_results["completely_missing_documents"] or dataset_results["documents_with_issues"]:
            missing_summaries[dataset_name] = {}
            
            # Add completely missing documents
            for doc_info in dataset_results["completely_missing_documents"]:
                missing_summaries[dataset_name][doc_info["document"]] = {
                    "total_chunks": doc_info["total_chunks"],
                    "missing_summaries": doc_info["total_chunks"],
                    "missing_chunk_indices": list(range(doc_info["total_chunks"]))
                }
            
            # Add partially missing documents
            for doc_name, doc_info in dataset_results["documents_with_issues"].items():
                missing_summaries[dataset_name][doc_name] = doc_info
    
    return missing_summaries

def validate_summary_file_structure(file_path: str) -> Dict:
    """Validate and normalize summary file structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
            
        logger.info(f"Validating structure of {file_path}")
        
        if not isinstance(summaries, dict):
            raise ValueError(f"Invalid root structure in {file_path}")
            
        # Check first level structure
        first_key = next(iter(summaries), None)
        if first_key and isinstance(summaries[first_key], list):
            # Direct document structure
            logger.info(f"Found direct document structure in {file_path}")
            return summaries
        elif first_key and isinstance(summaries[first_key], dict):
            # Nested dataset structure
            logger.info(f"Found nested dataset structure in {file_path}")
            return summaries
        else:
            raise ValueError(f"Unrecognized structure in {file_path}")
            
    except Exception as e:
        logger.error(f"Error validating {file_path}: {str(e)}")
        raise

def main():
    # File paths
    processed_docs_path = "/home/yperezhohin/Legal-Summarization-RAG/processed_documents.json"
    summary_files = {
        "cuad": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_cuad.json",
        "contractnli": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_contractnli.json",
        "maud": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_maud.json",
        "privacy_qa": "/home/yperezhohin/Legal-Summarization-RAG/summarized_docs_privacy_qa.json"
    }
    
    # Load and validate processed documents
    logger.info("Loading processed documents...")
    processed_docs = load_json_file(processed_docs_path)
    
    # Validate summary files structure
    for dataset_name, file_path in summary_files.items():
        if Path(file_path).exists():
            try:
                validate_summary_file_structure(file_path)
            except Exception as e:
                logger.error(f"Invalid structure in {dataset_name} summary file: {str(e)}")
                raise
    
    # Verify summaries
    logger.info("Starting verification process...")
    results = verify_summaries(processed_docs, summary_files)
    
    # Generate missing summaries file
    missing_summaries = generate_missing_summaries_file(results)
    
    # Save results
    save_json_file("utils/verification_results.json", results)
    save_json_file("utils/missing_summaries.json", missing_summaries)
    
    # Print final statistics
    logger.info("\n" + "="*50)
    logger.info("FINAL STATISTICS")
    logger.info("="*50)
    
    total_docs = sum(r["statistics"]["total_documents"] for r in results.values())
    total_missing = sum(r["statistics"]["completely_missing"] for r in results.values())
    total_partial = sum(r["statistics"]["partially_missing"] for r in results.values())
    total_chunks = sum(r["statistics"]["total_chunks"] for r in results.values())
    total_missing_chunks = sum(r["statistics"]["missing_chunks"] for r in results.values())
    
    logger.info(f"Total Documents: {total_docs}")
    logger.info(f"Completely Missing Documents: {total_missing}")
    logger.info(f"Documents with Partial Missing Summaries: {total_partial}")
    logger.info(f"Total Chunks: {total_chunks}")
    logger.info(f"Missing Chunks: {total_missing_chunks}")
    logger.info(f"Overall Completion: {((total_chunks - total_missing_chunks) / total_chunks * 100):.2f}%")
    logger.info("="*50)
    
    logger.info("\nResults have been saved to:")
    logger.info("1. verification_results.json - Detailed verification results")
    logger.info("2. missing_summaries.json - File ready for processing missing summaries")

if __name__ == "__main__":
    main()