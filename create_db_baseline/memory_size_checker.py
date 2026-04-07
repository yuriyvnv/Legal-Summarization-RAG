import os
import math

def get_folder_size(folder_path: str) -> int:
    """
    Recursively calculates the total size of the folder in bytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Ensure we are not counting symbolic links
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def convert_size(size_bytes: int) -> str:
    """
    Converts size in bytes into a human-readable format.
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# Specify the path to your vector DB folder (e.g., "hierarchical_rag_db")
folder_path = "/home/yperezhohin/Legal-Summarization-RAG/data/benchmarks/baseline_benchmark_original_openai"

# Get size in bytes and convert it
size_bytes = get_folder_size(folder_path)
readable_size = convert_size(size_bytes)

print(f"📁 Folder '{folder_path}' size: {readable_size}")
