import html
import re
import zstandard as zstd
import json
from huggingface_hub import HfApi
from spacy.lang.en import English
import multiprocessing as mp
import os 
import re

def clean_html(text):
    # Decode HTML entities (e.g., &lt;, &amp;, &nbsp;)
    text = html.unescape(text)

    # Collapse multiple spaces
    text = re.sub(r'</?(br|div|p|b|i|span|a)[^>]*>', ' ', text, flags=re.IGNORECASE)

    return text

nlp = English()
nlp.add_pipe("sentencizer")

def split_sentences(text):
    sentences = []
    paragraphs = text.split('\n') # split newline first then add all sentences in the line
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
    return sentences


def parallel_split_sentences(docs, n_processes=None):
    """Split sentences in parallel across multiple processes."""
    if n_processes:
        n_processes = min(n_processes, os.cpu_count() -1)
    else:
        n_processes = os.cpu_count() - 1
    with mp.Pool(n_processes) as pool:
        all_sent_lists = pool.map(split_sentences, docs, chunksize=50)
    # Flatten the list of lists
    return [s for sublist in all_sent_lists for s in sublist if s]


def get_token(key: str, filename: str = "token_file.ini") -> str | None:
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip("'\"")
    except FileNotFoundError:
        print(f"Token file '{filename}' not found.")
    return None


def read_zst_jsonl(file_path):
    """Read a zst-compressed JSONL file and return list of records"""
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
        
        dctx = zstd.ZstdDecompressor()
        try:
            decompressed_data = dctx.decompress(compressed_data)
        except zstd.ZstdError:
            # Fallback with estimated output size
            estimated_size = len(compressed_data) * 4
            decompressed_data = dctx.decompress(compressed_data, max_output_size=estimated_size)
        
        content = decompressed_data.decode('utf-8')
        
        # Parse JSONL
        records = []
        for line in content.strip().split('\n'):
            if line.strip():
                records.append(json.loads(line))
        
        return records
    


def load_all_folder_files(repo_id, folder_name, ending='.npy'):
    """Get all token files and load them into a list"""
    
    # Get repo info
    api = HfApi()
    repo_info = api.repo_info(repo_id, repo_type="dataset")
    
    # Filter for token files
    token_files = [
        sibling.rfilename for sibling in repo_info.siblings 
        if sibling.rfilename.startswith(f"{folder_name}/") and sibling.rfilename.endswith(ending)
    ]
    
    print(f"Found {len(token_files)} token files")
    
    # Download and load all files
    loaded_tokens = []
    for i, filename in enumerate(sorted(token_files)):
        print(f"Loading {i+1}/{len(token_files)}: {filename}")
        
        # Download file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
        # Load numpy array
        tokens = np.load(file_path)
        loaded_tokens.append({
            'filename': filename,
            'tokens': tokens,
            'shape': tokens.shape,
            'size': len(tokens)
        })
    
    return loaded_tokens



from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from pathlib import Path

def fetch_tokens_from_repo(repo_name, n_tokens):
    """
    Download token files until sum reaches n_tokens, but only for files that exist 
    in all folders (tokens, documents, domains_topics, domains_formats).
    """
    
    # Get repo info
    api = HfApi()
    repo_info = api.repo_info(repo_name, repo_type="dataset")
    
    # Get all files from each folder with correct extensions
    folders_extensions = {
        "tokens": ".npy",
        "documents": ".jsonl.zst", 
        "domains_topics": "__choice.npy",
        "domains_formats": "__choice.npy"
    }
    
    folder_files = {}
    for folder, extension in folders_extensions.items():
        files = [
            sibling.rfilename for sibling in repo_info.siblings 
            if sibling.rfilename.startswith(f"{folder}/") and sibling.rfilename.endswith(extension)
        ]
        
        # Extract base names (remove folder prefix and the specific extension for each folder)
        base_names = set()
        for file_path in files:
            # Remove the folder prefix
            filename = file_path.replace(f"{folder}/", "")
            # Remove the specific extension for this folder
            if filename.endswith(extension):
                base_name = filename[:-len(extension)]
                base_names.add(base_name)
        
        folder_files[folder] = base_names
        print(f"Found {len(base_names)} files in {folder}/ folder")
        if len(base_names) <= 5:  # Show examples if few files
            print(f"  Examples: {list(base_names)[:5]}")
    
    # Find intersection - files that exist in ALL folders
    common_base_names = set.intersection(*folder_files.values())
    print(f"\nFound {len(common_base_names)} files that exist in ALL folders")
    
    if not common_base_names:
        print("No files found that exist in all folders!")
        # Debug: show what's in each folder
        for folder, names in folder_files.items():
            print(f"{folder}: {list(names)[:3]}...")
        return None
    
    # Sort the common base names to process in order
    common_base_names = sorted(list(common_base_names))
    print(f"First few common files: {common_base_names[:5]}")
    
    # Download token files until we hit n_tokens (only from common files)
    selected_files = []
    total_tokens = 0
    
    for i, base_name in enumerate(common_base_names):
        token_file = f"tokens/{base_name}.npy"
        print(f"Downloading token file {i+1}/{len(common_base_names)}: {token_file}")
        
        # Download token file
        token_path = hf_hub_download(
            repo_id=repo_name,
            filename=token_file,
            repo_type="dataset"
        )
        
        # Load and sum token counts
        tokens = np.load(token_path)
        file_token_count = tokens.sum()
        
        selected_files.append({
            'base_name': base_name,
            'token_count': file_token_count
        })
        
        total_tokens += file_token_count
        print(f"  Tokens in file: {file_token_count:,}")
        print(f"  Total tokens so far: {total_tokens:,}")
        
        if total_tokens >= n_tokens:
            print(f"Reached target of {n_tokens:,} tokens with {total_tokens:,} tokens total")
            break
    
    print(f"\nSelected {len(selected_files)} files with {total_tokens:,} tokens")
    
    # Now download corresponding files from all folders
    downloaded_paths = {
        "tokens": [],
        "documents": [],
        "domains_topics": [],
        "domains_formats": []
    }
    
    # Download files from each folder for selected base names
    for folder, extension in folders_extensions.items():
        print(f"\nDownloading {len(selected_files)} files from {folder}/ folder...")
        
        for i, file_info in enumerate(selected_files):
            base_name = file_info['base_name']
            file_path_in_repo = f"{folder}/{base_name}{extension}"
            
            print(f"  Downloading {i+1}/{len(selected_files)}: {file_path_in_repo}")
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_name,
                    filename=file_path_in_repo,
                    repo_type="dataset"
                )
                downloaded_paths[folder].append(downloaded_path)
                
            except Exception as e:
                print(f"  Error downloading {file_path_in_repo}: {e}")
                downloaded_paths[folder].append(None)
    
    # Summary
    print(f"\n=== Download Summary ===")
    print(f"Files available in all folders: {len(common_base_names)}")
    print(f"Files selected based on token target: {len(selected_files)}")
    print(f"Target tokens: {n_tokens:,}")
    print(f"Actual tokens: {total_tokens:,}")
    print(f"Files downloaded per folder:")
    for folder, paths in downloaded_paths.items():
        successful_downloads = sum(1 for p in paths if p is not None)
        print(f"  {folder}: {successful_downloads}/{len(selected_files)}")
    
    return {
        'downloaded_paths': downloaded_paths,
        'total_tokens': total_tokens,
        'selected_files_info': selected_files,
        'common_files_count': len(common_base_names)
    }
