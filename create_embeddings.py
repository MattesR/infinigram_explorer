#!/usr/bin/env python3
"""
Script to create embeddings from pickle files containing lists of sentences.

Usage:
    python embed_sentences.py <model_name> <in_path> <out_path>

Arguments:
    model_name: Name of the sentence-transformers model to use
    in_path: Path to pickle file or directory containing pickle files
    out_path: Output directory for embeddings (will be created if it doesn't exist)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from huggingface_hub import login
from utils import get_token

HF_TOKEN = get_token('HF_TOKEN')
login(HF_TOKEN)



def load_sentences_from_pickle(pickle_path):
    """Load list of sentences from pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            sentences = pickle.load(f)
        
        if not isinstance(sentences, list):
            raise ValueError(f"Expected list of strings, got {type(sentences)}")
        
        # Ensure all items are strings
        sentences = [str(sentence) for sentence in sentences]
        return sentences
    
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None


def save_embeddings(embeddings, output_path):
    """Save embeddings as numpy array."""
    try:
        np.save(output_path, embeddings)
        print(f"Saved embeddings to {output_path}")
    except Exception as e:
        print(f"Error saving embeddings to {output_path}: {e}")


def process_single_file(model, pickle_path, output_dir):
    """Process a single pickle file and save embeddings."""
    print(f"Processing {pickle_path}...")
    
    # Load sentences
    sentences = load_sentences_from_pickle(pickle_path)
    if sentences is None:
        return False
    
    print(f"Loaded {len(sentences)} sentences")
    
    # Create embeddings
    try:
        embeddings = model.encode(sentences, show_progress_bar=True)
        print(f"Created embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False
    
    # Generate output filename
    input_stem = Path(pickle_path).stem
    output_path = output_dir / f"{input_stem}_embeddings.npy"
    
    # Save embeddings
    save_embeddings(embeddings, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create sentence embeddings from pickle files"
    )
    parser.add_argument("model_name", help="Sentence-transformers model name")
    parser.add_argument("in_path", help="Input pickle file or directory")
    parser.add_argument("out_path", help="Output directory for embeddings")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    
    # Validate input path
    if not in_path.exists():
        print(f"Error: Input path {in_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_path}")
    
    # Load model
    print(f"Loading model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        sys.exit(1)
    
    # Determine if input is file or directory
    if in_path.is_file():
        # Process single file
        if not in_path.suffix == '.pickle':
            print(f"Warning: Expected .pickle file, got {in_path.suffix}")
        
        success = process_single_file(model, in_path, out_path)
        if not success:
            sys.exit(1)
    
    elif in_path.is_dir():
        # Process all pickle files in directory
        pickle_files = list(in_path.glob("*.pickle"))
        
        if not pickle_files:
            print(f"No .pickle files found in {in_path}")
            sys.exit(1)
        
        print(f"Found {len(pickle_files)} pickle files to process")
        
        failed_files = []
        for pickle_file in tqdm(pickle_files, desc="Processing files"):
            success = process_single_file(model, pickle_file, out_path)
            if not success:
                failed_files.append(pickle_file)
        
        if failed_files:
            print(f"\nFailed to process {len(failed_files)} files:")
            for failed_file in failed_files:
                print(f"  - {failed_file}")
            sys.exit(1)
    
    else:
        print(f"Error: {in_path} is neither a file nor a directory")
        sys.exit(1)
    
    print("\nEmbedding creation completed successfully!")


if __name__ == "__main__":
    main()