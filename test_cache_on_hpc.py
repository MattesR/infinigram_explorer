#!/usr/bin/env python3
"""
Simple test for in-memory dataset loading with no caching
"""

import os
import sys
import traceback
import datasets
from datasets import load_dataset

# Print all relevant environment variables
print("="*60)
print("ENVIRONMENT VARIABLES")
print("="*60)
env_vars = [
    'HF_HOME',
    'HF_DATASETS_CACHE',
    'HF_DATASETS_OFFLINE',
    'HF_DATASETS_IN_MEMORY_MAX_SIZE',
    'TRANSFORMERS_CACHE',
    'TRANSFORMERS_OFFLINE',
    'HF_HUB_CACHE',
    'HF_HUB_OFFLINE',
    'HOME',
    'USER',
    'TMPDIR',
    'TEMP',
    'TMP'
]

for var in env_vars:
    value = os.environ.get(var, '<not set>')
    print(f"{var}: {value}")

# Show actual cache directory
cache_dir = os.path.expanduser('~/.cache/huggingface')
print(f"\nDefault cache dir: {cache_dir}")
print(f"Cache dir exists: {os.path.exists(cache_dir)}")
if os.path.exists(cache_dir):
    subdirs = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
    print(f"Cache subdirectories: {subdirs}")

print("\n" + "="*60)
print("TESTING IN-MEMORY DATASET LOADING")
print("="*60)

# Disable caching globally
print("\nDisabling caching...")
datasets.disable_caching()

# Test parameters
dataset_name = "allenai/olmo-mix-1124"
subset = "wiki"
max_examples = 10

print(f"\nDataset: {dataset_name}")
print(f"Subset: {subset}")
print(f"Max examples to test: {max_examples}")

try:
    print("\nLoading dataset with keep_in_memory=True and caching disabled...")
    
    ds = load_dataset(
        dataset_name,
        name=subset,
        split="train",
        streaming=False,
        keep_in_memory=True,
        trust_remote_code=True
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset type: {type(ds)}")
    print(f"Dataset length: {len(ds) if hasattr(ds, '__len__') else 'unknown'}")
    print(f"Dataset features: {ds.features if hasattr(ds, 'features') else 'no features'}")
    
    # Try to access some examples
    print(f"\nAccessing first {max_examples} examples...")
    for i in range(min(max_examples, len(ds) if hasattr(ds, '__len__') else max_examples)):
        example = ds[i]
        print(f"Example {i}: {list(example.keys())}, text length: {len(example.get('text', ''))}")
    
    print("\n✓ SUCCESS: Dataset loaded and accessed without issues")
    
except Exception as e:
    print("\n✗ FAILED: Dataset loading failed")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\nFULL TRACEBACK:")
    print("-"*60)
    traceback.print_exc()
    print("-"*60)
    
    # Also try to get more details about the error
    print("\nAdditional error details:")
    print(f"Args: {e.args if hasattr(e, 'args') else 'no args'}")
    
    # If it's an import or module error, show more context
    if isinstance(e, (ImportError, ModuleNotFoundError)):
        print(f"Missing module details: {e.name if hasattr(e, 'name') else 'unknown'}")
        print(f"Path: {e.path if hasattr(e, 'path') else 'unknown'}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)