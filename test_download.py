#!/usr/bin/env python3
"""
Improved debug test for the fixed HFCorpusBuffered class
with better cleanup monitoring and error handling.
"""

import os
import shutil
import psutil
import time
import tempfile
import signal
import sys
from pathlib import Path
from loguru import logger
import click
import gc
from word2vec import HFCorpusBuffered, FEATURES

# Global variable to handle graceful shutdown
corpus_instance = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    global corpus_instance
    if corpus_instance:
        try:
            corpus_instance._stop_event.set()
            corpus_instance.cleanup_all_datasets()
        except:
            pass
    sys.exit(0)

def get_disk_usage():
    """Get current disk usage of current directory"""
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    used_gb = (usage.total - usage.free) / (1024**3)
    total_gb = usage.total / (1024**3)
    return free_gb, used_gb, total_gb

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024**2)
    return mem_mb

def get_cache_size(cache_dir):
    """Calculate size of cache directory"""
    if not os.path.exists(cache_dir):
        return 0
    
    total_size = 0
    file_count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not calculate cache size: {e}")
    
    return total_size / (1024**2), file_count  # Return in MB and file count

def monitor_system(cache_dir, label=""):
    """Monitor and log system resources"""
    free_gb, used_gb, total_gb = get_disk_usage()
    mem_mb = get_memory_usage()
    cache_mb, cache_files = get_cache_size(cache_dir)
    
    logger.info(f"{label} - Disk: {free_gb:.2f}GB free/{total_gb:.2f}GB total, "
                f"Memory: {mem_mb:.1f}MB, Cache: {cache_mb:.1f}MB ({cache_files} files)")
    
    return {
        'free_gb': free_gb,
        'used_gb': used_gb,
        'total_gb': total_gb,
        'memory_mb': mem_mb,
        'cache_mb': cache_mb,
        'cache_files': cache_files
    }

@click.command()
@click.option('--dataset', default="allenai/olmo-mix-1124", help="Dataset name")
@click.option('--subset', default="wiki", help="Subset to test (use comma-separated for multiple: wiki,algebraic-stack)")
@click.option('--max-sentences', default=10, type=int, help="Max sentences to process PER BATCH")
@click.option('--max-files', default=1, type=int, help="Max files per stream")
@click.option('--buffer-size', default=2, type=int, help="Buffer size for HFCorpusBuffered")
@click.option('--use-features/--no-features', default=True, help="Use FEATURES or not")
@click.option('--log-level', default="INFO", help="Log level")
@click.option('--monitor-interval', default=5, type=int, help="Monitor every N sentences")
def main(dataset, subset, max_sentences, max_files, buffer_size, use_features, log_level, monitor_interval):
    """Test the fixed HFCorpusBuffered class with comprehensive monitoring"""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global corpus_instance
    
    # Setup logging
    logger.remove()
    logger.add(
        "improved_debug_test.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
        rotation="10 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="{time:HH:mm:ss} | {level} | {message}\n"
    )
    
    # Handle comma-separated subsets
    if ',' in subset:
        subset_list = [s.strip() for s in subset.split(',')]
        logger.info(f"Using multiple subsets: {subset_list}")
    else:
        subset_list = subset
        logger.info(f"Using single subset: {subset}")
    
    logger.info("="*80)
    logger.info("IMPROVED HFCorpusBuffered DEBUG TEST")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  dataset: {dataset}")
    logger.info(f"  subset: {subset_list}")
    logger.info(f"  max_sentences_per_batch: {max_sentences}")
    logger.info(f"  max_files_per_stream: {max_files}")
    logger.info(f"  buffer_size: {buffer_size}")
    logger.info(f"  use_features: {use_features}")
    logger.info(f"  monitor_interval: {monitor_interval}")
    
    # Use default HuggingFace cache directory
    hf_cache_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    cache_dir = os.path.join(hf_cache_home, 'datasets')
    logger.info(f"Using default HuggingFace cache directory: {cache_dir}")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initial system monitoring
    initial_stats = monitor_system(cache_dir, "INITIAL")
    
    corpus = None
    try:
        logger.info("-" * 40)
        logger.info("CREATING CORPUS")
        logger.info("-" * 40)
        
        # Create corpus using default cache (don't pass cache_dir)
        corpus = HFCorpusBuffered(
            dataset_name=dataset,
            subset=subset_list,  # Use the processed subset list
            split="train",
            text_field="text",
            max_sentences=max_sentences,  # Now means per-batch
            max_files_per_stream=max_files,
            buffer_size=buffer_size,
            data_dir="data",
            revision=None,
            tokenizer=None,
            use_features=use_features
            # No cache_dir parameter - use default
        )
        
        # Set global reference for signal handler
        corpus_instance = corpus
        
        logger.info(f"Created HFCorpusBuffered with {len(corpus.batches)} batches")
        for batch_name, batch_info in corpus.batches.items():
            logger.info(f"  Batch: {batch_name} - {len(batch_info['files'])} files")
        
        # Monitor after corpus creation
        monitor_system(cache_dir, "AFTER CORPUS CREATION")
        
        logger.info("-" * 40)
        logger.info("STARTING ITERATION")
        logger.info("-" * 40)
        
        # Test iteration with monitoring
        sentence_count = 0
        start_time = time.time()
        last_monitor_time = start_time
        
        for i, tokens in enumerate(corpus):
            sentence_count += 1
            
            # Log first sentence
            if i == 0:
                logger.info(f"First sentence: {len(tokens)} tokens")
                if len(tokens) > 0:
                    preview = ' '.join(tokens[:10])
                    logger.info(f"  Preview: {preview}...")
            
            # Periodic monitoring
            current_time = time.time()
            if (sentence_count % monitor_interval == 0 or 
                sentence_count <= 3 or
                current_time - last_monitor_time >= 30):  # At least every 30 seconds
                
                elapsed = current_time - start_time
                rate = sentence_count / elapsed if elapsed > 0 else 0
                
                stats = monitor_system(cache_dir, f"SENTENCE {sentence_count}")
                logger.info(f"  Rate: {rate:.1f} sentences/second, Time: {elapsed:.1f}s")
                
                last_monitor_time = current_time
            
            # NO MORE GLOBAL LIMIT CHECK - let the corpus handle it per-batch
        
        # Final iteration stats
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("-" * 40)
        logger.info("ITERATION COMPLETED")
        logger.info("-" * 40)
        logger.info(f"Processed: {sentence_count} sentences in {duration:.2f} seconds")
        if duration > 0:
            logger.info(f"Average rate: {sentence_count/duration:.1f} sentences/second")
        
        # Monitor before cleanup
        before_cleanup_stats = monitor_system(cache_dir, "BEFORE CLEANUP")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
    finally:
        logger.info("-" * 40)
        logger.info("STARTING CLEANUP")
        logger.info("-" * 40)
        
        try:
            # Clean up corpus object if it exists
            if corpus is not None:
                logger.info("Cleaning up corpus...")
                # This should now actually delete cache files from disk
                try:
                    freed_mb = corpus.cleanup_all_datasets()
                    logger.info(f"Cleaned up downloaded dataset objects and freed {freed_mb:.1f}MB from disk")
                except Exception as e:
                    logger.warning(f"Error cleaning dataset objects: {e}")
                del corpus
                
            # Force garbage collection
            gc.collect()
            logger.info("Forced garbage collection")
            
            # Final system monitoring - should show reduced cache size
            final_stats = monitor_system(cache_dir, "AFTER CLEANUP")
            
            # Summary
            logger.info("-" * 40)
            logger.info("CLEANUP SUMMARY")
            logger.info("-" * 40)
            
            memory_freed = before_cleanup_stats['memory_mb'] - final_stats['memory_mb']
            cache_freed = before_cleanup_stats['cache_mb'] - final_stats['cache_mb']
            
            logger.info(f"Memory freed: {memory_freed:.1f}MB")
            logger.info(f"Cache freed from disk: {cache_freed:.1f}MB")
            logger.info(f"Final cache size: {final_stats['cache_mb']:.1f}MB ({final_stats['cache_files']} files)")
            
            # Verify cleanup effectiveness
            if cache_freed > 0:
                logger.info("✓ Cache cleanup was effective - files were deleted from disk")
            elif cache_freed < -10:  # Cache grew significantly
                logger.warning("⚠ Cache size increased - cleanup may not be working properly")
            else:
                logger.info("ℹ Cache size stable - may indicate no files were downloaded or cleanup didn't work")
            
            if memory_freed < -50:  # If memory usage increased significantly
                logger.warning("Memory usage increased during cleanup - possible memory leak")
            elif memory_freed > 0:
                logger.info("Memory usage decreased - cleanup was effective")
            else:
                logger.info("Memory usage stable")
                
        except Exception as cleanup_e:
            logger.error(f"Cleanup failed: {cleanup_e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("="*80)
    logger.info("DEBUG TEST COMPLETED")
    logger.info("="*80)

if __name__ == "__main__":
    main()