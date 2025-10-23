import os
import json
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset, Features, Value
import time
from tqdm import tqdm
from pathlib import Path, PurePosixPath
from collections import defaultdict
import click
import yaml
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim import utils
from gensim.utils import simple_preprocess
from typing import Iterator
from utils import get_token, clean_html
from loguru import logger
import logging
from datetime import timedelta
from huggingface_hub import list_repo_files, login
import sys
import multiprocessing
import threading
from queue import Queue
import gc
import os
# import psutil

# Set HuggingFace timeout and cache settings
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '3600'  # 1 hour timeout
os.environ['DATASETS_DOWNLOAD_TIMEOUT'] = '3600'  # 1 hour timeout
os.environ['HF_DATASETS_CACHE_MAX_SIZE'] = '50GB'  # Limit cache size
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'false'  # Keep progress bars for debugging

class LoguruHandler(logging.Handler):
    def emit(self, record):
        # Use Loguru to handle logs from gensim
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())
logger.add("word2vec_training.log", level="INFO")

FEATURES = Features({
    "text": Value("string"),
    "added": Value("string"),
    "created": Value("string"),
    "attributes": Value("string"),
    "doc": Value("string"),
    "id": Value("string"),
    "metadata": Value("string"),
    "source": Value("string"),
    "version": Value("string"),
    "bff_contained_ngram_count_before_dedupe": Value("int64"),
    "previous_word_count": Value("int64"),
    "url": Value("string"),
    "warcinfo": Value("string"),
    "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob": Value("float64"),
    "language_id_whole_page_fasttext": {
        "en": Value("float64")
    },
})


HF_TOKEN = get_token('HF_TOKEN')
login(HF_TOKEN)
INDEX='v4_olmo-2-0325-32b-instruct_llama'
TOKENIZER_NAME='meta-llama/Llama-2-7b-hf'
## TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)


class HFStreamingCorpus:
    def __init__(self, 
                 dataset_name='allenai/olmo-mix-1124', 
                 split="train", 
                 text_field="text", 
                 subset=None,
                 max_sentences=None, 
                 max_files_per_stream=100,
                 data_dir="data",
                 revision=None,
                 tokenizer=None,
                 use_features=False,
                 disable_caching=False,
                 yield_style='raw',
                 return_metadata=False
                ):
        
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.text_field = text_field
        self.max_sentences = max_sentences
        self.max_files_per_stream = max_files_per_stream
        self.data_dir = data_dir.strip("/")
        self.revision = revision
        self.repo_id = dataset_name
        self.tokenizer = tokenizer
        self.use_features = use_features
        self.disable_caching = disable_caching
        if yield_style not in ['preproc', 'ids', 'raw', 'tokenize']:
            raise ValueError(f'no matching yield_style, must be preproc, ids, raw or tokenize, found {yield_style}')
        self.yield_style = yield_style  
        self.return_metadata = return_metadata
        
        if self.disable_caching:
            datasets.disable_caching()
        else:
            # Use default HuggingFace cache
            self.cache_dir = None  # Let HF handle caching
            
        self.batches = self._prepare_batches()
        self.stop_signal = object()

    def _prepare_batches(self):
        """Fixed version that respects max_files_per_stream and separates by subset"""
        logger.info(f"Fetching file list from repository: {self.dataset_name}, revision {self.revision if self.revision else 'Main'}")
        all_files = list_repo_files(self.dataset_name, repo_type="dataset", revision=self.revision)
        
        batches = {}
        batch_counter = 0
        
        if self.subset:
            if isinstance(self.subset, str):
                self.subset = [self.subset]
            
            # Process each subset separately
            for subset in self.subset:
                subset_folder = f'{self.data_dir}/{subset}'
                logger.info(f'Processing subset: {subset}')
                
                # Get files for this specific subset
                subset_files = [f for f in all_files if f.startswith(subset_folder)]
                logger.info(f'Files in {subset}: {len(subset_files)}')
                
                if not subset_files:
                    logger.warning(f'No files found for subset: {subset}')
                    continue
                
                # Create batches for this subset
                for i in range(0, len(subset_files), self.max_files_per_stream):
                    chunk = subset_files[i:i + self.max_files_per_stream]
                    
                    batch_name = f"batch_{batch_counter:04d}_{subset}"
                    batches[batch_name] = {
                        'files': chunk,
                        'subset': subset
                    }
                    
                    logger.info(f"Created batch '{batch_name}' with {len(chunk)} files from subset '{subset}'")
                    logger.info(f"  Files: {[f.split('/')[-1] for f in chunk]}")  # Show just filenames
                    batch_counter += 1
        else:
            files = [f for f in all_files if f.startswith(self.data_dir)]
            logger.info(f"Total files under `{self.data_dir}/`: {len(files)}")
            
            # Group files by subset
            subset_groups = {}
            for file_path in files:
                # Extract subset from file path: data/subset/... -> subset
                path_parts = PurePosixPath(file_path).parts
                if len(path_parts) > 1:
                    subset_name = path_parts[1]  # e.g., "wiki", "algebraic-stack"
                else:
                    subset_name = "unknown"
                
                if subset_name not in subset_groups:
                    subset_groups[subset_name] = []
                subset_groups[subset_name].append(file_path)
            
            logger.info(f"Found subsets: {list(subset_groups.keys())}")
            
            # Create batches for each subset
            for subset_name, subset_files in subset_groups.items():
                logger.info(f'Processing auto-detected subset: {subset_name} ({len(subset_files)} files)')
                
                # Create batches for this subset
                for i in range(0, len(subset_files), self.max_files_per_stream):
                    chunk = subset_files[i:i + self.max_files_per_stream]
                    
                    batch_name = f"batch_{batch_counter:04d}_{subset_name}"
                    batches[batch_name] = {
                        'files': chunk,
                        'subset': subset_name
                    }
                    
                    logger.info(f"Created batch '{batch_name}' with {len(chunk)} files from subset '{subset_name}'")
                    logger.info(f"  Files: {[f.split('/')[-1] for f in chunk]}")  # Show just filenames
                    batch_counter += 1
        
        logger.info(f"Created {len(batches)} total batches")
        return batches


    def get_default_cache_info(self):
        """Get information about the default HuggingFace cache directory"""
        hf_cache_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        datasets_cache = os.path.join(hf_cache_home, 'datasets')
        
        if os.path.exists(datasets_cache):
            try:
                total_size = sum(f.stat().st_size for f in Path(datasets_cache).rglob('*') if f.is_file())
                file_count = len(list(Path(datasets_cache).rglob('*')))
                return datasets_cache, total_size / (1024**2), file_count
            except Exception as e:
                logger.warning(f"Could not calculate cache info: {e}")
                return datasets_cache, 0, 0
        return datasets_cache, 0, 0

    def __iter__(self):
        count = 0
        if self.batches:
            for path, batch in self.batches.items():
                logger.info(f"Streaming batch: {path}")
                try:
                    ds = load_dataset(
                        self.dataset_name,
                        name=batch['subset'],
                        split=self.split,
                        streaming=True,
                        data_files={self.split: batch['files']},
                        revision=self.revision,
                        features=FEATURES if self.use_features else None,
                    )
                except Exception as ds_exception:
                    logger.error(f"Failed to load batch {path}, from {batch['subset']} with files\n: {batch['files']}: {ds_exception}") 
                    continue
                    
                try:
                    for example in tqdm(ds, desc=f"Streaming from {path}", unit="docs"):
                        try:
                            text = example.get(self.text_field, "")
                            if text:
                                match self.yield_style:
                                    case 'tokenize':
                                        yield self.tokenizer.tokenize(text)
                                    case 'ids':
                                        yield self.tokenizer.encode(text, add_special_tokens=False)
                                    case 'raw':
                                        yield text
                                    case 'preproc':
                                        yield simple_preprocess(clean_html(text.lower()))
                                count += 1
                                if self.max_sentences and count >= self.max_sentences:
                                    return
                        except Exception as ex:
                            logger.warning(f"Error processing example in {path}: {ex}")
                            continue
                except Exception as stream_ex:
                    logger.error(f"Error while streaming from batch {path}: {stream_ex}")
                    continue
        else:
            logger.info("Streaming dataset without batching")
            try:
                if self.subset:
                    ds = load_dataset(
                        self.dataset_name, 
                        name=self.subset[0] if isinstance(self.subset, list) else self.subset, 
                        split=self.split, 
                        streaming=True,
                        features=FEATURES if self.use_features else None,
                    )
                else:
                    ds = load_dataset(
                        self.dataset_name, 
                        split=self.split, 
                        streaming=True,
                        features=FEATURES if self.use_features else None,
                    )
                    
                for example in tqdm(ds, desc="Streaming examples", unit="docs"):
                    text = example.get(self.text_field, "")
                    if text:
                        match self.yield_style:
                            case 'tokenize':
                                yield self.tokenizer.tokenize(text)
                            case 'ids':
                                yield self.tokenizer.encode(text, add_special_tokens=False)
                            case 'raw':
                                yield text
                            case 'preproc':
                                yield simple_preprocess(clean_html(text.lower()))
                        count += 1
                        if self.max_sentences and count >= self.max_sentences:
                            break
                            
            except Exception as ds_exception:
                logger.error(f"Failed to load dataset: {ds_exception}")
                return

    def iter_batches(self):
        """Yield one generator per batch (and its metadata)."""
        for batch_name, batch_info in self.batches.items():
            try:
                yield self._stream_batch(batch_name, batch_info)
            except Exception as batch_iter_exception:
                logger.error(f"Error preparing batch {batch_name}: {batch_iter_exception}")
                continue


    def _stream_batch(self, batch_name, batch_info):
        """Stream one batch and collect metadata."""
        start = time.time()
        num_examples = 0

        try:
            ds = load_dataset(
                self.dataset_name,
                name=batch_info["subset"],
                split=self.split,
                streaming=True,
                data_files={self.split: batch_info["files"]},
                revision=self.revision,
                features=FEATURES if self.use_features else None,
            )
        except Exception as ds_exception:
            logger.error(
                f"Failed to load batch {batch_name} from subset '{batch_info['subset']}' "
                f"with files: {batch_info['files']}\nError: {ds_exception}"
            )
            return None  # Skip this batch entirely if dataset fails

        def gen():
            nonlocal num_examples
            try:
                for example in tqdm(ds, desc=f"Streaming from {batch_name}", unit="docs"):
                    try:
                        text = example.get(self.text_field, "")
                        if not text:
                            continue

                        match self.yield_style:
                            case 'tokenize':
                                yield self.tokenizer.tokenize(text)
                            case 'ids':
                                yield self.tokenizer.encode(text, add_special_tokens=False)
                            case 'raw':
                                yield text
                            case 'preproc':
                                yield simple_preprocess(clean_html(text.lower()))

                        num_examples += 1
                        if self.max_sentences and num_examples >= self.max_sentences:
                            return
                    except Exception as ex:
                        logger.warning(f"Error processing example in {batch_name}: {ex}")
                        continue

            except Exception as stream_ex:
                logger.error(f"Error while streaming from batch {batch_name}: {stream_ex}")

        # Return generator + metadata accessor
        return {
            "batch_name": batch_name,
            "subset": batch_info["subset"],
            "files": batch_info["files"],
            "start_time": start,
            "gen": gen,
            "meta": lambda: {
                "batch_name": batch_name,
                "subset": batch_info["subset"],
                "files": batch_info["files"],
                "num_examples": num_examples,
                "duration_sec": time.time() - start,
            },
        }

class HFCorpusBuffered(HFStreamingCorpus):
    def __init__(self, *args, buffer_size=2, yield_batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self.yield_batch_size = yield_batch_size
    def _producer(self):
        """Producer thread that downloads and queues batches"""
        try:
            for path, batch in self.batches.items():
                if self._stop_event.is_set():
                    logger.info("[Producer] Stop event set, ending early")
                    break
                    
                logger.info(f"[Producer] Starting download of batch {path} (queue size: {self.queue.qsize()}/{self.buffer_size})")
                ds = None
                try:
                    ds = load_dataset(
                        self.dataset_name,
                        name=batch['subset'],
                        split=self.split,
                        streaming=False,
                        data_files={self.split: batch['files']},
                        revision=self.revision,
                        features=FEATURES if self.use_features else None,
                        keep_in_memory=True if self.disable_caching else False,
                    )
                    if self.disable_caching:
                        # Dataset is now in memory, safe to delete cache files
                        import shutil
                        hf_cache_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
                        datasets_cache = os.path.join(hf_cache_home, 'datasets') ## don't needed right now, want to delete all
                        
                        if os.path.exists(hf_cache_home):
                            try:
                                # Calculate size before deletion
                                cache_size = sum(f.stat().st_size for f in Path(hf_cache_home).rglob('*') if f.is_file()) / (1024**2)
                                
                                # Remove all cached datasets since this one is in memory
                                shutil.rmtree(hf_cache_home)
                                logger.info(f"[Producer] Deleted cache directory after loading {path} into memory ({cache_size:.1f}MB freed)")
                                
                                # Recreate the empty directory so HF doesn't complain
                                os.makedirs(hf_cache_home, exist_ok=True)
                                
                            except Exception as e:
                                logger.warning(f"[Producer] Failed to clean cache after loading {path}: {e}")
                    # ADD THIS DEBUG CODE:
                    logger.info(f"[Producer] Dataset loaded successfully for {path}")
                    logger.info(f"[Producer] Dataset length: {len(ds)}")
                    logger.info(f"[Producer] Dataset features: {ds.features}")
                    # Check stop event before queuing
                    if self._stop_event.is_set():
                        logger.info("[Producer] Stop event set after download, cleaning up and stopping")
                        try:
                            del ds
                        except:
                            pass
                        break
                    # Track this dataset for later cleanup
                    self.queue.put((path, ds)) 
                except Exception as e:
                    logger.error(f"[Producer] Failed to load batch {path}: {e}")
                    
                    # Clean up any partial downloads from failed batch
                    if ds is not None:
                        try:
                            logger.info(f"[Producer] Cleaning up failed download for {path}")
                            self._cleanup_dataset_files_only(ds)
                            del ds
                        except Exception as cleanup_e:
                            logger.warning(f"[Producer] Failed to cleanup failed download: {cleanup_e}")
                    
                    # Try to free up space by cleaning temp files
                    import tempfile
                    import shutil
                    try:
                        # Clean HF temp files
                        hf_cache_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
                        temp_dirs = [
                            os.path.join(hf_cache_home, 'hub', 'temp'),
                            tempfile.gettempdir()
                        ]
                        for temp_dir in temp_dirs:
                            if os.path.exists(temp_dir):
                                for item in os.listdir(temp_dir):
                                    if item.startswith('tmp') or item.startswith('hf-'):
                                        try:
                                            item_path = os.path.join(temp_dir, item)
                                            if os.path.isdir(item_path):
                                                shutil.rmtree(item_path)
                                            else:
                                                os.remove(item_path)
                                            logger.info(f"[Producer] Cleaned temp file: {item_path}")
                                        except:
                                            pass
                    except Exception as temp_cleanup_e:
                        logger.warning(f"[Producer] Failed to clean temp files: {temp_cleanup_e}")
                    
                    continue
        except Exception as e:
            logger.error(f"[Producer] Unexpected error: {e}")
        finally:
            # Signal end of production
            if not self._stop_event.is_set():
                self.queue.put(None)
                logger.info("[Producer] Finished loading all batches")
            else:
                self.queue.put(self.stop_signal)
                logger.info("[Producer] Stopped early due to stop event")

    def _cleanup_dataset(self, ds, path):
        """Clean up a single dataset and its cached files"""
        if not self.disable_caching:
            try:
                # Get cache files before deleting the dataset
                cache_files = []
                if hasattr(ds, 'cache_files') and ds.cache_files:
                    if isinstance(ds.cache_files, dict):
                        # If it's a dict, get all values and flatten
                        cache_files = list(ds.cache_files.values())
                        cache_files = [f for sublist in cache_files for f in sublist] if cache_files else []
                    elif isinstance(ds.cache_files, list):
                        # If it's already a list, use it directly
                        cache_files = ds.cache_files
                    else:
                        logger.warning(f"[Cleanup] Unexpected cache_files type: {type(ds.cache_files)}")
                
                # Delete the dataset object first
                del ds
                gc.collect()
                
                # Now delete the actual cache files from disk
                deleted_size = 0
                deleted_count = 0
                for cache_file in cache_files:
                    file_path = None
                    
                    if isinstance(cache_file, dict) and 'filename' in cache_file:
                        file_path = cache_file['filename']
                    elif isinstance(cache_file, str):
                        file_path = cache_file
                    elif hasattr(cache_file, 'filename'):
                        file_path = cache_file.filename
                    else:
                        logger.debug(f"[Cleanup] Skipping cache file with unknown format: {cache_file}")
                        continue
                        
                    if file_path:
                        try:
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                deleted_size += file_size
                                deleted_count += 1
                                logger.debug(f"[Cleanup] Deleted cache file: {file_path}")
                            else:
                                logger.debug(f"[Cleanup] Cache file not found: {file_path}")
                        except Exception as e:
                            logger.warning(f"[Cleanup] Failed to delete cache file {file_path}: {e}")
                
                deleted_size_mb = deleted_size / (1024**2)
                if deleted_count > 0:
                    logger.info(f"[Cleanup] Cleaned up dataset for {path}: deleted {deleted_count} files ({deleted_size_mb:.1f}MB)")
                else:
                    logger.info(f"[Cleanup] Cleaned up dataset for {path}: no cache files found to delete")
                return deleted_size_mb
                
            except Exception as e:
                logger.error(f"[Cleanup] Failed to clean dataset for {path}: {e}")
                import traceback
                logger.debug(f"[Cleanup] Traceback: {traceback.format_exc()}")
                return 0
        else:
            try:
                # Just delete the dataset object - no files to clean
                # memory_usage_before = psutil.Process().memory_info().rss
                del ds
                gc.collect()
                # memory_usage_after = psutil.Process().memory_info().rss
                
                # memory_freed_mb = (memory_usage_before - memory_usage_after) / (1024**2)
                # logger.info(f"[Cleanup] Freed {memory_freed_mb:.1f}MB of memory for {path}")
                return 0 # memory_freed_mb
            except Exception as e:
                logger.error(f"[Cleanup] Failed to clean dataset for {path}: {e}")
                return 0

    
    def _cleanup_dataset_files_only(self, ds):
        """Helper to clean up just the cache files of a dataset"""
        try:
            # Get cache files before deleting the dataset
            cache_files = []
            if hasattr(ds, 'cache_files') and ds.cache_files:
                if isinstance(ds.cache_files, dict):
                    # If it's a dict, get all values and flatten
                    cache_files = list(ds.cache_files.values())
                    cache_files = [f for sublist in cache_files for f in sublist] if cache_files else []
                elif isinstance(ds.cache_files, list):
                    # If it's already a list, use it directly
                    cache_files = ds.cache_files
                else:
                    logger.warning(f"[Cleanup] Unexpected cache_files type: {type(ds.cache_files)}")
            
            # Delete the actual cache files from disk
            deleted_size = 0
            deleted_count = 0
            for cache_file in cache_files:
                file_path = None
                
                if isinstance(cache_file, dict) and 'filename' in cache_file:
                    file_path = cache_file['filename']
                elif isinstance(cache_file, str):
                    file_path = cache_file
                elif hasattr(cache_file, 'filename'):
                    file_path = cache_file.filename
                else:
                    logger.debug(f"[Cleanup] Skipping cache file with unknown format: {cache_file}")
                    continue
                    
                if file_path:
                    try:
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_size += file_size
                            deleted_count += 1
                            logger.debug(f"[Cleanup] Deleted cache file: {file_path}")
                        else:
                            logger.debug(f"[Cleanup] Cache file not found: {file_path}")
                    except Exception as e:
                        logger.warning(f"[Cleanup] Failed to delete cache file {file_path}: {e}")
            
            deleted_size_mb = deleted_size / (1024**2)
            if deleted_count > 0:
                logger.info(f"[Cleanup] Deleted {deleted_count} cache files ({deleted_size_mb:.1f}MB)")
            return deleted_size_mb
            
        except Exception as e:
            logger.error(f"[Cleanup] Failed to clean cache files: {e}")
            import traceback
            logger.debug(f"[Cleanup] Traceback: {traceback.format_exc()}")
            return 0


    def iter_batches(self):
        """Yield one generator per *pre-downloaded* batch (and its metadata)."""
        # Start producer thread in background
        producer_thread = threading.Thread(target=self._producer, daemon=False)
        producer_thread.start()

        while True:
            logger.info(f"[Consumer] Waiting for next dataset (queue: {self.queue.qsize()}/{self.buffer_size})")
            item = self.queue.get()

            # Stop signals
            if item is None or item is self.stop_signal:
                logger.info("[Consumer] All batches processed, stopping iter_batches()")
                break

            batch_name, ds = item
            batch_info = self.batches[batch_name]
            start_time = time.time()
            num_examples = 0

            logger.info(f"[Consumer] Starting iteration for batch {batch_name} ({len(ds)} examples)")

            def gen(ds=ds):
                nonlocal num_examples
                try:
                    # Iterate dataset in small HF batches
                    for batch in tqdm(
                        ds.iter(batch_size=self.yield_batch_size),
                        desc=f"Processing {batch_name} from iter_batch",
                        unit="batches",
                    ):
                        # HF returns dict of lists (e.g. {"text": ["doc1", "doc2", ...]})
                        texts = batch.get("text", [])
                        if not texts:
                            continue

                        match self.yield_style:
                            case "raw":
                                yield texts
                            case "preproc":
                                yield [simple_preprocess(clean_html(t.lower())) for t in texts]
                            case "tokenize":
                                yield [self.tokenizer.tokenize(t) for t in texts]
                            case "ids":
                                yield [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]

                        num_examples += len(texts)
                        if self.max_sentences and num_examples >= self.max_sentences:
                            logger.info(f"[Consumer] Reached max_sentences={self.max_sentences} for {batch_name}")
                            break

                except Exception as e:
                    logger.error(f"[Consumer] Error iterating {batch_name}: {e}")
                finally:
                    try:
                        del ds
                        gc.collect()
                    except Exception:
                        pass

            # yield batch descriptor
            yield {
                "batch_name": batch_name,
                "subset": batch_info["subset"],
                "files": batch_info["files"],
                "start_time": start_time,
                "gen": gen,
                "meta": lambda: {
                    "batch_name": batch_name,
                    "subset": batch_info["subset"],
                    "files": batch_info["files"],
                    "num_examples": num_examples,
                    "duration_sec": time.time() - start_time,
                },
            }

            logger.info("[Consumer] iter_batches() finished all datasets")  


    def __iter__(self):
        """Iterator that processes downloaded batches"""
        # Start background producer thread
        producer_thread = threading.Thread(target=self._producer, daemon=False)  # Not daemon!
        producer_thread.start()

        count = 0
        total_deleted_mb = 0
        
        try:
            while True:
                # Get next item from queue
                logger.info(f"[Consumer] Waiting for next batch (queue size: {self.queue.qsize()}/{self.buffer_size})")
                item = self.queue.get()
                logger.info(f"[Consumer] Retrieved batch from queue (queue size: {self.queue.qsize()}/{self.buffer_size})")
                
                # Check for end signal
                if item is self.stop_signal or item is None:
                    logger.info("[Consumer] All batches processed")
                    break
                path, ds = item
                logger.info(f"[Consumer] iterating over batch {path}")
                batch_sentence_count = 0  # Track sentences per batch
                
                try:
                    for batch in tqdm(ds.iter(batch_size=self.yield_batch_size), desc=f"Processing {path}", unit="batches"):
                        try:
                            if 'text' in batch.keys():
                                match self.yield_style:
                                    case 'tokenize':
                                        yield self.tokenizer.tokenize(batch['text']) # this might be wrong, idk whether it works with a list of text like that
                                    case 'ids':
                                        yield self.tokenizer.encode(batch['text'], add_special_tokens=False) ## This migth be wrong
                                    case 'raw':
                                        yield batch['text']
                                    case 'preproc':
                                        yield [simple_preprocess(clean_html(text.lower())) for text in batch['text']] # this screws up the earlier training pipeline I guess
                                count += 1
                                batch_sentence_count += 1
                                
                                # Check per-batch limit (max_sentences now means per-batch)
                                if self.max_sentences and batch_sentence_count >= self.max_sentences:
                                    logger.info(f"[Consumer] Reached per-batch limit ({self.max_sentences}) for {path}")
                                    break           
                        except Exception as ex:
                            logger.warning(f"[Consumer] Error in example from {path}: {ex}")
                            continue
                except Exception as stream_ex:
                    logger.error(f"[Consumer] Error iterating over data in {path}: {stream_ex}")
                    continue
                
                logger.info(f"[Consumer] Batch {path} completed: {batch_sentence_count} sentences processed")
                logger.info(f"[Consumer] Deleting the ds object")
                del ds
                gc.collect()
                
        except Exception as e:
            logger.error(f"[Consumer] Unexpected error during iteration: {e}")

def train_word2vec_model(
        corpus_iterable=None, 
        output_dir=None, 
        vector_size=200, 
        window=5, 
        min_count=5, 
        workers=4, 
        epochs=5,
        tokenizer=None,
        **kwargs):
    if isinstance(output_dir,str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_model_path = output_dir / '.tmp_word2vec.model'

    # Setup Loguru logger
    log_path = output_dir / "word2vec_training.log"
    logger.remove()  # Remove default logger to avoid duplicate logs
    logger.add(log_path, level="INFO")
    logger.add(sys.stderr, level="INFO") 

    # Redirect gensim logs to loguru
    gensim_logger = logging.getLogger("gensim")
    gensim_logger.setLevel(logging.INFO)
    gensim_logger.handlers.clear()
    gensim_logger.addHandler(LoguruHandler())
    if tokenizer:
        vocab = list(tokenizer.get_vocab().keys())
        logger.info(f'built vocab from tokenizer, length {len(vocab)}')
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=1,  # Set to 1 since we're using fixed vocab
            workers=workers,
            sg=kwargs.get('sg', 0),  # Add sg parameter if in kwargs
            sample=1e-05 ## more downsampling of frequent words 
        )
        model.build_vocab([vocab]) #needs to be a list of list > [vocab]
        logger.info("Starting Word2Vec training")
        model.train(corpus_iterable, 
            total_examples=3_080_000_000, ## this is hardcoded the amount of docs in olmo_mix, change if necessary
            epochs=epochs)
    else:
        logger.info("Starting Word2Vec training")
        model = Word2Vec(
            sentences=corpus_iterable,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sample=1e-05,
            batch_words=kwargs.get('batch_words', 10_000)
            )

    # Save model
    if output_dir:
        model_path = output_dir / "word2vec.model"
        model.save(str(model_path))
        logger.info(f"Model training complete and saved to {model_path}")
    return model


def read_config(config_path: str) -> dict:
    """
    Reads a YAML config file and fills omitted values with defaults.
    Logs warnings for each defaulted value.
    """
    DEFAULT_CONFIG = {
    "vector_size": 100,
    "window": 5,
    "min_count": 5,
    "workers": 4,
    "sg": 0,
    "epochs": 5,
    "batch_words": 10_000,
    "tokenizer_name": None,
    'buffered': False,
    'revision': None,
    'use_features': False,
    'max_files_per_stream': 4,
    'disable_caching': False,
    'total_examples': 3_08_000_000
    }
    config_path = Path(config_path)
    with open(config_path / 'config.yml', 'r') as f:
        user_config = yaml.safe_load(f) or {}

    config = user_config.copy()  # start with all user keys (including extras)

    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value
            logger.warning(f"Config key '{key}' missing; using default value: {default_value}")
    if config['tokenizer_name']:
        logger.info(f'load tokenizer, {config["tokenizer_name"]}')
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], token=HF_TOKEN)
        config['tokenizer'] = tokenizer
    else:
        config['tokenizer'] = None
    
    if 'path' in user_config:
        config['corpus_iterable'] = MyJsonCorpus(user_config['path'])
    elif 'huggingface_url' in user_config:
        hf_kwargs = {
            "dataset_name": user_config["huggingface_url"],
            "split": user_config.get("split", "train"),
            "text_field": user_config.get("text_field", "text"),
            "subset": user_config.get("subset"), 
            "max_sentences": user_config.get("max_sentences", None),
            "revision": user_config.get("revision", None),
            "use_features": config.get("use_features", False),  # PASS THE PARAMETER
            "max_files_per_stream": config.get("max_files_per_stream", 4),
            "disable_caching": config.get("disable_caching", False),
            "tokenizer": config.get("tokenizer", None)
        }
        if config['buffered']:
            config['corpus_iterable'] = HFCorpusBuffered(**hf_kwargs)      
        else:
            config['corpus_iterable'] = HFStreamingCorpus(**hf_kwargs)
    else:
        raise ValueError("Config must contain either 'path' or 'huggingface_url'")
    print(config)
    if 'outpath' not in config:
        config['output_dir'] = Path(config_path)
        print('adding outpath')
        logger.warning(f'adding default outpath {config["output_dir"]} to config')
    
    if isinstance(config['workers'], str):
        total_cpus = multiprocessing.cpu_count()
        if config['workers'].lower() == "all":
            config['workers'] = max(1, total_cpus - 1)
            logger.info(f"setting workers to {config['workers']} (all)")
        elif config['workers'].lower() == 'max':
            config['workers'] = max(1, total_cpus)
            logger.info(f"setting workers to {config['workers']} (max)")
        else:
            raise ValueError(f'malformed workers count, must be an int, ALL or MAX, got {config["workers"]}')

    return config


def model_from_config(config_dir):
    if isinstance(config_dir,str):
        config_dir = Path(config_dir)
    config_path = config_dir / "config.yml"
    if not config_path.exists():
        raise click.ClickException(f"No config.yml found in {config_dir}")

    config=read_config(config_dir)
    logger.info("Starting training...")
    start_time = time.time()
    model = train_word2vec_model(**config)
    duration_seconds = time.time() - start_time
    human_readable = str(timedelta(seconds=round(duration_seconds)))
    logger.info(f"Training completed in {human_readable} (≈ {duration_seconds:.2f} seconds)")

    return model


@click.command()
@click.argument("config_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(config_dir: Path):
    model_from_config(config_dir)


if __name__ == "__main__":
    main()