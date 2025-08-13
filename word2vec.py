"""
Modified version of your streaming code with FEATURES support
"""
import json
from transformers import AutoTokenizer
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


class MyJsonCorpus:
    """
    An iterator over a directory of JSON files.
    For each file, prep_text_file() is called, the result is lowercased,
    tokenized, and yielded as a list of tokens.
    """

    def __init__(self, dir_path, max_sentences=None):
        self.dir_path = Path(dir_path)
        self.max_sentences = max_sentences

    def __iter__(self):
        files = list(self.dir_path.glob("*.json"))
        count = 0
        for file_path in tqdm(files, desc="Processing files"):
            try:
                text = self.prep_text_file(file_path)
                text = text.lower()
                count += 1
                if self.max_sentences and count > self.max_sentences:
                    break
                yield simple_preprocess(clean_html(text))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def prep_text_file(self, file: str) -> str:
        with open(file) as f:
            json_file = json.load(f)
        if isinstance(json_file, list): # this is the output that I expect from infinigram
            text_file = ' '.join(element[0] for element in json_file)
        else:
            print(f'no list in {json_file}')
            return ''
        return text_file

    

    def search_string(self, query: str, expand=0) -> list[str] | list[tuple[str, str]]:
        """
        Return a list of filenames where the lowercase text contains the given query.
        If `expand` > 0, return a tuple of (filename, surrounding text) for the first match.
        Also prints the number of matches.
        """
        query = query.lower()
        results = []
    
        for file_path in self.dir_path.glob("*.json"):
            try:
                text = self.prep_text_file(file_path)
                text_lower = text.lower()
                match_index = text_lower.find(query)
    
                if match_index != -1:
                    if expand > 0:
                        start = max(0, match_index - expand)
                        end = min(len(text), match_index + len(query) + expand)
                        snippet = text[start:end].replace('\n', ' ').strip()
                        results.append((file_path.name, snippet))
                    else:
                        results.append(file_path.name)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
        print(f"Found {len(results)} occurrences")
        return results



class HFStreamingCorpus:
    def __init__(self, 
                 dataset_name, 
                 split="train", 
                 text_field="text", 
                 subset=None,
                 max_sentences=None, 
                 max_files_per_stream=100,
                 data_dir="data",
                 revision=None,
                 tokenizer=None,
                 use_features=False
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
            # Original logic for when no subset is specified
            files = [f for f in all_files if f.startswith(self.data_dir)]
            logger.info(f"Total files under `{self.data_dir}/`: {len(files)}")

            # Get subset name from first file
            if files:
                subset_name = PurePosixPath(files[0]).parts[1] if len(PurePosixPath(files[0]).parts) > 1 else "unknown"
            else:
                subset_name = "unknown"
            
            # Create batches
            for i in range(0, len(files), self.max_files_per_stream):
                chunk = files[i:i + self.max_files_per_stream]
                
                batch_name = f"batch_{batch_counter:04d}_{subset_name}"
                batches[batch_name] = {
                    'files': chunk,
                    'subset': subset_name
                }
                
                logger.info(f"Created batch '{batch_name}' with {len(chunk)} files")
                batch_counter += 1

        logger.info(f"Created {len(batches)} total batches")
        
        # Debug: Print all batches
        for batch_name, batch_info in batches.items():
            logger.info(f"Batch {batch_name}: {batch_info['subset']} - {len(batch_info['files'])} files")
        
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
                logger.info(f"Streaming batch from path: {path}")
                try:
                    ds = load_dataset(
                        self.dataset_name,
                        name=batch['subset'],
                        split=self.split,
                        streaming=True,
                        data_files={self.split: batch['files']},
                        revision=self.revision,
                        features=FEATURES if self.use_features else None
                    )
                except Exception as ds_exception:
                    logger.error(f"Failed to load batch from {path}: {ds_exception}") 
                    continue
                    
                try:
                    for example in tqdm(ds, desc=f"Streaming from {path}", unit="docs"):
                        try:
                            text = example.get(self.text_field, "")
                            if text:
                                if self.tokenizer:
                                    yield self.tokenizer.tokenize(text)
                                else:
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
                        features=FEATURES if self.use_features else None
                    )
                else:
                    ds = load_dataset(
                        self.dataset_name, 
                        split=self.split, 
                        streaming=True,
                        features=FEATURES if self.use_features else None
                    )
                    
                for example in tqdm(ds, desc="Streaming examples", unit="docs"):
                    text = example.get(self.text_field, "")
                    if text:
                        if self.tokenizer:
                            yield self.tokenizer.tokenize(text)
                        else:
                            yield simple_preprocess(clean_html(text.lower()))
                        count += 1
                        if self.max_sentences and count >= self.max_sentences:
                            break
                            
            except Exception as ds_exception:
                logger.error(f"Failed to load dataset: {ds_exception}")
                return


class HFStreamingCorpus:
    def __init__(self, 
                 dataset_name, 
                 split="train", 
                 text_field="text", 
                 subset=None,
                 max_sentences=None, 
                 max_files_per_stream=100,
                 data_dir="data",
                 revision=None,
                 tokenizer=None,
                 use_features=False
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
            # Original logic for when no subset is specified
            files = [f for f in all_files if f.startswith(self.data_dir)]
            logger.info(f"Total files under `{self.data_dir}/`: {len(files)}")

            # Get subset name from first file
            if files:
                subset_name = PurePosixPath(files[0]).parts[1] if len(PurePosixPath(files[0]).parts) > 1 else "unknown"
            else:
                subset_name = "unknown"
            
            # Create batches
            for i in range(0, len(files), self.max_files_per_stream):
                chunk = files[i:i + self.max_files_per_stream]
                
                batch_name = f"batch_{batch_counter:04d}_{subset_name}"
                batches[batch_name] = {
                    'files': chunk,
                    'subset': subset_name
                }
                
                logger.info(f"Created batch '{batch_name}' with {len(chunk)} files")
                batch_counter += 1

        logger.info(f"Created {len(batches)} total batches")
        
        # Debug: Print all batches
        for batch_name, batch_info in batches.items():
            logger.info(f"Batch {batch_name}: {batch_info['subset']} - {len(batch_info['files'])} files")
        
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
                logger.info(f"Streaming batch from path: {path}")
                try:
                    ds = load_dataset(
                        self.dataset_name,
                        name=batch['subset'],
                        split=self.split,
                        streaming=True,
                        data_files={self.split: batch['files']},
                        revision=self.revision,
                        features=FEATURES if self.use_features else None
                    )
                except Exception as ds_exception:
                    logger.error(f"Failed to load batch from {path}: {ds_exception}") 
                    continue
                    
                try:
                    for example in tqdm(ds, desc=f"Streaming from {path}", unit="docs"):
                        try:
                            text = example.get(self.text_field, "")
                            if text:
                                if self.tokenizer:
                                    yield self.tokenizer.tokenize(text)
                                else:
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
                        features=FEATURES if self.use_features else None
                    )
                else:
                    ds = load_dataset(
                        self.dataset_name, 
                        split=self.split, 
                        streaming=True,
                        features=FEATURES if self.use_features else None
                    )
                    
                for example in tqdm(ds, desc="Streaming examples", unit="docs"):
                    text = example.get(self.text_field, "")
                    if text:
                        if self.tokenizer:
                            yield self.tokenizer.tokenize(text)
                        else:
                            yield simple_preprocess(clean_html(text.lower()))
                        count += 1
                        if self.max_sentences and count >= self.max_sentences:
                            break
                            
            except Exception as ds_exception:
                logger.error(f"Failed to load dataset: {ds_exception}")
                return


class HFCorpusBuffered(HFStreamingCorpus):
    def __init__(self, *args, buffer_size=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._downloaded_datasets = []  # Track datasets for cleanup
                    
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
                        streaming=False,  # Download the data
                        data_files={self.split: batch['files']},
                        revision=self.revision,
                        features=FEATURES if self.use_features else None
                    )
                    
                    # Check stop event before queuing
                    if self._stop_event.is_set():
                        logger.info("[Producer] Stop event set after download, cleaning up and stopping")
                        try:
                            self._cleanup_dataset_files_only(ds)
                            del ds
                        except:
                            pass
                        break
                    
                    # Track this dataset for later cleanup
                    self._downloaded_datasets.append(ds)
                    
                    logger.info(f"[Producer] Download complete for {path}, queuing... (queue size: {self.queue.qsize()}/{self.buffer_size})")
                    self.queue.put((path, ds))  # This will block if queue is full
                    logger.info(f"[Producer] Batch {path} queued successfully (queue size: {self.queue.qsize()}/{self.buffer_size})")
                    
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
                self.queue.put(None)
                logger.info("[Producer] Stopped early due to stop event")

    def _cleanup_dataset(self, ds, path):
        """Clean up a single dataset and its cached files"""
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

    def cleanup_all_datasets(self):
        """Clean up all downloaded datasets and their cache files"""
        logger.info("[Cleanup] Starting cleanup of dataset objects and cache files")
        
        total_deleted_mb = 0
        
        # Clean up any remaining datasets and their cache files
        for ds in self._downloaded_datasets:
            try:
                deleted_mb = self._cleanup_dataset_files_only(ds)
                total_deleted_mb += deleted_mb
                del ds
            except:
                pass
        self._downloaded_datasets.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"[Cleanup] Dataset cleanup completed, freed {total_deleted_mb:.1f}MB from disk")
        
        return total_deleted_mb
    
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

    def __iter__(self):
        """Iterator that processes downloaded batches"""
        # Start background producer thread
        producer_thread = threading.Thread(target=self._producer, daemon=False)  # Not daemon!
        producer_thread.start()

        count = 0
        processed_datasets = []
        total_deleted_mb = 0
        
        try:
            while True:
                # Get next item from queue
                logger.info(f"[Consumer] Waiting for next batch (queue size: {self.queue.qsize()}/{self.buffer_size})")
                item = self.queue.get()
                logger.info(f"[Consumer] Retrieved batch from queue (queue size: {self.queue.qsize()}/{self.buffer_size})")
                
                # Check for end signal
                if item is None:
                    logger.info("[Consumer] All batches processed")
                    break
                    
                path, ds = item
                processed_datasets.append(ds)
                logger.info(f"[Consumer] Processing batch from {path}")
                
                batch_sentence_count = 0  # Track sentences per batch
                
                try:
                    for example in tqdm(ds, desc=f"Processing {path}", unit="docs"):
                        try:
                            text = example.get(self.text_field, "")
                            if text:
                                if self.tokenizer:
                                    yield self.tokenizer.tokenize(text)
                                else:
                                    yield simple_preprocess(clean_html(text.lower()))
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
                
                # Clean up this specific dataset and its cache files
                logger.info(f"[Consumer] Starting cleanup for {path} (queue size: {self.queue.qsize()}/{self.buffer_size})")
                deleted_mb = self._cleanup_dataset(ds, path)
                total_deleted_mb += deleted_mb
                logger.info(f"[Consumer] Cleanup complete for {path} (queue size: {self.queue.qsize()}/{self.buffer_size})")
                
        except Exception as e:
            logger.error(f"[Consumer] Unexpected error during iteration: {e}")
        finally:
            # Clean up any remaining datasets
            for ds in processed_datasets:
                try:
                    deleted_mb = self._cleanup_dataset_files_only(ds)
                    total_deleted_mb += deleted_mb
                    del ds
                except:
                    pass
            
            logger.info(f"[Consumer] Total disk space freed: {total_deleted_mb:.1f}MB")
            
            # Always wait for producer to finish properly
            if producer_thread.is_alive():
                logger.info("[Consumer] Waiting for producer thread to finish...")
                self._stop_event.set()
                producer_thread.join(timeout=30)
                if producer_thread.is_alive():
                    logger.warning("[Consumer] Producer thread did not finish in time")
                else:
                    logger.info("[Consumer] Producer thread finished successfully")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup_all_datasets()
        except:
            pass

def train_word2vec_model(corpus_iterable=None, output_dir=None, vector_size=200, window=5, min_count=5, workers=4, epochs=5, **kwargs):
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
    logger.info("Starting Word2Vec training")
    model = Word2Vec(
        sentences=corpus_iterable,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
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
    "batch_words": 10000,
    "tokenizer": None,
    'buffered': False,
    'revision': None,
    'use_features': False,
    'max_files_per_stream': 4
    }
    config_path = Path(config_path)
    with open(config_path / 'config.yml', 'r') as f:
        user_config = yaml.safe_load(f) or {}

    config = user_config.copy()  # start with all user keys (including extras)

    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value
            logger.warning(f"Config key '{key}' missing; using default value: {default_value}")
    
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
            "max_files_per_stream": config.get("max_files_per_stream", 4)
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