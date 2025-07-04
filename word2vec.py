"""
"""
from transformers import AutoTokenizer
from datasets import load_dataset
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
from huggingface_hub import list_repo_files

class LoguruHandler(logging.Handler):
    def emit(self, record):
        # Use Loguru to handle logs from gensim
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())
logger.add("word2vec_training.log", level="INFO")


HF_TOKEN = get_token('HF_TOKEN')
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
                 max_files_per_stream=1000,
                 data_dir="data",
                 revision=None):
        
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.text_field = text_field
        self.max_sentences = max_sentences
        self.max_files_per_stream = max_files_per_stream
        self.data_dir = data_dir.strip("/")
        self.revision = revision
        self.repo_id = dataset_name  # HF dataset name doubles as repo_id

        self.batches = self._prepare_batches()

    def _prepare_batches(self):
        logger.info(f"Listing files from dataset: {self.repo_id}")
        files = list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision)
        files = [f for f in files if f.startswith(self.data_dir)]
        logger.info(f"Total files under `{self.data_dir}/`: {len(files)}")

        # Group files by top-level subset (data/starcoder/ → starcoder)
        subsets = defaultdict(list)
        for f in files:
            parts = PurePosixPath(f).parts
            if len(parts) >= 2:
                subset = parts[1]
                subsets[subset].append(f)

        logger.info(f"Found {len(subsets)} top-level subsets: {list(subsets.keys())}")

        batches = {}
        for subset_name, subset_files in subsets.items():
            if len(subset_files) <= self.max_files_per_stream:
                batch_path = f"{self.data_dir}/{subset_name}/**/*"
                batches[batch_path] = subset_files
                logger.info(f"Added batch: {batch_path} with {len(subset_files)} files")
            else:
                # Chunk files within large subset
                for i in range(0, len(subset_files), self.max_files_per_stream):
                    chunk = subset_files[i:i + self.max_files_per_stream]
                    batch_path = f"{self.data_dir}/{subset_name}/chunk_{i//self.max_files_per_stream}/**/*"
                    batches[batch_path] = chunk
                    logger.info(f"Added chunked batch: {batch_path} with {len(chunk)} files")

        return batches if batches else None


    def __iter__(self):
        count = 0
        if self.batches:
            for path in self.batches:
                logger.info(f"Streaming batch from path: {path}")
                try:
                    ds = load_dataset(self.dataset_name,
                                        name=self.subset,
                                        split=self.split,
                                        streaming=True,
                                        data_files={self.split: path},
                                        revision=self.revision)
                except Exception as ds_exeption:
                   logger.error(f"Failed to load batch from {path}: {ds_exeption}") 
                   continue
                try:
                    for example in tqdm(ds, desc=f"Streaming from {path}", unit="docs"):
                        try:
                            text = example.get(self.text_field, "")
                            if text:
                                yield simple_preprocess(text.lower())
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
                    ds = load_dataset(self.dataset_name, name=self.subset if self.subset else None, split=self.split, streaming=True)
                else:
                    ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
            except Exception as ds_exeption:
                logger.error(f"Failed to load dataset: {ds_exeption}")
                return
        try:
            for example in tqdm(ds, desc="Streaming examples", unit="docs"):
                text = example.get(self.text_field, ""  )
                if text:
                    if self.tokenizer:
                        yield self.tokenizer.tokenize(text)
                    else:
                        yield simple_preprocess(clean_html(text))
                    count += 1
                    if self.max_sentences and count >= self.max_sentences:
                        break
        except Exception as stream_ex:
            logger.error(f"Error during dataset streaming: {stream_ex}")
            return


def train_word2vec_model(corpus_iterable=None, output_dir=None, vector_size=200, window=5, min_count=5, workers=4, epochs=5, **kwargs):
    if isinstance(output_dir,str):
        utput_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Loguru logger
    log_path = output_dir / "word2vec_training.log"
    logger.remove()  # Remove default logger to avoid duplicate logs
    logger.add(log_path, level="INFO")

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
    "batch_words": 10000
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
            "subset": user_config.get("subset"),  # Can be None
            "max_sentences": user_config.get("max_sentences", None)
        }
        config['corpus_iterable'] = HFStreamingCorpus(**hf_kwargs)
    else:
        raise ValueError("Config must contain either 'path' or 'huggingface_url'")
    print(config)
    if 'outpath' not in config:
        config['output_dir'] = Path(config_path)
        print('adding outpath')
        logger.warning(f'adding default outpath {config["output_dir"]} to config')

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