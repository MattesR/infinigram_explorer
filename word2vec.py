import time
import click
import yaml
from gensim.models import Word2Vec
from loguru import logger
import logging
from datetime import timedelta
import sys
import multiprocessing
from pathlib import Path
from utils import get_token


class LoguruHandler(logging.Handler):
    def emit(self, record):
        # Use Loguru to handle logs from gensim
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())
logger.add("word2vec_training.log", level="INFO")


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
        vocab = [i for i,_ in enumerate(tokenizer.get_vocab().keys())]
        logger.info(f'built vocab from tokenizer, length {len(vocab)}')
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            batch_words=50000,
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
    Logs warnings for each defaulted value and constructs the corpus object
    depending on 'corpus_type' (hf_stream, hf_buffered, token_shards).
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
        "corpus_type": "hf_stream",   # new unified corpus selector
        "revision": None,
        "use_features": False,
        "max_files_per_stream": 4,
        "disable_caching": False,
        "total_examples": 3_080_000_000,
        "shard_root": None,            # for token_shards mode
        "batch_size": 1
    }

    config_path = Path(config_path)
    with open(config_path / 'config.yml', 'r') as f:
        user_config = yaml.safe_load(f) or {}

    config = user_config.copy()

    # === Fill defaults ===
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value
            logger.warning(f"Config key '{key}' missing; using default value: {default_value}")

    # === Tokenizer handling ===
    if config["tokenizer_name"]:
        logger.info(f"Loading tokenizer: {config['tokenizer_name']}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], token=get_token("HF_TOKEN"))
        config["tokenizer"] = tokenizer
    else:
        config["tokenizer"] = None

    # === Corpus setup ===
    corpus_type = config["corpus_type"].lower()
    logger.info(f"Initializing corpus of type '{corpus_type}'")

    if corpus_type in ("hf_stream", "hf_buffered"):
        if "huggingface_url" not in user_config:
            raise ValueError("Config must contain 'huggingface_url' when using hf_stream or hf_buffered corpus types")

        hf_kwargs = {
            "dataset_name": user_config["huggingface_url"],
            "split": user_config.get("split", "train"),
            "text_field": user_config.get("text_field", "text"),
            "subset": user_config.get("subset"),
            "max_sentences": user_config.get("max_sentences", None),
            "revision": user_config.get("revision", None),
            "use_features": config.get("use_features", False),
            "max_files_per_stream": config.get("max_files_per_stream", 4),
            "disable_caching": config.get("disable_caching", False),
            "tokenizer": config.get("tokenizer", None),
        }

        if corpus_type == "hf_buffered":
            from hf_corpus import HFCorpusBuffered
            config["corpus_iterable"] = HFCorpusBuffered(**hf_kwargs)
        else:
            from hf_corpus import HFStreamingCorpus
            config["corpus_iterable"] = HFStreamingCorpus(**hf_kwargs)

    elif corpus_type == "token_shards":
        shard_root = config.get("shard_root")
        if not shard_root:
            raise ValueError("For corpus_type='token_shards', 'shard_root' must be specified in config.yml")
        from token_corpus import TokenCorpus
        config["corpus_iterable"] = TokenCorpus(shard_root, batch_size=config['batch_size'], max_sentences=user_config.get("max_sentences", None))
    else:
        raise ValueError(f"Invalid corpus_type '{corpus_type}'. Must be one of: hf_stream, hf_buffered, token_shards")

    # === Output path ===
    if "outpath" not in config:
        config["output_dir"] = Path(config_path)
        logger.warning(f"Adding default output_dir={config['output_dir']} to config")

    # === Worker handling ===
    if isinstance(config["workers"], str):
        total_cpus = multiprocessing.cpu_count()
        workers_str = config["workers"].lower()
        if workers_str == "all":
            config["workers"] = max(1, total_cpus - 1)
            logger.info(f"Setting workers to {config['workers']} (all cores minus one)")
        elif workers_str == "max":
            config["workers"] = max(1, total_cpus)
            logger.info(f"Setting workers to {config['workers']} (all cores)")
        else:
            raise ValueError(f"Malformed workers count, must be int, 'ALL', or 'MAX'; got '{config['workers']}'")

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