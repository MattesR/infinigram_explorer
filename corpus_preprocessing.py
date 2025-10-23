"""Preprocess the olmomix-corpus for word2vec training, shard by shard
"""
from utils import get_token, split_sentences, parallel_split_sentences

from transformers import AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm
from pathlib import Path, PurePosixPath
from gensim.models import Word2Vec
from loguru import logger
from word2vec import HFStreamingCorpus
import numpy as np
import json
import multiprocessing as mp
import os

HF_TOKEN = get_token('HF_TOKEN')
login(HF_TOKEN)
INDEX='v4_olmo-2-0325-32b-instruct_llama'
TOKENIZER_NAME='meta-llama/Llama-2-7b-hf'
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_text_batch(batch, tokenizer=TOKENIZER, total_offset=0, split_processes=1):
    if split_processes ==1:
        sentences = [s for doc in batch if doc for s in split_sentences(doc)]
        if not sentences:
            return np.array([], dtype=np.uint16), np.array([total_offset], dtype=np.int64)
    else:
        sentences = parallel_split_sentences(batch, n_processes=split_processes)

    batch_ids = tokenizer(sentences, add_special_tokens=False, padding=False, return_attention_mask=False)
    ids_list = batch_ids['input_ids']
    flat_ids = np.fromiter((id for ids in ids_list for id in ids),dtype=np.uint16)
    sentence_lengths = np.fromiter((len(ids) for ids in ids_list), dtype=np.int64)
    offsets = np.concatenate(([total_offset], total_offset + np.cumsum(sentence_lengths)))
    return flat_ids, offsets


def shards_from_corpus(
    corpus,
    out_dir="shards",
    tokenizer=TOKENIZER,
    max_tokens_per_shard=int(5 * 1024**3 / 2),  # ≈ 2.68B uint16 tokens (~5 GB)
    split_processes=1, 
):
    """
    Stream the corpus batch by batch, tokenize, and write fixed-size (~5 GB) shards to disk.
    Uses preallocated NumPy buffers for high efficiency.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for batch in corpus.iter_batches():
        if batch is None:
            continue

        logger.info(f"Starting batch {batch['batch_name']} with {len(batch['files'])} source files...")

        gen = batch["gen"]()
        shard_index = 0

        # Preallocate large buffers for one shard
        ids_arr = np.empty(max_tokens_per_shard, dtype=np.uint16)

        offset_capacity = 50_000_000  # start with ~400 MB offsets
        offset_arr = np.empty(offset_capacity, dtype=np.int64)
        offset_arr[0] = 0
        offset_pos = 1
        pos = 0
        offset_pos = 1  # current position in offsets array
        total_tokens = 0

        def flush_shard():
            """Write current shard to disk and reset buffers."""
            nonlocal shard_index, pos, offset_pos, total_tokens
            if pos == 0:
                return  # nothing to write

            shard_prefix = out_path / f"{batch['batch_name']}_shard_{shard_index:05d}"
            np.save(f"{shard_prefix}_ids.npy", ids_arr[:pos])
            np.save(f"{shard_prefix}_offsets.npy", offset_arr[:offset_pos])

            meta = batch["meta"]()
            meta.update({
                "total_tokens": int(total_tokens),
                "shard_index": shard_index,
                "ids_path": f"{shard_prefix}_ids.npy",
                "offsets_path": f"{shard_prefix}_offsets.npy",
            })
            with open(f"{shard_prefix}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(
                f"🧩 Saved {shard_prefix.name}: "
                f"{pos:,} tokens, {meta['num_examples']} docs"
            )

            shard_index += 1
            pos = 0
            offset_pos = 1
            offset_arr[0] = 0
            total_tokens = 0

        # Iterate through the streaming batch
        for text_batch in gen:
            try:
                new_ids, new_offsets = process_text_batch(
                    text_batch, total_offset=offset_arr[offset_pos - 1], split_processes=split_processes
                )
                n_ids = len(new_ids)
                if pos + n_ids >= max_tokens_per_shard:
                    # flush current shard before overflow
                    flush_shard()

                # copy into preallocated arrays
                ids_arr[pos:pos + n_ids] = new_ids
                n_offsets = len(new_offsets)
                if offset_pos + n_offsets > offset_capacity:
                    new_cap = int(offset_capacity * 1.5)
                    offset_arr = np.resize(offset_arr, new_cap)
                    offset_capacity = new_cap
                offset_arr[offset_pos:offset_pos + n_offsets] = new_offsets
                pos += n_ids
                offset_pos += n_offsets
                total_tokens += n_ids

            except Exception as e:
                logger.warning(f"Error processing document in {batch['batch_name']}: {e}")
                continue

        # flush final partial shard
        flush_shard()

        logger.info(f"✅ Finished batch {batch['batch_name']}")