"""Preprocess the olmomix-corpus for word2vec training, shard by shard
"""
from utils import get_token
from sentence_splitter import split_sentences
from huggingface_hub import login
from loguru import logger
import numpy as np
import json
import os
from pathlib import Path
from functools import lru_cache




import time
HF_TOKEN = get_token('HF_TOKEN')
login(HF_TOKEN)
INDEX='v4_olmo-2-0325-32b-instruct_llama'
TOKENIZER_NAME='meta-llama/Llama-2-7b-hf'
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)


def process_text_batch(batch, total_offset=0, split_processes=1, chunk_size=1000):
    sentences = split_sentences(batch, n_processes=split_processes, chunk_size=chunk_size)

    if not sentences:
        return np.array([], dtype=np.uint16), np.array([total_offset], dtype=np.int64)
    
    tokenizer = get_tokenizer()

    batch_ids = tokenizer(
        sentences, add_special_tokens=False, padding=False, return_attention_mask=False
    )
    ids_list = batch_ids["input_ids"]
    flat_ids = np.fromiter((tid for ids in ids_list for tid in ids), dtype=np.uint16)
    sentence_lengths = np.fromiter((len(ids) for ids in ids_list), dtype=np.int64)
    offsets = np.concatenate(([total_offset], total_offset + np.cumsum(sentence_lengths)))
    return flat_ids, offsets


def shards_from_corpus(
    corpus,
    out_dir="shards",
    max_tokens_per_shard=int(5 * 1024**3 / 2),  # ≈ 2.68B uint16 tokens (~5 GB)
    split_processes=1,
    chunk_size=1000
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for batch in corpus.iter_batches():
        if batch is None:
            continue

        logger.info(f"⚙️ Starting batch {batch['batch_name']} ({len(batch['files'])} files)...")
        gen = batch["gen"]()

        shard_index = 0
        ids_arr = np.empty(max_tokens_per_shard, dtype=np.uint16)

        offset_capacity = 50_000_000
        offset_arr = np.empty(offset_capacity, dtype=np.int64)
        offset_arr[0] = 0
        offset_pos = 1
        pos = 0
        total_tokens = 0

        def flush_shard():
            nonlocal shard_index, pos, offset_pos, total_tokens
            if pos == 0:
                return
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

        # Iterate through generator
        batch_idx = 0
        for text_batch in gen:
            batch_idx += 1
            t0 = time.perf_counter()

            try:
                new_ids, new_offsets = process_text_batch(
                    text_batch,
                    total_offset=offset_arr[offset_pos - 1],
                    split_processes=split_processes,
                    chunk_size=chunk_size
                )
            except Exception as e:
                logger.warning(f"Error processing document in {batch['batch_name']}: {e}")
                continue

            elapsed = time.perf_counter() - t0
            logger.info(
                f"⏱️ Batch {batch_idx}: split_processes={split_processes}, "
                f"{len(text_batch):,} docs → {len(new_ids):,} tokens in {elapsed:.2f}s "
                f"({len(new_ids) / max(elapsed,1e-6):,.0f} tok/s)"
            )

            n_ids = len(new_ids)
            if pos + n_ids >= max_tokens_per_shard:
                flush_shard()

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

        flush_shard()

        logger.info(f"✅ Finished batch {batch['batch_name']}")