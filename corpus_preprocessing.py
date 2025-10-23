"""Preprocess the olmomix-corpus for word2vec training, shard by shard
"""
from utils import get_token, split_sentences

from transformers import AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm
from pathlib import Path, PurePosixPath
from gensim.models import Word2Vec
from loguru import logger
from word2vec import HFStreamingCorpus
import numpy as np
import json

HF_TOKEN = get_token('HF_TOKEN')
login(HF_TOKEN)
INDEX='v4_olmo-2-0325-32b-instruct_llama'
TOKENIZER_NAME='meta-llama/Llama-2-7b-hf'
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)


def get_token_ids(text, tokenizer=TOKENIZER):
    return tokenizer(text, add_special_tokens=False, padding=False,return_attention_mask=False)['input_ids']

def process_text(text, tokenizer=TOKENIZER, total_offset=0):
    ids=[]
    offsets = [total_offset]
    total_offset = total_offset
    sentences = split_sentences(text)
    for sentence in sentences:
        if sentence:
            tokens = get_token_ids(sentence, tokenizer)
            ids.extend(tokens)
            total_offset += len(tokens)
            offsets.append(total_offset)
    return ids, offsets


def save_shard(batch, ids, offsets, shard_index, total_tokens, out_path):
    """Write one shard (IDs, offsets, metadata) to disk."""
    shard_prefix = out_path / f"{batch['batch_name']}_shard_{shard_index:05d}"

    ids_arr = np.array(ids, dtype=np.uint16)
    offs_arr = np.array(offsets, dtype=np.int64)

    np.save(f"{shard_prefix}_ids.npy", ids_arr)
    np.save(f"{shard_prefix}_offsets.npy", offs_arr)

    # Compute final metadata (calls the lambda)
    metadata = batch["meta"]()
    metadata.update({
        "total_tokens": int(total_tokens),
        "shard_index": shard_index,
        "ids_path": f"{shard_prefix}_ids.npy",
        "offsets_path": f"{shard_prefix}_offsets.npy",
    })

    with open(f"{shard_prefix}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"🧩 Saved {shard_prefix.name} "
        f"({len(ids_arr):,} tokens, {metadata['num_examples']} docs)"
    )


def shards_from_corpus(corpus, out_dir="shards", tokenizer=TOKENIZER, max_batch_tokens=None, max_tokens_per_shard=int(5 * 1024**3 / 2)):
    """stream the corpus as batches and create shards from it
    """
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    for batch in corpus.iter_batches():  
        if batch is None:
            continue
        logger.info(f"Starting batch {batch['batch_name']} with {len(batch['files'])} source files...")
        shard_index = 0      
        gen = batch["gen"]()
        batch_ids = []
        batch_offsets = []
        total_tokens = 0
        for text in gen:
            try:
                new_ids, new_offsets = process_text(text, total_offset=batch_offsets[-1] if batch_offsets else 0)
                batch_ids.extend(new_ids)
                batch_offsets.extend(new_offsets)
                total_tokens += len(new_ids)

                if len(batch_ids) >= max_tokens_per_shard:
                    save_shard(
                        batch, batch_ids, batch_offsets,
                        shard_index, total_tokens, out_path
                    )
                    shard_index += 1
                    batch_ids = []
                    batch_offsets = [0]
                    total_tokens = 0

            except Exception as e:
                logger.warning(f"Error processing document in {batch['batch_name']}: {e}")
                continue
            # rest of the data 
            if batch_ids:
                save_shard(
                    batch, batch_ids, batch_offsets,
                    shard_index, total_tokens, out_path
                )
            logger.info(f"✅ Finished batch {batch['batch_name']}")
        batch_ids = np.array(batch_ids, dtype=np.int32)
        batch_offsets = np.array(batch_offsets, dtype=np.int64)

        shard_prefix = Path(out_dir) / f"{batch['batch_name']}"
        np.save(f"{shard_prefix}_ids.npy", batch_ids)
        np.save(f"{shard_prefix}_offsets.npy", batch_offsets)

        # This calls the lambda function which is the value of the dict field meta 💡💡💡
        metadata = batch["meta"]() 
        metadata = batch["meta"]() 
        metadata.update({
        "total_tokens": total_tokens,
        "ids_path": f"{shard_prefix}_ids.npy",
        "offsets_path": f"{shard_prefix}_offsets.npy",
    })
        logger.info(f"Finished batch {batch['batch_name']} with {metadata['num_examples']}")
        with open(f"{shard_prefix}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
        
