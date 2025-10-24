import numpy as np
from pathlib import Path
from loguru import logger

class TokenCorpus:
    """
    Iterable corpus over tokenized shards stored as .npy arrays.
    Loads one shard (ids + offsets) fully into memory at a time for fast iteration.
    """

    def __init__(self, path, limit=None, log_every=10000):
        """
        Args:
            path (str | Path): Directory containing shard files.
            limit (int, optional): Max number of sentences to yield (for debugging).
            log_every (int): Log progress every N shards.
        """
        self.path = Path(path)
        self.limit = limit
        self.log_every = log_every

        self.shards = sorted(self.path.glob("**/*_ids.npy"))
        if not self.shards:
            raise FileNotFoundError(f"No **/*_ids.npy shards found in {self.path}")

    def __iter__(self):
        sentence_count = 0

        for i, ids_file in enumerate(self.shards):
            shard_prefix = ids_file.stem.replace("_ids", "")
            offsets_file = self.path / f"{shard_prefix}_offsets.npy"

            if not offsets_file.exists():
                logger.warning(f"⚠️ Missing offsets for {ids_file.name}, skipping shard")
                continue

            try:
                # Fully load arrays (faster, less I/O overhead)
                ids = np.load(ids_file)
                offsets = np.load(offsets_file)

                if len(offsets) < 2:
                    logger.warning(f"⚠️ Shard {shard_prefix} has no valid sentences")
                    continue

                num_sentences = len(offsets) - 1
                logger.info(f"📦 Loaded shard {shard_prefix} ({num_sentences:,} sentences, {len(ids):,} tokens)")

                # Yield one sentence at a time (list of ints)
                for start, end in zip(offsets[:-1], offsets[1:]):
                    yield ids[start:end].tolist()
                    sentence_count += 1

                    if self.limit and sentence_count >= self.limit:
                        logger.info(f"Reached limit ({self.limit}) → stopping early.")
                        return

                if (i + 1) % self.log_every == 0:
                    logger.info(f"✅ Processed {i+1}/{len(self.shards)} shards ({sentence_count:,} sentences total)")

                # Free memory explicitly
                del ids, offsets

            except Exception as e:
                logger.error(f"💥 Error processing shard {shard_prefix}: {e}")
                continue

        logger.info(f"🎯 Finished iterating {sentence_count:,} total sentences from {len(self.shards)} shards")