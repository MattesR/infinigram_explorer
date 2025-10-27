import re
import multiprocessing as mp
from loguru import logger

# ============================================================
# Precompiled regex for English sentence splitting
# ============================================================
_SENTENCE_BOUNDARY_RE = re.compile(
    r'([.!?])(?:\s+|\n+)(?=[A-Z0-9])'
)


# ============================================================
# Single-process splitter
# ============================================================
def _split_single(text: str) -> list[str]:
    """
    Split English text into sentences using simple regex rules.

    Rules:
    - Splits on '.', '!', or '?' followed by whitespace/newline and a capital letter or digit.
    - Keeps the punctuation with the sentence.
    - Handles newlines as potential breaks.
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize whitespace (collapse multiple spaces)
    text = re.sub(r'\s+', ' ', text.strip())

    # Split text into sentences, keeping the delimiters
    parts = _SENTENCE_BOUNDARY_RE.split(text)

    # Reconstruct full sentences by merging [sentence, punctuation]
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()
        if i + 1 < len(parts):
            sentence += parts[i + 1]  # append punctuation
        if sentence:
            sentences.append(sentence.strip())

    return sentences


# ============================================================
# Multiprocessing-enabled version
# ============================================================
def split_sentences(docs, n_processes: int | None = None, chunk_size: int = 1000) -> list[str]:
    """
    Split a list of documents into sentences — optionally in parallel.

    Args:
        docs: list of text documents
        n_processes: number of processes (if 1 or None → single-threaded)
        chunk_size: how many docs per worker batch

    Returns:
        A flat list of sentences (same as your old implementation)
    """
    if not docs:
        return []

    # Default to CPU count - 1, but disable multiprocessing if only one process
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    # Fast path: single process
    if n_processes <= 1 or len(docs) < 2:
        sentences = []
        for d in docs:
            sentences.extend(_split_single(d))
        return sentences

    with mp.Pool(processes=n_processes, maxtasksperchild=100) as pool:
        results = pool.map(_split_worker, _chunk_list(docs, chunk_size))

    # Flatten all results into one sentence list
    sentences = [s for sublist in results for s in sublist]
    return sentences


def _split_worker(chunk):
    sentences = []
    for doc in chunk:
        try:
            sentences.extend(_split_single(doc))
        except Exception as e:
            logger.warning(f"Regex split failed for doc: {e}")
    return sentences


def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]