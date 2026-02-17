from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
import numpy as np

Evaluator = Callable[[Tuple[int, ...]], float]


@dataclass(frozen=True)
class Seq:
    token_ids: Tuple[int, ...]
    logprob: float                 # cumulative log P(generated_tokens | prompt)
    eval_score: float = 0.0        # score returned by evaluator (higher = better)


def _top_w_next_tokens(
    engine,
    prompt_ids: List[int],
    w: int,
    *,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
    banned_token_ids: Optional[set[int]] = None,
) -> List[Tuple[int, float]]:
    next_token_method = engine.infgram_ntd if use_infgram else engine.ntd
    kwargs = {} if max_support is None else {"max_support": max_support}
    result_dict = next_token_method(prompt_ids=prompt_ids, **kwargs)

    token_dict = result_dict.get("result_by_token_id", {})
    if not token_dict:
        return []

    banned = banned_token_ids or set()

    items = sorted(
        (
            (int(token_id), float(info.get("prob", 0.0)))
            for token_id, info in token_dict.items()
            if float(info.get("prob", 0.0)) > 0.0 and int(token_id) not in banned
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    return items[:w]

def make_semantic_cosine_evaluator(
    tokenizer,
    st_model,                 # preloaded SentenceTransformer
    reference_text: str,
    *,
    normalize: bool = True,
    prefix_text: str = "",    # optional: include a fixed prefix (e.g., original anchor sentence)
) -> Callable[[Tuple[int, ...]], float]:
    """
    Returns evaluator(token_ids)->float where higher is better (cosine similarity).
    This evaluator decodes token_ids to text using your tokenizer.

    Note: This is the simplest form (no batching). For speed, prefer the batched version below.
    """
    # Precompute reference embedding once
    ref_emb = st_model.encode(reference_text, normalize_embeddings=normalize)
    ref_emb = np.asarray(ref_emb, dtype=np.float32)

    def evaluator(token_ids: Tuple[int, ...]) -> float:
        text = prefix_text + tokenizer.decode(list(token_ids))
        emb = st_model.encode(text, normalize_embeddings=normalize)
        emb = np.asarray(emb, dtype=np.float32)

        if normalize:
            # If normalized, cosine = dot
            return float(np.dot(emb, ref_emb))
        # Otherwise compute cosine manually
        denom = (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
        return float(np.dot(emb, ref_emb) / denom) if denom > 0 else float("-inf")

    return evaluator


def score_semantic_batch(
    tokenizer,
    st_model,
    reference_text: str,
    token_id_seqs: List[Tuple[int, ...]],
    *,
    normalize: bool = True,
    prefix_text: str = "",
) -> List[float]:
    """
    Scores a list of hypotheses in one batch call to the sentence-transformer.
    Returns a list of cosine similarities aligned with token_id_seqs.
    """
    ref_emb = st_model.encode(reference_text, normalize_embeddings=normalize)
    ref_emb = np.asarray(ref_emb, dtype=np.float32)

    texts = [prefix_text + tokenizer.decode(list(ids)) for ids in token_id_seqs]
    embs = st_model.encode(texts, normalize_embeddings=normalize, batch_size=64, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)

    if normalize:
        # cosine = dot product when embeddings are normalized
        return [float(np.dot(e, ref_emb)) for e in embs]

    ref_norm = np.linalg.norm(ref_emb)
    out = []
    for e in embs:
        denom = (np.linalg.norm(e) * ref_norm)
        out.append(float(np.dot(e, ref_emb) / denom) if denom > 0 else float("-inf"))
    return out


def beam_search_delayed_prune(
    engine,
    tokenizer,
    query: str,
    *,
    w: int = 5,
    k: int = 1,
    B: int = 5,
    d: int = 20,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
    evaluator: Optional[Evaluator] = None,   # must return a float; higher = better
    intermediate_cap: Optional[int] = None,  # optional safety cap during expansion
    ban_eos: bool = False,
    verbose: bool = False,
) -> Tuple[List[dict], int]:
    """
    Beam search that expands for `k` tokens, then prunes to `B` hypotheses, repeating until `d` tokens generated.

    Pruning is based on Seq.eval_score (from evaluator). If evaluator is None, falls back to logprob.

    Returns:
      (results, candidates_generated)
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if w <= 0 or B <= 0 or d < 0:
        raise ValueError("w and B must be >= 1; d must be >= 0")

    prompt_ids = tokenizer.encode(query, add_special_tokens=False)

    # If evaluator is provided, initialize eval_score for the initial prompt.
    init_eval = float(evaluator(tuple(prompt_ids))) if evaluator is not None else 0.0
    beam: List[Seq] = [Seq(token_ids=tuple(prompt_ids), logprob=0.0, eval_score=init_eval)]

    banned = set()
    if ban_eos and tokenizer.eos_token_id is not None:
        banned.add(int(tokenizer.eos_token_id))

    def rank_score(s: Seq) -> float:
        # This is the ONLY score used for pruning.
        # If no evaluator, use logprob so the function remains usable.
        return s.eval_score if evaluator is not None else s.logprob

    total_created = 0
    generated = 0

    while generated < d and beam:
        chunk = min(k, d - generated)
        frontier = beam

        # Expand chunk steps (optionally with intermediate capping)
        for step in range(1, chunk + 1):
            new_frontier: List[Seq] = []
            dead = 0
            min_branch = 10**9

            for seq in frontier:
                nexts = _top_w_next_tokens(
                    engine,
                    list(seq.token_ids),
                    w,
                    use_infgram=use_infgram,
                    max_support=max_support,
                    banned_token_ids=banned,
                )
                b = len(nexts)
                min_branch = min(min_branch, b)
                if b == 0:
                    dead += 1
                    continue

                for tok_id, p in nexts:
                    lp = -float("inf") if p <= 0.0 else math.log(p)
                    child_ids = seq.token_ids + (tok_id,)

                    # Evaluate child (only place eval_score changes)
                    child_eval = float(evaluator(child_ids)) if evaluator is not None else 0.0

                    new_frontier.append(
                        Seq(token_ids=child_ids, logprob=seq.logprob + lp, eval_score=child_eval)
                    )

            total_created += len(new_frontier)

            if verbose:
                mb = 0 if min_branch == 10**9 else min_branch
                print(f"[expand {step}/{chunk}] in={len(frontier)} out={len(new_frontier)} dead={dead} min_branch={mb}")

            frontier = new_frontier
            if not frontier:
                break

            # Safety cap during the wide part (kept by eval_score if evaluator exists)
            if intermediate_cap is not None and len(frontier) > intermediate_cap:
                frontier.sort(key=rank_score, reverse=True)
                frontier = frontier[:intermediate_cap]
                if verbose:
                    print(f"[intermediate_cap] kept={len(frontier)}")

        if not frontier:
            beam = []
            break

        # Prune to B using evaluator score
        frontier.sort(key=rank_score, reverse=True)
        beam = frontier[:B]
        if verbose:
            top = beam[0].eval_score if evaluator is not None else beam[0].logprob
            print(f"[prune] kept={len(beam)} top_score={top}")

        generated += chunk

    # Format final beam results
    results: List[dict] = []
    for seq in beam:
        full_ids = list(seq.token_ids)
        gen_ids = full_ids[len(prompt_ids):]
        results.append(
            {
                "token_ids": full_ids,
                "gen_token_ids": gen_ids,
                "text": tokenizer.decode(full_ids),
                "gen_text": tokenizer.decode(gen_ids),
                "logprob": seq.logprob,
                "eval_score": seq.eval_score,
            }
        )

    # Sort by eval_score (or logprob if evaluator missing)
    results.sort(key=lambda r: (r["eval_score"] if evaluator is not None else r["logprob"]), reverse=True)
    return results, total_created


def load_default_engine():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
    engine = InfiniGramEngine(index_dir="/home/mruc/first_index/", eos_token_id=tokenizer.eos_token_id)
    return tokenizer, engine
