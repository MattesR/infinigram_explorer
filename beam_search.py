from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# infini-gram docs:
# - InfiniGramEngine(index_dir=..., eos_token_id=...)
# - engine.infgram_ntd(prompt_ids=[...]) -> {'result_by_token_id': {tok: {'prob': ...}, ...}, ...}
# :contentReference[oaicite:0]{index=0}


@dataclass(frozen=True)
class Seq:
    token_ids: Tuple[int, ...]
    logprob: float  # cumulative log P(tokens_added | prompt)


def _top_w_next_tokens(
    engine,
    prompt_ids: List[int],
    w: int,
    *,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    Returns [(token_id, prob), ...] for the top-w tokens by prob.
    """
    if use_infgram:
        out = engine.infgram_ntd(prompt_ids=prompt_ids, **({} if max_support is None else {"max_support": max_support}))
    else:
        out = engine.ntd(prompt_ids=prompt_ids, **({} if max_support is None else {"max_support": max_support}))

    dist = out.get("result_by_token_id", {})  # {token_id: {'prob': ..., 'cont_cnt': ...}, ...}
    if not dist:
        return []

    # sort by probability descending and take top w
    items = sorted(
        ((tok, info["prob"]) for tok, info in dist.items() if info.get("prob", 0.0) > 0.0),
        key=lambda x: x[1],
        reverse=True,
    )
    return items[:w]


def expand_wk(
    engine,
    tokenizer,
    query: str,
    *,
    w: int = 5,
    k: int = 5,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
) -> List[dict]:
    """
    Expands a string query by taking top-w next tokens at each node for k steps.
    If all nodes have >= w continuations, you will get exactly w**k sequences (e.g., 3125 for w=5,k=5).

    Returns a list of dicts with:
      - token_ids: full token id sequence (prompt + generated)
      - gen_token_ids: generated token ids only
      - text: decoded full text
      - gen_text: decoded generated suffix
      - logprob: cumulative log probability of the generated suffix
    """
    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    frontier: List[Seq] = [Seq(token_ids=tuple(prompt_ids), logprob=0.0)]

    for _step in range(k):
        new_frontier: List[Seq] = []
        for seq in frontier:
            nexts = _top_w_next_tokens(
                engine,
                list(seq.token_ids),
                w,
                use_infgram=use_infgram,
                max_support=max_support,
            )
            for tok_id, p in nexts:
                # guard against log(0)
                lp = -float("inf") if p <= 0.0 else math.log(p)
                new_frontier.append(
                    Seq(token_ids=seq.token_ids + (tok_id,), logprob=seq.logprob + lp)
                )
        frontier = new_frontier
        if not frontier:
            break  # no continuations anywhere

    # Format results
    results: List[dict] = []
    for seq in frontier:
        full_ids = list(seq.token_ids)
        gen_ids = full_ids[len(prompt_ids):]
        results.append(
            {
                "token_ids": full_ids,
                "gen_token_ids": gen_ids,
                "text": tokenizer.decode(full_ids),
                "gen_text": tokenizer.decode(gen_ids),
                "logprob": seq.logprob,
            }
        )

    # Optional: sort best-first by logprob (highest = best)
    results.sort(key=lambda r: r["logprob"], reverse=True)
    return results


# Example usage:
#
# from infini_gram.engine import InfiniGramEngine
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Llama-2-7b-hf",
#     add_bos_token=False,
#     add_eos_token=False,
# )
# engine = InfiniGramEngine(index_dir="index/v4_pileval_llama", eos_token_id=tokenizer.eos_token_id)
#
# seqs = expand_wk(engine, tokenizer, "natural language", w=5, k=5, use_infgram=True, max_support=1000)
# print(len(seqs))          # often 3125 if branching never collapses
# print(seqs[0]["text"])    # best continuation by cumulative logprob
