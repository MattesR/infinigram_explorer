from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Seq:
    token_ids: Tuple[int, ...]
    logprob: float  # cumulative log P(generated_tokens | prompt)
    similarity: float



def _top_w_next_tokens(
    engine,
    prompt_ids: List[int],
    w: int,
    *,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
    banned_token_ids: Optional[set[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Returns [(token_id, prob), ...] for the top-w tokens by prob.
    """
    next_token_method = engine.infgram_ntd if use_infgram else engine.ntd
    kwargs = {} if max_support is None else {"max_support": max_support}
    result_dict = next_token_method(prompt_ids=prompt_ids, **kwargs)

    token_dict = result_dict.get("result_by_token_id", {})  # {token_id: {'prob': ..., ...}, ...}
    if not dist:
        return []

    banned = banned_token_ids or set()

    items = sorted(
        (
            (int(token_id), float(info["prob"]))
            for token_id, info in token_dict.items()
            if float(info.get("prob", 0.0)) > 0.0 and int(token_id) not in banned
        ),
        key=lambda x: x[1],
        reverse=True, ## sort in ascending order
    )
    return items[:w] ## return top_w tokens


def expand_wk(
    engine,
    tokenizer,
    query: str,
    *,
    w: int = 5,
    k: int = 5,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
    ban_eos: bool = False,
) -> List[dict]:
    """
    Tree expansion (no pruning). Expands query by top-w next tokens for k steps.
    WARNING: exponential in k (approximately w**k).
    """
    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    frontier: List[Seq] = [Seq(token_ids=tuple(prompt_ids), logprob=0.0)]

    banned = set()
    if ban_eos and tokenizer.eos_token_id is not None:
        banned.add(int(tokenizer.eos_token_id))

    for _ in range(k):
        new_frontier: List[Seq] = []
        for seq in frontier:
            nexts = _top_w_next_tokens(
                engine,
                list(seq.token_ids),
                w,
                use_infgram=use_infgram,
                max_support=max_support,
                banned_token_ids=banned,
            )
            for tok_id, p in nexts:
                lp = -float("inf") if p <= 0.0 else math.log(p)
                new_frontier.append(Seq(token_ids=seq.token_ids + (tok_id,), logprob=seq.logprob + lp))
        frontier = new_frontier
        if not frontier:
            break

    results: List[dict] = []
    for seq in frontier:
        full_ids = list(seq.token_ids)
        gen_ids = full_ids[len(prompt_ids) :]
        results.append(
            {
                "token_ids": full_ids,
                "gen_token_ids": gen_ids,
                "text": tokenizer.decode(full_ids),
                "gen_text": tokenizer.decode(gen_ids),
                "logprob": seq.logprob,
            }
        )
    results.sort(key=lambda r: r["logprob"], reverse=True)
    return results


# -------------------------
# Beam search with pruning
# -------------------------

Evaluator = Callable[[Tuple[int, ...]], float]


def beam_search_delayed_prune(
    engine,
    tokenizer,
    query: str,
    *,
    w: int = 5,                 # branching factor: top-w per hypothesis per token step
    k: int = 1,                 # prune interval in tokens (k=1 = classic beam search)
    B: int = 5,                 # beam size after pruning
    d: int = 20,                # total generated tokens
    use_infgram: bool = True,
    max_support: Optional[int] = None,
    evaluator: Optional[Evaluator] = None,
    alpha_logprob: float = 1.0, # weight for logprob
    beta_eval: float = 1.0,     # weight for evaluator score
    intermediate_cap: Optional[int] = None,  # optional cap during the k-step expansion (prevents blow-ups)
    ban_eos: bool = False,
    verbose: bool = False,
) -> Tuple[List[dict], int]:
    """
    Beam search that expands for `k` tokens, then prunes to `B` hypotheses, repeating until `d` tokens generated.

    Returns:
      (results, candidates_generated)

    results: list of dicts in the same format as expand_wk (for the final beam)
    candidates_generated: total number of child hypotheses created across the run
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if w <= 0 or B <= 0 or d < 0:
        raise ValueError("w and B must be >= 1; d must be >= 0")

    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    beam: List[Seq] = [Seq(token_ids=tuple(prompt_ids), logprob=0.0)]
    total_created = 0

    banned = set()
    if ban_eos and tokenizer.eos_token_id is not None:
        banned.add(int(tokenizer.eos_token_id))

    def score(seq: Seq) -> float:
        if evaluator is None:
            return seq.logprob
        return alpha_logprob * seq.logprob + beta_eval * float(evaluator(seq.token_ids))

    generated = 0
    while generated < d and beam:
        chunk = min(k, d - generated)
        frontier = beam

        # Expand chunk steps WITHOUT pruning (or with a loose intermediate cap)
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
                    new_frontier.append(Seq(token_ids=seq.token_ids + (tok_id,), logprob=seq.logprob + lp))

            total_created += len(new_frontier)

            if verbose:
                mb = 0 if min_branch == 10**9 else min_branch
                print(f"[expand {step}/{chunk}] in={len(frontier)} out={len(new_frontier)} dead={dead} min_branch={mb}")

            frontier = new_frontier
            if not frontier:
                break

            # Optional safety: keep only the best M (by cheap score) during wide expansion
            if intermediate_cap is not None and len(frontier) > intermediate_cap:
                frontier.sort(key=score, reverse=True)
                frontier = frontier[:intermediate_cap]
                if verbose:
                    print(f"[intermediate_cap] kept={len(frontier)}")

        if not frontier:
            beam = []
            break

        # Prune to beam size B
        frontier.sort(key=score, reverse=True)
        beam = frontier[:B]
        if verbose:
            print(f"[prune] kept={len(beam)}")

        generated += chunk

    # Format final beam results
    results: List[dict] = []
    for seq in beam:
        full_ids = list(seq.token_ids)
        gen_ids = full_ids[len(prompt_ids) :]
        results.append(
            {
                "token_ids": full_ids,
                "gen_token_ids": gen_ids,
                "text": tokenizer.decode(full_ids),
                "gen_text": tokenizer.decode(gen_ids),
                "logprob": seq.logprob,
            }
        )
    results.sort(key=lambda r: r["logprob"], reverse=True)
    return results, total_created


def load_default_engine():
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        add_eos_token=False,
    )
    engine = InfiniGramEngine(index_dir="/home/mruc/first_index/", eos_token_id=tokenizer.eos_token_id)
    return tokenizer, engine
