"""
BLEnD cultural-knowledge experiment — API-based counting.

Hits the public infini-gram API (api.infini-gram.io) to get n-gram counts
for BLEnD cultural questions across all country splits.

Two paths:
  A) No expansion — extract content words from the English question directly.
  B) With expansion — LLM generates KEY_ENTITIES + ASSOCIATED_TERMS.

No local engine needed — only the public API and a HuggingFace tokenizer.

Usage:
    # Path A: no expansion, single split
    python blend_experiment.py --mode no-expansion --splits US

    # Path A: all known splits
    python blend_experiment.py --mode no-expansion --splits all

    # Path A: specific splits
    python blend_experiment.py --mode no-expansion --splits US UK South_Korea

    # Path B: with LLM keyword expansion (requires pre-generated expansions)
    python blend_experiment.py --mode expansion --expansions-path blend_expansions.jsonl --splits US

    # Include combo counts (cross-keyword AND queries — much slower)
    python blend_experiment.py --mode no-expansion --splits US --combos

    # Submit LLM expansion batch for all splits
    python blend_experiment.py --mode submit-expansion --splits all

    # Collect expansion batch results
    python blend_experiment.py --mode collect-expansion --batch-id msgbatch_XXX

    # List available BLEnD splits from HuggingFace
    python blend_experiment.py --mode list-splits
"""

import argparse
import csv
import json
import os
import re
import requests
import time
from collections import defaultdict
from itertools import combinations


# ═══════════════════════════════════════════════════════════════════════
#  Stopwords & question words
# ═══════════════════════════════════════════════════════════════════════

try:
    from keyword_filter import STOPWORDS
except ImportError:
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "shall", "it", "its", "this", "that", "these",
        "those", "what", "which", "who", "whom", "how", "when", "where",
        "why", "not", "no", "nor", "if", "then", "than", "so", "up",
        "out", "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "once",
        "here", "there", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "only", "own", "same", "too",
        "very", "just", "also",
    }

QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "might",
    "many", "much", "often", "long", "far",
    "go", "get", "give", "take", "make", "come", "say", "tell",
    "know", "think", "see", "look", "want", "use", "find",
    "like", "need", "try", "call", "keep", "let", "put",
    "usually", "typically", "generally", "commonly", "mostly",
    "provide", "name",
}

ALL_STOP = STOPWORDS | QUESTION_WORDS

# Aspect names that are structural labels, not keywords themselves
_GENERIC_ASPECTS = {"terms", "content", "words", "keywords", "question_terms"}


# ═══════════════════════════════════════════════════════════════════════
#  Infini-gram API
# ═══════════════════════════════════════════════════════════════════════

API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_olmo-2-0325-32b-instruct_llama"
API_DELAY = 0.5  # seconds between calls (rate-limit protection)


def _api_count(query_ids, index=DEFAULT_INDEX):
    """
    Count n-gram occurrences via the infini-gram public API.

    query_ids is CNF: list of list of list of ints.
    Each outer list is an AND clause; each middle list is an OR group
    of token-id sequences.
    """
    time.sleep(API_DELAY)
    payload = {
        "index": index,
        "query_type": "count",
        "query_ids": query_ids,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        result = resp.json()
        if "error" in result:
            return 0
        return result.get("count", 0)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════
#  Tokenization helpers
# ═══════════════════════════════════════════════════════════════════════

def _tokenize_keyword(keyword, tokenizer):
    """
    Tokenize a keyword into a CNF piece with case variants.

    "Vietnam War" → cnf = [[[ids_original], [ids_lower], [ids_title]]]
    i.e. one AND clause containing OR'd case variants.
    """
    variants = []
    seen = set()
    for v in [keyword, keyword.lower(), keyword.title()]:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            variants.append(ids)
    if not variants:
        return None
    return {
        "description": keyword,
        "keyword": keyword,
        "cnf": [variants],
        "variants": variants,
    }


def _build_pieces(expansion, tokenizer):
    """
    Build tokenized CNF pieces from an expansion dict.

    Expects KEY_ENTITIES (dict of aspect→terms) and ASSOCIATED_TERMS (list).
    For generic aspect names like "terms", the name itself is not treated
    as a keyword (only the listed terms are).
    """
    key_entities = expansion.get("KEY_ENTITIES", {})
    associated = expansion.get("ASSOCIATED_TERMS", expansion.get("ASSOCIATED", []))

    key_pieces = {}
    for name, terms in key_entities.items():
        # Handle nested dict format (lexical/conceptual/referential)
        if isinstance(terms, dict):
            flat = []
            for level in ["lexical", "conceptual", "referential"]:
                flat.extend(terms.get(level, []))
            terms = flat

        # For proper-noun / meaningful aspect names, include the name itself
        # as a keyword; for generic labels like "terms", skip the name.
        if name.lower() in _GENERIC_ASPECTS:
            all_terms = terms if isinstance(terms, list) else []
        else:
            all_terms = [name] + (terms if isinstance(terms, list) else [])

        aspect_pieces = []
        for kw in all_terms:
            piece = _tokenize_keyword(kw, tokenizer)
            if piece:
                piece["source_aspect"] = name
                aspect_pieces.append(piece)
        if aspect_pieces:
            key_pieces[name] = aspect_pieces

    assoc_pieces = []
    for kw in associated:
        piece = _tokenize_keyword(kw, tokenizer)
        if piece:
            assoc_pieces.append(piece)

    return {"key_pieces": key_pieces, "associated": assoc_pieces}


# ═══════════════════════════════════════════════════════════════════════
#  Known BLEnD splits
# ═══════════════════════════════════════════════════════════════════════

BLEND_SPLITS = [
    "Algeria", "Australia", "Brazil", "China", "Ethiopia",
    "France", "Germany", "India", "Iran", "Japan",
    "Kenya", "Nigeria", "South_Korea", "Spain", "Turkey",
    "UK", "US",
]


def list_blend_splits():
    """Try to list available BLEnD splits from HuggingFace, fall back to known list."""
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder("uilab/BLEnD", "annotations")
        splits = list(builder.info.splits.keys())
        print(f"Available splits from HuggingFace ({len(splits)}):")
        for s in sorted(splits):
            n = builder.info.splits[s].num_examples
            print(f"  {s:20s}  {n:>6,d} rows")
        return splits
    except Exception as e:
        print(f"Could not query HuggingFace: {e}")
        print(f"Known splits ({len(BLEND_SPLITS)}):")
        for s in BLEND_SPLITS:
            print(f"  {s}")
        return BLEND_SPLITS


# ═══════════════════════════════════════════════════════════════════════
#  BLEnD data loading
# ═══════════════════════════════════════════════════════════════════════

def load_blend_hf(split: str = "US", max_rows: int = None):
    """Load BLEnD annotations from HuggingFace datasets library."""
    from datasets import load_dataset
    ds = load_dataset("uilab/BLEnD", "annotations", split=split)
    rows = []
    for i, row in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        rows.append({
            "id": str(row["ID"]),
            "question": row["question"],
            "en_question": row["en_question"],
            "annotations": row["annotations"],
            "idks": row["idks"],
        })
    return rows


def load_blend_jsonl(path: str, max_rows: int = None):
    """Load BLEnD from a JSONL file (pre-downloaded)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "id": str(obj.get("ID", obj.get("id", len(rows)))),
                "question": obj.get("question", obj.get("en_question", "")),
                "en_question": obj.get("en_question", obj.get("question", "")),
                "annotations": obj.get("annotations", []),
                "idks": obj.get("idks", 0),
            })
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def load_blend(path_or_split: str = "US", max_rows: int = None):
    """Load BLEnD data — from file if path exists, else from HuggingFace."""
    if os.path.isfile(path_or_split):
        ext = os.path.splitext(path_or_split)[1].lower()
        if ext == ".jsonl":
            return load_blend_jsonl(path_or_split, max_rows)
        elif ext == ".json":
            with open(path_or_split) as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = []
                for obj in data[:max_rows]:
                    rows.append({
                        "id": str(obj.get("ID", obj.get("id", len(rows)))),
                        "question": obj.get("question", obj.get("en_question", "")),
                        "en_question": obj.get("en_question", obj.get("question", "")),
                        "annotations": obj.get("annotations", []),
                        "idks": obj.get("idks", 0),
                    })
                return rows
            raise ValueError(f"Unexpected JSON structure in {path_or_split}")
        else:
            return load_blend_jsonl(path_or_split, max_rows)
    else:
        return load_blend_hf(path_or_split, max_rows)


# ═══════════════════════════════════════════════════════════════════════
#  Path A: No-expansion keyword extraction
# ═══════════════════════════════════════════════════════════════════════

BRIDGE_STOPS = {"of", "the", "de", "la", "le", "del", "van", "von", "di", "da", "and"}


def _extract_proper_phrases(text: str) -> list[str]:
    """
    Extract multi-word proper-noun phrases from text.

    A proper phrase is 2+ adjacent capitalized words, optionally bridged by
    a small stop word ("of", "the", "de", etc.).
    Returns phrases of 2+ meaningful words, longest first, deduplicated.
    """
    clean = re.sub(r"[^\w\s\-]", " ", text)
    words = clean.split()

    def _is_proper(w):
        if not w:
            return False
        return w[0].isupper() or (w[0].isdigit() and len(w) <= 5)

    phrases = []
    i = 0
    while i < len(words):
        if _is_proper(words[i]):
            run = [words[i]]
            j = i + 1
            while j < len(words):
                if _is_proper(words[j]):
                    run.append(words[j])
                    j += 1
                elif (words[j].lower() in BRIDGE_STOPS
                      and j + 1 < len(words)
                      and _is_proper(words[j + 1])):
                    run.append(words[j])
                    run.append(words[j + 1])
                    j += 2
                else:
                    break
            if len([w for w in run if w.lower() not in BRIDGE_STOPS]) >= 2:
                phrases.append(" ".join(run))
            i = j
        else:
            i += 1

    seen = set()
    unique = []
    for p in sorted(phrases, key=len, reverse=True):
        p_low = p.lower()
        if p_low not in seen:
            seen.add(p_low)
            unique.append(p)
    return unique


def extract_keywords_no_expansion(question: str) -> dict:
    """
    Extract content words from question text — no LLM, just stopword removal
    with proper-noun phrase detection.

    Returns a dict in KEY_ENTITIES / ASSOCIATED_TERMS format.
    """
    phrases = _extract_proper_phrases(question)

    text = re.sub(r"[^\w\s\-]", " ", question)
    words = text.split()

    # Find which word indices are consumed by detected phrases
    consumed_indices = set()
    for phrase in phrases:
        pwords = phrase.lower().split()
        for start in range(len(words) - len(pwords) + 1):
            if [words[start + k].lower() for k in range(len(pwords))] == pwords:
                for k in range(len(pwords)):
                    consumed_indices.add(start + k)
                break

    # Single content words not consumed by any phrase
    singles = []
    for i, w in enumerate(words):
        if i in consumed_indices:
            continue
        w_lower = w.lower()
        if w_lower not in ALL_STOP and len(w_lower) > 1:
            singles.append(w)

    if not phrases and not singles:
        by_len = sorted(words, key=len, reverse=True)
        singles = [w for w in by_len[:3] if len(w) > 2]

    # Build KEY_ENTITIES
    key_entities = {}
    for phrase in phrases:
        key_entities[phrase] = []
    if singles:
        key_entities["terms"] = singles  # "terms" is a generic label, not a keyword

    # ASSOCIATED_TERMS
    associated = []
    seen_assoc = set()

    for phrase in phrases:
        for pw in phrase.split():
            pw_lower = pw.lower()
            if pw_lower not in ALL_STOP and len(pw_lower) > 1 and pw_lower not in seen_assoc:
                if pw not in singles:
                    associated.append(pw)
                    seen_assoc.add(pw_lower)

    for i in range(len(singles) - 1):
        bigram = f"{singles[i]} {singles[i+1]}"
        if bigram.lower() not in seen_assoc:
            associated.append(bigram)
            seen_assoc.add(bigram.lower())

    return {
        "KEY_ENTITIES": key_entities,
        "ASSOCIATED_TERMS": associated,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Path B: LLM keyword expansion (batch submission & collection)
# ═══════════════════════════════════════════════════════════════════════

BLEND_EXPANSION_PROMPT = """\
You are a search keyword extraction expert. Given a cultural knowledge question, \
extract and generate relevant search keywords for a boolean search system that \
uses exact keyword matching with AND and OR operators.

The questions are about everyday cultural knowledge (food, holidays, sports, \
traditions, education, etc.) for a specific country. Generate keywords that \
would literally appear in web documents or encyclopedic text discussing these \
cultural topics.

Return a JSON object with these keys:
- "KEY_ENTITIES": an object mapping each core aspect of the question to a list \
of synonyms and related terms. Each aspect name should be a short phrase (1-3 \
words). Example: {"Thanksgiving": ["Turkey Day", "November holiday"], \
"traditional food": ["side dishes", "feast", "dinner"]}
- "ASSOCIATED_TERMS": a list of supporting terms that help narrow context — \
related cultural concepts, specific examples, regional variations. These should \
be terms likely to appear near the key entities in documents.

Rules:
- Each entry should be a keyword or short keyphrase (1-4 words)
- Focus on terms that would literally appear in relevant documents
- Include common colloquial names, abbreviations, and cultural-specific terms
- For questions asking "how many" or "what is the most popular", include the \
category being asked about and specific well-known examples
- Order by importance
- Return ONLY valid JSON, no explanation, no markdown fences"""


def submit_expansion_batch(blend_rows, model="claude-sonnet-4-6"):
    """Submit a batch of BLEnD questions for LLM keyword expansion."""
    import anthropic
    client = anthropic.Anthropic()

    requests_list = []
    for row in blend_rows:
        qid = row["id"]
        question = row["en_question"]

        answer_hints = []
        for ann in row.get("annotations", []):
            for ans in ann.get("en_answers", []):
                if ans and ans.lower() not in {"idk", "i don't know", "n/a", ""}:
                    answer_hints.append(ans)
        answer_hints = list(dict.fromkeys(answer_hints))[:10]

        user_msg = question
        if answer_hints:
            user_msg += (
                f"\n\n[Context: Known answers to this question include: "
                f"{', '.join(answer_hints)}. "
                f"Generate keywords that would help find documents discussing "
                f"both the question topic and these answers.]"
            )

        requests_list.append({
            "custom_id": qid,
            "params": {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0,
                "system": BLEND_EXPANSION_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            },
        })

    print(f"Submitting batch with {len(requests_list)} requests...")
    batch = client.messages.batches.create(requests=requests_list)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    return batch.id


def collect_expansion_results(batch_id, output_path):
    """Collect batch results and save as JSONL."""
    import anthropic
    client = anthropic.Anthropic()

    status = client.messages.batches.retrieve(batch_id)
    if status.processing_status != "ended":
        print(f"Batch not finished yet: {status.processing_status}")
        print(f"  Counts: {status.request_counts}")
        return None

    results = {}
    errors = 0
    for result in client.messages.batches.results(batch_id):
        qid = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            idx = text.find("{")
            if idx > 0:
                text = text[idx:]
            # Fix trailing commas
            text = re.sub(r',\s*([}\]])', r'\1', text)
            try:
                data = json.loads(text)
                data["qid"] = qid
                results[qid] = data
            except json.JSONDecodeError:
                print(f"  JSON parse error for {qid}: {text[:100]}...")
                errors += 1
        else:
            print(f"  Error for {qid}: {result.result.type}")
            errors += 1

    print(f"Collected {len(results)} expansions ({errors} errors)")

    with open(output_path, "w") as f:
        for qid in sorted(results.keys()):
            f.write(json.dumps(results[qid]) + "\n")
    print(f"Wrote: {output_path}")
    return results


def load_expansions(path: str) -> dict:
    """Load expansions from JSONL file. Returns {qid: expansion_dict}."""
    expansions = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry.pop("qid", None)
            if qid is not None:
                expansions[str(qid)] = entry
    return expansions


# ═══════════════════════════════════════════════════════════════════════
#  API-based peek counting
# ═══════════════════════════════════════════════════════════════════════

def peek_counts_api(
    expansion: dict,
    tokenizer,
    index: str = DEFAULT_INDEX,
    do_combos: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Get n-gram counts from the infini-gram API for one question's keywords.

    Phase 1: Standalone count for each keyword/phrase.
    Phase 2 (if do_combos): 2-way AND counts (cross-aspect k×k, k×assoc).

    Returns count summary — no document retrieval.
    """
    t0 = time.perf_counter()
    pieces = _build_pieces(expansion, tokenizer)

    n_api_calls = 0

    # ── Phase 1: standalone counts ────────────────────────────────────
    key_counts = {}   # desc → {count, aspect}
    assoc_counts = {} # desc → {count}

    for aspect_name, piece_list in pieces["key_pieces"].items():
        for p in piece_list:
            cnt = _api_count(p["cnf"], index)
            n_api_calls += 1
            key_counts[p["description"]] = {
                "count": cnt,
                "aspect": aspect_name,
                "piece": p,
            }
            if verbose:
                print(f"    [key/{aspect_name}] {p['description']}: {cnt:,d}")

    for p in pieces["associated"]:
        cnt = _api_count(p["cnf"], index)
        n_api_calls += 1
        assoc_counts[p["description"]] = {
            "count": cnt,
            "piece": p,
        }
        if verbose:
            print(f"    [assoc] {p['description']}: {cnt:,d}")

    # ── Phase 2: combo counts ─────────────────────────────────────────
    combos = []
    if do_combos:
        remaining_key = {d: i for d, i in key_counts.items() if i["count"] > 0}
        remaining_assoc = {d: i for d, i in assoc_counts.items() if i["count"] > 0}

        by_aspect = {}
        for desc, info in remaining_key.items():
            by_aspect.setdefault(info["aspect"], []).append((desc, info))

        aspect_names = list(by_aspect.keys())

        # k×k (cross-aspect)
        if len(aspect_names) >= 2:
            for a, b in combinations(aspect_names, 2):
                for d_a, i_a in by_aspect[a]:
                    for d_b, i_b in by_aspect[b]:
                        cnf = i_a["piece"]["cnf"] + i_b["piece"]["cnf"]
                        desc = f"({d_a}) AND ({d_b})"
                        cnt = _api_count(cnf, index)
                        n_api_calls += 1
                        if cnt > 0:
                            combos.append({
                                "description": desc, "count": cnt, "type": "kk",
                            })
                            if verbose:
                                print(f"    [k*k] {desc}: {cnt:,d}")

        # k×a (first aspect × associated)
        if aspect_names and remaining_assoc:
            first = aspect_names[0]
            for d_k, i_k in by_aspect.get(first, []):
                for d_a, i_a in remaining_assoc.items():
                    cnf = i_k["piece"]["cnf"] + i_a["piece"]["cnf"]
                    desc = f"({d_k}) AND ({d_a})"
                    cnt = _api_count(cnf, index)
                    n_api_calls += 1
                    if cnt > 0:
                        combos.append({
                            "description": desc, "count": cnt, "type": "ka",
                        })
                        if verbose:
                            print(f"    [k*a] {desc}: {cnt:,d}")

    elapsed = time.perf_counter() - t0

    # ── Aggregate ─────────────────────────────────────────────────────
    total_key = sum(i["count"] for i in key_counts.values())
    total_assoc = sum(i["count"] for i in assoc_counts.values())
    total_combo = sum(c["count"] for c in combos)
    nonzero_key = sum(1 for i in key_counts.values() if i["count"] > 0)
    nonzero_assoc = sum(1 for i in assoc_counts.values() if i["count"] > 0)

    return {
        "n_key_pieces": len(key_counts),
        "n_assoc_pieces": len(assoc_counts),
        "nonzero_key": nonzero_key,
        "nonzero_assoc": nonzero_assoc,
        "total_key_count": total_key,
        "total_assoc_count": total_assoc,
        "n_combos": len(combos),
        "total_combo_count": total_combo,
        "n_api_calls": n_api_calls,
        "time": elapsed,
        "key_counts": {d: i["count"] for d, i in key_counts.items()},
        "assoc_counts": {d: i["count"] for d, i in assoc_counts.items()},
        "combos": combos,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_blend_experiment(
    splits: list[str],
    mode: str = "no-expansion",
    expansions: dict = None,
    tokenizer=None,
    index: str = DEFAULT_INDEX,
    do_combos: bool = False,
    max_questions: int = None,
    output_dir: str = "./blend_results",
    blend_dir: str = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run BLEnD counting experiment across specified splits.

    Results are saved incrementally per question (checkpoint).  On restart,
    already-processed question IDs are skipped automatically.

    Args:
        splits:       List of BLEnD split names (country names).
        mode:         "no-expansion" or "expansion".
        expansions:   Pre-loaded {qid: expansion_dict} for mode="expansion".
        tokenizer:    HuggingFace tokenizer (LLaMA-2).
        index:        Infini-gram API index name.
        do_combos:    If True, also compute 2-way AND combo counts.
        max_questions: Cap per split.
        output_dir:   Where to save results.
        blend_dir:    Directory with pre-downloaded <split>.jsonl files.
        verbose:      Print progress.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for split in splits:
        print(f"\n{'='*70}")
        print(f"Split: {split}  |  mode={mode}  combos={do_combos}  index={index}")
        print(f"{'='*70}")

        # ── Load data ─────────────────────────────────────────────────
        if blend_dir:
            local_path = os.path.join(blend_dir, f"{split}.jsonl")
            if os.path.isfile(local_path):
                rows = load_blend(local_path, max_questions)
            else:
                rows = load_blend(split, max_questions)
        else:
            rows = load_blend(split, max_questions)

        print(f"  Loaded {len(rows)} questions")

        # ── Checkpoint ────────────────────────────────────────────────
        tag = f"{split}_{mode}"
        if do_combos:
            tag += "_combos"
        out_path = os.path.join(output_dir, f"{tag}_counts.jsonl")

        done_ids = set()
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_ids.add(json.loads(line).get("qid", ""))
                    except json.JSONDecodeError:
                        pass
            if done_ids:
                print(f"  Checkpoint: {len(done_ids)} already done, resuming...")

        # ── Process questions ─────────────────────────────────────────
        n_done = 0
        n_skipped = 0
        t_split = time.perf_counter()

        with open(out_path, "a") as out_f:
            for i, row in enumerate(rows):
                qid = row["id"]
                question = row["en_question"]

                if qid in done_ids:
                    continue

                # Build expansion
                if mode == "no-expansion":
                    expansion = extract_keywords_no_expansion(question)
                elif mode == "expansion":
                    if not expansions:
                        raise ValueError("expansions dict required for mode=expansion")
                    expansion = expansions.get(qid, {})
                    if not expansion:
                        n_skipped += 1
                        continue
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Get counts from API
                counts = peek_counts_api(
                    expansion, tokenizer, index=index,
                    do_combos=do_combos, verbose=False,
                )

                # Build result row
                result = {
                    "qid": qid,
                    "split": split,
                    "question": question,
                    "mode": mode,
                    "index": index,
                    "do_combos": do_combos,
                    # Expansion (keywords used)
                    "expansion": expansion,
                    # Counts
                    "n_key_pieces": counts["n_key_pieces"],
                    "n_assoc_pieces": counts["n_assoc_pieces"],
                    "nonzero_key": counts["nonzero_key"],
                    "nonzero_assoc": counts["nonzero_assoc"],
                    "total_key_count": counts["total_key_count"],
                    "total_assoc_count": counts["total_assoc_count"],
                    "n_combos": counts["n_combos"],
                    "total_combo_count": counts["total_combo_count"],
                    "n_api_calls": counts["n_api_calls"],
                    "peek_time": counts["time"],
                    # Per-keyword breakdowns
                    "key_counts": counts["key_counts"],
                    "assoc_counts": counts["assoc_counts"],
                    "combos": counts["combos"],
                    # BLEnD metadata
                    "annotations": row.get("annotations", []),
                    "idks": row.get("idks", 0),
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

                all_results.append(result)
                n_done += 1

                if verbose and (n_done % 100 == 0 or n_done <= 3):
                    elapsed = time.perf_counter() - t_split
                    rate = n_done / elapsed if elapsed > 0 else 0
                    todo = len(rows) - len(done_ids) - n_done
                    eta = todo / rate if rate > 0 else 0
                    print(
                        f"  [{n_done}/{len(rows) - len(done_ids)}] {qid}: "
                        f"{counts['nonzero_key']}/{counts['n_key_pieces']} key hits, "
                        f"total_key={counts['total_key_count']:,d} | "
                        f"{counts['n_api_calls']} calls {counts['time']:.1f}s | "
                        f"{rate:.1f} q/s, ~{eta/60:.0f}m left"
                    )

        elapsed_split = time.perf_counter() - t_split
        print(f"\n  {split} done: {n_done} processed, {n_skipped} skipped "
              f"({elapsed_split:.0f}s)")
        print(f"  Saved: {out_path}")

    # ── Cross-split summary ───────────────────────────────────────────
    if all_results:
        _save_summary(all_results, mode, do_combos, output_dir)

    return all_results


def _save_summary(results, mode, do_combos, output_dir):
    """Write a cross-split summary CSV from all results."""
    tag = mode
    if do_combos:
        tag += "_combos"
    csv_path = os.path.join(output_dir, f"blend_{tag}_summary.csv")

    fieldnames = [
        "split", "qid", "question",
        "n_key_pieces", "n_assoc_pieces",
        "nonzero_key", "nonzero_assoc",
        "total_key_count", "total_assoc_count",
        "n_combos", "total_combo_count",
        "n_api_calls", "peek_time",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["question"] = r.get("question", "")[:120]
            writer.writerow(row)
    print(f"  Summary CSV: {csv_path}")

    # Per-split aggregates
    by_split = defaultdict(list)
    for r in results:
        by_split[r["split"]].append(r)

    agg_path = os.path.join(output_dir, f"blend_{tag}_aggregates.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "n_questions",
            "avg_key_pieces", "avg_assoc_pieces",
            "avg_nonzero_key", "avg_nonzero_assoc",
            "avg_total_key_count", "avg_total_assoc_count",
            "zero_key_pct",
            "avg_combos", "avg_combo_count",
            "avg_api_calls", "avg_time",
        ])
        for split in sorted(by_split.keys()):
            rs = by_split[split]
            n = len(rs)
            writer.writerow([
                split, n,
                f"{sum(r['n_key_pieces'] for r in rs)/n:.1f}",
                f"{sum(r['n_assoc_pieces'] for r in rs)/n:.1f}",
                f"{sum(r['nonzero_key'] for r in rs)/n:.1f}",
                f"{sum(r['nonzero_assoc'] for r in rs)/n:.1f}",
                f"{sum(r['total_key_count'] for r in rs)/n:.0f}",
                f"{sum(r['total_assoc_count'] for r in rs)/n:.0f}",
                f"{sum(1 for r in rs if r['total_key_count']==0)/n*100:.1f}%",
                f"{sum(r['n_combos'] for r in rs)/n:.1f}",
                f"{sum(r['total_combo_count'] for r in rs)/n:.0f}",
                f"{sum(r['n_api_calls'] for r in rs)/n:.1f}",
                f"{sum(r['peek_time'] for r in rs)/n:.2f}",
            ])
    print(f"  Aggregates CSV: {agg_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Compare no-expansion vs expansion
# ═══════════════════════════════════════════════════════════════════════

def load_results(path: str) -> list[dict]:
    """Load results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compare_modes(
    no_exp_path: str,
    exp_path: str,
    output_dir: str = "./blend_results",
):
    """
    Compare no-expansion vs expansion results side by side.
    Loads from JSONL files and writes comparison CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_no = load_results(no_exp_path)
    results_exp = load_results(exp_path)

    no_exp = {r["qid"]: r for r in results_no}
    exp = {r["qid"]: r for r in results_exp}
    all_qids = sorted(set(no_exp.keys()) | set(exp.keys()))

    print(f"\n{'='*90}")
    print(f"Comparison: no-expansion vs expansion ({len(all_qids)} questions)")
    print(f"{'='*90}")

    for label, data in [("No expansion", no_exp), ("With expansion", exp)]:
        vals = list(data.values())
        if not vals:
            continue
        n = len(vals)
        avg_key = sum(r["n_key_pieces"] for r in vals) / n
        avg_assoc = sum(r["n_assoc_pieces"] for r in vals) / n
        avg_nz_key = sum(r["nonzero_key"] for r in vals) / n
        avg_key_cnt = sum(r["total_key_count"] for r in vals) / n
        zero = sum(1 for r in vals if r["total_key_count"] == 0)
        print(f"\n  {label} ({n} questions):")
        print(f"    Avg key pieces:   {avg_key:.1f}")
        print(f"    Avg assoc pieces: {avg_assoc:.1f}")
        print(f"    Avg nonzero key:  {avg_nz_key:.1f}")
        print(f"    Avg total key ct: {avg_key_cnt:,.0f}")
        print(f"    Zero-key-count:   {zero}/{n}")

    csv_path = os.path.join(output_dir, "blend_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "qid", "split", "question",
            "no_exp_key_count", "exp_key_count",
            "no_exp_nonzero", "exp_nonzero",
            "ratio",
        ])
        for qid in all_qids:
            ne = no_exp.get(qid, {})
            ex = exp.get(qid, {})
            ne_ct = ne.get("total_key_count", 0)
            ex_ct = ex.get("total_key_count", 0)
            ratio = (ex_ct / ne_ct) if ne_ct > 0 else (float("inf") if ex_ct > 0 else 0)
            writer.writerow([
                qid,
                (ne or ex).get("split", ""),
                (ne or ex).get("question", "")[:80],
                ne_ct, ex_ct,
                ne.get("nonzero_key", ""), ex.get("nonzero_key", ""),
                f"{ratio:.2f}" if ratio != float("inf") else "inf",
            ])
    print(f"\n  Wrote: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Trace — dry-run showing the full derivation chain (no API calls)
# ═══════════════════════════════════════════════════════════════════════

def trace_questions(
    split: str = "US",
    n: int = 5,
    mode: str = "no-expansion",
    expansions: dict = None,
    tokenizer=None,
    blend_dir: str = None,
    do_combos: bool = False,
    seed: int = 42,
):
    """
    Sample n questions from a split and print the full derivation chain
    without hitting the API.  Shows:
      1. Question text
      2. Keyword extraction / expansion
      3. Tokenized pieces (with token IDs and case variants)
      4. CNF queries that would be sent to the API
      5. Summary of API calls that would be made
    """
    import random

    # ── Load data ─────────────────────────────────────────────────────
    if blend_dir:
        local = os.path.join(blend_dir, f"{split}.jsonl")
        rows = load_blend(local if os.path.isfile(local) else split, max_rows=None)
    else:
        rows = load_blend(split, max_rows=None)

    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))

    W = 76  # print width
    total_api_calls = 0

    for qi, row in enumerate(sample, 1):
        qid = row["id"]
        question = row["en_question"] or row["question"]

        print(f"\n{'━' * W}")
        print(f"  [{qi}/{len(sample)}]  {qid}")
        print(f"{'━' * W}")

        # ── Step 1: question ──────────────────────────────────────────
        print(f"\n  QUESTION:  {question}")

        # Show ground-truth answers for context
        annotations = row.get("annotations", [])
        if annotations:
            ans_strs = []
            for ann in annotations:
                en = ann.get("en_answers", ann.get("answers", []))
                c = ann.get("count", "?")
                ans_strs.append(f"{'/'.join(en)} (×{c})")
            print(f"  ANSWERS:   {';  '.join(ans_strs)}")

        # ── Step 2: keyword extraction / expansion ────────────────────
        if mode == "no-expansion":
            expansion = extract_keywords_no_expansion(question)
            print(f"\n  EXTRACTION (no-expansion, stopword removal):")
        else:
            expansion = expansions.get(qid) if expansions else None
            if expansion is None:
                print(f"\n  ⚠ No LLM expansion found for {qid}, falling back to no-expansion")
                expansion = extract_keywords_no_expansion(question)
                print(f"\n  EXTRACTION (fallback):")
            else:
                print(f"\n  EXPANSION (LLM):")

        key_entities = expansion.get("KEY_ENTITIES", {})
        associated = expansion.get("ASSOCIATED_TERMS", expansion.get("ASSOCIATED", []))

        for aspect, terms in key_entities.items():
            is_generic = aspect.lower() in _GENERIC_ASPECTS
            label = f"  ├─ {aspect}" + (" (generic → name skipped)" if is_generic else " (→ name IS a keyword)")
            vals = terms if isinstance(terms, list) else list(terms.values()) if isinstance(terms, dict) else []
            print(f"{label}")
            print(f"  │    terms: {vals}")
        if associated:
            print(f"  └─ ASSOCIATED: {associated}")

        # ── Step 3: tokenized pieces ──────────────────────────────────
        if tokenizer is None:
            print(f"\n  TOKENIZATION:  (no tokenizer — showing keywords only)")
            # Count what WOULD happen
            n_key = 0
            for name, terms in key_entities.items():
                if isinstance(terms, dict):
                    flat = []
                    for lvl in ["lexical", "conceptual", "referential"]:
                        flat.extend(terms.get(lvl, []))
                    terms = flat
                if name.lower() in _GENERIC_ASPECTS:
                    n_key += len(terms) if isinstance(terms, list) else 0
                else:
                    n_key += 1 + (len(terms) if isinstance(terms, list) else 0)
            n_assoc = len(associated)
            standalone_calls = n_key + n_assoc
            combo_calls = 0
            if do_combos and n_key >= 2:
                combo_calls = n_key * (n_key - 1) // 2 + n_key * n_assoc
            print(f"  Would produce ~{n_key} key pieces, ~{n_assoc} assoc pieces")
            print(f"  API calls: {standalone_calls} standalone" +
                  (f" + ~{combo_calls} combos = ~{standalone_calls + combo_calls} total" if do_combos else ""))
            total_api_calls += standalone_calls + combo_calls
            continue

        pieces = _build_pieces(expansion, tokenizer)

        print(f"\n  TOKENIZED PIECES:")
        n_key_pieces = 0
        for aspect_name, piece_list in pieces["key_pieces"].items():
            for p in piece_list:
                n_key_pieces += 1
                nvars = len(p["variants"])
                # Show first variant's token IDs
                first_ids = p["variants"][0]
                ids_str = str(first_ids)
                if len(ids_str) > 50:
                    ids_str = ids_str[:47] + "..."
                print(f"  ├─ [key/{aspect_name}] \"{p['description']}\"")
                print(f"  │    {nvars} variant(s),  token IDs e.g. {ids_str}")

        for p in pieces["associated"]:
            nvars = len(p["variants"])
            first_ids = p["variants"][0]
            ids_str = str(first_ids)
            if len(ids_str) > 50:
                ids_str = ids_str[:47] + "..."
            print(f"  ├─ [assoc] \"{p['description']}\"")
            print(f"  │    {nvars} variant(s),  token IDs e.g. {ids_str}")

        n_assoc_pieces = len(pieces["associated"])

        # ── Step 4: CNF queries ───────────────────────────────────────
        print(f"\n  CNF QUERIES that would be sent:")
        standalone_calls = n_key_pieces + n_assoc_pieces
        print(f"  Phase 1 — {n_key_pieces} key + {n_assoc_pieces} assoc = {standalone_calls} standalone calls")

        # Show one example CNF in detail
        example_shown = False
        for aspect_name, piece_list in pieces["key_pieces"].items():
            if piece_list and not example_shown:
                p = piece_list[0]
                cnf = p["cnf"]
                print(f"\n    Example CNF for \"{p['description']}\":")
                print(f"      POST {API_URL}")
                print(f"      {{'index': '{DEFAULT_INDEX}',")
                print(f"       'query_type': 'count',")
                # Pretty-print the CNF
                cnf_str = json.dumps(cnf)
                if len(cnf_str) > 70:
                    print(f"       'query_ids': [")
                    for clause in cnf:
                        print(f"         {json.dumps(clause)},  // OR group")
                    print(f"       ]  // AND of the above")
                else:
                    print(f"       'query_ids': {cnf_str}")
                print(f"      }}")
                print(f"    → returns {{'count': N}}")
                example_shown = True

        combo_calls = 0
        if do_combos:
            # Count cross-aspect k×k pairs
            aspects = list(pieces["key_pieces"].keys())
            kk_pairs = 0
            if len(aspects) >= 2:
                from itertools import combinations as _comb
                for a, b in _comb(aspects, 2):
                    kk_pairs += len(pieces["key_pieces"][a]) * len(pieces["key_pieces"][b])

            # k×a pairs (first aspect × all assoc)
            ka_pairs = 0
            if aspects and n_assoc_pieces:
                ka_pairs = len(pieces["key_pieces"][aspects[0]]) * n_assoc_pieces

            combo_calls = kk_pairs + ka_pairs
            print(f"  Phase 2 — {kk_pairs} k×k + {ka_pairs} k×a = {combo_calls} combo calls")

            if kk_pairs > 0:
                # Show one example combo
                a, b = list(pieces["key_pieces"].keys())[:2]
                pa = pieces["key_pieces"][a][0]
                pb = pieces["key_pieces"][b][0]
                combo_cnf = pa["cnf"] + pb["cnf"]
                print(f"\n    Example combo: \"{pa['description']}\" AND \"{pb['description']}\"")
                print(f"      query_ids = {json.dumps(pa['cnf'])}  // {pa['description']}")
                print(f"                 + {json.dumps(pb['cnf'])}  // {pb['description']}")

        total_calls = standalone_calls + combo_calls
        total_api_calls += total_calls
        print(f"\n  TOTAL for this question: {total_calls} API call(s)")

    # ── Grand summary ─────────────────────────────────────────────────
    print(f"\n{'━' * W}")
    print(f"  TRACE SUMMARY  ({len(sample)} questions from {split})")
    print(f"{'━' * W}")
    print(f"  Mode:            {mode}")
    print(f"  Combos:          {'yes' if do_combos else 'no'}")
    print(f"  Total API calls: {total_api_calls}" +
          (f"  (≈ {total_api_calls * 0.5:.0f}s at 0.5s delay)" if total_api_calls else ""))
    if not do_combos:
        print(f"  (add --combos to see combo call counts)")
    print(f"{'━' * W}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    global API_DELAY
    parser = argparse.ArgumentParser(
        description="BLEnD cultural-knowledge experiment — API-based counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["no-expansion", "expansion",
                 "submit-expansion", "collect-expansion",
                 "list-splits", "compare", "trace"],
        default="no-expansion",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["US"],
        help='BLEnD splits to process (default: US). Use "all" for all known splits.',
    )
    parser.add_argument("--blend-dir", default=None,
                        help="Directory with pre-downloaded <split>.jsonl files")
    parser.add_argument("--expansions-path", default=None,
                        help="Path to LLM expansions JSONL (for mode=expansion)")
    parser.add_argument("--batch-id", default=None,
                        help="Anthropic batch ID (for mode=collect-expansion)")
    parser.add_argument("--output", default=None,
                        help="Output path for collected expansions")
    parser.add_argument("--output-dir", default="./blend_results",
                        help="Output directory for results (default: ./blend_results)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions per split")
    parser.add_argument("--index", default=DEFAULT_INDEX,
                        help=f"Infini-gram index (default: {DEFAULT_INDEX})")
    parser.add_argument("--delay", type=float, default=API_DELAY,
                        help=f"Seconds between API calls (default: {API_DELAY})")
    parser.add_argument("--combos", action="store_true", default=False,
                        help="Also compute 2-way AND combo counts (much slower)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Model for expansion batch submission")
    parser.add_argument("--no-exp-results", default=None,
                        help="No-expansion results JSONL (for mode=compare)")
    parser.add_argument("--exp-results", default=None,
                        help="Expansion results JSONL (for mode=compare)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("-n", type=int, default=5,
                        help="Number of questions to sample (for mode=trace, default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (for mode=trace, default: 42)")
    args = parser.parse_args()

    # Override API delay
    API_DELAY = args.delay

    # Resolve splits
    if args.splits == ["all"]:
        splits = BLEND_SPLITS
    else:
        splits = args.splits

    # ── List splits ───────────────────────────────────────────────────
    if args.mode == "list-splits":
        list_blend_splits()
        return

    # ── Compare ───────────────────────────────────────────────────────
    if args.mode == "compare":
        if not args.no_exp_results or not args.exp_results:
            parser.error("--no-exp-results and --exp-results required for compare")
        compare_modes(args.no_exp_results, args.exp_results, args.output_dir)
        return

    # ── Submit expansion batch ────────────────────────────────────────
    if args.mode == "submit-expansion":
        all_rows = []
        for split in splits:
            if args.blend_dir:
                local = os.path.join(args.blend_dir, f"{split}.jsonl")
                if os.path.isfile(local):
                    rows = load_blend(local, args.max_questions)
                else:
                    rows = load_blend(split, args.max_questions)
            else:
                rows = load_blend(split, args.max_questions)
            # Tag rows with split
            for r in rows:
                r["split"] = split
            all_rows.extend(rows)
            print(f"  {split}: {len(rows)} questions")
        print(f"Total: {len(all_rows)} questions across {len(splits)} splits")
        submit_expansion_batch(all_rows, model=args.model)
        return

    # ── Collect expansion results ─────────────────────────────────────
    if args.mode == "collect-expansion":
        if not args.batch_id:
            parser.error("--batch-id required for collect-expansion")
        output = args.output or "blend_expansions.jsonl"
        collect_expansion_results(args.batch_id, output)
        return

    # ── Trace (dry-run) ─────────────────────────────────────────────
    if args.mode == "trace":
        # Try to load the tokenizer for full trace; fall back gracefully
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            print("Loading tokenizer (LLaMA-2)...")
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                add_bos_token=False,
                add_eos_token=False,
            )
        except Exception as e:
            print(f"Tokenizer not available ({e})")
            print("Trace will show keywords and estimated call counts (no token IDs)\n")

        expansions = None
        exp_mode = "no-expansion"
        if args.expansions_path:
            expansions = load_expansions(args.expansions_path)
            exp_mode = "expansion"
            print(f"Loaded {len(expansions)} expansions")

        trace_questions(
            split=splits[0],
            n=args.n,
            mode=exp_mode,
            expansions=expansions,
            tokenizer=tokenizer,
            blend_dir=args.blend_dir,
            do_combos=args.combos,
            seed=args.seed,
        )
        return

    # ── Counting modes ────────────────────────────────────────────────
    from transformers import AutoTokenizer

    print(f"Loading tokenizer (LLaMA-2)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        add_eos_token=False,
    )

    expansions = None
    if args.mode == "expansion":
        if not args.expansions_path:
            parser.error("--expansions-path required for mode=expansion")
        expansions = load_expansions(args.expansions_path)
        print(f"Loaded {len(expansions)} expansions from {args.expansions_path}")

    print(f"\nAPI: {API_URL}")
    print(f"Index: {args.index}")
    print(f"Delay: {API_DELAY}s between calls")
    print(f"Splits: {splits}")
    print(f"Mode: {args.mode}")
    print(f"Combos: {args.combos}")

    results = run_blend_experiment(
        splits=splits,
        mode=args.mode,
        expansions=expansions,
        tokenizer=tokenizer,
        index=args.index,
        do_combos=args.combos,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        blend_dir=args.blend_dir,
        verbose=args.verbose,
    )

    print(f"\nDone — {len(results)} results across {len(splits)} split(s)")


if __name__ == "__main__":
    main()