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
#  Known BLEnD splits  (ISO-style codes used by uilab/BLEnD dataset)
# ═══════════════════════════════════════════════════════════════════════

BLEND_SPLITS = [
    "DZ", "AS", "AZ", "CN", "ET", "GR", "ID", "IR",
    "MX", "KP", "NG", "KR", "ES", "GB", "US", "JB",
]

# Human-readable names for output/logging
BLEND_SPLIT_NAMES = {
    "DZ": "Algeria",
    "AS": "Assam",
    "AZ": "Azerbaijan",
    "CN": "China",
    "ET": "Ethiopia",
    "GR": "Greece",
    "ID": "Indonesia",
    "IR": "Iran",
    "MX": "Mexico",
    "KP": "North Korea",
    "NG": "Northern Nigeria",
    "KR": "South Korea",
    "ES": "Spain",
    "GB": "UK",
    "US": "US",
    "JB": "West Java",
}

# Surface forms used as the "country" clause in answer ∧ country queries.
#
# The infini-gram API allows at most 4 terms per OR clause, so each tier is a
# complete clause of <= 4 terms, most important first.  Terms are used EXACTLY
# as written — no case variants ("US" must never become "us").  A term that
# contains another term of the same clause is redundant (an n-gram count of
# "Korea" already includes every "South Korea"), so build_clause drops it.
#
#   core: the main clause — region name as BLEnD writes it + demonyms.
#   alt:  an alternative clause for robustness checks (replaces core):
#         language names, the parent country for sub-national splits,
#         stricter or wider readings.  Used with --country-tier alt.
BLEND_COUNTRY_TERMS = {
    "DZ": {"core": ["Algeria", "Algerian", "Algerians"],
           "alt":  ["Algeria", "Algerian", "Algerians", "Algiers"]},
    "AS": {"core": ["Assam"],  # covers "Assamese"
           "alt":  ["Assam", "Northeast India", "North East India"]},
    "AZ": {"core": ["Azerbaijan", "Azerbaijani", "Azeri", "Azeris"],  # "Azerbaijan" covers "Azerbaijanis"
           "alt":  ["Azerbaijan", "Azerbaijani", "Azeri", "Baku"]},
    "CN": {"core": ["China", "Chinese", "PRC"],
           "alt":  ["China", "Chinese", "PRC", "Mandarin"]},
    "ET": {"core": ["Ethiopia", "Ethiopian", "Ethiopians"],
           "alt":  ["Ethiopia", "Ethiopian", "Ethiopians", "Habesha"]},
    "GR": {"core": ["Greece", "Greek", "Greeks"],
           "alt":  ["Greece", "Greek", "Greeks", "Hellenic"]},
    "ID": {"core": ["Indonesia", "Indonesian", "Indonesians"],
           "alt":  ["Indonesia", "Indonesian", "Indonesians", "Jakarta"]},
    "IR": {"core": ["Iran"],  # covers "Iranian", "Iranians"
           "alt":  ["Iran", "Persian", "Persians", "Persia"]},
    "MX": {"core": ["Mexico", "Mexican", "Mexicans"],  # NB: "Mexico" also matches "New Mexico"
           "alt":  ["Mexico", "Mexican", "Mexicans", "Mexicano"]},
    "KP": {"core": ["North Korea", "North Korean", "North Koreans", "DPRK"],
           "alt":  ["North Korea", "North Korean", "DPRK", "Pyongyang"]},
    "NG": {"core": ["Northern Nigeria", "Northern Nigerian", "Hausa"],
           "alt":  ["Nigeria", "Nigerian", "Nigerians", "Hausa"]},  # all of Nigeria
    # Bare "Korea"/"Korean" also cover North Korea mentions; alt is the strict reading.
    "KR": {"core": ["Korea", "Korean", "Koreans"],
           "alt":  ["South Korea", "South Korean", "South Koreans"]},
    "ES": {"core": ["Spain", "Spaniard", "Spaniards"],
           "alt":  ["Spain", "Spaniard", "Spaniards", "Spanish"]},  # "Spanish" is mostly the language
    "GB": {"core": ["UK", "Britain", "British", "England"],  # "Britain" covers "Great Britain"
           "alt":  ["UK", "U.K.", "United Kingdom", "British"]},
    "US": {"core": ["US", "U.S.", "United States", "American"],
           "alt":  ["USA", "America", "Americans", "United States"]},
    # "Sunda" alone hits geography (Sunda Strait/Shelf); bare "Java" omitted: programming language
    "JB": {"core": ["West Java", "Sundanese"],
           "alt":  ["West Java", "Sundanese", "Sunda", "Bandung"]},
}


def _get_hf_splits():
    """Auto-detect available splits from HuggingFace."""
    from datasets import load_dataset_builder
    builder = load_dataset_builder("uilab/BLEnD", "annotations")
    return list(builder.info.splits.keys()), builder.info.splits


def list_blend_splits():
    """Try to list available BLEnD splits from HuggingFace, fall back to known list."""
    try:
        splits, split_info = _get_hf_splits()
        print(f"Available splits from HuggingFace ({len(splits)}):")
        for s in sorted(splits):
            name = BLEND_SPLIT_NAMES.get(s, "")
            n = split_info[s].num_examples
            label = f"{s} ({name})" if name else s
            print(f"  {label:30s}  {n:>6,d} rows")
        return splits
    except Exception as e:
        print(f"Could not query HuggingFace: {e}")
        print(f"Known splits ({len(BLEND_SPLITS)}):")
        for s in BLEND_SPLITS:
            name = BLEND_SPLIT_NAMES.get(s, "")
            label = f"{s} ({name})" if name else s
            print(f"  {label}")
        return BLEND_SPLITS


def resolve_splits(requested):
    """Resolve --splits argument. 'all' auto-detects from HuggingFace."""
    if requested == ["all"]:
        try:
            splits, _ = _get_hf_splits()
            print(f"Auto-detected {len(splits)} splits from HuggingFace: {', '.join(splits)}")
            return splits
        except Exception:
            print(f"Could not auto-detect splits, using built-in list of {len(BLEND_SPLITS)}")
            return BLEND_SPLITS
    return requested


# ═══════════════════════════════════════════════════════════════════════
#  BLEnD data loading
# ═══════════════════════════════════════════════════════════════════════

def load_blend_hf(split: str = "US", max_rows: int = None):
    """Load BLEnD annotations from HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        ds = load_dataset("uilab/BLEnD", "annotations", split=split)
    except Exception as e:
        # Common: old datasets library doesn't support Json feature type
        raise RuntimeError(
            f"Could not load BLEnD split '{split}' from HuggingFace.\n"
            f"  Error: {e}\n\n"
            f"  Fix:\n"
            f"    pip install -U datasets   (upgrade to latest)\n"
            f"  Then either load directly or pre-download as JSONL:\n"
            f"    python blend_experiment.py --mode download --splits all --blend-dir ./blend_data"
        ) from e
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


def download_blend_splits(output_dir: str, splits: list[str] = None):
    """
    Download BLEnD annotation splits as JSONL files using the HuggingFace
    datasets library.

    Requires: pip install -U datasets
    Each split is saved as <output_dir>/<Split>.jsonl
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' library is required for download mode.\n"
            "  Install/upgrade it with:  pip install -U datasets"
        )

    if splits is None:
        splits = BLEND_SPLITS

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        out_path = os.path.join(output_dir, f"{split}.jsonl")
        if os.path.isfile(out_path):
            with open(out_path) as f:
                n_existing = sum(1 for line in f if line.strip())
            print(f"  {split}: already exists ({n_existing} rows), skipping")
            continue

        print(f"  {split}: downloading...", end="", flush=True)
        try:
            ds = load_dataset("uilab/BLEnD", "annotations", split=split)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        rows = []
        for row in ds:
            rows.append({
                "ID": row.get("ID", ""),
                "question": row.get("question", ""),
                "en_question": row.get("en_question", ""),
                "annotations": row.get("annotations", []),
                "idks": row.get("idks", {}),
            })

        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f" {len(rows)} rows → {out_path}")


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
        # path_or_split is a split name — try HuggingFace datasets library
        return load_blend_hf(path_or_split, max_rows)


def _resolve_blend_path(split: str, blend_dir: str = None) -> str:
    """Return local file path for a split, or the split name for HF fallback."""
    if blend_dir:
        local = os.path.join(blend_dir, f"{split}.jsonl")
        if os.path.isfile(local):
            return local
    return split


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
#  Path C: answer ∧ country (∧ anchor) — LLM anchor extraction
# ═══════════════════════════════════════════════════════════════════════
#
# Why not "AND every question concept": infini-gram AND queries subsample
# any clause with > max_clause_freq hits, so 4–5-clause topic queries come
# back as approx 0.  Instead we count, per ground-truth answer:
#     n_answer                 exact n-gram count of the answer (OR of variants)
#     answer ∧ country         does the corpus tie this answer to the region?
#     answer ∧ country ∧ anchor   only for generic answers ("fruit", "rice")
# plus one topic count per question: anchor ∧ country.
# An LLM supplies the anchors, answer surface variants and needs_anchor flags.

ANCHOR_MODEL = "claude-haiku-5-5"

ANCHOR_PROMPT = """\
You prepare search terms for measuring how often a pretraining corpus mentions \
cultural facts. The corpus is searched with exact, case-sensitive string \
matching, and terms are combined with AND inside a window of a few hundred words.

You receive a question about everyday culture in one region, plus the answer \
groups that annotators from that region gave. Return a JSON object with:

1. "anchors": 1-2 concepts that pin down the question's topic, most specific \
first. An anchor is a word a passage about this topic would almost certainly \
contain, apart from the region and the answer itself.
   - Never use the region/country name or demonym; that is handled separately.
   - Never use generic words such as common, popular, traditional, typical, \
usual, people, most, best, kind, type, thing, way, time, day (unless the day \
itself is the topic).
   - Use a concrete noun or short noun phrase from the question ("snack", \
"preschool", "wedding", "Lunar New Year"), not the question's phrasing.
   - "concept": the anchor in its base form.
   - "variants": 1-4 surface forms, most important first, as they would \
appear in English text: the base form, plural/singular, hyphenation or \
spelling variants, and near-synonyms only if they mean the same thing here \
("pre-school" for "preschool"; not "kindergarten").
   - If the answer carries the whole topic (e.g. "What is the most popular \
sport?"), the anchor is the category ("sport", "sports").

2. "answers": one entry per numbered answer group, in the given order:
   - "id": the group number.
   - "variants": 1-4 English surface forms, most important first: the \
annotator alternatives in the group, plural/singular, and common alternate \
spellings or transliterations of local terms ("jollof rice", "jollof"). Use \
only terms that name this group's answer, not broader, narrower or other \
groups' items. Strip leading articles and filler verbs ("eating", "going \
to") unless they are part of a name. If the answer is a long description, \
give the short distinctive phrases a document would literally contain \
("12 grapes", "eat grapes"), not the whole sentence.
   - "needs_anchor": true if the answer is a generic word that often appears \
near the region's name for unrelated reasons ("fruit", "rice", "football", \
"family", "church", "Sunday", "tea", "red", "morning"), so answer AND region \
alone would not show that the corpus links it to this question. false if it \
is specific enough that co-occurring with the region is mostly about this \
topic (named dishes, brands, festivals, specific places, people, institutions).
   - "skip": true only if the answer is not a countable term ("none", \
"nothing", "it depends", "I don't know", an opinion sentence); otherwise false.

Rules for all variants:
- Capitalization variants are added automatically: do not list a term again \
with only different capitalization, and never lowercase proper names.
- Every term matches anywhere it appears, so a short term also matches inside \
longer phrases. Never shorten a name to a part that is ambiguous on its own \
("Jordan" for "Michael Jordan", "Magic" for "Magic Johnson", "bars" for \
"sports bars"). Include a short form only if it is widely used alone and \
unambiguous ("Shaq", "jollof").

Return only the JSON object."""

ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "variants": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["concept", "variants"],
                "additionalProperties": False,
            },
        },
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "variants": {"type": "array", "items": {"type": "string"}},
                    "needs_anchor": {"type": "boolean"},
                    "skip": {"type": "boolean"},
                },
                "required": ["id", "variants", "needs_anchor", "skip"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["anchors", "answers"],
    "additionalProperties": False,
}


_FORMAT_MARKERS = ("Arabic numerals", "HH:MM", "MM/DD")


def is_numeric_question(question: str) -> bool:
    """
    BLEnD numeric / time / date questions ("Provide Arabic numerals",
    "HH:MM format", "MM/DD format"): answer ∧ country is meaningless there.
    """
    return any(m in question for m in _FORMAT_MARKERS)


def answer_groups(row) -> list[dict]:
    """Annotation groups as [{id, en_answers, votes}]; id = index in row['annotations']."""
    groups = []
    for i, ann in enumerate(row.get("annotations", [])):
        answers = [a.strip() for a in ann.get("en_answers", []) if a and a.strip()]
        answers = list(dict.fromkeys(answers))
        if answers:
            groups.append({"id": i, "en_answers": answers, "votes": ann.get("count", 0)})
    return groups


def anchor_custom_id(split: str, qid: str) -> str:
    """BLEnD question IDs repeat across splits, so the batch ID carries the split."""
    return f"{split}_{qid}"


def build_anchor_user_msg(row, split: str) -> str:
    region = BLEND_SPLIT_NAMES.get(split, split)
    lines = [f"Region: {region}", f"Question: {row['en_question']}",
             "Answer groups (alternatives separated by ' | '):"]
    for g in answer_groups(row):
        lines.append(f"{g['id']}: {' | '.join(g['en_answers'])}")
    return "\n".join(lines)


def anchor_request_params(row, split: str, model: str = ANCHOR_MODEL) -> dict:
    return {
        "model": model,
        "max_tokens": 4000,
        "system": ANCHOR_PROMPT,
        "messages": [{"role": "user", "content": build_anchor_user_msg(row, split)}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": ANCHOR_SCHEMA},
        },
    }


def anchor_rows(splits, blend_dir=None, max_questions=None):
    """(split, row) pairs for all non-numeric questions that have answers."""
    out = []
    for split in splits:
        for row in load_blend(_resolve_blend_path(split, blend_dir), max_questions):
            if is_numeric_question(row["en_question"]) or not answer_groups(row):
                continue
            out.append((split, row))
    return out


def submit_anchor_batch(pairs, model: str = ANCHOR_MODEL):
    """Submit one anchor-extraction request per (split, row) to the Message Batches API."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    requests_list = [
        Request(custom_id=anchor_custom_id(split, row["id"]),
                params=MessageCreateParamsNonStreaming(**anchor_request_params(row, split, model)))
        for split, row in pairs
    ]
    print(f"Submitting anchor batch: {len(requests_list)} requests, model={model}")
    batch = client.messages.batches.create(requests=requests_list)
    print(f"Batch ID: {batch.id}  status: {batch.processing_status}")
    return batch.id


def collect_anchor_results(batch_id: str, output_path: str):
    """Write {split, qid, anchors, answers} JSONL from a finished anchor batch."""
    import anthropic
    client = anthropic.Anthropic()

    status = client.messages.batches.retrieve(batch_id)
    if status.processing_status != "ended":
        print(f"Batch not finished: {status.processing_status}  {status.request_counts}")
        return None

    n_ok, failed = 0, []
    with open(output_path, "w") as f:
        for result in client.messages.batches.results(batch_id):
            cid = result.custom_id
            split, qid = cid.split("_", 1)
            if result.result.type != "succeeded":
                failed.append((cid, result.result.type))
                continue
            msg = result.result.message
            if msg.stop_reason in ("refusal", "max_tokens"):
                failed.append((cid, msg.stop_reason))
                continue
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                failed.append((cid, "bad_json"))
                continue
            f.write(json.dumps({"split": split, "qid": qid, **data}, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"Collected {n_ok} anchor results → {output_path}  ({len(failed)} failed)")
    for cid, why in failed[:20]:
        print(f"  failed: {cid}  {why}")
    return n_ok


def load_anchors(path: str) -> dict:
    """{(split, qid): {anchors, answers}} from collected anchor JSONL."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[(obj["split"], obj["qid"])] = obj
    return out


# ── Question-level entities (one per question ID, region-independent) ──
#
# All 16 splits share the same 500 question IDs; the English question only
# differs in the region name.  Entities (topic phrases + key terms) are
# therefore extracted once per ID from a "[REGION]" template.

ENTITY_PROMPT = """\
You prepare search units for measuring how often a pretraining corpus \
discusses the topic of a question. The corpus is searched by exact, \
case-sensitive string matching: a term matches only that exact string, so a \
singular does NOT match its plural ("snack" does not match "snacks"). Units \
can be combined with AND inside a window of a few hundred words.

You receive a question about everyday culture. [REGION] stands for the \
region; ignore it. Return a JSON object with:

1. "phrases": 0-3 multi-word expressions that name the question's topic as a \
unit and that documents would literally contain ("preschool snack", "gifted \
education", "New Year's Eve", "sports broadcast", "high school"). Build them \
from the question's words in natural word order; you may drop words in \
between ("snack for preschool kids" -> "preschool snack"). Only include \
expressions people actually write, not paraphrases.

2. "terms": 1-4 single key concepts the topic depends on, most specific \
first: content words or fixed compounds ("preschool", "snack"). Exclude \
generic words (common, popular, traditional, typical, usual, famous, most, \
best, people, kind, type, thing, way, time), question words, the verbs of the \
question frame (do, usually, consider, celebrate only if it is not the topic), \
and the region.

For every phrase and term give:
- "text": the base form.
- "variants": 1-4 surface forms, most important first: the base form; its \
plural or singular whenever the word is countable (always include it: \
"snack" -> "snacks", "festival" -> "festivals"; for phrases inflect the head \
noun: "preschool snack" -> "preschool snacks"); then common spelling or \
hyphenation variants ("pre-school"). Do not list capitalization-only \
variants (they are added automatically) and never lowercase proper names.

Return only the JSON object."""

_UNIT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "variants": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text", "variants"],
        "additionalProperties": False,
    },
}
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"phrases": _UNIT_SCHEMA, "terms": _UNIT_SCHEMA},
    "required": ["phrases", "terms"],
    "additionalProperties": False,
}


def _region_template(question: str, split: str) -> str:
    name = BLEND_SPLIT_NAMES.get(split, split)
    for pat in (f"the {name}", name):
        if pat in question:
            return question.replace(pat, "[REGION]")
    return question


def question_templates(blend_dir: str = "blend_data") -> dict:
    """
    {qid: {"template", "n_wordings", "wordings", "numeric"}} over all splits.
    The template is the majority wording; 14 IDs have >1 wording (mostly
    "of"/"in" differences, a few genuinely different questions).
    """
    from collections import Counter
    wordings = defaultdict(Counter)
    for split in BLEND_SPLITS:
        for row in load_blend(_resolve_blend_path(split, blend_dir)):
            wordings[row["id"]][_region_template(row["en_question"], split)] += 1
    out = {}
    for qid, c in wordings.items():
        template = c.most_common(1)[0][0]
        out[qid] = {"template": template, "n_wordings": len(c),
                    "wordings": dict(c) if len(c) > 1 else None,
                    "numeric": is_numeric_question(template)}
    return out


def entity_request_params(template: str, model: str = ANCHOR_MODEL) -> dict:
    return {
        "model": model,
        "max_tokens": 2000,
        "system": ENTITY_PROMPT,
        "messages": [{"role": "user", "content": f"Question: {template}"}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": ENTITY_SCHEMA},
        },
    }


# ── Counting ──────────────────────────────────────────────────────────

CNF_MAX_CLAUSE_FREQ = 500000   # API maximum; fewer clauses get subsampled
CNF_MAX_DIFF_TOKENS = 1000     # API maximum window between adjacent clauses


def _api_query(query_ids, index=DEFAULT_INDEX, cnf_params=True, retries=4):
    """
    Count query with retries. Returns {"count", "approx", "error"}; on failure
    count is None (never a silent 0, so failures can't pass as real zeros).
    """
    payload = {"index": index, "query_type": "count", "query_ids": query_ids}
    if cnf_params:
        payload["max_clause_freq"] = CNF_MAX_CLAUSE_FREQ
        payload["max_diff_tokens"] = CNF_MAX_DIFF_TOKENS
    err = None
    for attempt in range(retries):
        time.sleep(API_DELAY if attempt == 0 else API_DELAY * 4 * 2 ** attempt)
        try:
            resp = requests.post(API_URL, json=payload, timeout=120)
        except Exception as e:  # network error, timeout
            err = f"{type(e).__name__}: {e}"
            continue
        if resp.status_code == 429:  # rate limited: back off, then retry
            err = "HTTP 429 Too Many Requests"
            time.sleep(60)
            continue
        if resp.status_code == 403:  # IP blocked: retrying makes it worse
            raise RuntimeError("HTTP 403 Forbidden from infini-gram API — stop and wait before retrying")
        try:
            result = resp.json()
        except Exception as e:  # non-JSON body
            err = f"HTTP {resp.status_code} non-JSON: {e}"
            continue
        if "error" in result:
            err = str(result["error"])
            if err.startswith("Please"):  # validation error (e.g. too many terms): retrying won't help
                break
            continue
        return {"count": result.get("count"), "approx": bool(result.get("approx", False)),
                "error": None}
    return {"count": None, "approx": None, "error": err}


def _surface_forms(term: str) -> list[str]:
    """The term, plus a capitalized form if it starts lowercase. Never lowercases."""
    term = term.strip()
    if not term:
        return []
    forms = [term]
    if term[0].islower():
        forms.append(term[0].upper() + term[1:])
    return forms


MAX_OR_TERMS = 4  # API limit: "at most 4 terms in each disjunctive clause"


def _contains(seq, sub):
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def build_clause(terms, tokenizer, max_terms=MAX_OR_TERMS) -> list[list[int]]:
    """
    One CNF clause (OR of token sequences) for surface terms in priority order.
    Drops duplicates and any sequence that contains another one: n-gram
    counts are substring counts in token space, so "Iran" already covers
    "Iranian" and "Korea" covers "South Korea" (keeping both risks double
    counting).  Then keeps the first max_terms, preserving priority order.
    """
    # Given variants first (plurals etc.), auto-capitalized forms after them,
    # so capitalization never pushes a real variant past the 4-term cap.
    forms = [t.strip() for t in terms if t and t.strip()]
    forms += [f for t in forms for f in _surface_forms(t)[1:]]
    seqs, seen = [], set()
    for form in forms:
        ids = tokenizer.encode(form, add_special_tokens=False)
        if ids and tuple(ids) not in seen:
            seen.add(tuple(ids))
            seqs.append(ids)
    kept = [s for s in seqs
            if not any(o is not s and len(o) < len(s) and _contains(s, o) for o in seqs)]
    return kept[:max_terms]


def count_forms(forms, tokenizer, index=DEFAULT_INDEX, delay=0.0):
    """
    Exact total count for a list of surface forms, however many there are.
    Forms are pruned once (duplicates and containing sequences removed, as in
    build_clause), then split into OR clauses of at most MAX_OR_TERMS terms.
    The kept sequences are disjoint as token strings, so the per-clause counts
    add up to the exact count of the whole OR. One API call per clause.
    """
    seqs = build_clause(forms, tokenizer, max_terms=10_000)
    clauses = [seqs[i:i + MAX_OR_TERMS] for i in range(0, len(seqs), MAX_OR_TERMS)]
    total, parts, error = 0, [], None
    for c in clauses:
        r = _api_query([c], index, cnf_params=False)
        if delay:
            time.sleep(delay)
        if r["count"] is None:
            error = r["error"]
            total = None
            break
        total += r["count"]
        parts.append({"forms": [tokenizer.decode(s) for s in c], "count": r["count"]})
    return {"count": total, "clauses": parts, "error": error}


def run_phrase_counts(phrases_path, out_path, tokenizer, index=DEFAULT_INDEX, delay=1.5):
    """
    Exact counts for the literal question phrases (one row per qid × phrase).
    Input: question_phrases.jsonl rows {qid, phrases: [{text, variants, specificity}]}.
    Checkpointed: rows already written without an error are skipped.
    """
    with open(phrases_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    done = set()
    if os.path.exists(out_path):
        done = {(r["qid"], r["text"]) for r in load_results(out_path) if not r.get("error")}
    todo = [(r["qid"], p) for r in rows for p in r.get("phrases", []) if (r["qid"], p["text"]) not in done]
    print(f"phrase counts: {len(todo)} to do, {len(done)} done → {out_path}", flush=True)
    t0 = time.perf_counter()
    with open(out_path, "a") as f:
        for i, (qid, p) in enumerate(todo, 1):
            r = count_forms(p.get("variants") or [p["text"]], tokenizer, index=index, delay=delay)
            rec = {"qid": qid, "text": p["text"], "specificity": p.get("specificity"), **r}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(todo)}] ~{el / i * (len(todo) - i) / 60:.0f}m left", flush=True)


def answer_query_forms(q):
    """All surface forms of one reviewed answer query row (queries, then native)."""
    forms = [v for g in q.get("queries", []) for v in (g.get("variants") or [g["text"]])]
    return forms + [n for n in q.get("native", []) if isinstance(n, str)]


def count_answer_query(q, tokenizer, index=DEFAULT_INDEX, delay=0.0):
    """
    Counts for one answer_queries.jsonl row.
      count     exact occurrences of any of its forms (count_forms, summed clauses)
      and_sides per AND group: its OR clause (≤4 forms) alone, exact
      and_count AND of the groups within CNF_MAX_DIFF_TOKENS; exact only when
                every side is ≤ CNF_MAX_CLAUSE_FREQ (and_exact)
    """
    forms = answer_query_forms(q)
    rec = count_forms(forms, tokenizer, index=index, delay=delay) if forms else \
        {"count": None, "clauses": [], "error": None}
    rec.update({"and_sides": [], "and_count": None, "and_exact": None})
    groups = [g for g in q.get("and", []) if g.get("variants") or g.get("text")]
    if len(groups) >= 2 and rec["error"] is None:
        clauses = [build_clause(g.get("variants") or [g["text"]], tokenizer) for g in groups]
        for g, c in zip(groups, clauses):
            r = _api_query([c], index, cnf_params=False)
            if delay:
                time.sleep(delay)
            rec["and_sides"].append({"text": g.get("text"), "count": r["count"]})
            if r["count"] is None:
                rec["error"] = r["error"]
                return rec
        r = _api_query(clauses, index)
        if delay:
            time.sleep(delay)
        if r["count"] is None:
            rec["error"] = r["error"]
            return rec
        rec["and_count"] = r["count"]
        rec["and_exact"] = all(s["count"] <= CNF_MAX_CLAUSE_FREQ for s in rec["and_sides"])
    return rec


def run_answer_query_counts(queries_path, out_path, tokenizer, index=DEFAULT_INDEX, delay=1.5):
    """
    Counts for every reviewed answer query (answer_queries.jsonl → answer_query_counts.jsonl).
    One row per answer key. Checkpointed: keys already written without an error are skipped,
    and a row is recounted when its forms changed since it was counted (e.g. after review).
    """
    with open(queries_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    done = {}
    if os.path.exists(out_path):
        done = {r["key"]: r for r in load_results(out_path) if not r.get("error")}

    def sig(q):
        return [answer_query_forms(q), [g.get("variants") for g in q.get("and", [])]]

    todo = [q for q in rows if q["key"] not in done or done[q["key"]].get("sig") != sig(q)]
    print(f"answer query counts: {len(todo)} to do, {len(rows) - len(todo)} done → {out_path}", flush=True)
    t0 = time.perf_counter()
    with open(out_path, "a") as f:
        for i, q in enumerate(todo, 1):
            r = count_answer_query(q, tokenizer, index=index, delay=delay)
            rec = {"key": q["key"], "id": q["id"], "type": q["type"], "generic": q["generic"],
                   "sig": sig(q), **r}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(todo)}] ~{el / i * (len(todo) - i) / 3600:.1f}h left", flush=True)


def count_answer_anchor(row, split, anchor_entry, tokenizer, index=DEFAULT_INDEX,
                        country_tier="core", anchor_all=False):
    """All counts for one question. Returns a result dict (one JSONL row)."""
    t0 = time.perf_counter()
    tiers = BLEND_COUNTRY_TERMS[split]
    country_terms = tiers[country_tier]
    country = build_clause(country_terms, tokenizer)

    anchors = [a for a in anchor_entry.get("anchors", []) if a.get("variants")]
    anchor_clause = build_clause(anchors[0]["variants"], tokenizer) if anchors else None

    n_calls = 0

    def q(cnf, cnf_params=True):
        nonlocal n_calls
        n_calls += 1
        return _api_query(cnf, index, cnf_params=cnf_params)

    topic = q([anchor_clause, country]) if anchor_clause else None
    anchor_alone = q([anchor_clause], cnf_params=False) if anchor_clause else None

    llm_by_id = {a["id"]: a for a in anchor_entry.get("answers", [])}
    answers = []
    for g in answer_groups(row):
        llm = llm_by_id.get(g["id"], {})
        variants = list(dict.fromkeys(g["en_answers"] + llm.get("variants", [])))
        rec = {
            "id": g["id"], "en_answers": g["en_answers"], "votes": g["votes"],
            "variants": variants,
            "needs_anchor": bool(llm.get("needs_anchor", False)),
            "skip": bool(llm.get("skip", False)),
            "missing_llm": g["id"] not in llm_by_id,
        }
        if rec["skip"]:
            answers.append(rec)
            continue
        ans = build_clause(variants, tokenizer)
        if not ans:
            rec["skip"] = True
            answers.append(rec)
            continue
        a = q([ans], cnf_params=False)  # pure OR → exact
        rec["n_answer"] = a["count"]
        rec["error"] = a["error"]
        if a["count"] == 0:  # nothing to intersect
            rec.update(n_answer_country=0, approx_answer_country=False)
            if anchor_clause and (rec["needs_anchor"] or anchor_all):
                rec.update(n_answer_country_anchor=0, approx_answer_country_anchor=False)
            answers.append(rec)
            continue
        ac = q([ans, country])
        rec["n_answer_country"] = ac["count"]
        rec["approx_answer_country"] = ac["approx"]
        rec["error"] = rec["error"] or ac["error"]
        if anchor_clause and (rec["needs_anchor"] or anchor_all):
            aca = q([ans, country, anchor_clause])
            rec["n_answer_country_anchor"] = aca["count"]
            rec["approx_answer_country_anchor"] = aca["approx"]
            rec["error"] = rec["error"] or aca["error"]
        answers.append(rec)

    return {
        "qid": row["id"], "split": split, "question": row["en_question"],
        "index": index, "country_tier": country_tier,
        "max_clause_freq": CNF_MAX_CLAUSE_FREQ, "max_diff_tokens": CNF_MAX_DIFF_TOKENS,
        "country_terms": country_terms,
        "anchors": anchors,
        "n_anchor": anchor_alone["count"] if anchor_alone else None,
        "topic_count": topic["count"] if topic else None,
        "topic_approx": topic["approx"] if topic else None,
        "answers": answers,
        "n_api_calls": n_calls,
        "time": time.perf_counter() - t0,
        "has_errors": any(r.get("error") for r in answers)
                      or bool(topic and topic["error"]),
    }


def _unit_words(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def count_question_entities(qid, info, ents, tokenizer, index=DEFAULT_INDEX):
    """
    Region-independent overview for one question ID.

    units: every phrase and term, with an exact count per token sequence
           (plain n-gram counts are always exact) and their sum = unit count.
           A unit over CNF_MAX_CLAUSE_FREQ makes any AND containing it an
           unreliable estimate.
    ands:  pairwise ANDs of units that share no word, plus the AND of all
           terms.  Queried only when every clause is <= the cap (then the
           result is exact); otherwise recorded as over_limit, not queried.
    """
    t0 = time.perf_counter()
    n_calls = 0
    units = []
    for kind in ("phrases", "terms"):
        for u in ents.get(kind, []):
            seqs = build_clause(u.get("variants") or [u["text"]], tokenizer)
            seq_counts = []
            for s in seqs:
                r = _api_query(s, index, cnf_params=False)  # flat ids → exact n-gram count
                n_calls += 1
                seq_counts.append({"text": tokenizer.decode(s), "count": r["count"], "error": r["error"]})
            ok = all(c["count"] is not None for c in seq_counts)
            total = sum(c["count"] for c in seq_counts) if ok else None
            units.append({
                "kind": kind[:-1], "text": u["text"], "variants": u.get("variants", []),
                "seqs": seq_counts, "count": total,
                "over_limit": total is not None and total > CNF_MAX_CLAUSE_FREQ,
                "_clause": seqs,
            })

    combos = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if not (_unit_words(units[i]["text"]) & _unit_words(units[j]["text"])):
                combos.append((i, j))
    term_ix = tuple(i for i, u in enumerate(units) if u["kind"] == "term")
    if len(term_ix) >= 3:
        combos.append(term_ix)

    ands = []
    for combo in combos:
        us = [units[i] for i in combo]
        rec = {"units": [u["text"] for u in us]}
        if any(u["count"] is None or not u["_clause"] for u in us):
            rec["status"] = "unit_error"
        elif any(u["count"] == 0 for u in us):
            rec.update(status="exact", count=0)
        elif any(u["over_limit"] for u in us):
            rec.update(status="over_limit",
                       over=[u["text"] for u in us if u["over_limit"]])
        else:
            r = _api_query([u["_clause"] for u in us], index)
            n_calls += 1
            rec.update(status="exact" if r["approx"] is False else "approx",
                       count=r["count"], error=r["error"])
        ands.append(rec)

    for u in units:
        del u["_clause"]
    return {"qid": qid, "template": info["template"], "numeric": info["numeric"],
            "n_wordings": info["n_wordings"], "units": units, "ands": ands,
            "cap": CNF_MAX_CLAUSE_FREQ, "n_api_calls": n_calls,
            "time": time.perf_counter() - t0}


def run_entity_counts(entities_path, templates, tokenizer, out_path, index=DEFAULT_INDEX):
    """Checkpointed overview over all question IDs in entities_path (JSONL with qid)."""
    done = set()
    if os.path.exists(out_path):
        for r in load_results(out_path):
            errs = any(c["error"] for u in r["units"] for c in u["seqs"]) or \
                   any(a.get("error") or a["status"] == "unit_error" for a in r["ands"])
            if not errs:
                done.add(r["qid"])
    ents = [json.loads(l) for l in open(entities_path) if l.strip()]
    todo = [e for e in ents if e["qid"] not in done]
    print(f"entity counts: {len(todo)} to do, {len(done)} done → {out_path}")
    t_start = time.perf_counter()
    with open(out_path, "a") as f:
        for k, e in enumerate(todo, 1):
            res = count_question_entities(e["qid"], templates[e["qid"]], e, tokenizer, index)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if k <= 3 or k % 25 == 0:
                el = time.perf_counter() - t_start
                print(f"  [{k}/{len(todo)}] {e['qid']}: {res['n_api_calls']} calls "
                      f"| ~{el / k * (len(todo) - k) / 60:.0f}m left", flush=True)


def _answer_key(a: str) -> str:
    return re.sub(r"\s+", " ", a.strip()).lower()


def _is_numeric_answer(a: str) -> bool:
    return bool(re.fullmatch(r"[\d\s.,:/~\-–]+(am|pm|AM|PM)?", a.strip()))


def answer_forms(a: str) -> list[str]:
    """
    Surface forms for an annotator answer: as written, lowercase (unless it
    contains an acronym such as "BBQ"/"US", which must not be lowercased),
    then capitalized.  build_clause dedupes and caps at 4.
    """
    a = re.sub(r"\s+", " ", a.strip())
    forms = [a]
    if not re.search(r"\b[A-Z]{2,}\b", a):
        forms.append(a.lower())
    return forms  # build_clause appends capitalized forms of lowercase-initial ones


def answer_index(blend_dir="blend_data") -> tuple[list[dict], dict]:
    """
    (occurrences, unique): every annotator answer as
    {split, qid, group, votes, answer, key}, and {key: first-seen spelling}.
    """
    occ, uniq = [], {}
    for split in BLEND_SPLITS:
        for row in load_blend(_resolve_blend_path(split, blend_dir)):
            for g in answer_groups(row):
                for a in g["en_answers"]:
                    k = _answer_key(a)
                    uniq.setdefault(k, a.strip())
                    occ.append({"split": split, "qid": row["id"], "group": g["id"],
                                "votes": g["votes"], "answer": a, "key": k})
    return occ, uniq


def run_answer_counts(out_path, tokenizer, blend_dir="blend_data", index=DEFAULT_INDEX):
    """Exact corpus count (OR of surface forms) per unique answer. Checkpointed."""
    _, uniq = answer_index(blend_dir)
    done = set()
    if os.path.exists(out_path):
        done = {r["key"] for r in load_results(out_path) if not r.get("error")}
    todo = [(k, a) for k, a in uniq.items() if k not in done]
    print(f"answer counts: {len(todo)} to do, {len(done)} done → {out_path}", flush=True)
    t0 = time.perf_counter()
    with open(out_path, "a") as f:
        for i, (k, a) in enumerate(todo, 1):
            if _is_numeric_answer(a):
                rec = {"key": k, "answer": a, "skipped": "numeric", "count": None, "error": None}
            else:
                clause = build_clause(answer_forms(a), tokenizer)
                r = _api_query([clause], index, cnf_params=False) if clause else \
                    {"count": None, "error": "empty"}
                rec = {"key": k, "answer": a, "forms": [tokenizer.decode(s) for s in clause],
                       "count": r["count"], "error": r["error"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(todo)}] ~{el / i * (len(todo) - i) / 60:.0f}m left", flush=True)


def run_country_counts(out_path, tokenizer, index=DEFAULT_INDEX):
    """Exact count per country term and per clause (sum over its kept sequences)."""
    out = []
    for split, tiers in BLEND_COUNTRY_TERMS.items():
        for tier, terms in tiers.items():
            seqs = build_clause(terms, tokenizer)
            kept = {tokenizer.decode(s) for s in seqs}
            per_term = []
            for t in terms:
                ids = tokenizer.encode(t, add_special_tokens=False)
                r = _api_query(ids, index, cnf_params=False)
                per_term.append({"term": t, "count": r["count"], "error": r["error"],
                                 "in_clause": t in kept})
            clause_total = sum(p["count"] or 0 for p in per_term if p["in_clause"])
            out.append({"split": split, "name": BLEND_SPLIT_NAMES[split], "tier": tier,
                        "terms": per_term, "clause_total": clause_total,
                        "over_cap": clause_total > CNF_MAX_CLAUSE_FREQ})
            print(f"  {split} {tier:4s} clause_total={clause_total:>13,}", flush=True)
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


def summarize_entity_counts(path, out_csv):
    """
    Per-question overview CSV + aggregate printout from entity_counts.jsonl.

    category:
      exact_and    some AND of units is under the cap (exact) and > 0
      exact_unit   no usable AND, but some unit is under the cap and > 0
      over_cap     every unit with hits is over the cap (no exact handle)
      zero         no unit has any hits
    """
    rows = load_results(path)
    cap = rows[0]["cap"] if rows else CNF_MAX_CLAUSE_FREQ
    out, cats = [], defaultdict(int)
    plural_gain = []
    for r in rows:
        units = [u for u in r["units"] if u["count"] is not None]
        under = [u for u in units if 0 < u["count"] <= cap]
        good_ands = [a for a in r["ands"] if a.get("status") == "exact" and (a.get("count") or 0) > 0]
        if good_ands:
            cat = "exact_and"
        elif under:
            cat = "exact_unit"
        elif any(u["count"] > 0 for u in units):
            cat = "over_cap"
        else:
            cat = "zero"
        cats[cat] += 1
        for u in units:
            if u["count"] and len(u["seqs"]) > 1:
                plural_gain.append(1 - (u["seqs"][0]["count"] or 0) / u["count"])
        best_and = max(good_ands, key=lambda a: a["count"], default=None)
        out.append({
            "qid": r["qid"], "category": cat, "numeric": r["numeric"],
            "template": r["template"],
            "units": "; ".join(f"{u['text']}={u['count']:,}" + ("*" if u["over_limit"] else "")
                               for u in units),
            "n_units": len(units), "n_units_under_cap": len(under),
            "min_unit": min((u["count"] for u in units if u["count"]), default=0),
            "n_exact_ands": len(good_ands),
            "best_exact_and": " ∧ ".join(best_and["units"]) if best_and else "",
            "best_exact_and_count": best_and["count"] if best_and else "",
            "n_over_limit_ands": sum(a.get("status") == "over_limit" for a in r["ands"]),
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    n = len(rows)
    print(f"{n} questions  (cap = {cap:,}; * = over cap)")
    for c in ("exact_and", "exact_unit", "over_cap", "zero"):
        print(f"  {c:11s} {cats[c]:>4}  ({cats[c] / n:.0%})")
    if plural_gain:
        plural_gain.sort()
        print(f"  share of a unit's count from non-first variants: median "
              f"{plural_gain[len(plural_gain) // 2]:.0%}, mean {sum(plural_gain) / len(plural_gain):.0%}")
    print(f"  wrote {out_csv}")
    return out


def run_answer_anchor(splits, anchors, tokenizer, blend_dir=None, output_dir="./blend_results",
                      index=DEFAULT_INDEX, country_tier="core", anchor_all=False,
                      max_questions=None, workers=1, only_qids=None):
    """
    Count answer ∧ country (∧ anchor) for every question that has an anchor
    entry. Checkpointed per split; rows with API errors are re-run on restart.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    os.makedirs(output_dir, exist_ok=True)
    tag = f"answer-anchor_{country_tier}" + ("_all" if anchor_all else "")
    for split in splits:
        out_path = os.path.join(output_dir, f"{split}_{tag}.jsonl")
        done = set()
        if os.path.exists(out_path):
            for r in load_results(out_path):
                if not r.get("has_errors"):
                    done.add(r["qid"])
        todo = []
        for row in load_blend(_resolve_blend_path(split, blend_dir), max_questions):
            if only_qids and row["id"] not in only_qids:
                continue
            entry = anchors.get((split, row["id"]))
            if entry is None or row["id"] in done:
                continue
            todo.append((row, entry))
        print(f"\n{split}: {len(todo)} to do, {len(done)} already done → {out_path}")
        if not todo:
            continue

        lock = threading.Lock()
        t_split = time.perf_counter()
        with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(count_answer_anchor, row, split, entry, tokenizer, index,
                                country_tier, anchor_all) for row, entry in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                if i <= 3 or i % 25 == 0:
                    el = time.perf_counter() - t_split
                    print(f"  [{i}/{len(todo)}] {res['qid']}: {res['n_api_calls']} calls, "
                          f"{res['time']:.0f}s | ~{el / i * (len(todo) - i) / 60:.0f}m left")


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
                 "list-splits", "compare", "trace", "download",
                 "submit-anchors", "collect-anchors", "answer-anchor",
                 "entity-counts", "answer-counts", "country-counts", "phrase-counts", "answer-query-counts", "summarize-entities"],
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
    parser.add_argument("--anchors-path", default="blend_anchors.jsonl",
                        help="Anchor JSONL (output of collect-anchors, input of answer-anchor)")
    parser.add_argument("--anchor-model", default=ANCHOR_MODEL,
                        help=f"Model for submit-anchors (default: {ANCHOR_MODEL})")
    parser.add_argument("--country-tier", choices=["core", "alt"], default="core",
                        help="Which BLEND_COUNTRY_TERMS clause to use (answer-anchor)")
    parser.add_argument("--anchor-all", action="store_true", default=False,
                        help="answer-anchor: run the 3-way query for every answer, not only needs_anchor ones")
    parser.add_argument("--entities-path", default="blend_entities.jsonl",
                        help="entity-counts: per-question entities JSONL ({qid, phrases, terms})")
    parser.add_argument("--templates-path", default="blend_question_templates.json",
                        help="entity-counts: question templates JSON (from question_templates())")
    parser.add_argument("--workers", type=int, default=1,
                        help="answer-anchor: parallel questions (be polite to the public API)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("-n", type=int, default=5,
                        help="Number of questions to sample (for mode=trace, default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (for mode=trace, default: 42)")
    args = parser.parse_args()

    # Override API delay
    API_DELAY = args.delay

    # Resolve splits
    splits = resolve_splits(args.splits)

    # ── List splits ───────────────────────────────────────────────────
    if args.mode == "list-splits":
        list_blend_splits()
        return

    # ── Download ──────────────────────────────────────────────────────
    if args.mode == "download":
        out = args.blend_dir or args.output_dir
        print(f"Downloading BLEnD splits to {out}/")
        download_blend_splits(out, splits)
        print(f"\nDone. Use --blend-dir {out} with other modes.")
        return

    # ── Anchor extraction batch (Path C) ──────────────────────────────
    if args.mode == "submit-anchors":
        pairs = anchor_rows(splits, args.blend_dir, args.max_questions)
        print(f"{len(pairs)} non-numeric questions across {len(splits)} split(s)")
        submit_anchor_batch(pairs, model=args.anchor_model)
        return

    if args.mode == "collect-anchors":
        if not args.batch_id:
            parser.error("--batch-id required for collect-anchors")
        collect_anchor_results(args.batch_id, args.anchors_path)
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

    if args.mode == "summarize-entities":
        summarize_entity_counts(os.path.join(args.output_dir, "entity_counts.jsonl"),
                                os.path.join(args.output_dir, "entity_overview.csv"))
        return

    # ── Counting modes ────────────────────────────────────────────────
    from transformers import AutoTokenizer

    print(f"Loading tokenizer (LLaMA-2)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        add_eos_token=False,
    )

    if args.mode == "answer-counts":
        os.makedirs(args.output_dir, exist_ok=True)
        run_answer_counts(os.path.join(args.output_dir, "answer_counts.jsonl"), tokenizer,
                          blend_dir=args.blend_dir or "blend_data", index=args.index)
        return

    if args.mode == "phrase-counts":
        os.makedirs(args.output_dir, exist_ok=True)
        run_phrase_counts(os.path.join(args.output_dir, "question_phrases.jsonl"),
                          os.path.join(args.output_dir, "question_phrase_counts.jsonl"), tokenizer,
                          index=args.index, delay=args.delay)
        return

    if args.mode == "answer-query-counts":
        os.makedirs(args.output_dir, exist_ok=True)
        run_answer_query_counts(os.path.join(args.output_dir, "answer_queries.jsonl"),
                                os.path.join(args.output_dir, "answer_query_counts.jsonl"), tokenizer,
                                index=args.index, delay=args.delay)
        return

    if args.mode == "country-counts":
        os.makedirs(args.output_dir, exist_ok=True)
        run_country_counts(os.path.join(args.output_dir, "country_counts.jsonl"), tokenizer,
                           index=args.index)
        return

    if args.mode == "entity-counts":
        with open(args.templates_path) as f:
            templates = json.load(f)
        os.makedirs(args.output_dir, exist_ok=True)
        run_entity_counts(args.entities_path, templates, tokenizer,
                          os.path.join(args.output_dir, "entity_counts.jsonl"), index=args.index)
        return

    if args.mode == "answer-anchor":
        run_answer_anchor(
            splits, load_anchors(args.anchors_path), tokenizer,
            blend_dir=args.blend_dir, output_dir=args.output_dir, index=args.index,
            country_tier=args.country_tier, anchor_all=args.anchor_all,
            max_questions=args.max_questions, workers=args.workers,
        )
        return

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