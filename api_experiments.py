"""
API experiments: keyword expansion + peek and grab via public infini-gram API.

No local engine, no benchmark needed. Just a query string → keywords → retrieval counts.

Usage:
    from api_experiments import run_experiment

    result = run_experiment(
        query="how did the Vietnam War devastate the economy in 1968",
        model="claude-sonnet-4-20250514",
        tokenizer=tokenizer,
    )
"""

import os
import time
import json
import requests
from itertools import combinations
from transformers import AutoTokenizer
from utils import get_token


API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_olmo-2-0325-32b-instruct_llama"
API_DELAY = 0.1  # seconds between API calls

EXPANSION_PROMPT = """For the following search query, generate keyword expansions in JSON format.

Query: {query}

Return ONLY valid JSON with this exact structure:
{{
    "KEY_ENTITIES": {{
        "<aspect1>": ["<expansion1>", "<expansion2>", "<expansion3>", "<expansion4>", "<expansion5>"],
        "<aspect2>": ["<expansion1>", "<expansion2>", "<expansion3>", "<expansion4>", "<expansion5>"]
    }},
    "ASSOCIATED_TERMS": ["<term1>", "<term2>", "<term3>", "<term4>", "<term5>", "<term6>", "<term7>", "<term8>", "<term9>", "<term10>"],
    "VERBS": {{
        "<verb1>": ["<synonym1>", "<synonym2>", "<synonym3>", "<synonym4>", "<synonym5>"]
    }}
}}

Rules:
- KEY_ENTITIES: Break the query into 2-4 key aspects. For each aspect, give 5 alternative phrasings.
- ASSOCIATED_TERMS: 10 domain-specific terms strongly associated with the query topic.
- VERBS: Main action verb(s) from the query with 5 synonyms each.
- Keep it simple. Only include terms that would actually appear in relevant documents."""


# ================================================================
# Expansion generation
# ================================================================

def get_expansions(
    query: str,
    model: str = "claude-sonnet-4-20250514",
    cache_path: str = "./api_expansions.jsonl",
):
    """
    Get keyword expansions for a query string.

    Checks cache first. If not found, calls LLM API.
    Supports Anthropic models (claude-*) and OpenRouter models (everything else).

    Returns expansion dict with KEY_ENTITIES, ASSOCIATED_TERMS, VERBS.
    """
    # Check cache
    cached = _load_cached(query, model, cache_path)
    if cached is not None:
        print(f"  [cache hit] {model}: {query[:50]}...")
        return cached

    print(f"  [calling {model}] {query[:50]}...")

    if model.startswith("claude"):
        result = _call_anthropic(query, model)
    else:
        result = _call_openrouter(query, model)

    if result is not None:
        # Save to cache
        entry = {
            "text": query,
            "model": model,
        }
        entry.update(result)
        with open(cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return result


def _load_cached(query, model, cache_path):
    """Check if expansion already exists in cache."""
    if not os.path.exists(cache_path):
        return None

    with open(cache_path) as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if obj.get("text") == query and obj.get("model") == model:
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _call_anthropic(query, model):
    """Call Anthropic API for expansions."""
    import anthropic

    client = anthropic.Anthropic(api_key=get_token('ANTHROPIC_API_KEY'))
    prompt = EXPANSION_PROMPT.format(query=query)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return _parse_expansion(text)
    except Exception as e:
        print(f"    Anthropic error: {e}")
        return None


def _call_openrouter(query, model):
    """Call OpenRouter API for expansions."""
    api_key = get_token('OPENROUTER_API_KEY')

    # Fall back to OpenAI for OpenAI models
    if not api_key and ("gpt" in model or "o1" in model or "o3" in model):
        return _call_openai(query, model)

    prompt = EXPANSION_PROMPT.format(query=query)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return _parse_expansion(text)
    except Exception as e:
        print(f"    OpenRouter error: {e}")
        return None


def _call_openai(query, model):
    """Call OpenAI API for expansions."""
    from openai import OpenAI

    client = OpenAI(api_key=get_token('OPENAI_API_KEY'))
    prompt = EXPANSION_PROMPT.format(query=query)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        return _parse_expansion(text)
    except Exception as e:
        print(f"    OpenAI error: {e}")
        return None


def _parse_expansion(text):
    """Parse LLM response into expansion dict."""
    import re

    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Strip preamble
    idx = text.find('{')
    if idx > 0:
        text = text[idx:]

    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"    JSON parse error")
        return None


# ================================================================
# Infini-gram API calls
# ================================================================

def _api_count(query_ids, index):
    """
    Count via API. query_ids can be:
    - flat list of ints: contiguous n-gram
    - list of list of list of ints: CNF query
    """
    time.sleep(API_DELAY)
    payload = {
        "index": index,
        "query_type": "count",
        "query_ids": query_ids,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        result = response.json()
        if "error" in result:
            return 0
        return result.get("count", 0)
    except Exception:
        return 0


def _api_search(query_ids, index, max_disp_len=500):
    """Search via API. Same query_ids format as _api_count."""
    time.sleep(API_DELAY)
    payload = {
        "index": index,
        "query_type": "search_docs",
        "query_ids": query_ids,
        "max_disp_len": max_disp_len,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        result = response.json()
        if "error" in result:
            # Try alternative query type
            payload["query_type"] = "search"
            response = requests.post(API_URL, json=payload, timeout=60)
            result = response.json()
        return result
    except Exception as e:
        return {"error": str(e)}


def retrieve_documents(
    queries: list,
    tokenizer,
    index: str = DEFAULT_INDEX,
    verbose: bool = True,
):
    """
    Execute queries and retrieve documents from infini-gram API.

    Args:
        queries: List of query dicts from peek_and_grab (all_queries).
        tokenizer: HuggingFace tokenizer for decoding.
        index: Infini-gram index name.

    Returns list of unique document dicts with doc_id, text, from_queries.
    """
    doc_map = {}

    for i, q in enumerate(queries):
        if verbose:
            print(f"  [{i+1}/{len(queries)}] {q['description']} (est: {q['estimated_count']:,d})")

        result = _api_search(q["cnf"], index)

        if "error" in result:
            if verbose:
                print(f"    Error: {result['error']}")
            continue

        # Debug: show response keys on first successful call
        if i == 0 and verbose:
            print(f"    Response keys: {list(result.keys())}")
            # Print a sample of the response
            for k, v in result.items():
                if isinstance(v, list) and v:
                    print(f"    {k}[0]: {str(v[0])[:200]}")
                elif isinstance(v, (int, float, str)):
                    print(f"    {k}: {v}")

        # Parse response
        docs = result.get("documents", [])

        if isinstance(docs, list):
            for doc in docs:
                if doc.get("blocked"):
                    continue

                # Extract text from spans
                spans = doc.get("spans", [])
                text = ""
                if spans and isinstance(spans[0], list) and len(spans[0]) >= 2:
                    text = spans[0][1] if isinstance(spans[0][1], str) else str(spans[0][1])
                elif spans and isinstance(spans[0], str):
                    text = spans[0]

                # Extract doc_id from metadata
                doc_id = f"doc_{doc.get('doc_ix', i)}"
                meta_str = doc.get("metadata", "")
                if isinstance(meta_str, str) and meta_str:
                    try:
                        meta = json.loads(meta_str)
                        # Try to find a URL or ID
                        doc_id = meta.get("path", "")
                        if "metadata" in meta and isinstance(meta["metadata"], dict):
                            url = meta["metadata"].get("url", "")
                            if url:
                                doc_id = url
                            linenum = meta.get("linenum", "")
                            if linenum:
                                doc_id = f"{meta.get('path', '')}:{linenum}"
                    except json.JSONDecodeError:
                        pass

                if doc_id in doc_map:
                    doc_map[doc_id]["from_queries"].append(q["description"])
                else:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "doc_ix": doc.get("doc_ix", -1),
                        "doc_len": doc.get("doc_len", 0),
                        "text": text,
                        "from_queries": [q["description"]],
                    }

        if verbose:
            n_new = len(doc_map) - sum(1 for _ in doc_map)  # just print running total
            print(f"    Got {len(docs) if isinstance(docs, list) else 0} docs, "
                  f"total unique: {len(doc_map)}")

    documents = list(doc_map.values())
    if verbose:
        print(f"\n  Total: {len(documents)} unique documents retrieved")

    return documents


# ================================================================
# Tokenization
# ================================================================

def _tokenize_keyword(keyword, tokenizer):
    """
    Tokenize a keyword into a CNF piece with case variants.
    
    "Vietnam War" with variants becomes:
    cnf = [[[token_ids_for_"Vietnam War"], [token_ids_for_"vietnam war"], [token_ids_for_"Vietnam war"]]]
    
    This is a single AND clause with OR'd variants.
    """
    variants = []
    seen = set()

    for v in [keyword, keyword.lower(), keyword.title()]:
        token_ids = tokenizer.encode(v, add_special_tokens=False)
        if not token_ids:
            continue
        key = tuple(token_ids)
        if key not in seen:
            seen.add(key)
            variants.append(token_ids)

    if not variants:
        return None

    # CNF: one disjunctive clause containing all variants OR'd
    # [[[variant1_tokens], [variant2_tokens], ...]]
    cnf = [variants]

    return {
        "description": keyword,
        "keyword": keyword,
        "cnf": cnf,
        "variants": variants,
    }


def _build_pieces(expansion, tokenizer):
    """Build CNF pieces from expansion dict."""
    key_entities = expansion.get("KEY_ENTITIES", {})
    associated = expansion.get("ASSOCIATED_TERMS", expansion.get("ASSOCIATED", []))

    key_pieces = {}
    for name, terms in key_entities.items():
        if isinstance(terms, dict):
            flat = []
            for level in ["lexical", "conceptual", "referential"]:
                flat.extend(terms.get(level, []))
            terms = flat

        all_terms = [name] + (terms if isinstance(terms, list) else [])
        aspect_pieces = []

        for kw in all_terms:
            piece = _tokenize_keyword(kw, tokenizer)
            if piece:
                piece["source_aspect"] = name
                aspect_pieces.append(piece)

        key_pieces[name] = aspect_pieces

    assoc_pieces = []
    for kw in associated:
        piece = _tokenize_keyword(kw, tokenizer)
        if piece:
            assoc_pieces.append(piece)

    return {"key_pieces": key_pieces, "associated": assoc_pieces}


# ================================================================
# Peek and grab
# ================================================================

def peek_and_grab(
    expansion: dict,
    tokenizer,
    index: str = DEFAULT_INDEX,
    max_standalone_key: int = 1500,
    max_standalone_assoc: int = 750,
    max_budget: int = 20000,
    max_assoc_combo: int = 50000,
    verbose: bool = True,
):
    """
    Two-phase peek and grab using the public API.

    Args:
        expansion: Dict with KEY_ENTITIES, ASSOCIATED_TERMS (from get_expansions).
        tokenizer: HuggingFace tokenizer.
        index: Infini-gram index name.
    """
    t0 = time.perf_counter()

    pieces = _build_pieces(expansion, tokenizer)

    # ================================================================
    # Phase 1: Standalone peek and grab
    # ================================================================
    all_key_pieces = {}
    all_assoc_pieces = {}
    grabbed = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 1: Standalone peek [API: {index}]")
        print(f"{'='*70}")

    for aspect_name, piece_list in pieces["key_pieces"].items():
        for p in piece_list:
            cnt = _api_count(p["cnf"], index)

            desc = p["description"]
            info = {
                "piece": p,
                "count": cnt,
                "aspect": aspect_name,
                "grabbed": cnt <= max_standalone_key and cnt > 0,
            }
            all_key_pieces[desc] = info

            if info["grabbed"]:
                grabbed.append({
                    "type": "cnf",
                    "cnf": p["cnf"],
                    "description": desc,
                    "estimated_count": cnt,
                    "level": "S0_key",
                })

            if verbose:
                status = "GRAB" if info["grabbed"] else ("skip" if cnt == 0 else "remain")
                print(f"  [{aspect_name}] {desc}: {cnt:>10,d}  {status}")

    for p in pieces["associated"]:
        cnt = _api_count(p["cnf"], index)

        desc = p["description"]
        info = {
            "piece": p,
            "count": cnt,
            "grabbed": cnt <= max_standalone_assoc and cnt > 0,
        }
        all_assoc_pieces[desc] = info

        if info["grabbed"]:
            grabbed.append({
                "type": "cnf",
                "cnf": p["cnf"],
                "description": desc,
                "estimated_count": cnt,
                "level": "S0_assoc",
            })

        if verbose:
            status = "GRAB" if info["grabbed"] else ("skip" if cnt == 0 else "remain")
            print(f"  [assoc] {desc}: {cnt:>10,d}  {status}")

    grabbed_total = sum(q["estimated_count"] for q in grabbed)
    remaining_key = {d: i for d, i in all_key_pieces.items() if not i["grabbed"] and i["count"] > 0}
    remaining_assoc = {d: i for d, i in all_assoc_pieces.items() if not i["grabbed"] and i["count"] > 0}

    print(f"  Phase 1: {len(grabbed)} standalone grabbed (~{grabbed_total:,d} docs), "
          f"{len(remaining_key)} key + {len(remaining_assoc)} assoc remaining")

    # ================================================================
    # Phase 2: Combo peek
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 2: Combo peek — 2-way ANDs")
        print(f"{'='*70}")

    remaining_by_aspect = {}
    for desc, info in remaining_key.items():
        aspect = info["aspect"]
        remaining_by_aspect.setdefault(aspect, []).append(info)

    aspect_names = list(remaining_by_aspect.keys())
    combo_queries = []
    n_combo_checked = 0

    # k×k (cross-aspect)
    if len(aspect_names) >= 2:
        for a, b in combinations(aspect_names, 2):
            for p_a in remaining_by_aspect[a]:
                for p_b in remaining_by_aspect[b]:
                    # AND = concatenate CNF clauses
                    cnf = p_a["piece"]["cnf"] + p_b["piece"]["cnf"]
                    desc = f"({p_a['piece']['description']}) AND ({p_b['piece']['description']})"

                    cnt = _api_count(cnf, index)
                    n_combo_checked += 1

                    if cnt == 0:
                        continue

                    combo_queries.append({
                        "type": "cnf",
                        "cnf": cnf,
                        "description": desc,
                        "estimated_count": cnt,
                        "level": "S_combo_kk",
                    })

                    if verbose:
                        print(f"  [k×k] {desc}: {cnt:>10,d}")

    # k×a (first aspect only)
    if aspect_names:
        first_aspect = aspect_names[0]
        first_aspect_pieces = remaining_by_aspect[first_aspect]

        for info_k in first_aspect_pieces:
            for desc_a, info_a in remaining_assoc.items():
                if info_a["count"] > max_assoc_combo:
                    continue

                desc = f"({info_k['piece']['description']}) AND ({info_a['piece']['description']})"

                cnf = info_k["piece"]["cnf"] + info_a["piece"]["cnf"]
                cnt = _api_count(cnf, index)
                n_combo_checked += 1

                if cnt == 0:
                    continue

                if cnt > max_standalone_key:
                    continue

                combo_queries.append({
                    "type": "cnf",
                    "cnf": cnf,
                    "description": desc,
                    "estimated_count": cnt,
                    "level": "S_combo_ka",
                    "assoc_used": desc_a,
                })

                if verbose:
                    print(f"  [k×a] {desc}: {cnt:>10,d}")

    # Sort: k×k first, then k×a
    combo_kk = sorted([q for q in combo_queries if q["level"] == "S_combo_kk"],
                       key=lambda q: q["estimated_count"])
    combo_ka = sorted([q for q in combo_queries if q["level"] == "S_combo_ka"],
                       key=lambda q: q["estimated_count"])
    combo_queries = combo_kk + combo_ka

    combo_total = sum(q["estimated_count"] for q in combo_queries)
    remaining_assoc_total = sum(i["count"] for i in remaining_assoc.values())

    print(f"  Phase 2: {n_combo_checked} combos checked, "
          f"{len(combo_kk)} k×k + {len(combo_ka)} k×a (~{combo_total:,d} docs). "
          f"Total: ~{grabbed_total + combo_total:,d} / {max_budget:,d}")

    # ================================================================
    # Phase 3: Budget cut
    # ================================================================
    budget = max_budget
    budget_remaining = budget - grabbed_total
    cheap_ka_threshold = max_standalone_assoc / 2

    if combo_total > budget_remaining:
        kept_combos = []
        running = 0
        skipped = 0

        for q in combo_queries:
            if q["level"] == "S_combo_ka" and q["estimated_count"] <= cheap_ka_threshold:
                kept_combos.append(q)
                running += q["estimated_count"]
            elif running + q["estimated_count"] <= budget_remaining:
                kept_combos.append(q)
                running += q["estimated_count"]
            else:
                skipped += 1

        print(f"  Phase 3: budget cut — kept {len(kept_combos)}, skipped {skipped}. "
              f"Total: ~{grabbed_total + running:,d} / {budget:,d}")
        combo_queries = kept_combos
        combo_total = running
    else:
        print(f"  Phase 3: all combos fit (~{grabbed_total + combo_total:,d} / {budget:,d})")

    all_queries = list(grabbed) + list(combo_queries)
    current_total = grabbed_total + combo_total

    # ================================================================
    # Phase 4: Fill remaining budget
    # ================================================================
    filled = []
    if current_total < budget:
        fired_descs = {q["description"] for q in all_queries}
        remaining_all = []

        for desc, info in remaining_key.items():
            if desc not in fired_descs:
                remaining_all.append(info)
        for desc, info in remaining_assoc.items():
            if desc not in fired_descs:
                remaining_all.append(info)

        remaining_all.sort(key=lambda x: x["count"])

        for info in remaining_all:
            if current_total + info["count"] > budget:
                continue
            p = info["piece"]
            q = {
                "type": "cnf",
                "cnf": p["cnf"],
                "description": p["description"],
                "estimated_count": info["count"],
                "level": "S_fill",
            }
            all_queries.append(q)
            current_total += info["count"]
            filled.append({"description": p["description"], "count": info["count"]})

            if verbose:
                print(f"  Fill: {p['description']}: {info['count']:>10,d}  (total: {current_total:,d})")

        print(f"  Phase 4: filled {len(filled)} (~{sum(f['count'] for f in filled):,d} docs). "
              f"Final: ~{current_total:,d} / {budget:,d}")
    else:
        print(f"  Phase 4: skipped (at budget: ~{current_total:,d} / {budget:,d})")

    elapsed = time.perf_counter() - t0

    print(f"\n  Done in {elapsed:.1f}s — {len(all_queries)} queries, ~{current_total:,d} estimated docs")

    return {
        "query": expansion.get("text", ""),
        "model": expansion.get("model", ""),
        "all_queries": all_queries,
        "grabbed": [q for q in all_queries if q["level"].startswith("S0")],
        "combo_queries": combo_queries,
        "filled": filled,
        "all_key_pieces": all_key_pieces,
        "all_assoc_pieces": all_assoc_pieces,
        "grabbed_total": grabbed_total,
        "combo_total": combo_total,
        "remaining_assoc_total": remaining_assoc_total,
        "current_total": current_total,
        "max_budget": max_budget,
        "time": elapsed,
    }


# ================================================================
# Full experiment
# ================================================================

def run_experiment(
    query: str,
    model: str = "claude-sonnet-4-20250514",
    tokenizer=None,
    index: str = DEFAULT_INDEX,
    cache_path: str = "./api_expansions.jsonl",
    max_standalone_key: int = 1500,
    max_budget: int = 20000,
    retrieve_docs: bool = True,
    verbose: bool = True,
):
    """
    Full experiment: get expansions → peek and grab → retrieve documents.

    Args:
        query: Search query string.
        model: LLM model for expansion generation.
        tokenizer: HuggingFace tokenizer (loaded if None).
        index: Infini-gram index name.
        retrieve_docs: If True, actually pull documents from the index.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            add_bos_token=False,
            add_eos_token=False,
        )

    print(f"\nQuery: {query}")
    print(f"Model: {model}")
    print(f"Index: {index}")

    # Get expansions
    expansion = get_expansions(query, model=model, cache_path=cache_path)
    if expansion is None:
        print("Failed to get expansions")
        return None

    if verbose:
        n_aspects = len(expansion.get("KEY_ENTITIES", {}))
        n_assoc = len(expansion.get("ASSOCIATED_TERMS", expansion.get("ASSOCIATED", [])))
        print(f"Expansions: {n_aspects} aspects, {n_assoc} associated terms")

    # Peek and grab
    result = peek_and_grab(
        expansion=expansion,
        tokenizer=tokenizer,
        index=index,
        max_standalone_key=max_standalone_key,
        max_budget=max_budget,
        verbose=verbose,
    )

    result["query"] = query
    result["model"] = model
    result["expansion"] = expansion

    # Retrieve documents
    if retrieve_docs and result["all_queries"]:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Retrieving documents...")
            print(f"{'='*70}")

        documents = retrieve_documents(
            result["all_queries"],
            tokenizer=tokenizer,
            index=index,
            verbose=verbose,
        )
        result["documents"] = documents
        result["n_documents"] = len(documents)
    else:
        result["documents"] = []
        result["n_documents"] = 0

    return result