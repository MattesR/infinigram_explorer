"""
gold_mining.py -- mine CNF terms from gold documents and measure the CNF ceiling.

For one TREC query this script
  1. loads the relevant segments (qrels grade >= 1) and their text,
  2. splits them into K folds by web domain, so one site never sits on both sides,
  3. extracts candidate phrases (word 1-4-grams) from the mining documents (grade >= min_grade),
  4. counts every candidate in the infini-gram index (exact phrase, case forms OR'ed),
  5. selects terms per fold with Robertson's offer weight (relevance feedback) and
     credits each selected term only with relevant segments of the held-out fold,
  6. builds windowed pairs (a AND b within `window` tokens) among the selected terms,
  7. runs a greedy budgeted oracle and reports recall at several budgets.

Three vocabularies go through the same procedure:
  mined_heldout   terms mined from gold docs, selected on the other folds   <- the real ceiling
  mined_insample  same terms, selected and credited on the same documents  <- what hindsight claims
  opus_heldout    Opus expansion terms (as exact phrases), held-out        <- today's vocabulary
BM25 recall at InfiniSearch's average pool size (25,662 docs) is printed as a reference.

Everything is printed and written to <out_dir>/<qid>/.

Python:
    from gold_mining import analyze_query
    res = analyze_query("2024-32912", engine, tokenizer, corpus_dir="/path/to/msmarco_v2.1_doc_segmented")

CLI:
    python gold_mining.py --qid 2024-32912 --index_dir /home/mruc/msmarco_segmented_index/ \
        --corpus_dir /path/to/msmarco_v2.1_doc_segmented

Notes
  * Costs are index counts (occurrences, summed over queries). They overstate the number of
    unique documents a query set retrieves.
  * Local matching runs on the segment text re-tokenised with the same tokenizer. It should agree
    with the engine; terms or pairs that match gold docs locally but have count 0 in the index
    are reported as warnings and kept out of the oracle.
"""

import argparse
import csv
import gzip
import json
import os
import pickle
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import combinations
from math import log
from urllib.parse import urlparse

from llm_keyword_filter import STOPWORDS, load_all_expansions
from trec_output import load_qrels


DEFAULTS = dict(
    qrels_path="./qrels.rag24.test-umbrela-all.txt",
    topics_path="./topics_rag24_test.txt",
    expansions_path="./kiss.jsonl",          # Opus expansions used in the paper
    corpus_dir=None,                         # folder with msmarco_v2.1_doc_segmented_XX.json.gz
    out_dir="./gold_mining",
    bm25_pool_pkl="./bm_25_baseline_results_large_pool.pkl",
    n_docs_in_index=113_520_750,             # N for the RSJ weight: segments in MS MARCO v2.1 (check)
    min_grade=2,           # mine only from segments with grade >= min_grade
    n_folds=5,
    max_words=4,           # longest candidate phrase, in words
    max_tokens=16,         # longest candidate phrase, in tokens (local matching index)
    min_support=2,         # a mined term must occur in >= this many parent docs of the mining set
    max_candidates=1500,   # cap on mined candidates that get counted (highest support first)
    top_k=30,              # terms kept per fold by offer weight
    window=100,            # max_diff_tokens for pairs
    broad_floor=1000,      # a pair needs at least one term with count >= this
    min_pair_cover=2,      # a pair must cover >= this many held-out relevant segments
    max_pairs=400,         # cap on pairs that get counted
    beta=10.0,             # smoothing in the greedy score gain / (count + beta)
    budgets=(1_000, 5_000, 10_000, 25_000, 40_000, 100_000),
    max_clause_freq=80_000_000,
    show=40,               # rows to print per table
)


# ----------------------------------------------------------------------------
# 1. Relevant documents and folds
# ----------------------------------------------------------------------------

_DOCID_RE = re.compile(r'"docid":\s*"([^"]+)"')


def load_topics(path):
    """TSV topics file -> {qid: query text}."""
    topics = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def fetch_segments(doc_ids, corpus_dir):
    """Read segments from the gzipped MS MARCO v2.1 segmented corpus.

    Segment id msmarco_v2.1_doc_30_346287628#1_805120016 lives in file 30.
    Returns {doc_id: {"text", "url", "title"}}. Each needed file is scanned once.
    """
    by_file = defaultdict(set)
    for did in doc_ids:
        by_file[int(did.split("_")[3])].add(did)

    found = {}
    for fnum, wanted in sorted(by_file.items()):
        path = os.path.join(corpus_dir, f"msmarco_v2.1_doc_segmented_{fnum:02d}.json.gz")
        print(f"  scanning {path} for {len(wanted)} segments")
        with gzip.open(path, "rt") as f:
            for line in f:
                m = _DOCID_RE.search(line, 0, 200)
                if not m or m.group(1) not in wanted:
                    continue
                obj = json.loads(line)
                found[obj["docid"]] = {
                    "text": obj.get("segment", ""),
                    "url": obj.get("url", ""),
                    "title": obj.get("title", ""),
                }
                wanted.discard(obj["docid"])
                if not wanted:
                    break
    return found


def relevant_ids(qrels, qid):
    """{doc_id: grade} for all segments with grade >= 1."""
    return {d: g for d, g in qrels.get(qid, {}).items() if g >= 1}


def prefetch_relevant_docs(qids, qrels, corpus_dir, out_dir):
    """Fetch the relevant segments of several queries in one corpus pass and cache them per query."""
    todo = [q for q in qids if not os.path.exists(_docs_cache_path(out_dir, q))]
    if not todo:
        return
    all_ids = set()
    for q in todo:
        all_ids |= set(relevant_ids(qrels, q))
    texts = fetch_segments(all_ids, corpus_dir)
    for q in todo:
        _write_docs_cache(out_dir, q, relevant_ids(qrels, q), texts)


def _docs_cache_path(out_dir, qid):
    return os.path.join(out_dir, qid, "relevant_docs.jsonl")


def _write_docs_cache(out_dir, qid, grades, texts):
    path = _docs_cache_path(out_dir, qid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for did, g in grades.items():
            if did in texts:
                f.write(json.dumps({"doc_id": did, "grade": g, **texts[did]}) + "\n")


def load_relevant_docs(qid, qrels, corpus_dir, out_dir):
    """Relevant segments of one query as dicts with doc_id, grade, parent, group, text.

    parent = the web page the segment belongs to; group = its domain (used for folds).
    Texts are cached in <out_dir>/<qid>/relevant_docs.jsonl.
    """
    grades = relevant_ids(qrels, qid)
    path = _docs_cache_path(out_dir, qid)
    if not os.path.exists(path):
        if corpus_dir is None:
            raise ValueError(f"No cached texts at {path}; pass corpus_dir to fetch them.")
        _write_docs_cache(out_dir, qid, grades, fetch_segments(grades, corpus_dir))

    docs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            did = rec["doc_id"]
            parent = did.split("#")[0]
            domain = urlparse(rec.get("url", "")).netloc.lower().removeprefix("www.")
            docs.append({
                "doc_id": did,
                "grade": grades.get(did, rec["grade"]),
                "parent": parent,
                "group": domain or parent,
                "text": rec["text"],
            })
    n_missing = len(grades) - len(docs)
    return docs, n_missing


def assign_folds(docs, n_folds):
    """Put whole domains into folds: largest domain first, always into the smallest fold."""
    group_size = Counter(d["group"] for d in docs)
    n_folds = min(n_folds, len(group_size))
    fold_size = [0] * n_folds
    fold_of = {}
    for group, size in group_size.most_common():
        f = fold_size.index(min(fold_size))
        fold_of[group] = f
        fold_size[f] += size
    for d in docs:
        d["fold"] = fold_of[d["group"]]
    return n_folds


def make_splits(docs, n_folds, min_grade, held_out=True):
    """List of (mining doc indices, evaluation doc indices).

    held_out=True : one split per fold. Mining = grade >= min_grade outside the fold,
                    evaluation = every relevant segment inside the fold.
    held_out=False: one split. Mining = all grade >= min_grade, evaluation = all relevant.
    """
    mine = [i for i, d in enumerate(docs) if d["grade"] >= min_grade]
    if not held_out:
        return [(mine, list(range(len(docs))))]
    return [
        ([i for i in mine if docs[i]["fold"] != f],
         [i for i, d in enumerate(docs) if d["fold"] == f])
        for f in range(n_folds)
    ]


# ----------------------------------------------------------------------------
# 2. Candidate phrases
# ----------------------------------------------------------------------------

_CHUNK_SPLIT = re.compile(r'[.,;:!?()\[\]{}"“”|/\n\t•]+')


def _words(chunk):
    words = [w.strip("'’-*_#") for w in chunk.split()]
    return [w for w in words if w]


def _clean_edges(gram):
    """First and last word are no stopwords; numbers only as single words; no 1-letter words."""
    first, last = gram[0].lower(), gram[-1].lower()
    if first in STOPWORDS or last in STOPWORDS:
        return False
    if len(gram) == 1:
        return len(first) >= 2
    return not (first.isdigit() or last.isdigit())


def extract_candidates(mining_docs, max_words, min_support):
    """Word 1..max_words-grams that never cross punctuation and have clean edges.

    Returns {lowercased phrase: {"phrase": most frequent surface form, "support": #parent docs}},
    keeping phrases found in >= min_support parent docs.
    """
    parents = defaultdict(set)
    forms = defaultdict(Counter)
    for d in mining_docs:
        for chunk in _CHUNK_SPLIT.split(d["text"]):
            words = _words(chunk)
            for n in range(1, max_words + 1):
                for i in range(len(words) - n + 1):
                    gram = words[i:i + n]
                    if not _clean_edges(gram):
                        continue
                    phrase = " ".join(gram)
                    key = phrase.lower()
                    parents[key].add(d["parent"])
                    forms[key][phrase] += 1
    return {
        key: {"phrase": forms[key].most_common(1)[0][0], "support": len(p)}
        for key, p in parents.items() if len(p) >= min_support
    }


def opus_phrases(qid, expansions_path):
    """All Opus terms of a query (aspect names, their expansions, associated terms)."""
    data = load_all_expansions(expansions_path).get(qid, {})
    phrases = []
    for aspect, terms in data.get("KEY_ENTITIES", {}).items():
        phrases.append(aspect)
        if isinstance(terms, list):
            phrases.extend(terms)
    phrases.extend(data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", [])))
    return {p.strip().lower(): p.strip() for p in phrases if p.strip()}


# ----------------------------------------------------------------------------
# 3. Matching and counting
# ----------------------------------------------------------------------------

def index_tokens(tokens, max_len):
    """Map every token n-gram (up to max_len tokens) of a document to its start positions."""
    idx = defaultdict(list)
    for i in range(len(tokens)):
        for n in range(1, min(max_len, len(tokens) - i) + 1):
            idx[tuple(tokens[i:i + n])].append(i)
    return idx


def phrase_variants(phrase, tokenizer, max_len):
    """Token ids of the case forms: as written, lowercase, first letter capitalised."""
    lower = phrase.lower()
    out = []
    for form in dict.fromkeys([phrase, lower, lower[:1].upper() + lower[1:]]):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if ids and len(ids) <= max_len and ids not in out:
            out.append(ids)
    return out


def find_positions(doc, variants):
    """Start positions of any variant in a document (sorted)."""
    pos = []
    for v in variants:
        pos.extend(doc["ngrams"].get(tuple(v), []))
    return sorted(pos)


def within_window(pos_a, pos_b, window):
    """True if some occurrence of a and some occurrence of b start <= window tokens apart."""
    for i in pos_a:
        j = bisect_left(pos_b, i - window)
        if j < len(pos_b) and pos_b[j] <= i + window:
            return True
    return False


def count_piece(engine, variants, cache, cfg):
    """Index count of one exact-phrase piece (its case forms OR'ed in one clause)."""
    key = ("piece", tuple(map(tuple, variants)))
    if key not in cache:
        res = engine.count_cnf([variants], max_clause_freq=cfg["max_clause_freq"])
        cache[key] = res.get("count", 0)
    return cache[key]


def count_pair(engine, va, vb, cache, cfg):
    """Index count of (a) AND (b) within cfg['window'] tokens. Returns (count, approx)."""
    a, b = sorted([tuple(map(tuple, va)), tuple(map(tuple, vb))])
    key = ("pair", a, b, cfg["window"])
    if key not in cache:
        res = engine.count_cnf([va, vb], max_clause_freq=cfg["max_clause_freq"],
                               max_diff_tokens=cfg["window"])
        cache[key] = (res.get("count", 0), bool(res.get("approx", False)))
    return cache[key]


def prepare_terms(phrases, docs, tokenizer, engine, cache, cfg):
    """For each phrase: token variants, positions in every relevant doc, index count."""
    terms = {}
    for key, phrase in phrases.items():
        variants = phrase_variants(phrase, tokenizer, cfg["max_tokens"])
        if not variants:
            continue
        positions = {}
        for i, d in enumerate(docs):
            p = find_positions(d, variants)
            if p:
                positions[i] = p
        terms[key] = {
            "key": key,
            "phrase": phrase,
            "variants": variants,
            "positions": positions,
            "docs": set(positions),
            "count": count_piece(engine, variants, cache, cfg),
        }
    return terms


# ----------------------------------------------------------------------------
# 4. Relevance feedback: offer weight
# ----------------------------------------------------------------------------

def offer_weight(r, R, n, N):
    """Robertson-Sparck Jones weight w and offer weight r*w.

    r of R relevant units contain the term, n of N documents in the corpus.
    n is an occurrence count here, so it is clipped to >= r.
    """
    n = max(n, r)
    w = log(((r + 0.5) * (N - n - R + r + 0.5)) / ((R - r + 0.5) * (n - r + 0.5)))
    return r * w, w


def score_terms(terms, docs, mining_idx, cfg, min_support):
    """Offer weight of every term on the mining documents, counted per parent doc. Sorted by OW."""
    mining = set(mining_idx)
    R = len({docs[i]["parent"] for i in mining})
    rows = []
    for t in terms.values():
        r = len({docs[i]["parent"] for i in t["docs"] & mining})
        ow, w = offer_weight(r, R, t["count"], cfg["n_docs_in_index"]) if r else (0.0, 0.0)
        rows.append({
            "key": t["key"], "phrase": t["phrase"], "r": r, "R": R,
            "count": t["count"], "w": w, "ow": ow,
            "eligible": r >= min_support and w > 0 and t["count"] > 0,
        })
    rows.sort(key=lambda x: -x["ow"])
    return rows


# ----------------------------------------------------------------------------
# 5.-6. Selection per split, pairs, oracle
# ----------------------------------------------------------------------------

def _contains(a, b):
    return f" {a} " in f" {b} "


def build_pairs(terms, per_split, splits, engine, cache, cfg):
    """Windowed pairs among selected terms, credited on held-out docs where both were selected."""
    keys = sorted(set().union(*per_split)) if per_split else []
    local = []
    for a, b in combinations(keys, 2):
        if _contains(a, b) or _contains(b, a):
            continue
        ta, tb = terms[a], terms[b]
        if max(ta["count"], tb["count"]) < cfg["broad_floor"]:
            continue
        H = set()
        for selected, (_, eval_idx) in zip(per_split, splits):
            if a in selected and b in selected:
                for i in set(eval_idx) & ta["docs"] & tb["docs"]:
                    if within_window(ta["positions"][i], tb["positions"][i], cfg["window"]):
                        H.add(i)
        if len(H) >= cfg["min_pair_cover"]:
            local.append((len(H), a, b, H))
    local.sort(key=lambda x: -x[0])

    pairs, warnings = [], []
    for _, a, b, H in local[:cfg["max_pairs"]]:
        ta, tb = terms[a], terms[b]
        count, approx = count_pair(engine, ta["variants"], tb["variants"], cache, cfg)
        query = f"({ta['phrase']}) AND ({tb['phrase']})"
        if count == 0:
            warnings.append(f"pair matches {len(H)} gold docs locally but count 0: {query}")
            continue
        pairs.append({"kind": "pair", "query": query, "count": count, "approx": approx, "H": H})
    return pairs, warnings


def greedy_oracle(candidates, docs, cfg):
    """Pick, step by step, the candidate with most new relevant docs per count, within the budget."""
    n_all = len(docs)
    n_top = sum(d["grade"] >= cfg["min_grade"] for d in docs) or 1
    b_max = cfg["budgets"][-1]
    pool = [c for c in candidates if c["H"] and c["count"] > 0]
    covered, cost, trace = set(), 0, []
    while True:
        best, best_score = None, 0.0
        for c in pool:
            if cost + c["count"] > b_max:
                continue
            gain = len(c["H"] - covered)
            if gain and gain / (c["count"] + cfg["beta"]) > best_score:
                best, best_score = c, gain / (c["count"] + cfg["beta"])
        if best is None:
            break
        new = best["H"] - covered
        covered |= new
        cost += best["count"]
        pool.remove(best)
        trace.append({
            "step": len(trace) + 1, "kind": best["kind"], "query": best["query"],
            "count": best["count"], "new_docs": len(new), "cum_cost": cost,
            "recall": len(covered) / n_all,
            "recall_top": sum(docs[i]["grade"] >= cfg["min_grade"] for i in covered) / n_top,
        })
    return trace


def recall_at_budgets(trace, budgets, key="recall"):
    out = {}
    for B in budgets:
        steps = [t for t in trace if t["cum_cost"] <= B]
        out[B] = steps[-1][key] if steps else 0.0
    return out


def run_pool(terms, docs, splits, engine, cache, cfg, min_support):
    """Select terms per split, credit them on that split's evaluation docs, add pairs, run the oracle."""
    selected = {}
    per_split = []
    for s, (mining_idx, eval_idx) in enumerate(splits):
        rows = [r for r in score_terms(terms, docs, mining_idx, cfg, min_support) if r["eligible"]]
        chosen = rows[:cfg["top_k"]]
        per_split.append({r["key"] for r in chosen})
        ev = set(eval_idx)
        for r in chosen:
            entry = selected.setdefault(r["key"], {
                "kind": "single", "key": r["key"], "query": r["phrase"],
                "count": r["count"], "splits": [], "best_ow": 0.0, "H": set(),
            })
            entry["splits"].append(s)
            entry["best_ow"] = max(entry["best_ow"], r["ow"])
            entry["H"] |= terms[r["key"]]["docs"] & ev

    singles = sorted(selected.values(), key=lambda e: (-len(e["H"]), e["count"]))
    pairs, warnings = build_pairs(terms, per_split, splits, engine, cache, cfg)
    for t in terms.values():
        if t["count"] == 0 and t["docs"]:
            warnings.append(f"term matches {len(t['docs'])} gold docs locally but count 0: {t['phrase']}")
    trace = greedy_oracle(singles + pairs, docs, cfg)
    return {
        "singles": singles, "pairs": pairs, "trace": trace, "warnings": warnings,
        "recall_at": recall_at_budgets(trace, cfg["budgets"]),
        "recall_top_at": recall_at_budgets(trace, cfg["budgets"], key="recall_top"),
    }


# ----------------------------------------------------------------------------
# Reference, printing, saving
# ----------------------------------------------------------------------------

_BM25 = None


def bm25_reference(qid, pkl_path):
    """BM25 recall at the InfiniSearch pool size, from the existing large-pool run."""
    global _BM25
    if not pkl_path or not os.path.exists(pkl_path):
        return None
    if _BM25 is None:
        print("  loading BM25 large-pool results (one-off, ~400 MB)")
        with open(pkl_path, "rb") as f:
            _BM25 = {r["qid"]: {"pool": r["n_retrieved"], "recall": r["recall"]} for r in pickle.load(f)}
    return _BM25.get(qid)


def _load_cache(out_dir):
    path = os.path.join(out_dir, "count_cache.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(out_dir, cache):
    with open(os.path.join(out_dir, "count_cache.pkl"), "wb") as f:
        pickle.dump(cache, f)


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt_budget(B):
    return f"{B // 1000}k"


def print_header(qid, query, docs, n_missing, n_folds, cfg):
    grades = Counter(d["grade"] for d in docs)
    top = [d for d in docs if d["grade"] >= cfg["min_grade"]]
    fold_sizes = Counter(d["fold"] for d in docs)
    print("=" * 100)
    print(f"{qid}: {query}")
    print("=" * 100)
    print(f"relevant segments with text: {len(docs)}  (missing text: {n_missing})  "
          f"grades: {dict(sorted(grades.items(), reverse=True))}")
    print(f"mining set (grade >= {cfg['min_grade']}): {len(top)} segments from "
          f"{len({d['parent'] for d in top})} pages, {len({d['group'] for d in top})} domains")
    print(f"folds: {n_folds} by domain, relevant segments per fold: "
          f"{[fold_sizes[f] for f in range(n_folds)]}")


def print_candidates(rows, heldout, n_total, cfg):
    sel = {e["key"]: e for e in heldout["singles"]}
    print(f"\nMINED CANDIDATES: top {cfg['show']} of {n_total} by offer weight on all grade>="
          f"{cfg['min_grade']} docs (in-sample, for reading only)")
    print(f"  {'phrase':<40} {'pages':>5} {'count':>11} {'w':>6} {'OW':>7}   held-out: folds  H")
    for r in rows[:cfg["show"]]:
        e = sel.get(r["key"])
        held = f"{str(e['splits']):<14} {len(e['H']):>3}" if e else "-"
        print(f"  {r['phrase'][:40]:<40} {r['r']:>5} {r['count']:>11,} {r['w']:>6.2f} {r['ow']:>7.1f}   {held}")


def print_pool(name, res, docs, cfg):
    n = len(docs)
    print(f"\n--- {name} " + "-" * (95 - len(name)))
    print(f"selected terms: {len(res['singles'])}   pairs counted: {len(res['pairs'])}")
    print(f"  {'selected term':<40} {'count':>11} {'H':>4}  folds")
    for e in res["singles"][:cfg["show"]]:
        print(f"  {e['query'][:40]:<40} {e['count']:>11,} {len(e['H']):>4}  {e['splits']}")
    if res["pairs"]:
        print(f"  {'top pairs':<60} {'count':>11} {'H':>4}")
        for p in sorted(res["pairs"], key=lambda p: -len(p["H"]))[:cfg["show"] // 2]:
            flag = " (approx)" if p["approx"] else ""
            print(f"  {p['query'][:60]:<60} {p['count']:>11,} {len(p['H']):>4}{flag}")
    print(f"  oracle trace (first {cfg['show']} of {len(res['trace'])} steps), recall over {n} relevant segments")
    print(f"  {'step':>4} {'query':<60} {'count':>9} {'new':>4} {'cum cost':>10} {'recall':>7}")
    for t in res["trace"][:cfg["show"]]:
        print(f"  {t['step']:>4} {t['query'][:60]:<60} {t['count']:>9,} {t['new_docs']:>4} "
              f"{t['cum_cost']:>10,} {t['recall']:>7.3f}")
    for w in res["warnings"][:10]:
        print(f"  WARNING {w}")
    if len(res["warnings"]) > 10:
        print(f"  ... {len(res['warnings']) - 10} more warnings")


def print_comparison(results, bm25, cfg):
    budgets = cfg["budgets"]
    print("\nRECALL AT BUDGET (summed index counts; all relevant segments, grade >= 1)")
    print(f"  {'pool':<16}" + "".join(f"{_fmt_budget(B):>8}" for B in budgets))
    for name, res in results.items():
        print(f"  {name:<16}" + "".join(f"{res['recall_at'][B]:>8.3f}" for B in budgets))
    if bm25:
        print(f"  BM25 reference: recall {bm25['recall']:.3f} at {bm25['pool']:,} unique docs")


def save_outputs(qdir, cand_rows, results, summary):
    _write_csv(os.path.join(qdir, "candidates_mined.csv"), cand_rows,
               ["phrase", "r", "R", "count", "w", "ow", "eligible", "heldout_folds", "heldout_H"])
    for name, res in results.items():
        _write_csv(os.path.join(qdir, f"{name}_selected.csv"),
                   [{**e, "H": len(e["H"]), "splits": e["splits"]} for e in res["singles"]],
                   ["query", "count", "H", "splits", "best_ow"])
        _write_csv(os.path.join(qdir, f"{name}_pairs.csv"),
                   [{**p, "H": len(p["H"])} for p in res["pairs"]],
                   ["query", "count", "approx", "H"])
        _write_csv(os.path.join(qdir, f"{name}_oracle.csv"), res["trace"],
                   ["step", "kind", "query", "count", "new_docs", "cum_cost", "recall", "recall_top"])
    with open(os.path.join(qdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

def analyze_query(qid, engine, tokenizer, query_text=None, **overrides):
    """Run the full analysis for one query; prints everything and saves to <out_dir>/<qid>/."""
    cfg = {**DEFAULTS, **overrides}
    qdir = os.path.join(cfg["out_dir"], qid)
    os.makedirs(qdir, exist_ok=True)
    cache = _load_cache(cfg["out_dir"])
    qrels = load_qrels(cfg["qrels_path"])
    if query_text is None:
        query_text = load_topics(cfg["topics_path"]).get(qid, "")

    # 1. relevant documents, tokens, folds
    docs, n_missing = load_relevant_docs(qid, qrels, cfg["corpus_dir"], cfg["out_dir"])
    for d in docs:
        d["tokens"] = tokenizer.encode(d["text"], add_special_tokens=False)
        d["ngrams"] = index_tokens(d["tokens"], cfg["max_tokens"])
    n_folds = assign_folds(docs, cfg["n_folds"])
    heldout_splits = make_splits(docs, n_folds, cfg["min_grade"], held_out=True)
    insample_splits = make_splits(docs, n_folds, cfg["min_grade"], held_out=False)
    print_header(qid, query_text, docs, n_missing, n_folds, cfg)

    # 2.-3. candidates, matched and counted
    mining_docs = [d for d in docs if d["grade"] >= cfg["min_grade"]]
    mined = extract_candidates(mining_docs, cfg["max_words"], cfg["min_support"])
    n_mined = len(mined)
    mined = dict(sorted(mined.items(), key=lambda kv: (-kv[1]["support"], -len(kv[0])))[:cfg["max_candidates"]])
    print(f"\nmined candidates: {n_mined} with support >= {cfg['min_support']}, counting {len(mined)}")
    mined_terms = prepare_terms({k: v["phrase"] for k, v in mined.items()}, docs, tokenizer, engine, cache, cfg)
    opus_terms = prepare_terms(opus_phrases(qid, cfg["expansions_path"]), docs, tokenizer, engine, cache, cfg)
    _save_cache(cfg["out_dir"], cache)

    # 4.-6. selection, pairs, oracle for each vocabulary
    results = {
        "mined_heldout": run_pool(mined_terms, docs, heldout_splits, engine, cache, cfg, cfg["min_support"]),
        "mined_insample": run_pool(mined_terms, docs, insample_splits, engine, cache, cfg, cfg["min_support"]),
        "opus_heldout": run_pool(opus_terms, docs, heldout_splits, engine, cache, cfg, 1),
    }
    _save_cache(cfg["out_dir"], cache)

    # report
    cand_rows = score_terms(mined_terms, docs, insample_splits[0][0], cfg, cfg["min_support"])
    held = {e["key"]: e for e in results["mined_heldout"]["singles"]}
    for r in cand_rows:
        e = held.get(r["key"])
        r["heldout_folds"] = e["splits"] if e else []
        r["heldout_H"] = len(e["H"]) if e else 0
    print_candidates(cand_rows, results["mined_heldout"], len(mined_terms), cfg)
    for name, res in results.items():
        print_pool(name, res, docs, cfg)
    bm25 = bm25_reference(qid, cfg["bm25_pool_pkl"])
    print_comparison(results, bm25, cfg)

    summary = {
        "qid": qid, "query": query_text, "n_relevant": len(docs), "n_missing_text": n_missing,
        "n_folds": n_folds, "bm25": bm25,
        "config": {k: v for k, v in cfg.items() if k != "budgets"} | {"budgets": list(cfg["budgets"])},
        "recall_at": {name: res["recall_at"] for name, res in results.items()},
        "recall_top_at": {name: res["recall_top_at"] for name, res in results.items()},
    }
    save_outputs(qdir, cand_rows, results, summary)
    print(f"\nsaved to {qdir}/")
    return {"docs": docs, "mined_terms": mined_terms, "opus_terms": opus_terms,
            "results": results, "summary": summary}


def load_resources(index_dir, tokenizer_name):
    """Tokenizer and engine, loaded as in run_experiment.py."""
    from transformers import AutoTokenizer
    from infini_gram.engine import InfiniGramEngine
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_bos_token=False, add_eos_token=False)
    engine = InfiniGramEngine(index_dir=index_dir, eos_token_id=tokenizer.eos_token_id)
    return tokenizer, engine


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", nargs="+", help="one or more query ids")
    ap.add_argument("--all_topics", action="store_true", help="run every topic in the topics file")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--corpus_dir", default=None)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    for key in ["qrels_path", "topics_path", "expansions_path", "out_dir", "bm25_pool_pkl"]:
        ap.add_argument(f"--{key}", default=DEFAULTS[key])
    for key in ["min_grade", "n_folds", "top_k", "window", "max_candidates"]:
        ap.add_argument(f"--{key}", type=int, default=DEFAULTS[key])
    args = ap.parse_args()

    overrides = {k: getattr(args, k) for k in [
        "corpus_dir", "qrels_path", "topics_path", "expansions_path", "out_dir", "bm25_pool_pkl",
        "min_grade", "n_folds", "top_k", "window", "max_candidates"]}
    qids = list(load_topics(args.topics_path)) if args.all_topics else args.qid
    if not qids:
        ap.error("give --qid or --all_topics")

    if len(qids) > 1 and args.corpus_dir:
        prefetch_relevant_docs(qids, load_qrels(args.qrels_path), args.corpus_dir, args.out_dir)

    tokenizer, engine = load_resources(args.index_dir, args.tokenizer)
    summaries = [analyze_query(q, engine, tokenizer, **overrides)["summary"] for q in qids]

    if len(summaries) > 1:
        budgets = DEFAULTS["budgets"]
        print("\nMEAN RECALL AT BUDGET over", len(summaries), "queries")
        print(f"  {'pool':<16}" + "".join(f"{_fmt_budget(B):>8}" for B in budgets))
        rows = []
        for name in summaries[0]["recall_at"]:
            means = [sum(s["recall_at"][name][B] for s in summaries) / len(summaries) for B in budgets]
            print(f"  {name:<16}" + "".join(f"{m:>8.3f}" for m in means))
            rows.append({"pool": name, **{_fmt_budget(B): m for B, m in zip(budgets, means)}})
        bm = [s["bm25"]["recall"] for s in summaries if s["bm25"]]
        if bm:
            print(f"  BM25 reference: mean recall {sum(bm) / len(bm):.3f} at ~25.7k unique docs")
        _write_csv(os.path.join(args.out_dir, "summary_mean.csv"), rows,
                   ["pool"] + [_fmt_budget(B) for B in budgets])


if __name__ == "__main__":
    main()
