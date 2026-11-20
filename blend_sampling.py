"""
Sampling estimates for co-occurrence counts the infini-gram API cannot count exactly.

The API counts an AND (A within 1000 tokens of B) exactly only if both clauses have
<= 500k hits; above that it subsamples and its estimate is badly biased. Instead:

  1. `find` every surface form of the small side A: exact count and rank ranges.
  2. Draw n occurrences of A uniformly (without replacement) across all forms/shards.
  3. Fetch the text around each one (get_doc_by_rank) and check locally whether a
     form of B occurs within +-WINDOW tokens (and anywhere in the shown document).
  4. count(A AND B) ~= count(A) * p_hat, with a Wilson interval on p_hat.

    python blend_sampling.py validate --n 100     # check against exact AND counts
"""

import argparse
import json
import math
import os
import random
import re
import time

import requests

import blend_experiment as be

API_URL = be.API_URL
INDEX = be.DEFAULT_INDEX
WINDOW = be.CNF_MAX_DIFF_TOKENS   # same window as the API's AND
DISP = 20_000                     # tokens of document shown around the match
DELAY = 1.3                       # public API: ~1 call / 1.3 s, never parallel
RES = "blend_results"
_last = [0.0]


def _post(payload, retries=4):
    err = None
    for attempt in range(retries):
        wait = DELAY - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        if attempt:
            time.sleep(DELAY * 4 * 2 ** attempt)
        _last[0] = time.time()
        try:
            r = requests.post(API_URL, json={"index": INDEX, **payload}, timeout=120)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            continue
        if r.status_code == 403:
            raise RuntimeError("HTTP 403 from infini-gram API: stop and wait before retrying")
        if r.status_code == 429:
            err = "HTTP 429"
            time.sleep(60)
            continue
        try:
            out = r.json()
        except Exception as e:
            err = f"HTTP {r.status_code} non-JSON: {e}"
            continue
        if "error" in out:
            err = str(out["error"])
            continue
        return out
    raise RuntimeError(f"API failed: {err}")


def find_forms(forms, tokenizer):
    """Exact occurrences of each pruned form, with rank ranges. One `find` call per form."""
    seqs = be.build_clause(forms, tokenizer, max_terms=10_000)
    out = []
    for ids in seqs:
        r = _post({"query_type": "find", "query_ids": ids})
        segs = [(s, a, b) for s, (a, b) in enumerate(r["segment_by_shard"]) if b > a]
        out.append({"form": tokenizer.decode(ids), "ids": ids, "count": r["cnt"], "segs": segs})
    return out


def draw(finds, n, rng):
    """n occurrences uniformly without replacement over all forms and shards."""
    total = sum(f["count"] for f in finds)
    picks = sorted(rng.sample(range(total), min(n, total)))
    out, i, base = [], 0, 0
    flat = [(f, s, a, b) for f in finds for s, a, b in f["segs"]]
    for f, s, a, b in flat:
        size = b - a
        while i < len(picks) and picks[i] < base + size:
            out.append((f, s, a + picks[i] - base))
            i += 1
        base += size
    return out


def fetch(f, shard, rank):
    r = _post({"query_type": "get_doc_by_rank", "s": shard, "rank": rank,
               "max_disp_len": DISP, "query_ids": f["ids"]})
    spans = r["spans"]
    k = next(i for i, (_, tag) in enumerate(spans) if tag is not None)
    return {"doc_ix": r["doc_ix"], "doc_len": r["doc_len"], "disp_len": r["disp_len"],
            "left": "".join(t for t, _ in spans[:k]), "needle": spans[k][0],
            "right": "".join(t for t, _ in spans[k + 1:])}


def _pattern(forms):
    # Token n-grams start at a word boundary ("▁Alg...") but may end inside a word,
    # like the engine: "Algeria" also matches "Algerian".
    alts = sorted({f.strip() for f in forms if f.strip()}, key=len, reverse=True)
    return re.compile(r"(?<![0-9A-Za-zÀ-ÿ])(?:" + "|".join(map(re.escape, alts)) + ")")


def check(doc, pattern, tokenizer, window=WINDOW):
    """Does a B form occur within +-window tokens of the match / anywhere in the shown doc?"""
    lt = tokenizer.encode(doc["left"], add_special_tokens=False)
    rt = tokenizer.encode(doc["right"], add_special_tokens=False)
    left = tokenizer.decode(lt[-window:]) if lt else ""
    right = tokenizer.decode(rt[:window]) if rt else ""
    near = bool(pattern.search(left + " " + doc["needle"] + " " + right))
    anywhere = bool(pattern.search(doc["left"] + " " + doc["needle"] + " " + doc["right"]))
    return near, anywhere, doc["disp_len"] < doc["doc_len"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def estimate(a_forms, b_forms, tokenizer, n=100, seed=0, finds=None):
    """Sample n occurrences of A; estimate count(A within WINDOW of B) and docs-level share."""
    rng = random.Random(seed)
    finds = finds or find_forms(a_forms, tokenizer)
    N = sum(f["count"] for f in finds)
    pat = _pattern(b_forms)
    hits = []
    for f, s, rank in draw(finds, n, rng):
        d = fetch(f, s, rank)
        near, anywhere, trunc = check(d, pat, tokenizer)
        hits.append({"form": f["form"], "s": s, "rank": rank, "doc_ix": d["doc_ix"],
                     "near": near, "doc": anywhere, "trunc": trunc})
    k = sum(h["near"] for h in hits)
    m = len(hits)
    lo, hi = wilson(k, m)
    return {"N_a": N, "n": m, "k_near": k, "k_doc": sum(h["doc"] for h in hits),
            "n_trunc": sum(h["trunc"] for h in hits), "n_docs": len({(h["s"], h["doc_ix"]) for h in hits}),
            "est": N * k / m if m else None, "lo": N * lo, "hi": N * hi, "draws": hits}


def _tok():
    """LLaMA-2 tokenizer; uses the copy in answer_samples/tokenizer if present (no HF login needed)."""
    from transformers import AutoTokenizer
    local = os.path.join(PLAN_DIR, "tokenizer")
    src = local if os.path.isdir(local) else "meta-llama/Llama-2-7b-hf"
    return AutoTokenizer.from_pretrained(src, add_bos_token=False, add_eos_token=False)


def validation_pairs(k_and=8, seed=0):
    """Exact AND pairs from the answer counts (both sides <= cap), spread over sizes."""
    q = {r["key"]: r for r in be.load_results(os.path.join(RES, "answer_queries.jsonl"))}
    c = [r for r in be.load_results(os.path.join(RES, "answer_query_counts.jsonl"))
         if r.get("and_exact") and r.get("and_count") and len(r["and_sides"]) == 2]
    c.sort(key=lambda r: r["and_count"])
    pick = [c[int(i * (len(c) - 1) / (k_and - 1))] for i in range(k_and)]
    pairs = []
    for r in pick:
        g = q[r["key"]]["and"]
        pairs.append({"name": f"{g[0]['text']} AND {g[1]['text']}", "a": g[0]["variants"], "b": g[1]["variants"],
                      "exact": r["and_count"], "sides": [s["count"] for s in r["and_sides"]]})
    return pairs


REGION_TERMS = {"AS": ["Northeast India", "North East India"], "AZ": ["Azeris"], "ET": ["Habesha"],
                "NG": ["Northern Nigeria", "Northern Nigerian"], "JB": ["West Java"]}


def region_pairs(per_region=2, tokenizer=None):
    """Answer AND region-term pairs whose exact count the API can give (both sides <= cap)."""
    import csv
    share = sorted(d for d in os.listdir(os.path.join(RES, "share")) if not d.endswith(".zip"))[-1]
    rows = list(csv.DictReader(open(os.path.join(RES, "share", share, "answers.csv"))))
    q = {r["id"]: r for r in be.load_results(os.path.join(RES, "answer_queries.jsonl"))}
    pairs = []
    for split, terms in REGION_TERMS.items():
        cand = [r for r in rows if r["region_code"] == split and r["hits"] and r["generic"] == "False"
                and r["type"] in ("name", "common") and 2_000 <= int(r["hits"]) <= 300_000]
        cand.sort(key=lambda r: (-int(r["votes"]), r["answer_id"]))
        seen = set()
        for r in cand:
            if r["answer_id"] in seen or len(seen) == per_region:
                continue
            seen.add(r["answer_id"])
            a = [tokenizer.decode(s) for s in be.build_clause(be.answer_query_forms(q[r["answer_id"]]), tokenizer)]
            ca, cb = be.build_clause(a, tokenizer), be.build_clause(terms, tokenizer)
            exact = be._api_query([ca, cb])
            side_a = be._api_query([ca], cnf_params=False)["count"]
            if side_a > be.CNF_MAX_CLAUSE_FREQ or exact["approx"]:
                continue
            pairs.append({"name": f"{a[0]} AND {terms[0]}", "a": a, "b": terms, "exact": exact["count"],
                          "sides": [side_a, None], "a_only": True})
    return pairs


def _cap_forms(forms):
    out = []
    for f in forms:
        f = f.strip()
        if f:
            out.append(f)
            if f[0].islower():
                out.append(f[0].upper() + f[1:])
    return out


def run_targets(scope="top"):
    """Answers to sample: non-generic, counted, >0 hits; scope 'top' = top-voted group only."""
    import blend_review as br
    (templates, entities, phrases, phrase_counts, queries,
     raw_counts, new_counts, countries, regional) = br.load()
    where = {}
    for qid, regs in regional.items():
        for split, r in regs.items():
            for gi, g in enumerate(r["groups"]):
                if scope == "top" and gi > 0:
                    continue
                for key in g["keys"]:
                    q, c = queries.get(key), new_counts.get(key)
                    if not q or q["generic"] or not br.hits_now(c) or not c.get("count"):
                        continue
                    where.setdefault(key, set()).add((split, qid))
    phrase_pats = {qid: [(p["text"], _pattern(_cap_forms(p.get("variants") or [p["text"]])))
                         for p in r.get("phrases", [])] for qid, r in phrases.items()}
    return [(queries[k], sorted(w)) for k, w in sorted(where.items())], phrase_pats


PLAN_DIR = os.path.join(RES, "answer_samples")
PLAN = os.path.join(PLAN_DIR, "plan.json")


def make_plan(scope="top", n=100, seed=0, path=PLAN):
    """
    Freeze the work into plan.json: every answer to sample (with its surface forms and
    where it occurs), the question phrases and the region terms, and per region the
    answers in a fixed random order. Runs on any machine need only this file.
    """
    targets, _ = run_targets(scope)
    import blend_review as br
    phrases = {qid: [{"text": p["text"], "forms": _cap_forms(p.get("variants") or [p["text"]])}
                     for p in r.get("phrases", [])]
               for qid, r in br.load()[2].items()}
    answers, regions = {}, {s: [] for s in be.BLEND_SPLITS}
    for q, where in targets:
        answers[q["id"]] = {"key": q["key"], "answer": q["answer"], "type": q["type"],
                            "forms": be.answer_query_forms(q), "where": [list(w) for w in where]}
        for split in sorted({s for s, _ in where}):
            regions[split].append(q["id"])
    for split in regions:
        random.Random(f"{seed}:{split}").shuffle(regions[split])
    plan = {"created": time.strftime("%Y-%m-%d %H:%M"), "scope": scope, "n": n, "seed": seed,
            "window": WINDOW, "index": INDEX, "region_terms": be.BLEND_COUNTRY_TERMS,
            "phrases": phrases, "answers": answers, "regions": regions}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan, f, ensure_ascii=False)
    print(f"plan: {len(answers)} answers, " +
          ", ".join(f"{s} {len(v)}" for s, v in regions.items()) + f" → {path}")
    return plan


def sample_answer(aid, a, plan, phrase_pats, region_pats, tokenizer):
    """Draw n occurrences of one answer; check all regions and its question phrases in each.
    Returns (summary row, list of draw records)."""
    rng = random.Random(f"{plan['seed']}:{aid}")
    finds = find_forms(a["forms"], tokenizer)
    N = sum(f["count"] for f in finds)
    qids = sorted({qid for _, qid in a["where"]})
    reg = {s: {"near_core": 0, "doc_core": 0, "near_alt": 0, "doc_alt": 0} for s in region_pats}
    phr = {qid: [{"text": t, "near": 0, "doc": 0} for t, _ in phrase_pats.get(qid, [])] for qid in qids}
    trunc, docs, draws = 0, set(), []
    for f, s, rank in draw(finds, plan["n"], rng):
        d = fetch(f, s, rank)
        lt = tokenizer.encode(d["left"], add_special_tokens=False)
        rt = tokenizer.encode(d["right"], add_special_tokens=False)
        win = (tokenizer.decode(lt[-WINDOW:]) if lt else "") + " " + d["needle"] + " " + \
              (tokenizer.decode(rt[:WINDOW]) if rt else "")
        full = d["left"] + " " + d["needle"] + " " + d["right"]
        trunc += d["disp_len"] < d["doc_len"]
        docs.add((s, d["doc_ix"]))
        rec = {"id": aid, "form": f["form"], "s": s, "rank": rank, "doc_ix": d["doc_ix"],
               "doc_len": d["doc_len"], "near": [], "doc": []}
        for split, (pc, pa) in region_pats.items():
            for tier, p in (("core", pc), ("alt", pa)):
                if p.search(win):
                    reg[split]["near_" + tier] += 1
                    rec["near"].append(f"{split}:{tier}")
                if p.search(full):
                    reg[split]["doc_" + tier] += 1
                    rec["doc"].append(f"{split}:{tier}")
        for qid in qids:
            for i, (_, p) in enumerate(phrase_pats.get(qid, [])):
                if p.search(win):
                    phr[qid][i]["near"] += 1
                    rec["near"].append(f"{qid}:{i}")
                if p.search(full):
                    phr[qid][i]["doc"] += 1
                    rec["doc"].append(f"{qid}:{i}")
        rec["window"] = win
        draws.append(rec)
    m = len(draws)
    row = {"key": a["key"], "id": aid, "answer": a["answer"], "type": a["type"], "where": a["where"],
           "N": N, "n": m, "all": m == N, "n_docs": len(docs), "n_trunc": trunc,
           "forms": [{"form": f["form"], "count": f["count"]} for f in finds],
           "regions": reg, "phrases": phr, "error": None}
    return row, draws


def _paths(split, root=PLAN_DIR):
    return os.path.join(root, f"{split}.jsonl"), os.path.join(root, "draws", f"{split}.jsonl.gz")


def _done_rows(root=PLAN_DIR):
    """All finished answer rows in this folder, by answer id (from any region file)."""
    out = {}
    for s in be.BLEND_SPLITS:
        path = _paths(s, root)[0]
        if os.path.exists(path):
            for r in be.load_results(path):
                if not r.get("error") and not r.get("copied_from"):
                    out.setdefault(r["id"], (s, r))
    return out


def _lock(split, root=PLAN_DIR):
    """One process per region per folder. Returns False if another live process holds it."""
    import socket
    path = os.path.join(root, f"{split}.lock")
    host = socket.gethostname()
    if os.path.exists(path):
        h, pid = open(path).read().split()
        if h != host:
            print(f"{split}: locked by {h} (pid {pid}); delete {path} if that run is gone", flush=True)
            return False
        try:
            os.kill(int(pid), 0)
            print(f"{split}: already running here (pid {pid})", flush=True)
            return False
        except OSError:
            pass  # stale lock from a dead process
    with open(path, "w") as f:
        f.write(f"{host} {os.getpid()}")
    return True


def run(regions, root=PLAN_DIR):
    """
    Sample region by region. Per region two files: <root>/<SPLIT>.jsonl (one row per answer)
    and <root>/draws/<SPLIT>.jsonl.gz (the draws, one gzip member per answer). Restart-safe:
    answers already in the region file are skipped. An answer already sampled for another
    region in this folder is copied (no API calls) and marked copied_from.
    """
    import gzip
    plan = json.load(open(os.path.join(root, "plan.json")))
    tok = _tok()
    region_pats = {s: (_pattern(t["core"]), _pattern(t["alt"])) for s, t in plan["region_terms"].items()}
    phrase_pats = {qid: [(p["text"], _pattern(p["forms"])) for p in ps] for qid, ps in plan["phrases"].items()}
    os.makedirs(os.path.join(root, "draws"), exist_ok=True)
    for split in regions:
        if not _lock(split, root):
            continue
        out, draws_out = _paths(split, root)
        done = set()
        if os.path.exists(out):
            done = {r["id"] for r in be.load_results(out) if not r.get("error")}
        todo = [aid for aid in plan["regions"][split] if aid not in done]
        print(f"{split}: {len(todo)} to do, {len(done)} done", flush=True)
        t0, sampled = time.time(), 0
        with open(out, "a") as f:
            for i, aid in enumerate(todo, 1):
                shared = len({w[0] for w in plan["answers"][aid]["where"]}) > 1
                other = _done_rows(root).get(aid) if shared else None
                if other:
                    row = {**other[1], "copied_from": other[0]}
                else:
                    try:
                        row, draws = sample_answer(aid, plan["answers"][aid], plan, phrase_pats, region_pats, tok)
                    except RuntimeError as e:
                        if "403" in str(e):
                            raise
                        row, draws = {"key": plan["answers"][aid]["key"], "id": aid, "error": str(e)}, []
                    if draws:  # one complete gzip member per answer, written in one go
                        blob = gzip.compress("".join(json.dumps(d, ensure_ascii=False) + "\n"
                                                     for d in draws).encode())
                        with open(draws_out, "ab") as df:
                            df.write(blob)
                            df.flush()
                            os.fsync(df.fileno())
                    sampled += 1
                row["region"] = split
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    left = el / max(sampled, 1) * (len(todo) - i)
                    print(f"  {split} [{i}/{len(todo)}] {sampled} sampled, ~{left / 3600:.1f}h left in {split}",
                          flush=True)
        os.remove(os.path.join(root, f"{split}.lock"))


def status(root=PLAN_DIR):
    plan = json.load(open(os.path.join(root, "plan.json")))
    tot_d = tot = 0
    for s, ids in plan["regions"].items():
        path = _paths(s, root)[0]
        rows = be.load_results(path) if os.path.exists(path) else []
        ok = {r["id"] for r in rows if not r.get("error")}
        err = {r["id"] for r in rows if r.get("error")} - ok
        lock = os.path.join(root, f"{s}.lock")
        who = open(lock).read() if os.path.exists(lock) else ""
        print(f"{s}: {len(ok):>4}/{len(ids)}  errors {len(err)}  {('running: ' + who) if who else ''}")
        tot_d += len(ok)
        tot += len(ids)
    print(f"total {tot_d}/{tot}")
    return tot_d == tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "plan", "run", "status"])
    ap.add_argument("--regions", default="all", help="comma-separated region codes, or 'all'")
    ap.add_argument("--root", default=PLAN_DIR)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--scope", choices=["top", "all"], default="top")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "plan":
        make_plan(args.scope, args.n, args.seed, os.path.join(args.root, "plan.json"))
        return
    if args.mode == "status":
        raise SystemExit(0 if status(args.root) else 3)   # exit 0 = everything done
    if args.mode == "run":
        regions = be.BLEND_SPLITS if args.regions == "all" else args.regions.split(",")
        run(regions, args.root)
        return
    args.out = args.out or os.path.join(RES, "sampling_validation.jsonl")
    tok = _tok()
    done = set()
    if os.path.exists(args.out):
        done = {(r["name"], r["side"]) for r in be.load_results(args.out)}
    with open(args.out, "a") as f:
        for p in validation_pairs(args.pairs) + region_pairs(tokenizer=tok):
            sides = [("a", p["a"], p["b"])] + ([] if p.get("a_only") else [("b", p["b"], p["a"])])
            for side, a, b in sides:
                if (p["name"], side) in done:
                    continue
                t0 = time.time()
                e = estimate(a, b, tok, n=args.n)
                rec = {"name": p["name"], "side": side, "exact": p["exact"], "sides": p["sides"], **e}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{p['name'][:50]:50s} from {side}: N={e['N_a']:>7,} near {e['k_near']}/{e['n']} "
                      f"doc {e['k_doc']} est {e['est']:,.0f} [{e['lo']:,.0f}, {e['hi']:,.0f}] "
                      f"exact {p['exact']:,} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
