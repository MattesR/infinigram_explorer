"""
Two-phase peek and grab v2.

Phase 1: Standalone peek — grab cheap pieces (< threshold)
Phase 2: Combo peek — build 2-way ANDs from remaining KEY pieces, peek counts,
         sort by tightness, report totals.

Usage:
    from peek_grab_v2 import peek_and_grab_v2

    result = peek_and_grab_v2(
        qid="2024-32912",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import time
from itertools import combinations
from progressive_queries import build_pieces


def peek_and_grab_v2(
    qid,
    expansions_path,
    tokenizer,
    engine,
    max_standalone_key=1500,
    max_standalone_assoc=750,
    max_clause_freq=80000000,
    prox_peek=10,
    prox_cross=50,
    max_budget=20000,
    max_tighten_attempts=20,
    verbose=True,
    _pieces=None,
):
    """
    Two-phase peek and grab.

    Phase 1: Standalone — peek all pieces, grab cheap ones.
    Phase 2: Combo — build 2-way ANDs from remaining KEY pieces (no assoc),
             peek counts, sort by count.
    Phase 3: If over budget, tighten broadest queries with associated terms.

    Returns dict with grabbed, combo queries, remaining assoc, and summaries.
    """
    t0 = time.perf_counter()

    if _pieces is not None:
        pieces = _pieces
    else:
        pieces = build_pieces(qid, expansions_path, tokenizer, engine, verbose=False)

    # ================================================================
    # Phase 1: Standalone peek and grab
    # ================================================================
    all_key_pieces = {}  # description -> {piece, count, aspect, grabbed}
    all_assoc_pieces = {}  # description -> {piece, count, grabbed}
    grabbed = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 1: Standalone peek ({qid})")
        print(f"{'='*70}")

    # Peek key pieces
    for aspect_name, piece_list in pieces["key_pieces"].items():
        for p in piece_list:
            prox = prox_peek if len(p["cnf"]) > 1 else None
            try:
                kwargs = {"max_clause_freq": max_clause_freq}
                if prox:
                    kwargs["max_diff_tokens"] = prox
                cnt = engine.count_cnf(p["cnf"], **kwargs).get("count", 0)
            except Exception:
                cnt = 0

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
                    "max_diff_tokens": prox,
                    "description": desc,
                    "estimated_count": cnt,
                    "level": "S0_key",
                })

            if verbose:
                status = "GRAB" if info["grabbed"] else ("skip" if cnt == 0 else "remain")
                print(f"  [{aspect_name}] {desc}: {cnt:>10,d}  {status}")

    # Peek assoc pieces
    for p in pieces["associated"]:
        prox = prox_peek if len(p["cnf"]) > 1 else None
        try:
            kwargs = {"max_clause_freq": max_clause_freq}
            if prox:
                kwargs["max_diff_tokens"] = prox
            cnt = engine.count_cnf(p["cnf"], **kwargs).get("count", 0)
        except Exception:
            cnt = 0

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
                "max_diff_tokens": prox,
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

    print(f"  [{qid}] Phase 1: {len(grabbed)} standalone grabbed (~{grabbed_total:,d} docs), "
          f"{len(remaining_key)} key + {len(remaining_assoc)} assoc remaining")

    # ================================================================
    # Phase 2: Combo peek (2-way ANDs from remaining KEY pieces only)
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 2: Combo peek — 2-way ANDs from remaining key pieces")
        print(f"{'='*70}")

    # Group remaining key pieces by aspect
    remaining_by_aspect = {}
    for desc, info in remaining_key.items():
        aspect = info["aspect"]
        remaining_by_aspect.setdefault(aspect, []).append(info)

    aspect_names = list(remaining_by_aspect.keys())
    combo_queries = []
    n_combo_checked = 0

    if len(aspect_names) >= 2:
        for a, b in combinations(aspect_names, 2):
            for p_a in remaining_by_aspect[a]:
                for p_b in remaining_by_aspect[b]:
                    cnf = p_a["piece"]["cnf"] + p_b["piece"]["cnf"]
                    desc = f"({p_a['piece']['description']}) AND ({p_b['piece']['description']})"

                    try:
                        cnt = engine.count_cnf(
                            cnf, max_clause_freq=max_clause_freq,
                            max_diff_tokens=prox_cross,
                        ).get("count", 0)
                    except Exception:
                        cnt = 0

                    n_combo_checked += 1

                    if cnt == 0:
                        continue

                    combo_queries.append({
                        "type": "cnf",
                        "cnf": cnf,
                        "max_diff_tokens": prox_cross,
                        "description": desc,
                        "estimated_count": cnt,
                        "level": "S_combo",
                        "aspects": (a, b),
                    })

                    if verbose:
                        print(f"  {desc}: {cnt:>10,d}")

    # Sort by count (tightest first)
    combo_queries.sort(key=lambda q: q["estimated_count"])

    combo_total = sum(q["estimated_count"] for q in combo_queries)
    remaining_assoc_total = sum(i["count"] for i in remaining_assoc.values())

    print(f"  [{qid}] Phase 2: {n_combo_checked} combos checked, "
          f"{len(combo_queries)} with hits (~{combo_total:,d} docs). "
          f"Total so far: ~{grabbed_total + combo_total:,d} / {max_budget:,d} budget")

    # Sort by count (tightest first)
    combo_queries.sort(key=lambda q: q["estimated_count"])

    combo_total = sum(q["estimated_count"] for q in combo_queries)
    remaining_assoc_total = sum(i["count"] for i in remaining_assoc.values())

    # ================================================================
    # Phase 3: Tighten broad queries if over budget
    # ================================================================
    budget = max_budget
    all_fire_queries = list(grabbed) + list(combo_queries)
    current_total = sum(q["estimated_count"] for q in all_fire_queries)
    tightened = []

    if current_total > budget and remaining_assoc and all_fire_queries:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Phase 3: Over budget ({current_total:,d} > {budget:,d}), tightening broadest queries")
            print(f"{'='*70}")

        # Sort ALL queries by count descending — tighten broadest first
        all_fire_queries.sort(key=lambda q: q["estimated_count"], reverse=True)

        i = 0
        attempts = 0
        while current_total > budget and i < len(all_fire_queries) and attempts < max_tighten_attempts:
            query_q = all_fire_queries[i]
            original_count = query_q["estimated_count"]

            # Skip if already tight enough
            if original_count <= 100:
                i += 1
                continue

            attempts += 1

            if verbose:
                print(f"\n  Tightening: {query_q['description']} ({original_count:,d}) [{query_q['level']}]")

            # AND this query with each remaining assoc term
            replacement_queries = []
            for desc, assoc_info in remaining_assoc.items():
                cnf = query_q["cnf"] + assoc_info["piece"]["cnf"]
                new_desc = f"{query_q['description']} AND ({desc})"

                try:
                    cnt = engine.count_cnf(
                        cnf, max_clause_freq=max_clause_freq,
                        max_diff_tokens=prox_cross,
                    ).get("count", 0)
                except Exception:
                    cnt = 0

                if cnt == 0:
                    continue

                replacement_queries.append({
                    "type": "cnf",
                    "cnf": cnf,
                    "max_diff_tokens": prox_cross,
                    "description": new_desc,
                    "estimated_count": cnt,
                    "level": "S_tightened",
                    "original": query_q["description"],
                })

                if verbose:
                    print(f"    {new_desc}: {cnt:>10,d}")

            replacement_total = sum(q["estimated_count"] for q in replacement_queries)

            # Only replace if tightening actually reduces the count
            if replacement_total >= original_count or not replacement_queries:
                if verbose:
                    print(f"    Tightening didn't help ({replacement_total:,d} >= {original_count:,d}), keeping original")
                i += 1
                continue

            # Replace original with tightened versions
            all_fire_queries.pop(i)
            all_fire_queries.extend(replacement_queries)
            tightened.append({
                "original": query_q["description"],
                "original_level": query_q["level"],
                "original_count": original_count,
                "replacements": len(replacement_queries),
                "replacement_total": replacement_total,
            })

            current_total = current_total - original_count + replacement_total

            if verbose:
                print(f"    Replaced {original_count:,d} with {replacement_total:,d} "
                      f"({len(replacement_queries)} queries). New total: {current_total:,d}")

            # Don't increment i — next broadest is now at position i

        print(f"  [{qid}] Phase 3: tightened {len(tightened)} broad queries. "
              f"Final total: ~{current_total:,d} / {budget:,d} budget")
    else:
        if current_total <= budget:
            print(f"  [{qid}] Phase 3: skipped (under budget: ~{current_total:,d} / {budget:,d})")
        elif not remaining_assoc:
            print(f"  [{qid}] Phase 3: skipped (no assoc terms for tightening, ~{current_total:,d} docs)")

    # Split back into grabbed and combos for clarity
    grabbed = [q for q in all_fire_queries if q["level"].startswith("S0")]
    combo_queries = [q for q in all_fire_queries if not q["level"].startswith("S0")]

    # Sort combos by count (tightest first)
    combo_queries.sort(key=lambda q: q["estimated_count"])
    grabbed_total = sum(q["estimated_count"] for q in grabbed)
    combo_total = sum(q["estimated_count"] for q in combo_queries)
    current_total = grabbed_total + combo_total

    # ================================================================
    # Phase 4: Fill remaining budget with cheapest remaining pieces
    # ================================================================
    filled = []
    if current_total < budget:
        # Collect all remaining pieces (key + assoc) not yet in queries
        fired_descs = {q["description"] for q in grabbed + combo_queries}
        remaining_all = []

        for desc, info in remaining_key.items():
            if desc not in fired_descs:
                remaining_all.append(info)
        for desc, info in remaining_assoc.items():
            if desc not in fired_descs:
                remaining_all.append(info)

        # Sort by count ascending (cheapest first)
        remaining_all.sort(key=lambda x: x["count"])

        for info in remaining_all:
            if current_total + info["count"] > budget:
                continue
            p = info["piece"]
            prox = prox_peek if len(p["cnf"]) > 1 else None
            q = {
                "type": "cnf",
                "cnf": p["cnf"],
                "max_diff_tokens": prox,
                "description": p["description"],
                "estimated_count": info["count"],
                "level": "S_fill",
            }
            grabbed.append(q)
            all_fire_queries.append(q)
            current_total += info["count"]
            filled.append({"description": p["description"], "count": info["count"]})

            if verbose:
                print(f"  Fill: {p['description']}: {info['count']:>10,d}  (total: {current_total:,d})")

        grabbed_total = sum(q["estimated_count"] for q in grabbed)

        print(f"  [{qid}] Phase 4: filled {len(filled)} standalone pieces (~{sum(f['count'] for f in filled):,d} docs). "
              f"Final total: ~{current_total:,d} / {budget:,d} budget")
    else:
        print(f"  [{qid}] Phase 4: skipped (at budget: ~{current_total:,d} / {budget:,d})")

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n{'='*70}")
        print(f"Summary ({elapsed:.1f}s)")
        print(f"{'='*70}")
        print(f"  Phase 1 grabbed:       {len(grabbed):>5d} queries, ~{grabbed_total:>12,d} docs")
        print(f"  Phase 2 combos:        {len(combo_queries):>5d} queries, ~{combo_total:>12,d} docs")
        if tightened:
            n_tight = sum(t["replacements"] for t in tightened)
            print(f"    (incl. {n_tight} tightened from {len(tightened)} broad combos)")
        print(f"  Remaining assoc:       {len(remaining_assoc):>5d} pieces,  ~{remaining_assoc_total:>12,d} docs")
        print(f"  Total (grab+combo):    ~{grabbed_total + combo_total:>12,d} docs")
        print(f"  Budget:                 {budget:>12,d}")
        print(f"  Total (all):           ~{grabbed_total + combo_total + remaining_assoc_total:>12,d} docs")
        print(f"{'='*70}")

    return {
        "qid": qid,
        "grabbed": grabbed,
        "combo_queries": combo_queries,
        "tightened": tightened,
        "filled": filled,
        "remaining_key": remaining_key,
        "remaining_assoc": remaining_assoc,
        "all_key_pieces": all_key_pieces,
        "all_assoc_pieces": all_assoc_pieces,
        "grabbed_total": grabbed_total,
        "combo_total": combo_total,
        "remaining_assoc_total": remaining_assoc_total,
        "max_budget": max_budget,
        "time": elapsed,
    }