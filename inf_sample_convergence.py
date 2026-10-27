import math

def get_dist(engine, prompt_ids, *, use_infgram=True, max_support=None):
    fn = engine.infgram_ntd if use_infgram else engine.ntd
    kwargs = {} if max_support is None else {"max_support": max_support}
    out = fn(prompt_ids=prompt_ids, **kwargs)
    dist = out.get("result_by_token_id", {}) or {}
    # normalize defensively
    probs = {int(t): float(info.get("prob", 0.0)) for t, info in dist.items()}
    s = sum(probs.values())
    if s > 0:
        probs = {t: p/s for t, p in probs.items()}
    return probs

def js_divergence(p, q, eps=1e-12):
    # JS over union of keys
    keys = set(p) | set(q)
    def kl(a, b):
        out = 0.0
        for t in keys:
            pa = a.get(t, 0.0)
            pb = b.get(t, 0.0)
            if pa > 0:
                out += pa * math.log(pa / max(pb, eps))
        return out
    m = {t: 0.5*(p.get(t,0.0)+q.get(t,0.0)) for t in keys}
    return 0.5*kl(p, m) + 0.5*kl(q, m)

def topk_overlap(p, q, k=50):
    tp = [t for t,_ in sorted(p.items(), key=lambda x: x[1], reverse=True)[:k]]
    tq = [t for t,_ in sorted(q.items(), key=lambda x: x[1], reverse=True)[:k]]
    return len(set(tp) & set(tq)) / k

def convergence_sweep(engine, tokenizer, query, supports, use_infgram=True):
    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    prev = None
    for N in supports:
        cur = get_dist(engine, prompt_ids, use_infgram=use_infgram, max_support=N)
        if prev is None:
            print(f"N={N}: |V|={len(cur)}")
        else:
            js = js_divergence(prev, cur)
            ov = topk_overlap(prev, cur, k=50)
            print(f"N={N}: JS(prev,cur)={js:.6f}  top50_overlap={ov:.2f}  |V|={len(cur)}")
        prev = cur
