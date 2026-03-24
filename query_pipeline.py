"""
Full query construction pipeline.

Input: a natural language query string
Output: ranked CNF queries ready for engine.find_cnf()

Pipeline:
1. Encode query with SPLADE -> sparse embedding
2. Extract tokens above min_splade_score
3. Compute combined score: splade_score * log(1/tup)
4. Cluster by WordPiece stems
5. Build CNF queries (anchor + non-anchor pairs)

Usage:
    from query_pipeline import QueryPipeline

    pipeline = QueryPipeline(
        splade_model=splade,
        infini_tokenizer=tokenizer,
        probs=probs,  # numpy array from load_tup
    )

    queries = pipeline.build(
        "what makes up a community, including its definitions",
    )

    # queries is a list of dicts with 'cnf', 'description', 'score'
    # ready for engine.find_cnf(cnf=queries[0]['cnf'])
"""

import numpy as np
from query_construction import build_cnf_queries, run_queries


class QueryPipeline:
    def __init__(
        self,
        splade_model,
        infini_tokenizer,
        probs: np.ndarray,
    ):
        """
        Args:
            splade_model: SparseEncoder model for query encoding.
            infini_tokenizer: Infini-gram tokenizer (e.g. Llama).
            probs: Token unigram probability array (indexed by infini-gram token ID).
        """
        self.splade = splade_model
        self.infini_tokenizer = infini_tokenizer
        self.probs = probs
        self.splade_tokenizer = splade_model.tokenizer

    def encode_query(self, query: str) -> list[tuple[str, float]]:
        """
        Encode query with SPLADE and return all non-zero tokens with scores.

        Returns:
            List of (token_string, splade_score) sorted by score descending.
        """
        embedding = self.splade.encode_query([query])

        # Extract non-zero entries
        if hasattr(embedding, 'to_dense'):
            dense = embedding.to_dense().cpu().numpy()[0]
        else:
            dense = np.array(embedding[0])

        nonzero_indices = np.nonzero(dense)[0]
        vocab = self.splade_tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        tokens = [
            (id_to_token.get(int(idx), f"[UNK_{idx}]"), float(dense[idx]))
            for idx in nonzero_indices
        ]
        tokens.sort(key=lambda x: x[1], reverse=True)
        return tokens

    def word_tup(self, word: str) -> float:
        """
        Compute token unigram probability for a word as the product
        of its constituent token probabilities.
        """
        clean = word.lstrip("#")
        ids = self.infini_tokenizer.encode(clean, add_special_tokens=False)
        if not ids:
            return 1.0  # unknown word, treat as maximally common (will be filtered)
        return float(np.prod([self.probs[tid] for tid in ids if tid < len(self.probs)]))

    def score_tokens(
        self,
        splade_tokens: list[tuple[str, float]],
        min_splade_score: float = 0.3,
    ) -> list[tuple[str, float, float, float]]:
        """
        Filter by min SPLADE score and compute combined score.

        Returns:
            List of (token, splade_score, tup, combined_score) sorted by
            combined_score descending.
        """
        scored = []
        for token, splade_score in splade_tokens:
            if splade_score < min_splade_score:
                continue
            tup = self.word_tup(token)
            if tup <= 0:
                tup = 1e-15  # avoid log(0)
            combined = splade_score * np.log(1.0 / tup)
            scored.append((token, splade_score, tup, combined))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored

    def build(
        self,
        query: str,
        min_splade_score: float = 0.3,
        anchor_score: float = 0.9,
        min_cluster_score: float = 1.0,
        min_stem_len: int = 5,
        max_queries: int = 50,
        strategy: str = "anchor",
        verbose: bool = True,
        dry: bool = False,
    ) -> list[dict]:
        """
        Full pipeline: query string -> ranked CNF queries.

        Args:
            query: Natural language query string.
            min_splade_score: Minimum SPLADE score to keep a token.
            anchor_score: Minimum raw SPLADE score for a cluster to be an anchor.
            min_cluster_score: Minimum combined score for non-anchor clusters.
            min_stem_len: For WordPiece clustering.
            max_queries: Maximum number of CNF queries to return.
            strategy: "anchor" or "all_pairs".
            verbose: Print intermediate steps.
            dry: If True, just print the queries without returning CNF data.
                 Implies verbose=True.

        Returns:
            List of query dicts with 'cnf', 'description', 'score'.
            If dry=True, returns the list but skips nothing — just forces verbose.
        """
        if dry:
            verbose = True
        # Step 1: SPLADE encode
        if verbose:
            print(f"Query: {query}")
            print(f"\nStep 1: SPLADE encoding...")
        all_tokens = self.encode_query(query)
        if verbose:
            print(f"  {len(all_tokens)} non-zero tokens in SPLADE embedding")

        # Step 2: Filter and score
        if verbose:
            print(f"\nStep 2: Filtering (min_splade={min_splade_score}) and scoring...")
        scored = self.score_tokens(all_tokens, min_splade_score=min_splade_score)
        if verbose:
            print(f"  {len(scored)} tokens after filtering")
            print(f"\n  {'Token':<20s} {'SPLADE':>8s} {'TUP':>12s} {'Combined':>10s}")
            print(f"  {'-'*52}")
            for token, splade, tup, combined in scored:
                print(f"  {token:<20s} {splade:>8.3f} {tup:>12.2e} {combined:>10.2f}")

        if len(scored) < 2:
            print("  Not enough tokens to form CNF queries")
            return []

        # Step 3: Build CNF queries
        if verbose:
            print(f"\nStep 3: Building CNF queries (strategy={strategy})...")
        queries = build_cnf_queries(
            scored,
            self.infini_tokenizer,
            min_stem_len=min_stem_len,
            max_queries=max_queries,
            min_cluster_score=min_cluster_score,
            anchor_score=anchor_score,
            strategy=strategy,
        )

        return queries

    def build_and_run(
        self,
        query: str,
        engine,
        max_clause_freq: int = None,
        dry: bool = False,
        **build_kwargs,
    ) -> list[dict]:
        """
        Full pipeline including hitting the index.
        If dry=True, prints everything but skips the index queries.

        Returns:
            List of query dicts with 'cnf', 'description', 'score', 'cnt', etc.
        """
        queries = self.build(query, dry=dry, **build_kwargs)
        if queries and not dry:
            queries = run_queries(engine, queries, max_clause_freq=max_clause_freq)
        return queries