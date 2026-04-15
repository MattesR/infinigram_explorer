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

        # Reconstruct WordPiece fragments using the original query
        tokens = self.reconstruct_wordpieces(tokens, query)
        return tokens

    def reconstruct_wordpieces(
        self,
        tokens: list[tuple[str, float]],
        query: str,
    ) -> list[tuple[str, float]]:
        """
        Merge BERT WordPiece subtokens (##-prefixed) back into full words.

        Tokenizes the original query with the BERT tokenizer to get the
        correct token order, then glues ##-tokens onto the preceding token.
        Finally, maps scores from the SPLADE output onto the reconstructed words.

        E.g.: query "vicarious trauma" tokenizes to ['vi', '##car', '##ious', 'trauma']
              SPLADE has scores for 'vicar' and '##ious'
              -> reconstructs 'vicarious' with max(score_vicar, score_ious)
        """
        if not any(t.startswith("##") for t, _ in tokens):
            return tokens

        # Build score lookup from SPLADE output
        score_lookup = {t: s for t, s in tokens}

        # Tokenize the original query with BERT tokenizer to get correct order
        bert_tokens = self.splade_tokenizer.tokenize(query)

        # Reconstruct words by gluing ## tokens onto previous
        words = []  # list of (word_string, [constituent_tokens])
        for bt in bert_tokens:
            if bt.startswith("##") and words:
                word, constituents = words[-1]
                words[-1] = (word + bt[2:], constituents + [bt])
            else:
                words.append((bt, [bt]))

        # Map reconstructed words to SPLADE scores
        merged = []
        used_tokens = set()

        for word, constituents in words:
            # Collect scores from constituents that appear in SPLADE output
            constituent_scores = []
            for c in constituents:
                if c in score_lookup:
                    constituent_scores.append(score_lookup[c])
                    used_tokens.add(c)

            if constituent_scores:
                # Use max score among constituents
                merged.append((word, max(constituent_scores)))

        # Add any SPLADE tokens that weren't part of the query tokenization
        # (SPLADE can activate tokens not in the original query)
        for token, score in tokens:
            if token not in used_tokens and not token.startswith("##"):
                merged.append((token, score))

        # Deduplicate by word string, keeping highest score
        deduped = {}
        for word, score in merged:
            if word not in deduped or score > deduped[word]:
                deduped[word] = score
        merged = [(w, s) for w, s in deduped.items()]

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

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
        max_anchor_tup: float = 1e-4,
        min_cluster_score: float = 1.0,
        min_stem_len: int = 5,
        max_queries: int = 50,
        strategy: str = "anchor",
        use_chunking: bool = False,
        verbose: bool = True,
        dry: bool = False,
    ) -> list[dict]:
        """
        Full pipeline: query string -> ranked CNF queries.

        Args:
            query: Natural language query string.
            min_splade_score: Minimum SPLADE score to keep a token.
            anchor_score: Minimum raw SPLADE score for a cluster to be an anchor.
            max_anchor_tup: Maximum TUP for anchor/informative tokens.
                Tokens above this go into the common pool.
            min_cluster_score: Minimum combined score for non-anchor clusters.
            min_stem_len: For WordPiece clustering.
            max_queries: Maximum number of CNF queries to return.
            strategy: "anchor", "anchor_plus_common", or "all_pairs".
            use_chunking: If True, use spaCy to extract multi-word phrases
                from the original query and merge matching SPLADE tokens
                into phrase-level entries before clustering.
            verbose: Print intermediate steps.
            dry: If True, just print the queries without hitting the index.
                 Implies verbose=True.

        Returns:
            List of query dicts with 'cnf', 'description', 'score'.
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

        # Step 2b (optional): Extract syntactic links from query
        syntactic_links = None
        if use_chunking:
            from phrase_extraction import extract_syntactic_links
            if verbose:
                print(f"\nStep 2b: Extracting syntactic links with spaCy...")
            syntactic_links = extract_syntactic_links(query, verbose=verbose)

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
            max_anchor_tup=max_anchor_tup,
            strategy=strategy,
            syntactic_links=syntactic_links,
        )

        return queries

    def build_and_run(
        self,
        query: str,
        engine,
        max_clause_freq: int = 8000000,
        min_retrieved_docs: int = None,
        lower_query_bound: float = None,
        dry: bool = False,
        **build_kwargs,
    ) -> list[dict]:
        """
        Full pipeline including hitting the index.
        If dry=True, prints everything but skips the index queries.

        Args:
            query: Query text.
            engine: Infini-gram engine.
            max_clause_freq: Passed to engine.find_cnf.
            min_retrieved_docs: Stop executing queries after accumulating
                this many pointers.
            lower_query_bound: Skip queries with score below this.
            dry: Print only, don't execute.
            **build_kwargs: Passed to build().

        Returns:
            List of query dicts with 'cnf', 'description', 'score', 'cnt', etc.
        """
        queries = self.build(query, dry=dry, **build_kwargs)
        if queries and not dry:
            queries = run_queries(
                engine, queries,
                max_clause_freq=max_clause_freq,
                min_retrieved_docs=min_retrieved_docs,
                lower_query_bound=lower_query_bound,
            )
        return queries

    def build_and_run_adaptive(
        self,
        query: str,
        engine,
        min_splade_score: float = 0.3,
        anchor_score: float = 0.9,
        max_anchor_tup: float = 1e-4,
        min_cluster_score: float = 1.0,
        min_stem_len: int = 5,
        max_standalone: int = 10000,
        max_refined: int = 20000,
        max_queries: int = 50,
        max_clause_freq: int = 8000000,
        min_retrieved_docs: int = None,
        verbose: bool = True,
    ) -> tuple[list[dict], list[tuple]]:
        """
        Full adaptive pipeline: query string -> count-adaptive queries -> executed results.

        Handles encoding, scoring, adaptive query building, and execution
        in one call.

        Args:
            query: Natural language query string.
            engine: Infini-gram engine.
            min_splade_score: Min SPLADE score for token filtering.
            anchor_score: Min SPLADE score for anchor clusters.
            max_anchor_tup: Max TUP for anchor/informative tokens.
            min_cluster_score: Min combined score for non-anchor clusters.
            min_stem_len: For WordPiece clustering.
            max_standalone: Max count for direct find() queries.
            max_refined: Max count before adding more AND clauses.
            max_queries: Max total queries.
            max_clause_freq: For CNF sampling.
            min_retrieved_docs: Stop executing after enough pointers.
            verbose: Print intermediate steps.

        Returns:
            Tuple of (executed_queries, scored_tokens).
            scored_tokens is the list of (token, splade, tup, combined) for crude scoring.
        """
        from adaptive_queries import build_adaptive_queries, run_adaptive

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

        if len(scored) < 1:
            print("  Not enough tokens")
            return [], scored

        # Step 3: Build adaptive queries (peeks at counts)
        if verbose:
            print(f"\nStep 3: Building adaptive queries...")
        queries = build_adaptive_queries(
            scored,
            self.infini_tokenizer,
            engine,
            query_text=query,
            min_stem_len=min_stem_len,
            anchor_score=anchor_score,
            max_anchor_tup=max_anchor_tup,
            max_standalone=max_standalone,
            max_refined=max_refined,
            min_cluster_score=min_cluster_score,
            max_queries=max_queries,
            max_clause_freq=max_clause_freq,
            verbose=verbose,
        )

        if not queries:
            return [], scored

        # Step 4: Execute queries
        if verbose:
            print(f"\nStep 4: Executing queries...")
        executed = run_adaptive(
            engine, queries,
            max_clause_freq=max_clause_freq,
            min_retrieved_docs=min_retrieved_docs,
            verbose=verbose,
        )

        return executed, scored

    def build_and_run_llm_adaptive(
        self,
        query: str,
        qid: str,
        engine,
        expansions_path: str,
        min_splade_score: float = 0.3,
        max_standalone: int = 5000,
        max_standalone_sup: int = 1000,
        max_count: int = 500000,
        max_queries: int = 50,
        max_clause_freq: int = 8000000,
        min_retrieved_docs: int = None,
        use_core_only: bool = False,
        filter_mode: str = "stopword",
        verbose: bool = True,
    ) -> tuple[list[dict], list[tuple]]:
        """
        Full LLM-keyword-driven pipeline:
        query -> SPLADE encode (for scoring) -> LLM keywords -> adaptive queries -> execute.

        Args:
            query: Natural language query string.
            qid: Query ID (for looking up LLM expansions).
            engine: Infini-gram engine.
            expansions_path: Path to LLM keyword expansions JSONL.
            min_splade_score: Min SPLADE score for token filtering (scoring only).
            max_standalone: Max count for direct find() queries.
            max_refined: Max count for AND-refined queries.
            max_count: Skip terms above this entirely.
            max_queries: Max total queries.
            max_clause_freq: For CNF sampling.
            min_retrieved_docs: Stop executing after enough pointers.
            use_core_only: Only use core facets from LLM keywords.
            verbose: Print intermediate steps.

        Returns:
            Tuple of (executed_queries, scored_tokens, all_validated).
            scored_tokens: SPLADE tokens for fallback scoring.
            all_validated: LLM keyword dicts for LLM-based scoring.
        """
        from llm_adaptive_queries import build_llm_adaptive_queries, run_llm_adaptive

        # Step 1: SPLADE encode for fallback scoring
        if verbose:
            print(f"Query: [{qid}] {query}")
            print(f"\nStep 1: SPLADE encoding (for fallback scoring)...")
        all_tokens = self.encode_query(query)
        scored = self.score_tokens(all_tokens, min_splade_score=min_splade_score)
        if verbose:
            print(f"  {len(scored)} scored tokens")

        # Step 2: Build queries from LLM keywords
        if verbose:
            print(f"\nStep 2: Building queries from LLM keywords...")
        queries, all_validated = build_llm_adaptive_queries(
            qid=qid,
            expansions_path=expansions_path,
            tokenizer=self.infini_tokenizer,
            engine=engine,
            max_standalone=max_standalone,
            max_standalone_sup=max_standalone_sup,
            max_count=max_count,
            max_queries=max_queries,
            max_clause_freq=max_clause_freq,
            use_core_only=use_core_only,
            filter_mode=filter_mode,
            verbose=verbose,
        )

        if not queries:
            if verbose:
                print("  No queries generated!")
            return [], scored, all_validated

        # Step 3: Execute queries
        if verbose:
            print(f"\nStep 3: Executing {len(queries)} queries...")
        executed = run_llm_adaptive(
            engine, queries,
            max_clause_freq=max_clause_freq,
            min_retrieved_docs=min_retrieved_docs,
            verbose=verbose,
        )

        return executed, scored, all_validated