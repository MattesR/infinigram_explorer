"""
Lightweight phrase extraction using spaCy to recombine SPLADE tokens
into multi-word expressions for stronger index queries.

SPLADE atomizes queries into individual tokens, losing multi-word
expressions like "make up", "civilized community", "grassroots organizations".
This module extracts those phrases from the original query and cross-references
them with the SPLADE token list to create combined phrase queries.

Usage:
    from phrase_extraction import extract_phrases, merge_phrases_into_tokens

    phrases = extract_phrases(query_text)
    merged_tokens = merge_phrases_into_tokens(top_tokens, phrases, tokenizer)
"""

import spacy

# Load once, reuse
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_phrases(query: str, verbose: bool = False) -> list[dict]:
    """
    Extract multi-word expressions from a query using spaCy.

    Extracts:
    - Noun chunks (e.g. "civilized community", "grassroots organizations")
    - Phrasal verbs (e.g. "makes up")
    - Compound nouns (e.g. "nation building")
    - Adjective + noun pairs (e.g. "civilized community")

    Args:
        query: Original query text.
        verbose: Print extracted phrases.

    Returns:
        List of phrase dicts, each with:
            - 'text': the phrase as a string
            - 'tokens': list of lowercase token strings
            - 'type': phrase type (noun_chunk, phrasal_verb, compound, amod)
    """
    nlp = _get_nlp()
    doc = nlp(query)
    phrases = []
    seen = set()

    # Noun chunks (filter out single-word and determiner-only chunks)
    for chunk in doc.noun_chunks:
        # Remove leading determiners/pronouns
        tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON")]
        if len(tokens) >= 2:
            text = " ".join(t.text.lower() for t in tokens)
            if text not in seen:
                seen.add(text)
                phrases.append({
                    "text": text,
                    "tokens": [t.text.lower() for t in tokens],
                    "type": "noun_chunk",
                })

    # Phrasal verbs (verb + particle)
    for token in doc:
        if token.dep_ == "prt":
            text = f"{token.head.lemma_.lower()} {token.text.lower()}"
            if text not in seen:
                seen.add(text)
                phrases.append({
                    "text": text,
                    "tokens": [token.head.lemma_.lower(), token.text.lower()],
                    "type": "phrasal_verb",
                })

    # Compound nouns
    for token in doc:
        if token.dep_ == "compound":
            text = f"{token.text.lower()} {token.head.text.lower()}"
            if text not in seen:
                seen.add(text)
                phrases.append({
                    "text": text,
                    "tokens": [token.text.lower(), token.head.text.lower()],
                    "type": "compound",
                })

    # Adjective modifiers (amod)
    for token in doc:
        if token.dep_ == "amod":
            text = f"{token.text.lower()} {token.head.text.lower()}"
            if text not in seen:
                seen.add(text)
                phrases.append({
                    "text": text,
                    "tokens": [token.text.lower(), token.head.text.lower()],
                    "type": "amod",
                })

    if verbose:
        print(f"Extracted {len(phrases)} phrases from query:")
        for p in phrases:
            print(f"  [{p['type']:<15s}] {p['text']}")

    return phrases


def merge_phrases_into_tokens(
    top_tokens: list[tuple],
    phrases: list[dict],
    tokenizer,
    verbose: bool = False,
) -> list[tuple]:
    """
    Cross-reference extracted phrases with SPLADE tokens. Where multiple
    SPLADE tokens form a known phrase, merge them into a single entry
    with combined score and the full phrase as the token string.

    Merged tokens get their input_ids from tokenizing the full phrase,
    which means engine.find() will search for the contiguous sequence.

    Args:
        top_tokens: List of (token, splade_score, tup, combined_score).
        phrases: From extract_phrases().
        tokenizer: Infini-gram tokenizer for encoding phrases.
        verbose: Print merge decisions.

    Returns:
        Updated top_tokens list where merged tokens replace their
        constituents. Unmerged tokens are kept as-is.
    """
    # Build lookup: lowercase token -> index in top_tokens
    token_lookup = {}
    for i, (token, splade, tup, combined) in enumerate(top_tokens):
        clean = token.lstrip("#").lower()
        token_lookup[clean] = i

    # Build reconstructed words from WordPiece sequences.
    # E.g. ['vicar', '##ious'] -> 'vicarious' mapping to indices [0, 4]
    # Walk through tokens in order: a non-## token starts a new word,
    # subsequent ##-prefixed tokens continue it.
    wordpiece_words = []  # list of (reconstructed_word, [indices])
    current_word = ""
    current_indices = []
    for i, (token, splade, tup, combined) in enumerate(top_tokens):
        if token.startswith("##"):
            # Continuation of previous word
            current_word += token[2:].lower()
            current_indices.append(i)
        else:
            # Save previous word if it was multi-token
            if len(current_indices) > 1:
                wordpiece_words.append((current_word, list(current_indices)))
            # Start new word
            current_word = token.lower()
            current_indices = [i]
    # Don't forget the last word
    if len(current_indices) > 1:
        wordpiece_words.append((current_word, list(current_indices)))

    # Also build all pairwise concatenations of adjacent subtokens
    # since SPLADE doesn't guarantee order matches the original word
    for i, (tok_a, _, _, _) in enumerate(top_tokens):
        for j, (tok_b, _, _, _) in enumerate(top_tokens):
            if i == j:
                continue
            if tok_b.startswith("##"):
                reconstructed = tok_a.lstrip("#").lower() + tok_b[2:].lower()
                wordpiece_words.append((reconstructed, [i, j]))

    # Deduplicate: keep longest index list per word
    word_to_indices = {}
    for word, indices in wordpiece_words:
        if word not in word_to_indices or len(indices) > len(word_to_indices[word]):
            word_to_indices[word] = indices

    # Also try lemma matching for verbs (e.g. "makes" -> "make")
    nlp = _get_nlp()

    merged_indices = set()
    merged_entries = []

    for phrase in phrases:
        phrase_tokens = phrase["tokens"]

        # Check if all tokens in the phrase are present in SPLADE tokens
        # Try three strategies: direct match, lemma match, WordPiece reconstruction
        indices = []
        for pt in phrase_tokens:
            if pt in token_lookup:
                indices.append(token_lookup[pt])
            elif pt in word_to_indices:
                # Match via reconstructed WordPiece word
                indices.extend(word_to_indices[pt])
            else:
                # Try lemma
                lemma = nlp(pt)[0].lemma_.lower()
                if lemma in token_lookup:
                    indices.append(token_lookup[lemma])
                elif lemma in word_to_indices:
                    indices.extend(word_to_indices[lemma])
                else:
                    break
        else:
            # All tokens found — merge them
            if len(indices) < 2:
                continue

            # Aggregate scores
            constituents = [top_tokens[i] for i in indices]
            max_splade = max(s for _, s, _, _ in constituents)
            min_tup_val = min(t for _, _, t, _ in constituents)
            sum_combined = sum(c for _, _, _, c in constituents)

            # Encode the full phrase for index queries
            phrase_text = phrase["text"]
            phrase_ids = tokenizer.encode(phrase_text, add_special_tokens=False)

            if not phrase_ids:
                continue

            merged_entry = (
                phrase_text,       # token string is now the full phrase
                max_splade,        # use max splade of constituents
                min_tup_val,       # use min tup (most discriminative)
                sum_combined,      # sum combined scores
            )
            merged_entries.append(merged_entry)
            merged_indices.update(indices)

            if verbose:
                parts = " + ".join(f"{t[0]}({t[3]:.1f})" for t in constituents)
                print(f"  Merged: {parts} -> '{phrase_text}' "
                      f"(splade={max_splade:.2f}, combined={sum_combined:.2f})")

    # Build new token list: merged entries + unmerged originals
    new_tokens = list(merged_entries)
    for i, entry in enumerate(top_tokens):
        if i not in merged_indices:
            new_tokens.append(entry)

    # Sort by combined score descending
    new_tokens.sort(key=lambda x: x[3], reverse=True)

    if verbose:
        n_merged = len(merged_entries)
        n_consumed = len(merged_indices)
        print(f"\n  {n_merged} phrases merged, consuming {n_consumed} tokens")
        print(f"  {len(top_tokens)} tokens -> {len(new_tokens)} tokens")

    return new_tokens