"""
Extract syntactic relationships from the query using spaCy and combine
with SPLADE token clusters to form structural query units.

Two distinct concerns:
1. SPLADE expansion: civilized -> civilized|civilization (synonym clusters)
2. spaCy syntax: "civilized" modifies "community" (AND constraints)

Combined: (civilized OR civilization) AND (community OR communities)
This forms one "unit" that can be AND'd with other clusters in CNF queries.

Usage:
    from phrase_extraction import extract_syntactic_links, build_query_units

    links = extract_syntactic_links(query_text)
    units = build_query_units(links, clusters)
"""

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_syntactic_links(query: str, verbose: bool = False) -> list[dict]:
    """
    Extract syntactic relationships between words in the query.

    Returns links between words that should be AND'd together:
    - amod: adjective + noun ("civilized community")
    - compound: compound noun ("grassroots organization")
    - prt: phrasal verb ("make up")
    - nsubj/dobj: subject/object relations for tight coupling

    Args:
        query: Original query text.
        verbose: Print extracted links.

    Returns:
        List of link dicts, each with:
            - 'words': list of lowercase word strings that belong together
            - 'type': relationship type
            - 'text': the phrase as it appears in the query
    """
    nlp = _get_nlp()
    doc = nlp(query)
    links = []
    seen = set()

    # Adjective modifiers: "civilized community"
    for token in doc:
        if token.dep_ == "amod":
            key = (token.lemma_.lower(), token.head.lemma_.lower())
            if key not in seen:
                seen.add(key)
                links.append({
                    "words": [token.lemma_.lower(), token.head.lemma_.lower()],
                    "type": "amod",
                    "text": f"{token.text} {token.head.text}",
                })

    # Compound nouns: "grassroots organization", "nation building"
    for token in doc:
        if token.dep_ == "compound":
            key = (token.lemma_.lower(), token.head.lemma_.lower())
            if key not in seen:
                seen.add(key)
                links.append({
                    "words": [token.lemma_.lower(), token.head.lemma_.lower()],
                    "type": "compound",
                    "text": f"{token.text} {token.head.text}",
                })

    # Phrasal verbs: "make up", "cope with"
    for token in doc:
        if token.dep_ == "prt":
            key = (token.head.lemma_.lower(), token.lemma_.lower())
            if key not in seen:
                seen.add(key)
                links.append({
                    "words": [token.head.lemma_.lower(), token.lemma_.lower()],
                    "type": "prt",
                    "text": f"{token.head.text} {token.text}",
                })

    if verbose:
        print(f"Extracted {len(links)} syntactic links:")
        for link in links:
            print(f"  [{link['type']:<10s}] {link['text']:<30s} -> {link['words']}")

    return links


def build_query_units(
    links: list[dict],
    clusters: list[dict],
    verbose: bool = False,
) -> list[dict]:
    """
    Combine syntactic links with SPLADE clusters to form query units.

    A query unit is one or more clusters that are AND'd together because
    spaCy says they modify each other. Clusters not involved in any link
    remain standalone units.

    Args:
        links: From extract_syntactic_links().
        clusters: From wordpiece_cluster.cluster_tokens(), each with
            'stem', 'tokens' list of (word, score), 'combined_score'.

    Returns:
        List of unit dicts, each with:
            - 'clusters': list of cluster dicts that form this unit
            - 'type': 'linked' or 'standalone'
            - 'description': human-readable description
            - 'score': unit score
    """
    # Build lookup: word -> cluster index
    word_to_cluster = {}
    for i, c in enumerate(clusters):
        for token, score in c["tokens"]:
            word_to_cluster[token.lower()] = i
        # Also map the stem
        word_to_cluster[c["stem"].lower()] = i

    # Try to also match via lemma
    nlp = _get_nlp()

    # Find which clusters are linked together
    linked_groups = []  # list of sets of cluster indices
    used_clusters = set()

    for link in links:
        cluster_indices = set()
        for word in link["words"]:
            # Direct match
            if word in word_to_cluster:
                cluster_indices.add(word_to_cluster[word])
            else:
                # Try lemma
                lemma = nlp(word)[0].lemma_.lower()
                if lemma in word_to_cluster:
                    cluster_indices.add(word_to_cluster[lemma])

        if len(cluster_indices) >= 2:
            # Check if this overlaps with an existing group
            merged = False
            for group in linked_groups:
                if group & cluster_indices:
                    group.update(cluster_indices)
                    merged = True
                    break
            if not merged:
                linked_groups.append(cluster_indices)
            used_clusters.update(cluster_indices)

            if verbose:
                cluster_names = [clusters[i]["stem"] for i in cluster_indices]
                print(f"  Link [{link['type']}] '{link['text']}' "
                      f"-> clusters: {cluster_names}")

    # Build units
    units = []

    # Linked units (multi-cluster)
    for group in linked_groups:
        group_clusters = [clusters[i] for i in sorted(group)]
        parts = []
        for c in group_clusters:
            tokens_str = " OR ".join(t for t, s in c["tokens"])
            parts.append(f"({tokens_str})")
        description = " AND ".join(parts)
        score = sum(c.get("combined_score", 0) for c in group_clusters)

        units.append({
            "clusters": group_clusters,
            "type": "linked",
            "description": description,
            "score": score,
        })

    # Standalone units (single cluster)
    for i, c in enumerate(clusters):
        if i not in used_clusters:
            tokens_str = " OR ".join(t for t, s in c["tokens"])
            units.append({
                "clusters": [c],
                "type": "standalone",
                "description": f"({tokens_str})",
                "score": c.get("combined_score", 0),
            })

    units.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print(f"\nQuery units ({len(units)}):")
        for u in units:
            marker = "* " if u["type"] == "linked" else "  "
            print(f"  {marker}[{u['score']:6.2f}] {u['description']}")

    return units