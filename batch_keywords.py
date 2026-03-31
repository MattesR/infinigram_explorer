"""
Batch keyword expansion for TREC topics using Anthropic Batch API.

Usage:
    # Submit batch
    python batch_keywords.py submit --topics topics_rag24_test.txt

    # Check status
    python batch_keywords.py status --batch-id msgbatch_XXXXX

    # Collect results
    python batch_keywords.py collect --batch-id msgbatch_XXXXX --output keyword_expansions.json
"""

import argparse
import json
import re
import time
import anthropic
import os

SYSTEM_PROMPT = """You are generating search terms for an exact-match keyword search engine where terms must appear verbatim in documents.  The goal is to also find highly relevant documents, that might have little or no verbatim overlap with the query.

Given a query, identify its key aspects and provide search terms for each.

KEY aspects (2-3): The main topics of the query. A key term should be specific to this topic — if someone searches for it, they are likely looking for something related to this query.

SUPPORTING aspects (3-6): Secondary topics that help narrow or contextualize results. These terms are broader and will be combined with key terms during search.

VERBS (1, optional): Key verbs relevant to the query's intent in base form (e.g. "cope", "develop", "prevent"). Only include if the query has a clear action focus.

For each aspect, provide three levels of search terms:
1. Lexical: Different ways to say the same thing — synonyms, abbreviations, alternative phrasings (e.g. "AI" and "artificial intelligence", "holistic" and "comprehensive")
2. Conceptual: Domain-specific technical terms, established frameworks, and jargon from the field
3. Referential: Named entities — key figures, landmark papers, specific theories, organizations, or standards associated with this topic (only if applicable)

Mix all three levels within each aspect's term list, ordered by specificity (most specific first).

Format: JSON object where keys are "KEY: <aspect name>", "SUP: <aspect name>", or "VERBS" and values are lists of 3-8 search terms.

Rules:
- Every term must be a noun phrase (1-4 words) that literally appears in real documents
- Key terms: specific to this topic, domain jargon, technical phrases
- Supporting terms: can be broader, will be combined with key terms
- Order terms within each aspect by specificity (most specific first)
- No sentences, no questions
- Return ONLY valid JSON, no explanation"""

from utils import get_token
os.environ['ANTHROPIC_API_KEY'] = get_token('ANTHROPIC_API_KEY')

def load_topics(path):
    topics = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                topics.append((parts[0], parts[1]))
    return topics


def submit_batch(topics_path, model="claude-sonnet-4-6"):
    client = anthropic.Anthropic()
    topics = load_topics(topics_path)

    requests = []
    for qid, query in topics:
        requests.append({
            "custom_id": qid,
            "params": {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": query}],
            }
        })

    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"\nTo check status:")
    print(f"  python batch_keywords.py status --batch-id {batch.id}")
    print(f"\nTo collect results:")
    print(f"  python batch_keywords.py collect --batch-id {batch.id}")
    return batch.id


def check_status(batch_id):
    client = anthropic.Anthropic()
    status = client.messages.batches.retrieve(batch_id)
    print(f"Batch: {batch_id}")
    print(f"Status: {status.processing_status}")
    print(f"Request counts: {status.request_counts}")
    if hasattr(status, 'ended_at') and status.ended_at:
        print(f"Ended at: {status.ended_at}")
    return status.processing_status


def wait_for_completion(batch_id, poll_interval=30):
    client = anthropic.Anthropic()
    print(f"Waiting for batch {batch_id}...")
    while True:
        status = client.messages.batches.retrieve(batch_id)
        print(f"  Status: {status.processing_status} | {status.request_counts}")
        if status.processing_status == "ended":
            print("Batch completed!")
            return True
        time.sleep(poll_interval)


def collect_results(batch_id, output_path):
    client = anthropic.Anthropic()

    # Check status first
    status = client.messages.batches.retrieve(batch_id)
    if status.processing_status != "ended":
        print(f"Batch not done yet: {status.processing_status}")
        print(f"Request counts: {status.request_counts}")
        return None

    print(f"Collecting results from batch {batch_id}...")
    results = {}
    errors = []

    for result in client.messages.batches.results(batch_id):
        qid = result.custom_id

        if result.result.type != "succeeded":
            errors.append((qid, result.result.type))
            continue

        text = result.result.message.content[0].text

        # Strip markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        try:
            parsed = json.loads(text)
            results[qid] = parsed
        except json.JSONDecodeError:
            # Store raw text if JSON parsing fails
            results[qid] = {"raw": text}
            errors.append((qid, "json_parse_error"))

    print(f"Collected {len(results)} results, {len(errors)} errors")

    if errors:
        print(f"Errors:")
        for qid, err in errors[:10]:
            print(f"  {qid}: {err}")

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    # Also save as JSONL for easier streaming
    jsonl_path = output_path.replace(".json", ".jsonl")
    with open(jsonl_path, "w") as f:
        for qid, data in results.items():
            f.write(json.dumps({"qid": qid, **data}) + "\n")
    print(f"Also saved as {jsonl_path}")

    return results


def load_keyword_expansions(path):
    """
    Load cached keyword expansions.

    Returns:
        Dict mapping qid -> {"core": [...], "expansion": [...]}
    """
    with open(path) as f:
        return json.load(f)


def get_all_keywords(expansions, qid):
    """
    Get all keywords (core + expansion) for a query.

    Returns:
        List of keyword strings.
    """
    data = expansions.get(qid, {})
    core = data.get("core", [])
    expansion = data.get("expansion", [])
    return core + expansion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch keyword expansion")
    subparsers = parser.add_subparsers(dest="command")

    sub = subparsers.add_parser("submit")
    sub.add_argument("--topics", required=True)
    sub.add_argument("--model", default="claude-sonnet-4-6")

    sub = subparsers.add_parser("status")
    sub.add_argument("--batch-id", required=True)

    sub = subparsers.add_parser("wait")
    sub.add_argument("--batch-id", required=True)
    sub.add_argument("--interval", type=int, default=30)

    sub = subparsers.add_parser("collect")
    sub.add_argument("--batch-id", required=True)
    sub.add_argument("--output", default="keyword_expansions.json")

    args = parser.parse_args()

    if args.command == "submit":
        submit_batch(args.topics, args.model)
    elif args.command == "status":
        check_status(args.batch_id)
    elif args.command == "wait":
        wait_for_completion(args.batch_id, args.interval)
    elif args.command == "collect":
        collect_results(args.batch_id, args.output)
    else:
        parser.print_help()