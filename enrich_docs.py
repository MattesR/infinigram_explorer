"""
Enrich JSONL files with document text from a Pyserini Lucene index.

Usage:
    from pyserini.search.lucene import LuceneSearcher
    from enrich_docs import enrich_jsonl

    searcher = LuceneSearcher('./lucene_index_msmarco_segmented')
    enrich_jsonl("inspection/lcr_opus/2024-32912_missed.jsonl", searcher)
"""

import json


def enrich_jsonl(path: str, searcher, output_path: str = None, text_field: str = "segment"):
    """
    Read a JSONL file, look up each doc_id in the Pyserini index,
    add the document text, and write back.

    Args:
        path: Path to JSONL file with {"doc_id": "...", ...} lines.
        searcher: LuceneSearcher instance.
        output_path: Output path. Defaults to overwriting the input file.
        text_field: Field name to store the text under.
    """
    if output_path is None:
        output_path = path

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    found = 0
    for record in records:
        did = record.get("doc_id", "")
        if not did or text_field in record:
            continue
        doc = searcher.doc(did)
        if doc:
            raw = json.loads(doc.raw())
            text = raw.get("segment", raw.get("body", raw.get("text", "")))
            record[text_field] = text
            found += 1

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Enriched {found}/{len(records)} docs in {output_path}")


def enrich_all_in_dir(directory: str, searcher, pattern: str = "*_missed.jsonl"):
    """
    Enrich all matching JSONL files in a directory.

    Args:
        directory: Path to inspection directory.
        searcher: LuceneSearcher instance.
        pattern: Glob pattern for files to enrich.
    """
    from pathlib import Path
    files = sorted(Path(directory).glob(pattern))
    print(f"Found {len(files)} files matching {pattern} in {directory}")
    for f in files:
        enrich_jsonl(str(f), searcher)