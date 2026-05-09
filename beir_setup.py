"""
Prepare a BEIR dataset for the infini-gram retrieval pipeline.

Steps:
1. Download BEIR dataset
2. Convert corpus to infini-gram format (with docid in metadata)
3. Build infini-gram index
4. Convert queries and qrels to pipeline format

Usage:
    # Full setup
    python beir_setup.py --dataset scifact --out_dir ./beir/scifact/

    # Or step by step:
    from beir_setup import download_beir, convert_corpus, convert_queries_qrels
    from beir_setup import build_index, create_config

    download_beir("scifact", "./beir/scifact/raw/")
    convert_corpus("./beir/scifact/raw/", "./beir/scifact/corpus/")
    build_index("./beir/scifact/corpus/", "./beir/scifact/index/", cpus=16, mem=64)
    convert_queries_qrels("./beir/scifact/raw/", "./beir/scifact/")
    create_config("./beir/scifact/")
"""

import os
import json
import csv
import argparse
from pathlib import Path


def download_beir(dataset_name, out_dir):
    """
    Download a BEIR dataset using the beir library.

    Args:
        dataset_name: e.g. 'scifact', 'nfcorpus', 'fiqa', 'trec-covid'
        out_dir: Directory to store raw data.
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, out_dir)
        print(f"Downloaded {dataset_name} to {data_path}")
        return data_path

    except ImportError:
        # Fallback: download manually
        import urllib.request
        import zipfile

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        zip_path = os.path.join(out_dir, f"{dataset_name}.zip")

        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)

        print(f"Extracting to {out_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
        os.remove(zip_path)

        data_path = os.path.join(out_dir, dataset_name)
        print(f"Downloaded {dataset_name} to {data_path}")
        return data_path


def convert_corpus(raw_dir, corpus_dir):
    """
    Convert BEIR corpus to infini-gram format.

    BEIR: {"_id": "doc1", "title": "...", "text": "..."}
    Infini-gram: {"text": "title. text", "doc_id": "doc1"}

    Also creates a doc_id mapping file for resolving results back.
    """
    os.makedirs(corpus_dir, exist_ok=True)

    # Find corpus file
    corpus_file = None
    for name in ["corpus.jsonl", "corpus.jsonl.gz"]:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            corpus_file = path
            break

    if corpus_file is None:
        raise FileNotFoundError(f"No corpus file found in {raw_dir}")

    # Open potentially gzipped file
    if corpus_file.endswith(".gz"):
        import gzip
        opener = gzip.open
    else:
        opener = open

    # Convert
    out_path = os.path.join(corpus_dir, "corpus.jsonl")

    n_docs = 0
    with opener(corpus_file, 'rt', encoding='utf-8') as fin, \
         open(out_path, 'w') as fout:

        for line in fin:
            doc = json.loads(line.strip())
            doc_id = doc.get("_id", "")
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()

            # Combine title and text
            if title:
                full_text = f"{title}. {text}"
            else:
                full_text = text

            # Infini-gram format: "text" field required, other fields stored as metadata
            fout.write(json.dumps({"text": full_text, "docid": doc_id}) + "\n")

            n_docs += 1

    print(f"Converted {n_docs} documents to {out_path}")
    return n_docs


def convert_queries_qrels(raw_dir, out_dir, split="test"):
    """
    Convert BEIR queries and qrels to pipeline format.

    Creates:
    - topics.txt: tab-separated qid, query_text
    - qrels.txt: TREC format qrels
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load queries
    queries_file = os.path.join(raw_dir, "queries.jsonl")
    queries = {}
    if os.path.exists(queries_file):
        with open(queries_file) as f:
            for line in f:
                q = json.loads(line.strip())
                queries[q["_id"]] = q.get("text", "")
    print(f"Loaded {len(queries)} queries")

    # Load qrels
    qrels_file = os.path.join(raw_dir, "qrels", f"{split}.tsv")
    if not os.path.exists(qrels_file):
        # Try other locations
        for alt in [f"qrels/{split}.txt", "qrels/test.tsv", "qrels/dev.tsv"]:
            alt_path = os.path.join(raw_dir, alt)
            if os.path.exists(alt_path):
                qrels_file = alt_path
                break

    qrels = {}  # qid -> {doc_id -> rel}
    if os.path.exists(qrels_file):
        with open(qrels_file) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3 and row[0] != "query-id":
                    qid, doc_id, rel = row[0], row[1], int(row[2])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = rel
    print(f"Loaded qrels for {len(qrels)} queries")

    # Filter queries to those with qrels
    active_qids = set(qrels.keys())

    # Write topics.txt
    topics_path = os.path.join(out_dir, "topics.txt")
    n_topics = 0
    with open(topics_path, 'w') as f:
        for qid in sorted(active_qids):
            if qid in queries:
                f.write(f"{qid}\t{queries[qid]}\n")
                n_topics += 1
    print(f"Wrote {n_topics} topics to {topics_path}")

    # Write qrels in TREC format: qid 0 doc_id relevance
    qrels_path = os.path.join(out_dir, "qrels.txt")
    n_judgments = 0
    with open(qrels_path, 'w') as f:
        for qid in sorted(qrels.keys()):
            for doc_id, rel in sorted(qrels[qid].items()):
                f.write(f"{qid} 0 {doc_id} {rel}\n")
                n_judgments += 1
    print(f"Wrote {n_judgments} judgments to {qrels_path}")

    return n_topics, n_judgments


def build_index(corpus_dir, index_dir, cpus=16, mem=64, tokenizer="llama"):
    """
    Build infini-gram index from converted corpus.

    Args:
        corpus_dir: Directory with corpus.jsonl
        index_dir: Where to save the index
        cpus: Number of CPU cores
        mem: RAM in GB
        tokenizer: Tokenizer name (llama, gpt2, olmo)
    """
    os.makedirs(index_dir, exist_ok=True)

    # Estimate tokens to determine shards
    # Rough estimate: 1 token per 4 chars
    corpus_file = os.path.join(corpus_dir, "corpus.jsonl")
    file_size = os.path.getsize(corpus_file)
    est_tokens = file_size // 4
    max_tokens_per_shard = int(0.4 * mem * 1e9)  # 0.4 * RAM in bytes / 2 bytes per token
    shards = max(1, est_tokens // max_tokens_per_shard + 1)

    print(f"Estimated {est_tokens:,} tokens, using {shards} shard(s)")
    print(f"Building index with {cpus} CPUs, {mem}GB RAM...")

    cmd = (
        f"python -m infini_gram.indexing "
        f"--data_dir {corpus_dir} "
        f"--save_dir {index_dir} "
        f"--tokenizer {tokenizer} "
        f"--cpus {cpus} --mem {mem} "
        f"--shards {shards} --add_metadata "
        f"--ulimit 1048576"
    )

    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"WARNING: Indexing returned non-zero exit code: {ret}")
    else:
        print(f"Index built at {index_dir}")

    return index_dir


def create_config(base_dir, dataset_name=None):
    """
    Create a pipeline config.yaml for this BEIR dataset.
    """
    if dataset_name is None:
        dataset_name = Path(base_dir).name

    config = {
        "topics_path": os.path.join(base_dir, "topics.txt"),
        "qrels_path": os.path.join(base_dir, "qrels.txt"),
        "expansions_path": os.path.join(base_dir, "kiss.jsonl"),
        "index_dir": os.path.join(base_dir, "index"),
        "tokenizer_name": "meta-llama/Llama-2-7b-hf",
        "stages": ["retrieval", "biencoder", "crossencoder"],
        "biencoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "crossencoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "max_topics": None,
        "top_k_list": [10, 100, 1000],
        "max_standalone_key": 1500,
        "max_standalone_assoc": 750,
        "prox_peek": 10,
        "max_clause_freq": 80000000,
        "max_docs": 20000,
        "prox_cross": 100,
        "prox_assoc": 100,
        "max_total": 200,
        "biencoder_top_k": 1000,
        "crossencoder_top_k": 100,
        "max_doc_len": 2000,
    }

    config_path = os.path.join(base_dir, "config.yaml")

    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config saved to {config_path}")
    return config_path


def setup_full(dataset_name, base_dir, cpus=16, mem=64):
    """Run the full setup pipeline (download, convert, index)."""
    os.makedirs(base_dir, exist_ok=True)

    raw_dir = os.path.join(base_dir, "raw")
    corpus_dir = os.path.join(base_dir, "corpus")
    index_dir = os.path.join(base_dir, "index")

    print(f"\n{'='*60}")
    print(f"Setting up BEIR dataset: {dataset_name}")
    print(f"Base directory: {base_dir}")
    print(f"{'='*60}")

    # Step 1: Download
    print(f"\n[1/4] Downloading {dataset_name}...")
    data_path = download_beir(dataset_name, raw_dir)

    # Step 2: Convert corpus
    print(f"\n[2/4] Converting corpus...")
    n_docs = convert_corpus(data_path, corpus_dir)

    # Step 3: Convert queries and qrels
    print(f"\n[3/4] Converting queries and qrels...")
    n_topics, n_judgments = convert_queries_qrels(data_path, base_dir)

    # Step 4: Build index
    print(f"\n[4/4] Building infini-gram index ({n_docs} docs)...")
    build_index(corpus_dir, index_dir, cpus=cpus, mem=mem)

    # Create config
    create_config(base_dir, dataset_name)

    print(f"\n{'='*60}")
    print(f"Setup complete!")
    print(f"  Corpus: {n_docs} docs")
    print(f"  Topics: {n_topics}")
    print(f"  Judgments: {n_judgments}")
    print(f"  Index: {index_dir}")
    print(f"  Config: {os.path.join(base_dir, 'config.yaml')}")
    print(f"\nNext steps:")
    print(f"  1. Generate keyword expansions -> {os.path.join(base_dir, 'kiss.jsonl')}")
    print(f"  2. run_from_config('{os.path.join(base_dir, 'config.yaml')}')")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup BEIR dataset for infini-gram pipeline")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name (e.g. scifact, nfcorpus)")
    parser.add_argument("--out_dir", required=True, help="Base output directory")
    parser.add_argument("--cpus", type=int, default=16, help="CPUs for indexing")
    parser.add_argument("--mem", type=int, default=64, help="RAM in GB for indexing")

    args = parser.parse_args()
    setup_full(args.dataset, args.out_dir, cpus=args.cpus, mem=args.mem)