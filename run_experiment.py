"""
Config-based experiment runner.

Usage:
    from run_experiment import run_from_config

    run_from_config("./experiments/exp01/config.yaml")

Or with pre-loaded engine:
    run_from_config("./experiments/exp01/config.yaml", engine=engine)
"""

import os
import sys
import yaml
import time
import csv
import pickle
import shutil
from datetime import datetime
from pathlib import Path

from pipeline import run_batch, evaluate, DEFAULT_CONFIG


DEFAULT_EXPERIMENT_CONFIG = {
    # Paths
    "topics_path": "./topics_rag24_test.txt",
    "qrels_path": "./qrels.rag24.test-umbrela-all.txt",
    "expansions_path": "./kiss.jsonl",
    "out_path": None,  # defaults to config file's directory

    # Index and tokenizer
    "index_dir": "/home/mruc/msmarco_segmented_index/",
    "tokenizer_name": "meta-llama/Llama-2-7b-hf",

    # Stages to run
    "stages": ["retrieval", "biencoder", "crossencoder"],

    # Models
    "biencoder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "crossencoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",

    # Limits
    "max_topics": None,

    # Eval
    "top_k_list": [10, 100, 1000],
}


def load_config(config_path):
    """Load config from YAML file, merged with defaults."""
    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    config = dict(DEFAULT_EXPERIMENT_CONFIG)
    config.update(user_config)

    # Pipeline hyperparams: merge with DEFAULT_CONFIG
    pipeline_cfg = dict(DEFAULT_CONFIG)
    for k, v in user_config.items():
        if k in DEFAULT_CONFIG:
            pipeline_cfg[k] = v
    # Pass index_dir through to pipeline
    pipeline_cfg["index_dir"] = config["index_dir"]
    config["pipeline"] = pipeline_cfg

    # Default out_path to config file's directory
    if config["out_path"] is None:
        config["out_path"] = str(Path(config_path).parent.resolve())

    return config


def load_resources(config, engine=None, tokenizer=None):
    """
    Load engine, tokenizer, and models.

    Pass engine/tokenizer to reuse existing instances.
    """
    resources = {}

    # Tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {config['tokenizer_name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_name"],
            add_bos_token=False,
            add_eos_token=False,
        )
    resources["tokenizer"] = tokenizer

    # Engine
    if engine is None:
        from infini_gram.engine import InfiniGramEngine
        print(f"Loading engine: {config['index_dir']}")
        engine = InfiniGramEngine(
            index_dir=config["index_dir"],
            eos_token_id=tokenizer.eos_token_id,
        )
    resources["engine"] = engine

    # Bi-encoder
    if "biencoder" in config.get("stages", []):
        from sentence_transformers import SentenceTransformer
        model_name = config.get("biencoder_model")
        if model_name:
            print(f"Loading bi-encoder: {model_name}")
            resources["biencoder_model"] = SentenceTransformer(model_name)
    else:
        resources["biencoder_model"] = None

    # Cross-encoder
    if "crossencoder" in config.get("stages", []):
        from sentence_transformers import CrossEncoder
        model_name = config.get("crossencoder_model")
        if model_name:
            print(f"Loading cross-encoder: {model_name}")
            resources["crossencoder_model"] = CrossEncoder(model_name)
    else:
        resources["crossencoder_model"] = None

    return resources


def _write_trec_run(results, stage, out_path, run_name):
    """Write TREC format run file for a stage."""
    filepath = os.path.join(out_path, f"trec_run_{stage}.txt")

    if stage == "pool":
        doc_key = "pool_docs"
    elif stage == "biencoder":
        doc_key = "biencoder_ranked"
    elif stage == "crossencoder":
        doc_key = "crossencoder_ranked"
    else:
        return

    with open(filepath, "w") as f:
        for r in results:
            qid = r["qid"]
            docs = r.get(doc_key, [])
            if docs is None:
                continue
            for rank, doc in enumerate(docs):
                did = doc.get("doc_id", "")
                score = doc.get("score", 1000 - rank)
                f.write(f"{qid} Q0 {did} {rank+1} {score:.6f} {run_name}\n")

    print(f"  Wrote: {filepath}")


def _write_per_query_csv(eval_results, top_k_list, out_path):
    """Write per-query metrics CSV."""
    filepath = os.path.join(out_path, "per_query.csv")

    rows = []
    for stage, results in eval_results.items():
        for r in results:
            row = {
                "stage": stage,
                "qid": r["qid"],
                "n_relevant": r["n_relevant"],
                "pool_found": r.get("pool_found", r.get("pool_size", 0)),
                "mrr": r["cutoffs"].get("mrr", 0),
            }
            for k in top_k_list:
                if k in r["cutoffs"]:
                    row[f"recall_{k}"] = r["cutoffs"][k]["recall"]
                    row[f"ndcg_{k}"] = r["cutoffs"][k]["ndcg"]
                    row[f"precision_{k}"] = r["cutoffs"][k]["precision"]
                    row[f"found_{k}"] = r["cutoffs"][k]["found"]
            rows.append(row)

    if rows:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote: {filepath}")


def run_from_config(config_path, engine=None, tokenizer=None, verbose=True):
    """
    Run full experiment from a config file.

    Args:
        config_path: Path to YAML config file.
        engine: Pre-loaded InfiniGramEngine (optional).
        tokenizer: Pre-loaded tokenizer (optional).

    Returns (results, eval_results).
    """
    t_start = time.perf_counter()

    # Load config
    config = load_config(config_path)
    out_path = config["out_path"]
    os.makedirs(out_path, exist_ok=True)

    if verbose:
        print(f"{'='*70}")
        print(f"EXPERIMENT: {config_path}")
        print(f"  Output: {out_path}")
        print(f"  Stages: {config['stages']}")
        print(f"  Topics: {config.get('max_topics', 'all')}")
        print(f"{'='*70}")

    # Save resolved config (separate file, never overwrites original)
    resolved_path = os.path.join(out_path, "config_resolved.yaml")
    with open(resolved_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load resources
    resources = load_resources(config, engine=engine, tokenizer=tokenizer)

    # Run pipeline
    if verbose:
        print(f"\nRunning pipeline...")

    results = run_batch(
        topics_path=config["topics_path"],
        expansions_path=config["expansions_path"],
        config=config["pipeline"],
        resources=resources,
        max_topics=config.get("max_topics"),
        verbose=verbose,
    )

    # Save results
    results_path = os.path.join(out_path, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Wrote: {results_path}")

    # Evaluate
    if verbose:
        print(f"\nEvaluating...")

    eval_stages = []
    if "retrieval" in config["stages"]:
        eval_stages.append("pool")
    if "biencoder" in config["stages"]:
        eval_stages.append("biencoder")
    if "crossencoder" in config["stages"]:
        eval_stages.append("crossencoder")

    top_k_list = config.get("top_k_list", [10, 100, 1000])

    eval_results = evaluate(
        results,
        qrels_path=config["qrels_path"],
        stages=eval_stages,
        top_k_list=top_k_list,
        verbose=verbose,
    )

    # Write TREC run files
    if verbose:
        print(f"\nWriting output files...")

    run_name = Path(config_path).parent.name or "run"
    for stage in eval_stages:
        _write_trec_run(results, stage, out_path, run_name)

    # Write per-query CSV
    _write_per_query_csv(eval_results, top_k_list, out_path)

    # Write metrics CSV (pytrec_eval based)
    from recalc_metrics import recalc_from_pickle
    recalc_from_pickle(
        results_path, config["qrels_path"],
        stages=eval_stages, top_k_list=top_k_list,
        output_csv=os.path.join(out_path, "metrics.csv"),
        verbose=False,
    )

    # Write eval summary
    summary_path = os.path.join(out_path, "eval_summary.txt")
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    evaluate(results, qrels_path=config["qrels_path"],
             stages=eval_stages, top_k_list=top_k_list, verbose=True)
    sys.stdout = old_stdout
    with open(summary_path, "w") as f:
        f.write(buffer.getvalue())
    print(f"  Wrote: {summary_path}")

    t_total = time.perf_counter() - t_start
    print(f"\nDone in {t_total:.1f}s. Outputs in {out_path}")

    return results, eval_results