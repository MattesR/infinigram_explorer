"""
Local LLM keyword expansion using Ollama.

Usage:
    # Run a single model
    python ollama_keywords.py run --topics topics_rag24_test.txt --model llama3.2

    # Compare multiple models
    python ollama_keywords.py compare --topics topics_rag24_test.txt \
        --models llama3.2,mistral,phi4-mini --output-dir ./ollama_results/

    # List available models
    python ollama_keywords.py list-models
"""

import argparse
import json
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import ollama as ollama_lib
except ImportError:
    print("Install the ollama package: pip install ollama")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

from batch_keywords import SYSTEM_PROMPT, load_topics

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _make_client():
    return ollama_lib.Client(host=OLLAMA_HOST)


def _strip_fences(text: str) -> str:
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _expand_one(client, qid: str, query: str, model: str) -> tuple[str, dict | None, str | None]:
    """
    Run keyword expansion for one query.
    Returns (qid, parsed_dict_or_None, error_str_or_None).
    """
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0},
        )
        text = response.message.content
        text = _strip_fences(text)
        try:
            parsed = json.loads(text)
            return qid, parsed, None
        except json.JSONDecodeError:
            return qid, {"raw": text}, "json_parse_error"
    except Exception as e:
        return qid, None, str(e)


def _load_done(output_path: str) -> set:
    """Return set of qids already written to output_path (for resume support)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "qid" in obj:
                        done.add(obj["qid"])
                except json.JSONDecodeError:
                    pass
    return done


def run(topics_path: str, model: str, output_path: str, workers: int = 1):
    """Run keyword expansion for all topics with one model, writing to a JSONL file."""
    client = _make_client()
    topics = load_topics(topics_path)
    done = _load_done(output_path)

    pending = [(qid, query) for qid, query in topics if qid not in done]
    if done:
        print(f"Resuming: {len(done)} already done, {len(pending)} remaining.")

    errors = []
    out_f = open(output_path, "a")

    def process(item):
        qid, query = item
        return _expand_one(client, qid, query, model)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, item): item for item in pending}
            for fut in tqdm(as_completed(futures), total=len(pending), desc=model):
                qid, result, err = fut.result()
                if result is not None:
                    out_f.write(json.dumps({"qid": qid, **result}) + "\n")
                    out_f.flush()
                if err:
                    errors.append((qid, err))
    else:
        for item in tqdm(pending, desc=model):
            qid, result, err = process(item)
            if result is not None:
                out_f.write(json.dumps({"qid": qid, **result}) + "\n")
                out_f.flush()
            if err:
                errors.append((qid, err))

    out_f.close()

    total = len(topics)
    success = total - len(done) - len(errors)
    print(f"\nDone. {success + len(done)}/{total} succeeded, {len(errors)} errors.")
    if errors:
        err_path = output_path.replace(".jsonl", "_errors.json")
        with open(err_path, "w") as f:
            json.dump([{"qid": q, "error": e} for q, e in errors], f, indent=2)
        print(f"Errors saved to {err_path}")
    print(f"Results saved to {output_path}")


def compare(topics_path: str, models: list[str], output_dir: str, workers: int = 1):
    """Run all models and save each to a separate file in output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for model in models:
        safe_name = model.replace(":", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_name}.jsonl")
        print(f"\n=== Model: {model} -> {output_path} ===")
        run(topics_path, model, output_path, workers=workers)


def list_models():
    """Print available Ollama models."""
    client = _make_client()
    try:
        models = client.list()
        if not models.models:
            print("No models pulled yet. Example: ollama pull llama3.2")
            return
        print(f"Available models on {OLLAMA_HOST}:")
        for m in models.models:
            size_gb = (m.size or 0) / 1e9
            print(f"  {m.model:<40} {size_gb:.1f} GB")
    except Exception as e:
        print(f"Could not connect to Ollama at {OLLAMA_HOST}: {e}")
        print("Is 'ollama serve' running?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword expansion via local Ollama models")
    parser.add_argument("--host", default=None,
                        help="Ollama host URL (default: $OLLAMA_HOST or http://localhost:11434)")
    subparsers = parser.add_subparsers(dest="command")

    sub = subparsers.add_parser("run", help="Run one model")
    sub.add_argument("--topics", required=True)
    sub.add_argument("--model", required=True)
    sub.add_argument("--output", default=None,
                     help="Output JSONL path (default: keyword_expansions_<model>.jsonl)")
    sub.add_argument("--workers", type=int, default=1,
                     help="Parallel workers (default: 1)")

    sub = subparsers.add_parser("compare", help="Run multiple models and compare")
    sub.add_argument("--topics", required=True)
    sub.add_argument("--models", required=True,
                     help="Comma-separated model names, e.g. llama3.2,mistral,phi4-mini")
    sub.add_argument("--output-dir", default="ollama_results")
    sub.add_argument("--workers", type=int, default=1)

    subparsers.add_parser("list-models", help="List models available on the Ollama server")

    args = parser.parse_args()

    if args.host:
        os.environ["OLLAMA_HOST"] = args.host
        OLLAMA_HOST = args.host

    if args.command == "run":
        safe_name = args.model.replace(":", "_").replace("/", "_")
        output = args.output or f"keyword_expansions_{safe_name}.jsonl"
        run(args.topics, args.model, output, workers=args.workers)

    elif args.command == "compare":
        models = [m.strip() for m in args.models.split(",")]
        compare(args.topics, models, args.output_dir, workers=args.workers)

    elif args.command == "list-models":
        list_models()

    else:
        parser.print_help()
