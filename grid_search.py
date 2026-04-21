"""
Grid search over retrieval hyperparameters.

Usage:
    from grid_search import grid_search_recall

    results = grid_search_recall(
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        engine=engine,
        tokenizer=tokenizer,
        expansions_path="./kiss.jsonl",
        max_topics=10,
        param_grid={
            "max_docs": [5000, 10000, 20000],
            "max_combo_grab": [3000, 5000],
            "prox_cross": [30, 50, 100],
        },
    )
"""

from itertools import product as iter_product
from recall_ceiling import compare_recall_ceiling


def grid_search_recall(
    topics_path: str,
    qrels_path: str,
    engine,
    tokenizer,
    expansions_path: str,
    param_grid: dict,
    max_topics: int = None,
    max_clause_freq: int = 80000000,
    save_inspection: bool = False,
    inspection_base_dir: str = "./inspection/grid",
    # Default values for params not in grid
    max_standalone_key: int = 1000,
    max_standalone_assoc: int = 200,
    prox_peek: int = 10,
    max_docs: int = 10000,
    max_combo_grab: int = 5000,
    prox_cross: int = 50,
    prox_assoc: int = 80,
    return_docs: bool = False,
    verbose: bool = True,
):
    """
    Grid search over progressive retrieval hyperparameters.

    Args:
        param_grid: Dict of param_name -> list of values to try.
            e.g. {"max_docs": [5000, 10000], "prox_cross": [30, 50, 100]}
        Other args: default values used when param not in grid.

    Returns dict of config_label -> results.
    """
    # Build all configs
    defaults = {
        "max_standalone_key": max_standalone_key,
        "max_standalone_assoc": max_standalone_assoc,
        "prox_peek": prox_peek,
        "max_docs": max_docs,
        "max_combo_grab": max_combo_grab,
        "prox_cross": prox_cross,
        "prox_assoc": prox_assoc,
    }

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    configs = []
    for combo in iter_product(*param_values):
        config = dict(defaults)
        label_parts = []
        for name, val in zip(param_names, combo):
            config[name] = val
            label_parts.append(f"{name}={val}")
        label = "_".join(label_parts)
        configs.append((label, config))

    if verbose:
        print(f"Grid search: {len(configs)} configurations")
        print(f"  Parameters: {param_names}")
        print(f"  Topics: {max_topics or 'all'}")
        for label, config in configs:
            print(f"  {label}")
        print()

    all_results = {}

    for i, (label, config) in enumerate(configs):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Config {i+1}/{len(configs)}: {label}")
            print(f"{'='*70}")

        insp_dir = f"{inspection_base_dir}/{label}" if save_inspection else "./inspection"

        results = compare_recall_ceiling(
            topics_path=topics_path,
            qrels_path=qrels_path,
            engine=engine,
            tokenizer=tokenizer,
            progressive_paths={"prog": expansions_path},
            max_topics=max_topics,
            max_clause_freq=max_clause_freq,
            save_inspection=save_inspection,
            inspection_dir=insp_dir,
            return_docs=return_docs,
            **config,
        )

        all_results[label] = results.get("prog", [])

    # Summary comparison
    if verbose:
        print(f"\n{'='*80}")
        print(f"GRID SEARCH SUMMARY")
        print(f"{'='*80}")

        header = f"{'Config':<45s} {'Topics':>7s} {'Recall':>8s} {'Retr':>8s} {'Found':>7s} {'Time':>7s}"
        print(header)
        print("-" * len(header))

        sorted_configs = sorted(
            all_results.items(),
            key=lambda x: sum(r["recall"] for r in x[1]) / len(x[1]) if x[1] else 0,
            reverse=True,
        )

        for label, results_list in sorted_configs:
            if not results_list:
                print(f"{label:<45s}  No results")
                continue
            n = len(results_list)
            avg_recall = sum(r["recall"] for r in results_list) / n
            avg_retr = sum(r["n_retrieved"] for r in results_list) / n
            avg_found = sum(r["n_found"] for r in results_list) / n
            avg_time = sum(r.get("time_query", 0) + r.get("time_resolve", 0) for r in results_list) / n
            print(f"{label:<45s} {n:>7d} {avg_recall:>8.4f} {avg_retr:>8.0f} {avg_found:>7.1f} {avg_time:>6.1f}s")

        # Best config
        if sorted_configs:
            best_label = sorted_configs[0][0]
            print(f"\n  Best: {best_label}")

    return all_results