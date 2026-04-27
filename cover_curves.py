"""
Generate set cover curve data for visualization.

Usage:
    from cover_curves import compute_cover_curves, save_curves_html

    curves = compute_cover_curves(
        found_dir="./inspection/prog/",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
        prox=100,
    )
    save_curves_html(curves, "./cover_curves.html")
"""

import os
import json
from glob import glob
from tqdm import tqdm
from tightest_queries import find_tightest_per_doc


def _greedy_set_cover_curve(all_query_coverage, query_engine_counts, all_doc_ids):
    """
    Run greedy set cover and return the cumulative curve.
    Returns list of dicts with cumulative stats at each step.
    """
    uncovered = set(all_doc_ids)
    n_total = len(all_doc_ids)
    steps = []
    cum_engine = 0
    cum_covered = 0

    step = 0
    while uncovered:
        best_desc = None
        best_eff = -1
        for desc, doc_ids in all_query_coverage.items():
            new_covered = len(doc_ids & uncovered)
            if new_covered == 0:
                continue
            eng = query_engine_counts.get(desc, 1)
            if eng <= 0:
                eng = 1
            eff = new_covered / eng
            if eff > best_eff:
                best_eff = eff
                best_desc = desc

        if best_desc is None:
            break

        doc_ids = all_query_coverage[best_desc]
        new_covered = doc_ids & uncovered
        eng = query_engine_counts.get(best_desc, 0)
        uncovered -= new_covered
        cum_engine += eng
        cum_covered += len(new_covered)
        step += 1

        steps.append({
            "step": step,
            "query": best_desc,
            "new_docs": len(new_covered),
            "cum_docs": cum_covered,
            "cum_recall": cum_covered / n_total if n_total else 0,
            "engine_count": eng,
            "cum_engine": cum_engine,
            "efficiency": len(new_covered) / eng if eng > 0 else 0,
        })

    return steps


def compute_cover_curves(
    found_dir,
    expansions_path,
    tokenizer,
    engine,
    prox=100,
    max_clause_freq=80000000,
    pattern="*_found.jsonl",
    verbose=True,
):
    """Compute set cover curves for all queries."""
    files = sorted(glob(os.path.join(found_dir, pattern)))
    all_curves = {}

    for fpath in tqdm(files, desc="Computing curves", disable=not verbose):
        fname = os.path.basename(fpath)
        qid = fname.replace("_found.jsonl", "")

        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        n_with_text = sum(1 for l in lines if json.loads(l).get("text"))
        if n_with_text == 0:
            continue

        try:
            results = find_tightest_per_doc(
                found_path=fpath, qid=qid,
                expansions_path=expansions_path,
                tokenizer=tokenizer, engine=engine,
                prox=prox, max_clause_freq=max_clause_freq,
                verbose=False,
            )
        except Exception as e:
            if verbose:
                print(f"  {qid}: ERROR {e}")
            continue

        all_doc_ids = {r["doc_id"] for r in results["doc_results"]}
        steps = _greedy_set_cover_curve(
            results["all_query_coverage"],
            results["query_engine_counts"],
            all_doc_ids,
        )
        all_curves[qid] = {
            "steps": steps,
            "n_docs": n_with_text,
        }

    return all_curves


def save_curves_html(curves, output_path="./cover_curves.html"):
    """Save interactive HTML visualization of set cover curves."""

    # Prepare data for charts
    curves_data = []
    for qid, data in curves.items():
        steps = data["steps"]
        n_docs = data["n_docs"]
        curve_points = [{"step": 0, "cum_recall": 0, "cum_engine": 0, "query": ""}]
        for s in steps:
            curve_points.append({
                "step": s["step"],
                "cum_recall": round(s["cum_recall"], 4),
                "cum_engine": s["cum_engine"],
                "query": s["query"][:60],
                "new_docs": s["new_docs"],
                "engine_count": s["engine_count"],
                "efficiency": round(s["efficiency"], 6),
            })
        curves_data.append({
            "qid": qid,
            "n_docs": n_docs,
            "points": curve_points,
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Set Cover Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
    body {{ font-family: 'Menlo', 'Consolas', monospace; background: #0a0a0a; color: #e0e0e0; margin: 0; padding: 20px; }}
    h1 {{ color: #00ff88; font-size: 18px; margin-bottom: 5px; }}
    h2 {{ color: #88aaff; font-size: 14px; margin: 20px 0 10px; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .chart-box {{ background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 15px; }}
    .chart-box canvas {{ max-height: 350px; }}
    .summary {{ background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 15px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ padding: 4px 8px; text-align: right; border-bottom: 1px solid #333; }}
    th {{ color: #00ff88; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .highlight {{ color: #ffaa00; }}
</style>
</head>
<body>
<div class="container">
<h1>Set Cover Analysis — Engine Count vs Recall</h1>

<div class="summary">
<h2>Summary</h2>
<table>
<tr><th>QID</th><th>Docs</th><th>Queries</th><th>Total Engine</th>
<th>R@50%</th><th>Eng@50%</th><th>R@80%</th><th>Eng@80%</th><th>R@95%</th><th>Eng@95%</th></tr>
"""

    for cd in curves_data:
        qid = cd["qid"]
        n_docs = cd["n_docs"]
        points = cd["points"]
        n_queries = len(points) - 1
        total_eng = points[-1]["cum_engine"] if points else 0

        # Find engine count at recall thresholds
        def eng_at_recall(target):
            for p in points:
                if p["cum_recall"] >= target:
                    return p["cum_engine"], round(p["cum_recall"], 2)
            return total_eng, round(points[-1]["cum_recall"], 2) if points else (0, 0)

        e50, r50 = eng_at_recall(0.5)
        e80, r80 = eng_at_recall(0.8)
        e95, r95 = eng_at_recall(0.95)

        html += f"""<tr><td>{qid}</td><td>{n_docs}</td><td>{n_queries}</td>
<td>{total_eng:,}</td>
<td>{r50:.0%}</td><td>{e50:,}</td>
<td>{r80:.0%}</td><td>{e80:,}</td>
<td>{r95:.0%}</td><td>{e95:,}</td></tr>
"""

    html += """</table></div>

<div class="charts">
"""

    # Generate a chart for each query
    colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a29bfe',
              '#fd79a8', '#00cec9', '#fab1a0', '#81ecec', '#dfe6e9']

    for i, cd in enumerate(curves_data):
        qid = cd["qid"]
        points = cd["points"]
        color = colors[i % len(colors)]

        recalls = [p["cum_recall"] * 100 for p in points]
        engines = [p["cum_engine"] for p in points]
        steps = [p["step"] for p in points]

        # Recall vs step
        html += f"""
<div class="chart-box">
<h2>{qid} ({cd['n_docs']} docs)</h2>
<canvas id="chart_recall_{i}"></canvas>
<canvas id="chart_engine_{i}" style="margin-top:10px;"></canvas>
<script>
new Chart(document.getElementById('chart_recall_{i}'), {{
    type: 'line',
    data: {{
        labels: {steps},
        datasets: [{{
            label: 'Recall %',
            data: {recalls},
            borderColor: '{color}',
            backgroundColor: '{color}22',
            fill: true,
            tension: 0.1,
            pointRadius: 2,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Query #', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
            y: {{ title: {{ display: true, text: 'Recall %', color: '#888' }},
                  min: 0, max: 100,
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});
new Chart(document.getElementById('chart_engine_{i}'), {{
    type: 'line',
    data: {{
        labels: {steps},
        datasets: [{{
            label: 'Cumulative Engine Count',
            data: {engines},
            borderColor: '#ff6b6b',
            tension: 0.1,
            pointRadius: 2,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Query #', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
            y: {{ title: {{ display: true, text: 'Engine Count', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});
</script>
</div>
"""

    # Aggregate chart: all queries overlaid (recall vs cum_engine)
    html += f"""
<div class="chart-box" style="grid-column: 1 / -1;">
<h2>All Queries: Recall vs Engine Count</h2>
<canvas id="chart_all"></canvas>
<script>
new Chart(document.getElementById('chart_all'), {{
    type: 'scatter',
    data: {{
        datasets: [
"""
    for i, cd in enumerate(curves_data):
        points = cd["points"]
        color = colors[i % len(colors)]
        data_points = [{"x": p["cum_engine"], "y": round(p["cum_recall"] * 100, 1)}
                       for p in points]
        html += f"""{{
            label: '{cd["qid"]}',
            data: {data_points},
            borderColor: '{color}',
            backgroundColor: '{color}44',
            showLine: true,
            tension: 0.1,
            pointRadius: 1,
        }},
"""

    html += f"""
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#888', font: {{ size: 10 }} }} }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Cumulative Engine Count', color: '#888' }},
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }}, type: 'linear' }},
            y: {{ title: {{ display: true, text: 'Recall %', color: '#888' }},
                  min: 0, max: 100,
                  ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}
        }}
    }}
}});
</script>
</div>
"""

    html += """
</div>
</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved to {output_path}")

    return output_path