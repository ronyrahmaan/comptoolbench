#!/usr/bin/env python3
"""Generate the final leaderboard table combining all benchmark runs.

Merges:
  1. V3 unified results (6 cloud models from checkpoint files)
  2. V3 leaderboard CSV (has additional Cohere + Cerebras models)
  3. Local/Ollama models (hardcoded from paper v1, no checkpoint files)
  4. Wave 1: GPT-4o, GPT-4o Mini, Claude Sonnet 4
  5. Wave 2: GPT-5.4, GPT-4.1, o3, Claude Opus 4, Claude Haiku 4.5, Gemini 2.5 Flash

Outputs:
  - leaderboard_final.csv (all models)
  - leaderboard_final.tex (LaTeX table for paper)
  - summary stats (averages, model counts)

Usage:
    uv run python scripts/generate_final_leaderboard.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ──────────────────────────────────────────────────────────────────────
# Level prefixes in checkpoint files
# ──────────────────────────────────────────────────────────────────────
LEVEL_PREFIXES = {
    "L0": "L0_node",
    "L1": "L1_chain",
    "L2": "L2_parallel",
    "L3": "L3_dag",
}

LEVEL_COUNTS = {"L0": 48, "L1": 64, "L2": 40, "L3": 48}


def parse_checkpoint(path: Path) -> dict[str, Any]:
    """Parse a JSONL checkpoint file and compute per-level accuracy."""
    results: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if len(results) < 200:
        print(f"  WARNING: {path.name} has only {len(results)}/200 tasks")

    # Per-level accuracy
    level_scores: dict[str, list[float]] = {k: [] for k in LEVEL_PREFIXES}
    all_scores: list[float] = []

    for r in results:
        task_id = r.get("task_id", "")
        score_data = r.get("score", {})
        score = score_data.get("overall", 0.0) if isinstance(score_data, dict) else float(score_data)
        all_scores.append(score)

        for level_key, prefix in LEVEL_PREFIXES.items():
            if task_id.startswith(prefix):
                level_scores[level_key].append(score)
                break

    # Compute accuracy as mean score (weighted scoring)
    overall = np.mean(all_scores) if all_scores else 0.0
    level_acc = {}
    for lk, scores in level_scores.items():
        level_acc[lk] = np.mean(scores) if scores else None

    # Selection Gap = L0 - avg(L1, L2, L3)
    composed_levels = [level_acc[k] for k in ["L1", "L2", "L3"] if level_acc[k] is not None]
    sel_gap = level_acc["L0"] - np.mean(composed_levels) if composed_levels and level_acc["L0"] is not None else None

    return {
        "n_tasks": len(results),
        "overall": overall,
        "L0": level_acc.get("L0"),
        "L1": level_acc.get("L1"),
        "L2": level_acc.get("L2"),
        "L3": level_acc.get("L3"),
        "sel_gap": sel_gap,
    }


# ──────────────────────────────────────────────────────────────────────
# Hardcoded local model results (from paper v1, no checkpoint files)
# ──────────────────────────────────────────────────────────────────────
LOCAL_MODELS = [
    {"model": "Granite4 3B", "provider": "Ollama", "L0": 45.8, "L1": 57.3, "L2": 56.1, "L3": 30.2, "overall": 47.8, "delta": 15.6, "sel_gap_flag": True},
    {"model": "Granite4 1B", "provider": "Ollama", "L0": 41.7, "L1": 56.3, "L2": 55.9, "L3": 29.9, "overall": 46.4, "delta": 11.8, "sel_gap_flag": True},
    {"model": "Mistral 7B", "provider": "Ollama", "L0": 43.8, "L1": 57.7, "L2": 49.2, "L3": 30.5, "overall": 46.1, "delta": 13.3, "sel_gap_flag": True},
    {"model": "Llama 3.1 8B", "provider": "Ollama", "L0": 39.6, "L1": 56.7, "L2": 56.1, "L3": 29.5, "overall": 45.9, "delta": 10.1, "sel_gap_flag": True},
    {"model": "Mistral Nemo 12B", "provider": "Ollama", "L0": 37.5, "L1": 58.4, "L2": 51.0, "L3": 31.8, "overall": 45.5, "delta": 5.7, "sel_gap_flag": True},
    {"model": "Qwen 2.5 7B", "provider": "Ollama", "L0": 39.6, "L1": 56.7, "L2": 53.8, "L3": 25.8, "overall": 44.6, "delta": 13.8, "sel_gap_flag": True},
    {"model": "Mistral Small 24B", "provider": "Ollama", "L0": 37.5, "L1": 51.1, "L2": 47.7, "L3": 22.6, "overall": 40.3, "delta": 14.9, "sel_gap_flag": True},
    {"model": "Qwen3 8B", "provider": "Ollama", "L0": 35.4, "L1": 52.0, "L2": 36.9, "L3": 21.8, "overall": 37.7, "delta": 13.7, "sel_gap_flag": True},
]

# Model display names and providers for checkpoint-based models
MODEL_META = {
    # V3 cloud models
    "groq-llama3.1-8b": ("Llama 3.1 8B", "Groq"),
    "groq-llama4-scout": ("Llama 4 Scout 17B", "Groq"),
    "mistral-small": ("Mistral Small", "Mistral"),
    "mistral-medium": ("Mistral Medium", "Mistral"),
    "mistral-large": ("Mistral Large", "Mistral"),
    "or-gemini-2.0-flash": ("Gemini 2.0 Flash", "OpenRouter"),
    # V3 CSV-only models (Cohere, Cerebras)
    # Wave 1
    "gpt-4o": ("GPT-4o", "OpenAI"),
    "gpt-4o-mini": ("GPT-4o Mini", "OpenAI"),
    "claude-sonnet-4": ("Claude Sonnet 4", "Anthropic"),
    # Wave 2
    "gpt-5.4": ("GPT-5.4", "OpenAI"),
    "gpt-4.1": ("GPT-4.1", "OpenAI"),
    "o3": ("OpenAI o3", "OpenAI"),
    "claude-opus-4": ("Claude Opus 4", "Anthropic"),
    "claude-haiku-4.5": ("Claude Haiku 4.5", "Anthropic"),
    "gemini-2.5-flash": ("Gemini 2.5 Flash", "Google"),
    "or-gemini-2.5-flash": ("Gemini 2.5 Flash", "Google"),
}

# V3 leaderboard CSV models not available as checkpoints
CSV_ONLY_MODELS = {
    "Command A": {"provider": "Cohere", "L0": 45.83, "L1": 62.69, "L2": 87.76, "L3": 40.78, "overall": 58.40},
    "Command R+": {"provider": "Cohere", "L0": 43.75, "L1": 57.47, "L2": 87.97, "L3": 40.33, "overall": 56.16},
    "Llama 3.1 8B (Cerebras)": {"provider": "Cerebras", "L0": 31.25, "L1": 66.08, "L2": 81.18, "L3": 46.39, "overall": 56.01},
    "GPT-OSS 120B": {"provider": "Cerebras", "L0": 45.83, "L1": 56.29, "L2": 56.06, "L3": 29.03, "overall": 47.19},
}


def main() -> None:
    """Generate the final combined leaderboard."""
    all_models: list[dict[str, Any]] = []

    # 1. Parse checkpoint files from all runs
    checkpoint_dirs = [
        RESULTS_DIR / "unified_v3",           # V3 cloud
        RESULTS_DIR / "run_20260311_000147",   # Wave 1
        RESULTS_DIR / "run_20260311_003312",   # Wave 2
        RESULTS_DIR / "run_20260311_020636",   # Gemini 2.5 Flash (via OpenRouter)
    ]

    seen_models: set[str] = set()

    for run_dir in checkpoint_dirs:
        if not run_dir.exists():
            print(f"Skipping {run_dir} (not found)")
            continue
        for cp in sorted(run_dir.glob("checkpoint_*.jsonl")):
            model_key = cp.stem.replace("checkpoint_", "")
            if model_key in seen_models:
                continue

            stats = parse_checkpoint(cp)
            if stats["n_tasks"] < 200:
                print(f"  Skipping {model_key} ({stats['n_tasks']}/200 incomplete)")
                continue

            seen_models.add(model_key)
            display_name, provider = MODEL_META.get(model_key, (model_key, "Unknown"))

            l0 = stats["L0"] * 100 if stats["L0"] is not None else None
            l1 = stats["L1"] * 100 if stats["L1"] is not None else None
            l2 = stats["L2"] * 100 if stats["L2"] is not None else None
            l3 = stats["L3"] * 100 if stats["L3"] is not None else None
            overall = stats["overall"] * 100

            delta = (l0 - l3) if l0 is not None and l3 is not None else None
            composed_avg = np.mean([x for x in [l1, l2, l3] if x is not None])
            has_sel_gap = l0 is not None and l0 < composed_avg

            all_models.append({
                "model": display_name,
                "provider": provider,
                "L0": l0, "L1": l1, "L2": l2, "L3": l3,
                "overall": overall,
                "delta": delta,
                "sel_gap_flag": has_sel_gap,
                "deployment": "cloud" if provider != "Ollama" else "local",
            })

    # 2. Add CSV-only models (Cohere, Cerebras)
    for name, data in CSV_ONLY_MODELS.items():
        l0 = data["L0"] * 100 if data["L0"] < 1 else data["L0"]
        l1 = data["L1"] * 100 if data["L1"] < 1 else data["L1"]
        l2 = data["L2"] * 100 if data["L2"] < 1 else data["L2"]
        l3 = data["L3"] * 100 if data["L3"] < 1 else data["L3"]
        overall = data["overall"] * 100 if data["overall"] < 1 else data["overall"]

        delta = l0 - l3
        composed_avg = np.mean([l1, l2, l3])
        has_sel_gap = l0 < composed_avg

        all_models.append({
            "model": name,
            "provider": data["provider"],
            "L0": l0, "L1": l1, "L2": l2, "L3": l3,
            "overall": overall,
            "delta": delta,
            "sel_gap_flag": has_sel_gap,
            "deployment": "cloud",
        })

    # 3. Add local models
    for lm in LOCAL_MODELS:
        lm_copy = dict(lm)
        lm_copy["deployment"] = "local"
        all_models.append(lm_copy)

    # Sort: frontier first (by overall desc), then cloud (by overall desc), then local (by overall desc)
    def sort_key(m: dict) -> tuple:
        is_frontier = m["provider"] in ("OpenAI", "Anthropic", "Google")
        is_local = m["deployment"] == "local"
        return (not is_frontier, is_local, -m["overall"])

    all_models.sort(key=sort_key)

    # Print summary
    print(f"\n{'='*90}")
    print(f"COMPTOOLBENCH FINAL LEADERBOARD — {len(all_models)} MODELS")
    print(f"{'='*90}")
    print(f"{'Model':<28s} {'Provider':<12s} {'L0':>6s} {'L1':>6s} {'L2':>6s} {'L3':>6s} {'Overall':>8s} {'Δ':>7s} {'SelGap':>7s}")
    print("-" * 90)

    frontier_models = []
    cloud_models = []
    local_models = []

    for m in all_models:
        is_frontier = m["provider"] in ("OpenAI", "Anthropic", "Google")
        is_local = m["deployment"] == "local"

        l0 = f"{m['L0']:.1f}" if m['L0'] is not None else "—"
        l1 = f"{m['L1']:.1f}" if m['L1'] is not None else "—"
        l2 = f"{m['L2']:.1f}" if m['L2'] is not None else "—"
        l3 = f"{m['L3']:.1f}" if m['L3'] is not None else "—"
        overall = f"{m['overall']:.1f}"
        delta = f"{m['delta']:.1f}" if m['delta'] is not None else "—"
        gap = "†" if m.get("sel_gap_flag") else ""

        print(f"{m['model']:<28s} {m['provider']:<12s} {l0:>6s} {l1:>6s} {l2:>6s} {l3:>6s} {overall:>8s} {delta:>7s} {gap:>7s}")

        if is_frontier:
            frontier_models.append(m)
        elif is_local:
            local_models.append(m)
        else:
            cloud_models.append(m)

    # Averages
    print("-" * 90)
    for label, group in [("Frontier", frontier_models), ("Free Cloud", cloud_models), ("Local", local_models), ("All", all_models)]:
        if not group:
            continue
        avg_l0 = np.mean([m["L0"] for m in group if m["L0"] is not None])
        avg_l1 = np.mean([m["L1"] for m in group if m["L1"] is not None])
        avg_l2 = np.mean([m["L2"] for m in group if m["L2"] is not None])
        avg_l3 = np.mean([m["L3"] for m in group if m["L3"] is not None])
        avg_overall = np.mean([m["overall"] for m in group])
        avg_delta = avg_l0 - avg_l3
        n_sel_gap = sum(1 for m in group if m.get("sel_gap_flag"))
        print(f"{'  ' + label + ' avg':<28s} {'':12s} {avg_l0:>6.1f} {avg_l1:>6.1f} {avg_l2:>6.1f} {avg_l3:>6.1f} {avg_overall:>8.1f} {avg_delta:>7.1f}  ({n_sel_gap}/{len(group)} gap)")

    # Save CSV
    out_csv = RESULTS_DIR / "leaderboard_final.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "provider", "deployment", "L0", "L1", "L2", "L3", "overall", "delta", "sel_gap_flag"])
        writer.writeheader()
        for m in all_models:
            writer.writerow({k: m.get(k) for k in writer.fieldnames})
    print(f"\nSaved: {out_csv}")

    # Generate LaTeX table
    generate_latex_table(all_models, frontier_models, cloud_models, local_models)


def generate_latex_table(
    all_models: list[dict],
    frontier: list[dict],
    cloud: list[dict],
    local: list[dict],
) -> None:
    """Generate LaTeX leaderboard table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results on \bench{}. Models ranked by overall accuracy within deployment category. \textbf{Bold} = best per column. $\Delta$ = \lzero{} $-$ \lthree{} gap (positive $=$ degradation). $\dagger$ = exhibits Selection Gap (\lzero{} $<$ avg of \lone{}--\lthree{}).}")
    lines.append(r"\label{tab:leaderboard}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Provider & \lzero{} & \lone{} & \ltwo{} & \lthree{} & Overall & $\Delta$\,$\downarrow$ \\")
    lines.append(r"\midrule")

    # Find best per column across all models
    best = {
        "L0": max(m["L0"] for m in all_models if m["L0"] is not None),
        "L1": max(m["L1"] for m in all_models if m["L1"] is not None),
        "L2": max(m["L2"] for m in all_models if m["L2"] is not None),
        "L3": max(m["L3"] for m in all_models if m["L3"] is not None),
        "overall": max(m["overall"] for m in all_models),
    }

    def fmt_val(val: float | None, col: str) -> str:
        if val is None:
            return "---"
        s = f"{val:.1f}"
        if col in best and abs(val - best[col]) < 0.05:
            s = r"\textbf{" + s + "}"
        return s

    def model_row(m: dict) -> str:
        dag = "$^\\dagger$" if m.get("sel_gap_flag") else ""
        delta = f"{m['delta']:.1f}" if m['delta'] is not None else "---"
        if m['delta'] is not None and m['delta'] < 0:
            delta = f"$-${abs(m['delta']):.1f}"

        return (
            f"{m['model']}{dag} & {m['provider']} & "
            f"{fmt_val(m['L0'], 'L0')} & {fmt_val(m['L1'], 'L1')} & "
            f"{fmt_val(m['L2'], 'L2')} & {fmt_val(m['L3'], 'L3')} & "
            f"{fmt_val(m['overall'], 'overall')} & {delta} \\\\"
        )

    # Frontier models
    lines.append(r"\multicolumn{8}{l}{\textit{Frontier models (paid API)}} \\")
    for m in frontier:
        lines.append(model_row(m))
    lines.append(r"\midrule")

    # Free cloud models
    lines.append(r"\multicolumn{8}{l}{\textit{Free cloud models}} \\")
    for m in cloud:
        lines.append(model_row(m))
    lines.append(r"\midrule")

    # Local models
    lines.append(r"\multicolumn{8}{l}{\textit{Local models (Ollama)}} \\")
    for m in local:
        lines.append(model_row(m))
    lines.append(r"\midrule")

    # Averages
    for label, group in [("All models", all_models), ("Frontier", frontier), ("Free cloud", cloud), ("Local", local)]:
        if not group:
            continue
        avg_l0 = np.mean([m["L0"] for m in group if m["L0"] is not None])
        avg_l1 = np.mean([m["L1"] for m in group if m["L1"] is not None])
        avg_l2 = np.mean([m["L2"] for m in group if m["L2"] is not None])
        avg_l3 = np.mean([m["L3"] for m in group if m["L3"] is not None])
        avg_overall = np.mean([m["overall"] for m in group])
        avg_delta = avg_l0 - avg_l3
        lines.append(
            f"\\textit{{{label} avg.}} & & \\textit{{{avg_l0:.1f}}} & "
            f"\\textit{{{avg_l1:.1f}}} & \\textit{{{avg_l2:.1f}}} & "
            f"\\textit{{{avg_l3:.1f}}} & \\textit{{{avg_overall:.1f}}} & "
            f"\\textit{{{avg_delta:.1f}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = Path(__file__).parent.parent / "paper" / "tables" / "leaderboard_final.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
