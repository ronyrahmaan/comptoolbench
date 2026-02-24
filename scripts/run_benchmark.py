#!/usr/bin/env python3
"""Run the CompToolBench benchmark across available models.

Usage:
    # Quick smoke test (20 tasks, 1 model):
    uv run python scripts/run_benchmark.py --smoke

    # Run with all local Ollama models (free, $0):
    uv run python scripts/run_benchmark.py --local-only

    # Run cloud models only (Groq, Mistral, Gemini, xAI, DeepSeek):
    uv run python scripts/run_benchmark.py --cloud-only

    # Full benchmark: all 27 models × 2,500 tasks:
    uv run python scripts/run_benchmark.py --full

    # Specific models with custom task count:
    uv run python scripts/run_benchmark.py --models groq-llama3.3-70b mistral-large --tasks 500

    # Resume an interrupted run:
    uv run python scripts/run_benchmark.py --output-dir results/run_20260222_120000

    # Generate figures from existing results:
    uv run python scripts/run_benchmark.py --figures-only results/run_001/results_*.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comptoolbench.evaluation.model_adapter import AVAILABLE_MODELS, Provider
from comptoolbench.evaluation.runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("comptoolbench")

# Model groups for different run modes
LOCAL_MODELS = [
    k for k, v in AVAILABLE_MODELS.items()
    if v.provider == Provider.OLLAMA and v.supports_tools
]

CLOUD_MODELS = [
    k for k, v in AVAILABLE_MODELS.items()
    if v.provider != Provider.OLLAMA and v.supports_tools
]

ALL_MODELS = [
    k for k, v in AVAILABLE_MODELS.items()
    if v.supports_tools
]

# Models that need specific env vars
PROVIDER_ENV_VARS = {
    Provider.GEMINI: "GEMINI_API_KEY",
    Provider.GROQ: "GROQ_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.MISTRAL: "MISTRAL_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.XAI: "XAI_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.CEREBRAS: "CEREBRAS_API_KEY",
    Provider.SAMBANOVA: "SAMBANOVA_API_KEY",
    Provider.COHERE: "COHERE_API_KEY",
}


def get_available_models(model_keys: list[str]) -> list[str]:
    """Filter to models that have their required API keys set."""
    available = []
    missing_keys: dict[str, list[str]] = {}

    for key in model_keys:
        config = AVAILABLE_MODELS.get(key)
        if config is None:
            logger.warning("Unknown model key '%s' — skipping", key)
            continue
        if config.provider == Provider.OLLAMA:
            available.append(key)
            continue

        env_var = PROVIDER_ENV_VARS.get(config.provider)
        if env_var and not os.environ.get(env_var):
            missing_keys.setdefault(env_var, []).append(key)
            continue

        available.append(key)

    if missing_keys:
        for env_var, models in missing_keys.items():
            logger.warning(
                "Skipping %s (missing %s): %s",
                ", ".join(models), env_var, ", ".join(models),
            )

    return available


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CompToolBench benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test: 20 tasks, first local model only",
    )
    group.add_argument(
        "--local-only", action="store_true",
        help="Run all local Ollama models with tool support ($0 cost)",
    )
    group.add_argument(
        "--cloud-only", action="store_true",
        help="Run cloud API models only (Groq, Gemini, Mistral, xAI, DeepSeek)",
    )
    group.add_argument(
        "--full", action="store_true",
        help="Run ALL models with tool support (27 models)",
    )
    group.add_argument(
        "--figures-only", type=str, metavar="RESULTS_JSON",
        help="Skip evaluation, just generate figures from existing results",
    )

    parser.add_argument(
        "--models", nargs="+", type=str,
        help="Specific model keys to evaluate (overrides mode selection)",
    )
    parser.add_argument(
        "--tasks", type=int, default=None,
        help="Total number of tasks (auto-split across levels). Default: 150 for smoke, 2500 for full",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of independent trials per model (default: 1)",
    )
    parser.add_argument(
        "--use-v1", action="store_true",
        help="Use the original TaskGenerator (v1, 36 tools) instead of CompositionEngine",
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark."""
    args = parse_args()

    # --- Figures-only mode ---
    if args.figures_only:
        from comptoolbench.analysis.figures import generate_all_figures
        from comptoolbench.analysis.loader import load_results
        from comptoolbench.analysis.style import apply_style

        apply_style()
        results = load_results(args.figures_only)
        fig_dir = Path(args.figures_only).parent / "figures"
        outputs = generate_all_figures(results, output_dir=fig_dir)
        logger.info("Generated %d figures in %s", len(outputs), fig_dir)
        return

    # --- Determine output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"run_{timestamp}"

    # --- Determine task counts ---
    if args.tasks:
        total = args.tasks
        # Split proportionally: 24% L0, 32% L1, 20% L2, 24% L3
        l0_count = int(total * 0.24)
        l1_count = int(total * 0.32)
        l2_count = int(total * 0.20)
        l3_count = total - l0_count - l1_count - l2_count
    elif args.smoke:
        l0_count, l1_count, l2_count, l3_count = 5, 5, 5, 5
    else:
        # Default: full 2,500 task suite
        l0_count, l1_count, l2_count, l3_count = 600, 800, 500, 600

    # --- Determine models ---
    if args.models:
        model_keys = args.models
    elif args.full:
        model_keys = ALL_MODELS
    elif args.cloud_only:
        model_keys = CLOUD_MODELS
    elif args.local_only:
        model_keys = LOCAL_MODELS
    elif args.smoke:
        model_keys = [LOCAL_MODELS[0]] if LOCAL_MODELS else CLOUD_MODELS[:1]
    else:
        model_keys = LOCAL_MODELS

    # Filter to models with available API keys
    model_keys = get_available_models(model_keys)

    if not model_keys:
        logger.error("No valid models to evaluate. Check API keys. Exiting.")
        sys.exit(1)

    # --- Create runner and generate/load tasks ---
    runner = BenchmarkRunner(output_dir=output_dir)

    total_tasks = l0_count + l1_count + l2_count + l3_count
    logger.info("=" * 70)
    logger.info("  CompToolBench v2.0 — Compositional Tool-Use Benchmark")
    logger.info("=" * 70)
    logger.info("  Output:    %s", output_dir)
    logger.info("  Seed:      %d", args.seed)
    logger.info("  Tasks:     L0=%d L1=%d L2=%d L3=%d (total=%d)",
                l0_count, l1_count, l2_count, l3_count, total_tasks)
    logger.info("  Models:    %d (%s)", len(model_keys), ", ".join(model_keys))
    logger.info("  Trials:    %d", args.trials)
    logger.info("  Engine:    %s", "v1 (TaskGenerator)" if args.use_v1 else "v2 (CompositionEngine)")
    logger.info("=" * 70)

    # Reuse existing task suite if resuming
    existing_suite = output_dir / "task_suite.json"
    if existing_suite.exists():
        suite = runner.load_tasks(existing_suite)
        logger.info("Loaded existing %d tasks from %s", len(suite.tasks), existing_suite)
    elif args.use_v1:
        suite = runner.generate_tasks(
            seed=args.seed,
            l0_count=l0_count, l1_count=l1_count,
            l2_count=l2_count, l3_count=l3_count,
        )
    else:
        suite = runner.generate_tasks_v2(
            seed=args.seed,
            l0_count=l0_count, l1_count=l1_count,
            l2_count=l2_count, l3_count=l3_count,
        )
    logger.info("Task suite ready: %d tasks, %d tools", len(suite.tasks), suite.stats.get("unique_tools", 0))

    # --- Evaluate each model (with trials) ---
    for trial in range(1, args.trials + 1):
        if args.trials > 1:
            logger.info("\n>>> TRIAL %d/%d <<<", trial, args.trials)

        for i, model_key in enumerate(model_keys, start=1):
            config = AVAILABLE_MODELS[model_key]
            logger.info("\n" + "=" * 70)
            logger.info(
                "[%d/%d] Evaluating: %s (%s)",
                i, len(model_keys), config.name, config.provider.value,
            )
            logger.info("=" * 70)

            try:
                # For multiple trials, use different model key suffixes
                eval_key = model_key if args.trials == 1 else f"{model_key}_t{trial}"
                result = runner.evaluate_model(eval_key, config=config, resume=True)
                gap = result.composition_gap
                if gap:
                    logger.info(
                        "[%s] Results: Overall=%.1f%%, Gap=%.1f%%, "
                        "L0=%.1f%% L1=%.1f%% L2=%.1f%% L3=%.1f%%",
                        config.name,
                        gap.overall_accuracy * 100,
                        gap.gap_overall * 100,
                        gap.accuracy_l0 * 100,
                        gap.accuracy_l1 * 100,
                        gap.accuracy_l2 * 100,
                        gap.accuracy_l3 * 100,
                    )
            except Exception:
                logger.exception("[%s] Evaluation failed", config.name)
                continue

    # --- Save results ---
    results_path = runner.save_results()
    leaderboard_path = runner.save_leaderboard()
    logger.info("Results saved to: %s", results_path)
    logger.info("Leaderboard: %s", leaderboard_path)

    # --- Generate figures ---
    try:
        from comptoolbench.analysis.figures import generate_all_figures
        from comptoolbench.analysis.loader import load_results
        from comptoolbench.analysis.style import apply_style

        apply_style()
        results_data = load_results(results_path)
        fig_dir = output_dir / "figures"
        outputs = generate_all_figures(results_data, output_dir=fig_dir)
        logger.info("Generated %d figures in %s", len(outputs), fig_dir)
    except Exception:
        logger.warning("Figure generation failed (non-critical)")

    logger.info("\n" + "=" * 70)
    logger.info("  DONE! Results in: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
