#!/bin/bash
cd /Users/rono/Garage/phd/projects/comptoolbench

MODELS=("mistral-nemo-12b" "llama3.1-8b" "qwen2.5-7b" "mistral-7b" "granite4-3b")
OUTPUT="results/local_run_001"

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Starting: $model at $(date)"
    echo "=========================================="
    uv run python scripts/run_benchmark.py --models "$model" --output-dir "$OUTPUT" 2>&1 | tee -a "$OUTPUT/benchmark_all.log"
    echo ""
    echo "Finished: $model at $(date)"
    echo ""
done

echo "ALL MODELS COMPLETE at $(date)"
