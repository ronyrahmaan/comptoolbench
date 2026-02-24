#!/bin/bash
# Watch for Wave 1 benchmark (PID 39747) to finish, then launch Wave 2
cd /Users/rono/Garage/phd/projects/comptoolbench

echo "[$(date)] Waiting for Wave 1 (PID 39747) to finish..."

while kill -0 39747 2>/dev/null; do
    LINES=$(wc -l < results/run_20260311_000147/checkpoint_claude-sonnet-4.jsonl 2>/dev/null || echo 0)
    echo "[$(date)] Claude Sonnet 4: ${LINES}/200 tasks"
    sleep 60
done

echo "[$(date)] Wave 1 DONE. Launching Wave 2..."

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY env var}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY env var}"

uv run python scripts/run_benchmark.py --models gpt-5.4 gemini-2.5-flash claude-haiku-4.5 --tasks 200

echo "[$(date)] Wave 2 COMPLETE."
