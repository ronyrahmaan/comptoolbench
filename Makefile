.PHONY: test lint format check smoke bench-local bench-cloud figures demo paper arxiv clean

# ── Development ──────────────────────────────────────────────────────
test:
	uv run pytest tests/ -x -q

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

check: lint test

# ── Benchmarking ─────────────────────────────────────────────────────
smoke:
	uv run python scripts/run_benchmark.py --smoke

bench-local:
	uv run python scripts/run_benchmark.py --local-only

bench-cloud:
	uv run python scripts/run_benchmark.py --cloud-only --tasks 500

# ── Analysis & Visualization ─────────────────────────────────────────
figures:
	uv run python scripts/run_benchmark.py --figures-only results/run_*/results_*.json

demo:
	uv run python demo/app.py

# ── Paper ────────────────────────────────────────────────────────────
paper:
	tectonic paper/main.tex

arxiv:
	bash scripts/bundle_arxiv.sh

# ── Cleanup ──────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .cache htmlcov .coverage
