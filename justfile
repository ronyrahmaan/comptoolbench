# CompToolBench — Task Runner
# Run `just` to see all available commands

default:
    @just --list

# Install all dependencies
install:
    uv sync --all-extras

# Run linting
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

# Format code
format:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/

# Type check
typecheck:
    uv run pyright src/

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run tests with coverage
test-cov:
    uv run pytest --cov=src/comptoolbench --cov-report=term-missing --cov-report=html

# Run all quality checks
check: lint typecheck test

# Generate benchmark tasks
generate LEVEL="all":
    uv run python scripts/generate_tasks.py --level {{LEVEL}}

# Run evaluation on a model
eval MODEL="gpt-4o":
    uv run python scripts/evaluate.py --model {{MODEL}}

# Run evaluation on all models
eval-all:
    uv run python scripts/evaluate.py --all

# Generate figures for paper
figures:
    uv run python scripts/generate_figures.py

# Launch Jupyter notebook
notebook:
    uv run jupyter lab notebooks/

# Show benchmark statistics
stats:
    uv run python scripts/show_stats.py
