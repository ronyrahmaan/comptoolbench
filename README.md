# CompToolBench

**Measuring the Compositional Tool-Use Gap in Large Language Models**

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/coming-soon)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/mdarahmanxAI/comptoolbench)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace_Spaces-orange)](https://huggingface.co/spaces/mdarahmanxAI/comptoolbench-demo)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-305%20passing-brightgreen.svg)]()

> **The Selection Gap:** 17 of 18 models score *higher* on composed multi-tool tasks than on isolated single-tool selection — the *opposite* of what benchmarks assume.
> CompToolBench is the first controlled benchmark to reveal that natural-language tool selection from a 106-tool catalog is systematically harder than multi-step composition.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **Selection Gap is near-universal** | 17/18 models score higher on composed tasks than single-tool selection (13.2pp avg gap) |
| **Cloud-local divergence** | Cloud models dominate L2 (80.5%), local models do not (50.8%) — same model, 31pp spread |
| **Parallel is easier than sequential** | L2 (67.3%) > L1 (58.0%) overall, driven by cloud models (80.5% vs 59.7%) |
| **Traditional composition gap nearly vanishes** | L0→L3 delta is only 5.8pp overall (0.6pp for cloud), down from 26.8pp at smaller scale |
| **Infrastructure matters** | Llama 3.1 8B: 66.4% (Groq) vs 56.0% (Cerebras) vs 45.9% (Ollama) — same weights |

## Leaderboard (18 Models, 200 Tasks, 106 Tools)

| Model | Provider | L0 | L1 | L2 | L3 | Overall | Delta |
|-------|----------|-----|-----|-----|-----|---------|-------|
| Llama 3.1 8B | Groq | 27.1 | **75.8** | 87.1 | **76.0** | **66.4** | **-48.9** |
| Command A | Cohere | **45.8** | 62.7 | 87.8 | 40.8 | 58.4 | 5.1 |
| Mistral Small | Mistral | **45.8** | 59.7 | 87.6 | 40.9 | 57.5 | 4.9 |
| Command R+ | Cohere | 43.8 | 57.5 | **88.0** | 40.3 | 56.2 | 3.4 |
| Llama 3.1 8B | Cerebras | 31.2 | 66.1 | 81.2 | 46.4 | 56.0 | -15.1 |
| Mistral Large | Mistral | 39.6 | 59.5 | 87.9 | 38.5 | 55.4 | 1.1 |
| Mistral Medium | Mistral | 43.8 | 57.5 | 87.9 | 36.3 | 55.2 | 7.4 |
| Gemini 2.0 Flash | OpenRouter | 39.6 | 52.4 | 85.7 | 39.0 | 52.8 | 0.6 |
| GPT-OSS 120B | Cerebras | **45.8** | 56.3 | 56.1 | 29.0 | 47.2 | 16.8 |
| Llama 4 Scout 17B | Groq | 37.5 | 49.6 | 55.8 | 7.0 | 37.7 | 30.5 |
| | | | | | | | |
| Granite4 3B | Ollama | **45.8** | 57.3 | 56.1 | 30.2 | 47.8 | 15.6 |
| Granite4 1B | Ollama | 41.7 | 56.3 | 55.9 | 29.9 | 46.4 | 11.8 |
| Mistral 7B | Ollama | 43.8 | 57.7 | 49.2 | 30.5 | 46.1 | 13.3 |
| Llama 3.1 8B | Ollama | 39.6 | 56.7 | 56.1 | 29.5 | 45.9 | 10.1 |
| Mistral Nemo 12B | Ollama | 37.5 | 58.4 | 51.0 | 31.8 | 45.5 | 5.7 |
| Qwen 2.5 7B | Ollama | 39.6 | 56.7 | 53.8 | 25.8 | 44.6 | 13.8 |
| Mistral Small 24B | Ollama | 37.5 | 51.1 | 47.7 | 22.6 | 40.3 | 14.9 |
| Qwen3 8B | Ollama | 35.4 | 52.0 | 36.9 | 21.8 | 37.7 | 13.7 |

Delta = L0 - L3 (positive = degradation). All models achieve 100% tool *selection* accuracy.

## How It Works

CompToolBench evaluates models across **four composition levels**, each representable as a directed acyclic graph (DAG):

```
L0 (Single)       L1 (Chain)         L2 (Parallel)       L3 (DAG)

  [A]              [A] -> [B]         [A]   [B]          [A]   [B]
                                        \   /              |     |
                                         [C]              [C]   [D]
                                                            \   /
                                                             [E]
```

- **L0**: One tool call (baseline). *"What's the weather in Paris?"*
- **L1**: Sequential chain (A -> B). *"Look up Paris coordinates, then get weather for those coordinates."*
- **L2**: Parallel fork-join (A | B -> C). *"Get weather for Tokyo AND London, then compare them."*
- **L3**: DAG (branching + merging). *"Get weather for 2 cities, convert currencies for each, then compose a summary email."*

The **Selection Gap** = avg(L1, L2, L3) accuracy - L0 accuracy. A positive value means composed tasks are *easier* than single-tool selection — the opposite of conventional wisdom.

## Quick Start

### Installation

```bash
git clone https://github.com/ronyrahmaan/comptoolbench.git
cd comptoolbench
uv sync
```

### Run a Smoke Test (Local, Free)

```bash
# Requires Ollama with at least one tool-calling model:
# ollama pull granite4:3b

uv run python scripts/run_benchmark.py --smoke
```

### Run Full Benchmark

```bash
# Local models only (free, ~90 min on M4 Pro):
uv run python scripts/run_benchmark.py --local-only

# Cloud models (requires API keys):
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
uv run python scripts/run_benchmark.py --cloud-only

# Generate paper figures from results:
uv run python scripts/generate_paper_v3.py
```

## Project Structure

```
comptoolbench/
├── src/comptoolbench/
│   ├── tools/              # 106 deterministic tool simulations
│   ├── generators/         # Task generation engine (CompositionEngine)
│   ├── evaluation/         # Scoring, metrics, model adapter (LiteLLM)
│   ├── analysis/           # Publication-quality analysis & figures
│   └── tasks/models.py     # Task, Trace, TaskSuite data models
├── tests/                  # 305 tests
├── paper/                  # NeurIPS-format LaTeX paper
├── scripts/                # Entry point scripts
├── demo/                   # Gradio interactive demo
└── pyproject.toml
```

## Scoring System

| Dimension | Description | L0 | L1 | L2 | L3 |
|-----------|-------------|-----|-----|-----|-----|
| **Tool Sequence** | Correct tools in correct order | Binary | 0.40 | 0.35 | 0.30 |
| **Arguments** | Correct parameter values | Binary | 0.35 | 0.35 | 0.30 |
| **Data Flow** | Outputs correctly feed into inputs | --- | --- | 0.15 | 0.25 |
| **Completeness** | All required calls made | Binary | 0.25 | 0.15 | 0.15 |

L0 uses strict binary scoring (pass/fail). L1-L3 use weighted partial credit.

## Reproducibility

All results are deterministic with seed 42. Tools use a simulated mode that produces deterministic outputs from a hash of inputs — no live API calls needed.

```python
from comptoolbench.generators.composition_engine import CompositionEngine

engine = CompositionEngine(seed=42)
suite = engine.generate_suite(l0_count=48, l1_count=64, l2_count=40, l3_count=48)
# Same seed = identical 200-task suite, every time
```

## Citation

```bibtex
@article{rahman2026comptoolbench,
  title={CompToolBench: Measuring the Compositional Tool-Use Gap in Large Language Models},
  author={Rahman, Md A},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
