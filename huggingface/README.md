---
dataset_info:
  features:
    - name: task_id
      dtype: string
    - name: level
      dtype: string
    - name: prompt
      dtype: string
    - name: available_tools
      sequence: string
    - name: expected_trace
      dtype: string
    - name: expected_final_answer
      dtype: string
    - name: num_steps
      dtype: int32
    - name: num_tools_offered
      dtype: int32
    - name: category
      dtype: string
    - name: pattern
      dtype: string
  splits:
    - name: test
      num_examples: 200
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - tool-use
  - function-calling
  - benchmark
  - compositional
  - llm-evaluation
  - agents
  - dag
  - composition-gap
pretty_name: CompToolBench
size_categories:
  - n<1K
---

# CompToolBench: Measuring Compositional Tool-Use Generalization in LLMs

## Dataset Summary

**CompToolBench** is a benchmark for evaluating how well large language models generalize from simple, single-tool calls to complex, multi-step compositional tool use. It contains **200 tasks** spanning four composition levels of increasing structural complexity, built on top of **106 deterministic tool simulators** covering 9 functional categories.

The key insight behind CompToolBench is the **composition gap**: models that can reliably call individual tools often fail dramatically when those same tools must be composed into chains, parallel fan-outs, or directed acyclic graphs (DAGs). CompToolBench quantifies this gap with fine-grained diagnostic metrics.

### Key Features

- **4 composition levels**: single calls (L0), sequential chains (L1), parallel fan-outs (L2), and full DAGs with branching + merging (L3)
- **106 deterministic tool simulators**: no external API dependencies, fully reproducible
- **Fine-grained scoring**: tool selection accuracy, argument accuracy, data-flow correctness, completion rate
- **18-model leaderboard**: spanning cloud APIs (Mistral, Cohere, Groq, Cerebras, OpenRouter) and local models (Ollama)
- **Composition gap metric**: directly measures how much accuracy degrades as structural complexity increases

## Dataset Structure

Each example in the dataset contains the following fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | Unique identifier (e.g., `L0_node_0001`, `L3_dag_0153`) |
| `level` | `string` | Composition level: `L0_node`, `L1_chain`, `L2_parallel`, or `L3_dag` |
| `prompt` | `string` | Natural language instruction given to the model |
| `available_tools` | `list[string]` | Tool names provided to the model (includes distractors) |
| `expected_trace` | `object` | Ground-truth execution plan with steps, dependencies, and arguments |
| `expected_final_answer` | `string` | JSON-serialized expected output |
| `num_steps` | `int` | Number of tool calls in the expected trace |
| `num_tools_offered` | `int` | Number of tools offered (correct + distractors) |
| `category` | `string` | Functional category of the task |
| `pattern` | `string` | Composition pattern (e.g., `retrieve-transform`, `fan-out-compare`) |

### Expected Trace Structure

Each step in `expected_trace.steps` contains:

- `step_id`: Step identifier (e.g., `step_1`)
- `tool_name`: Which tool to call
- `arguments`: JSON-serialized expected arguments
- `depends_on`: List of step IDs this step depends on (defines the DAG structure)
- `output_key`: Variable name for the step's output (used by downstream steps)

## Composition Levels

| Level | Name | Description | Tasks | Avg Steps | Avg Tools Offered |
|---|---|---|---|---|---|
| **L0** | Single Node | One tool call, no composition | 48 | 1.0 | 4.0 |
| **L1** | Chain | Sequential pipeline (A -> B) | 64 | 2.0 | 5.0 |
| **L2** | Parallel | Independent fan-out (A \|\| B \|\| C) | 40 | 2.8 | 4.2 |
| **L3** | DAG | Full directed acyclic graph with branching and merging | 48 | 4.4 | 6.6 |

### Task Categories

Tasks cover 9 functional categories: `chain`, `communication`, `computation`, `dag`, `external_services`, `information_retrieval`, `parallel`, `text_processing`, and `time_scheduling`.

### Composition Patterns

Over 40 distinct composition patterns are represented, including `retrieve-transform`, `fan-out-compare`, `chain-fanout-merge-chain`, `parallel-merge-chain`, `true-dag-parallel-reads-merge`, and many more. See the paper for full details.

## Leaderboard

Results from evaluating 18 models (10 cloud, 8 local) on all 200 tasks. Models are ranked by overall accuracy. All models achieve 100% tool *selection* accuracy (when they issue a call, they name the correct tool).

### Cloud Models

| Model | Provider | L0 | L1 | L2 | L3 | Overall | Delta |
|---|---|---|---|---|---|---|---|
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

### Local Models (Ollama)

| Model | Provider | L0 | L1 | L2 | L3 | Overall | Delta |
|---|---|---|---|---|---|---|---|
| Granite4 3B | Ollama | **45.8** | 57.3 | 56.1 | 30.2 | 47.8 | 15.6 |
| Granite4 1B | Ollama | 41.7 | 56.3 | 55.9 | 29.9 | 46.4 | 11.8 |
| Mistral 7B | Ollama | 43.8 | 57.7 | 49.2 | 30.5 | 46.1 | 13.3 |
| Llama 3.1 8B | Ollama | 39.6 | 56.7 | 56.1 | 29.5 | 45.9 | 10.1 |
| Mistral Nemo 12B | Ollama | 37.5 | 58.4 | 51.0 | 31.8 | 45.5 | 5.7 |
| Qwen 2.5 7B | Ollama | 39.6 | 56.7 | 53.8 | 25.8 | 44.6 | 13.8 |
| Mistral Small 24B | Ollama | 37.5 | 51.1 | 47.7 | 22.6 | 40.3 | 14.9 |
| Qwen3 8B | Ollama | 35.4 | 52.0 | 36.9 | 21.8 | 37.7 | 13.7 |

### Aggregate Statistics

| Segment | L0 | L1 | L2 | L3 | Overall | Delta |
|---|---|---|---|---|---|---|
| *All models avg.* | 40.0 | 58.0 | 67.3 | 34.2 | 49.8 | 5.8 |
| *Cloud avg.* | 40.0 | 59.7 | 80.5 | 39.4 | 54.3 | 0.6 |
| *Local avg.* | 40.1 | 55.8 | 50.8 | 27.8 | 44.3 | 12.3 |

**Delta** = L0 accuracy minus L3 accuracy (positive means degradation at higher composition levels). Models marked with a dagger in the paper exhibit a *Selection Gap*, where L0 accuracy is lower than the average of L1-L3.

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("mdarahmanxAI/comptoolbench", split="test")

# Browse tasks by composition level
l3_tasks = dataset.filter(lambda x: x["level"] == "L3_dag")
print(f"L3 DAG tasks: {len(l3_tasks)}")
print(l3_tasks[0]["prompt"])
```

### Evaluating a Model

CompToolBench evaluates models by comparing their tool-call traces against the expected trace. The evaluation harness is available in the [GitHub repository](https://github.com/ronyrahmaan/comptoolbench).

```python
import json

for task in dataset:
    # 1. Build the tool-use prompt from task["prompt"] and task["available_tools"]
    # 2. Send to your model with the tool schemas
    # 3. Compare the model's tool calls against:
    trace = json.loads(task["expected_trace"])
    answer = json.loads(task["expected_final_answer"])

    # Scoring dimensions:
    #   - Tool selection: did the model call the right tools?
    #   - Argument accuracy: were the arguments correct?
    #   - Data flow: did outputs flow correctly between steps?
    #   - Completion: did all required steps execute?
```

### Scoring Metrics

| Metric | Description |
|---|---|
| **Overall Accuracy** | Weighted combination of all sub-metrics |
| **Tool Selection** | Whether the model called the correct tool names |
| **Argument Accuracy** | Whether arguments matched expected values |
| **Data Flow Accuracy** | Whether inter-step data dependencies were satisfied |
| **Completion Rate** | Fraction of expected steps that were executed |
| **Composition Gap** | L0 accuracy minus Lk accuracy (measures degradation) |

## Citation

If you use CompToolBench in your research, please cite:

```bibtex
@article{rahmaan2026comptoolbench,
  title={CompToolBench: Measuring Compositional Tool-Use Generalization in Large Language Models},
  author={Rahmaan, Rony},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. You are free to share and adapt the dataset for any purpose, provided you give appropriate credit.

## Links

- **Paper**: [arXiv (coming soon)]()
- **Code**: [GitHub](https://github.com/ronyrahmaan/comptoolbench)
- **Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/mdarahmanxAI/comptoolbench-demo)
