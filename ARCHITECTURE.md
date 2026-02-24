# CompToolBench: Complete Architecture Specification

> **Version:** 0.1.0
> **Date:** 2026-02-21
> **Status:** Design Phase (Pre-Implementation)

---

## Table of Contents

1. [Core Concept](#1-core-concept)
2. [Evaluation Protocol](#2-evaluation-protocol)
3. [Task Generation Pipeline](#3-task-generation-pipeline)
4. [Tool Implementation](#4-tool-implementation)
5. [Scoring System](#5-scoring-system)
6. [File Structure](#6-file-structure)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Appendix: Data Schemas](#appendix-a-data-schemas)

---

## 1. Core Concept

CompToolBench is a benchmark that measures **compositional tool-use generalization** in large language models. The central hypothesis is that an LLM's ability to use tools A, B, and C individually does not predict its ability to compose them into novel pipelines like A(B(C(x))). The benchmark operationalizes this by (a) confirming baseline competence on each tool in isolation via **Node-level tasks**, then (b) testing the same tools in **Chain**, **Parallel**, and **DAG** compositions that the model has never seen during any prior evaluation in the same session. All tasks are programmatically generated from parameterized templates with deterministic simulated tool backends, producing verifiable ground-truth execution traces. The primary metric is the **Composition Gap** -- the delta between a model's Node-level accuracy and its accuracy on composed tasks involving the same tools -- which isolates the failure mode unique to composition rather than conflating it with individual tool confusion. By controlling the train/eval split at the level of *compositions* (not tools), CompToolBench provides the first systematic measurement of whether tool-use competence transfers to novel multi-tool scenarios.

---

## 2. Evaluation Protocol

### 2.1 Composition Levels

Every task in CompToolBench belongs to exactly one of four composition levels, defined by the topology of its **Tool Execution Graph (TEG)** -- a directed acyclic graph where nodes are tool invocations and edges are data dependencies.

| Level | Topology | # Tools | Description | Example |
|-------|----------|---------|-------------|---------|
| **L0: Node** | Single node | 1 | One tool, one invocation, correct args | `get_weather("London", "2026-03-01")` |
| **L1: Chain** | Linear path | 2-4 | Sequential: output of tool_i feeds tool_{i+1} | `web_search -> summarize_text -> send_email` |
| **L2: Parallel** | Fork-join | 2-4 | Independent calls whose results merge into a downstream tool | `[get_weather("London"), get_weather("Paris")] -> data_sort(results, "temp")` |
| **L3: DAG** | General DAG | 3-6 | Branching, merging, and sequential edges combined | `web_search -> [extract_entities, summarize_text] -> merge_data -> write_file` |

**Invariant:** Every tool that appears in an L1/L2/L3 task also appears in at least one L0 task. This is what enables the composition gap measurement.

### 2.2 The Composition Split

The key methodological innovation. We split on **composition patterns**, not tools:

```
Tool Universe: {T1, T2, ..., T36}  (all 36 tools)

L0 Tasks: Every tool gets 5-8 Node-level tasks with varying parameters
          Total: ~200 L0 tasks
          Purpose: Establish per-tool baseline accuracy

L1-L3 Tasks: Generated from composition templates
          The MODEL sees all 36 tool schemas at all times
          The TASKS present novel compositions the model hasn't practiced
```

There is no "training set" in the traditional sense -- CompToolBench is purely an evaluation benchmark. The split means: we first run L0 to get per-tool accuracy, then run L1-L3 to get per-composition accuracy, then compute the gap. The model sees the same tool schemas throughout; what changes is the complexity of the requested composition.

### 2.3 The Composition Gap Metric

The headline metric of the benchmark. Formally:

```
For a model M and a composition task C involving tools {T_i, T_j, T_k}:

  individual_accuracy(M, C) = min(L0_accuracy(M, T_i), L0_accuracy(M, T_j), L0_accuracy(M, T_k))
  composed_accuracy(M, C)   = L_x_accuracy(M, C)   where x in {1, 2, 3}

  composition_gap(M, C)     = individual_accuracy(M, C) - composed_accuracy(M, C)
```

Aggregated across all compositions at each level:

```
  CompositionGap_Lx(M) = mean(individual_accuracy(M, C) - composed_accuracy(M, C))
                          for all C at level Lx
```

**Interpretation:**
- `CompositionGap = 0`: Model composes perfectly (no degradation from individual to composed)
- `CompositionGap > 0`: Model loses accuracy when composing (the gap we want to measure)
- `CompositionGap < 0`: Model is *better* at composition than individual (unlikely but possible with some compositions being easier than individual parameterizations)

We also report:
- **CompositionGap_L1**, **CompositionGap_L2**, **CompositionGap_L3**: Per-level gaps
- **CompositionGap_overall**: Weighted average across levels
- **GapProfile**: How the gap changes as composition complexity (number of tools, depth of DAG) increases

### 2.4 Evaluation Modes

CompToolBench supports two evaluation modes:

**Mode 1: Single-Turn Function Calling (Primary)**
The model receives a natural language prompt and the full tool schema set. It must return a structured function call (or sequence of calls). We compare the returned call sequence against the ground-truth execution trace. This is the primary evaluation mode and what we report in the paper.

**Mode 2: Multi-Turn Agentic (Extended)**
The model interacts in a loop: it makes a tool call, receives the simulated result, then decides the next call. This tests the model's ability to handle intermediate results and adapt. This is a secondary mode for deeper analysis.

Both modes use the Inspect AI framework for orchestration.

### 2.5 Per-Task Evaluation Structure

Each task is evaluated as follows:

```
Input to model:
  - System prompt (fixed across all tasks)
  - Tool schemas (all 36 tools, OpenAI function calling format)
  - User prompt (natural language task description)

Expected output (ground truth):
  - Ordered list of tool calls with exact arguments
  - Expected return value for each call (from simulated backend)
  - Final answer (if the task asks for one)

Evaluation:
  - Compare model output against ground truth using scoring rubric
  - Record per-tool, per-level, and per-composition metrics
```

---

## 3. Task Generation Pipeline

### 3.1 Overview

Tasks are generated programmatically from **Composition Templates** -- parameterized blueprints that define the topology, the tool slots, and the argument generation strategy. This design achieves three goals: (1) verifiable ground truth (every task has a deterministic expected trace), (2) contamination resistance (parameters are randomized, tasks can be regenerated with new seeds), and (3) scalability (generating 1000 tasks is as easy as generating 100).

### 3.2 Architecture

```
                    ┌─────────────────────┐
                    │  Template Registry  │
                    │  (YAML definitions) │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Composition Engine │
                    │  - Selects template │
                    │  - Fills tool slots │
                    │  - Generates params │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │   Trace Executor    │
                    │  - Runs simulated   │
                    │    tools in order   │
                    │  - Records trace    │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Prompt Generator   │
                    │  - NL description   │
                    │  - from template +  │
                    │    concrete params  │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │    Task Instance    │
                    │  {prompt, schemas,  │
                    │   ground_truth,     │
                    │   metadata}         │
                    └─────────────────────┘
```

### 3.3 Composition Templates

A **Composition Template** is a YAML file that defines:

```yaml
# templates/chain/search_summarize_email.yaml
template_id: "chain_search_summarize_email"
level: "L1"  # Chain
topology: "chain"
description: "Search the web, summarize results, and email the summary"

# Tool slots in execution order
tool_graph:
  - step: 1
    tool: "web_search"
    args_template:
      query: "{{search_query}}"
      num_results: 3
    output_binding: "search_results"

  - step: 2
    tool: "summarize_text"
    args_template:
      text: "{{search_results.content}}"
      max_length: 200
      style: "professional"
    output_binding: "summary"
    depends_on: [1]

  - step: 3
    tool: "send_email"
    args_template:
      to: "{{recipient_email}}"
      subject: "Summary: {{search_query}}"
      body: "{{summary.summary}}"
    depends_on: [2]

# Parameter generation
parameters:
  search_query:
    type: "sampled"
    source: "data/search_queries.json"
    # OR: type: "generated", pattern: "latest {topic} research"
  recipient_email:
    type: "sampled"
    source: "data/email_addresses.json"

# Natural language prompt template (multiple variants for diversity)
prompt_templates:
  - "Search the web for '{{search_query}}', summarize the results in a professional tone (max 200 words), and email the summary to {{recipient_email}}."
  - "I need a brief professional summary of web results for '{{search_query}}'. Please search, summarize, and send it to {{recipient_email}}."
  - "Find information about '{{search_query}}' online, create a concise professional summary, and email it to {{recipient_email}}."

# Metadata for analysis
tags: ["cross-category", "information-retrieval", "text-processing", "communication"]
cross_category: true
difficulty: "medium"
```

**DAG Template Example:**

```yaml
# templates/dag/travel_research.yaml
template_id: "dag_travel_research"
level: "L3"  # DAG
topology: "dag"
description: "Research a travel destination with weather, directions, and costs in parallel"

tool_graph:
  - step: 1
    tool: "get_location_info"
    args_template:
      query: "{{destination}}"
    output_binding: "location"

  - step: 2
    tool: "get_weather"
    args_template:
      location: "{{location.address}}"
      date: "{{travel_date}}"
    output_binding: "weather"
    depends_on: [1]

  - step: 3
    tool: "get_directions"
    args_template:
      origin: "{{origin}}"
      destination: "{{location.address}}"
      mode: "transit"
    output_binding: "directions"
    depends_on: [1]

  - step: 4
    tool: "search_products"
    args_template:
      query: "{{destination}} tickets"
      category: "tourism"
      max_price: "{{budget}}"
    output_binding: "products"
    depends_on: [1]

  - step: 5
    tool: "summarize_text"
    args_template:
      text: "Weather: {{weather}}\nDirections: {{directions}}\nProducts: {{products}}"
      max_length: 300
      style: "casual"
    output_binding: "summary"
    depends_on: [2, 3, 4]  # Fan-in: merges all parallel results

  - step: 6
    tool: "send_email"
    args_template:
      to: "{{recipient}}"
      subject: "Travel info for {{destination}}"
      body: "{{summary.summary}}"
    depends_on: [5]

parameters:
  destination:
    type: "sampled"
    source: "data/destinations.json"
  travel_date:
    type: "generated"
    pattern: "2026-{month:03-06}-{day:01-28}"
  origin:
    type: "sampled"
    source: "data/cities.json"
  budget:
    type: "uniform_float"
    min: 50.0
    max: 500.0
  recipient:
    type: "sampled"
    source: "data/email_addresses.json"

prompt_templates:
  - "I'm planning a trip to {{destination}} on {{travel_date}}. I'm leaving from {{origin}} with a budget of ${{budget}}. Can you check the weather, find transit directions, search for tickets under budget, then summarize everything and email it to {{recipient}}?"
```

### 3.4 Parameter Pools

Each parameter has a data pool (JSON file) that provides realistic values:

```json
// data/param_pools/search_queries.json
{
  "values": [
    "renewable energy trends 2026",
    "machine learning in healthcare",
    "best practices for remote work",
    "quantum computing applications",
    "sustainable agriculture technology",
    // ... 200+ entries
  ],
  "metadata": {
    "source": "manually curated + GPT-4 generated",
    "last_updated": "2026-02-21",
    "contamination_note": "Regenerate with new seed before public release"
  }
}
```

### 3.5 Trace Executor

The Trace Executor runs the simulated tools in topological order to produce the ground-truth execution trace:

```python
@dataclass
class ToolCall:
    """A single tool invocation in the ground-truth trace."""
    step: int
    tool_name: str
    arguments: dict[str, Any]
    expected_output: Any
    depends_on: list[int]

@dataclass
class ExecutionTrace:
    """Complete ground truth for a task."""
    task_id: str
    level: str  # L0, L1, L2, L3
    topology: str  # node, chain, parallel, dag
    tool_calls: list[ToolCall]
    final_answer: str | None
    tools_involved: list[str]
    template_id: str
    seed: int
```

The executor:
1. Topologically sorts the tool graph
2. Executes each tool with its simulated backend, passing outputs forward via bindings
3. Records every call and its output
4. Produces the `ExecutionTrace` as the ground truth

### 3.6 Prompt Generator

Converts a filled template into natural language. Multiple strategies:

1. **Direct template**: Fill `{{placeholders}}` in the prompt_templates from the YAML
2. **Paraphrased variant**: Use an LLM to rephrase the direct template (for diversity), but cache the rephrasings so they are deterministic across benchmark runs
3. **Implicit variant**: Remove explicit tool names from the prompt so the model must infer which tools to use (harder)

For v1 of the benchmark, we use strategy 1 (direct templates) for reproducibility. Strategies 2 and 3 are planned for v2.

### 3.7 Contamination Resistance

Three mechanisms:

1. **Seeded randomness**: Every task is generated with a seed. Changing the seed produces different parameter values but the same composition structure. For public releases, we generate with a new seed.
2. **Parameter pool rotation**: The parameter pools (search queries, city names, email addresses) can be regenerated. The composition templates are the stable part; the concrete values are ephemeral.
3. **Held-out test set**: We generate 2x the needed tasks and randomly hold out 50%. The held-out set is never published; only released to reproduce results via a seed.

---

## 4. Tool Implementation

### 4.1 Design Decision: Simulated Tools

All tools are **simulated with deterministic outputs**. This is a deliberate design choice:

| Consideration | Real APIs | Simulated Tools | **Our Choice** |
|--------------|-----------|-----------------|----------------|
| Reproducibility | Non-deterministic | Fully deterministic | **Simulated** |
| Cost | API costs per eval | Free | **Simulated** |
| Rate limits | Yes | No | **Simulated** |
| Realism | High | Medium-high | **Simulated** (acceptable trade-off) |
| Ground truth | Hard to verify | Trivially verifiable | **Simulated** |
| Offline usage | No | Yes | **Simulated** |

Simulated tools use a **seed-based deterministic response generator** that produces realistic-looking outputs. The outputs are consistent: calling `get_weather("London", "2026-03-01")` with the same seed always returns the same result.

### 4.2 Tool Schema Format

We use the **OpenAI function calling format** as the canonical schema, since it is the de facto standard and supported by Inspect AI, LiteLLM, and every major model provider:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather and forecast for a location on a specific date.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name or address (e.g., 'London, UK')"
        },
        "date": {
          "type": "string",
          "description": "Date in YYYY-MM-DD format"
        }
      },
      "required": ["location", "date"]
    }
  }
}
```

### 4.3 Tool Implementation Architecture

Each simulated tool is a Python class that inherits from `SimulatedTool`:

```python
class SimulatedTool(ABC):
    """Base class for all simulated tools in CompToolBench."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name matching the function schema."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Tool category (e.g., 'information_retrieval')."""
        ...

    @property
    @abstractmethod
    def schema(self) -> dict:
        """OpenAI function calling schema for this tool."""
        ...

    @property
    @abstractmethod
    def input_types(self) -> dict[str, type]:
        """Map of parameter names to their Python types."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> type:
        """The Python type of the tool's return value."""
        ...

    @abstractmethod
    def execute(self, seed: int, **kwargs) -> Any:
        """Execute the tool with deterministic output based on seed."""
        ...

    def validate_args(self, **kwargs) -> tuple[bool, str]:
        """Validate that arguments match the schema."""
        ...
```

### 4.4 Simulated Output Strategy

Each tool category has a domain-specific output generator:

```python
class GetWeatherTool(SimulatedTool):
    """Simulated weather API."""

    name = "get_weather"
    category = "external_services"

    def execute(self, seed: int, location: str, date: str) -> dict:
        rng = random.Random(hash((seed, location, date)))
        return {
            "location": location,
            "date": date,
            "temperature_celsius": rng.randint(-10, 40),
            "humidity_percent": rng.randint(20, 95),
            "conditions": rng.choice([
                "sunny", "partly cloudy", "cloudy", "rainy",
                "thunderstorms", "snowy", "foggy", "windy"
            ]),
            "wind_speed_kmh": rng.randint(0, 80),
            "forecast_summary": f"Expected {rng.choice(['fair', 'mixed', 'poor'])} weather in {location} on {date}."
        }
```

**Critical property:** The seed is derived from `hash((global_seed, *args))`, so the same tool + args + seed always produces the same output, but different args produce different outputs. This enables deterministic execution traces while keeping outputs varied.

### 4.5 Tool Registry

All tools are registered in a central `ToolRegistry`:

```python
class ToolRegistry:
    """Central registry for all simulated tools."""

    def __init__(self) -> None:
        self._tools: dict[str, SimulatedTool] = {}

    def register(self, tool: SimulatedTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> SimulatedTool:
        return self._tools[name]

    def get_all_schemas(self) -> list[dict]:
        """Return OpenAI function calling schemas for all tools."""
        return [tool.schema for tool in self._tools.values()]

    def get_schemas_for(self, tool_names: list[str]) -> list[dict]:
        """Return schemas for a subset of tools."""
        return [self._tools[name].schema for name in tool_names]

    def execute(self, name: str, seed: int, **kwargs) -> Any:
        """Execute a tool by name."""
        return self._tools[name].execute(seed=seed, **kwargs)
```

### 4.6 The 36 Tools (Trimmed from 55)

We trim the original 55-tool taxonomy to **36 tools** for v1. The principles for trimming:

1. **Remove redundancy**: If two tools do nearly the same thing, keep one.
2. **Maximize composition surface**: Keep tools whose outputs naturally feed other tools' inputs.
3. **Cover all 9 categories**: Every category retains at least 2 tools.
4. **Prioritize common real-world patterns**: Keep tools that appear in the most composition templates.

| Category | Kept (36) | Cut (19) | Rationale for Cuts |
|----------|-----------|----------|-------------------|
| **Information Retrieval (5)** | web_search, web_page_fetch, knowledge_base_query, database_query, lookup_entity | get_document, search_news | get_document overlaps with read_file; search_news overlaps with web_search |
| **Computation (5)** | calculator, execute_python, data_filter, data_sort, data_aggregate | statistical_analysis, unit_convert | statistical_analysis subsumable by execute_python; unit_convert is too narrow |
| **Communication (4)** | send_email, send_message, create_notification, schedule_meeting | create_task, post_to_feed | create_task overlaps with create_notification; post_to_feed is niche |
| **File & Data (5)** | read_file, write_file, list_files, transform_format, merge_data | extract_text, generate_chart | extract_text overlaps with web_page_fetch; generate_chart adds multimodal complexity for v2 |
| **External Services (5)** | get_weather, get_location_info, get_directions, translate_text, get_stock_price | get_exchange_rate, search_products, book_reservation | get_exchange_rate overlaps calculator; search_products/book_reservation are stateful complexity for v2 |
| **State Management (4)** | store_memory, retrieve_memory, list_memories, get_session_context | delete_memory, update_user_preferences | delete/update are CRUD completeness but not needed for composition testing |
| **Text Processing (4)** | summarize_text, extract_entities, sentiment_analysis, classify_text | generate_text, compare_texts | generate_text is too open-ended for deterministic eval; compare_texts is narrow |
| **Time & Scheduling (2)** | get_current_time, convert_timezone | calculate_date_diff, set_reminder | calculate_date_diff is subsumable by calculator; set_reminder overlaps with schedule_meeting |
| **Media (2)** | generate_image, transcribe_audio | analyze_image, text_to_speech | Keep one input and one output modality; v2 adds the rest |

**Total: 36 tools across 9 categories.**

---

## 5. Scoring System

### 5.1 Metrics Hierarchy

CompToolBench reports metrics at four levels of granularity:

```
Level 4 (Headline):   CompositionGap_overall, Overall_Accuracy
Level 3 (Per-Level):  Accuracy_L0, Accuracy_L1, Accuracy_L2, Accuracy_L3,
                      CompositionGap_L1, CompositionGap_L2, CompositionGap_L3
Level 2 (Per-Task):   task_score (0.0 - 1.0)
Level 1 (Per-Call):   call_correct (bool), args_correct (bool), output_used_correctly (bool)
```

### 5.2 Per-Call Scoring (Level 1)

Each individual tool call in the model's response is scored on three dimensions:

```python
@dataclass
class CallScore:
    """Score for a single tool call."""
    tool_selected_correctly: bool    # Did the model pick the right tool?
    args_correct: float              # 0.0-1.0, proportion of args that match
    output_routing_correct: bool     # Was the output passed to the correct next tool?
```

**Argument matching rules:**
- **Exact match** for: tool name, enum-type args, boolean args
- **Fuzzy match** for: string args (using normalized Levenshtein distance, threshold 0.85)
- **Numeric tolerance** for: float args (relative tolerance 1e-6)
- **Structural match** for: list/dict args (recursive comparison with above rules)

### 5.3 Per-Task Scoring (Level 2)

A task score combines call-level scores based on the topology:

**For L0 (Node) tasks:**
```
task_score = 1.0 if (tool_correct AND args_correct >= 0.85) else 0.0
```
Binary pass/fail for single-tool tasks.

**For L1 (Chain) tasks:**
```
task_score = weighted_sum(
    tool_sequence_score * 0.40,    # Are the right tools called in the right order?
    argument_score * 0.35,          # Are arguments correct (including data flow)?
    completeness_score * 0.25       # Did the model complete all steps?
)
```

Where:
- `tool_sequence_score` = longest common subsequence of tool names / expected length
- `argument_score` = mean of per-call arg_correct scores
- `completeness_score` = number of expected calls made / total expected calls

**For L2 (Parallel) tasks:**
```
task_score = weighted_sum(
    tool_set_score * 0.35,          # Are the right tools called (order-independent for parallel segment)?
    argument_score * 0.35,
    fan_in_score * 0.15,            # Did the merge step receive all parallel outputs?
    completeness_score * 0.15
)
```

**For L3 (DAG) tasks:**
```
task_score = weighted_sum(
    graph_structure_score * 0.30,   # Does the call graph match the expected DAG?
    argument_score * 0.30,
    data_flow_score * 0.25,         # Are outputs routed correctly through the DAG?
    completeness_score * 0.15
)
```

The `graph_structure_score` compares the model's execution graph against the expected TEG using graph edit distance normalized to [0, 1].

### 5.4 Per-Level Scoring (Level 3)

```python
accuracy_Lx = mean(task_score for task in level_Lx_tasks)
```

### 5.5 Composition Gap (Level 4)

```python
def composition_gap(model_results: ModelResults) -> dict[str, float]:
    """Compute the composition gap at each level."""
    gaps = {}
    for level in ["L1", "L2", "L3"]:
        level_gaps = []
        for task in model_results.tasks_at_level(level):
            # Per-tool L0 accuracy for tools in this composition
            tool_accs = [
                model_results.l0_accuracy(tool)
                for tool in task.tools_involved
            ]
            individual_acc = min(tool_accs)  # Bottleneck: weakest tool
            composed_acc = task.task_score
            level_gaps.append(individual_acc - composed_acc)
        gaps[f"CompositionGap_{level}"] = mean(level_gaps)

    gaps["CompositionGap_overall"] = weighted_mean(
        [gaps["CompositionGap_L1"], gaps["CompositionGap_L2"], gaps["CompositionGap_L3"]],
        weights=[0.30, 0.30, 0.40]  # DAG weighted higher as it's the hardest
    )
    return gaps
```

### 5.6 Additional Diagnostic Metrics

Beyond the headline metrics, we compute diagnostics for deeper analysis:

| Metric | Description |
|--------|-------------|
| **Tool Selection Accuracy** | % of calls where the correct tool was chosen (ignoring args) |
| **Argument Accuracy** | % of arguments that exactly match ground truth |
| **Data Flow Accuracy** | % of inter-tool data dependencies correctly satisfied |
| **Completion Rate** | % of tasks where the model attempted all expected calls |
| **Hallucinated Tool Rate** | % of calls to tools not in the schema |
| **Early Termination Rate** | % of chains/DAGs where the model stopped before the final step |
| **Cross-Category Composition Gap** | Gap specifically for compositions spanning multiple tool categories |
| **Within-Category Composition Gap** | Gap for compositions within a single category |

---

## 6. File Structure

### 6.1 Complete Project Layout

```
projects/comptoolbench/
├── pyproject.toml                          # Project metadata, dependencies, entry points
├── Makefile                                # Common tasks (generate, evaluate, report)
├── ARCHITECTURE.md                         # This document
├── README.md                               # Public-facing documentation
├── .python-version                         # Python 3.13
├── .gitignore
│
├── configs/                                # Hydra YAML configurations
│   ├── config.yaml                         # Default top-level config
│   ├── experiment/
│   │   ├── full_benchmark.yaml             # All levels, all tools, all models
│   │   ├── quick_smoke.yaml                # 5 tasks per level, 1 model (for testing)
│   │   ├── node_only.yaml                  # L0 only (baseline establishment)
│   │   └── composition_only.yaml           # L1-L3 only (assumes L0 already run)
│   ├── model/
│   │   ├── gpt4o.yaml                      # GPT-4o via LiteLLM
│   │   ├── claude_sonnet.yaml              # Claude 3.5/4 Sonnet
│   │   ├── claude_opus.yaml                # Claude Opus 4
│   │   ├── gemini_pro.yaml                 # Gemini 2.0 Pro
│   │   ├── llama_70b.yaml                  # Llama 3 70B (via Together/Fireworks)
│   │   ├── qwen_72b.yaml                   # Qwen 2.5 72B
│   │   └── deepseek_v3.yaml                # DeepSeek-V3
│   └── generation/
│       ├── default.yaml                    # Default task generation config
│       └── high_diversity.yaml             # More parameter variation
│
├── src/comptoolbench/
│   ├── __init__.py
│   │
│   ├── tools/                              # Tool implementations (simulated)
│   │   ├── __init__.py
│   │   ├── base.py                         # SimulatedTool ABC, ToolRegistry
│   │   ├── registry.py                     # Global registry, tool discovery
│   │   ├── schemas.py                      # Schema generation utilities
│   │   ├── information_retrieval.py        # web_search, web_page_fetch, knowledge_base_query,
│   │   │                                   # database_query, lookup_entity
│   │   ├── computation.py                  # calculator, execute_python, data_filter,
│   │   │                                   # data_sort, data_aggregate
│   │   ├── communication.py                # send_email, send_message, create_notification,
│   │   │                                   # schedule_meeting
│   │   ├── file_data.py                    # read_file, write_file, list_files,
│   │   │                                   # transform_format, merge_data
│   │   ├── external_services.py            # get_weather, get_location_info, get_directions,
│   │   │                                   # translate_text, get_stock_price
│   │   ├── state_management.py             # store_memory, retrieve_memory, list_memories,
│   │   │                                   # get_session_context
│   │   ├── text_processing.py              # summarize_text, extract_entities,
│   │   │                                   # sentiment_analysis, classify_text
│   │   ├── time_scheduling.py              # get_current_time, convert_timezone
│   │   └── media.py                        # generate_image, transcribe_audio
│   │
│   ├── generators/                         # Task generation pipeline
│   │   ├── __init__.py
│   │   ├── template_loader.py              # Load and validate YAML templates
│   │   ├── param_sampler.py                # Sample parameters from pools
│   │   ├── composition_engine.py           # Instantiate templates into concrete tasks
│   │   ├── trace_executor.py               # Execute simulated tools, produce ground truth
│   │   ├── prompt_generator.py             # Generate natural language prompts
│   │   ├── task_factory.py                 # Orchestrates the full generation pipeline
│   │   └── contamination.py               # Seed management, deduplication, holdout logic
│   │
│   ├── tasks/                              # Task data structures and I/O
│   │   ├── __init__.py
│   │   ├── models.py                       # Pydantic models: Task, ToolCall, ExecutionTrace,
│   │   │                                   # TaskSuite, etc.
│   │   ├── io.py                           # Serialize/deserialize tasks to JSON/JSONL
│   │   └── validators.py                   # Validate task integrity, schema compliance
│   │
│   ├── evaluation/                         # Scoring and evaluation logic
│   │   ├── __init__.py
│   │   ├── scorers.py                      # Per-call, per-task, per-level scoring functions
│   │   ├── metrics.py                      # CompositionGap, diagnostic metrics
│   │   ├── matchers.py                     # Argument matching (exact, fuzzy, numeric)
│   │   ├── graph_compare.py                # TEG comparison (for DAG scoring)
│   │   └── inspect_task.py                 # Inspect AI task definition (the glue)
│   │
│   ├── models/                             # Model interface layer
│   │   ├── __init__.py
│   │   ├── litellm_adapter.py              # LiteLLM wrapper for unified model access
│   │   ├── response_parser.py              # Parse model responses into ToolCall sequences
│   │   └── prompts.py                      # System prompts, few-shot examples (if any)
│   │
│   ├── analysis/                           # Post-evaluation analysis
│   │   ├── __init__.py
│   │   ├── aggregator.py                   # Aggregate results across models/levels
│   │   ├── gap_analysis.py                 # Composition gap breakdowns
│   │   ├── error_taxonomy.py               # Classify error types (wrong tool, wrong args,
│   │   │                                   # wrong order, missing step, hallucinated tool)
│   │   └── visualizations.py               # Matplotlib figures for paper
│   │
│   └── utils/                              # Shared utilities
│       ├── __init__.py
│       ├── config.py                       # Hydra config loading
│       ├── logging.py                      # Structured logging setup
│       ├── hashing.py                      # Deterministic hashing for seeds
│       └── types.py                        # Shared type aliases and enums
│
├── templates/                              # Composition templates (YAML)
│   ├── L0_node/                            # Single-tool templates
│   │   ├── web_search.yaml
│   │   ├── calculator.yaml
│   │   ├── get_weather.yaml
│   │   └── ...                             # One per tool (36 files)
│   ├── L1_chain/                           # Chain templates
│   │   ├── search_summarize_email.yaml
│   │   ├── fetch_translate_summarize.yaml
│   │   ├── read_transform_write.yaml
│   │   ├── stock_calculate_notify.yaml
│   │   ├── query_filter_sort.yaml
│   │   ├── lookup_extract_store.yaml
│   │   └── ...                             # ~25-30 templates
│   ├── L2_parallel/                        # Parallel templates
│   │   ├── multi_city_weather.yaml
│   │   ├── multi_source_search.yaml
│   │   ├── multi_stock_compare.yaml
│   │   ├── parallel_translate.yaml
│   │   └── ...                             # ~15-20 templates
│   └── L3_dag/                             # DAG templates
│       ├── travel_research.yaml
│       ├── research_report.yaml
│       ├── multimodal_meeting.yaml
│       ├── data_pipeline.yaml
│       └── ...                             # ~15-20 templates
│
├── data/                                   # Parameter pools and generated data
│   ├── param_pools/                        # Reusable parameter value pools
│   │   ├── search_queries.json
│   │   ├── cities.json
│   │   ├── destinations.json
│   │   ├── email_addresses.json
│   │   ├── file_paths.json
│   │   ├── stock_symbols.json
│   │   ├── languages.json
│   │   ├── topics.json
│   │   └── ...
│   ├── generated/                          # Generated task suites (gitignored)
│   │   ├── v1_seed42/                      # Task suite with seed 42
│   │   │   ├── L0_tasks.jsonl
│   │   │   ├── L1_tasks.jsonl
│   │   │   ├── L2_tasks.jsonl
│   │   │   ├── L3_tasks.jsonl
│   │   │   └── metadata.json
│   │   └── ...
│   └── reference/                          # Reference data for simulated tools
│       ├── knowledge_base.json             # Simulated KB entries
│       ├── database_tables.json            # Simulated DB tables
│       ├── file_system.json                # Simulated file system
│       └── memory_store.json               # Simulated memory state
│
├── scripts/                                # Entry point scripts
│   ├── generate_tasks.py                   # Generate a task suite
│   ├── run_evaluation.py                   # Run evaluation on a model
│   ├── compute_metrics.py                  # Compute metrics from raw results
│   ├── generate_report.py                  # Generate paper-ready figures and tables
│   └── validate_templates.py              # Validate all templates are well-formed
│
├── tests/                                  # Test suite
│   ├── __init__.py
│   ├── conftest.py                         # Shared fixtures
│   ├── test_tools/
│   │   ├── test_base.py                    # Test SimulatedTool ABC
│   │   ├── test_registry.py                # Test ToolRegistry
│   │   ├── test_information_retrieval.py
│   │   ├── test_computation.py
│   │   ├── test_communication.py
│   │   ├── test_external_services.py
│   │   └── ...                             # One per tool category
│   ├── test_generators/
│   │   ├── test_template_loader.py
│   │   ├── test_composition_engine.py
│   │   ├── test_trace_executor.py
│   │   └── test_prompt_generator.py
│   ├── test_evaluation/
│   │   ├── test_scorers.py
│   │   ├── test_matchers.py
│   │   ├── test_graph_compare.py
│   │   └── test_metrics.py
│   └── test_tasks/
│       ├── test_models.py
│       └── test_validators.py
│
├── notebooks/                              # Analysis notebooks
│   ├── 01_task_distribution.ipynb          # Visualize task distribution across levels
│   ├── 02_composition_gap_analysis.ipynb   # Deep dive into gap patterns
│   ├── 03_error_taxonomy.ipynb             # Error analysis
│   └── 04_paper_figures.ipynb              # Generate all paper figures
│
├── results/                                # Evaluation results (gitignored)
│   └── {model_name}_{timestamp}/
│       ├── raw_responses.jsonl             # Raw model responses
│       ├── scored_results.jsonl            # Per-task scores
│       ├── metrics.json                    # Aggregated metrics
│       └── config.yaml                     # Experiment config snapshot
│
├── paper/                                  # LaTeX paper
│   ├── main.tex
│   ├── references.bib
│   ├── figures/                            # Paper figures (symlinked from figures/)
│   └── tables/                             # Auto-generated LaTeX tables
│
└── figures/                                # Generated figures
    ├── composition_gap_by_model.pdf
    ├── accuracy_by_level.pdf
    ├── gap_vs_complexity.pdf
    ├── error_breakdown.pdf
    └── ...
```

### 6.2 Key Module Responsibilities

| Module | Primary Responsibility | Key Classes/Functions |
|--------|----------------------|----------------------|
| `tools/base.py` | Define the tool abstraction | `SimulatedTool`, `ToolRegistry` |
| `tools/{category}.py` | Implement simulated tools per category | One class per tool |
| `generators/composition_engine.py` | Instantiate templates into tasks | `CompositionEngine.generate()` |
| `generators/trace_executor.py` | Execute tool graphs deterministically | `TraceExecutor.execute()` |
| `tasks/models.py` | Data models for tasks and traces | `Task`, `ToolCall`, `ExecutionTrace` |
| `evaluation/scorers.py` | Score model outputs against ground truth | `score_task()`, `score_call()` |
| `evaluation/metrics.py` | Compute composition gap and diagnostics | `composition_gap()`, `diagnostic_metrics()` |
| `evaluation/inspect_task.py` | Integrate with Inspect AI framework | `comptoolbench_task()` |
| `models/litellm_adapter.py` | Unified model API via LiteLLM | `ModelAdapter.generate()` |
| `analysis/gap_analysis.py` | Post-hoc analysis of composition gaps | `GapAnalyzer` |

---

## 7. Key Design Decisions

### 7.1 Tool Count: 36 (trimmed from 55)

**Rationale:** 55 tools is too many to implement and validate for v1. 36 tools across 9 categories provides sufficient coverage for all four topology types while remaining manageable. The 19 cut tools are not thrown away -- they are earmarked for v2 expansion. The cut prioritizes removing redundancy (e.g., `get_exchange_rate` is `get_stock_price` + `calculator`) and deferring complex features (e.g., `book_reservation` requires multi-turn state).

### 7.2 Task Count Per Level

| Level | Templates | Tasks per Template | Total Tasks | Rationale |
|-------|-----------|-------------------|-------------|-----------|
| **L0: Node** | 36 (one per tool) | 6 | **216** | Need sufficient per-tool sample size for reliable L0 accuracy |
| **L1: Chain** | 25 | 8 | **200** | Most templates, moderate instances for good coverage |
| **L2: Parallel** | 15 | 8 | **120** | Fewer templates (parallel has less structural variety) |
| **L3: DAG** | 15 | 8 | **120** | Fewer templates but complex; 8 instances for statistical power |
| **Total** | **91** | -- | **656** | Manageable size for full evaluation (~$50-100 per model at GPT-4o pricing) |

**Justification for 656 tasks:**
- BFCL has ~2,000 tasks but most are simple function calls
- NESTful has 1,800 but only chain topology
- TaskBench has ~600 across node/chain/DAG
- 656 tasks at our complexity level (with 36 tools presented each time) is substantial
- At ~$0.15 per GPT-4o call, a full evaluation costs roughly $100 per model -- affordable for 7 models (~$700 total)

### 7.3 Models to Evaluate (Priority Order)

**Tier 1 (Must have for paper -- 4 models):**

| Model | Provider | Why |
|-------|----------|-----|
| GPT-4o | OpenAI | The function calling reference model, used by NESTful and BFCL |
| Claude Opus 4 | Anthropic | Strongest on complex reasoning, tool use built into the model |
| Gemini 2.0 Pro | Google | Strong function calling, competitive with GPT-4o |
| Claude Sonnet 4 | Anthropic | Best cost/performance ratio, widely used in production agents |

**Tier 2 (Ideal for paper -- 3 more models):**

| Model | Provider | Why |
|-------|----------|-----|
| Llama 3 70B | Meta (via Together) | Top open-weight model, tests OSS ecosystem |
| DeepSeek-V3 | DeepSeek | Strong reasoning, popular in research community |
| Qwen 2.5 72B | Alibaba (via Together) | Strong multilingual, tests non-Western model |

**Tier 3 (Extended analysis -- if budget permits):**

| Model | Provider | Why |
|-------|----------|-----|
| GPT-4o-mini | OpenAI | Tests scaling law for composition gap |
| Claude Haiku 3.5 | Anthropic | Tests scaling law for composition gap |
| Mistral Large | Mistral | European model, function calling support |

The Tier 1 vs Tier 2 split lets us get a solid paper with 4 models, then strengthen it with 3 more.

### 7.4 Inspect AI Integration

CompToolBench is implemented as an Inspect AI task. The integration point:

```python
# src/comptoolbench/evaluation/inspect_task.py
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import scorer, Score, Target
from inspect_ai.solver import generate, use_tools

from comptoolbench.tools.registry import get_global_registry
from comptoolbench.evaluation.scorers import score_task


@task
def comptoolbench(
    level: str = "all",          # L0, L1, L2, L3, or "all"
    suite_path: str = "data/generated/v1_seed42/",
    tools_presented: str = "all", # "all" (all 36) or "relevant" (only needed tools)
) -> Task:
    """CompToolBench evaluation task for Inspect AI."""
    dataset = load_task_suite(suite_path, level=level)
    registry = get_global_registry()

    return Task(
        dataset=dataset,
        solver=[
            use_tools(registry.as_inspect_tools()),
            generate(),
        ],
        scorer=comptoolbench_scorer(registry),
    )


@scorer(metrics=["accuracy", "composition_gap"])
def comptoolbench_scorer(registry):
    async def score(state, target: Target) -> Score:
        model_calls = parse_tool_calls(state.output.completion)
        expected_trace = ExecutionTrace.from_json(target.text)
        task_score = score_task(model_calls, expected_trace, registry)
        return Score(
            value=task_score.overall,
            answer=json.dumps(model_calls),
            metadata=task_score.to_dict(),
        )
    return score
```

### 7.5 System Prompt

The system prompt is fixed across all tasks to avoid confounding:

```
You are a helpful assistant with access to a set of tools. When the user asks
you to perform a task, you should use the provided tools to accomplish it.

Rules:
1. Use only the tools provided. Do not make up tool names.
2. Pass the correct arguments to each tool as specified in its schema.
3. When a task requires multiple tool calls, you may call them in sequence or
   in parallel as appropriate.
4. When the output of one tool is needed as input to another, use the actual
   output value from the first tool call.
5. If no tool is relevant to the user's request, say so without making a tool call.

Make all necessary tool calls to fully complete the user's request.
```

### 7.6 Why Not Real APIs?

Addressed in Section 4.1. To reiterate the strongest argument: **reproducibility is non-negotiable for a benchmark**. If we call a real weather API today and next month, we get different results, making ground-truth comparison impossible. Every existing benchmark that uses real APIs (ToolBench, SambaNova) struggles with this. NESTful and BFCL's simulated environments are far more reliable. We follow their lead.

### 7.7 Single-Turn vs Multi-Turn

We prioritize **single-turn function calling** (Mode 1) for the paper because:
1. It isolates composition ability from conversation management ability
2. It is cheaper (one API call per task vs N calls)
3. It is easier to score deterministically
4. It aligns with BFCL's methodology

Multi-turn agentic evaluation (Mode 2) is implemented but reported as a secondary analysis. It tests whether models can recover from mid-chain errors, which is interesting but a different research question.

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Deliverable | Priority |
|------|-------------|----------|
| Implement `SimulatedTool` base class and `ToolRegistry` | `tools/base.py`, `tools/registry.py` | P0 |
| Implement 10 core tools (2+ per category minimum) | `tools/{category}.py` | P0 |
| Implement task data models (Pydantic) | `tasks/models.py` | P0 |
| Write tests for all implemented tools | `tests/test_tools/` | P0 |
| Set up Hydra configs | `configs/` | P1 |

**Milestone:** Can instantiate tools, call them with deterministic output, and serialize results.

### Phase 2: Generation (Week 3-4)

| Task | Deliverable | Priority |
|------|-------------|----------|
| Implement template loader and validator | `generators/template_loader.py` | P0 |
| Implement composition engine | `generators/composition_engine.py` | P0 |
| Implement trace executor | `generators/trace_executor.py` | P0 |
| Implement prompt generator | `generators/prompt_generator.py` | P0 |
| Create 10 L0 templates + 5 L1 templates | `templates/` | P0 |
| Create parameter pools | `data/param_pools/` | P1 |
| Implement remaining 26 tools | `tools/{category}.py` | P0 |

**Milestone:** Can generate a complete task suite (L0 + L1) with ground-truth traces.

### Phase 3: Evaluation (Week 5-6)

| Task | Deliverable | Priority |
|------|-------------|----------|
| Implement per-call and per-task scorers | `evaluation/scorers.py`, `evaluation/matchers.py` | P0 |
| Implement composition gap metric | `evaluation/metrics.py` | P0 |
| Implement Inspect AI integration | `evaluation/inspect_task.py` | P0 |
| Implement LiteLLM adapter | `models/litellm_adapter.py` | P0 |
| Create L2 and L3 templates | `templates/L2_parallel/`, `templates/L3_dag/` | P0 |
| Implement graph comparison for DAG scoring | `evaluation/graph_compare.py` | P1 |

**Milestone:** Can run end-to-end evaluation on one model and compute the composition gap.

### Phase 4: Full Evaluation (Week 7-8)

| Task | Deliverable | Priority |
|------|-------------|----------|
| Generate full task suite (656 tasks) | `data/generated/v1_seed42/` | P0 |
| Evaluate Tier 1 models (4 models) | `results/` | P0 |
| Implement analysis and visualization | `analysis/` | P0 |
| Evaluate Tier 2 models (3 models) | `results/` | P1 |
| Generate paper figures and tables | `figures/`, `paper/tables/` | P0 |

**Milestone:** Full benchmark results for 7 models with composition gap analysis.

### Phase 5: Paper (Week 9-10)

| Task | Deliverable | Priority |
|------|-------------|----------|
| Write paper | `paper/main.tex` | P0 |
| Error taxonomy analysis | `analysis/error_taxonomy.py` | P1 |
| Ablation studies (tool count, prompt style) | Additional experiment configs | P1 |
| Public release preparation | README, documentation, seed rotation | P1 |

---

## Appendix A: Data Schemas

### A.1 Task Instance (JSON)

```json
{
  "task_id": "L1_chain_search_summarize_email_042",
  "template_id": "chain_search_summarize_email",
  "level": "L1",
  "topology": "chain",
  "seed": 42,
  "prompt": "Search the web for 'renewable energy trends 2026', summarize the results in a professional tone (max 200 words), and email the summary to alice@example.com.",
  "tools_presented": ["web_search", "web_page_fetch", "knowledge_base_query", "database_query", "lookup_entity", "calculator", "execute_python", "data_filter", "data_sort", "data_aggregate", "send_email", "send_message", "create_notification", "schedule_meeting", "read_file", "write_file", "list_files", "transform_format", "merge_data", "get_weather", "get_location_info", "get_directions", "translate_text", "get_stock_price", "store_memory", "retrieve_memory", "list_memories", "get_session_context", "summarize_text", "extract_entities", "sentiment_analysis", "classify_text", "get_current_time", "convert_timezone", "generate_image", "transcribe_audio"],
  "tools_involved": ["web_search", "summarize_text", "send_email"],
  "ground_truth": {
    "tool_calls": [
      {
        "step": 1,
        "tool_name": "web_search",
        "arguments": {
          "query": "renewable energy trends 2026",
          "num_results": 3
        },
        "expected_output": {
          "results": [
            {"title": "Global Renewable Energy Outlook 2026", "snippet": "Solar and wind capacity expected to double...", "url": "https://example.com/renewable-2026"},
            {"title": "Wind Energy Breakthroughs", "snippet": "New turbine designs promise 40% efficiency gains...", "url": "https://example.com/wind-2026"},
            {"title": "Battery Storage Revolution", "snippet": "Grid-scale storage costs fall below $100/MWh...", "url": "https://example.com/battery-2026"}
          ]
        },
        "depends_on": []
      },
      {
        "step": 2,
        "tool_name": "summarize_text",
        "arguments": {
          "text": "Global Renewable Energy Outlook 2026: Solar and wind capacity expected to double... Wind Energy Breakthroughs: New turbine designs promise 40% efficiency gains... Battery Storage Revolution: Grid-scale storage costs fall below $100/MWh...",
          "max_length": 200,
          "style": "professional"
        },
        "expected_output": {
          "summary": "The renewable energy sector in 2026 shows significant momentum across three key areas. Solar and wind capacity is projected to double, driven by falling costs and supportive policies. Wind energy is experiencing breakthroughs with new turbine designs offering 40% efficiency improvements. Meanwhile, grid-scale battery storage costs have dropped below $100/MWh, a critical threshold for widespread adoption. These developments collectively suggest a transformative year for clean energy infrastructure."
        },
        "depends_on": [1]
      },
      {
        "step": 3,
        "tool_name": "send_email",
        "arguments": {
          "to": "alice@example.com",
          "subject": "Summary: renewable energy trends 2026",
          "body": "The renewable energy sector in 2026 shows significant momentum across three key areas. Solar and wind capacity is projected to double, driven by falling costs and supportive policies. Wind energy is experiencing breakthroughs with new turbine designs offering 40% efficiency improvements. Meanwhile, grid-scale battery storage costs have dropped below $100/MWh, a critical threshold for widespread adoption. These developments collectively suggest a transformative year for clean energy infrastructure.",
          "attachments": []
        },
        "expected_output": {
          "status": "sent",
          "message_id": "msg_a1b2c3d4"
        },
        "depends_on": [2]
      }
    ],
    "final_answer": null
  },
  "metadata": {
    "tags": ["cross-category", "information-retrieval", "text-processing", "communication"],
    "difficulty": "medium",
    "cross_category": true,
    "num_tools": 3,
    "max_depth": 3,
    "generated_at": "2026-02-21T10:00:00Z"
  }
}
```

### A.2 Scored Result (JSON)

```json
{
  "task_id": "L1_chain_search_summarize_email_042",
  "model": "gpt-4o-2025-11-20",
  "level": "L1",
  "topology": "chain",
  "task_score": 0.85,
  "call_scores": [
    {
      "step": 1,
      "tool_selected_correctly": true,
      "args_correct": 1.0,
      "output_routing_correct": true
    },
    {
      "step": 2,
      "tool_selected_correctly": true,
      "args_correct": 0.67,
      "output_routing_correct": true
    },
    {
      "step": 3,
      "tool_selected_correctly": true,
      "args_correct": 0.90,
      "output_routing_correct": true
    }
  ],
  "sub_scores": {
    "tool_sequence_score": 1.0,
    "argument_score": 0.857,
    "completeness_score": 1.0
  },
  "diagnostics": {
    "tools_involved": ["web_search", "summarize_text", "send_email"],
    "individual_accuracy_min": 1.0,
    "composition_gap_this_task": 0.15,
    "error_type": "argument_mismatch",
    "error_detail": "summarize_text: text argument was truncated, style was 'formal' instead of 'professional'"
  },
  "raw_response": "...",
  "latency_ms": 2340,
  "evaluated_at": "2026-02-21T14:30:00Z"
}
```

### A.3 Aggregated Metrics (JSON)

```json
{
  "model": "gpt-4o-2025-11-20",
  "timestamp": "2026-02-21T15:00:00Z",
  "suite_version": "v1_seed42",
  "headline_metrics": {
    "overall_accuracy": 0.72,
    "composition_gap_overall": 0.18,
    "composition_gap_L1": 0.12,
    "composition_gap_L2": 0.19,
    "composition_gap_L3": 0.25
  },
  "per_level_accuracy": {
    "L0_node": 0.91,
    "L1_chain": 0.79,
    "L2_parallel": 0.68,
    "L3_dag": 0.52
  },
  "diagnostic_metrics": {
    "tool_selection_accuracy": 0.94,
    "argument_accuracy": 0.81,
    "data_flow_accuracy": 0.73,
    "completion_rate": 0.88,
    "hallucinated_tool_rate": 0.02,
    "early_termination_rate": 0.09,
    "cross_category_gap": 0.22,
    "within_category_gap": 0.11
  },
  "per_tool_L0_accuracy": {
    "web_search": 1.0,
    "calculator": 0.95,
    "get_weather": 1.0,
    "summarize_text": 0.88,
    "send_email": 0.92,
    "data_filter": 0.83,
    "merge_data": 0.75,
    "...": "..."
  },
  "task_count": {
    "L0": 216,
    "L1": 200,
    "L2": 120,
    "L3": 120,
    "total": 656
  }
}
```

### A.4 Composition Template Schema (YAML)

```yaml
# Full schema for a composition template
template_id: string            # Unique identifier
level: enum[L0, L1, L2, L3]   # Composition level
topology: enum[node, chain, parallel, dag]  # Topology type
description: string            # Human-readable description

tool_graph:                    # Execution graph
  - step: int                  # Execution order (topological)
    tool: string               # Tool name (must exist in registry)
    args_template: dict        # Arguments with {{placeholders}}
    output_binding: string     # Name to reference this output in later steps
    depends_on: list[int]      # Steps whose output this step consumes

parameters:                    # Parameter definitions
  param_name:
    type: enum[sampled, generated, uniform_int, uniform_float, choice, constant]
    source: string             # For 'sampled': path to JSON pool
    pattern: string            # For 'generated': generation pattern
    min: number                # For 'uniform_int/float'
    max: number                # For 'uniform_int/float'
    options: list              # For 'choice'
    value: any                 # For 'constant'

prompt_templates: list[string] # NL prompt variants with {{placeholders}}

tags: list[string]             # For analysis/filtering
cross_category: bool           # Whether tools span multiple categories
difficulty: enum[easy, medium, hard]  # Estimated difficulty
```

---

## Appendix B: Type System for Tool Composition

A key enabler of programmatic task generation is the **type system** that governs how tools can compose. Each tool has typed inputs and outputs, and composition is valid when types match:

```python
# Core types used across all tools
ToolType = Literal[
    "str",           # Plain text
    "int",           # Integer
    "float",         # Floating point
    "bool",          # Boolean
    "list[str]",     # List of strings
    "list[dict]",    # Structured data (table-like)
    "dict",          # Key-value mapping
    "SearchResults", # web_search output
    "FileContent",   # read_file output
    "EmailResult",   # send_email output
    "WeatherData",   # get_weather output
    # ... domain-specific types
]

# Type compatibility matrix (which outputs can feed which inputs)
# This matrix drives the template generator:
TYPE_COMPATIBILITY = {
    "SearchResults": ["str", "list[dict]"],  # Can be used as text or structured data
    "FileContent": ["str"],                   # Can be used as text
    "list[dict]": ["list[dict]", "str"],      # Structured data, or stringified
    "dict": ["dict", "str"],                  # Mapping, or stringified
    "str": ["str"],                           # Text
    "float": ["float", "str"],               # Number, or stringified
    # ...
}
```

This type system is used by the composition engine to:
1. **Validate templates**: Ensure that data flows between tools are type-compatible
2. **Generate new templates**: Automatically discover valid compositions by matching output types to input types
3. **Score data flow**: Check whether the model correctly routes typed outputs between tools

---

## Appendix C: Error Taxonomy

Errors in compositional tool use fall into a systematic taxonomy that we use for analysis:

| Error Class | Code | Description | Example |
|------------|------|-------------|---------|
| **Wrong Tool** | E1 | Model selects incorrect tool | Uses `classify_text` instead of `sentiment_analysis` |
| **Missing Step** | E2 | Model skips a required step in the chain | Skips `summarize_text` in search->summarize->email |
| **Wrong Order** | E3 | Model calls tools in wrong sequence | Calls `send_email` before `web_search` |
| **Wrong Arguments** | E4 | Correct tool, incorrect arguments | Passes wrong column name to `data_filter` |
| **Broken Data Flow** | E5 | Model does not pass output of step N to step N+1 | Hardcodes a value instead of using search results |
| **Hallucinated Tool** | E6 | Model invokes a tool not in the schema | Calls `analyze_sentiment` (nonexistent) |
| **Unnecessary Tool** | E7 | Model calls extra tools not needed | Adds `store_memory` when not asked |
| **Partial Completion** | E8 | Model completes some steps then stops | Searches and summarizes but does not email |
| **Parallel as Sequential** | E9 | Model serializes parallel-independent calls | Calls 3 `get_weather` sequentially when parallel is valid |
| **Format Error** | E10 | Model output cannot be parsed as tool calls | Returns natural language instead of function call |

---

*End of Architecture Specification*
