"""CompToolBench Gradio Demo — Interactive Benchmark Explorer.

Designed for HuggingFace Spaces (free CPU tier).
Launch locally:  python demo/app.py
"""

from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# DATA — extracted verbatim from paper/tables/leaderboard.tex
# Columns: Model, Provider, L0, L1, L2, L3, Overall, Delta, SelectionGap
# Delta = L0 - L3 (positive = degradation).
# SelectionGap (dagger) = L0 < avg(L1,L2,L3).
# ---------------------------------------------------------------------------

CLOUD_MODELS = [
    # (Model, Provider, L0, L1, L2, L3, Overall, Delta, SelectionGap)
    ("Llama 3.1 8B",      "Groq",       27.1, 75.8, 87.1, 76.0, 66.4, -48.9, True),
    ("Command A",          "Cohere",     45.8, 62.7, 87.8, 40.8, 58.4,   5.1, True),
    ("Mistral Small",      "Mistral",    45.8, 59.7, 87.6, 40.9, 57.5,   4.9, True),
    ("Command R+",         "Cohere",     43.8, 57.5, 88.0, 40.3, 56.2,   3.4, True),
    ("Llama 3.1 8B",      "Cerebras",   31.2, 66.1, 81.2, 46.4, 56.0, -15.1, True),
    ("Mistral Large",      "Mistral",    39.6, 59.5, 87.9, 38.5, 55.4,   1.1, True),
    ("Mistral Medium",     "Mistral",    43.8, 57.5, 87.9, 36.3, 55.2,   7.4, True),
    ("Gemini 2.0 Flash",   "OpenRouter", 39.6, 52.4, 85.7, 39.0, 52.8,   0.6, True),
    ("GPT-OSS 120B",       "Cerebras",   45.8, 56.3, 56.1, 29.0, 47.2,  16.8, True),
    ("Llama 4 Scout 17B",  "Groq",       37.5, 49.6, 55.8,  7.0, 37.7,  30.5, False),
]

LOCAL_MODELS = [
    ("Granite4 3B",        "Ollama",     45.8, 57.3, 56.1, 30.2, 47.8,  15.6, True),
    ("Granite4 1B",        "Ollama",     41.7, 56.3, 55.9, 29.9, 46.4,  11.8, True),
    ("Mistral 7B",         "Ollama",     43.8, 57.7, 49.2, 30.5, 46.1,  13.3, True),
    ("Llama 3.1 8B",      "Ollama",     39.6, 56.7, 56.1, 29.5, 45.9,  10.1, True),
    ("Mistral Nemo 12B",   "Ollama",     37.5, 58.4, 51.0, 31.8, 45.5,   5.7, True),
    ("Qwen 2.5 7B",        "Ollama",     39.6, 56.7, 53.8, 25.8, 44.6,  13.8, True),
    ("Mistral Small 24B",  "Ollama",     37.5, 51.1, 47.7, 22.6, 40.3,  14.9, True),
    ("Qwen3 8B",           "Ollama",     35.4, 52.0, 36.9, 21.8, 37.7,  13.7, True),
]

# Averages from the table
AVERAGES = {
    "All models":  {"L0": 40.0, "L1": 58.0, "L2": 67.3, "L3": 34.2, "Overall": 49.8, "Delta": 5.8},
    "Cloud avg":   {"L0": 40.0, "L1": 59.7, "L2": 80.5, "L3": 39.4, "Overall": 54.3, "Delta": 0.6},
    "Local avg":   {"L0": 40.1, "L1": 55.8, "L2": 50.8, "L3": 27.8, "Overall": 44.3, "Delta": 12.3},
}


def _build_display_name(model: str, provider: str) -> str:
    """Build a unique display name like 'Llama 3.1 8B (Groq)'."""
    return f"{model} ({provider})"


def build_full_dataframe() -> pd.DataFrame:
    """Build the full leaderboard DataFrame with all 18 models."""
    rows = []
    for model, provider, l0, l1, l2, l3, overall, delta, sgap in CLOUD_MODELS:
        composed_avg = (l1 + l2 + l3) / 3.0
        rows.append({
            "Rank": 0,
            "Model": _build_display_name(model, provider),
            "Provider": provider,
            "Type": "Cloud",
            "L0": l0,
            "L1": l1,
            "L2": l2,
            "L3": l3,
            "Overall": overall,
            "Delta": delta,
            "Selection Gap": sgap,
            "Composed Avg": round(composed_avg, 1),
        })
    for model, provider, l0, l1, l2, l3, overall, delta, sgap in LOCAL_MODELS:
        composed_avg = (l1 + l2 + l3) / 3.0
        rows.append({
            "Rank": 0,
            "Model": _build_display_name(model, provider),
            "Provider": provider,
            "Type": "Local",
            "L0": l0,
            "L1": l1,
            "L2": l2,
            "L3": l3,
            "Overall": overall,
            "Delta": delta,
            "Selection Gap": sgap,
            "Composed Avg": round(composed_avg, 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Overall", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# PLOTLY THEME CONSTANTS
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
CARD_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
ACCENT_BLUE = "#4fc3f7"
ACCENT_GREEN = "#66bb6a"
ACCENT_ORANGE = "#ffa726"
ACCENT_RED = "#ef5350"
ACCENT_PURPLE = "#ab47bc"

LEVEL_COLORS = {
    "L0": ACCENT_BLUE,
    "L1": ACCENT_GREEN,
    "L2": ACCENT_ORANGE,
    "L3": ACCENT_RED,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=60, r=30, t=60, b=80),
    hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT_COLOR, bordercolor=GRID_COLOR),
)


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Apply consistent dark theme to a plotly figure."""
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# TAB 1: LEADERBOARD (styled DataFrame)
# ---------------------------------------------------------------------------
def format_leaderboard_html(df: pd.DataFrame) -> str:
    """Build a styled HTML leaderboard table with color-coded scores."""

    def _score_color(val: float, low: float = 20.0, high: float = 80.0) -> str:
        """Map a score to a green-yellow-red gradient."""
        ratio = max(0.0, min(1.0, (val - low) / (high - low)))
        if ratio > 0.5:
            # green zone
            r = int(255 * (1 - (ratio - 0.5) * 2))
            g = 200
        else:
            # red zone
            r = 240
            g = int(200 * ratio * 2)
        return f"rgb({r},{g},80)"

    def _gap_badge(has_gap: bool) -> str:
        if has_gap:
            return '<span style="color:#66bb6a;font-weight:600;">Yes</span>'
        return '<span style="color:#999;">No</span>'

    def _type_badge(model_type: str) -> str:
        if model_type == "Cloud":
            return '<span style="background:#1e3a5f;color:#4fc3f7;padding:2px 8px;border-radius:4px;font-size:0.8em;">Cloud</span>'
        return '<span style="background:#2e3a1f;color:#a5d6a7;padding:2px 8px;border-radius:4px;font-size:0.8em;">Local</span>'

    css = """
    <style>
    .lb-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 14px;
    }
    .lb-table th {
        background: #0d1b2a;
        color: #b0bec5;
        padding: 12px 10px;
        text-align: center;
        font-weight: 600;
        border-bottom: 2px solid #2a2a4a;
        cursor: pointer;
        user-select: none;
        white-space: nowrap;
    }
    .lb-table th:first-child, .lb-table th:nth-child(2) {
        text-align: left;
    }
    .lb-table td {
        padding: 10px 10px;
        text-align: center;
        border-bottom: 1px solid #1a1a3a;
    }
    .lb-table td:first-child {
        font-weight: 700;
        color: #ffd54f;
        text-align: center;
        width: 40px;
    }
    .lb-table td:nth-child(2) {
        text-align: left;
        font-weight: 500;
        color: #e0e0e0;
        max-width: 220px;
    }
    .lb-table tr:hover {
        background: #1e2d4a !important;
    }
    .lb-table tr:nth-child(even) {
        background: #111827;
    }
    .lb-table tr:nth-child(odd) {
        background: #0f1729;
    }
    .lb-table .score-cell {
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .lb-table .overall-cell {
        font-weight: 700;
        font-size: 15px;
    }
    .lb-avg-row td {
        background: #1a1a2e !important;
        border-top: 2px solid #4fc3f7;
        font-style: italic;
        color: #90caf9;
    }
    .lb-divider td {
        background: #1a1a2e !important;
        border-top: 2px solid #2a2a4a;
        padding: 2px;
        height: 4px;
    }
    </style>
    """

    header = """
    <table class="lb-table">
    <thead>
    <tr>
        <th>#</th>
        <th>Model</th>
        <th>Type</th>
        <th>L0</th>
        <th>L1</th>
        <th>L2</th>
        <th>L3</th>
        <th>Overall</th>
        <th>Selection Gap</th>
    </tr>
    </thead>
    <tbody>
    """

    rows_html = ""
    for _, row in df.iterrows():
        l0_c = _score_color(row["L0"])
        l1_c = _score_color(row["L1"])
        l2_c = _score_color(row["L2"])
        l3_c = _score_color(row["L3"])
        ov_c = _score_color(row["Overall"])

        rows_html += f"""
        <tr>
            <td>{row['Rank']}</td>
            <td>{row['Model']}</td>
            <td>{_type_badge(row['Type'])}</td>
            <td class="score-cell" style="color:{l0_c}">{row['L0']:.1f}</td>
            <td class="score-cell" style="color:{l1_c}">{row['L1']:.1f}</td>
            <td class="score-cell" style="color:{l2_c}">{row['L2']:.1f}</td>
            <td class="score-cell" style="color:{l3_c}">{row['L3']:.1f}</td>
            <td class="overall-cell" style="color:{ov_c}">{row['Overall']:.1f}</td>
            <td>{_gap_badge(row['Selection Gap'])}</td>
        </tr>
        """

    # Divider
    rows_html += '<tr class="lb-divider"><td colspan="9"></td></tr>'

    # Averages
    for label, avg in AVERAGES.items():
        l0_c = _score_color(avg["L0"])
        l1_c = _score_color(avg["L1"])
        l2_c = _score_color(avg["L2"])
        l3_c = _score_color(avg["L3"])
        ov_c = _score_color(avg["Overall"])
        rows_html += f"""
        <tr class="lb-avg-row">
            <td></td>
            <td><em>{label}</em></td>
            <td></td>
            <td class="score-cell" style="color:{l0_c}">{avg['L0']:.1f}</td>
            <td class="score-cell" style="color:{l1_c}">{avg['L1']:.1f}</td>
            <td class="score-cell" style="color:{l2_c}">{avg['L2']:.1f}</td>
            <td class="score-cell" style="color:{l3_c}">{avg['L3']:.1f}</td>
            <td class="overall-cell" style="color:{ov_c}">{avg['Overall']:.1f}</td>
            <td></td>
        </tr>
        """

    footer = "</tbody></table>"
    return css + header + rows_html + footer


# ---------------------------------------------------------------------------
# TAB 2: SELECTION GAP VISUALIZATION
# ---------------------------------------------------------------------------
def plot_selection_gap(df: pd.DataFrame) -> go.Figure:
    """Bar chart: L0 vs Composed Average for each model, with gap arrows."""
    df_sorted = df.sort_values("Overall", ascending=True)

    fig = go.Figure()

    # L0 bars
    fig.add_trace(go.Bar(
        y=df_sorted["Model"],
        x=df_sorted["L0"],
        name="L0 (Single Tool)",
        orientation="h",
        marker=dict(color=ACCENT_BLUE, line=dict(width=0)),
        text=[f"{v:.1f}" for v in df_sorted["L0"]],
        textposition="inside",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{y}</b><br>L0: %{x:.1f}%<extra></extra>",
    ))

    # Composed average bars
    fig.add_trace(go.Bar(
        y=df_sorted["Model"],
        x=df_sorted["Composed Avg"],
        name="Composed Avg (L1-L3)",
        orientation="h",
        marker=dict(color=ACCENT_ORANGE, line=dict(width=0)),
        text=[f"{v:.1f}" for v in df_sorted["Composed Avg"]],
        textposition="inside",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{y}</b><br>Composed Avg: %{x:.1f}%<extra></extra>",
    ))

    # Add gap annotations
    for _, row in df_sorted.iterrows():
        gap = row["Composed Avg"] - row["L0"]
        direction = "+" if gap > 0 else ""
        color = ACCENT_GREEN if gap > 0 else ACCENT_RED
        x_pos = max(row["L0"], row["Composed Avg"]) + 2
        fig.add_annotation(
            x=x_pos,
            y=row["Model"],
            text=f"<b>{direction}{gap:.1f}</b>",
            showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left",
        )

    fig = _apply_layout(
        fig,
        title=dict(text="Selection Gap: L0 (Single Tool) vs Composed Average (L1-L3)", font=dict(size=16)),
        barmode="group",
        xaxis=dict(title="Accuracy (%)", range=[0, 100], gridcolor=GRID_COLOR),
        yaxis=dict(title="", gridcolor=GRID_COLOR, tickfont=dict(size=11)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=700,
    )
    return fig


# ---------------------------------------------------------------------------
# TAB 3: LEVEL COMPARISON
# ---------------------------------------------------------------------------
def plot_level_comparison(df: pd.DataFrame, model_type: str = "All") -> go.Figure:
    """Grouped bar chart: L0/L1/L2/L3 per model, filterable by type."""
    if model_type == "Cloud":
        df_plot = df[df["Type"] == "Cloud"].copy()
    elif model_type == "Local":
        df_plot = df[df["Type"] == "Local"].copy()
    else:
        df_plot = df.copy()

    df_plot = df_plot.sort_values("Overall", ascending=True)

    fig = go.Figure()

    for level, color in LEVEL_COLORS.items():
        fig.add_trace(go.Bar(
            y=df_plot["Model"],
            x=df_plot[level],
            name=level,
            orientation="h",
            marker=dict(color=color, line=dict(width=0.5, color="#111")),
            text=[f"{v:.1f}" for v in df_plot[level]],
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate=f"<b>%{{y}}</b><br>{level}: %{{x:.1f}}%<extra></extra>",
        ))

    n_models = len(df_plot)
    fig = _apply_layout(
        fig,
        title=dict(
            text=f"Performance by Composition Level ({model_type} Models)",
            font=dict(size=16),
        ),
        barmode="group",
        xaxis=dict(title="Accuracy (%)", range=[0, 105], gridcolor=GRID_COLOR),
        yaxis=dict(title="", gridcolor=GRID_COLOR, tickfont=dict(size=11)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=max(400, n_models * 50 + 150),
    )
    return fig


def plot_level_radar() -> go.Figure:
    """Radar/spider chart comparing cloud vs local averages."""
    categories = ["L0", "L1", "L2", "L3"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[AVERAGES["Cloud avg"]["L0"], AVERAGES["Cloud avg"]["L1"],
           AVERAGES["Cloud avg"]["L2"], AVERAGES["Cloud avg"]["L3"],
           AVERAGES["Cloud avg"]["L0"]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Cloud Avg",
        line=dict(color=ACCENT_BLUE, width=2),
        fillcolor="rgba(79, 195, 247, 0.2)",
    ))

    fig.add_trace(go.Scatterpolar(
        r=[AVERAGES["Local avg"]["L0"], AVERAGES["Local avg"]["L1"],
           AVERAGES["Local avg"]["L2"], AVERAGES["Local avg"]["L3"],
           AVERAGES["Local avg"]["L0"]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Local Avg",
        line=dict(color=ACCENT_PURPLE, width=2),
        fillcolor="rgba(171, 71, 188, 0.2)",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(
                visible=True, range=[0, 90],
                gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                tickfont=dict(color=TEXT_COLOR, size=10),
            ),
            angularaxis=dict(
                gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                tickfont=dict(color=TEXT_COLOR, size=13, family="Inter, system-ui, sans-serif"),
            ),
        ),
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        title=dict(text="Cloud vs Local: Performance Profile", font=dict(size=16, color=TEXT_COLOR)),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# TAB 4: ABOUT
# ---------------------------------------------------------------------------
ABOUT_MD = """
## CompToolBench: Measuring Compositional Tool-Use in LLMs

**CompToolBench** is a benchmark that measures *compositional tool-use generalization* in large
language models. The central question: if an LLM can use tools A, B, and C individually, can it
compose them into novel pipelines like `A(B(C(x)))`?

---

### Composition Levels

| Level | Topology | Description |
|:------|:---------|:------------|
| **L0 (Node)** | Single call | One tool, correct arguments -- the baseline |
| **L1 (Chain)** | A -> B -> C | Sequential: output of tool_i feeds tool_{i+1} |
| **L2 (Parallel)** | [A, B] -> C | Independent calls whose results merge downstream |
| **L3 (DAG)** | Complex graph | Branching, merging, and sequential edges combined |

---

### Key Finding: The Selection Gap

> **17 out of 18 models exhibit a Selection Gap**: their L0 (single-tool) accuracy is *lower*
> than their average accuracy on composed tasks (L1-L3).

This is counter-intuitive. Models are *better* at multi-step composition than at simple
single-tool selection. The explanation: L0 tests pure tool *selection* (choosing the right
tool from a large catalogue), while L1-L3 tasks provide more structural context that narrows
the search space. The hardest part of tool use is not execution -- it is *selection*.

---

### Benchmark Details

- **18 models** evaluated (10 cloud API, 8 local via Ollama)
- **106 deterministic tool simulations** across 15 categories
- **200 tasks** at 4 composition levels (L0-L3)
- **Deterministic scoring** with verifiable ground-truth execution traces

---

### Links

| Resource | Link |
|:---------|:-----|
| Paper | [ArXiv (coming soon)](#) |
| Code | [github.com/ronyrahmaan/comptoolbench](https://github.com/ronyrahmaan/comptoolbench) |
| Author | Md A Rahman, Texas Tech University |

---

<p style="text-align:center;color:#666;font-size:0.85em;">
CompToolBench -- February 2026
</p>
"""


# ---------------------------------------------------------------------------
# GRADIO APP
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Build the full 4-tab Gradio Blocks application."""
    df = build_full_dataframe()

    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-header h1 {
        font-size: 2em;
        font-weight: 700;
        background: linear-gradient(135deg, #4fc3f7, #ab47bc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #aaa;
        font-size: 1.1em;
    }
    .stat-row {
        display: flex;
        justify-content: center;
        gap: 40px;
        padding: 15px 0;
        flex-wrap: wrap;
    }
    .stat-item {
        text-align: center;
    }
    .stat-num {
        font-size: 1.8em;
        font-weight: 700;
        color: #4fc3f7;
    }
    .stat-label {
        font-size: 0.85em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    footer {visibility: hidden;}
    """

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f0f1a",
        body_background_fill_dark="#0f0f1a",
        block_background_fill="#1a1a2e",
        block_background_fill_dark="#1a1a2e",
        block_border_color="#2a2a4a",
        block_border_color_dark="#2a2a4a",
        block_label_text_color="#b0bec5",
        block_label_text_color_dark="#b0bec5",
        block_title_text_color="#e0e0e0",
        block_title_text_color_dark="#e0e0e0",
        body_text_color="#e0e0e0",
        body_text_color_dark="#e0e0e0",
        body_text_color_subdued="#888",
        body_text_color_subdued_dark="#888",
        background_fill_primary="#16213e",
        background_fill_primary_dark="#16213e",
        background_fill_secondary="#1a1a2e",
        background_fill_secondary_dark="#1a1a2e",
        border_color_accent="#4fc3f7",
        border_color_accent_dark="#4fc3f7",
        color_accent_soft="#1e3a5f",
        color_accent_soft_dark="#1e3a5f",
        button_primary_background_fill="#4fc3f7",
        button_primary_background_fill_dark="#4fc3f7",
        button_primary_text_color="#0f0f1a",
        button_primary_text_color_dark="#0f0f1a",
    )

    # Gradio 6+ moved theme/css from Blocks() to launch().
    # Detect version and pass params accordingly.
    _gradio_major = int(gr.__version__.split(".")[0])
    _blocks_kwargs: dict = {"title": "CompToolBench"}
    if _gradio_major < 6:
        _blocks_kwargs["theme"] = theme
        _blocks_kwargs["css"] = custom_css

    with gr.Blocks(**_blocks_kwargs) as app:
        # ── Header ──
        gr.HTML("""
        <div class="main-header">
            <h1>CompToolBench</h1>
            <p>Measuring Compositional Tool-Use Generalization in LLMs</p>
        </div>
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-num">18</div>
                <div class="stat-label">Models</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">106</div>
                <div class="stat-label">Tools</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">4</div>
                <div class="stat-label">Composition Levels</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">17/18</div>
                <div class="stat-label">Show Selection Gap</div>
            </div>
        </div>
        """)

        # ── Tab 1: Leaderboard ──
        with gr.Tab("Leaderboard", id="leaderboard"):
            gr.HTML(format_leaderboard_html(df))
            gr.Markdown(
                """
                **Reading the table:** Scores are accuracy percentages. Colors range from
                <span style="color:#ef5350">red</span> (low) to
                <span style="color:#66bb6a">green</span> (high).
                **Selection Gap** = model's L0 is lower than its average of L1-L3
                (i.e., models are *better* at composed tasks than single-tool selection).
                **Delta** in the paper = L0 minus L3 (positive means degradation from single to DAG).
                """,
                elem_classes=["block"],
            )

        # ── Tab 2: Selection Gap ──
        with gr.Tab("Selection Gap", id="selection-gap"):
            gr.Markdown(
                "### The Selection Gap: Why are models better at *composed* tasks than single-tool calls?"
            )
            gr.Plot(plot_selection_gap(df))
            gr.Markdown(
                """
                **How to read this chart:** For each model, the blue bar shows L0 accuracy
                (single-tool selection) and the orange bar shows the average of L1, L2, L3
                (composed tasks). The number on the right is the gap.

                A **positive gap** (green number) means the model performs *better* on composed
                tasks -- the Selection Gap. This happens because multi-step prompts provide
                richer structural context that narrows the tool search space.

                Only **Llama 4 Scout 17B** does not exhibit a Selection Gap, because its L3
                accuracy collapses to 7.0% (catastrophic DAG failure).
                """
            )

        # ── Tab 3: Level Comparison ──
        with gr.Tab("Level Comparison", id="level-comparison"):
            gr.Markdown("### Performance breakdown by composition level")
            model_filter = gr.Radio(
                choices=["All", "Cloud", "Local"],
                value="All",
                label="Filter by deployment type",
            )
            level_chart = gr.Plot(plot_level_comparison(df, "All"))
            model_filter.change(
                fn=lambda t: plot_level_comparison(df, t),
                inputs=[model_filter],
                outputs=[level_chart],
            )

            gr.Markdown("### Cloud vs Local: Aggregate Profile")
            gr.Plot(plot_level_radar())
            gr.Markdown(
                """
                **Key insight:** Cloud models massively outperform local models on L2
                (parallel composition): 80.5% vs 50.8%. This 30-point gap is the largest
                difference between deployment types at any level, suggesting that parallel
                tool orchestration is where API-served models have the biggest advantage.
                """
            )

        # ── Tab 4: About ──
        with gr.Tab("About", id="about"):
            gr.Markdown(ABOUT_MD)

    # Store launch kwargs for Gradio 6+ theme/css
    app._ctb_launch_kwargs = {}  # type: ignore[attr-defined]
    if _gradio_major >= 6:
        app._ctb_launch_kwargs["theme"] = theme  # type: ignore[attr-defined]
        app._ctb_launch_kwargs["css"] = custom_css  # type: ignore[attr-defined]

    return app


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    launch_kwargs = getattr(app, "_ctb_launch_kwargs", {})
    app.launch(share=False, **launch_kwargs)
