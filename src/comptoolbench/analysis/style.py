"""Publication-quality matplotlib style configuration for CompToolBench paper figures.

Design decisions (all verified against primary sources):

NeurIPS 2024 format specs:
  - Text area: 5.5 inches wide, 9 inches tall (source: NeurIPS 2024 style guide)
  - Body font: 10pt Times New Roman
  - Single-column figure width: 5.5 inches (full text width)
  - Two-subfigure row: ~2.65 inches each (half text width minus gap)
  - Three-subfigure row: ~1.7 inches each (third text width minus gaps)

Font sizes (matched to 10pt body text, per academic best practice):
  - Axis labels: 9pt (slightly smaller than body, standard for data-dense figures)
  - Tick labels: 7pt (smaller than labels, keeps clutter low)
  - Legend: 7pt
  - Subplot titles: 9pt
  Source: https://allanchain.github.io/blog/post/mpl-paper-tips/
          https://jwalton.info/Embed-Publication-Matplotlib-Latex/

Color palette - Okabe-Ito / Bang Wong (8 colors, colorblind-safe):
  Source: Okabe & Ito (2008) "Color Universal Design"
          Wong (2011) Nature Methods "Points of view: Color blindness"
          Used as default in Claus Wilke "Fundamentals of Data Visualization"
  Hex codes: #E69F00, #56B4E9, #009E73, #F0E442, #0072B2, #D55E00, #CC79A7, #000000

DPI:
  - 300 DPI minimum for most journals; 600 DPI for final camera-ready
  Source: https://pythonforthelab.com/blog/python-tip-ready-publish-matplotlib-figures/

File format:
  - PDF for vector figures (line plots, bar charts, scatter plots)
  - PNG at 600 DPI as fallback for complex plots
  Source: https://fraserlab.com/2014/08/20/Figures-with-Python/
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Okabe-Ito / Bang Wong colorblind-safe palette (8 colors)
# Source: Okabe & Ito (2008); Wong, Nature Methods 2011
# ---------------------------------------------------------------------------
OKABE_ITO: list[str] = [
    "#E69F00",  # orange       — strong, warm; good for prominent categories
    "#56B4E9",  # sky blue     — light, cool
    "#009E73",  # bluish green — accessible alternative to pure green
    "#F0E442",  # yellow       — use sparingly (low contrast on white)
    "#0072B2",  # blue         — strong, high contrast
    "#D55E00",  # vermillion   — strong red-orange, distinguishable from orange
    "#CC79A7",  # reddish purple
    "#000000",  # black        — for reference lines, borders
]

# Recommended ordering for most figures (highest contrast first, yellow last)
PALETTE: list[str] = [
    OKABE_ITO[4],  # blue       #0072B2  — best single-color choice
    OKABE_ITO[5],  # vermillion #D55E00  — strong contrast against blue
    OKABE_ITO[2],  # bg-green   #009E73
    OKABE_ITO[0],  # orange     #E69F00
    OKABE_ITO[6],  # r-purple   #CC79A7
    OKABE_ITO[1],  # sky blue   #56B4E9
    OKABE_ITO[7],  # black      #000000
    OKABE_ITO[3],  # yellow     #F0E442  — last resort
]

# Model-family color assignments (stable across all figures)
# Mapping: provider family → color index
MODEL_COLORS: dict[str, str] = {
    "gpt": OKABE_ITO[4],        # OpenAI  → blue
    "claude": OKABE_ITO[5],     # Anthropic → vermillion
    "gemini": OKABE_ITO[2],     # Google  → bluish green
    "llama": OKABE_ITO[0],      # Meta    → orange
    "deepseek": OKABE_ITO[6],   # DeepSeek → reddish purple
    "qwen": OKABE_ITO[1],       # Alibaba → sky blue
    "mistral": OKABE_ITO[3],    # Mistral → yellow
    "command": OKABE_ITO[5],    # Cohere  → vermillion
    "grok": OKABE_ITO[7],       # xAI     → black
    "granite": OKABE_ITO[6],    # IBM     → reddish purple
}

# Level colors — used for L0/L1/L2/L3 in grouped bar charts
LEVEL_COLORS: dict[str, str] = {
    "L0": OKABE_ITO[4],   # blue       — baseline, "safe"
    "L1": OKABE_ITO[2],   # bg-green   — chain
    "L2": OKABE_ITO[0],   # orange     — parallel
    "L3": OKABE_ITO[5],   # vermillion — DAG, hardest → warm/urgent color
}

# Gap heatmap colormap — diverging, blue-white-red
# Blue = negative gap (better at composition), Red = positive gap (worse)
GAP_CMAP: str = "RdBu_r"  # reversed: red=high gap (bad), blue=low/neg gap (good)

# ---------------------------------------------------------------------------
# NeurIPS 2024 figure dimensions (inches)
# NeurIPS text area: 5.5 inches wide. Source: NeurIPS 2024 LaTeX style guide
# https://media.neurips.cc/Conferences/NeurIPS2022/Styles/neurips_2022.pdf
# ---------------------------------------------------------------------------
TEXTWIDTH: float = 5.5       # Full text width (single-column NeurIPS)
GOLDEN: float = (5**0.5 - 1) / 2  # ~0.618 — golden ratio height/width

# Standard figure widths
FIG_FULL: float = TEXTWIDTH                    # 5.50 in — full-width figure
FIG_HALF: float = (TEXTWIDTH - 0.3) / 2       # 2.60 in — half-width (side-by-side pair)
FIG_THIRD: float = (TEXTWIDTH - 0.6) / 3      # 1.63 in — third-width (three columns)

# Standard figure heights (using golden ratio unless noted)
FIG_SQUARE: float = TEXTWIDTH * GOLDEN        # ~3.40 in — full-width, golden ratio height
FIG_TALL: float = TEXTWIDTH * 0.75            # 4.13 in — taller than golden (for heatmaps)
FIG_SHORT: float = TEXTWIDTH * 0.50           # 2.75 in — short wide (overview bar charts)


def figure_size(
    width: float = FIG_FULL,
    aspect: float = GOLDEN,
) -> tuple[float, float]:
    """Return (width, height) in inches using the given aspect ratio.

    Args:
        width: Figure width in inches. Use FIG_FULL, FIG_HALF, FIG_THIRD, etc.
        aspect: Height-to-width ratio. Default is golden ratio (~0.618).

    Returns:
        (width, height) tuple suitable for plt.figure(figsize=...).
    """
    return (width, width * aspect)


# ---------------------------------------------------------------------------
# rcParams — applied globally via apply_style()
# ---------------------------------------------------------------------------
RCPARAMS: dict[str, object] = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size": 9,                   # Base font size (NeurIPS body = 10pt; we use 9pt for tighter figures)
    "axes.labelsize": 9,              # Axis label size
    "axes.titlesize": 9,              # Subplot title size
    "xtick.labelsize": 7,             # Tick label size
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,

    # Figure
    "figure.dpi": 150,                # Screen preview DPI (save at 300-600)
    "figure.autolayout": True,        # Tight layout by default
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,               # Default save DPI (increase to 600 for camera-ready)
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",

    # Axes
    "axes.linewidth": 0.5,
    "axes.spines.top": False,         # Remove top spine (cleaner look)
    "axes.spines.right": False,       # Remove right spine
    "axes.grid": True,
    "axes.grid.axis": "y",            # Horizontal grid only (for bar charts)
    "axes.prop_cycle": mpl.cycler(color=PALETTE),

    # Grid
    "grid.linewidth": 0.4,
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,

    # Lines
    "lines.linewidth": 1.2,
    "lines.markersize": 4,

    # Ticks
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.4,
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.4,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend
    "legend.frameon": False,           # No box around legend
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,

    # Hatching (for accessibility in B&W print)
    "hatch.linewidth": 0.5,

    # PDF font embedding (critical for camera-ready submission)
    "pdf.fonttype": 42,               # TrueType fonts in PDF (not Type 3 bitmaps)
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply the CompToolBench publication style globally.

    Call once at the top of any script or notebook that generates figures.

    Example:
        >>> from comptoolbench.analysis.style import apply_style
        >>> apply_style()
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(figsize=(5.5, 3.4))
    """
    plt.rcParams.update(RCPARAMS)


def get_model_color(model_name: str) -> str:
    """Return a consistent color for a model based on its provider family.

    Args:
        model_name: Model name string (e.g., 'gpt-4o', 'claude-opus-4').

    Returns:
        Hex color string from OKABE_ITO palette.
    """
    model_lower = model_name.lower()
    for prefix, color in MODEL_COLORS.items():
        if model_lower.startswith(prefix):
            return color
    # Fallback: cycle through palette by hash
    return PALETTE[hash(model_name) % len(PALETTE)]


def get_level_color(level: str) -> str:
    """Return the canonical color for a composition level.

    Args:
        level: One of 'L0', 'L1', 'L2', 'L3'.

    Returns:
        Hex color string.
    """
    return LEVEL_COLORS.get(level.upper(), PALETTE[0])
