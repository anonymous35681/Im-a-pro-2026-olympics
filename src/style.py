import matplotlib.pyplot as plt

# Color constants
BACKGROUND_COLOR = "#D9D9D9"
NEON_CYAN = "#82C2AB"
LIGHT_GRAY = "#494949"
WHITE = "#FFFFFF"
TEXT_COLOR = LIGHT_GRAY
RED = "#E06561"
YELLOW = "#F0DC58"
CYAN = "#82C2AB"
BLUE = "#81A6DF"
GREEN = "#77B96E"
GRAY = "#494949"


def apply_global_style() -> None:
    """Apply global matplotlib style configuration."""
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": GRAY,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "text.color": GRAY,
            "grid.color": GRAY,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "legend.facecolor": WHITE,
            "legend.edgecolor": GRAY,
            "font.family": "sans-serif",
        }
    )
    # Ensure Cyrillic fonts work (depending on local system availability)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
