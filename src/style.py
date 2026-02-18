import matplotlib.pyplot as plt

# Color constants
BACKGROUND_COLOR = "#000000"
NEON_CYAN = "#00FFFF"
LIGHT_GRAY = "#D3D3D3"
WHITE = "#FFFFFF"
TEXT_COLOR = WHITE


def apply_global_style() -> None:
    """Apply global matplotlib style configuration."""
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.facecolor": BACKGROUND_COLOR,
            "axes.edgecolor": LIGHT_GRAY,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "grid.color": LIGHT_GRAY,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "legend.facecolor": BACKGROUND_COLOR,
            "legend.edgecolor": LIGHT_GRAY,
            "font.family": "sans-serif",
        }
    )
    # Ensure Cyrillic fonts work (depending on local system availability)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
