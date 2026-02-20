import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style


def run() -> None:
    """Generate Theme 2: Dunning-Kruger effect slope chart."""
    logger.info("Generating Graph 3: Dunning-Kruger effect slope chart.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Define "Can Distinguish" labels
    can_labels = [
        "1. Практически всегда могут отличить",
        "2. В большинстве случаев могут отличить",
    ]

    # Calculate % (using literal column names directly)
    others_can = (
        df[
            "29. Как Вы думаете, большинство людей могут или нет отличить выдуманные, фейковые новости от правдивых, достоверных (ОДИН ответ)?"
        ]
        .isin(can_labels)
        .sum()
        / len(df)
    ) * 100
    self_can = (
        df[
            "30. Как Вы думаете, Вы лично чаще всего можете или нет отличить выдуманные, фейковые новости от правдивых, достоверных (ОДИН ответ)?"
        ]
        .isin(can_labels)
        .sum()
        / len(df)
    ) * 100

    # Visualization
    apply_global_style()
    plt.figure(figsize=(10, 8))  # Keeping same size but making elements larger

    # Slope data
    categories = ["Общество\n(оценка других)", "Лично Я\n(самооценка)"]
    values = [others_can, self_can]

    # Line - making it much thicker
    plt.plot(
        categories,
        values,
        marker="o",
        markersize=30,
        color=NEON_CYAN,
        linewidth=6,
        alpha=0.9,
        zorder=2,
    )

    # Annotate values - increasing font size
    for i, v in enumerate(values):
        plt.text(
            i,
            v + 4,
            f"{v:.1f}%",
            ha="center",
            fontsize=24,
            fontweight="bold",
            color=NEON_CYAN,
        )
        plt.text(
            i,
            v - 8,
            "Могут отличить",
            ha="center",
            fontsize=14,
            color="white",
            alpha=0.8,
        )

    # Gap annotation - more prominent
    gap = self_can - others_can
    plt.annotate(
        f"+{gap:.1f}% разрыв",
        xy=(0.5, (others_can + self_can) / 2),
        xytext=(0.5, (others_can + self_can) / 2 + 10),
        ha="center",
        fontsize=18,
        color="white",
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": "white",
            "lw": 2,
            "connectionstyle": "arc3,rad=.2",
        },
    )

    plt.title(
        "Эффект Даннинга-Крюгера в медиа-грамотности\n(Иллюзия превосходства)",
        fontsize=22,
        pad=40,
        color=TEXT_COLOR,
    )

    plt.ylim(0, 100)
    plt.xlim(-0.3, 1.3)  # Tightened x-axis to make the line longer/more prominent

    plt.grid(axis="y", color="white", alpha=0.1, linestyle="--")

    # Clean up axes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "graph3.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 3 saved to: {output_path}")


if __name__ == "__main__":
    run()
