import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import apply_global_style

# Custom colors for this graph
TRUST_COLOR = "#E06561"  # Red
DISTRUST_COLOR = "#82C2AB"  # Cyan


def run() -> None:
    """Generate Graph 12: Trust vs Distrust in Media."""
    logger.info("Generating Graph 12: Trust vs Distrust in Media dumbbell chart.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    results = []
    total = len(df)

    # Columns mapping
    channels = [
        ("[Объединенный] Телевидение", "Телевидение"),
        ("[Объединенный] Интернет-издания", "Интернет-издания"),
        ("[Объединенный] Социальные сети", "Социальные сети"),
        ("[Объединенный] Друзья", "Друзья"),
        ("[Объединенный] Газеты", "Газеты"),
        ("[Объединенный] Радио", "Радио"),
        ("[Объединенный] Журналы", "Печатные журналы"),
        ("[Объединенный] Телеграм-каналы", "Телеграм-каналы"),
    ]

    for col, label in channels:
        # Calculate Trust %
        trust_mask = df[col].isin(["Пользуюсь и доверяю", "Доверяю"])
        trust_pct = (trust_mask.sum() / total) * 100

        # Calculate Distrust %
        distrust_mask = df[col].isin(["Пользуюсь, но не доверяю", "Не доверяю"])
        distrust_pct = (distrust_mask.sum() / total) * 100

        results.append({"channel": label, "trust": trust_pct, "distrust": distrust_pct})

    # Convert to DataFrame and sort by Trust for better visualization
    res_df = pd.DataFrame(results).sort_values("trust", ascending=True)

    # Visualization
    apply_global_style()
    plt.figure(figsize=(10, 10))

    # Horizontal lines connecting the dots
    plt.hlines(
        y=res_df["channel"],
        xmin=res_df["distrust"],
        xmax=res_df["trust"],
        color="#CCCCCC",
        alpha=0.5,
        linewidth=2,
    )

    # Plot Distrust (Blue)
    plt.scatter(
        res_df["distrust"],
        res_df["channel"],
        color=DISTRUST_COLOR,
        s=250,
        label="Не доверяю (%)",
        zorder=3,
    )

    # Plot Trust (Orange)
    plt.scatter(
        res_df["trust"],
        res_df["channel"],
        color=TRUST_COLOR,
        s=250,
        label="Доверяю (%)",
        zorder=3,
    )

    plt.title(
        "Доверие и недоверие к СМИ",
        fontsize=24,
        pad=35,
        color="#494949",
        loc="center",
        fontweight="bold",
    )

    plt.xlabel("Доля респондентов (%)", fontsize=14, labelpad=15)
    plt.xlim(
        0, max(res_df["trust"].max(), res_df["distrust"].max()) + 10
    )  # Add some padding

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#494949",
    )

    # Increase Y-axis tick labels size
    plt.tick_params(axis="y", labelsize=12)

    plt.grid(axis="x", color="#CCCCCC", alpha=0.5, linestyle="--")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "graph12.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Add percentage labels for Телевидение and Друзья  # noqa: RUF003
    for _, row in res_df.iterrows():
        if row["channel"] == "Телевидение":
            y_pos = row["channel"]
            # Телевидение - Distrust (green/cyan) # noqa: RUF003
            plt.text(
                row["distrust"] + 3.5,
                y_pos,
                f"{row['distrust']:.1f}%",
                fontsize=11,
                va="center",
                ha="right",
                color=DISTRUST_COLOR,
                fontweight="bold",
            )
            # Телевидение - Trust (red) # noqa: RUF003
            plt.text(
                row["trust"] - 0.7,
                y_pos,
                f"{row['trust']:.1f}%",
                fontsize=11,
                va="center",
                ha="right",
                color=TRUST_COLOR,
                fontweight="bold",
            )
        if row["channel"] == "Друзья":
            y_pos = row["channel"]
            # Друзья - Distrust (green/cyan)
            plt.text(
                row["distrust"] - 3.1,
                y_pos,
                f" {row['distrust']:.1f}%",
                fontsize=11,
                va="center",
                ha="left",
                color=DISTRUST_COLOR,
                fontweight="bold",
            )
            # Друзья - Trust (red)
            plt.text(
                row["trust"] - 0.7,
                y_pos,
                f"{row['trust']:.1f}%",
                fontsize=11,
                va="center",
                ha="right",
                color=TRUST_COLOR,
                fontweight="bold",
            )

    # Add footer
    plt.annotate(
        "Источник: Опрос Мордовского государственного университета имени Н. П. Огарёва",
        xy=(0.48, 0.0208),
        xycoords="figure fraction",
        fontsize=12,
        color="#494949",
        ha="center",
        va="top",
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Graph 12 saved to: {output_path}")


if __name__ == "__main__":
    run()
