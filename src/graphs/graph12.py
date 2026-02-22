import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import TEXT_COLOR, apply_global_style

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
        ("[Объединенный] Телевидение", "ТВ"),
        ("[Объединенный] Интернет-издания", "Интернет-СМИ"),
        ("[Объединенный] Социальные сети", "Соцсети"),
        ("[Объединенный] Друзья", "Друзья/Коллеги"),
        ("[Объединенный] Газеты", "Газеты"),
        ("[Объединенный] Радио", "Радио"),
        ("[Объединенный] Журналы", "Журналы"),
        ("[Объединенный] Телеграм-каналы", "Telegram"),
    ]

    for col, label in channels:
        # Calculate Trust %
        trust_mask = df[col].isin(["Пользуюсь и доверяю", "Доверяю"])
        trust_pct = (trust_mask.sum() / total) * 100

        # Calculate Distrust %
        distrust_mask = df[col].isin(["Пользуюсь, но не доверяю", "Не доверяю"])
        distrust_pct = (distrust_mask.sum() / total) * 100

        results.append(
            {"channel": label, "trust": trust_pct, "distrust": distrust_pct}
        )

    # Convert to DataFrame and sort by Trust for better visualization
    res_df = pd.DataFrame(results).sort_values("trust", ascending=True)

    # Visualization
    apply_global_style()
    plt.figure(figsize=(10, 8))

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
        s=150,
        label="Не доверяю (%)",
        zorder=3,
    )

    # Plot Trust (Orange)
    plt.scatter(
        res_df["trust"],
        res_df["channel"],
        color=TRUST_COLOR,
        s=150,
        label="Доверяю (%)",
        zorder=3,
    )

    # Add text labels
    for _, row in res_df.iterrows():
        # Label for Distrust
        plt.text(
            row["distrust"] - 1.5,
            row["channel"],
            f"{row['distrust']:.1f}%",
            va="center",
            ha="right",
            color=DISTRUST_COLOR,
            fontweight="bold",
        )
        
        # Label for Trust
        plt.text(
            row["trust"] + 1.5,
            row["channel"],
            f"{row['trust']:.1f}%",
            va="center",
            ha="left",
            color=TRUST_COLOR,
            fontweight="bold",
        )

    plt.title(
        "Уровень доверия и недоверия к источникам информации\n(ТВ и Telegram — лидеры доверия среди медиа)",
        fontsize=16,
        pad=25,
        color=TEXT_COLOR,
    )
    plt.xlabel("Доля респондентов (%)", fontsize=12, labelpad=15)
    plt.xlim(0, max(res_df["trust"].max(), res_df["distrust"].max()) + 10) # Add some padding

    plt.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#494949")
    plt.grid(axis="x", color="#CCCCCC", alpha=0.5, linestyle="--")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "graph12.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 12 saved to: {output_path}")


if __name__ == "__main__":
    run()
