# ruff: noqa: RUF001
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style


def run() -> None:
    """Generate Theme 1: Media consumption vs trust dumbbell chart."""
    logger.info("Generating Graph 2: Media consumption vs trust dumbbell chart.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    results = []
    total = len(df)

    # Using literal strings for column names as per "no python variables" preference
    for col, label in [
        ("[Объединенный] Телевидение", "ТВ"),
        ("[Объединенный] Интернет-издания", "Интернет-СМИ"),
        ("[Объединенный] Социальные сети", "Соцсети"),
        ("[Объединенный] Друзья", "Друзья/Коллеги"),
        ("[Объединенный] Газеты", "Газеты"),
        ("[Объединенный] Радио", "Радио"),
        ("[Объединенный] Журналы", "Журналы"),
        ("[Объединенный] Телеграм-каналы", "Telegram"),
    ]:
        usage_mask = df[col].isin(
            ["Пользуюсь", "Пользуюсь и доверяю", "Пользуюсь, но не доверяю", "Доверяю"]
        )
        trust_mask = df[col].isin(["Пользуюсь и доверяю", "Доверяю"])

        usage_pct = (usage_mask.sum() / total) * 100
        trust_pct = (trust_mask.sum() / total) * 100

        results.append({"channel": label, "usage": usage_pct, "trust": trust_pct})

    res_df = pd.DataFrame(results).sort_values("usage", ascending=True)

    # Visualization
    apply_global_style()
    plt.figure(figsize=(10, 8))

    plt.hlines(
        y=res_df["channel"],
        xmin=res_df["trust"],
        xmax=res_df["usage"],
        color="white",
        alpha=0.3,
        linewidth=2,
    )

    plt.scatter(
        res_df["usage"],
        res_df["channel"],
        color=NEON_CYAN,
        s=150,
        label="Потребление (%)",
        zorder=3,
    )
    plt.scatter(
        res_df["trust"],
        res_df["channel"],
        color="#FF0055",
        s=150,
        label="Доверие (%)",
        zorder=3,
    )

    for _, row in res_df.iterrows():
        plt.text(
            row["usage"] + 1.5,
            row["channel"],
            f"{row['usage']:.1f}%",
            va="center",
            color=NEON_CYAN,
            fontweight="bold",
        )
        plt.text(
            row["trust"] - 1.5,
            row["channel"],
            f"{row['trust']:.1f}%",
            va="center",
            ha="right",
            color="#FF0055",
            fontweight="bold",
        )

        gap = row["usage"] - row["trust"]
        plt.text(
            (row["usage"] + row["trust"]) / 2,
            row["channel"],
            f"Δ {gap:.1f}",
            va="bottom",
            ha="center",
            color="white",
            fontsize=9,
            alpha=0.7,
        )

    plt.title(
        "Медиа-ландшафт: Потребление vs Доверие\n(Разрыв в легитимности источников)",
        fontsize=16,
        pad=25,
        color=TEXT_COLOR,
    )
    plt.xlabel("Процент респондентов (%)", fontsize=12, labelpad=15)
    plt.xlim(0, 100)

    plt.legend(loc="lower right", frameon=True, facecolor="black", edgecolor="white")
    plt.grid(axis="x", color="white", alpha=0.1, linestyle="--")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "graph2.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 2 saved to: {output_path}")


if __name__ == "__main__":
    run()
