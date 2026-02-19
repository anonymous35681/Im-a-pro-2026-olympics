# ruff: noqa: RUF001
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style


def run() -> None:
    """Generate Theme 4: Perceived media-literacy gap (Lollipop chart)."""
    logger.info("Generating Graph 4: Perceived media-literacy gap.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    countries = [
        "Великобритания",
        "Германия",
        "Китай",
        "Латвия",
        "Польша",
        "Россия",
        "Северная Корея",
        "Сербия",
        "США",
        "Турция",
        "Украина",
        "Финляндия",
        "Франция",
        "Эстония",
    ]

    cannot_labels = [
        "3. В большинстве случаев не могут отличить «фейковые новости»",
        "4. Почти всегда не могут отличить «фейковые новости»",
    ]
    can_labels = [
        "1. Практически всегда могут отличить «фейковые новости»",
        "2. В большинстве случаев могут отличить «фейковые новости»",
    ]

    results = []
    for country in countries:
        count_cannot = df[country].isin(cannot_labels).sum()
        count_can = df[country].isin(can_labels).sum()

        if (count_can + count_cannot) > 0:
            index_val = (count_cannot / (count_can + count_cannot)) * 100
        else:
            index_val = 0

        results.append({"country": country, "index": index_val})

    res_df = pd.DataFrame(results).sort_values("index", ascending=True)

    # Visualization
    apply_global_style()
    plt.figure(figsize=(10, 10))

    plt.hlines(
        y=res_df["country"], xmin=0, xmax=res_df["index"], color="white", alpha=0.3
    )
    plt.scatter(
        res_df["index"],
        res_df["country"],
        color=NEON_CYAN,
        s=200,
        edgecolors="white",
        zorder=3,
    )

    for _, row in res_df.iterrows():
        plt.text(
            row["index"] + 1,
            row["country"],
            f"{row['index']:.1f}",
            va="center",
            color=NEON_CYAN,
            fontweight="bold",
            fontsize=12,
        )

    plt.title(
        "География 'информационных пузырей'\n(Восприятие неспособности жителей стран отличать фейки)",
        fontsize=16,
        pad=30,
        color=TEXT_COLOR,
    )
    plt.xlabel(
        "Индекс закрытости (Чем выше, тем ниже доверие к медиа-грамотности нации)",
        fontsize=12,
        labelpad=15,
    )

    plt.xlim(0, 100)
    plt.grid(axis="x", color="white", alpha=0.1, linestyle="--")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "graph4.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 4 saved to: {output_path}")


if __name__ == "__main__":
    run()
