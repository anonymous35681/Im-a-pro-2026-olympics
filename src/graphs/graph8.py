import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import TEXT_COLOR, apply_global_style


def map_age_to_group(age: int) -> str:
    """Map individual age to age group."""
    if age < 18:
        return "Другое"

    for low, high, label in [
        (18, 24, "18-24"),
        (25, 34, "25-34"),
        (35, 44, "35-44"),
        (45, 54, "45-54"),
        (55, 64, "55-64"),
    ]:
        if low <= age <= high:
            return label
    return "65+"


def run() -> None:
    """Generate Graph 8: News sources by age group (Sankey + Line charts)."""
    logger.info("Generating Graph 8: News sources by age group.")

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Map age to groups
    df["age_group"] = df["5. Ваш возраст (количество полных лет):"].apply(
        map_age_to_group
    )

    # Filter only target age groups
    df = df[df["age_group"].isin(age_groups)]

    # Define source columns and their display names
    sources = [
        ("[Телевидение]", "Телевидение"),
        ("[Интернет-издания]", "Интернет-издания"),
        ("[Друзья]", "Друзья, знакомые, родственники"),
        ("[Телеграм-каналы]", "Телеграм-каналы"),
        ("[Газеты]", "Печатные газеты"),
        ("[Радио]", "Радио"),
        ("[Журналы]", "Печатные журналы"),
    ]

    # Calculate usage percentages for each age group and source
    # Value "1. Пользуюсь" or 1 means user uses the source
    data = []
    for age_group in age_groups:
        group_df = df[df["age_group"] == age_group]
        group_size = len(group_df)

        row = {"age_group": age_group}
        for col, display_name in sources:
            # Count users who use this source (value is "1. Пользуюсь" or numeric 1)
            users = (
                group_df[col]
                .apply(lambda x: 1 if (x == "1. Пользуюсь" or x == 1) else 0)
                .sum()
            )
            percentage = (users / group_size) * 100
            row[display_name] = percentage
        data.append(row)

    plot_df = pd.DataFrame(data)
    plot_df = plot_df.set_index("age_group")

    logger.info(f"Data for plotting:\n{plot_df}")

    # Create figure with single subplot
    apply_global_style()
    _fig, ax = plt.subplots(figsize=(14, 8))

    # === Line Chart ===
    source_display_names = [display_name for _, display_name in sources]
    colors = [
        "#E57373",  # Телевидение - красный
        "#81C784",  # Интернет-издания - зеленый
        "#64B5F6",  # Друзья - синий
        "#FFD54F",  # Телеграм-каналы - желтый
        "#BA68C8",  # Печатные газеты - фиолетовый
        "#4DD0E1",  # Радио - бирюзовый
        "#FFB74D",  # Печатные журналы - оранжевый
    ]

    # Plot lines for each source
    x = np.arange(len(age_groups))
    for i, source_name in enumerate(source_display_names):
        values = plot_df[source_name].values
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=source_name,
            color=colors[i],
            alpha=0.9,
        )

    ax.set_xlabel("Возрастная группа", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("Процент пользователей (%)", fontsize=12, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(age_groups, fontsize=11)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        facecolor="black",
        edgecolor="white",
        fontsize=10,
    )

    ax.set_title(
        "Динамика предпочтений новостных источников\nпо возрастным группам",
        fontsize=14,
        pad=20,
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend

    # Save
    output_path = OUTPUT_DIR / "graph8.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Graph 8 saved to: {output_path}")


if __name__ == "__main__":
    run()
