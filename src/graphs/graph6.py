# ruff: noqa: RUF001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.projections.polar import PolarAxes

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style


def run() -> None:
    """Generate Spider Chart comparing fake news sources by encounter frequency."""
    logger.info("Generating Graph 6: Spider Chart of Fake News Sources.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    col19 = "19. Скажите, сталкивались ли Вы за последний год с подобного рода сообщениями, и если да, то как часто (ОДИН ответ)?"
    source_cols = [
        "20. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
        "21. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
        "22. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
        "23. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
    ]

    # Map frequency to categories
    freq_map = {
        "1. Практически ежедневно": "Часто (День/Неделя)",
        "2. По нескольку раз в неделю": "Часто (День/Неделя)",
        "3. По нескольку раз в месяц": "Редко (Месяц/Год)",
        "4. Несколько раз в год": "Редко (Месяц/Год)",
    }

    df["freq_group"] = df[col19].map(freq_map)
    
    # Filter only those who encountered fakes and have a group
    df = df.dropna(subset=["freq_group"])

    # Define main sources mapping
    source_mapping = {
        "1. Телевидение": "ТВ",
        "2. Интернет-издания (газеты, журналы, информационные порталы)": "Интернет-СМИ",
        "3. Социальные сети": "Соцсети",
        "4. Друзья, родные, знакомые": "Окружение",
        "5. Газеты": "Газеты",
        "6. Радио": "Радио",
        "7. Журналы": "Журналы",
        "8. Телеграм-каналы": "Телеграм",
    }
    
    categories = list(source_mapping.values())
    groups = ["Часто (День/Неделя)", "Редко (Месяц/Год)"]
    
    # Calculate stats
    stats = {}
    for group in groups:
        group_df = df[df["freq_group"] == group]
        group_size = len(group_df)

        counts = dict.fromkeys(categories, 0)
        for col in source_cols:
            val_counts = group_df[col].value_counts()
            for raw_val, cat_label in source_mapping.items():
                if raw_val in val_counts:
                    counts[cat_label] += val_counts[raw_val]

        # Normalize to percentages
        percentages = {cat: (counts[cat] / group_size) * 100 for cat in categories}
        stats[group] = percentages

    # Sort categories by total percentage (descending) for better visualization
    category_totals = {cat: sum(stats[group][cat] for group in groups) for cat in categories}
    sorted_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)

    # Reorder stats based on sorted categories
    sorted_stats = {}
    for group in groups:
        sorted_stats[group] = [stats[group][cat] for cat in sorted_categories]

    stats = sorted_stats
    categories = sorted_categories

    # Radar chart setup
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the loop
    for group in stats:
        stats[group].append(stats[group][0])
    angles.append(angles[0])

    # Visualization
    apply_global_style()
    _, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    assert isinstance(ax, PolarAxes)

    # Plot each group
    colors = [NEON_CYAN, "#FF0055"]
    for i, (group, values) in enumerate(stats.items()):
        ax.plot(angles, values, color=colors[i], linewidth=2, label=group)
        ax.fill(angles, values, color=colors[i], alpha=0.25)
        # Add points at each vertex
        ax.scatter(angles[:-1], values[:-1], color=colors[i], s=40, zorder=5)

    # Fix axis to start from top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, color=TEXT_COLOR)

    # Add grid lines and labels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20%", "40%", "60%", "80%"], color="grey", size=8)
    plt.ylim(0, 100)

    plt.title(
        "Источники фейковых новостей\nв зависимости от частоты столкновения",
        fontsize=14,
        pad=20,
        color=TEXT_COLOR,
    )

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True, facecolor="black", edgecolor="white")

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "graph6.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 6 (Spider Chart) saved to: {output_path}")


if __name__ == "__main__":
    run()
