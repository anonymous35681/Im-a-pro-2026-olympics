# ruff: noqa: RUF001
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.projections.polar import PolarAxes

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style


def map_age_to_group(age: int) -> str:
    """Map individual age to age group."""
    if age < 18:
        return "Другое"

    for low, high, label in [(18, 24, "18-24"), (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"), (55, 64, "55-64")]:
        if low <= age <= high:
            return label
    return "65+"


def run() -> None:
    """Generate Graph 7: Demographics comparison spider chart."""
    logger.info("Generating Graph 7: Demographics comparison spider chart.")

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    # Process survey data
    df_survey = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")
    df_survey["age_group"] = df_survey["5. Ваш возраст (количество полных лет):"].apply(map_age_to_group)
    survey_pct = (df_survey[df_survey["age_group"] != "Другое"]["age_group"].value_counts(normalize=True) * 100).reindex(age_groups, fill_value=0)

    # Process demographics data
    df_demo = pd.read_csv(ROOT_DIR / "data" / "demographics_dataset.csv")
    df_demo = df_demo[~df_demo["age"].isin(["Всего", "моложе трудоспособного", "трудоспособного", "старше трудоспособного"])]
    df_demo["age_num"] = pd.to_numeric(df_demo["age"], errors="coerce")
    df_demo["age_group"] = df_demo["age_num"].apply(map_age_to_group)
    demo_pct = (df_demo[df_demo["age_group"] != "Другое"].groupby("age_group")["total_both"].sum() / df_demo[df_demo["age_group"] != "Другое"]["total_both"].sum() * 100).reindex(age_groups, fill_value=0)

    # Prepare data for radar chart
    survey_values = [*survey_pct.tolist(), survey_pct.iloc[0]]
    demo_values = [*demo_pct.tolist(), demo_pct.iloc[0]]
    angles = [*np.linspace(0, 2 * np.pi, len(age_groups), endpoint=False), 0]

    # Visualization
    apply_global_style()
    _, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    assert isinstance(ax, PolarAxes)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot both datasets
    for values, color, label in [(survey_values, NEON_CYAN, "Выборка (n=1000)"), (demo_values, "#FF0055", "Демография региона")]:
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.scatter(angles[:-1], values[:-1], color=color, s=40, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(age_groups, color="white", size=12)
    ax.set_rlabel_position(0)
    plt.yticks([5, 10, 15, 20, 25], ["5%", "10%", "15%", "20%", "25%"], color="grey", size=8)
    plt.ylim(0, 30)

    plt.title("Сравнение возрастной структуры:\nВыборка vs Демография региона", fontsize=14, pad=20, color=TEXT_COLOR)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True, facecolor="black", edgecolor="white")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "graph7.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Graph 7 saved to: {output_path}")


if __name__ == "__main__":
    run()
