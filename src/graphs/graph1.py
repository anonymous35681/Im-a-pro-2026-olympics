# ruff: noqa: RUF001
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style


def run() -> None:
    """Generate Theme 3: Socio-demographic profile heatmap."""
    logger.info("Generating Graph 1: Socio-demographic profile heatmap.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Preprocessing
    # 1. Age cleaning and binning
    df["5. Ваш возраст (количество полных лет):"] = pd.to_numeric(
        df["5. Ваш возраст (количество полных лет):"], errors="coerce"
    )
    df = df.dropna(subset=["5. Ваш возраст (количество полных лет):"])

    df["age_group"] = pd.cut(
        df["5. Ваш возраст (количество полных лет):"],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=False,
    )

    # 2. Frequency mapping
    df["freq_score"] = df[
        "19. Скажите, сталкивались ли Вы за последний год с подобного рода сообщениями, и если да, то как часто (ОДИН ответ)?"
    ].map(
        {
            "1. Практически ежедневно": 4,
            "2. По нескольку раз в неделю": 3,
            "3. По нескольку раз в месяц": 2,
            "4. Несколько раз в год": 1,
            "5. Не сталкивался": 0,
        }
    )

    # Drop rows where frequency is unknown
    df = df.dropna(subset=["freq_score"])

    # 3. Education cleaning (shorten labels for display)
    df["education"] = df["6. Уровень Вашего образования (ОДИН ответ):"].map(
        {
            "1. Основное (до 9 классов), среднее (до 10—11 классов)": "Среднее общее",
            "2. Начальное профессиональное (училище)": "Нач. проф.",
            "3. Среднее профессиональное (техникум, колледж)": "Среднее проф.",
            "4. Незаконченное высшее, высшее, ученая степень": "Высшее/Незак. высшее",
        }
    )

    # Pivot for heatmap
    pivot_df = df.pivot_table(
        index="age_group", columns="education", values="freq_score", aggfunc="mean"
    )

    # Sort columns
    pivot_df = pivot_df.reindex(
        columns=["Среднее общее", "Нач. проф.", "Среднее проф.", "Высшее/Незак. высшее"]
    )

    # Visualization
    apply_global_style()
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap=sns.dark_palette(NEON_CYAN, as_cmap=True),
        cbar_kws={"label": "Интенсивность столкновения (0-4)"},
        linewidths=0.5,
    )

    plt.title(
        "Профиль жертвы фейков: Возраст vs Образование\n(Чем выше число, тем чаще сталкиваются)",
        fontsize=16,
        pad=20,
        color=TEXT_COLOR,
    )
    plt.xlabel("Уровень образования", fontsize=12, labelpad=10)
    plt.ylabel("Возрастная группа", fontsize=12, labelpad=10)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save the plot
    output_path = OUTPUT_DIR / "graph1.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 1 saved to: {output_path}")


if __name__ == "__main__":
    run()
