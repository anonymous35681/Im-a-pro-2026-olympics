import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR
from style import BACKGROUND_COLOR, NEON_CYAN, TEXT_COLOR, apply_global_style


def run() -> None:
    """
    Generate Graph 8: Usage of information sources by age group (Bubble Chart).
    """
    logger.info("Generating Graph 8: Information sources usage by age group.")

    # Load dataset
    data_path = ROOT_DIR / "data" / "origin_dataset.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Preprocessing: Age
    # Ensure numeric and drop NaNs
    df["age"] = pd.to_numeric(
        df["5. Ваш возраст (количество полных лет):"], errors="coerce"
    )
    df = df.dropna(subset=["age"])

    # Define age bins and labels
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)

    # Define source columns mapping
    # Assuming the 'Unified' columns are the ones to use based on analysis
    source_mapping = {
        "[Объединенный] Телевидение": "Телевидение",
        "[Объединенный] Социальные сети": "Социальные сети",
        "[Объединенный] Интернет-издания": "Интернет-издания",
        "[Объединенный] Друзья": "Друзья, знакомые, родственники",
        "[Объединенный] Газеты": "Печатные газеты",
        "[Объединенный] Журналы": "Печатные журналы",
        "[Объединенный] Телеграм-каналы": "Telegram-каналы",
        "[Объединенный] Радио": "Радио",
    }

    # Calculate usage percentages
    results = []

    # Helper function to check usage
    def is_user(val):
        if pd.isna(val):
            return False
        val_str = str(val)
        return "Пользуюсь" in val_str or "Доверяю" in val_str

    # 1. Calculate for each age group
    for group in age_labels:
        group_df = df[df["age_group"] == group]
        total_in_group = len(group_df)

        if total_in_group == 0:
            continue

        for col, label in source_mapping.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataset.")
                continue

            users_count = group_df[col].apply(is_user).sum()
            percentage = (users_count / total_in_group) * 100

            results.append(
                {"Age Group": group, "Source": label, "Percentage": percentage}
            )

    # 2. Calculate for "Total" (All ages)
    total_df = df
    total_count = len(total_df)
    if total_count > 0:
        for col, label in source_mapping.items():
            if col not in df.columns:
                continue

            users_count = total_df[col].apply(is_user).sum()
            percentage = (users_count / total_count) * 100

            results.append(
                {
                    "Age Group": "Всего (18-65+)",
                    "Source": label,
                    "Percentage": percentage,
                }
            )

    plot_df = pd.DataFrame(results)

    # Order of Sources (Y-axis) - can be sorted by Total percentage or manual
    # Let's sort by Total percentage descending for better readability
    total_percentages = plot_df[plot_df["Age Group"] == "Всего (18-65+)"][
        ["Source", "Percentage"]
    ]
    sorted_sources = total_percentages.sort_values("Percentage", ascending=True)[
        "Source"
    ].tolist()

    # Visualization
    apply_global_style()
    _, ax = plt.subplots(figsize=(12, 10))

    # X-axis order
    x_order = [*age_labels, "Всего (18-65+)"]

    # Y-axis order
    y_order = sorted_sources

    # Plot lines and bubbles
    # We iterate over age groups (X-axis) and draw a vertical line
    for i, group in enumerate(x_order):
        group_data = plot_df[plot_df["Age Group"] == group]

        # Draw vertical line
        # We need numeric coordinates for Y. Let's map sources to 0..N
        # But standard plot uses data coordinates.
        # Let's just plot the line from min to max source index

        # Filter data for this group
        # Map sources to indices
        group_data = group_data.set_index("Source").reindex(y_order).reset_index()

        # Y indices
        y_indices = range(len(y_order))

        # Plot vertical line
        ax.vlines(
            x=i,
            ymin=0,
            ymax=len(y_order) - 1,
            colors=NEON_CYAN,
            linestyles="-",
            linewidth=1.5,
            alpha=0.6,
        )

        # Plot bubbles
        # Size depends on percentage (scaled for visibility)
        # Reduced overall size for better fit
        sizes = [(row.Percentage**1.8) * 1.9 for row in group_data.itertuples()]

        # Determine colors for highlighting
        # Highlight social media for 18-24 and TV for 45-54
        face_colors = []
        edge_colors = []
        line_widths = []

        for row in group_data.itertuples():
            source = row.Source
            age_group = group

            # Highlight conditions: 18-24 + Социальные сети OR 45-54 + Телевидение
            if (age_group == "18-24" and source == "Социальные сети") or (
                age_group == "45-54" and source == "Телевидение"
            ):
                face_colors.append("#F0DC58")  # Highlight color (yellow)
                edge_colors.append("#F0DC58")
                line_widths.append(3)
            else:
                face_colors.append(BACKGROUND_COLOR)
                edge_colors.append(NEON_CYAN)
                line_widths.append(2)

        # Scatter plot for this group
        # Use simple scatter with mapped Y coordinates
        ax.scatter(
            x=[i] * len(group_data),
            y=y_indices,
            s=sizes,
            c=face_colors,
            edgecolors=edge_colors,
            linewidths=line_widths,
            zorder=3,
        )

        # Add text inside bubbles (only for highlighted yellow bubbles and "Total" column)
        for y_idx, row in enumerate(group_data.itertuples()):
            source = row.Source
            age_group = group
            pct = row.Percentage

            # Show percentage for: 1) highlighted bubbles (yellow ones), 2) Total column
            if (age_group == "18-24" and source == "Социальные сети") or (
                age_group == "45-54" and source == "Телевидение"
            ) or age_group == "Всего (18-65+)":
                ax.text(
                    x=i,
                    y=y_idx,
                    s=f"{pct:.0f}%",
                    color=TEXT_COLOR,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    zorder=4,
                )

    # Set ticks and labels
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, fontsize=12)

    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order, fontsize=12)

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Grid (optional, maybe vertical grid is redundant since we have lines, horizontal could help)
    # ax.grid(False) # Clean look

    # Title
    # Calculate key insight: Which source is most popular overall?
    title_text = "Самые востребованные источники информации по возрасту."

    plt.title(
        title_text,
        fontsize=16,
        pad=30,
        color=TEXT_COLOR,
        loc="center",
        fontweight="bold",
    )

    # Adjust layout with proper margins to prevent cutoff
    plt.subplots_adjust(left=0.25, right=0.95, top=0.88, bottom=0.08)

    # Add footer
    plt.annotate(
        "Источник: Опрос Мордовского государственного университет имени Н. П. Огарёва",
        xy=(0, 0),
        xycoords="figure points",
        fontsize=10,
        color="#494949",
        xytext=(10, 3),
    )

    # Save
    output_path = OUTPUT_DIR / "graph8.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Graph 8 saved to: {output_path}")


if __name__ == "__main__":
    run()
