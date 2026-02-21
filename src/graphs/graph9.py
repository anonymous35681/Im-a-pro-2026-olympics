import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR


def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convert hex color to rgba format with given alpha."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def run() -> None:
    """Generate Graph 9: News source preferences by settlement type (Sankey diagram)."""
    logger.info("Generating Graph 9: News source preferences by settlement type.")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Define settlement type labels
    settlement_mapping = {
        "1. Саранск": "Саранск",
        "2. Города и пгт": "Города",
        "3. Села": "Сёла",
    }

    # Define source columns with their display names
    sources = [
        ("[Объединенный] Телевидение", "Телевидение"),
        ("[Объединенный] Интернет-издания", "Интернет-издания"),
        ("[Объединенный] Социальные сети", "Социальные сети"),
        ("[Объединенный] Друзья", "Друзья, знакомые, родственники"),
        ("[Объединенный] Газеты", "Печатные газеты"),
        ("[Объединенный] Радио", "Радио"),
        ("[Объединенный] Журналы", "Печатные журналы"),
        ("[Объединенный] Телеграм-каналы", "Телеграм-каналы"),
    ]

    # Prepare data for Sankey diagram
    settlement_types = ["Саранск", "Города", "Сёла"]
    source_names = [display_name for _, display_name in sources]

    # Nodes: settlement types + source names
    all_nodes = settlement_types + source_names
    node_dict = {node: i for i, node in enumerate(all_nodes)}

    # Calculate flows (people who use each source in each settlement type)
    # User values: "Пользуюсь", "Пользуюсь и доверяю", or any value containing "Пользуюсь"
    source = []
    target = []
    value = []

    for settlement_original, settlement_display in settlement_mapping.items():
        # Get data for this settlement type
        settlement_df = df[df["3. Тип населенного пункта"] == settlement_original]

        for col, source_display in sources:
            # Count users who use this source
            users = (
                settlement_df[col]
                .apply(
                    lambda x: (
                        1
                        if pd.notna(x)
                        and ("Пользуюсь" in str(x) or "Доверяю" in str(x))
                        else 0
                    )
                )
                .sum()
            )

            if users > 0:
                source.append(node_dict[settlement_display])
                target.append(node_dict[source_display])
                value.append(users)

    # Color scheme - neon colors on dark background (matching other graphs)
    settlement_colors = ["#FF0055", "#00FFFF", "#00FF00"]  # Саранск, Города, Сёла
    source_colors = [
        "#FF0055",  # Телевидение - неоновый розовый
        "#00FFFF",  # Интернет-издания - неоновый циан
        "#00FF00",  # Социальные сети - неоновый зеленый
        "#FFFF00",  # Друзья - неоновый желтый
        "#FF00FF",  # Печатные газеты - неоновый пурпурный
        "#00CCFF",  # Радио - неоновый голубой
        "#FF6600",  # Печатные журналы - неоновый оранжевый
        "#9900FF",  # Телеграм-каналы - неоновый фиолетовый
    ]

    node_color = settlement_colors + source_colors

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "#FFFFFF", "width": 1},
                    "label": all_nodes,
                    "color": node_color,
                },
                link={
                    "source": source,
                    "target": target,
                    "value": value,
                    "color": [
                        hex_to_rgba(node_color[src], alpha=0.4) for src in source
                    ],  # Use source node color with 40% opacity
                },
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Предпочтения новостных источников<br>по типам населенных пунктов",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#FFFFFF"},
        },
        font={"size": 12, "color": "#FFFFFF"},
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        height=700,
    )

    # Save as PNG
    output_path = OUTPUT_DIR / "graph9.png"
    fig.write_image(output_path, width=1400, height=800, scale=2)
    logger.success(f"Graph 9 PNG saved to: {output_path}")


if __name__ == "__main__":
    run()
