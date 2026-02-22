import re

import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR


def run() -> None:
    """Generate Graph 11: 100% Stacked bar chart of fake news frequency by source."""
    logger.info("Generating Graph 11: Fake news frequency by source (Stacked Bar).")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Column definitions
    freq_col = "19. Скажите, сталкивались ли Вы за последний год с подобного рода сообщениями, и если да, то как часто (ОДИН ответ)?"

    source_cols = [
        "20. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
        "21. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
        "22. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
        "23. Скажите, где Вы чаще всего за последний год сталкивались с выдуманными, фейковыми новостями (не более ТРЕХ ответов)?",
    ]

    # Mappings
    sources_map = {
        "8. Телеграм-каналы": "Телеграм",
        "2. Интернет-издания (газеты, журналы, информационные порталы)": "Интернет-СМИ",
        "1. Телевидение": "ТВ",
        "7. Журналы": "Журналы",
        "5. Газеты": "Газеты",
        "3. Социальные сети": "Соцсети",
        "4. Друзья, родные, знакомые": "Окружение",
        "6. Радио": "Радио",
    }

    # Frequency scale mapping (Value -> Category Name)
    # Ordered Left to Right: Daily -> Weekly -> Monthly -> Yearly -> Never -> Hard to say
    freq_map = {
        "1. Практически ежедневно": "Ежедневно",
        "2. По нескольку раз в неделю": "По несколько раз в неделю",
        "3. По нескольку раз в месяц": "По несколько раз в месяц",
        "4. Несколько раз в год": "По несколько раз в год",
        "5. Не сталкивался": "Не сталкивался",
        "6. Затрудняюсь ответить": "Затрудняюсь ответить",
    }

    # Order for stacking (Left to Right)
    stack_order = [
        "Ежедневно",
        "По несколько раз в неделю",
        "По несколько раз в месяц",
        "По несколько раз в год",
        "Не сталкивался",
        "Затрудняюсь ответить",
    ]

    # Colors (Neon/Bright matching reference)
    colors = {
        "Ежедневно": "#FF0055",  # Bright Pink/Red
        "По несколько раз в неделю": "#FF8800",  # Bright Orange
        "По несколько раз в месяц": "#FFFF00",  # Bright Yellow
        "По несколько раз в год": "#00FF55",  # Bright Green
        "Не сталкивался": "#00FFFF",  # Cyan
        "Затрудняюсь ответить": "#888888",  # Gray
    }

    data_rows = []

    for source_full, source_short in sources_map.items():
        # Filter users who selected this source in ANY of the source columns
        mask_list = []
        for col in source_cols:
            if col not in df.columns:
                continue
            series = df[col].astype(str)
            mask_list.append(
                series.str.contains(re.escape(source_full), regex=True, na=False)
            )

        if not mask_list:
            continue
        final_mask = pd.concat(mask_list, axis=1).any(axis=1)
        subset = df[final_mask]

        if len(subset) == 0:
            continue

        counts = subset[freq_col].value_counts()

        # Calculate total (including all categories)
        valid_total = 0
        valid_counts = {}
        for original_cat, short_cat in freq_map.items():
            val = counts.get(original_cat, 0)
            valid_counts[short_cat] = val
            valid_total += val

        if valid_total == 0:
            continue

        row_data = {"Source": source_short, "Total": valid_total, "N": len(subset)}
        for short_cat, val in valid_counts.items():
            row_data[short_cat] = (val / valid_total) * 100

        data_rows.append(row_data)

    # Convert to DataFrame
    plot_df = pd.DataFrame(data_rows)

    # Sort by "Daily + Weekly" (Frequency) descending
    plot_df["High_Freq_Sum"] = plot_df["Ежедневно"] + plot_df["По несколько раз в неделю"]
    plot_df = plot_df.sort_values(
        "High_Freq_Sum", ascending=True
    )  # Ascending for horizontal bar (top is highest)

    fig = go.Figure()

    # Add traces in stack order
    for cat in stack_order:
        values = plot_df[cat]
        fig.add_trace(
            go.Bar(
                y=plot_df["Source"],
                x=values,
                name=cat,
                orientation="h",
                marker_color=colors[cat],
                text=values.apply(lambda x: f"{x:.0f}%" if x > 3 else ""),
                textposition="auto",
                hovertemplate=f"<b>{cat}</b><br>%{{y}}: %{{x:.1f}}%<extra></extra>",
                # Add thin black border for separation
                marker_line_color="black",
                marker_line_width=1.5,
            )
        )

    fig.update_layout(
        title={
            "text": "Частота столкновения с фейками по источникам<br><span style='font-size: 16px; color: #CCCCCC'>(Соцсети и Телеграм — главные каналы распространения)</span>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "color": "#FFFFFF"},
        },
        barmode="stack",  # Standard stacked bar
        xaxis={
            "title": "",
            "tickvals": [0, 20, 40, 60, 80, 100],
            "ticktext": ["0%", "20%", "40%", "60%", "80%", "100%"],
            "gridcolor": "#333333",
            "zeroline": False,
            "tickfont": {"size": 12, "color": "#FFFFFF"},
            "range": [0, 100],  # Force 0-100 scale
        },
        yaxis={
            "title": "",
            "tickfont": {"size": 14, "color": "#FFFFFF"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": "#FFFFFF", "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
            # Reverse legend order to match visual stacking (Left-to-Right)
            "traceorder": "normal",
        },
        paper_bgcolor="#000000",  # Pure black background
        plot_bgcolor="#000000",
        margin={"l": 120, "r": 50, "t": 100, "b": 100},
        height=700,
        width=1200,
    )

    output_path = OUTPUT_DIR / "graph11.png"
    fig.write_image(output_path, scale=2)
    logger.success(f"Graph 11 saved to {output_path}")


if __name__ == "__main__":
    run()
