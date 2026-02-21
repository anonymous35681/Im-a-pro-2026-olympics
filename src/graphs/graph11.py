import re
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR


def run() -> None:
    """Generate Graph 11: Diverging stacked bar chart of fake news frequency by source."""
    logger.info("Generating Graph 11: Fake news frequency by source (Diverging Bar).")

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
    # Ordered for the chart (Left to Right visually)
    # Daily -> Weekly -> (Center) -> Monthly -> Yearly -> Never
    freq_map = {
        "1. Практически ежедневно": "Ежедневно",
        "2. По нескольку раз в неделю": "Раз в неделю",
        "3. По нескольку раз в месяц": "Раз в месяц",
        "4. Несколько раз в год": "Раз в год",
        "5. Не сталкивался": "Не сталкивался",
    }

    # Categories to exclude from normalization
    exclude_cats = ["6. Затрудняюсь ответить", "#NULL!"]

    # Colors (Neon/Bright as requested)
    colors = {
        "Ежедневно": "#FF0055",  # Bright Red/Pink
        "Раз в неделю": "#FF8800",  # Bright Orange
        "Раз в месяц": "#FFFF00",  # Bright Yellow
        "Раз в год": "#00FF55",  # Bright Green
        "Не сталкивался": "#00FFFF",  # Cyan
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

        # Calculate valid total (excluding "Hard to say")
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

    # Sort by "Daily + Weekly" (Frequency) descending,
    # but put Radio at bottom if N is small?
    # The reference image has Radio (highest freq) at bottom.
    # I'll sort by Frequency but maybe the user wants the exact order from image?
    # The user said "approximately like this". Frequency sort is standard.
    # However, to match the "Reference" feel where Radio is an outlier/small base,
    # I will sort by N (Count) descending?
    # Telegram(207), Internet(305), TV(258)... Social(567) is max.
    # Let's stick to Frequency sort (Daily + Weekly) as it highlights the "problematic" channels.
    plot_df["High_Freq_Sum"] = plot_df["Ежедневно"] + plot_df["Раз в неделю"]
    plot_df = plot_df.sort_values(
        "High_Freq_Sum", ascending=True
    )  # Ascending for horizontal bar (top is highest)

    fig = go.Figure()

    # Left side: Daily, Weekly
    left_cats = ["Раз в неделю", "Ежедневно"]  # Inner to Outer

    for cat in left_cats:
        values = plot_df[cat]
        fig.add_trace(
            go.Bar(
                y=plot_df["Source"],
                x=-values,
                name=cat,
                orientation="h",
                marker_color=colors[cat],
                text=values.apply(lambda x: f"{x:.0f}%" if x > 3 else ""),
                textposition="auto",
                customdata=values,
                hovertemplate=f"<b>{cat}</b><br>%{{y}}: %{{customdata:.1f}}%<extra></extra>",
                # Ensure text is displayed for negative bars
                insidetextanchor="middle",
            )
        )

    # Right side: Monthly, Yearly, Never
    right_cats = ["Раз в месяц", "Раз в год", "Не сталкивался"]  # Inner to Outer

    for cat in right_cats:
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
                customdata=values,
                hovertemplate=f"<b>{cat}</b><br>%{{y}}: %{{customdata:.1f}}%<extra></extra>",
            )
        )

    # X-axis setup
    tick_vals = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
    tick_text = [str(abs(x)) + "%" if x != 0 else "0%" for x in tick_vals]

    fig.update_layout(
        title={
            "text": "Частота столкновения с фейками по источникам<br><span style='font-size: 16px; color: #CCCCCC'>(Соцсети и Телеграм — главные каналы распространения)</span>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "color": "#FFFFFF"},
        },
        barmode="relative",
        xaxis={
            "title": "",
            "tickvals": tick_vals,
            "ticktext": tick_text,
            "gridcolor": "#333333",
            "zeroline": True,
            "zerolinecolor": "#FFFFFF",
            "zerolinewidth": 1,
            "tickfont": {"size": 12, "color": "#FFFFFF"},
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
        },
        paper_bgcolor="#000000",  # Pure black background like reference
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
