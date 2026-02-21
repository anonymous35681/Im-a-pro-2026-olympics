import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from config import OUTPUT_DIR, ROOT_DIR


def run() -> None:
    """Generate Graph 9: Age distribution comparison (Rosstat vs Survey) for City and Village."""
    logger.info("Generating Graph 9: Age distribution comparison (Rosstat vs Survey).")

    # 1. Load Data
    survey_path = ROOT_DIR / "data" / "origin_dataset.csv"
    demo_path = ROOT_DIR / "data" / "demographics_dataset.csv"

    if not survey_path.exists() or not demo_path.exists():
        logger.error("Data files not found.")
        return

    df_survey = pd.read_csv(survey_path)
    df_demo = pd.read_csv(demo_path)

    # 2. Process Survey Data
    # Map settlements
    settlement_map = {
        "1. Саранск": "City",
        "2. Города и пгт": "City",
        "3. Села": "Village",
    }
    df_survey["Settlement"] = df_survey["3. Тип населенного пункта"].map(settlement_map)

    # Clean Age
    # Ensure age is numeric, coerce errors to NaN
    df_survey["Age"] = pd.to_numeric(
        df_survey["5. Ваш возраст (количество полных лет):"], errors="coerce"
    )
    df_survey = df_survey.dropna(subset=["Age"])
    df_survey = df_survey[df_survey["Age"] >= 18]  # Ensure 18+

    # Define bins
    bins = [18, 24, 34, 44, 54, 64, 150]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    df_survey["AgeGroup"] = pd.cut(
        df_survey["Age"], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # Calculate percentages for Survey
    survey_counts = (
        df_survey.groupby(["Settlement", "AgeGroup"], observed=True)
        .size()
        .reset_index(name="Count")
    )

    # Calculate totals per settlement
    survey_totals = df_survey.groupby("Settlement").size()

    survey_data = []
    for settlement in ["City", "Village"]:
        subset = survey_counts[survey_counts["Settlement"] == settlement].copy()
        total = survey_totals.get(settlement, 1)  # avoid div by zero
        subset["Percentage"] = (subset["Count"] / total) * 100
        subset["Source"] = "Опрос"
        subset["Type"] = settlement
        survey_data.append(subset)

    df_survey_final = pd.concat(survey_data)

    # 3. Process Rosstat Data
    # Columns: age, urban_both (City), rural_both (Village)
    # Filter 18+
    # 'age' column needs cleaning (it has "Total", "<1", "100+", etc.)

    # Helper to clean age string
    def clean_demo_age(x):
        if str(x).isdigit():
            return int(x)
        if x == "100 и более":
            return 100
        return None

    df_demo["AgeNumeric"] = df_demo["age"].apply(clean_demo_age)
    df_demo_clean = df_demo.dropna(subset=["AgeNumeric"])
    df_demo_clean = df_demo_clean[df_demo_clean["AgeNumeric"] >= 18]

    df_demo_clean["AgeGroup"] = pd.cut(
        df_demo_clean["AgeNumeric"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    # Group Rosstat
    # We need to sum the counts for each bin
    rosstat_data = []

    # City (urban_both)
    city_grouped = (
        df_demo_clean.groupby("AgeGroup", observed=True)["urban_both"]
        .sum()
        .reset_index(name="Count")
    )
    city_total = city_grouped["Count"].sum()
    city_grouped["Percentage"] = (city_grouped["Count"] / city_total) * 100
    city_grouped["Source"] = "Росстат"
    city_grouped["Type"] = "City"
    rosstat_data.append(city_grouped)

    # Village (rural_both)
    village_grouped = (
        df_demo_clean.groupby("AgeGroup", observed=True)["rural_both"]
        .sum()
        .reset_index(name="Count")
    )
    village_total = village_grouped["Count"].sum()
    village_grouped["Percentage"] = (village_grouped["Count"] / village_total) * 100
    village_grouped["Source"] = "Росстат"
    village_grouped["Type"] = "Village"
    rosstat_data.append(village_grouped)

    df_rosstat_final = pd.concat(rosstat_data)

    # Combine all data
    df_final = pd.concat([df_survey_final, df_rosstat_final])

    # Pivot to get City and Village percentages side-by-side
    df_pivot = df_final.pivot_table(
        index=["AgeGroup", "Source"],
        columns="Type",
        values="Percentage",
        observed=False,
    ).reset_index()

    # Calculate Skew (Visual Average)
    # Skew = (Village - City) / 2
    # If Skew < 0: Visual point is on Left (City side)
    # If Skew > 0: Visual point is on Right (Village side)
    df_pivot["Skew"] = (df_pivot["Village"] - df_pivot["City"]) / 2

    # 4. Visualization
    fig = go.Figure()

    # Colors
    color_map = {"Росстат": "#00FFFF", "Опрос": "#FF0055"}  # Cyan, Pink/Red

    # Order of age groups for Y-axis
    y_order = labels  # ["18-24", "25-34", ...]

    # Add traces
    for source in ["Росстат", "Опрос"]:
        subset = df_pivot[df_pivot["Source"] == source]
        fig.add_trace(
            go.Scatter(
                x=subset["Skew"],
                y=subset["AgeGroup"],
                mode="markers",
                name=source,
                marker={
                    "color": color_map[source],
                    "size": 20,
                    "line": {"width": 2, "color": "white"},
                },
                showlegend=True,
                hoverinfo="x+y+name",
                hovertemplate="%{y}<br>"
                + source
                + "<br>Сдвиг: %{x:.1f}%<br>(< 0: Город, > 0: Село)<extra></extra>",
            )
        )

    # Calculate max range for symmetry
    # We use Skew values now, which are smaller than raw percentages
    max_val = df_pivot["Skew"].abs().max()
    limit = (int(max_val / 2) + 1) * 2  # Round up to nearest 2
    if limit < 5:
        limit = 5  # Minimum range
    tick_vals = list(range(-limit, limit + 1, 1 if limit <= 5 else 2))
    tick_text = [str(abs(x)) for x in tick_vals]

    # Update Layout
    fig.update_layout(
        title={
            "text": "Демографический перекос: Город vs Село<br>(Росстат vs Опрос)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24, "color": "#FFFFFF"},
        },
        font={"size": 14, "color": "#FFFFFF"},
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        margin={"l": 50, "r": 50, "t": 120, "b": 100},
        height=700,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 16},
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={
            "tickmode": "array",
            "tickvals": tick_vals,
            "ticktext": tick_text,
            "title": "Сдвиг (%)",
            "showgrid": True,
            "gridcolor": "#333333",
            "zeroline": True,
            "zerolinecolor": "#FFFFFF",
            "zerolinewidth": 2,
            "range": [-limit - 0.5, limit + 0.5],
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#333333",
            "categoryorder": "array",
            "categoryarray": y_order,
            "tickfont": {"size": 14},
        },
    )

    # Annotations for "City" and "Village"
    fig.add_annotation(
        x=-limit / 2,
        y=1.05,
        xref="x",
        yref="paper",
        text="Больше в городе",
        showarrow=False,
        font={"size": 18, "color": "#FFFFFF"},
    )
    fig.add_annotation(
        x=limit / 2,
        y=1.05,
        xref="x",
        yref="paper",
        text="Больше в селе",
        showarrow=False,
        font={"size": 18, "color": "#FFFFFF"},
    )

    # Arrows
    # Left Arrow (City)
    fig.add_annotation(
        x=-limit / 2,
        y=1.08,
        xref="x",
        yref="paper",
        ax=40,
        ay=0,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="white",
    )
    # Right Arrow (Village)
    fig.add_annotation(
        x=limit / 2,
        y=1.08,
        xref="x",
        yref="paper",
        ax=-40,
        ay=0,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="white",
    )

    output_path = OUTPUT_DIR / "graph9.png"
    fig.write_image(output_path, width=1200, height=800, scale=2)
    logger.success(f"Graph 9 saved to {output_path}")


if __name__ == "__main__":
    run()
