import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from config import OUTPUT_DIR, ROOT_DIR


def extract_tv_channels(text: str) -> list[str]:
    """Extract potential TV channel names from text."""
    if pd.isna(text):
        return []

    text = str(text).strip()
    if not text:
        return []

    # Keywords that indicate non-TV sources
    non_tv_keywords = [
        "вконтакте",
        "вк",
        "telegram",
        "телеграм",
        "tg",
        "телеграмм-канал",
        "тг-канал",
        "сайт",
        "официальный сайт",
        "интернет",
        "социальн",
        "сети",
        "юoutube",
        "youtube",
        "дзен",
        "zen",
        "одноклассники",
        "tiktok",
        "instagram",
        "whatsapp",
        "viber",
        "чат",
        "воцап",
        "группа",
        "подписка",
        "подписчик",
        "блог",
        "блогер",
        "паблик",
        "страница",
        "профиль",
        "новостник",
        # Programs and personalities (not channels)
        "программа",
        "время",
        "соловьев",
        "подоляка",
        "юрий",
        "миг",
        "редакция",
        "российская газета",
        "минобрнауки",
        "московский комсомолец",
        "новости москвы",
        "призрак",
        "новороссии",
    ]

    # Split by common delimiters
    channels = re.split(r"[,;]\s*|\n|\t", text)

    # Clean up each channel name
    cleaned_channels = []
    for channel in channels:
        channel = channel.strip()
        # Remove common prefixes and suffixes
        channel = re.sub(r'^["\']|["\']$', "", channel)  # Remove quotes

        # Remove "тв/tv" prefix but NOT "канал" if preceded by digit (e.g., "1 канал")
        channel = re.sub(r"^(тв|tv)\s*", "", channel, flags=re.IGNORECASE)
        # Remove "телеканал" prefix but keep "X канал" patterns
        if not re.match(r"^\d+\s*канал", channel, flags=re.IGNORECASE):
            channel = re.sub(r"^телеканал\s*", "", channel, flags=re.IGNORECASE)
        # NOTE: Do NOT remove trailing "тв/tv" because "ТВ" is a channel name!  # noqa: RUF003
        # Only remove it if there's a space before (like "1 канал тв")
        if re.match(r".*\s+(тв|tv)\s*$", channel, flags=re.IGNORECASE):
            channel = re.sub(r"\s+(тв|tv)\s*$", "", channel, flags=re.IGNORECASE)
        channel = channel.strip()

        # Filter out non-TV sources
        channel_lower = channel.lower()
        is_non_tv = any(keyword in channel_lower for keyword in non_tv_keywords)

        if len(channel) > 2 and not is_non_tv:  # Only keep meaningful TV names
            cleaned_channels.append(channel)

    return cleaned_channels


def match_channel_to_known(
    channel_name: str,
    known_channels: list[str],
    model: SentenceTransformer,
    channel_aliases: dict[str, str],
    threshold: float = 0.85,
) -> tuple[str | None, float, str]:
    """Match a channel name to the closest known channel.

    Returns tuple of (matched_channel_name, similarity_score, match_method).
    match_method can be: 'alias', 'exact', 'substring', 'fuzzy', or None.
    """
    # First try aliases
    channel_name_lower = channel_name.lower()
    if channel_name_lower in channel_aliases:
        return channel_aliases[channel_name_lower], 1.0, "alias"

    # Then try exact case-insensitive match
    for known in known_channels:
        if channel_name_lower == known.lower():
            return known, 1.0, "exact"

    # Also try substring match (e.g., "нтв" in "нтв гтрк «мордовия»")
    for known in known_channels:
        known_lower = known.lower()
        if (
            channel_name_lower in known_lower or known_lower in channel_name_lower
        ) and len(channel_name) >= 3:
            return known, 0.95, "substring"

    # If no exact match, use sentence transformer
    channel_emb = model.encode(channel_name, convert_to_tensor=True)
    known_embs = model.encode(known_channels, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(channel_emb, known_embs)[0]

    # Get the best match
    best_idx = int(similarities.argmax().item())
    best_score = similarities[best_idx].item()

    if best_score >= threshold:
        return known_channels[best_idx], best_score, "fuzzy"
    return None, best_score, "none"


def run() -> None:
    """Generate Graph 10: TV channel popularity difference (Dumbbell plot)."""
    logger.info(
        "Generating Graph 10: TV channel popularity difference (Dumbbell plot)."
    )

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Known TV channels with their popularity statistics
    channels_popularity = {
        "Первый канал": 60.9,
        "Россия-1": 53.1,
        "НТВ": 42.3,
        "ГТРК 'Мордовия'": 26.4,
        "Россия-24": 24.3,
        "НТМ — Народное телевидение Мордовии": 22.9,
        "РЕН ТВ": 22.6,
        "ТНТ": 21.3,
        "СТС": 21.0,
        "10 канал — ТелеСеть Мордовии": 20.5,
        "Матч ТВ": 19.4,
        "Звезда": 18.9,
        "Домашний": 17.8,
        "Пятый канал": 15.1,
        "ТВЦ": 14.8,
        "Россия-Культура": 12.1,
        "Пятница": 11.9,
        "Мир": 11.3,
        "Мордовия-24": 10.8,
        "Спас": 10.2,
        "ОТР": 9.4,
        "Муз-ТВ": 8.9,
        "ТВ-З": 7.3,
        "Карусель": 5.1,
        "РН Рузаевские новости": 1.3,
        "ТВС": 1.1,
        "Канал соседнего региона": 0.3,
        "Другое": 1.6,
        "Затрудняюсь ответить": 0.3,
        "Не смотрю телевизор": 15.6,
    }

    # Channel aliases - map alternative names to official names
    channel_aliases = {
        "орт": "Первый канал",
        "1 канал": "Первый канал",
        "1-й канал": "Первый канал",
        "1й канал": "Первый канал",
        "первый": "Первый канал",
        "первый канал": "Первый канал",
        "россия": "Россия-1",
        "россия 1": "Россия-1",
        "россия-1": "Россия-1",
        "нтв": "НТВ",
        "нтв.": "НТВ",
        "гтрк": "ГТРК 'Мордовия'",
        "гтрк мордовия": "ГТРК 'Мордовия'",
    }

    # Load sentence transformer model for fuzzy matching
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    known_channel_names = list(channels_popularity.keys())

    # Extract and count TV channel mentions from the dataset
    logger.info("Extracting TV channel mentions from dataset...")
    channel_column = "16. Вспомните, пожалуйста, названия двух-трех конкретных источников, из которых Вы обычно получаете новости (напишите)?"

    all_mentions = []
    match_stats = {"alias": 0, "exact": 0, "substring": 0, "fuzzy": 0, "none": 0}
    match_examples = {
        "alias": [],
        "exact": [],
        "substring": [],
        "fuzzy": [],
        "none": [],
    }

    for text in df[channel_column]:
        extracted = extract_tv_channels(text)
        for mention in extracted:
            # Try to match to known channels
            matched, score, method = match_channel_to_known(
                mention, known_channel_names, model, channel_aliases
            )
            match_stats[method] += 1

            # Collect examples (first 3 of each type)
            if len(match_examples[method]) < 3:
                match_examples[method].append((mention, matched, score))

            if matched:
                all_mentions.append(matched)

    # Log matching statistics
    total_attempts = sum(match_stats.values())
    logger.info("Matching statistics:")
    logger.info(f"  Total attempts: {total_attempts}")
    for method, count in match_stats.items():
        if method != "none":
            pct = (count / total_attempts * 100) if total_attempts > 0 else 0
            logger.info(f"  {method.capitalize()}: {count} ({pct:.1f}%)")

    # Log matching examples
    logger.info("\nMatching examples:")
    for method in ["alias", "exact", "substring", "fuzzy", "none"]:
        if match_examples[method]:
            logger.info(f"  {method.capitalize()} matches:")
            for mention, matched, score in match_examples[method]:
                matched_str = matched if matched else "None"
                logger.info(f"    '{mention}' -> '{matched_str}' (score: {score:.4f})")

    # Count mentions in dataset
    dataset_counts = Counter(all_mentions)
    # Use total dataset size (1000) for comparison with official statistics
    # not just number of responses (534)
    total_responses = len(df)

    # Calculate frequencies in dataset (as percentages)
    dataset_frequency = {
        channel: (dataset_counts.get(channel, 0) / total_responses) * 100
        for channel in channels_popularity
    }

    # Calculate difference: dataset frequency - reference popularity
    channel_diffs = {
        channel: dataset_frequency[channel] - channels_popularity[channel]
        for channel in channels_popularity
    }

    # Sort by reference popularity and take top 7
    top_channels = sorted(
        channels_popularity.items(), key=lambda x: x[1], reverse=True
    )[:7]

    top_channel_names = [channel for channel, _ in top_channels]

    # Prepare data for dumbbell plot
    dataset_values = [dataset_frequency[ch] for ch in top_channel_names]
    reference_values = [channels_popularity[ch] for ch in top_channel_names]
    diff_values = [channel_diffs[ch] for ch in top_channel_names]

    # Create dumbbell plot with single traces per type
    fig = go.Figure()

    # Colors
    dataset_color = "#00FFFF"  # Cyan for dataset
    reference_color = "#FF0055"  # Pink for reference
    line_color = "#666666"  # Gray for connecting lines

    # Prepare all data for dataset points
    dataset_x = list(range(len(top_channel_names)))
    dataset_y = dataset_values

    # Prepare all data for reference points
    reference_x = list(range(len(top_channel_names)))
    reference_y = reference_values

    # Add all connecting lines at once
    for i in range(len(top_channel_names)):
        fig.add_trace(
            go.Scatter(
                x=[i, i],
                y=[dataset_values[i], reference_values[i]],
                mode="lines",
                line={"color": line_color, "width": 3},
                showlegend=False,
                hoverinfo="none",
            )
        )

    # Add all dataset points
    fig.add_trace(
        go.Scatter(
            x=dataset_x,
            y=dataset_y,
            mode="markers",
            name="Частота в опросе",
            marker={
                "color": dataset_color,
                "size": 16,
                "symbol": "circle",
                "line": {"color": "#000000", "width": 1},
            },
            hovertemplate="<b>%{x}</b><br>Частота в опросе: %{y:.1f}%<extra></extra>",
            text=top_channel_names,
        )
    )

    # Add all reference points
    fig.add_trace(
        go.Scatter(
            x=reference_x,
            y=reference_y,
            mode="markers",
            name="Официальная статистика",
            marker={
                "color": reference_color,
                "size": 16,
                "symbol": "circle",
                "line": {"color": "#000000", "width": 1},
            },
            hovertemplate="<b>%{x}</b><br>Официальная статистика: %{y:.1f}%<extra></extra>",
            text=top_channel_names,
        )
    )

    # Add difference annotations
    for i, (_channel, diff) in enumerate(
        zip(top_channel_names, diff_values, strict=True)
    ):
        mid_y = (dataset_values[i] + reference_values[i]) / 2
        diff_text = f"{diff:+.1f}%"
        color = "#00FF00" if diff > 0 else "#FF0055"

        fig.add_annotation(
            x=i,
            y=mid_y,
            text=diff_text,
            showarrow=False,
            font={"size": 14, "color": color, "family": "Arial Black"},
            xanchor="center",
            yanchor="middle",
            bgcolor="rgba(0,0,0,0.7)",
        )

    fig.update_layout(
        title={
            "text": "Разрыв в популярности: опрос vs официальная статистика<br>Топ-7 телеканалов",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#FFFFFF"},
        },
        xaxis={
            "title": "",
            "tickmode": "array",
            "tickvals": list(range(len(top_channel_names))),
            "ticktext": top_channel_names,
            "tickfont": {"size": 13, "color": "#FFFFFF"},
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "tickangle": -15,
        },
        yaxis={
            "title": "Популярность (%)",
            "tickfont": {"size": 12, "color": "#FFFFFF"},
            "gridcolor": "#333333",
            "showgrid": True,
            "gridwidth": 1,
            "showline": True,
            "linewidth": 1,
            "linecolor": "#333333",
        },
        font={"size": 12, "color": "#FFFFFF"},
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        margin={"l": 80, "r": 50, "t": 100, "b": 100},
        height=700,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.98,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(0,0,0,0.5)",
            "bordercolor": "#00FFFF",
            "borderwidth": 1,
        },
        hovermode="closest",
        bargap=0.3,
    )

    # Save as PNG
    output_path = OUTPUT_DIR / "graph10.png"
    fig.write_image(output_path, width=1600, height=800, scale=2)
    logger.success(f"Graph 10 PNG saved to: {output_path}")

    # Also log insights
    logger.info("Channel frequency analysis:")
    for channel in top_channel_names:
        logger.info(
            f"  {channel}: {dataset_frequency[channel]:.1f}% (dataset) vs "
            f"{channels_popularity[channel]:.1f}% (reference) "
            f"→ Diff: {channel_diffs[channel]:+.1f}%"
        )


if __name__ == "__main__":
    run()
