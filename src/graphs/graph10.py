import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from config import OUTPUT_DIR, ROOT_DIR

OPEN_SOURCES_COLUMN = (
    "16. Вспомните, пожалуйста, названия двух-трех конкретных источников, "
    "из которых Вы обычно получаете новости (напишите)?"
)
TV_USAGE_COLUMN = "[Телевидение]"
DATASET_PATH = ROOT_DIR / "data" / "origin_dataset.csv"


def extract_tv_channels(text: str) -> list[str]:
    """Extract potential TV channel names from free-text response."""
    if pd.isna(text):
        return []

    text = str(text).strip()
    if not text:
        return []

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
        "youtube",
        "ютуб",
        "дзен",
        "zen",
        "одноклассники",
        "tiktok",
        "instagram",
        "whatsapp",
        "viber",
        "чат",
        "группа",
        "подписка",
        "подписчик",
        "блог",
        "блогер",
        "паблик",
        "страница",
        "профиль",
        "новостник",
        "программа",
        "время",
        "соловьев",
        "подоляка",
        "редакция",
        "российская газета",
        "московский комсомолец",
    ]

    channels = re.split(r"[,;]\s*|\n|\t", text)
    cleaned_channels: list[str] = []
    for channel in channels:
        channel = channel.strip()
        channel = re.sub(r'^["\']|["\']$', "", channel)

        channel = re.sub(r"^(тв|tv)\s*", "", channel, flags=re.IGNORECASE)
        if not re.match(r"^\d+\s*канал", channel, flags=re.IGNORECASE):
            channel = re.sub(r"^телеканал\s*", "", channel, flags=re.IGNORECASE)
        if re.match(r".*\s+(тв|tv)\s*$", channel, flags=re.IGNORECASE):
            channel = re.sub(r"\s+(тв|tv)\s*$", "", channel, flags=re.IGNORECASE)
        channel = channel.strip()

        channel_lower = channel.lower()
        is_non_tv = any(keyword in channel_lower for keyword in non_tv_keywords)
        if len(channel) > 2 and not is_non_tv:
            cleaned_channels.append(channel)

    return cleaned_channels


def normalize_channel_name(value: str) -> str:
    """Normalize channel name for robust matching."""
    normalized = value.lower().replace("ё", "е")
    normalized = re.sub(r"[\"'«»().,:;!?]", " ", normalized)
    normalized = re.sub(r"[-–—_/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def match_channel_to_known(
    channel_name: str,
    known_channels: list[str],
    known_embeddings,
    model,
    channel_aliases: dict[str, str],
    threshold: float = 0.85,
) -> tuple[str | None, float, str]:
    """Match channel mention to known official channel name."""
    channel_name_lower = channel_name.lower()
    channel_name_norm = normalize_channel_name(channel_name)
    if channel_name_lower in channel_aliases:
        return channel_aliases[channel_name_lower], 1.0, "alias"
    if channel_name_norm in channel_aliases:
        return channel_aliases[channel_name_norm], 1.0, "alias"

    for known in known_channels:
        if channel_name_lower == known.lower():
            return known, 1.0, "exact"
        if channel_name_norm == normalize_channel_name(known):
            return known, 1.0, "exact"

    for known in known_channels:
        known_lower = known.lower()
        known_norm = normalize_channel_name(known)
        if (
            channel_name_lower in known_lower or known_lower in channel_name_lower
        ) and len(channel_name) >= 3:
            return known, 0.95, "substring"
        if (channel_name_norm in known_norm or known_norm in channel_name_norm) and len(
            channel_name_norm
        ) >= 3:
            return known, 0.95, "substring"

    channel_emb = model.encode(channel_name, convert_to_tensor=True)
    similarities = util.cos_sim(channel_emb, known_embeddings)[0]
    best_idx = int(similarities.argmax().item())
    best_score = similarities[best_idx].item()
    if best_score >= threshold:
        return known_channels[best_idx], best_score, "fuzzy"
    return None, best_score, "none"


def save_dumbbell_plot(
    channel_names: list[str],
    survey_values: list[float],
    reference_values: list[float],
    survey_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Save dumbbell chart for survey vs reference."""
    fig = go.Figure()
    x_pos = list(range(len(channel_names)))

    for i in x_pos:
        fig.add_trace(
            go.Scatter(
                x=[i, i],
                y=[survey_values[i], reference_values[i]],
                mode="lines",
                line={"color": "#9CA3AF", "width": 2},
                showlegend=False,
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=survey_values,
            mode="markers",
            name=survey_label,
            marker={"color": "#0EA5E9", "size": 12},
            customdata=channel_names,
            hovertemplate=(
                "<b>%{customdata}</b><br>" + survey_label + ": %{y:.1f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=reference_values,
            mode="markers",
            name="Официальная статистика",
            marker={"color": "#EF4444", "size": 12},
            customdata=channel_names,
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Официальная статистика: %{y:.1f}%<extra></extra>"
            ),
        )
    )

    for i, _channel in enumerate(channel_names):
        diff = survey_values[i] - reference_values[i]
        mid_y = (survey_values[i] + reference_values[i]) / 2
        fig.add_annotation(
            x=i,
            y=mid_y,
            text=f"{diff:+.1f} п.п.",
            showarrow=False,
            font={"size": 11, "color": "#374151"},
            bgcolor="rgba(255,255,255,0.85)",
        )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis={
            "tickmode": "array",
            "tickvals": x_pos,
            "ticktext": channel_names,
            "tickangle": -20,
            "title": "",
            "showgrid": False,
        },
        yaxis={"title": "Доля, %", "gridcolor": "#E5E7EB"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
        margin={"l": 90, "r": 30, "t": 90, "b": 140},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04},
    )

    fig.write_image(output_path, width=1700, height=900, scale=2)
    logger.success(f"Saved chart: {output_path}")


def save_denominator_effect_plot(
    channel_names: list[str],
    all_values: list[float],
    tv_users_values: list[float],
    open_values: list[float],
    official_values: list[float],
    n_all: int,
    n_tv_users: int,
    n_open: int,
    output_path: Path,
) -> None:
    """Save chart showing denominator effect."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=channel_names,
            y=all_values,
            name=f"От всех респондентов (n={n_all})",
            marker_color="#93C5FD",
        )
    )
    fig.add_trace(
        go.Bar(
            x=channel_names,
            y=tv_users_values,
            name=f"Только пользователи ТВ (n={n_tv_users})",
            marker_color="#60A5FA",
        )
    )
    fig.add_trace(
        go.Bar(
            x=channel_names,
            y=open_values,
            name=f"Ответившие на вопрос 16 (n={n_open})",
            marker_color="#2563EB",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=channel_names,
            y=official_values,
            mode="lines+markers",
            name="Официальная статистика",
            marker={"size": 10, "color": "#DC2626"},
            line={"width": 2, "color": "#DC2626"},
        )
    )

    fig.update_layout(
        title={
            "text": "Влияние базы расчета на доли каналов (Топ-7)",
            "x": 0.5,
        },
        xaxis={"title": "", "tickangle": -20},
        yaxis={"title": "Доля, %", "gridcolor": "#E5E7EB"},
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
        margin={"l": 90, "r": 30, "t": 90, "b": 140},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.08},
    )

    fig.write_image(output_path, width=1800, height=900, scale=2)
    logger.success(f"Saved chart: {output_path}")


def run() -> None:
    """Generate Graph 10 with corrected bases and denominator-effect view."""
    logger.info("Generating Graph 10 (corrected normalization).")

    df = pd.read_csv(DATASET_PATH)

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
        "ТВ-3": 7.3,
        "Карусель": 5.1,
        "РН Рузаевские новости": 1.3,
        "ТВС": 1.1,
        "Канал соседнего региона": 0.3,
        "Другое": 1.6,
        "Затрудняюсь ответить": 0.3,
        "Не смотрю телевизор": 15.6,
    }

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
        "канал россия": "Россия-1",
        "гтрк": "ГТРК 'Мордовия'",
        "гтрк мордовия": "ГТРК 'Мордовия'",
        "россия 24": "Россия-24",
        "россия24": "Россия-24",
        "вести 24": "Россия-24",
        "рен тв": "РЕН ТВ",
        "рен-тв": "РЕН ТВ",
        "рентв": "РЕН ТВ",
        "матч": "Матч ТВ",
        "матч тв": "Матч ТВ",
        "пятый": "Пятый канал",
        "пятый канал": "Пятый канал",
        "твц": "ТВЦ",
    }

    known_channel_names = list(channels_popularity.keys())
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    known_channel_embeddings = model.encode(
        known_channel_names, convert_to_tensor=True
    )

    tv_user_mask = df[TV_USAGE_COLUMN].fillna("").astype(str).str.startswith("1.")
    open_answer_mask = (
        df[OPEN_SOURCES_COLUMN].fillna("").astype(str).str.strip().ne("")
    )

    n_all = len(df)
    n_tv_users = int(tv_user_mask.sum())
    n_open = int(open_answer_mask.sum())
    logger.info(
        f"Base sizes: all={n_all}, tv_users={n_tv_users}, open_answered={n_open}"
    )

    counts_all = Counter()
    counts_tv_users = Counter()
    match_stats = {"alias": 0, "exact": 0, "substring": 0, "fuzzy": 0, "none": 0}
    match_examples = {
        "alias": [],
        "exact": [],
        "substring": [],
        "fuzzy": [],
        "none": [],
    }

    for idx, text in enumerate(df[OPEN_SOURCES_COLUMN]):
        extracted = extract_tv_channels(text)
        matched_channels_in_row: set[str] = set()
        for mention in extracted:
            matched, score, method = match_channel_to_known(
                channel_name=mention,
                known_channels=known_channel_names,
                known_embeddings=known_channel_embeddings,
                model=model,
                channel_aliases=channel_aliases,
            )
            match_stats[method] += 1
            if len(match_examples[method]) < 3:
                match_examples[method].append((mention, matched, score))
            if matched:
                matched_channels_in_row.add(matched)

        for channel in matched_channels_in_row:
            counts_all[channel] += 1
            if tv_user_mask.iat[idx]:
                counts_tv_users[channel] += 1

    total_attempts = sum(match_stats.values())
    logger.info("Matching statistics:")
    logger.info(f"  Total attempts: {total_attempts}")
    for method, count in match_stats.items():
        if method != "none":
            pct = (count / total_attempts * 100) if total_attempts else 0.0
            logger.info(f"  {method.capitalize()}: {count} ({pct:.1f}%)")

    logger.info("Matching examples:")
    for method in ["alias", "exact", "substring", "fuzzy", "none"]:
        if match_examples[method]:
            logger.info(f"  {method.capitalize()} matches:")
            for mention, matched, score in match_examples[method]:
                matched_str = matched if matched else "None"
                logger.info(f"    '{mention}' -> '{matched_str}' ({score:.4f})")

    def to_pct(counter: Counter, denom: int) -> dict[str, float]:
        if denom <= 0:
            return dict.fromkeys(channels_popularity, 0.0)
        return {
            channel: (counter.get(channel, 0) / denom) * 100
            for channel in channels_popularity
        }

    frequency_all = to_pct(counts_all, n_all)
    frequency_tv_users = to_pct(counts_tv_users, n_tv_users)
    frequency_open = to_pct(counts_all, n_open)

    top_channels = sorted(
        channels_popularity.items(), key=lambda x: x[1], reverse=True
    )[:7]
    top_channel_names = [channel for channel, _ in top_channels]
    official_values = [channels_popularity[ch] for ch in top_channel_names]
    open_values = [frequency_open[ch] for ch in top_channel_names]
    all_values = [frequency_all[ch] for ch in top_channel_names]
    tv_users_values = [frequency_tv_users[ch] for ch in top_channel_names]

    main_output = OUTPUT_DIR / "graph10.png"
    save_dumbbell_plot(
        channel_names=top_channel_names,
        survey_values=open_values,
        reference_values=official_values,
        survey_label=f"Открытый вопрос, база n={n_open}",
        title=(
            "Спонтанные упоминания ТВ-каналов vs официальная статистика\n"
            "(доля от ответивших на открытый вопрос)"
        ),
        output_path=main_output,
    )

    denominator_output = OUTPUT_DIR / "graph10_base_effect.png"
    save_denominator_effect_plot(
        channel_names=top_channel_names,
        all_values=all_values,
        tv_users_values=tv_users_values,
        open_values=open_values,
        official_values=official_values,
        n_all=n_all,
        n_tv_users=n_tv_users,
        n_open=n_open,
        output_path=denominator_output,
    )

    logger.info("Top-7 channels comparison:")
    for channel in top_channel_names:
        logger.info(
            f"  {channel}: official={channels_popularity[channel]:.1f}% | "
            f"all={frequency_all[channel]:.1f}% | "
            f"tv_users={frequency_tv_users[channel]:.1f}% | "
            f"open_base={frequency_open[channel]:.1f}%"
        )


if __name__ == "__main__":
    run()
