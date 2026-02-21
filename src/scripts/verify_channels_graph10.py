"""Critical verification of all channel matches."""

import re
from collections import defaultdict

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from config import ROOT_DIR


def extract_tv_channels(text: str) -> list[str]:
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []

    non_tv_keywords = [
        "вконтакте", "вк", "telegram", "телеграм", "tg", "телеграмм-канал", "тг-канал",
        "сайт", "официальный сайт", "интернет", "социальн", "сети",
        "юoutube", "youtube", "дзен", "zen", "одноклассники",
        "tiktok", "instagram", "whatsapp", "viber", "чат", "воцап",
        "группа", "подписка", "подписчик", "блог", "блогер",
        "паблик", "страница", "профиль", "новостник",
        # Programs and personalities (not channels)
        "программа", "время", "соловьев", "подоляка", "юрий", "миг",
        "редакция", "российская газета", "минобрнауки",
        "московский комсомолец", "новости москвы", "призрак", "новороссии",
    ]

    channels = re.split(r"[;,]\s*|\n|\t", text)
    cleaned_channels = []
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


def main() -> None:
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")
    col = "16. Вспомните, пожалуйста, названия двух-трех конкретных источников, из которых Вы обычно получаете новости (напишите)?"

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    channels_popularity = {
        "Первый канал": 60.9,
        "Россия-1": 53.1,
        "НТВ": 42.3,
        "Россия-24": 26.4,
        "НТМ — Народное телевидение Мордовии": 24.3,
        "РЕН ТВ": 22.9,
        "ТНТ": 22.6,
        "СТС": 21.3,
        "10 канал — ТелеСеть Мордовии": 21.0,
        "Матч ТВ": 20.5,
        "Звезда": 19.4,
        "Домашний": 18.9,
        "Пятый канал": 17.8,
        "ТВЦ": 15.1,
        "Россия-Культура": 14.8,
        "Пятница": 12.1,
        "Мир": 11.9,
        "Мордовия-24": 11.3,
        "Спас": 10.8,
        "ОТР": 10.2,
        "Муз-ТВ": 9.4,
        "ТВ-З": 8.9,
        "Карусель": 7.3,
        "РН Рузаевские новости": 5.1,
        "ГТРК Мордовия": 4.2,
        "РБК": 3.2,
        "Столица С": 1.5,
        "ТВС": 1.3,
        "Канал соседнего региона": 1.1,
        "Другое": 0.3,
        "Затрудняюсь ответить": 1.6,
        "Не смотрю телевизор": 0.3,
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
        "гтрк": "ГТРК Мордовия",
        "гтрк мордовия": "ГТРК Мордовия",
        "рбк": "РБК",
        "столица с": "Столица С",
        "столица": "Столица С",
    }

    known_channel_names = list(channels_popularity.keys())

    # Collect all matches with details
    matches = []

    for text in df[col]:
        if pd.isna(text) or not str(text).strip():
            continue
        extracted = extract_tv_channels(text)
        for mention in extracted:
            mention_lower = mention.lower()
            matched = None
            method = None
            score = None

            if mention_lower in channel_aliases:
                matched = channel_aliases[mention_lower]
                method = "alias"
                score = 1.0
            else:
                # Exact match
                for known in known_channel_names:
                    if mention_lower == known.lower():
                        matched = known
                        method = "exact"
                        score = 1.0
                        break

                if not matched:
                    # Substring match
                    for known in known_channel_names:
                        known_lower = known.lower()
                        if (
                            (mention_lower in known_lower or known_lower in mention_lower)
                            and len(mention) >= 3
                        ):
                            matched = known
                            method = "substring"
                            score = 0.95
                            break

                if not matched:
                    # Fuzzy match
                    channel_emb = model.encode(mention, convert_to_tensor=True)
                    known_embs = model.encode(known_channel_names, convert_to_tensor=True)
                    similarities = util.cos_sim(channel_emb, known_embs)[0]
                    best_idx = int(similarities.argmax().item())
                    best_score = float(similarities[best_idx].item())
                    if best_score >= 0.85:
                        matched = known_channel_names[best_idx]
                        method = "fuzzy"
                        score = best_score

            if matched:
                matches.append({
                    "mention": mention,
                    "matched": matched,
                    "method": method,
                    "score": score,
                })

    # Group by matched channel
    by_channel = defaultdict(list)
    for m in matches:
        by_channel[m["matched"]].append(m)

    logger.info("=" * 80)
    logger.info("CRITICAL VERIFICATION OF ALL CHANNEL MATCHES")
    logger.info("=" * 80)

    for channel in sorted(channels_popularity.keys(), key=lambda x: channels_popularity[x], reverse=True)[:15]:
        if channel in by_channel:
            channel_matches = by_channel[channel]
            logger.info(f"\n{channel} ({len(channel_matches)} matches):")
            logger.info(f"  Reference popularity: {channels_popularity[channel]}%")

            # Show unique mentions
            unique_mentions = {}
            for m in channel_matches:
                key = (m["mention"], m["method"], m["score"])
                unique_mentions[key] = unique_mentions.get(key, 0) + 1

            # Group by method
            by_method = defaultdict(list)
            for (mention, method, score), count in unique_mentions.items():
                by_method[method].append((mention, count, score))

            for method in ["alias", "exact", "substring", "fuzzy"]:
                if method in by_method:
                    logger.info(f"\n  {method.upper()} matches:")
                    for mention, count, score in sorted(by_method[method], key=lambda x: -x[1])[:10]:
                        logger.info(f'    "{mention}" -> {count}x (score: {score:.4f})')
                        if method == "fuzzy" and score < 0.90:
                            logger.warning("      LOW CONFIDENCE - MANUAL CHECK NEEDED")


if __name__ == "__main__":
    main()
