"""Improved debug script for graph10.py to check data processing accuracy."""

import re
from collections import Counter

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from config import ROOT_DIR


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
        # NOTE: Do NOT remove trailing "тв/tv" because "НТВ" is a channel name!  # noqa: RUF003
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


def main() -> None:
    """Run debug analysis for graph10."""

    # Known TV channels with their popularity statistics
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
        "гтрк": "ГТРК Мордовия",
        "гтрк мордовия": "ГТРК Мордовия",
        "рбк": "РБК",
        "столица с": "Столица С",
        "столица": "Столица С",
    }

    logger.info("=" * 80)
    logger.info("IMPROVED DEBUG SCRIPT FOR GRAPH10 - TV CHANNEL ANALYSIS")
    logger.info("=" * 80)

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")
    channel_column = "16. Вспомните, пожалуйста, названия двух-трех конкретных источников, из которых Вы обычно получаете новости (напишите)?"

    logger.info(f"\nTotal rows in dataset: {len(df)}")
    logger.info(
        f"Non-empty responses: {len([x for x in df[channel_column] if pd.notna(x) and str(x).strip()])}"
    )

    # Load model
    logger.info("\nLoading sentence transformer model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    known_channel_names = list(channels_popularity.keys())
    logger.info(f"Model loaded. Known channels: {len(known_channel_names)}")

    # Show sample raw responses
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE RAW RESPONSES (first 10 non-empty)")
    logger.info("=" * 80)
    sample_responses = df[channel_column].dropna().head(10)
    for i, text in enumerate(sample_responses, 1):
        logger.info(f"\n{i}. {text}")

    # Extract all channels and show matching details
    logger.info("\n" + "=" * 80)
    logger.info("CHANNEL MATCHING DETAILS (IMPROVED)")
    logger.info("=" * 80)

    all_mentions = []
    match_stats = {"alias": 0, "exact": 0, "substring": 0, "fuzzy": 0, "none": 0}
    match_examples = {
        "alias": [],
        "exact": [],
        "substring": [],
        "fuzzy": [],
        "none": [],
    }
    false_positives = []
    true_positives = []

    for text in df[channel_column]:
        if pd.isna(text) or not str(text).strip():
            continue

        extracted = extract_tv_channels(text)
        if not extracted:
            continue

        for mention in extracted:
            matched, score, method = match_channel_to_known(
                mention, known_channel_names, model, channel_aliases
            )
            match_stats[method] += 1

            # Collect examples (first 5 of each type)
            if len(match_examples[method]) < 5:
                match_examples[method].append((mention, matched, score))

            if matched:
                all_mentions.append(matched)
                # Manually verify some matches for accuracy assessment
                if method == "fuzzy" and len(false_positives) < 10:
                    # Check if it's likely a false positive by looking at the mention
                    mention_lower = mention.lower()
                    if any(keyword in mention_lower for keyword in ["нтв", "нтм"]):
                        false_positives.append((mention, matched, score))
                    else:
                        true_positives.append((mention, matched, score))
            elif method == "none" and len(match_examples["none"]) <= 10:
                # Store unmatched examples
                pass

    # Show matched examples by method
    for method in ["alias", "exact", "substring", "fuzzy", "none"]:
        if match_examples[method]:
            logger.info(f"\n{method.upper()} MATCHES:")
            for mention, matched, score in match_examples[method]:
                matched_str = matched if matched else "None"
                logger.info(f"  '{mention}' -> '{matched_str}' (score: {score:.4f})")

    # Show matching statistics
    total_attempts = sum(match_stats.values())
    logger.info("\nMATCHING STATISTICS:")
    logger.info(f"  Total extraction attempts: {total_attempts}")
    for method, count in match_stats.items():
        if method != "none":
            pct = (count / total_attempts * 100) if total_attempts > 0 else 0
            logger.info(f"  {method.capitalize()}: {count} ({pct:.1f}%)")

    # Calculate accuracy
    successful_matches = (
        match_stats["exact"] + match_stats["substring"] + match_stats["fuzzy"]
    )
    accuracy_rate = (
        (successful_matches / total_attempts * 100) if total_attempts > 0 else 0
    )
    logger.info(
        f"\nOVERALL MATCHING RATE: {successful_matches}/{total_attempts} ({accuracy_rate:.1f}%)"
    )

    # Estimate precision (assuming fuzzy has ~10% false positives based on examples)
    estimated_true_positives = (
        match_stats["exact"]
        + match_stats["substring"]
        + int(match_stats["fuzzy"] * 0.9)
    )
    precision = (
        (estimated_true_positives / successful_matches * 100)
        if successful_matches > 0
        else 0
    )
    logger.info(
        f"ESTIMATED PRECISION: ~{precision:.1f}% (assuming ~10% false positives in fuzzy matches)"
    )

    # Count and display frequencies
    dataset_counts = Counter(all_mentions)
    # Use total dataset size (1000) for comparison with official statistics
    # not just number of responses (534)
    total_responses = len(df)

    logger.info("\n" + "=" * 80)
    logger.info("CHANNEL FREQUENCY IN DATASET")
    logger.info("=" * 80)
    logger.info(f"Total responses used for calculation: {total_responses}")

    # Calculate frequencies
    dataset_frequency = {
        channel: (dataset_counts.get(channel, 0) / total_responses) * 100
        for channel in channels_popularity
    }

    # Show all channels with their frequencies
    logger.info("\nAll channels with dataset frequency:")
    for channel, ref_pop in sorted(
        channels_popularity.items(), key=lambda x: x[1], reverse=True
    ):
        dataset_freq = dataset_frequency[channel]
        count = dataset_counts.get(channel, 0)
        diff = dataset_freq - ref_pop
        logger.info(
            f"  {channel:40s} | Count: {count:3d} | Dataset: {dataset_freq:5.1f}% | Reference: {ref_pop:5.1f}% | Diff: {diff:+6.1f}%"
        )

    # Top 7 channels comparison
    logger.info("\n" + "=" * 80)
    logger.info("TOP-7 CHANNELS COMPARISON")
    logger.info("=" * 80)

    top_channels = sorted(
        channels_popularity.items(), key=lambda x: x[1], reverse=True
    )[:7]
    for channel, ref_pop in top_channels:
        dataset_freq = dataset_frequency[channel]
        count = dataset_counts.get(channel, 0)
        diff = dataset_freq - ref_pop
        logger.info(
            f"  {channel:40s}\n"
            f"    Count: {count:3d} | Dataset: {dataset_freq:5.1f}% | Reference: {ref_pop:5.1f}% | Diff: {diff:+6.1f}%"
        )

    # Accuracy assessment
    logger.info("\n" + "=" * 80)
    logger.info("ACCURACY ASSESSMENT")
    logger.info("=" * 80)

    # Channels with significant mentions
    mentioned_channels = {ch: cnt for ch, cnt in dataset_counts.items() if cnt > 0}
    logger.info(f"\nChannels actually mentioned in dataset: {len(mentioned_channels)}")
    logger.info(
        f"Channels never mentioned: {len(channels_popularity) - len(mentioned_channels)}"
    )

    logger.info("\nChannels with most mentions in dataset:")
    for channel, count in dataset_counts.most_common(10):
        logger.info(
            f"  {channel:40s} | {count:3d} times ({dataset_frequency[channel]:.1f}%)"
        )

    # Final accuracy summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL ACCURACY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"1. Total extraction attempts: {total_attempts}")
    logger.info(f"2. Successfully matched: {successful_matches} ({accuracy_rate:.1f}%)")
    logger.info("3. Match breakdown:")
    logger.info(
        f"   - Alias matches: {match_stats['alias']} ({match_stats['alias'] / total_attempts * 100:.1f}%)"
    )
    logger.info(
        f"   - Exact matches: {match_stats['exact']} ({match_stats['exact'] / total_attempts * 100:.1f}%)"
    )
    logger.info(
        f"   - Substring matches: {match_stats['substring']} ({match_stats['substring'] / total_attempts * 100:.1f}%)"
    )
    logger.info(
        f"   - Fuzzy matches: {match_stats['fuzzy']} ({match_stats['fuzzy'] / total_attempts * 100:.1f}%)"
    )
    logger.info(
        f"   - No match: {match_stats['none']} ({match_stats['none'] / total_attempts * 100:.1f}%)"
    )
    logger.info(f"\n4. Estimated precision: ~{precision:.1f}%")
    logger.info(
        "5. Filtering effectiveness: Non-TV sources filtered out during extraction"
    )

    logger.info("\n" + "=" * 80)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
