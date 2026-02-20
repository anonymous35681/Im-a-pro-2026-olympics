import os

# Suppress heavy logging from libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import torch
from huggingface_hub.utils import logging as hf_logging
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from transformers import logging as transformers_logging

from config import OUTPUT_DIR, ROOT_DIR
from style import NEON_CYAN, TEXT_COLOR, apply_global_style

# Set logging levels to error only to hide "UNEXPECTED" and "HF_TOKEN" warnings
transformers_logging.set_verbosity_error()
hf_logging.set_verbosity_error()


def run() -> None:
    """Generate Theme 5: Anatomy of verification using SBERT Semantic Analysis."""
    logger.info("Generating Graph 5: Semantic Anatomy of verification (SBERT).")

    # Load dataset
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")

    # Column 31 is the raw text response
    text_col = "31. По каким признакам можно отличить выдуманную, фейковую новость от правдивой, достоверной? На что полагаетесь лично Вы (напишите)?"

    # Filter valid responses
    responses = df[text_col].replace("#NULL!", pd.NA).dropna().unique().tolist()
    logger.info(f"Processing {len(responses)} unique text responses.")

    # Define reference categories (targets for semantic mapping)
    # We use descriptive phrases to help SBERT understand the context
    categories = {
        "Проверка через другие источники": "Проверка информации в других источниках, поиск опровержения или подтверждения в СМИ",
        "Анализ источника и автора": "Анализ репутации источника, проверка кто автор новости, доверие к площадке",
        "Наличие фактов и цифр": "Наличие конкретных фактов, доказательств, официальных цифр и ссылок",
        "Логика и здравый смысл": "Логическая последовательность, отсутствие противоречий, здравый смысл и критическое мышление",
        "Эмоциональный окрас": "Излишняя эмоциональность, кричащие заголовки, сенсационность, попытка вызвать панику",
        "Качество контента": "Качество видео, фото, грамотность текста, оформление сообщения",
        "Личный опыт и интуиция": "Собственный жизненный опыт, интуиция, внутреннее чутье, знание темы",
    }

    cat_labels = list(categories.keys())
    cat_descriptions = list(categories.values())

    # Initialize SBERT (Multilingual model)
    # This model is great for Russian and relatively lightweight
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode categories and responses
    cat_embeddings = model.encode(cat_descriptions, convert_to_tensor=True)
    response_embeddings = model.encode(responses, convert_to_tensor=True)

    # Compute cosine similarities
    # cosine_scores shape: [num_responses, num_categories]
    cosine_scores = util.cos_sim(response_embeddings, cat_embeddings)

    # Map responses to categories
    results_counts = dict.fromkeys(cat_labels, 0)
    results_counts["Другое / Неясно"] = 0

    threshold = 0.35  # Similarity threshold

    for i in range(len(responses)):
        max_score, max_idx = torch.max(cosine_scores[i], dim=0)
        if max_score > threshold:
            results_counts[cat_labels[max_idx]] += 1
        else:
            results_counts["Другое / Неясно"] += 1

    # Convert to DataFrame for plotting
    res_df = pd.DataFrame(
        list(results_counts.items()), columns=["category", "count"]
    ).sort_values("count", ascending=True)

    # Visualization
    apply_global_style()
    plt.figure(figsize=(12, 9))

    # Color logic: Highlight Emotional markers
    colors = [
        NEON_CYAN if "Эмоциональный" not in c else "#FF0055" for c in res_df["category"]
    ]
    # Make "Other" a bit more muted
    colors = [
        c if "Другое" not in res_df.iloc[i]["category"] else "#444444"
        for i, c in enumerate(colors)
    ]

    bars = plt.barh(res_df["category"], res_df["count"], color=colors, alpha=0.8)

    # Add count labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=12,
        )

    plt.title(
        "Семантическая анатомия проверки (SBERT Analysis)\nНа основе живых ответов респондентов",
        fontsize=18,
        pad=30,
        color=TEXT_COLOR,
    )
    plt.xlabel("Количество уникальных смысловых единиц", fontsize=12, labelpad=15)

    # Add info text about the model
    plt.figtext(
        0.99,
        0.01,
        f"Model: {model_name} | Total analyzed: {len(responses)}",
        ha="right",
        fontsize=8,
        color="white",
        alpha=0.5,
    )

    plt.grid(axis="x", color="white", alpha=0.1, linestyle="--")
    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "graph5.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f"Graph 5 (SBERT) saved to: {output_path}")


if __name__ == "__main__":
    run()
