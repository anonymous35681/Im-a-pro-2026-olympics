import os

# Suppress heavy logging from libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from huggingface_hub.utils import logging as hf_logging
from sentence_transformers import SentenceTransformer, util
from transformers import logging as transformers_logging

from config import ROOT_DIR

# Set logging levels
transformers_logging.set_verbosity_error()
hf_logging.set_verbosity_error()


def debug_other():
    """Analyze responses classified as 'Другое / Неясно'."""
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")
    text_col = "31. По каким признакам можно отличить выдуманную, фейковую новость от правдивой, достоверной? На что полагаетесь лично Вы (напишите)?"

    responses = df[text_col].replace("#NULL!", pd.NA).dropna().unique().tolist()

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

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)

    cat_embeddings = model.encode(cat_descriptions, convert_to_tensor=True)
    response_embeddings = model.encode(responses, convert_to_tensor=True)

    cosine_scores = util.cos_sim(response_embeddings, cat_embeddings)

    threshold = 0.35
    other_responses = []

    for i in range(len(responses)):
        max_score, max_idx = torch.max(cosine_scores[i], dim=0)
        score = max_score.item()
        if score <= threshold:
            other_responses.append(
                {
                    "response": responses[i],
                    "max_score": round(score, 3),
                    "closest_category": cat_labels[max_idx],
                }
            )

    # Sort by score descending to see those closest to the threshold
    other_responses.sort(key=lambda x: x["max_score"], reverse=True)

    print(f"Total 'Другое / Неясно': {len(other_responses)} out of {len(responses)}\n")
    print(f"{'Score':<8} | {'Closest Category':<30} | {'Response'}")
    print("-" * 100)

    # Print top 50 or so to not flood too much, but enough to see patterns
    for item in other_responses[:50]:
        print(
            f"{item['max_score']:<8} | {item['closest_category']:<30} | {item['response']}"
        )


if __name__ == "__main__":
    debug_other()
