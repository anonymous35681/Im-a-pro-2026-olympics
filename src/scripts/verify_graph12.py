import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add src to python path to import config
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root / "src"))

from config import ROOT_DIR  # noqa: E402


def run():
    logger.info("Starting verification for Graph 12 data...")
    df = pd.read_csv(ROOT_DIR / "data" / "origin_dataset.csv")
    total = len(df)

    channels = [
        ("[Объединенный] Телевидение", "ТВ"),
        ("[Объединенный] Интернет-издания", "Интернет-СМИ"),
        ("[Объединенный] Социальные сети", "Соцсети"),
        ("[Объединенный] Друзья", "Друзья"),
        ("[Объединенный] Газеты", "Газеты"),
        ("[Объединенный] Радио", "Радио"),
        ("[Объединенный] Журналы", "Журналы"),
        ("[Объединенный] Телеграм-каналы", "Telegram"),
    ]

    print(f"\nTotal respondents: {total}")
    print("-" * 80)
    print(f"{'Channel':<15} | {'Trust (%)':<10} | {'Distrust (%)':<12} | {'Neutral (%)':<12} | {'No Ans (%)':<10}")
    print("-" * 80)

    for col, label in channels:
        trust_mask = df[col].isin(["Пользуюсь и доверяю", "Доверяю"])
        distrust_mask = df[col].isin(["Пользуюсь, но не доверяю", "Не доверяю"])
        neutral_mask = df[col].isin(["Пользуюсь"])
        null_mask = df[col].isin(["#NULL!"])

        trust_pct = (trust_mask.sum() / total) * 100
        distrust_pct = (distrust_mask.sum() / total) * 100
        neutral_pct = (neutral_mask.sum() / total) * 100
        null_pct = (null_mask.sum() / total) * 100

        print(f"{label:<15} | {trust_pct:>9.1f}% | {distrust_pct:>11.1f}% | {neutral_pct:>11.1f}% | {null_pct:>9.1f}%")

if __name__ == "__main__":
    run()
