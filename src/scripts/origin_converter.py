import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Project root is now three levels up from src/scripts/converter.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Match format from src/logger.py
LOG_FORMAT = (
    "<green>{time:DD.MM.YYYY HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<white>{message}</white>"
)


def setup_simple_logger() -> None:
    """Configure a simple logger for the script matching project settings."""
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level="INFO")


def convert_excel_to_csv(excel_path: Path, csv_path: Path) -> None:
    """Convert Excel file to CSV."""
    try:
        logger.info(f"Reading Excel file: {excel_path}")
        # Read Excel file
        df = pd.read_excel(excel_path)

        # Ensure the parent directory for the CSV exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV (using utf-8-sig for better compatibility with Excel/Cyrillic)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Successfully converted to: {csv_path}")

    except Exception as e:
        logger.error(f"Failed to convert file: {e}")


if __name__ == "__main__":
    setup_simple_logger()

    # Define paths
    raw_dir = PROJECT_ROOT / "data" / "raw"
    excel_file = (
        raw_dir / "БАК 3. Социологический опрос населения Республики Мордовия.xlsx"
    )
    output_csv = PROJECT_ROOT / "data" / "origin_dataset.csv"

    convert_excel_to_csv(excel_file, output_csv)
