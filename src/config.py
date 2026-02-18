from pathlib import Path

# Project root directory (assumes src/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directory to save generated graphics
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Graph configuration
DPI = 300
FORMAT = "png"
