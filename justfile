@_default:
    just --list

# CI: lint, format (ruff, ty)
qa:
    uv run ruff check src/
    uv run ty check src/

# Run all convert recipes
convert_all: convert_origin convert_demographics

# Run the converter script for origin dataset (origin xlsx -> csv)
convert_origin:
    uv run src/scripts/origin_converter.py

# Run the converter script for demographics dataset (origin xlsx -> csv)
convert_demographics:
    uv run src/scripts/demographics_converter.py

# Generate all graphics (or specific graph with number)
run number="":
    uv run src/main.py {{number}}
