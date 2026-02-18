@_default:
    just --list

# CI: lint, format (ruff, ty)
qa:
    uv run ruff check src/
    uv run ty check src/

# Run the converter script
convert_origin:
    uv run src/scripts/converter.py

# Generate all graphics
run:
    uv run src/main.py
