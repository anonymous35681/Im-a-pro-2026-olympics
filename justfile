@_default:
    just --list

# CI: lint, format (ruff, ty)
qa:
    uv run ruff check src
    uv run ty check src

# Generate all graphics
run:
    uv run src/main.py
