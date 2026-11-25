#!/usr/bin/env bash
#
# Run linters
#
set -x
uv lock --check
uv run ruff check --fix
uv run ruff format src/
uv run mypy src
uv run mypy tests
uv run pylint src
