#!/usr/bin/env bash
#
# Run commands
#
set -e

uv run --no-sync python -m src "$@"
