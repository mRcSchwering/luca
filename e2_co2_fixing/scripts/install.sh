#!/usr/bin/env bash
#
# Installs all dependencies for local development and tests
# (project, dev, others)
#
#   bash scripts/install.sh
#
uv sync --all-groups "$@" 