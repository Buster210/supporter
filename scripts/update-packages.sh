#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
    cat <<'EOF'
Usage: scripts/update-packages.sh [dep|prek]

Modes:
  dep      Upgrade project dependencies only
  prek     Upgrade prek tooling only
  (none)   Upgrade everything (default)
EOF
}

refresh_prek_hooks() {
    echo "Updating prek hooks"
    uv run prek auto-update --config prek.toml
}

update_tooling_packages() {
    echo "Updating tooling packages"
    uv lock \
        --upgrade-package prek \
        --upgrade-package ruff \
        --upgrade-package mypy \
        --upgrade-package bandit \
        --upgrade-package detect-secrets
}

if [[ $# -gt 1 ]]; then
    usage
    exit 1
fi

mode="${1:-all}"

case "$mode" in
all)
    echo "Upgrading all packages"
    uv lock --upgrade
    refresh_prek_hooks
    ;;
dep)
    echo "Upgrading project dependencies"
    uv lock --upgrade
    ;;
prek)
    update_tooling_packages
    refresh_prek_hooks
    ;;
*)
    usage
    exit 1
    ;;
esac

clear