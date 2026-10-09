#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIRTUAL_ENV="${REPOSITORY_ROOT}/.venv"

if [[ ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "Isaac environment is missing. Run './setup_isaac.sh' from ${REPOSITORY_ROOT}." >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: ./run_isaac.sh python scripts/<script>.py [args...]" >&2
    exit 2
fi

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
cd "${REPOSITORY_ROOT}"

if [[ "$1" == "python" || "$1" == "python3" ]]; then
    shift
    exec "${VIRTUAL_ENV}/bin/python" "$@"
fi

exec "$@"
