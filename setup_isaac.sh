#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPOSITORY_ROOT}"

for command in git uv awk; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "${command} is required to set up this repository." >&2
        exit 1
    fi
done

revision() {
    awk -F '"' -v key="$1" '
        /^\[tool\.igris_c\.revisions\]$/ { in_revisions = 1; next }
        in_revisions && /^\[/ { exit }
        in_revisions && $0 ~ "^" key " = " { print $2; exit }
    ' pyproject.toml
}

ensure_submodule() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${actual}" != "${expected}" ]]; then
        git submodule update --init "${path}"
        actual="$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)"
    fi
    if [[ "${actual}" != "${expected}" ]]; then
        echo "${path} is at ${actual:-unknown}; expected ${expected}. Update the root submodule pin." >&2
        exit 1
    fi
}

ensure_submodule third_party/IsaacLab "$(revision isaac_lab)"
ensure_submodule third_party/IsaacSim "$(revision isaac_sim)"

uv sync --locked
echo "Setup complete. Run ./run_isaac.sh python scripts/motion_tracking/igris_c_motion_tracking.py --help"
