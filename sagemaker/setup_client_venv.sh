#!/usr/bin/env bash
# Create a small venv *outside* this repo for SageMaker submit/status/upload CLIs only.
# Avoids bloating sagemaker source_dir uploads when .venv lives under the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${MINIMIND_SM_VENV:-${HOME}/.venvs/minimind-sagemaker}"

mkdir -p "$(dirname "${VENV_PATH}")"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Creating venv at ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
else
  echo "Using existing venv at ${VENV_PATH}"
fi

"${VENV_PATH}/bin/pip" install -U pip
"${VENV_PATH}/bin/pip" install -r "${REPO_ROOT}/sagemaker/requirements-client.txt"

echo ""
echo "Activate before running submit/status/upload:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
echo "To use a different path: MINIMIND_SM_VENV=/path/to/venv bash sagemaker/setup_client_venv.sh"
