#!/usr/bin/env bash
# setup_tier1.sh — Create the synergy_new conda env on Tier1 (Dodrio)
#
# Run once from the project root on a login/interactive node:
#   bash env/setup_tier1.sh
#
# Assumes conda is already initialised in the shell (source conda.sh first if not).

set -euo pipefail

CONDA_SOURCE="/dodrio/scratch/projects/2026_029/kkontras/env/Anaconda/etc/profile.d/conda.sh"
ENV_PATH="/dodrio/scratch/projects/2026_029/kkontras/envs/synergy_new"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements_tier1.txt"

echo "=== Sourcing conda ==="
source "${CONDA_SOURCE}"

echo "=== Creating conda env at ${ENV_PATH} ==="
conda create -y -p "${ENV_PATH}" python=3.10

echo "=== Activating env ==="
conda activate "${ENV_PATH}"

echo "=== Installing PyTorch 2.9.1+cu128 ==="
pip install \
    torch==2.9.1+cu128 \
    torchvision==0.24.1+cu128 \
    torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing remaining requirements ==="
pip install -r "${REQUIREMENTS}"

echo "=== Done ==="
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU available:', torch.cuda.is_available())"
