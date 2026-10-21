#!/bin/bash
#SBATCH --job-name=chatarena-mixed-env
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/shared/s3/lab07/solbeecho/ChatArena/logs/%j-mixed-env-setup.out
#SBATCH --error=/shared/s3/lab07/solbeecho/ChatArena/logs/%j-mixed-env-setup.err

set -euo pipefail
cd /shared/s3/lab07/solbeecho/ChatArena

export HF_HOME=/shared/s3/lab07/solbeecho/huggingface_cache
export PIP_CACHE_DIR=/shared/s3/lab07/solbeecho/ChatArena/outputs/pip-cache
export XDG_CACHE_HOME=/shared/s3/lab07/solbeecho/ChatArena/outputs/cache
export XDG_CONFIG_HOME=/shared/s3/lab07/solbeecho/ChatArena/outputs/config
export TMPDIR=/shared/s3/lab07/solbeecho/ChatArena/outputs/runtime_tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" logs outputs

# The previous checkout already has the two installed runtimes. Restore their
# original entry paths without moving or reinstalling the shared environments.
PREVIOUS_ROOT=/shared/s3/lab07/solbeecho/ChatArena-old
for environment in .venv-qwen35 .venv-chatui; do
    if [[ ! -e "$PWD/$environment" && -x "$PREVIOUS_ROOT/$environment/bin/python" ]]; then
        ln -s "$PREVIOUS_ROOT/$environment" "$PWD/$environment"
        echo "Reusing environment: $PREVIOUS_ROOT/$environment"
    fi
done
if [[ ! -e "$PWD/outputs/python-dev" && -d "$PREVIOUS_ROOT/outputs/python-dev" ]]; then
    ln -s "$PREVIOUS_ROOT/outputs/python-dev" "$PWD/outputs/python-dev"
fi

# Triton compiles a CUDA helper at server startup and needs Python C headers.
# Extract the development package locally; no system installation is needed.
PYTHON_DEV_ROOT="$PWD/outputs/python-dev"
if [[ ! -f "$PYTHON_DEV_ROOT/usr/include/python3.12/Python.h" ]]; then
    mkdir -p "$PWD/outputs/python-dev-debs" "$PYTHON_DEV_ROOT"
    (
        cd "$PWD/outputs/python-dev-debs"
        apt-get download libpython3.12-dev
        for archive in libpython3.12-dev_*.deb; do
            dpkg-deb -x "$archive" "$PYTHON_DEV_ROOT"
        done
    )
fi

PIP_BOOTSTRAP="$PWD/outputs/get-pip.py"
if [[ ! -x "$PWD/.venv-qwen35/bin/python" ]]; then
    python3 -m venv --without-pip .venv-qwen35
    PYTHON="$PWD/.venv-qwen35/bin/python"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$PIP_BOOTSTRAP"
    "$PYTHON" "$PIP_BOOTSTRAP"
    rm -f "$PIP_BOOTSTRAP"
    "$PYTHON" -m pip install --upgrade pip setuptools wheel

    # vLLM's CUDA 12.9 wheel provides the matching torch/CUDA runtime for RTX 3090.
    "$PYTHON" -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --extra-index-url https://wheels.vllm.ai/0.22.0/cu129 \
        'vllm==0.22.0' pyarrow ninja
fi

# Gradio 4.44.1's schema converter requires Pydantic <2.11. Keep it separate
# from vLLM, which requires Pydantic >=2.12.
if [[ ! -x "$PWD/.venv-chatui/bin/python" ]]; then
    python3 -m venv --without-pip .venv-chatui
    UI_PYTHON="$PWD/.venv-chatui/bin/python"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$PIP_BOOTSTRAP"
    "$UI_PYTHON" "$PIP_BOOTSTRAP"
    rm -f "$PIP_BOOTSTRAP"
    "$UI_PYTHON" -m pip install --upgrade pip setuptools wheel
    "$UI_PYTHON" -m pip install -r requirements-mixed-ui.txt
fi

echo "Environment installation completed: $PWD/.venv-qwen35/bin/python"
echo "UI environment installation completed: $PWD/.venv-chatui/bin/python"
