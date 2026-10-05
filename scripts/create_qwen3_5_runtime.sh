#!/bin/bash
#SBATCH --job-name=qwen3.5-runtime
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-qwen3.5-runtime.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-qwen3.5-runtime.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
ENV_DIR="/home/solbeecho/data/conda_envs/chatarena_qwen35"

export CONDA_PKGS_DIRS="/home/solbeecho/data/conda_pkgs"
export PIP_CACHE_DIR="/home/solbeecho/data/.cache/pip"
export HF_HOME="/home/solbeecho/data/datasets/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"

source /home/solbeecho/data/.bashrc
source /home/solbeecho/data/miniconda3/etc/profile.d/conda.sh

mkdir -p "$PROJECT_DIR/logs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" \
    "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    conda create -y -p "$ENV_DIR" python=3.12 pip
fi

"$ENV_DIR/bin/python" -m pip install \
    vllm==0.19.1 \
    openai==2.36.0 \
    tenacity==8.2.2 \
    'pettingzoo>=1.24.0' \
    chess==1.9.4 \
    rlcard==1.0.5 \
    'pygame>=2.3.0' \
    'gymnasium>=0.28.1'

"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip show vllm
