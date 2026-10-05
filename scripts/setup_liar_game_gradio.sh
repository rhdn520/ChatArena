#!/bin/bash
#SBATCH --job-name=liar-game-gradio-setup
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-liar-game-gradio-setup.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-liar-game-gradio-setup.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
UI_ENV="/home/solbeecho/data/conda_envs/chatarena_gradio"
export CONDA_PKGS_DIRS="/home/solbeecho/data/conda_pkgs"
export CONDA_ENVS_PATH="/home/solbeecho/data/conda_envs"
export PIP_CACHE_DIR="/home/solbeecho/data/.cache/pip"
export HF_HOME="/home/solbeecho/data/datasets/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"

cd "$PROJECT_DIR"
mkdir -p logs "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" \
    "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
source /home/solbeecho/data/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "$UI_ENV/bin/python" ]]; then
    conda create -y -p "$UI_ENV" python=3.12 pip
fi
"$UI_ENV/bin/python" -m pip install \
    gradio==5.49.1 \
    openai==2.36.0 \
    tenacity==8.2.2 \
    rich \
    prompt_toolkit \
    'pettingzoo>=1.24.0' \
    chess==1.9.4 \
    rlcard==1.0.5 \
    'pygame>=2.3.0' \
    'gymnasium>=0.28.1'
"$UI_ENV/bin/python" -m pip check
"$UI_ENV/bin/python" -m pip show gradio
