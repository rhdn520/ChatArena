#!/bin/bash
#SBATCH --job-name=download-qwen3.5-9b
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-download-qwen3.5-9b.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-download-qwen3.5-9b.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
HF_CACHE="/home/solbeecho/data/datasets/huggingface_cache"

source /home/solbeecho/data/.bashrc
source /home/solbeecho/data/miniconda3/etc/profile.d/conda.sh
conda activate mhv_env

export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

cd "$PROJECT_DIR"
mkdir -p logs "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

srun hf download Qwen/Qwen3.5-9B \
    --cache-dir "$HUGGINGFACE_HUB_CACHE"
