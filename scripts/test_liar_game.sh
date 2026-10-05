#!/bin/bash
#SBATCH --job-name=liar-game-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-liar-game-test.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-liar-game-test.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
RUNTIME_ENV="/home/solbeecho/data/conda_envs/chatarena_qwen35"

export HF_HOME="/home/solbeecho/data/datasets/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"
export OPENAI_API_KEY="local-vllm"

cd "$PROJECT_DIR"
mkdir -p logs
srun "$RUNTIME_ENV/bin/python" -m unittest \
    tests.unit.test_liar_game tests.unit.test_liar_game_gradio -v
