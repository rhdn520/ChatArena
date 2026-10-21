#!/bin/bash
#SBATCH --job-name=chatarena-qwen-debug
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/s3/lab07/solbeecho/ChatArena/logs/%j-mixed-qwen-debug.out
#SBATCH --error=/shared/s3/lab07/solbeecho/ChatArena/logs/%j-mixed-qwen-debug.err

set -euo pipefail
cd /shared/s3/lab07/solbeecho/ChatArena
export HF_HOME=/shared/s3/lab07/solbeecho/huggingface_cache
export XDG_CACHE_HOME="$PWD/outputs/cache"
export TMPDIR="$PWD/outputs/runtime_tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export CHAT_ARENA_QWEN_2B_BASE_URL=http://127.0.0.1:8001/v1
export CHAT_ARENA_QWEN_9B_BASE_URL=http://127.0.0.1:8002/v1
mkdir -p "$TMPDIR"
echo "DEBUG_NODE=$(hostname) JOB=$SLURM_JOB_ID GPT_API=blocked"
"$PWD/.venv-chatui/bin/python" -u scripts/debug_mixed_qwen.py
