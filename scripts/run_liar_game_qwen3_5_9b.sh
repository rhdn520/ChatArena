#!/bin/bash
#SBATCH --job-name=dsp-chatarena-liar-qwen3.5-9b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-liar-game-qwen3.5-9b.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-liar-game-qwen3.5-9b.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
MODEL_CACHE="/home/solbeecho/data/datasets/huggingface_cache/hub/models--Qwen--Qwen3.5-9B"
MODEL_REVISION="$(<"$MODEL_CACHE/refs/main")"
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_REVISION"
SERVED_MODEL="Qwen/Qwen3.5-9B"
PILOT_SEEDS="${PILOT_SEEDS:-20261002}"
MODEL_SEED="${MODEL_SEED:-20261002}"
PORT="$((19000 + SLURM_JOB_ID % 1000))"
OUTPUT_DIR="$PROJECT_DIR/outputs/liar_game_qwen3_5_9b"
RUNTIME_ENV="/home/solbeecho/data/conda_envs/chatarena_qwen35"

if [[ ! -x "$RUNTIME_ENV/bin/vllm" ]]; then
    echo "Missing runtime. Submit scripts/create_qwen3_5_runtime.sh first." >&2
    exit 1
fi

export HF_HOME="/home/solbeecho/data/datasets/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="local-vllm"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

SERVER_LOG="$OUTPUT_DIR/${SLURM_JOB_ID}-server.log"
"$RUNTIME_ENV/bin/vllm" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --seed "$MODEL_SEED" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

SERVER_READY=0
for _ in $(seq 1 180); do
    if curl --silent --fail "$OPENAI_BASE_URL/models" >/dev/null; then
        SERVER_READY=1
        break
    fi
    sleep 5
done
if [[ "$SERVER_READY" -ne 1 ]]; then
    echo "vLLM server did not become ready. See $SERVER_LOG" >&2
    exit 1
fi

for seed in $PILOT_SEEDS; do
    HISTORY="$OUTPUT_DIR/seed_${seed}_history.json"
    SUMMARY="$OUTPUT_DIR/seed_${seed}_summary.json"
    "$RUNTIME_ENV/bin/python" scripts/run_liar_game_pilot.py \
        --config examples/liar_game_qwen3_5_9b.json \
        --seed "$seed" \
        --history "$HISTORY" \
        --summary "$SUMMARY"
done
