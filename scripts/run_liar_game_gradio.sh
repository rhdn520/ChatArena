#!/bin/bash
#SBATCH --job-name=dsp-chatarena-liar-human
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-liar-game-gradio.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-liar-game-gradio.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
RUNTIME_ENV="/home/solbeecho/data/conda_envs/chatarena_qwen35"
UI_ENV="/home/solbeecho/data/conda_envs/chatarena_gradio"
MODEL_CACHE="/home/solbeecho/data/datasets/huggingface_cache/hub/models--Qwen--Qwen3.5-9B"
MODEL_REVISION="$(<"$MODEL_CACHE/refs/main")"
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_REVISION"
MODEL_PORT="$((19000 + SLURM_JOB_ID % 1000))"
UI_PORT="$((21000 + SLURM_JOB_ID % 1000))"

if [[ ! -x "$RUNTIME_ENV/bin/vllm" || ! -x "$UI_ENV/bin/python" ]]; then
    echo "Missing vLLM or Gradio. Submit scripts/setup_liar_game_gradio.sh first." >&2
    exit 1
fi

export HF_HOME="/home/solbeecho/data/datasets/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"
export GRADIO_TEMP_DIR="/home/solbeecho/data/ChatArena/outputs/gradio_tmp"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="local-vllm"
export OPENAI_BASE_URL="http://127.0.0.1:${MODEL_PORT}/v1"

cd "$PROJECT_DIR"
mkdir -p logs "$GRADIO_TEMP_DIR"

SERVER_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}-liar-game-vllm.log"
"$RUNTIME_ENV/bin/vllm" serve "$MODEL_PATH" \
    --served-model-name "Qwen/Qwen3.5-9B" \
    --host 127.0.0.1 \
    --port "$MODEL_PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --seed 20261002 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
    if curl --silent --fail "$OPENAI_BASE_URL/models" >/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited. See $SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
if ! curl --silent --fail "$OPENAI_BASE_URL/models" >/dev/null; then
    echo "vLLM did not become ready. See $SERVER_LOG" >&2
    exit 1
fi

echo "Compute node: $(hostname)"
echo "Gradio port: $UI_PORT"
echo "From your computer, forward localhost:$UI_PORT on this compute node, then open http://127.0.0.1:$UI_PORT"
"$UI_ENV/bin/python" scripts/run_liar_game_gradio.py \
    --config examples/liar_game_qwen3_5_9b.json \
    --port "$UI_PORT"
