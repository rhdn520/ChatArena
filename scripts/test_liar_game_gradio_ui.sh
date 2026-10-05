#!/bin/bash
#SBATCH --job-name=liar-game-gradio-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/solbeecho/data/ChatArena/logs/%j-liar-game-gradio-smoke.out
#SBATCH --error=/home/solbeecho/data/ChatArena/logs/%j-liar-game-gradio-smoke.err

set -euo pipefail

PROJECT_DIR="/home/solbeecho/data/ChatArena"
UI_ENV="/home/solbeecho/data/conda_envs/chatarena_gradio"
PORT="$((23000 + SLURM_JOB_ID % 1000))"
export HF_HOME="/home/solbeecho/data/datasets/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/home/solbeecho/data/.cache"
export GRADIO_TEMP_DIR="$PROJECT_DIR/outputs/gradio_tmp"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY="local-vllm"

cd "$PROJECT_DIR"
mkdir -p logs "$GRADIO_TEMP_DIR"
"$UI_ENV/bin/python" -m unittest \
    tests.unit.test_liar_game_gradio.TestLiarGameGradio.test_pilot_config_replaces_one_backend_with_human -v
SERVER_LOG="$PROJECT_DIR/logs/${SLURM_JOB_ID}-liar-game-gradio-smoke-server.log"
"$UI_ENV/bin/python" scripts/run_liar_game_gradio.py \
    --config examples/liar_game_qwen3_5_9b.json --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    if curl --silent --fail --max-time 3 "http://127.0.0.1:${PORT}/" >/dev/null; then
        echo "Gradio UI HTTP smoke test passed on port $PORT"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Gradio server exited early; see $SERVER_LOG" >&2
        exit 1
    fi
    sleep 2
done

echo "Gradio server did not become ready; see $SERVER_LOG" >&2
exit 1
