#!/bin/bash
#SBATCH --job-name=chatarena-qwen35-games
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/data5/jaehoonjeong/DS_proj/ChatArena/logs/qwen35_%j.out
#SBATCH --error=/data5/jaehoonjeong/DS_proj/ChatArena/logs/qwen35_%j.err

set -euo pipefail
cd /data5/jaehoonjeong/DS_proj/ChatArena
export FLASHINFER_WORKSPACE_BASE=/data5/jaehoonjeong
export XDG_CACHE_HOME=/data5/jaehoonjeong/.cache
export XDG_CONFIG_HOME=/data5/jaehoonjeong/.config
export TRITON_CACHE_DIR=/data5/jaehoonjeong/.cache/triton
export TRITON_DUMP_DIR=/data5/jaehoonjeong/.cache/triton/dump
export TRITON_OVERRIDE_DIR=/data5/jaehoonjeong/.cache/triton/override
export TORCH_HOME=/data5/jaehoonjeong/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/data5/jaehoonjeong/.cache/torch/inductor
export VLLM_CACHE_ROOT=/data5/jaehoonjeong/.cache/vllm
export VLLM_CONFIG_ROOT=/data5/jaehoonjeong/.config/vllm
export CUDA_CACHE_PATH=/data5/jaehoonjeong/.cache/cuda
export NUMBA_CACHE_DIR=/data5/jaehoonjeong/.cache/numba
export MPLCONFIGDIR=/data5/jaehoonjeong/.cache/matplotlib
export HF_HUB_CACHE=/data5/jaehoonjeong/.cache/huggingface/hub
export HF_ASSETS_CACHE=/data5/jaehoonjeong/.cache/huggingface/assets
export HF_DATASETS_CACHE=/data5/jaehoonjeong/.cache/huggingface/datasets
export TMPDIR=/data5/jaehoonjeong/DS_proj/ChatArena/outputs/runtime_tmp
export TMP=/data5/jaehoonjeong/DS_proj/ChatArena/outputs/runtime_tmp
export TEMP=/data5/jaehoonjeong/DS_proj/ChatArena/outputs/runtime_tmp
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONFAULTHANDLER=1
export OPENAI_API_KEY=local-model
export ARROW_DEFAULT_MEMORY_POOL=system
export MALLOC_CONF=background_thread:false
export PYTHONPATH=/data5/jaehoonjeong/DS_proj/ChatArena/runtime_init:/data5/jaehoonjeong/DS_proj/ChatArena
PYTHON="$PWD/.venv-qwen35/bin/python"
PORT=$((21000 + SLURM_JOB_ID % 1000))
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
OUTPUT="$PWD/outputs/qwen35_pilot/job_$SLURM_JOB_ID"
mkdir -p "$OUTPUT" "$TMPDIR"
echo "Compute node: $(hostname); job: $SLURM_JOB_ID; output: $OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" scripts/start_qwen35_server.py \
    --model Qwen/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
    --language-model-only --max-model-len 16384 --max-num-seqs 8 \
    --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"\nI have finished reasoning. I will now output only the requested final response, with no analysis, speaker prefix, or visibility labels.\n</think>\n\n"}' \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    >"$OUTPUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
READY=0
for _ in $(seq 1 360); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited before readiness; see $OUTPUT/server.log" >&2
        exit 1
    fi
    if "$PYTHON" -c 'import os, urllib.request; urllib.request.urlopen(os.environ["OPENAI_BASE_URL"] + "/models", timeout=2).close()' 2>/dev/null; then
        READY=1
        break
    fi
    sleep 5
done
[[ "$READY" == 1 ]] || { echo "Server readiness timed out" >&2; exit 1; }
"$PYTHON" scripts/run_local_game_logs.py --game liar_game --players 5 --clue-rounds 1 \
    --model Qwen/Qwen3.5-9B --base-url "$OPENAI_BASE_URL" \
    --seeds 1 2 3 4 5 6 7 8 9 10 --enable-thinking --max-tokens 4096 --thinking-token-budget 2048 --control-thinking-token-budget 256 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repetition-penalty 1.0 \
    --request-timeout 300 --output "$OUTPUT/liar_game"
if [[ "${CHAT_ARENA_LIAR_ONLY:-0}" == "1" ]]; then
    echo "Liar Game pilot completed: $OUTPUT"
    exit 0
fi
"$PYTHON" scripts/run_local_game_logs.py --game mafia --players 7 \
    --model Qwen/Qwen3.5-9B --base-url "$OPENAI_BASE_URL" \
    --seeds 1 2 3 --enable-thinking --max-tokens 4096 --thinking-token-budget 2048 --control-thinking-token-budget 256 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repetition-penalty 1.0 \
    --request-timeout 300 --output "$OUTPUT/mafia"
echo "Both game pilots completed: $OUTPUT"
