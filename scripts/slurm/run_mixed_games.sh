#!/bin/bash
#SBATCH --job-name=chatarena-mixed-games
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_3090:2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH --output=/shared/s3/lab07/solbeecho/ChatArena/logs/%j-mixed-games.out
#SBATCH --error=/shared/s3/lab07/solbeecho/ChatArena/logs/%j-mixed-games.err

set -euo pipefail
cd /shared/s3/lab07/solbeecho/ChatArena

export HF_HOME=/shared/s3/lab07/solbeecho/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME=/shared/s3/lab07/solbeecho/ChatArena/outputs/cache
export XDG_CONFIG_HOME=/shared/s3/lab07/solbeecho/ChatArena/outputs/config
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TMPDIR=/shared/s3/lab07/solbeecho/ChatArena/outputs/runtime_tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TRITON_DUMP_DIR="$TRITON_CACHE_DIR/dump"
export TRITON_OVERRIDE_DIR="$TRITON_CACHE_DIR/override"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_HOME/inductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export CHAT_ARENA_QWEN_2B_MODEL="${CHAT_ARENA_QWEN_2B_MODEL:-Qwen/Qwen3.5-2B}"
export CHAT_ARENA_QWEN_9B_MODEL="${CHAT_ARENA_QWEN_9B_MODEL:-Qwen/Qwen3.5-9B}"
export CHAT_ARENA_GPT_MODEL="${CHAT_ARENA_GPT_MODEL:-openai/gpt-5.6-luna}"
export CHAT_ARENA_QWEN_2B_BASE_URL=http://127.0.0.1:8001/v1
export CHAT_ARENA_QWEN_9B_BASE_URL=http://127.0.0.1:8002/v1
export CHAT_ARENA_QWEN_API_KEY=local-model
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
# Match the existing pilot: avoid FlashInfer sampler's nvcc JIT dependency.
export VLLM_USE_FLASHINFER_SAMPLER=0
export ARROW_DEFAULT_MEMORY_POOL=system
export MALLOC_CONF=background_thread:false
export C_INCLUDE_PATH="$PWD/outputs/python-dev/usr/include/python3.12:$PWD/outputs/python-dev/usr/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TMPDIR" logs
OUTPUT="$PWD/outputs/mixed_games/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT"
PYTHON="$PWD/.venv-qwen35/bin/python"
UI_PYTHON="$PWD/.venv-chatui/bin/python"
# FlashInfer invokes ninja by name when building CUDA kernels.
export PATH="$PWD/.venv-qwen35/bin:$PATH"

CHILD_PIDS=()
cleanup() {
    if (( ${#CHILD_PIDS[@]} )); then
        kill "${CHILD_PIDS[@]}" 2>/dev/null || true
        sleep 2
        kill -KILL "${CHILD_PIDS[@]}" 2>/dev/null || true
        wait 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[MODEL_START] model=$CHAT_ARENA_QWEN_9B_MODEL port=8002 log=$OUTPUT/qwen35-9b-server.log"
"$PYTHON" scripts/start_qwen35_server.py \
    --model "$CHAT_ARENA_QWEN_9B_MODEL" --served-model-name "$CHAT_ARENA_QWEN_9B_MODEL" \
    --host 127.0.0.1 --port 8002 --dtype bfloat16 \
    --language-model-only --tensor-parallel-size 2 \
    --max-model-len 8192 --max-num-seqs 4 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.70 \
    --kv-cache-memory-bytes 2147483648 \
    --reasoning-parser qwen3 \
    >"$OUTPUT/qwen35-9b-server.log" 2>&1 &
QWEN_9B_PID=$!
CHILD_PIDS+=("$QWEN_9B_PID")

GPU_FOR_2B="${CUDA_VISIBLE_DEVICES%%,*}"
echo "[MODEL_START] model=$CHAT_ARENA_QWEN_2B_MODEL port=8001 log=$OUTPUT/qwen35-2b-server.log"
CUDA_VISIBLE_DEVICES="$GPU_FOR_2B" "$PYTHON" scripts/start_qwen35_server.py \
    --model "$CHAT_ARENA_QWEN_2B_MODEL" --served-model-name "$CHAT_ARENA_QWEN_2B_MODEL" \
    --host 127.0.0.1 --port 8001 --dtype bfloat16 \
    --language-model-only --max-model-len 8192 --max-num-seqs 8 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.18 \
    --kv-cache-memory-bytes 536870912 \
    --reasoning-parser qwen3 \
    >"$OUTPUT/qwen35-2b-server.log" 2>&1 &
QWEN_2B_PID=$!
CHILD_PIDS+=("$QWEN_2B_PID")

# Mirror both model logs to the Slurm output while retaining individual files.
tail -n +1 -F "$OUTPUT/qwen35-9b-server.log" "$OUTPUT/qwen35-2b-server.log" &
CHILD_PIDS+=("$!")

# Start the web UIs only after both local model servers finish initialization.
# Read their existing lifecycle logs; no inference or GPT API request is made.
STARTUP_DEADLINE=$((SECONDS + ${CHAT_ARENA_STARTUP_TIMEOUT:-2700}))
echo "Qwen 서버 초기화 중: 두 서버가 준비되면 게임 UI와 공개 주소를 생성합니다."
until grep -Fq 'Application startup complete' "$OUTPUT/qwen35-9b-server.log" && \
      grep -Fq 'Application startup complete' "$OUTPUT/qwen35-2b-server.log"; do
    if ! kill -0 "$QWEN_9B_PID" 2>/dev/null || ! kill -0 "$QWEN_2B_PID" 2>/dev/null; then
        echo "Qwen 서버 초기화 실패. 게임 UI는 시작하지 않습니다. 로그: $OUTPUT" >&2
        tail -n 35 "$OUTPUT/qwen35-9b-server.log" "$OUTPUT/qwen35-2b-server.log" >&2
        exit 1
    fi
    if (( SECONDS >= STARTUP_DEADLINE )); then
        echo "Qwen 서버 초기화 대기 시간 초과. 로그: $OUTPUT" >&2
        tail -n 35 "$OUTPUT/qwen35-9b-server.log" "$OUTPUT/qwen35-2b-server.log" >&2
        exit 1
    fi
    sleep 5
done
echo "[MODEL_READY] model=$CHAT_ARENA_QWEN_2B_MODEL port=8001"
echo "[MODEL_READY] model=$CHAT_ARENA_QWEN_9B_MODEL port=8002"
echo "Qwen 2B / 9B 서버 준비 완료. 게임 UI를 시작합니다."

"$UI_PYTHON" -u liar_app.py --port 8080 &
LIAR_UI_PID=$!
CHILD_PIDS+=("$LIAR_UI_PID")
"$UI_PYTHON" -u mafia_app.py --port 8081 &
MAFIA_UI_PID=$!
CHILD_PIDS+=("$MAFIA_UI_PID")

if wait -n "$LIAR_UI_PID" "$MAFIA_UI_PID" "$QWEN_2B_PID" "$QWEN_9B_PID"; then
    STATUS=0
else
    STATUS=$?
fi
echo "게임 UI 또는 모델 서버 종료 감지(exit=$STATUS). 두 UI를 종료합니다. 모델 서버 로그: $OUTPUT" >&2
if [[ "$STATUS" != 0 ]]; then
    tail -n 25 "$OUTPUT/qwen35-9b-server.log" "$OUTPUT/qwen35-2b-server.log" >&2
fi
exit "$STATUS"
