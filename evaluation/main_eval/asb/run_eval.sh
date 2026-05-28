#!/bin/bash
# ASB (Agent Safety Benchmark) evaluation. Iterates models x DPI/OPI configs x attack types.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

LLM_PATHS=(
    "${MODEL_DIR}/Qwen3-32B"
    # "${MODEL_DIR}/guardrail_model/Qwen2.5-7B-Instruct"
)

ATTACK_TYPES=(
    # "context_ignoring"
    "combined_attack"
)

CFG_PATHS=(
    "config/DPI.yml"
    # "config/OPI.yml"
)

NUM_WORKERS="${NUM_WORKERS:-20}"
AGENT_PORT="${AGENT_PORT:-8100}"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.9

is_large_model()  { [[ "$1" == *"72B"* || "$1" == *"72b"* ]]; }
is_medium_model() { [[ "$1" == *"32B"* || "$1" == *"32b"* ]]; }
is_api_model() {
    local m="$1"
    [[ "$m" == *"oss"* ]] && return 1
    [[ "$m" == *"/"* ]] && return 1
    [[ "$m" == "gpt-"* ]] || [[ "$m" == "claude-"* ]]
}

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

SERVE_PID=""
cleanup() {
    if [ -n "$SERVE_PID" ]; then
        kill "$SERVE_PID" 2>/dev/null
        wait "$SERVE_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

wait_for_serve() {
    local port="$1" max_wait=900 elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        curl -s "http://localhost:${port}/health" > /dev/null 2>&1 && return 0
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "vllm serve failed to start within ${max_wait}s"; return 1
}

for LLM_PATH in "${LLM_PATHS[@]}"; do
    MODEL_NAME=$(basename "$LLM_PATH")
    SERVE_PID=""

    if ! is_api_model "$MODEL_NAME"; then
        if   is_large_model  "$MODEL_NAME"; then TP_SIZE=4
        elif is_medium_model "$MODEL_NAME"; then TP_SIZE=2
        else TP_SIZE=1; fi

        SERVE_LOG="${LOG_DIR}/vllm_serve_${MODEL_NAME}_port${AGENT_PORT}_${RUN_TS}.log"
        python -m vllm.entrypoints.openai.api_server \
            --model "$LLM_PATH" --port "$AGENT_PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --trust-remote-code > "$SERVE_LOG" 2>&1 &
        SERVE_PID=$!
        wait_for_serve "$AGENT_PORT" || { echo "Aborting."; exit 1; }
    fi

    for CFG_PATH in "${CFG_PATHS[@]}"; do
        CFG_NAME=$(basename "$CFG_PATH" .yml)
        for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${CFG_NAME}_${ATTACK_TYPE}_${TIMESTAMP}.log"
            echo "Running ${MODEL_NAME} | ${CFG_NAME} | ${ATTACK_TYPE} | workers=${NUM_WORKERS}"

            if is_api_model "$MODEL_NAME"; then
                python run_eval.py \
                    --cfg_path "$CFG_PATH" \
                    --llm_path "$LLM_PATH" \
                    --agent_type "advanced_react" \
                    --save_interval 100 \
                    --attack_type "$ATTACK_TYPE" \
                    --num_workers "$NUM_WORKERS" \
                    2>&1 | tee "$LOG_FILE"
            else
                python run_eval.py \
                    --cfg_path "$CFG_PATH" \
                    --llm_path "$LLM_PATH" \
                    --agent_base_url "http://localhost:${AGENT_PORT}/v1" \
                    --agent_type "advanced_react" \
                    --save_interval 100 \
                    --attack_type "$ATTACK_TYPE" \
                    --num_workers "$NUM_WORKERS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    2>&1 | tee "$LOG_FILE"
            fi
        done
    done

    if [ -n "$SERVE_PID" ]; then
        kill "$SERVE_PID" 2>/dev/null
        wait "$SERVE_PID" 2>/dev/null
        SERVE_PID=""
    fi
done

echo "All evaluations completed."
