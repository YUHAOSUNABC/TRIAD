#!/bin/bash
# AgentHarm evaluation. Modes: API / vLLM serve (default) / offline vLLM (USE_OFFLINE_VLLM=true).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

LLM_PATHS=(
    # "gpt-4o-2024-08-06"
    # "claude-opus-4-6"
    "${MODEL_DIR}/guardrail_model/Qwen2.5-7B-Instruct"
)

TASK_NAMES=("harmful" "benign")
SPLIT="test_public"
DEFENSE_TYPE="no_defense"
GUARDRAIL_PATH=""

NUM_WORKERS=20
GPUS="0"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.9

USE_OFFLINE_VLLM="${USE_OFFLINE_VLLM:-false}"
AGENT_PORT="${AGENT_PORT:-8000}"

is_api_model() {
    local m="$1"
    [[ "$m" == *"oss"* ]] && return 1
    [[ "$m" == *"/"* ]] && return 1
    [[ "$m" == "gpt-"* ]] || [[ "$m" == "claude-"* ]]
}
is_large_model()  { [[ "$1" == *"72B"* || "$1" == *"72b"* ]]; }
is_medium_model() { [[ "$1" == *"32B"* || "$1" == *"32b"* ]]; }

export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "Using GPUs: $GPUS (count: $NUM_GPUS)"
mkdir -p "logs/${SPLIT}"

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

    if ! is_api_model "$MODEL_NAME" && [ "$USE_OFFLINE_VLLM" != "true" ]; then
        if   is_large_model  "$MODEL_NAME"; then TP_SIZE=4
        elif is_medium_model "$MODEL_NAME"; then TP_SIZE=2
        else TP_SIZE=1; fi

        SERVE_LOG="logs/vllm_serve_${MODEL_NAME}_port${AGENT_PORT}_${RUN_TS}.log"
        python -m vllm.entrypoints.openai.api_server \
            --model "$LLM_PATH" --port "$AGENT_PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --trust-remote-code > "$SERVE_LOG" 2>&1 &
        SERVE_PID=$!
        wait_for_serve "$AGENT_PORT" || { echo "Aborting."; exit 1; }
    fi

    for TASK_NAME in "${TASK_NAMES[@]}"; do
        LOG="logs/${SPLIT}/${MODEL_NAME}_${TASK_NAME}_${DEFENSE_TYPE}_$(date +%Y%m%d_%H%M%S).log"

        if is_api_model "$MODEL_NAME"; then
            python run_eval.py \
                --llm_path "$LLM_PATH" \
                --task_name $TASK_NAME --split $SPLIT \
                --max_rounds 20 --save_interval 5 \
                --defense_type $DEFENSE_TYPE \
                --num_workers $NUM_WORKERS \
                ${GUARDRAIL_PATH:+--guardrail_path $GUARDRAIL_PATH} \
                2>&1 | tee "$LOG"

        elif [ "$USE_OFFLINE_VLLM" = "true" ]; then
            if   is_large_model  "$MODEL_NAME"; then TP_SIZE=4
            elif is_medium_model "$MODEL_NAME"; then TP_SIZE=2
            else TP_SIZE=1; fi
            python run_eval.py \
                --llm_path "$LLM_PATH" \
                --task_name $TASK_NAME --split $SPLIT \
                --max_rounds 20 --save_interval 5 \
                --defense_type $DEFENSE_TYPE \
                --use_vllm \
                --gpu_memory_utilization "$GPU_MEM_UTIL" \
                --tensor_parallel_size "$TP_SIZE" \
                --max_model_len "$MAX_MODEL_LEN" \
                ${GUARDRAIL_PATH:+--guardrail_path $GUARDRAIL_PATH} \
                2>&1 | tee "$LOG"

        else
            python run_eval.py \
                --llm_path "$LLM_PATH" \
                --agent_base_url "http://localhost:${AGENT_PORT}/v1" \
                --task_name $TASK_NAME --split $SPLIT \
                --max_rounds 20 --save_interval 5 \
                --defense_type $DEFENSE_TYPE \
                --num_workers $NUM_WORKERS \
                --max_model_len "$MAX_MODEL_LEN" \
                ${GUARDRAIL_PATH:+--guardrail_path $GUARDRAIL_PATH} \
                2>&1 | tee "$LOG"
        fi
    done

    if [ -n "$SERVE_PID" ]; then
        kill "$SERVE_PID" 2>/dev/null
        wait "$SERVE_PID" 2>/dev/null
        SERVE_PID=""
    fi
done

echo "All evaluations completed."
