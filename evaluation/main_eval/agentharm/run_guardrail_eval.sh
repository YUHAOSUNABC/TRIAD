#!/bin/bash
# AgentHarm with guardrail. Modes: offline vLLM (default) / vLLM serve (USE_SERVE=true).
# GPUs: 2 needed (agent + guardrail). API agent only needs 1.
# Env vars: TASK_NAME SPLIT GPUS GUARDRAIL_PATH GUARDRAIL_NAME USE_SERVE NUM_WORKERS AGENT_PORT GUARD_PORT USE_COT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

if [ -n "$LLM_PATH" ]; then
    LLM_PATHS=("$LLM_PATH")
else
    LLM_PATHS=(
        # "gpt-4o-2024-08-06"
        # "claude-sonnet-4-20250514"
        "${MODEL_DIR}/guardrail_model/Qwen2.5-7B-Instruct"
    )
fi

TASK_NAME="${TASK_NAME:-benign}"
SPLIT="${SPLIT:-val}"

GUARDRAIL_PATH="${GUARDRAIL_PATH:-${MODEL_DIR}/guardrail_model/TS-Guard}"
GUARDRAIL_NAME="${GUARDRAIL_NAME:-TS-Guard}"
MAX_GUARDRAIL_ATTEMPTS=3

GPUS="${GPUS:-2,3}"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.9

USE_SERVE="${USE_SERVE:-false}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AGENT_PORT="${AGENT_PORT:-8000}"
GUARD_PORT="${GUARD_PORT:-8100}"
USE_COT="${USE_COT:-true}"

is_api_model() {
    local m="$1"
    [[ "$m" == *"oss"* ]] && return 1
    [[ "$m" == *"/"* ]] && return 1
    [[ "$m" == "gpt-"* ]] || [[ "$m" == "claude-"* ]]
}
is_large_model() {
    [[ "$1" == *"70B"* || "$1" == *"70b"* || "$1" == *"72B"* || "$1" == *"72b"* ]]
}

FIRST_LLM_PATH="${LLM_PATHS[0]}"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if is_api_model "$(basename "$FIRST_LLM_PATH")"; then MIN_GPUS=1; else MIN_GPUS=2; fi
if [ "$NUM_GPUS" -lt "$MIN_GPUS" ]; then
    echo "Error: need at least $MIN_GPUS GPU(s) (got: $NUM_GPUS)"; exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
AGENT_GPU="${GPU_ARRAY[0]}"
GUARD_GPU="${GPU_ARRAY[$((NUM_GPUS - 1))]}"

echo "AgentHarm guardrail eval | task=$TASK_NAME split=$SPLIT guardrail=$GUARDRAIL_NAME use_cot=$USE_COT gpus=$GPUS"

mkdir -p "logs/${SPLIT}"

SERVE_PIDS=()
if [ "$USE_SERVE" = "true" ]; then
    cleanup() {
        for pid in "${SERVE_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
        wait 2>/dev/null
    }
    trap cleanup EXIT INT TERM

    wait_for_serve() {
        local port="$1" label="$2" max_wait=900 elapsed=0
        while [ $elapsed -lt $max_wait ]; do
            curl -s "http://localhost:${port}/health" > /dev/null 2>&1 && return 0
            sleep 5; elapsed=$((elapsed + 5))
        done
        echo "ERROR: $label not ready in ${max_wait}s"; return 1
    }

    AGENT_SERVE_STARTED=0
    for LLM_PATH_ITER in "${LLM_PATHS[@]}"; do
        if is_api_model "$(basename "$LLM_PATH_ITER")"; then continue; fi
        CUDA_VISIBLE_DEVICES="$AGENT_GPU" python -m vllm.entrypoints.openai.api_server \
            --model "$LLM_PATH_ITER" --port "$AGENT_PORT" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" --seed 42 --trust-remote-code \
            > "logs/vllm_serve_agent_port${AGENT_PORT}_${RUN_TS}.log" 2>&1 &
        SERVE_PIDS+=($!); AGENT_SERVE_STARTED=1
        break
    done

    CUDA_VISIBLE_DEVICES="$GUARD_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$GUARDRAIL_PATH" --port "$GUARD_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" --seed 42 --trust-remote-code \
        > "logs/vllm_serve_guard_port${GUARD_PORT}_${RUN_TS}.log" 2>&1 &
    SERVE_PIDS+=($!)

    [ "$AGENT_SERVE_STARTED" = "1" ] && { wait_for_serve "$AGENT_PORT" "agent" || exit 1; }
    wait_for_serve "$GUARD_PORT" "guardrail" || exit 1
fi

for LLM_PATH in "${LLM_PATHS[@]}"; do
    MODEL_NAME=$(basename "$LLM_PATH")
    LOG="logs/${SPLIT}/${MODEL_NAME}_${TASK_NAME}_guardrail_${GUARDRAIL_NAME}_$(date +%Y%m%d_%H%M%S).log"

    if [ "$USE_SERVE" = "true" ] && ! is_api_model "$MODEL_NAME"; then
        python run_eval.py \
            --llm_path "$LLM_PATH" \
            --agent_base_url "http://localhost:${AGENT_PORT}/v1" \
            --task_name $TASK_NAME --split $SPLIT \
            --max_rounds 20 --save_interval 5 \
            --defense_type guardrail \
            --guardrail_path "$GUARDRAIL_PATH" \
            --guardrail_name "$GUARDRAIL_NAME" \
            --guardrail_type vllm_serve \
            --guardrail_base_url "http://localhost:${GUARD_PORT}/v1" \
            --max_guardrail_attempts $MAX_GUARDRAIL_ATTEMPTS \
            --num_workers "$NUM_WORKERS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --use_cot "$USE_COT" \
            2>&1 | tee "$LOG"

    elif is_api_model "$MODEL_NAME"; then
        GUARDRAIL_TYPE_ARG=$( [ "$USE_SERVE" = "true" ] && echo "vllm_serve" || echo "vllm" )
        EXTRA_ARGS=""
        if [ "$USE_SERVE" = "true" ]; then
            EXTRA_ARGS="--guardrail_base_url http://localhost:${GUARD_PORT}/v1 --num_workers $NUM_WORKERS"
        fi
        export CUDA_VISIBLE_DEVICES="$GPUS"
        python run_eval.py \
            --llm_path "$LLM_PATH" \
            --task_name $TASK_NAME --split $SPLIT \
            --max_rounds 20 --save_interval 5 \
            --defense_type guardrail \
            --guardrail_path "$GUARDRAIL_PATH" \
            --guardrail_name "$GUARDRAIL_NAME" \
            --guardrail_type "$GUARDRAIL_TYPE_ARG" \
            --max_guardrail_attempts $MAX_GUARDRAIL_ATTEMPTS \
            --num_gpus $NUM_GPUS \
            --gpu_memory_utilization "$GPU_MEM_UTIL" \
            --max_model_len "$MAX_MODEL_LEN" \
            --use_cot "$USE_COT" \
            $EXTRA_ARGS \
            2>&1 | tee "$LOG"
    else
        if is_large_model "$MODEL_NAME"; then AGENT_TP_SIZE=$((NUM_GPUS - 1)); else AGENT_TP_SIZE=1; fi
        export CUDA_VISIBLE_DEVICES="$GPUS"
        python run_eval.py \
            --llm_path "$LLM_PATH" \
            --task_name $TASK_NAME --split $SPLIT \
            --max_rounds 20 --save_interval 5 \
            --defense_type guardrail \
            --guardrail_path "$GUARDRAIL_PATH" \
            --guardrail_name "$GUARDRAIL_NAME" \
            --guardrail_type vllm \
            --max_guardrail_attempts $MAX_GUARDRAIL_ATTEMPTS \
            --use_vllm --num_gpus $NUM_GPUS \
            --tensor_parallel_size "$AGENT_TP_SIZE" \
            --gpu_memory_utilization "$GPU_MEM_UTIL" \
            --max_model_len "$MAX_MODEL_LEN" \
            --use_cot "$USE_COT" \
            2>&1 | tee "$LOG"
    fi
done

echo "All evaluations completed."
