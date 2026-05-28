#!/bin/bash
# ASB with guardrail. Runs DPI + OPI in parallel (4 GPUs).
# Usage:  bash run_guardrail_eval.sh [attack_type]
# Env:    GUARDRAIL_PATH GUARDRAIL_NAME AGENT_PATH NUM_WORKERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

GUARDRAIL_PATH="${GUARDRAIL_PATH:-${MODEL_DIR}/guardrail_model/TS-Guard}"
GUARDRAIL_NAME="${GUARDRAIL_NAME:-TS-Guard}"
AGENT_PATH="${AGENT_PATH:-${MODEL_DIR}/guardrail_model/Qwen2.5-7B-Instruct}"

if [ -n "$1" ]; then
    ATTACK_TYPES=("$1")
else
    ATTACK_TYPES=("combined_attack" "context_ignoring")
fi

NUM_WORKERS="${NUM_WORKERS:-20}"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.9

DPI_AGENT_GPU="${DPI_AGENT_GPU:-0}"
DPI_GUARD_GPU="${DPI_GUARD_GPU:-1}"
OPI_AGENT_GPU="${OPI_AGENT_GPU:-2}"
OPI_GUARD_GPU="${OPI_GUARD_GPU:-3}"

DPI_AGENT_PORT=8000; DPI_GUARD_PORT=8100
OPI_AGENT_PORT=8200; OPI_GUARD_PORT=8300

LOG_DIR="logs/full"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME=$(basename "$AGENT_PATH")

echo "ASB guardrail eval | agent=$MODEL_NAME guardrail=$GUARDRAIL_NAME workers=$NUM_WORKERS attacks=${ATTACK_TYPES[*]}"

SERVE_PIDS=()
cleanup() {
    for pid in "${SERVE_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

start_serve() {
    local model="$1" gpu="$2" port="$3" label="$4"
    CUDA_VISIBLE_DEVICES="$gpu" python -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" --seed 42 --trust-remote-code \
        > "$LOG_DIR/vllm_serve_${label}_port${port}_${RUN_TS}.log" 2>&1 &
    SERVE_PIDS+=($!)
}

wait_for_serve() {
    local port="$1" label="$2" max_wait=900 elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        curl -s "http://localhost:${port}/health" > /dev/null 2>&1 && return 0
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "ERROR: $label not ready in ${max_wait}s"; return 1
}

start_serve "$AGENT_PATH"     "$DPI_AGENT_GPU" "$DPI_AGENT_PORT" "dpi_agent"
start_serve "$GUARDRAIL_PATH" "$DPI_GUARD_GPU" "$DPI_GUARD_PORT" "dpi_guard"
start_serve "$AGENT_PATH"     "$OPI_AGENT_GPU" "$OPI_AGENT_PORT" "opi_agent"
start_serve "$GUARDRAIL_PATH" "$OPI_GUARD_GPU" "$OPI_GUARD_PORT" "opi_guard"

wait_for_serve "$DPI_AGENT_PORT" "dpi_agent" || exit 1
wait_for_serve "$DPI_GUARD_PORT" "dpi_guard" || exit 1
wait_for_serve "$OPI_AGENT_PORT" "opi_agent" || exit 1
wait_for_serve "$OPI_GUARD_PORT" "opi_guard" || exit 1

run_eval() {
    local cfg_path="$1" agent_port="$2" guard_port="$3" log_suffix="$4"
    local cfg_name=$(basename "$cfg_path" .yml)
    for attack_type in "${ATTACK_TYPES[@]}"; do
        python run_eval.py \
            --cfg_path "$cfg_path" \
            --llm_path "$AGENT_PATH" \
            --agent_base_url "http://localhost:${agent_port}/v1" \
            --agent_type "advanced_react" \
            --defense_type "guardrail" \
            --guardrail_path "$GUARDRAIL_PATH" \
            --guardrail_name "$GUARDRAIL_NAME" \
            --guardrail_type "vllm_serve" \
            --guardrail_base_url "http://localhost:${guard_port}/v1" \
            --attack_type "$attack_type" \
            --save_interval 10 \
            --max_guardrail_attempts 3 \
            --num_workers "$NUM_WORKERS" \
            --max_model_len "$MAX_MODEL_LEN" \
            2>&1 | tee "${LOG_DIR}/${MODEL_NAME}_${GUARDRAIL_NAME}_${cfg_name}_${attack_type}_${TIMESTAMP}.log"
    done
}

run_eval "config/DPI.yml" "$DPI_AGENT_PORT" "$DPI_GUARD_PORT" "DPI" &
DPI_PID=$!
run_eval "config/OPI.yml" "$OPI_AGENT_PORT" "$OPI_GUARD_PORT" "OPI" &
OPI_PID=$!

wait $DPI_PID && echo "DPI completed" || echo "DPI failed ($?)"
wait $OPI_PID && echo "OPI completed" || echo "OPI failed ($?)"

echo "All ASB evaluations completed."
