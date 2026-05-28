# Agent-Guardrail — Reproduction Package

Anonymized source code accompanying the paper. This package contains:

- `evaluation/main_eval/{agentharm,asb}/run_eval.py` — the two evaluation entrypoints
- `train/sft_trainer.py` and `format_sft_data.py` — SFT training pipeline
- `data/train/sft_training_data.json` — the curated SFT training set
- `models/` — empty placeholders (see `models/README.md` for what to drop in)
- `guardrail_prompts.py` — shared system prompts used by the guardrails
- `requirements.txt`, `accelerate_config.yaml`, `deepspeed_zero{2,3}.json` — runtime config

## Quickstart

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Additional runtime deps not in requirements.txt (install as needed):
#   pip install vllm anthropic python-dotenv pyyaml huggingface_hub inspect_evals

# 2. API keys (only needed if running closed-source models or LLM-as-judge)
cp .env.example .env  # then edit with your own keys

# 3. Model weights (place in models/ — see models/README.md)
export MODEL_DIR="$PWD/models"

# 4. Run AgentHarm evaluation
cd evaluation/main_eval/agentharm
bash run_eval.sh             # no-defense baseline
bash run_guardrail_eval.sh   # with guardrail

# 5. Run ASB evaluation
cd ../asb
bash run_eval.sh
bash run_guardrail_eval.sh
```

## Reproducing training

```bash
accelerate launch --config_file accelerate_config.yaml \
    train/sft_trainer.py \
    --model_path "$MODEL_DIR/Qwen3.5-9B" \
    --dataset_path data/train/sft_training_data.json \
    --output_dir "$MODEL_DIR/guardrail_model/TS-Guard" \
    --num_train_epochs 3 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --deepspeed deepspeed_zero3.json
```

## What's not in this package

Model weights are NOT included (size + third-party licensing). See `models/README.md` for download instructions. Logs, intermediate outputs, and result snapshots from our runs are also stripped — reviewers regenerate them by running `run_eval.sh`.
