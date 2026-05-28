# Model Weights

This directory ships empty. Place model weights (HuggingFace `snapshot_download` layout: `config.json`, `tokenizer*`, `*.safetensors`, etc.) in the subdirectory whose name matches the path expected by `evaluation/main_eval/{agentharm,asb}/run_eval.sh`.

## Agent models (under `models/`)

| Subdir                 | Source                                                                  |
|------------------------|-------------------------------------------------------------------------|
| `Qwen3-32B/`           | `Qwen/Qwen3-32B` (HuggingFace)                                          |
| `Qwen3-32B-thinking/`  | `Qwen/Qwen3-32B` with `enable_thinking=True` in chat template           |
| `Qwen3.5-9B/`          | Your base agent (e.g. `Qwen/Qwen2.5-7B-Instruct`) renamed locally       |
| `gpt-oss-120b/`        | `openai/gpt-oss-120b` (HuggingFace)                                     |

## Guardrail models (under `models/guardrail_model/`)

| Subdir                       | Source                                                  |
|------------------------------|---------------------------------------------------------|
| `Qwen2.5-7B-Instruct/`       | `Qwen/Qwen2.5-7B-Instruct` (baseline, no guardrail)     |
| `Qwen3Guard-Gen-8B/`         | `Qwen/Qwen3Guard-Gen-8B`                                |
| `Safiron/`                   | `safiron-team/Safiron` (or the released checkpoint)     |
| `ShieldLM-14B-qwen/`         | `thu-coai/ShieldLM-14B-qwen`                            |
| `TS-Guard/`                  | TS-Guard checkpoint (released with this paper)          |
| `Tri-Guard/`                 | Tri-Guard checkpoint (released with this paper)         |
| `gpt-oss-safeguard-20b/`     | `openai/gpt-oss-safeguard-20b`                          |

After placing weights, either edit the `LLM_PATHS` / `GUARDRAIL_PATH` variables in the shell scripts (which are now expressed as `${MODEL_DIR:-./models/...}`) or simply `export MODEL_DIR=/absolute/path/to/models` before invoking.
