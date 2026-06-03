<div align="center">

# 🛡️ TRIAD

### From Risk Classification to Action Plan Remediation:<br>A Guardrail Feedback-Driven Framework for LLM Agents

[Yuhao Sun](#)<sup>1</sup>&nbsp;·&nbsp;
[Jiacheng Zhang](#)<sup>1</sup>&nbsp;·&nbsp;
[Shaanan Cohney](#)<sup>1</sup>&nbsp;·&nbsp;
[Zhexin Zhang](#)<sup>2</sup>&nbsp;·&nbsp;
[Feng Liu](#)<sup>1</sup>&nbsp;·&nbsp;
[Xingliang Yuan](mailto:xingliang.yuan@unimelb.edu.au)<sup>1,†</sup>

<sup>1</sup>The University of Melbourne&nbsp;&nbsp;·&nbsp;&nbsp;<sup>2</sup>Tsinghua University<br>
<sup>†</sup>Corresponding author

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?logo=adobeacrobatreader&logoColor=white)](docs/paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b?logo=arxiv&logoColor=white)](#)
[![Project Page](https://img.shields.io/badge/Project-Page-1f5fff?logo=githubpages&logoColor=white)](https://yuhaosunabc.github.io/TRIAD/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](#)

</div>

<p align="center">
  <img src="docs/assets/pipeline.png" width="92%" alt="Overview of the TRIAD pipeline">
</p>

<p align="center"><em>
  At each planning step, <b>Tri-Guard</b> inspects the agent's proposed action and returns natural-language
  feedback plus a three-way decision — <b>Proceed</b>, <b>Update</b>, or <b>Refuse</b> — before any tool is executed.
</em></p>

---

## 📖 Overview

LLM-based guardrails usually safeguard agents by emitting **binary allow/deny** signals before execution. But agent
risks often arise when an otherwise **benign task is contaminated** by untrusted content or injected instructions —
and a binary guardrail blocks the whole task, sacrificing the legitimate goal.

**TRIAD** (*Tripartite Response for Iterative Agent Guardrailing*) is a guardrail-integrated agent framework that
turns guardrail outputs from static risk signals into **actionable verbal feedback**. We finetune a language model,
**Tri-Guard**, to produce structured natural-language feedback together with a three-way decision, and inject that
feedback back into the agent's context — forming a **closed loop** between guardrail feedback and agent planning.

## ✨ Key Idea — Three-Way Guardrail Decisions

| Decision | When | What happens |
|:--|:--|:--|
| 🟢 **Proceed** | the plan is safe and on-goal | execute the proposed action |
| 🟠 **Update** | the plan is *partially* unsafe | inject feedback → the agent **revises** its plan, dropping the harmful part while **preserving the benign task** |
| 🔴 **Refuse** | the request is purely harmful | block execution and refuse |

The **Update** decision is what lets TRIAD stop an attack *without* killing the user's legitimate task — the core
difference from allow-or-block guardrails.

## 📊 Results

Across four agent backbones (Qwen3-32B, Kimi-2.5, Gemini-2.5-Pro, GPT-5.1) on **ASB** and **AgentHarm**:

- 🛡️ Average **Attack Success Rate: 74.45% → 10.42%**
- ✅ Average **Task Success Rate: 28.45% → 68.60%**
- ⚖️ Best **Helpfulness–Safety score: 80.92** on AgentHarm

See the [**project page**](https://yuhaosunabc.github.io/TRIAD/) or the [**paper**](docs/paper.pdf) for the full tables and case studies.

## 📁 Repository Structure

```text
TRIAD/
├── evaluation/main_eval/
│   ├── agentharm/run_eval.py     # AgentHarm evaluation entrypoint
│   └── asb/run_eval.py           # ASB (ASB-DPI / ASB-IPI) evaluation entrypoint
├── train/sft_trainer.py          # weighted-SFT training for Tri-Guard
├── format_sft_data.py            # builds the SFT data from trajectories
├── data/train/sft_training_data.json   # curated SFT training set
├── guardrail_prompts.py          # shared guardrail system prompts
├── models/                       # weight placeholders (see models/README.md)
├── docs/                         # project page + paper PDF + figures
└── requirements.txt, accelerate_config.yaml, deepspeed_zero{2,3}.json
```

## 🛠️ Installation & Quickstart

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Additional runtime deps not in requirements.txt (install as needed):
#   pip install vllm anthropic python-dotenv pyyaml huggingface_hub inspect_evals

# 2. API keys (only needed for closed-source models or LLM-as-judge)
cp .env.example .env   # then edit with your own keys

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

## 🏋️ Reproducing Training

```bash
accelerate launch --config_file accelerate_config.yaml \
    train/sft_trainer.py \
    --model_path "$MODEL_DIR/Qwen3.5-9B" \
    --dataset_path data/train/sft_training_data.json \
    --output_dir "$MODEL_DIR/guardrail_model/Tri-Guard" \
    --num_train_epochs 3 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --deepspeed deepspeed_zero3.json
```

## 📝 Citation

```bibtex
@article{sun2026triad,
  title   = {From Risk Classification to Action Plan Remediation:
             A Guardrail Feedback-Driven Framework for LLM Agents},
  author  = {Sun, Yuhao and Zhang, Jiacheng and Cohney, Shaanan and
             Zhang, Zhexin and Liu, Feng and Yuan, Xingliang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 🙏 Acknowledgements

Our evaluation builds on [AgentSecurityBench (ASB)](https://github.com/agiresearch/ASB) and
[AgentHarm](https://github.com/UKGovernmentBEIS/inspect_evals). We thank the authors of these benchmarks.
