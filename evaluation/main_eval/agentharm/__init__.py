"""
AgentHarm Benchmark Evaluation Package.

This package provides tools for evaluating agent + guardrail systems
on the AgentHarm benchmark (harmful and benign behaviors).

Components:
- dataset_loader: Load AgentHarm dataset from HuggingFace
- tool_adapter: Adapt inspect_ai tools to our format
- scorer: Score agent performance using AgentHarm grading functions
- run_eval: Main evaluation script
"""

from .dataset_loader import load_agentharm_dataset
from .tool_adapter import AgentHarmTool, create_tools_from_metadata
from .scorer import (
    AgentHarmScorer,
    RefusalJudgeLLM,
    SemanticJudge,
)

__all__ = [
    "load_agentharm_dataset",
    "AgentHarmTool",
    "create_tools_from_metadata",
    "AgentHarmScorer",
    "RefusalJudgeLLM",
    "SemanticJudge",
]
