"""
AgentHarm Dataset Loader.

Loads the AgentHarm benchmark dataset from HuggingFace or local cache.
"""

import json
import os
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
from huggingface_hub import snapshot_download


# Default cache path for AgentHarm dataset
AGENTHARM_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".cache",
    "agentharm_dataset",
    "AgentHarm"
)


@dataclass
class AgentHarmBehavior:
    """Represents a single behavior from the AgentHarm dataset."""
    id: str
    id_original: str
    name: str
    category: str
    prompt: str
    detailed_prompt: bool
    hint_included: bool
    target_functions: List[str]
    grading_function: str
    metadata: Dict


def load_agentharm_dataset(
    task_name: Literal["harmful", "benign", "chat"] = "harmful",
    split: Literal["val", "test_public", "test_private"] = "test_public",
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
    behavior_ids: Optional[List[str]] = None,
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
) -> List[AgentHarmBehavior]:
    """
    Load AgentHarm dataset from HuggingFace or local cache.

    Args:
        task_name: Type of behaviors to load ("harmful", "benign", or "chat")
        split: Dataset split ("val", "test_public", or "test_private")
        cache_dir: Optional custom cache directory
        force_redownload: Force download even if cached
        behavior_ids: Optional list of specific behavior IDs to load
        detailed_behaviors: If True, filter to detailed behaviors only
        hint_included: If True, filter to hint-included behaviors only

    Returns:
        List of AgentHarmBehavior objects
    """
    cache_path = cache_dir or AGENTHARM_CACHE_PATH

    # File paths for each split
    split_files = {
        "harmful": {
            "val": "benchmark/harmful_behaviors_validation.json",
            "test_public": "benchmark/harmful_behaviors_test_public.json",
            "test_private": "benchmark/harmful_behaviors_test_private.json",
        },
        "benign": {
            "val": "benchmark/benign_behaviors_validation.json",
            "test_public": "benchmark/benign_behaviors_test_public.json",
            "test_private": "benchmark/benign_behaviors_test_private.json",
        },
        "chat": {
            "val": "benchmark/chat_validation.json",
            "test_public": "benchmark/chat_public_test.json",
            "test_private": "benchmark/chat_private_test.json",
        },
    }

    if task_name not in split_files:
        raise ValueError(f"Invalid task_name: {task_name}. Must be 'harmful', 'benign', or 'chat'")
    if split not in split_files[task_name]:
        raise ValueError(f"Invalid split: {split}. Must be 'val', 'test_public', or 'test_private'")

    # Download if needed
    benchmark_dir = os.path.join(cache_path, "benchmark")
    if not os.path.exists(benchmark_dir) or force_redownload:
        print(f"[AgentHarm] Downloading dataset from HuggingFace...")
        os.makedirs(cache_path, exist_ok=True)
        try:
            snapshot_download(
                repo_id="ai-safety-institute/AgentHarm",
                repo_type="dataset",
                local_dir=cache_path,
                allow_patterns="**/*.json",
                resume_download=True,
            )
            print(f"[AgentHarm] Dataset downloaded to: {cache_path}")
        except Exception as e:
            print(f"[AgentHarm] Error downloading dataset: {e}")
            raise

    # Load the JSON file
    json_path = os.path.join(cache_path, split_files[task_name][split])
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Parse behaviors
    behaviors = []
    for item in data.get("behaviors", []):
        behavior = AgentHarmBehavior(
            id=item.get("id", ""),
            id_original=item.get("id_original", ""),
            name=item.get("name", ""),
            category=item.get("category", ""),
            prompt=item.get("prompt", ""),
            detailed_prompt=item.get("detailed_prompt", False),
            hint_included=item.get("hint_included", False),
            target_functions=item.get("target_functions", []),
            grading_function=item.get("grading_function", ""),
            metadata=item,
        )
        behaviors.append(behavior)

    # Apply filters
    if behavior_ids:
        behaviors = [b for b in behaviors if b.id in behavior_ids]

    if detailed_behaviors is not None:
        behaviors = [b for b in behaviors if b.detailed_prompt == detailed_behaviors]

    if hint_included is not None:
        behaviors = [b for b in behaviors if b.hint_included == hint_included]

    print(f"[AgentHarm] Loaded {len(behaviors)} behaviors from {task_name}/{split}")
    return behaviors


def get_behavior_by_id(behaviors: List[AgentHarmBehavior], behavior_id: str) -> Optional[AgentHarmBehavior]:
    """Get a specific behavior by ID."""
    for b in behaviors:
        if b.id == behavior_id:
            return b
    return None


def get_behaviors_by_category(behaviors: List[AgentHarmBehavior], category: str) -> List[AgentHarmBehavior]:
    """Filter behaviors by category."""
    return [b for b in behaviors if b.category == category]


def list_categories(behaviors: List[AgentHarmBehavior]) -> List[str]:
    """List all unique categories in the dataset."""
    return list(set(b.category for b in behaviors))
