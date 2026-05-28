"""
Guardrail Factory - Create guardrail instances based on model type.

Usage:
    from guardrail_factory import create_guardrail

    # Create our fine-tuned guardrail
    guardrail = create_guardrail(
        model_path="/path/to/our/guardrail",
        guardrail_type="hf"
    )

    # Create Safiron guardrail
    guardrail = create_guardrail(
        model_path="Safiron/Safiron",
        guardrail_type="safiron"
    )

    # Use the guardrail
    result = guardrail.check(messages, agent_output, tools)
"""

from typing import Optional
from .guardrail_interface import GuardrailInterface, APIGuardrail


def create_guardrail(
    model_path: str,
    guardrail_type: str = "hf",
    max_new_tokens: int = 2048,
    api_key: Optional[str] = None,
    gpu_ids: Optional[list] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    base_url: Optional[str] = None,
    use_cot: bool = True,
) -> GuardrailInterface:
    """Create a guardrail instance based on the specified type.

    Args:
        model_path: Path to the guardrail model
        guardrail_type: Type of guardrail ("hf", "vllm", "vllm_serve", "safiron", "api")
        max_new_tokens: Maximum tokens for guardrail response
        api_key: API key for API-based guardrails (optional)
        gpu_ids: List of GPU IDs to use for the guardrail model (optional)
        tensor_parallel_size: Number of GPUs for tensor parallelism (vLLM only)
        gpu_memory_utilization: GPU memory utilization ratio (vLLM only)
        max_model_len: Maximum model context length (vLLM only)
        base_url: Base URL for vLLM serve endpoint (vllm_serve only)
        use_cot: Whether to use COT (chain-of-thought) system prompt (default: True)

    Returns:
        GuardrailInterface instance

    Raises:
        ValueError: If guardrail_type is not supported
        NotImplementedError: If API guardrail is requested
    """
    guardrail_type = guardrail_type.lower()

    if guardrail_type == "hf":
        from .hf_guardrail import HFGuardrail
        return HFGuardrail(model_path, max_new_tokens, gpu_ids=gpu_ids, use_cot=use_cot)

    elif guardrail_type == "vllm":
        from .vllm_guardrail import VLLMGuardrail
        return VLLMGuardrail(
            model_path,
            max_new_tokens=max_new_tokens,
            gpu_ids=gpu_ids,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            use_cot=use_cot,
        )

    elif guardrail_type == "vllm_serve":
        from .vllm_guardrail import VLLMServeGuardrail
        if not base_url:
            raise ValueError("base_url is required for vllm_serve guardrail type")
        return VLLMServeGuardrail(
            model_path,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            use_cot=use_cot,
        )

    elif guardrail_type == "safiron":
        from .safiron_guardrail import SafironGuardrail
        return SafironGuardrail(model_path, max_new_tokens)

    elif guardrail_type == "safiron_serve":
        from .safiron_guardrail import SafironServeGuardrail
        if not base_url:
            raise ValueError("base_url is required for safiron_serve guardrail type")
        return SafironServeGuardrail(model_path, base_url=base_url, max_new_tokens=max_new_tokens)

    elif guardrail_type == "safeguard_serve":
        from .safeguard_guardrail import SafeguardServeGuardrail
        if not base_url:
            raise ValueError("base_url is required for safeguard_serve guardrail type")
        return SafeguardServeGuardrail(model_path, base_url=base_url, max_new_tokens=max_new_tokens)

    elif guardrail_type == "qwen3guard_serve":
        from .qwen3guard_guardrail import Qwen3GuardServeGuardrail
        if not base_url:
            raise ValueError("base_url is required for qwen3guard_serve guardrail type")
        return Qwen3GuardServeGuardrail(model_path, base_url=base_url, max_new_tokens=max_new_tokens)

    elif guardrail_type == "shieldlm_serve":
        from .shieldlm_guardrail import ShieldLMServeGuardrail
        if not base_url:
            raise ValueError("base_url is required for shieldlm_serve guardrail type")
        return ShieldLMServeGuardrail(model_path, base_url=base_url, max_new_tokens=max_new_tokens)

    elif guardrail_type == "api":
        return APIGuardrail(model_path, api_key, max_new_tokens)

    else:
        raise ValueError(
            f"Unknown guardrail_type: {guardrail_type}. "
            f"Supported types: 'hf', 'vllm', 'vllm_serve', 'safiron', 'safiron_serve', 'safeguard_serve', 'qwen3guard_serve', 'shieldlm_serve', 'api'"
        )


def get_guardrail_type_from_path(model_path: str) -> str:
    """Infer guardrail type from model path.

    Args:
        model_path: Path to the guardrail model

    Returns:
        Inferred guardrail type
    """
    model_path_lower = model_path.lower()

    if "safiron" in model_path_lower:
        return "safiron"
    elif "gpt" in model_path_lower or "claude" in model_path_lower:
        return "api"
    else:
        return "hf"


def create_guardrail_auto(
    model_path: str,
    max_new_tokens: int = 2048,
    api_key: Optional[str] = None
) -> GuardrailInterface:
    """Automatically create a guardrail based on model path.

    Args:
        model_path: Path to the guardrail model
        max_new_tokens: Maximum tokens for guardrail response
        api_key: API key for API-based guardrails (optional)

    Returns:
        GuardrailInterface instance
    """
    guardrail_type = get_guardrail_type_from_path(model_path)
    return create_guardrail(
        model_path=model_path,
        guardrail_type=guardrail_type,
        max_new_tokens=max_new_tokens,
        api_key=api_key
    )