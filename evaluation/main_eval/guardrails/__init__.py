"""
Guardrails Package - Abstract interface and implementations for guardrail models.

Available classes:
- GuardrailInterface: Abstract base class for all guardrails
- HFGuardrailBase: Base class for HuggingFace-based guardrails
- HFGuardrail: Our fine-tuned guardrail implementation
- SafironGuardrail: Safiron model implementation
- APIGuardrail: Placeholder for API-based guardrails (NotImplemented)

Factory functions:
- create_guardrail: Create guardrail by type
- create_guardrail_auto: Auto-detect guardrail type from path

Usage:
    from guardrails import create_guardrail, HFGuardrail, SafironGuardrail

    # Using factory
    guardrail = create_guardrail("/path/to/model", guardrail_type="hf")

    # Direct instantiation
    guardrail = HFGuardrail("/path/to/model")
    guardrail = SafironGuardrail("Safiron/Safiron")

    # Check agent action
    result = guardrail.check(messages, agent_output, tools)
"""

from .guardrail_interface import (
    GuardrailInterface,
    HFGuardrailBase,
    APIGuardrail,
)
from .hf_guardrail import HFGuardrail
from .safiron_guardrail import SafironGuardrail, SafironServeGuardrail
from .guardrail_factory import (
    create_guardrail,
    create_guardrail_auto,
    get_guardrail_type_from_path,
)

__all__ = [
    # Base classes
    "GuardrailInterface",
    "HFGuardrailBase",
    "APIGuardrail",
    # Implementations
    "HFGuardrail",
    "SafironGuardrail",
    "SafironServeGuardrail",
    # Factory functions
    "create_guardrail",
    "create_guardrail_auto",
    "get_guardrail_type_from_path",
]