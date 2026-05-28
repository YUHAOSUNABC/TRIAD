"""
Guardrail Interface - Abstract base class for guardrail models.

This interface defines the contract for guardrail implementations.
Subclasses must implement the check() method to evaluate agent actions.

Implementations:
- HFGuardrail: Our fine-tuned guardrail model (guardrail.py)
- SafironGuardrail: Safiron model from Agentic-Guardian
- APIGuardrail: Placeholder for API-based models (NotImplemented)
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch


class GuardrailInterface(ABC):
    """Abstract base class for guardrail models."""

    def __init__(self, model_path: str, max_new_tokens: int = 2048):
        """Initialize guardrail with a model.

        Args:
            model_path: Path to the model (local or HuggingFace hub)
            max_new_tokens: Maximum tokens for guardrail response
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model_name = model_path.split('/')[-1]

    @abstractmethod
    def check(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str,
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Check agent's action with guardrail model.

        Args:
            messages: Conversation history
            agent_output: Agent's current output (Thought + Action + Action Input)
            tools: Optional list of available tools for context

        Returns:
            Dict with keys:
                - decision: "proceed" | "refuse" | "update" | "invalid"
                - reason: Explanation for the decision
                - feedback: Feedback text for the agent
        """
        pass

    @abstractmethod
    def _build_input(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Build input string for the guardrail model.

        Args:
            messages: Conversation history
            agent_output: Agent's current output
            tools: Optional list of available tools

        Returns:
            Formatted input string for the model
        """
        pass

    @abstractmethod
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse guardrail response to extract decision.

        Args:
            response_text: Raw response from the model

        Returns:
            Dict with decision, reason, and feedback
        """
        pass

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


class HFGuardrailBase(GuardrailInterface):
    """Base class for HuggingFace-based guardrail models."""

    def __init__(self, model_path: str, max_new_tokens: int = 2048, gpu_ids: list = None):
        super().__init__(model_path, max_new_tokens)
        self.gpu_ids = gpu_ids
        self._load_model()

    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.gpu_ids and len(self.gpu_ids) > 0:
            device_map = {"": self.gpu_ids[0]}
            print(f"[Guardrail] Loading model on GPU {self.gpu_ids[0]}")
        else:
            device_map = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
        print(f"[Guardrail] Loaded model: {self.model_name}")

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Generated response text
        """
        # Default to thinking OFF to match SFT config; opt in with GUARDRAIL_THINKING=1.
        template_kwargs = {}
        if "enable_thinking" in (self.tokenizer.chat_template or ""):
            template_kwargs["enable_thinking"] = os.environ.get("GUARDRAIL_THINKING", "0") == "1"

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        new_tokens = outputs[0][len(inputs["input_ids"][0]):]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response_text


class APIGuardrail(GuardrailInterface):
    """Placeholder for API-based guardrail models (e.g., GPT-4, Claude)."""

    def __init__(self, model_path: str, api_key: str = None, max_new_tokens: int = 2048):
        super().__init__(model_path, max_new_tokens)
        self.api_key = api_key
        raise NotImplementedError(
            "API-based guardrail is not implemented yet. "
            "Please use HuggingFace-based guardrail models."
        )

    def check(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str,
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError("API-based guardrail is not implemented yet.")

    def _build_input(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str,
        tools: Optional[List[Dict]] = None
    ) -> str:
        raise NotImplementedError("API-based guardrail is not implemented yet.")

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        raise NotImplementedError("API-based guardrail is not implemented yet.")