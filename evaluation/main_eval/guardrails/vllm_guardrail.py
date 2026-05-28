"""
VLLMGuardrail - vLLM-based guardrail implementations.

Two variants:
- VLLMGuardrail: Offline vLLM inference (loads model locally)
- VLLMServeGuardrail: Calls a running vLLM OpenAI-compatible server
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from .guardrail_interface import GuardrailInterface

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from guardrail_prompts import GUARDRAIL_SYSTEM_PROMPT_COT, GUARDRAIL_SYSTEM_PROMPT


class VLLMGuardrail(GuardrailInterface):
    """vLLM-based guardrail for faster inference."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        gpu_ids: Optional[List[int]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        use_cot: bool = True
    ):
        """Initialize VLLMGuardrail.

        Args:
            model_path: Path to the guardrail model
            max_new_tokens: Maximum tokens for guardrail response
            gpu_ids: List of GPU IDs to use (optional)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model context length
            use_cot: Whether to use COT (chain-of-thought) system prompt (default: True)
        """
        super().__init__(model_path, max_new_tokens)
        self.gpu_ids = gpu_ids
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.system_prompt = GUARDRAIL_SYSTEM_PROMPT_COT if use_cot else GUARDRAIL_SYSTEM_PROMPT
        self._load_model()

    def _load_model(self):
        """Load the vLLM model on specified GPU."""
        import os
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pin vLLM to a specific GPU via CUDA_VISIBLE_DEVICES so it doesn't grab all visible devices.
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        if self.gpu_ids and len(self.gpu_ids) > 0:
            target_gpu = self.gpu_ids[0]

            if original_cuda_visible:
                visible_gpus = [int(g) for g in original_cuda_visible.split(",")]
                if target_gpu < len(visible_gpus):
                    actual_gpu = visible_gpus[target_gpu]
                else:
                    actual_gpu = target_gpu
            else:
                actual_gpu = target_gpu

            os.environ["CUDA_VISIBLE_DEVICES"] = str(actual_gpu)
            print(f"[Guardrail vLLM] Setting CUDA_VISIBLE_DEVICES={actual_gpu} (was: {original_cuda_visible})")

        print(f"[Guardrail vLLM] Loading model with tensor_parallel_size={self.tensor_parallel_size}, "
              f"gpu_memory_utilization={self.gpu_memory_utilization}, max_model_len={self.max_model_len}")

        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            disable_log_stats=True,
        )

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )

        if original_cuda_visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible

        print(f"[Guardrail vLLM] Loaded model: {self.model_name}")

    def check(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str = None,
        tools: Optional[List[Dict]] = None,
        completion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check agent's action with guardrail model.

        Args:
            messages: Conversation history
            agent_output: Agent's current output as text (legacy, for backward compatibility)
            tools: Optional list of available tools for context
            completion: Agent's completion in native format (preferred)

        Returns:
            Dict with keys: decision, reason, feedback
        """
        guardrail_input = self._build_input(messages, agent_output, tools, completion)

        guardrail_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": guardrail_input}
        ]

        try:
            response_text = self._generate(guardrail_messages)
            return self._parse_response(response_text)

        except Exception as e:
            print(f"[Guardrail vLLM] Error during check: {e}")
            return {
                "decision": "proceed",
                "reason": f"Guardrail error: {str(e)}",
                "feedback": ""
            }

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using vLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Generated response text
        """
        # Default to thinking OFF to match SFT config; opt in with GUARDRAIL_THINKING=1.
        template_kwargs = {}
        if "enable_thinking" in (self.tokenizer.chat_template or ""):
            template_kwargs["enable_thinking"] = os.environ.get("GUARDRAIL_THINKING", "0") == "1"

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

        outputs = self.llm.generate([prompt], self.sampling_params)
        response_text = outputs[0].outputs[0].text

        return response_text

    def _build_input(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str = None,
        tools: Optional[List[Dict]] = None,
        completion: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build input string for guardrail model.

        Supports both native function calling format (preferred) and legacy text format.
        """
        parts = []

        if tools:
            parts.append("**Available Tools**")
            for tool in tools:
                func = tool.get("function", tool)
                name = func.get("name", "")
                desc = func.get("description", "")
                if desc:
                    parts.append(f"- {name}: {desc}")
                else:
                    parts.append(f"- {name}")
            parts.append("")

        parts.append("**Memory**")

        for msg in messages:
            if msg["role"] == "system":
                continue
            parts.append(json.dumps(msg, ensure_ascii=False))

        parts.append("\n**Current Action Plan**")

        if completion is not None:
            if completion.get('type') == 'tool':
                current_action = {
                    "role": "assistant",
                    "content": completion.get('thought', ''),
                    "tool_calls": [{
                        "id": completion.get('tool_call_id', 'call_0'),
                        "type": "function",
                        "function": {
                            "name": completion.get('tool_name', ''),
                            "arguments": json.dumps(completion.get('arguments', {}), ensure_ascii=False)
                        }
                    }]
                }
            else:
                current_action = {
                    "role": "assistant",
                    "content": completion.get('content', '')
                }
            parts.append(json.dumps(current_action, ensure_ascii=False, indent=2))
        elif agent_output is not None:
            parts.append(agent_output)
        else:
            parts.append("No action provided")

        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse guardrail response to extract decision.

        Expected format: Natural language paragraph ending with:
        - **Decision**: Proceed Action → decision = "proceed"
        - **Decision**: Refuse Action → decision = "refuse"
        - **Decision**: Update Action → decision = "update"
        """
        response_lower = response_text.lower()

        decision_match = re.search(r'\*\*decision\*\*:\s*(\w+(?:\s+\w+)?)', response_lower)

        if decision_match:
            decision_text = decision_match.group(1).strip()

            if "proceed" in decision_text:
                return {
                    "decision": "proceed",
                    "reason": response_text,
                    "feedback": response_text
                }
            elif "refuse" in decision_text:
                return {
                    "decision": "refuse",
                    "reason": response_text,
                    "feedback": response_text
                }
            elif "update" in decision_text:
                return {
                    "decision": "update",
                    "reason": response_text,
                    "feedback": response_text
                }

        return {
            "decision": "invalid",
            "reason": "Guardrail output did not contain valid **Decision**: pattern",
            "feedback": response_text
        }


class VLLMServeGuardrail(GuardrailInterface):
    """Guardrail that calls a running vLLM OpenAI-compatible server.

    Use this when you want to run guardrail as a service for concurrent access.
    Start the server with:
        vllm serve <model_path> --port 8100 --max-model-len 16384
    """

    def __init__(
        self,
        model_path: str,
        base_url: str = "http://localhost:8100/v1",
        max_new_tokens: int = 2048,
        use_cot: bool = True,
    ):
        super().__init__(model_path, max_new_tokens)
        # vLLM serve registers the model by its full path.
        self.model_name = model_path
        self.base_url = base_url
        self.system_prompt = GUARDRAIL_SYSTEM_PROMPT_COT if use_cot else GUARDRAIL_SYSTEM_PROMPT

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="unused")
        print(f"[Guardrail vLLM-Serve] Connected to {base_url}, model: {self.model_name}, use_cot: {use_cot}")

    def check(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str = None,
        tools: Optional[List[Dict]] = None,
        completion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        guardrail_input = self._build_input(messages, agent_output, tools, completion)

        guardrail_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": guardrail_input}
        ]

        try:
            response_text = self._generate(guardrail_messages)
            return self._parse_response(response_text)
        except Exception as e:
            print(f"[Guardrail vLLM-Serve] Error during check: {e}")
            return {
                "decision": "proceed",
                "reason": f"Guardrail error: {str(e)}",
                "feedback": ""
            }

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": os.environ.get("GUARDRAIL_THINKING", "0") == "1"}},
        )
        return response.choices[0].message.content

    _build_input = VLLMGuardrail._build_input
    _parse_response = VLLMGuardrail._parse_response
