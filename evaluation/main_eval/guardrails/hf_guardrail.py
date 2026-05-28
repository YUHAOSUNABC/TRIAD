"""
HFGuardrail - Our fine-tuned guardrail model/vanilla model implementation.

This is the guardrail model we trained for agent action verification.
It inherits from HFGuardrailBase and implements the check logic.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from .guardrail_interface import HFGuardrailBase

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from guardrail_prompts import GUARDRAIL_SYSTEM_PROMPT_COT, GUARDRAIL_SYSTEM_PROMPT


class HFGuardrail(HFGuardrailBase):
    """Our fine-tuned guardrail model for checking agent actions."""

    def __init__(self, model_path: str, max_new_tokens: int = 2048, gpu_ids: list = None, use_cot: bool = True):
        """Initialize HFGuardrail.

        Args:
            model_path: Path to the fine-tuned guardrail model
            max_new_tokens: Maximum tokens for guardrail response
            gpu_ids: List of GPU IDs to use for the model (optional)
            use_cot: Whether to use COT (chain-of-thought) system prompt (default: True)
        """
        super().__init__(model_path, max_new_tokens, gpu_ids=gpu_ids)
        self.system_prompt = GUARDRAIL_SYSTEM_PROMPT_COT if use_cot else GUARDRAIL_SYSTEM_PROMPT

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
                       Contains: type, thought, tool_name, arguments, etc.

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
            print(f"[Guardrail] Error during check: {e}")
            return {
                "decision": "proceed",
                "reason": f"Guardrail error: {str(e)}",
                "feedback": ""
            }

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