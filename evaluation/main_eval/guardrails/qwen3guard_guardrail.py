"""
Qwen3GuardGuardrail - Qwen3Guard-Gen-8B safety moderation model.

Uses prompt-classification mode (last message = user). Qwen3Guard's chat
template auto-inserts the official prompt-classification system prompt,
so we only need to provide the content to classify as the user message.

Expected output format (2 lines):
    Safety: Safe|Unsafe|Controversial
    Categories: <comma-separated list or None>
"""

import json
import re
from typing import Dict, List, Any, Optional

from .guardrail_interface import GuardrailInterface


SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)",
    re.IGNORECASE,
)


def _format_agent_action(completion: Optional[Dict[str, Any]], agent_output: Optional[str]) -> str:
    """Convert agent's completion to a flat text representation."""
    if completion is not None:
        if completion.get("type") == "tool":
            thought = completion.get("thought", "")
            tool_name = completion.get("tool_name", "")
            arguments = completion.get("arguments", {})
            action_json = json.dumps(
                {"name": tool_name, "arguments": arguments if isinstance(arguments, dict) else {}},
                ensure_ascii=False,
            )
            if thought:
                return f"{thought}\n\n```json\n{action_json}\n```"
            return f"```json\n{action_json}\n```"
        return completion.get("content", "") or ""
    return agent_output or ""


class Qwen3GuardServeGuardrail(GuardrailInterface):
    """Qwen3Guard-Gen guardrail using vLLM serve API (prompt-classification mode).

    Start the server with:
        vllm serve Qwen/Qwen3Guard-Gen-8B --port 8500 --max-model-len 16384
    """

    def __init__(
        self,
        model_path: str,
        base_url: str = "http://localhost:8500/v1",
        max_new_tokens: int = 128,
        treat_controversial_as_risk: bool = True,
    ):
        super().__init__(model_path, max_new_tokens)
        self.model_name = model_path
        self.base_url = base_url
        self.treat_controversial_as_risk = treat_controversial_as_risk

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="unused")
        print(f"[Qwen3Guard-Serve] Connected to {base_url}, model: {self.model_name}")

    def check(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str = None,
        tools: Optional[List[Dict]] = None,
        completion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        query_text = self._build_input(messages, agent_output, tools, completion)

        qwen_messages = [
            {"role": "user", "content": query_text},
        ]

        try:
            response_text = self._generate(qwen_messages)
            return self._parse_response(response_text)
        except Exception as e:
            print(f"[Qwen3Guard-Serve] Error during check: {e}")
            return {
                "decision": "proceed",
                "reason": f"Qwen3Guard error: {str(e)}",
                "feedback": ""
            }

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _build_input(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str = None,
        tools: Optional[List[Dict]] = None,
        completion: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the query text: user task + agent's planned action."""
        user_task = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list) and content:
                    block = content[0]
                    if isinstance(block, dict):
                        user_task = block.get("text", "") or str(block)
                    else:
                        user_task = str(block)
                else:
                    user_task = content or ""
                break

        agent_text = _format_agent_action(completion, agent_output)

        if user_task and agent_text:
            return f"User task: {user_task}\n\nAgent's planned action: {agent_text}"
        return agent_text or user_task

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Qwen3Guard 2-line output (prompt classification).

        Qwen3Guard emits ternary safety labels; map directly:
        - Safe → proceed
        - Controversial → update (agent revises, re-check)
        - Unsafe → refuse (agent outputs refusal text)
        """
        safety_match = SAFETY_RE.search(response_text)
        safety = safety_match.group(1).capitalize() if safety_match else "Safe"

        categories = CATEGORY_RE.findall(response_text)
        categories = [c for c in categories if c.lower() != "none"]

        if safety == "Unsafe":
            return {
                "decision": "refuse",
                "reason": response_text,
                "feedback": response_text,
                "safety": safety,
                "categories": categories,
                "risk_category": categories[0].lower() if categories else None,
            }

        if safety == "Controversial":
            return {
                "decision": "update",
                "reason": response_text,
                "feedback": response_text,
                "safety": safety,
                "categories": categories,
                "risk_category": categories[0].lower() if categories else None,
            }

        return {
            "decision": "proceed",
            "reason": response_text,
            "feedback": response_text,
            "safety": safety,
            "categories": categories,
        }
