"""
SafeguardGuardrail - gpt-oss-safeguard model implementation.

Reuses Safiron's input format and system prompt, but places the policy
in the `developer` role (OpenAI Harmony format used by gpt-oss models).
"""

from typing import Dict, List, Any, Optional

from .guardrail_interface import GuardrailInterface
from .safiron_guardrail import SAFIRON_SYSTEM_PROMPT, SafironGuardrail


class SafeguardServeGuardrail(GuardrailInterface):
    """gpt-oss-safeguard guardrail using vLLM serve API.

    Uses Harmony format with:
    - `developer` role: policy (reusing Safiron's system prompt)
    - `user` role: Safiron-style JSON input

    Start the server with:
        vllm serve openai/gpt-oss-safeguard-20b --port 8400 --max-model-len 16384
    """

    def __init__(
        self,
        model_path: str,
        base_url: str = "http://localhost:8400/v1",
        max_new_tokens: int = 512,
        reasoning_effort: str = "medium",
    ):
        super().__init__(model_path, max_new_tokens)
        self.model_name = model_path
        self.base_url = base_url
        self.policy_prompt = SAFIRON_SYSTEM_PROMPT
        self.reasoning_effort = reasoning_effort

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="unused")
        print(f"[Safeguard-Serve] Connected to {base_url}, model: {self.model_name}")

    def check(
        self,
        messages: List[Dict[str, Any]],
        agent_output: str = None,
        tools: Optional[List[Dict]] = None,
        completion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        safeguard_input = self._build_input(messages, agent_output, tools, completion)

        safeguard_messages = [
            {"role": "developer", "content": self.policy_prompt},
            {"role": "user", "content": safeguard_input}
        ]

        try:
            response_text = self._generate(safeguard_messages)
            return self._parse_response(response_text)
        except Exception as e:
            print(f"[Safeguard-Serve] Error during check: {e}")
            return {
                "decision": "proceed",
                "reason": f"Safeguard error: {str(e)}",
                "feedback": ""
            }

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            extra_body={"chat_template_kwargs": {"reasoning_effort": self.reasoning_effort}},
        )
        msg = response.choices[0].message
        # Empty content means the reasoning channel exhausted the token budget before the final channel was produced.
        if msg.content is None:
            finish = response.choices[0].finish_reason
            print(f"[Safeguard-Serve] Empty final channel (finish_reason={finish}). "
                  f"Consider raising --guardrail_max_new_tokens.")
            return ""
        return msg.content

    _build_input = SafironGuardrail._build_input
    _parse_response = SafironGuardrail._parse_response
