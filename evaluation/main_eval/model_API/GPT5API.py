from .BaseAPI import BaseAPI
import time
import json
from typing import Dict, List


class GPT5API(BaseAPI):
    """OpenAI GPT-5.x reasoning models with native function calling.

    Differs from OpenaiAPI (gpt-4o text-based path) in three ways required by
    the reasoning-model API contract:
      * `tools=` is passed so the model emits structured `message.tool_calls`
        instead of a ```json``` block in content.
      * `max_completion_tokens` replaces `max_tokens`.
      * `temperature` is omitted (reasoning models reject anything other
        than the default 1).
      * `reasoning_effort` is forwarded.

    Allowed `reasoning_effort` values are model-specific. Empirically on
    gpt-5.4-2026-03-05 Chat Completions REJECTS function tools whenever
    `reasoning_effort` is set ("Function tools with reasoning_effort are not
    supported... Please use /v1/responses instead"). To stay on Chat
    Completions for gpt-5.4 we therefore skip the param entirely (model picks
    its own default reasoning depth). For earlier gpt-5.x where the
    combination is allowed, `reasoning_effort` is still forwarded if set.

    Pass `reasoning_effort=None` (the default) to omit the param. Otherwise
    valid values depend on the specific model: gpt-5.0/5.1 accept
    {minimal, low, medium, high}; gpt-5.4 accepts {none, low, medium, high,
    xhigh} but rejects the combination with tools.

    On Chat Completions the model's reasoning is internal and not exposed in
    `message.content`, so unlike KimiAPI we do NOT echo a `reasoning_content`
    field back on subsequent turns — sending one would be rejected by OpenAI.
    """

    def __init__(self, max_retries, client, model_name, args, reasoning_effort=None):
        super().__init__()
        self.max_retries = max_retries
        self.client = client
        self.model_name = model_name
        self.args = args
        self.reasoning_effort = reasoning_effort

    def _generate_response(self, messages: List[Dict], tools: List[Dict]):
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.args.max_new_tokens,
                    seed=42,
                )
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                if tools:
                    kwargs["tools"] = tools

                completion = self.client.chat.completions.create(**kwargs)
                if completion is None or completion.choices is None:
                    continue

                return self._parse_response(completion.choices[0].message)

            except Exception as e:
                print(f"[Error] GPT-5 API call failed (attempt {attempt + 1}): {e}")
                time.sleep(1)
        return None

    def _parse_response(self, message) -> Dict:
        thought = (message.content or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            tc = tool_calls[0]
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            return {
                "type": "tool",
                "tool_name": tc.function.name,
                "arguments": arguments,
                "thought": thought,
                "tool_call_id": tc.id,
            }

        return {"type": "content", "content": thought, "thought": thought}
