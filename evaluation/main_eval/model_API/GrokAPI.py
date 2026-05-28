from .BaseAPI import BaseAPI
import time
import json
from typing import Dict, List


class GrokAPI(BaseAPI):
    """xAI Grok with native function calling.

    Targets the non-reasoning Grok 4 Fast variants (e.g. grok-4-1-fast-non-reasoning).
    The xAI endpoint is OpenAI-compatible at https://api.x.ai/v1, so the same
    `OpenAI(base_url=..., api_key=...)` client used elsewhere works directly.

    Differs from OpenaiAPI (gpt-4o text-based path) and GPT5API (gpt-5.x reasoning):
      * Native FC: `tools=` is forwarded so the model emits `message.tool_calls`
        instead of a markdown JSON fence in content.
      * `max_completion_tokens` (the spec name; xAI deprecated `max_tokens`).
      * `temperature=0.0` is supported on non-reasoning Grok variants and
        forwarded for reproducibility.
      * `reasoning_effort` / `seed` / `presence_penalty` / `frequency_penalty`
        / `stop` are NOT supported by Grok and would 400; we never send them.
      * No `reasoning_content` echo (the non-reasoning variant doesn't expose
        any internal reasoning, unlike Kimi which requires round-tripping it).

    If a future reasoning variant is added (e.g. grok-4-1-fast-reasoning), this
    class will likely need a separate GrokReasoningAPI subclass that follows
    the GPT5API pattern (omit temperature, conditionally forward reasoning_effort).
    """

    def __init__(self, max_retries, client, model_name, args):
        super().__init__()
        self.max_retries = max_retries
        self.client = client
        self.model_name = model_name
        self.args = args

    def _generate_response(self, messages: List[Dict], tools: List[Dict]):
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_completion_tokens=self.args.max_new_tokens,
                )
                if tools:
                    kwargs["tools"] = tools

                completion = self.client.chat.completions.create(**kwargs)
                if completion is None or completion.choices is None:
                    continue

                return self._parse_response(completion.choices[0].message)

            except Exception as e:
                print(f"[Error] Grok API call failed (attempt {attempt + 1}): {e}")
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
