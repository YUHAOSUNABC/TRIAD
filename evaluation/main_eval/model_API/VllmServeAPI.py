from .OpenaiAPI import OpenaiAPI
import time
import json
from typing import Dict


class VllmServeAPI(OpenaiAPI):
    """vLLM Serve API - wraps OpenAI-compatible endpoint with model-specific settings.

    Two modes:
      * Text-based (`fc_mode=False`, default): inherits OpenaiAPI's regex
        ```json``` fence parser. Used by Qwen3-32B and other models we run
        without vLLM's auto-tool-choice.
      * Native FC (`fc_mode=True`): passes `tools=` to chat completions and
        parses `message.tool_calls` directly. Required for gpt-oss-120b,
        whose harmony chat template trains it to emit OpenAI-format
        `tool_calls` (via vLLM's `--tool-call-parser=openai`) rather than
        markdown JSON fences. Without this, gpt-oss frequently emits bare
        JSON or pure prose, dropping ~30% of tasks at parser miss.
    """

    def __init__(self, max_retries, client, model_name, args, extra_body=None, fc_mode=False):
        super().__init__(max_retries, client, model_name, args)
        self.extra_body = extra_body or {}
        self.fc_mode = fc_mode

    def _generate_response(self, messages, tools):
        """Generate response via vLLM serve OpenAI-compatible API."""
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.args.max_new_tokens,
                )
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                if self.fc_mode and tools:
                    kwargs["tools"] = tools

                completion = self.client.chat.completions.create(**kwargs)

                if completion is None or completion.choices is None:
                    continue

                if self.fc_mode:
                    return self._parse_response_fc(completion.choices[0].message)
                content = completion.choices[0].message.content or ""
                return self._parse_response(content)

            except Exception as e:
                print(f"[Error] vLLM Serve API call failed (attempt {attempt + 1}): {e}")
                time.sleep(1)

        return None

    def _parse_response_fc(self, message) -> Dict:
        """Parse OpenAI-format response with native `tool_calls`.

        Mirrors GPT5API._parse_response: prefer `message.tool_calls[0]`, fall
        back to a content-only completion. We do not echo `reasoning_content`
        — vLLM's openai parser surfaces gpt-oss harmony commentary via
        content/tool_calls, not via a separate reasoning field, so there's
        nothing to round-trip.
        """
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
