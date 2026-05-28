from .BaseAPI import BaseAPI
import time
import json
from typing import Dict, List


class KimiAPI(BaseAPI):
    """Moonshot Kimi API with native OpenAI-compatible function calling.

    Kimi (k2.5/k2.6) speaks OpenAI's chat-completions schema and emits
    `message.content` (free-text thought) plus structured `message.tool_calls`
    in a single response — same shape as Claude's text + tool_use blocks but
    in OpenAI format. This API class therefore behaves like ClaudeAPI in spirit
    while staying in OpenAI-style messaging.

    Constraints (Moonshot platform):
    - Thinking enabled (default): only temperature=1.0 accepted.
    - Thinking disabled (instant mode): only temperature=0.6 accepted.
    """

    def __init__(self, max_retries, client, model_name, args, extra_body=None):
        super().__init__()
        self.max_retries = max_retries
        self.client = client
        self.model_name = model_name
        self.args = args
        self.extra_body = extra_body or {}
        # Moonshot only accepts a single temperature per thinking mode (1.0 / 0.6).
        thinking_disabled = (
            isinstance(self.extra_body.get("thinking"), dict)
            and self.extra_body["thinking"].get("type") == "disabled"
        )
        self.temperature = 0.6 if thinking_disabled else 1.0

    def _generate_response(self, messages: List[Dict], tools: List[Dict]):
        """Native function calling via Moonshot OpenAI-compatible endpoint."""
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.args.max_new_tokens,
                )
                if tools:
                    kwargs["tools"] = tools
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body

                completion = self.client.chat.completions.create(**kwargs)
                if completion is None or completion.choices is None:
                    continue

                return self._parse_response(completion.choices[0].message)

            except Exception as e:
                err = str(e)
                err_lower = err.lower()
                is_transient = (
                    "429" in err
                    or "503" in err
                    or "rate limit" in err_lower
                    or "rate_limit" in err_lower
                    or "too many requests" in err_lower
                    or "overloaded" in err_lower
                    or "service unavailable" in err_lower
                    or "try again later" in err_lower
                )
                backoff = min(60, 5 * (2 ** attempt)) if is_transient else 1
                print(
                    f"[Error] Kimi API call failed (attempt {attempt + 1}/"
                    f"{self.max_retries}): {e} (retry in {backoff}s)"
                )
                time.sleep(backoff)
        return None

    def _parse_response(self, message) -> Dict:
        """Parse OpenAI-style chat message into unified completion dict.

        Mirrors ClaudeAPI._parse_response semantics: when tool_calls exist,
        emit a 'tool' completion carrying the model's prose as `thought`;
        otherwise emit a 'content' completion. Also threads `reasoning_content`
        through so multi-turn tool-call sequences can echo it back to the
        platform (Moonshot rejects multi-turn tool calls otherwise).
        """
        thought = (message.content or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []
        reasoning_content = getattr(message, "reasoning_content", None)

        if tool_calls:
            tc = tool_calls[0]
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            out = {
                "type": "tool",
                "tool_name": tc.function.name,
                "arguments": arguments,
                "thought": thought,
                "tool_call_id": tc.id,
            }
            if reasoning_content:
                out["reasoning_content"] = reasoning_content
            return out

        out = {"type": "content", "content": thought, "thought": thought}
        if reasoning_content:
            out["reasoning_content"] = reasoning_content
        return out
