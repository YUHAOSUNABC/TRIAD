from .BaseAPI import BaseAPI
import time
import json
from typing import Dict, List


class GeminiAPI(BaseAPI):
    """Google Gemini via the official OpenAI-compatible Chat Completions endpoint.

    Targets gemini-2.5-pro through https://generativelanguage.googleapis.com/v1beta/openai/.
    The same OpenAI Python client used elsewhere works directly here — Google's
    endpoint speaks the standard OpenAI dialect including native function calling.

    Quirks vs other OpenAI-compatible providers:
      * Uses `max_tokens` (NOT `max_completion_tokens` like GPT5API) — the
        Google OpenAI shim accepts `max_tokens` as the spec name.
      * `temperature=0.0` is supported and forwarded for reproducibility.
      * Thinking is enabled by default on Gemini 2.5 Pro and CANNOT be disabled.
        Internal reasoning tokens consume part of the response budget but never
        appear in `message.content` — observed empirically: a single-word reply
        like "pong" used ~200 thinking tokens. `args.max_new_tokens` should be
        comfortably above the typical thinking budget (2048 is fine).
      * The Google OpenAI shim returns 400 INVALID_ARGUMENT on these params if
        sent: `top_p`, `presence_penalty`, `frequency_penalty`. We never send
        them. `reasoning_effort` works but maps to discrete thinking levels —
        we leave it unset to use Gemini's "dynamic" default.
      * No `reasoning_content` echo: thinking is not surfaced on this endpoint
        (unlike Kimi's thinking mode), so nothing to round-trip.

    The `client` MUST be an `openai.OpenAI(...)` instance with `base_url` set
    to `https://generativelanguage.googleapis.com/v1beta/openai/` and a valid
    AI Studio API key (env: GOOGLE_API_KEY or GEMINI_API_KEY — both work).
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
                    max_tokens=self.args.max_new_tokens,
                )
                if tools:
                    kwargs["tools"] = tools

                completion = self.client.chat.completions.create(**kwargs)
                if completion is None or completion.choices is None:
                    continue

                return self._parse_response(completion.choices[0].message)

            except Exception as e:
                err = str(e)
                err_lower = err.lower()
                is_transient = (
                    "503" in err
                    or "UNAVAILABLE" in err
                    or "429" in err
                    or "RESOURCE_EXHAUSTED" in err
                    or "overloaded" in err_lower
                    or "high demand" in err_lower
                    or "try again later" in err_lower
                )
                backoff = min(60, 5 * (2 ** attempt)) if is_transient else 1
                print(
                    f"[Error] Gemini API call failed (attempt {attempt + 1}/"
                    f"{self.max_retries}): {e} (retry in {backoff}s)"
                )
                time.sleep(backoff)
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
