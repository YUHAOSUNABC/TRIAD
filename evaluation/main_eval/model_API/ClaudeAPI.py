from .BaseAPI import BaseAPI
import time
from typing import List, Dict


class ClaudeAPI(BaseAPI):
    """Claude API using native Anthropic SDK.

    Expects Anthropic-format messages and tools.
    """

    def __init__(self, max_retries, client, model_name, args):
        """
        Initialize ClaudeAPI.

        Args:
            max_retries: Max retry attempts
            client: Anthropic client instance
            model_name: Model name (e.g., 'claude-sonnet-4-20250514')
            args: Arguments object with max_new_tokens
        """
        super().__init__()
        self.max_retries = max_retries
        self.client = client
        self.model_name = model_name
        self.args = args

    def _generate_response(self, messages, tools, system_prompt=None):
        """Generate response using Anthropic API with native function calling.

        Args:
            messages: Anthropic-format messages list (may include system message at index 0)
            tools: Anthropic-format tools list
            system_prompt: Optional system prompt string (overrides system message in messages)
        """
        # Anthropic requires `system` as a top-level kwarg; extract it from messages.
        actual_messages = messages
        if not system_prompt and messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", "")
            actual_messages = messages[1:]

        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": actual_messages,
                    "temperature": 0.0,
                    "max_tokens": self.args.max_new_tokens,
                }

                if tools:
                    # Enable prompt caching on [system + tools] prefix via cache_control on the last tool.
                    tools_cached = list(tools)
                    tools_cached[-1] = {**tools_cached[-1], "cache_control": {"type": "ephemeral"}}
                    kwargs["tools"] = tools_cached

                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self.client.messages.create(**kwargs)
                return self._parse_response(response)

            except Exception as e:
                print(f"[Error] Claude API call failed (attempt {attempt + 1}): {e}")
                time.sleep(1)

        return None

    def _parse_response(self, response) -> Dict:
        """Parse Anthropic-format response.

        Response contains content blocks: TextBlock and/or ToolUseBlock.
        """
        thought = ""
        tool_use_block = None

        for block in response.content:
            if block.type == "text":
                thought = block.text
            elif block.type == "tool_use":
                tool_use_block = block

        if tool_use_block:
            return {
                'type': 'tool',
                'tool_name': tool_use_block.name,
                'arguments': tool_use_block.input,
                'thought': thought,
                'tool_call_id': tool_use_block.id,
            }

        return {
            'type': 'content',
            'content': thought,
            'thought': thought,
        }
