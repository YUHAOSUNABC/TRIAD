from .BaseAPI import BaseAPI
import time
import re
from typing import Dict
import json


class OpenaiAPI(BaseAPI):
    """OpenAI API with text-based tool calling (no native function calling)."""

    def __init__(self, max_retries, client, model_name, args):
        super().__init__()
        self.max_retries = max_retries
        self.client = client
        self.model_name = model_name
        self.args = args
        self._tool_call_counter = 0

    def _generate_response(self, messages, tools):
        """Generate response using text-based tool calling."""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.args.max_new_tokens,
                )

                if completion is None or completion.choices is None:
                    continue

                content = completion.choices[0].message.content or ""
                return self._parse_response(content)

            except Exception as e:
                print(f"[Error] OpenAI API call failed (attempt {attempt + 1}): {e}")
                time.sleep(1)

        return None

    def _parse_response(self, content: str) -> Dict:
        """Parse text response, extracting JSON tool calls from content."""
        thought = content.strip()

        # Try to extract JSON tool call from ```json block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                tool_call = json.loads(json_match.group(1))
                tool_name = tool_call.get('name', '')
                arguments = tool_call.get('arguments', {})

                # Extract thought (text before JSON block)
                thought = content[:json_match.start()].strip()

                self._tool_call_counter += 1
                return {
                    'type': 'tool',
                    'tool_name': tool_name,
                    'arguments': arguments,
                    'thought': thought,
                    'tool_call_id': f'call_{self._tool_call_counter}',
                }
            except json.JSONDecodeError:
                pass

        # No tool call found, return as content
        return {
            'type': 'content',
            'content': content,
            'thought': thought,
        }
