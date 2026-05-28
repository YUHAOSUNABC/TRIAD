"""
Message Adapter for converting different LLM message formats to guardrail format.

Guardrail is trained on OpenAI function calling format, so all agent message
histories need to be converted to this format before passing to guardrail.

Supported formats:
- openai: Text-based format (```json in content)
- claude: Anthropic format (tool_use/tool_result blocks)
- vllm: OpenAI function calling format (no conversion needed)
"""

import json
from typing import List, Dict


class MessageAdapter:
    """Adapter to convert different message formats to guardrail format (OpenAI FC)."""

    @staticmethod
    def to_guardrail_format(messages: List[Dict], llm_type: str) -> List[Dict]:
        """Convert messages from any format to guardrail's training format (OpenAI FC).

        Args:
            messages: List of messages in the agent's native format
            llm_type: Type of LLM ("openai", "claude", "vllm")

        Returns:
            Messages converted to OpenAI function calling format
        """
        if llm_type in ("openai", "vllm_serve"):
            return MessageAdapter._convert_text_to_openai_fc(messages)
        elif llm_type == "claude":
            return MessageAdapter._convert_anthropic_to_openai_fc(messages)
        else:
            # vllm / kimi / gpt5 / vllm_serve_fc / grok / gemini already use OpenAI FC.
            return messages

    @staticmethod
    def _convert_text_to_openai_fc(messages: List[Dict]) -> List[Dict]:
        """Convert GPT text-based format to OpenAI function calling format.

        Text format:
        - assistant: "思考内容\n\n```json\n{\"name\": \"tool\", \"arguments\": {...}}\n```"
        - user: "Tool response:\n..."

        OpenAI FC format:
        - assistant: {"content": "思考", "tool_calls": [...]}
        - tool: {"tool_call_id": "...", "content": "..."}
        """
        converted = []
        tool_call_counter = 0

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system" or role == "user":
                if role == "user" and isinstance(content, str) and content.startswith("Tool response:"):
                    tool_result = content.replace("Tool response:\n", "", 1)
                    converted.append({
                        "role": "tool",
                        "tool_call_id": f"call_{tool_call_counter - 1}",
                        "content": tool_result
                    })
                else:
                    converted.append(msg)

            elif role == "assistant":
                if isinstance(content, str) and "```json" in content:
                    thought, tool_call = MessageAdapter._parse_text_tool_call(content)

                    if tool_call:
                        tool_call_id = f"call_{tool_call_counter}"
                        tool_call_counter += 1

                        converted.append({
                            "role": "assistant",
                            "content": thought,
                            "tool_calls": [{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.get("name", ""),
                                    "arguments": json.dumps(tool_call.get("arguments", {}))
                                }
                            }]
                        })
                    else:
                        converted.append(msg)
                else:
                    converted.append(msg)
            else:
                converted.append(msg)

            i += 1

        return converted

    @staticmethod
    def _parse_text_tool_call(content: str) -> tuple:
        """Parse text-based tool call format.

        Args:
            content: Assistant message content with ```json block

        Returns:
            (thought, tool_call_dict) or (thought, None) if parsing fails
        """
        parts = content.split("```json")
        thought = parts[0].strip()

        if len(parts) < 2:
            return thought, None

        json_part = parts[1]
        if "```" in json_part:
            json_str = json_part.split("```")[0].strip()
        else:
            json_str = json_part.strip()

        try:
            tool_call = json.loads(json_str)
            return thought, tool_call
        except json.JSONDecodeError:
            return thought, None

    @staticmethod
    def _convert_anthropic_to_openai_fc(messages: List[Dict]) -> List[Dict]:
        """Convert Anthropic format to OpenAI function calling format.

        Anthropic format:
        - assistant: {"content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]}
        - user: {"content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}

        OpenAI FC format:
        - assistant: {"content": "思考", "tool_calls": [...]}
        - tool: {"tool_call_id": "...", "content": "..."}
        """
        converted = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                converted.append(msg)

            elif role == "assistant":
                if isinstance(content, list):
                    thought = ""
                    tool_calls = []

                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                thought = block.get("text", "")
                            elif block.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": block.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input", {}))
                                    }
                                })

                    if tool_calls:
                        converted.append({
                            "role": "assistant",
                            "content": thought,
                            "tool_calls": tool_calls
                        })
                    else:
                        converted.append({
                            "role": "assistant",
                            "content": thought
                        })
                else:
                    converted.append(msg)

            elif role == "user":
                if isinstance(content, list):
                    tool_results = []
                    text_content = ""

                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                tool_results.append({
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": block.get("content", "")
                                })
                            elif block.get("type") == "text":
                                text_content = block.get("text", "")

                    converted.extend(tool_results)

                    if text_content:
                        converted.append({
                            "role": "user",
                            "content": text_content
                        })
                else:
                    converted.append(msg)

            elif role == "tool":
                converted.append(msg)

            else:
                converted.append(msg)

        return converted

    @staticmethod
    def convert_completion_to_openai_fc(completion: Dict) -> Dict:
        """Convert a single completion (current action) to OpenAI FC format.

        This is used for converting the current action plan before passing to guardrail.
        The completion format from model API is unified across all LLM types.

        Args:
            completion: The completion dict from model API

        Returns:
            Completion in OpenAI FC format for guardrail
        """
        if completion.get("type") == "tool":
            return {
                "role": "assistant",
                "content": completion.get("thought", ""),
                "tool_calls": [{
                    "id": completion.get("tool_call_id", "call_0"),
                    "type": "function",
                    "function": {
                        "name": completion.get("tool_name", ""),
                        "arguments": json.dumps(completion.get("arguments", {}))
                    }
                }]
            }
        else:
            return {
                "role": "assistant",
                "content": completion.get("content", "")
            }
