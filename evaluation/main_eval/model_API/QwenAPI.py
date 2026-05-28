from .BaseAPI import BaseAPI
from typing import List, Dict, Optional
import json
import re


class QwenAPI(BaseAPI):
    def __init__(self, model=None, tokenizer=None, args=None, llm=None):
        """
        Initialize QwenAPI with either HuggingFace model or vLLM.

        Args:
            model: HuggingFace model (for local mode)
            tokenizer: Tokenizer (required for both modes)
            args: Arguments object
            llm: vLLM LLM object (for vLLM mode)
        """
        super().__init__()
        self.model = model
        self.llm = llm
        self.tokenizer = tokenizer
        self.args = args
        self.use_vllm = llm is not None

    def _generate_response(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Generate response using either vLLM or HuggingFace model.
        """
        if self.use_vllm:
            return self._generate_response_vllm(messages, tools)
        else:
            return self._generate_response_hf(messages, tools)

    def _generate_response_vllm(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Generate response using vLLM with text-based tool calling (ReAct format).
        Tools are described in system prompt, not passed to chat template.
        """
        try:
            from vllm import SamplingParams

            # Tools are described in the system prompt (text-based ReAct), not passed to chat template.
            chat_template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if hasattr(self.args, 'llm_path') and "qwen3" in self.args.llm_path.lower():
                chat_template_kwargs["enable_thinking"] = False

            prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)

            sampling_params = SamplingParams(
                max_tokens=self.args.max_new_tokens,
                temperature=0.0,
            )

            outputs = self.llm.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text

            return self._parse_response(response_text)

        except Exception as e:
            print(f"[Error] vLLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_response_hf(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """
        Generate response using local HuggingFace model with text-based tool calling.
        Tools are described in system prompt, not passed to chat template.
        """
        try:
            chat_template_kwargs = {
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if hasattr(self.args, 'llm_path') and "qwen3" in self.args.llm_path.lower():
                chat_template_kwargs["enable_thinking"] = False

            inputs = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

            new_tokens = outputs[0][len(inputs["input_ids"][0]):]
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return self._parse_response(response_text)

        except Exception as e:
            print(f"[Error] Qwen model generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to format expected by Qwen chat templates."""
        local_tools = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                func = tool["function"].copy()
                if func.get("parameters") is None:
                    func["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                local_tools.append({
                    "type": "function",
                    "function": func
                })
        return local_tools

    def _parse_response(self, text: str) -> Dict:
        """
        Parse Qwen model output with text-based ReAct format.

        Expected format:
        **Reasoning**: [thought text]
        ```json
        {"name": "tool_name", "arguments": {...}}
        ```

        Also supports legacy <tool_call> format for backward compatibility.
        """
        json_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)

        if json_block_match:
            json_start = text.find('```json')
            thought = text[:json_start].strip()

            json_content = json_block_match.group(1).strip()
            json_obj, _, _ = self._extract_json_object(json_content)

            if json_obj and "name" in json_obj:
                tool_name = json_obj.get("name", "")
                arguments = json_obj.get("arguments", {})

                return {
                    'type': 'tool',
                    'tool_name': tool_name,
                    'arguments': arguments,
                    'thought': thought,
                }

        # Legacy <tool_call> format for backward compatibility.
        tool_call_start = text.find('<tool_call>')
        tool_call_end = text.find('</tool_call>')

        if tool_call_start != -1 and tool_call_end != -1:
            thought = text[:tool_call_start].strip()
            json_content = text[tool_call_start + len('<tool_call>'):tool_call_end].strip()
            json_obj, _, _ = self._extract_json_object(json_content)

            if json_obj and "name" in json_obj:
                return {
                    'type': 'tool',
                    'tool_name': json_obj.get("name", ""),
                    'arguments': json_obj.get("arguments", {}),
                    'thought': thought,
                }

        json_obj, start_idx, end_idx = self._extract_json_object(text)
        if json_obj and "name" in json_obj:
            thought = text[:start_idx].strip() if start_idx > 0 else ""
            return {
                'type': 'tool',
                'tool_name': json_obj.get("name", ""),
                'arguments': json_obj.get("arguments", {}),
                'thought': thought,
            }

        return {'type': 'content', 'content': text, 'thought': text}

    def _extract_json_object(self, text: str):
        """
        Extract JSON object from text, handling nested braces.

        Returns:
            tuple: (json_obj, start_index, end_index) or (None, -1, -1) if not found
        """
        start = text.find('{')
        if start == -1:
            return None, -1, -1

        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start:i+1]
                    try:
                        json_obj = json.loads(json_str)
                        return json_obj, start, i+1
                    except json.JSONDecodeError:
                        result = self._extract_json_object(text[i+1:])
                        if result[0] is not None:
                            return result[0], result[1] + i + 1, result[2] + i + 1
                        return None, -1, -1

        return None, -1, -1