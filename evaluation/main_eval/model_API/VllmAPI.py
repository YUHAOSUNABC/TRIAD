from typing import List, Dict, Optional


class VllmAPI:
    """
    Unified vLLM API wrapper for both agent and guardrail models.

    Supports:
    - Text generation with/without tools
    - Configurable GPU allocation
    - Response parsing
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_new_tokens: int = 2048,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
    ):
        """
        Initialize vLLM model.

        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
            max_model_len: Maximum sequence length (affects KV cache size)
            trust_remote_code: Whether to trust remote code
        """
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )

        print(f"[VllmAPI] Loaded {self.model_name} with tp={tensor_parallel_size}, mem={gpu_memory_utilization}, max_len={max_model_len}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate response from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tools in OpenAI format
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        from vllm import SamplingParams

        local_tools = self._convert_tools(tools) if tools else None

        # Disable Qwen3's thinking mode to reduce output verbosity.
        chat_template_kwargs = {
            "tools": local_tools,
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if "qwen3" in self.model_name.lower():
            chat_template_kwargs["enable_thinking"] = False

        prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text

        return response_text

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to format expected by chat templates."""
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
