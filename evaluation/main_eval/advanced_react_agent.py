import os
from typing import Any, Dict, Optional
from prompt import get_function_calling_instruction, get_text_tool_calling_instruction
from asb.tools.simulated_tools import SimulatedTool, AttackerTool
import copy
import pandas as pd
from model_API import Llama3API, OpenaiAPI, QwenAPI, ClaudeAPI, GeminiAPI, VllmServeAPI, KimiAPI, GPT5API, GrokAPI


class AdvancedReactAgent:
    """ReAct Agent with native function calling support."""

    def __init__(
        self,
        agent_name: str,
        user_task: str,
        llm_agent_info: Dict[str, Any],
        args: Any,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_name = agent_name
        self.user_task = user_task
        self.args = args
        self.agent_config = agent_config

        self.llm_agent_info = llm_agent_info
        self.llm_type = llm_agent_info["llm_type"]
        self.model_name = llm_agent_info["model_name"]

        self.tool_list = dict()
        self.tools = []
        self.tool_names = self.agent_config["tools"]
        self.load_tools_from_file(self.tool_names, self.args.tools_info_path)

        if self.llm_agent_info["llm_type"] == "local":
            self.model = self.llm_agent_info["model"]
            self.tokenizer = self.llm_agent_info["tokenizer"]
        elif self.llm_agent_info["llm_type"] == "vllm":
            self.llm = self.llm_agent_info["llm"]
            self.tokenizer = self.llm_agent_info["tokenizer"]
        elif self.llm_agent_info["llm_type"] == "openai":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
        elif self.llm_agent_info["llm_type"] == "claude":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
        elif self.llm_agent_info["llm_type"] == "kimi":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
        elif self.llm_agent_info["llm_type"] == "gpt5":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
            self.reasoning_effort = self.llm_agent_info.get("reasoning_effort", "minimal")
        elif self.llm_agent_info["llm_type"] == "grok":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
        elif self.llm_agent_info["llm_type"] == "vllm_serve":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
        elif self.llm_agent_info["llm_type"] == "vllm_serve_fc":
            self.client = self.llm_agent_info["client"]
            self.max_retries = 3
        elif self.llm_agent_info["llm_type"] == "gemini":
            self.client = self.llm_agent_info["client"]
            # Gemini 2.5 Pro emits 503 under capacity pressure — extra retries for transient spikes.
            self.max_retries = 6

        if self.llm_agent_info["llm_type"] == "vllm_serve":
            # Text-mode vLLM serve: Qwen3 needs enable_thinking=False via extra_body.
            extra_body = {}
            model_path = self.llm_agent_info.get("model_path", self.model_name)
            if "qwen3" in model_path.lower() and os.environ.get("QWEN3_THINKING", "0") != "1":
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            self.model_api = VllmServeAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args,
                extra_body=extra_body,
                fc_mode=False,
            )
        elif self.llm_agent_info["llm_type"] == "vllm_serve_fc":
            # Native-FC vLLM serve: requires --enable-auto-tool-choice --tool-call-parser=openai.
            self.model_api = VllmServeAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args,
                fc_mode=True,
            )
        elif self.llm_agent_info["llm_type"] == "openai":
            self.model_api = OpenaiAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args
            )
        elif self.llm_agent_info["llm_type"] == "claude":
            self.model_api = ClaudeAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args
            )
        elif self.llm_agent_info["llm_type"] == "kimi":
            extra_body = self.llm_agent_info.get("extra_body") or {}
            self.model_api = KimiAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args,
                extra_body=extra_body,
            )
        elif self.llm_agent_info["llm_type"] == "gpt5":
            self.model_api = GPT5API(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args,
                reasoning_effort=self.reasoning_effort,
            )
        elif self.llm_agent_info["llm_type"] == "grok":
            self.model_api = GrokAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args,
            )
        elif self.llm_agent_info["llm_type"] == "gemini":
            self.model_api = GeminiAPI(
                max_retries=self.max_retries,
                client=self.client,
                model_name=self.model_name,
                args=self.args
            )
        elif self.llm_agent_info["llm_type"] == "vllm":
            if "llama" in self.model_name.lower():
                self.model_api = Llama3API(
                    llm=self.llm,
                    tokenizer=self.tokenizer,
                    args=self.args
                )
            elif "qwen" in self.model_name.lower():
                self.model_api = QwenAPI(
                    llm=self.llm,
                    tokenizer=self.tokenizer,
                    args=self.args
                )
            else:
                raise NotImplementedError(f"Unsupported vLLM model: {self.model_name}")
        elif self.llm_agent_info["llm_type"] == "local":
            if "llama" in self.model_name.lower():
                self.model_api = Llama3API(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=self.args
                )
            elif "qwen" in self.model_name.lower():
                self.model_api = QwenAPI(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=self.args
                )
            else:
                raise NotImplementedError(f"Unsupported local model: {self.model_name}")
        else:
            raise NotImplementedError(f"Unsupported llm_type: {self.llm_agent_info['llm_type']}")

    def load_tools_from_file(self, tool_names, tools_info_path):
        """Load tools from file and convert to appropriate format based on llm_type."""
        if tools_info_path is None:
            # Tools will be set externally (e.g., AgentHarm).
            return
        # vllm_serve uses the text-based "openai" schema; native-FC providers use the full OpenAI FC schema.
        if self.llm_type == "vllm_serve":
            tool_format_type = "openai"
        elif self.llm_type in ("kimi", "gpt5", "vllm_serve_fc", "grok", "gemini"):
            tool_format_type = "vllm"
        else:
            tool_format_type = self.llm_type
        tools_info_df = pd.read_json(tools_info_path, lines=True)
        for tool_name in tool_names:
            _, name = tool_name.split("/")
            tool_instance = SimulatedTool(name, tools_info_df)
            self.tool_list[name] = tool_instance
            self.tools.append(tool_instance.get_tool_call_format(tool_format_type))
        self.normal_tools = copy.deepcopy(self.tool_list)

    def add_attacker_tool(self, attacker_tool_info: Dict[str, Any]):
        """Add an attacker tool to the agent's tool list."""
        attacker_tool = AttackerTool(attacker_tool_info)
        tool_name = attacker_tool.tool_name
        # vllm_serve uses the text-based "openai" schema; native-FC providers use the full OpenAI FC schema.
        if self.llm_type == "vllm_serve":
            tool_format_type = "openai"
        elif self.llm_type in ("kimi", "gpt5", "vllm_serve_fc", "grok", "gemini"):
            tool_format_type = "vllm"
        else:
            tool_format_type = self.llm_type
        self.tool_list[tool_name] = attacker_tool
        self.tools.append(attacker_tool.get_tool_call_format(tool_format_type))

    def get_system_prompt(self):
        """Generate system prompt for function calling mode."""
        if "system_prompt" in self.agent_config:
            self.complete_system_prompt = self.agent_config["system_prompt"]
            return self.complete_system_prompt

        prefix = "".join(self.agent_config["description"])
        self.prefix = prefix

        # Text-based tool calling for OpenAI/local/vLLM; native FC for Claude / Kimi / gpt-5.x / Grok / Gemini.
        if self.llm_agent_info["llm_type"] in ["openai", "vllm", "vllm_serve", "local"]:
            plan_instruction = get_text_tool_calling_instruction(self.tools)
        else:
            plan_instruction = get_function_calling_instruction(self.tools, self.llm_type)

        self.complete_system_prompt = prefix + plan_instruction
        return self.complete_system_prompt

    def generate_response(self, messages, tools):
        """Generate response using appropriate method based on LLM type."""
        return self.model_api._generate_response(messages, tools)
