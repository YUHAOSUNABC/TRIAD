"""
AgentHarm Tool Adapter.

Adapts inspect_ai @tool functions to work with our agent framework.
Uses inspect_ai's parse_tool_info for schema generation to ensure
compatibility with the original AgentHarm implementation.
"""

import asyncio
import sys
import os
import threading
from typing import Dict, List, Any, Optional, Callable

# Add inspect_evals to path for importing tools
INSPECT_EVALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "inspect_evals",
    "src"
)
if INSPECT_EVALS_PATH not in sys.path:
    sys.path.insert(0, INSPECT_EVALS_PATH)

# Tools (read_file, edit_txt, convert_pdf_to_text) open content files via
# relative paths like "benchmark/{benign,harmful}_tools/content/...".
# These resolve correctly only when cwd == inspect_evals/agentharm/.
# os.chdir is process-global, so we serialize tool execution with a lock to
# stay correct under ThreadPoolExecutor (NUM_WORKERS > 1).
_AGENTHARM_TOOL_CWD = os.path.join(INSPECT_EVALS_PATH, "inspect_evals", "agentharm")
_TOOL_CWD_LOCK = threading.Lock()

# Import inspect_ai's tool info parser for schema generation
from inspect_ai.tool._tool_info import parse_tool_info, ToolParam


def _tool_param_to_dict(param: ToolParam) -> Dict[str, Any]:
    """
    Convert inspect_ai ToolParam to OpenAI-compatible dict.

    This recursively converts the ToolParam structure to a clean dict
    that can be used with OpenAI's function calling API.
    """
    d: Dict[str, Any] = {"type": param.type}

    if param.description:
        d["description"] = param.description

    if param.items:
        d["items"] = _tool_param_to_dict(param.items)

    if param.enum:
        d["enum"] = param.enum

    if param.properties:
        d["properties"] = {k: _tool_param_to_dict(v) for k, v in param.properties.items()}

    if param.required:
        d["required"] = param.required

    if param.additionalProperties is not None:
        d["additionalProperties"] = param.additionalProperties

    return d


class AgentHarmTool:
    """
    Adapter class for AgentHarm tools.

    Wraps inspect_ai @tool decorated functions to work with our agent framework.
    Uses inspect_ai's parse_tool_info for accurate schema generation.
    """

    def __init__(self, tool_name: str, tool_func: Callable):
        """
        Initialize AgentHarmTool.

        Args:
            tool_name: Name of the tool
            tool_func: The tool function (result of calling @tool decorated function)
        """
        self.name = tool_name
        self._async_func = tool_func

        # Use inspect_ai to parse tool info (same as original AgentHarm)
        self._tool_info = parse_tool_info(tool_func)
        self.description = self._tool_info.description

    def run(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the tool synchronously.

        Args:
            params: Dictionary of parameter values

        Returns:
            Tool execution result as string
        """
        params = params or {}

        # Run async function synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            with _TOOL_CWD_LOCK:
                prev_cwd = os.getcwd()
                os.chdir(_AGENTHARM_TOOL_CWD)
                try:
                    result = loop.run_until_complete(self._async_func(**params))
                finally:
                    os.chdir(prev_cwd)
            return str(result) if result is not None else ""
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"

    def get_tool_call_format(self, llm_type: str = "vllm") -> Dict[str, Any]:
        """
        Get tool call format based on llm_type.

        Args:
            llm_type: "claude" or others (openai, vllm, local)

        Returns:
            Dict in the appropriate format for the llm_type
        """
        # Convert inspect_ai ToolParams to properties dict
        properties = {}
        for param_name, param in self._tool_info.parameters.properties.items():
            properties[param_name] = _tool_param_to_dict(param)

        required = self._tool_info.parameters.required or []

        if llm_type == "claude":
            # Anthropic format for native function calling
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        else:
            # OpenAI function calling format (for text-based and vllm)
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }


def load_tool_module(task_name: str):
    """
    Load the tool module for a given task.

    Args:
        task_name: "harmful" or "benign"

    Returns:
        The imported tool module
    """
    if task_name == "harmful":
        from inspect_evals.agentharm.benchmark import harmful_tools
        return harmful_tools
    elif task_name == "benign":
        from inspect_evals.agentharm.benchmark import benign_tools
        return benign_tools
    else:
        raise ValueError(f"Unknown task_name: {task_name}")


def create_tools_from_metadata(
    task_name: str,
    target_functions: List[str]
) -> Dict[str, AgentHarmTool]:
    """
    Create AgentHarmTool instances for the specified tools.

    Args:
        task_name: "harmful" or "benign"
        target_functions: List of tool names to load

    Returns:
        Dict mapping tool names to AgentHarmTool instances
    """
    tools_module = load_tool_module(task_name)
    tools = {}

    for tool_name in target_functions:
        try:
            # Get the tool function from the module
            tool_factory = getattr(tools_module, tool_name, None)
            if tool_factory is None:
                print(f"[Warning] Tool '{tool_name}' not found in {task_name}_tools")
                continue

            # Call the factory to get the async run function
            tool_func = tool_factory()

            # Create wrapper
            tools[tool_name] = AgentHarmTool(tool_name, tool_func)

        except Exception as e:
            print(f"[Warning] Error loading tool '{tool_name}': {e}")

    return tools
