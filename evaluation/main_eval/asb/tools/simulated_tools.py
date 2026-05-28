class BaseTool:
    """Base class for calling tool
    """
    def __init__(self):
        pass

    def run(self):
        pass

class SimulatedTool(BaseTool):
    def __init__(self, name, tools_info_df):
        """
        Args:
            name: Tool name
            tools_info_df: Pre-loaded DataFrame containing tool information
        """
        super().__init__()
        self.name = name
        tool = tools_info_df[tools_info_df['Tool Name'] == name]
        self.description = tool['Description'].iloc[0]
        self.expected_achivement = tool['Expected Achievements'].iloc[0]

    def run(self, params=None) -> str:
        return self.expected_achivement

    def get_tool_call_format(self, llm_type="vllm"):
        """Get tool format for different LLM types.

        Args:
            llm_type: "openai" (text-based), "vllm" (OpenAI FC), "claude" (Anthropic)
        """
        if llm_type == "openai":
            return {
                "name": self.name,
                "description": self.description,
            }
        elif llm_type == "claude":
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        else:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                }
            }


class AttackerTool(BaseTool):
    def __init__(self, attacker_tool):
        super().__init__()
        self.tool_name=attacker_tool['Attacker Tool']
        self.tool_description=attacker_tool['Description']
        self.attack_goal = attacker_tool['Attack goal']

    def run(self,params=None):
        return f"You have used the tool {self.tool_name}. You have achieved the goal: {self.attack_goal}"

    def get_tool_call_format(self, llm_type="vllm"):
        """Get tool format for different LLM types.

        Args:
            llm_type: "openai" (text-based), "vllm" (OpenAI FC), "claude" (Anthropic)
        """
        if llm_type == "openai":
            return {
                "name": self.tool_name,
                "description": self.tool_description,
            }
        elif llm_type == "claude":
            return {
                "name": self.tool_name,
                "description": self.tool_description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        else:
            return {
                "type": "function",
                "function": {
                    "name": self.tool_name,
                    "description": self.tool_description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                }
            }
