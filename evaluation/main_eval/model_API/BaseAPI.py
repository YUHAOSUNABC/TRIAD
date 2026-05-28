from typing import Any, List, Dict


class BaseAPI:
    def __init__(self):
        pass

    def _generate_response(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """Generate response from model."""
        raise NotImplementedError

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to model-specific format."""
        raise NotImplementedError

    def _parse_response(self, *args, **kwargs) -> Dict:
        """
        Parse model response to standardized format.

        Note: Signature varies by subclass:
        - OpenaiAPI: (content: str, tool_calls)
        - QwenAPI/Llama3API: (text: str)
        """
        raise NotImplementedError