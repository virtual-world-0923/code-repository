from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime


@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}：晴天"


tools = [get_weather]

tool_node = ToolNode(tools)

# 使用 AIMessage 对象
response = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "北京"},
            "id": "call_123",
            "type": "tool_call",
        }
    ],
)

result = tool_node.invoke({"messages": [response]}, runtime=Runtime())

print(result)
