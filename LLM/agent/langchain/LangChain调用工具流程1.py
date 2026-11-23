from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from LLM.agent.httpx截取 import http_client

# ==================================================
# 定义工具
# ==================================================

class get_weather_Input(BaseModel):
    city: str = Field(description=("查询指定城市的天气"))


@tool(args_schema=get_weather_Input)
def get_weather(city: str) -> str:
    """
    查询指定城市的天气。

    Args:
        city: 城市名称。
    """
    return f"{city}今天晴天，25度"


# 统一工具列表
tools = [get_weather]


# ==================================================
# 创建模型
# ==================================================

model = ChatOpenAI(
    model="GLM-4-Flash",
    temperature=0,
    api_key="5e4e165df1c24405873fcb3621b5d7a0.6CEDD1OabBprR4p3",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    http_client=http_client,
)


# ==================================================
# 绑定工具
# ==================================================

model_with_tools = model.bind_tools(tools)

# ------------------------------------------
# 创建消息历史
# ------------------------------------------

messages = [HumanMessage(content="查看北京天气")]

# ==================================================
# 调用模型
# ==================================================

response = model_with_tools.invoke(messages)

# 模型返回的内容添加到历史记录
messages.append(response)

# ==================================================
# 执行工具
# ==================================================

# 创建工具执行节点（直接传 list，内部自动做映射）
tool_node = ToolNode(tools)

"""
直接把模型返回的 AIMessage 交给 ToolNode，就可以自动查找并执行其中的所有工具调用，不需要自己维护 tool_map，也不需要手动调用 tool.invoke()
参数：
    messages：默认固定的状态字段名称。ToolNode 默认从这个字段中读取消息，并查找包含工具调用请求的 AIMessage。
    response：AIMessage 类型，即模型返回的信息，其中的 tool_calls 包含工具名称、调用参数和工具调用 ID。一个 AIMessage 可以包含多个工具调用。
    runtime=Runtime()：提供 LangGraph 运行时上下文。独立调用 ToolNode.invoke() 时，在当前使用环境中需要手动提供；如果通过 StateGraph 或 Agent 执行，则由框架管理。
返回：返回一个字典，其中 messages 列表包含工具执行结果，每个结果都是标准的 ToolMessage 对象。
    {
    "messages": [
        ToolMessage(content="北京晴，25度", tool_call_id="call_1", ...),
        ToolMessage(content="上海雨，20度", tool_call_id="call_2", ...)
    ]
}
"""
result = tool_node.invoke({"messages": [response]},runtime=Runtime())

# 将工具结果加入消息历史（注意用 extend）
messages.extend(result["messages"])


# ==================================================
# 带着工具执行的结果再次发给模型
# ==================================================

response = model_with_tools.invoke(messages)


# ==================================================
# 最终返回
# ==================================================

print("\n" + "=" * 70)
print(f"最终结果：{response.content}")
print("=" * 70)


# 关闭 HTTP 客户端
http_client.close()
