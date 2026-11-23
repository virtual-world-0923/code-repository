from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
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


# 根据工具名称建立映射关系
tool_map = {tool.name: tool for tool in tools}


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
for tool_call in response.tool_calls:
    # 获取工具名称
    tool_name = tool_call["name"]

    # 获取工具参数
    tool_args = tool_call["args"]

    # 获取工具调用 ID
    tool_call_id = tool_call["id"]

    # 根据名称查找工具
    selected_tool = tool_map.get(tool_name)

    # 执行工具
    tool_result = selected_tool.invoke(tool_args)

    # 创建 ToolMessage
    tool_message = ToolMessage(
        # 工具执行结果
        content=str(tool_result),
        # 必须对应模型请求中的工具调用 ID
        tool_call_id=tool_call_id,
        # 工具名称
        name=tool_name,
    )

    # 将工具结果加入消息历史
    messages.append(tool_message)


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
