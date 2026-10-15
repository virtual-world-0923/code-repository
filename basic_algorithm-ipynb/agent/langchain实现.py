import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# 1. 声明 LangChain 工具
@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气情况。

    Args:
        location: 目标城市名称，例如：北京、上海
        unit: 温度单位，celsius 或 fahrenheit
    """
    return f"{location} 的当前天气为晴朗，温度 22 {unit}。"


# 2. 将 ChatOpenAI 配置为与代码七相同的免费 API 终点
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY_HERE"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

# 绑定工具
llm_with_tools = llm.bind_tools([get_current_weather])

# 3. 真实运行请求
query = "北京今天天气怎么样？"
response = llm_with_tools.invoke(query)

if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_result = get_current_weather.invoke(tool_call)

        final_response = llm.invoke([
            ("user", query),
            response,
            tool_result
        ])
        print("【LangChain 真实调用最终回答】:\n", final_response.content)