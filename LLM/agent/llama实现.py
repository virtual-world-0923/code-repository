import os
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


# 1. 业务函数与工具定义
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气情况。

    :param location: 目标城市名称，例如：北京
    :param unit: 温度单位
    """
    return f"{location} 的当前天气为晴朗，温度 22 {unit}。"


weather_tool = FunctionTool.from_defaults(fn=get_current_weather, name="get_current_weather")

# 2. 将 LlamaIndex 的 OpenAI LLM 改指向免费接口
llm = OpenAI(
    model="glm-4-flash",
    api_key=os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY_HERE"),
    api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 3. 执行闭环
response = llm.predict_and_call(
    tools=[weather_tool],
    user_msg="北京今天天气怎么样？"
)

print("【LlamaIndex 真实调用最终回答】:\n", response.response)