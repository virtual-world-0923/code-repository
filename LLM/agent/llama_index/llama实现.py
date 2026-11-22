import os

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike


# 1. 业务函数与工具定义
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气情况。

    :param location: 目标城市名称，例如：北京
    :param unit: 温度单位
    """
    return f"{location} 的当前天气为晴朗，温度 22 {unit}。"


weather_tool = FunctionTool.from_defaults(fn=get_current_weather, name="get_current_weather")

# 2. 将 LlamaIndex 的 OpenAI LLM 改指向免费接口
llm = OpenAILike(
    is_chat_model=True,
    is_function_calling_model=True,
    model=os.environ.get("OPENAI_MODEL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_base=os.environ.get("OPENAI_BASE_URL"),
)

# 3. 执行闭环
response = llm.predict_and_call(
    tools=[weather_tool],
    user_msg="北京今天天气怎么样？"
)

print("【LlamaIndex 真实调用最终回答】:\n", response.response)