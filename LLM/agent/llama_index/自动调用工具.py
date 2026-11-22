import asyncio
import os

from langchain_core.messages import ChatMessage
from llama_index.core.agent.workflow import AgentInput, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike

from LLM.agent.httpx截取 import http_client

llm = OpenAILike(
    model=os.environ.get("OPENAI_MODEL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_base=os.environ.get("OPENAI_BASE_URL"),
    is_chat_model=True, # 调用没有支持的第三方模型时必须写
    is_function_calling_model=True,# 调用没有支持的第三方模型时必须写
    http_client=http_client,
)


def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}今天晴，25℃。"


weather_tool = FunctionTool.from_defaults(fn=get_weather)

agent = FunctionAgent(
    tools=[weather_tool],
    llm=llm,
)


async def main():
    event = AgentInput(
    input=[ChatMessage(role="user", content="北京今天天气怎么样？")]
    )
    response = await agent.run(start_event=event)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
