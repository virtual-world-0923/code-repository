from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import os

OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_URL = os.environ.get("OPENAI_URL")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

from LLM.agent.httpx截取 import http_client

model = ChatOpenAI(
    temperature=0,
    api_key=OPENAI_KEY,
    base_url=OPENAI_URL,
    model=OPENAI_MODEL,
    http_client=http_client,
)


# 1. 用 Pydantic 定义你想要的“标准数据表格”
class UserInfo(BaseModel):
    name: str = Field(description="用户的真实姓名")
    age: int = Field(description="用户的年龄，必须是整数")
    skills: list[str] = Field(description="用户掌握的技能列表")


# 2. 给模型绑定输出格式
structured_model = model.with_structured_output(UserInfo)

messages = [
    HumanMessage(
        content="大家好，我叫小明，今年刚满28岁，平时工作主要写Python和SQL，偶尔也搞搞Docker。"
    ),
    SystemMessage(
        content="你是一个信息提取助手。请从用户输入的文本中精准提取关键数据并按要求输出。"
    ),
]

# 3. 随便丢一段乱七八糟的自我介绍进去
result = structured_model.invoke(messages)

# 4. 拿到的直接就是干净的 Python 对象，可以直接点出属性！
print(type(result))
print(result.name)
print(result.age + 2)
print(result.skills)
