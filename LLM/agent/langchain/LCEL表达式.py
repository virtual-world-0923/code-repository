from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
OPENAI_KEY=os.environ.get("OPENAI_KEY")
OPENAI_URL=os.environ.get("OPENAI_URL")
OPENAI_MODEL=os.environ.get("OPENAI_MODEL")

# 1. 准备零件一：提示词模板
prompt = ChatPromptTemplate.from_template("请用一句话向小学生解释什么是：{topic}")

# 2. 准备零件二：大模型（只要兼容 OpenAI 格式都能这么写）
model = ChatOpenAI(
    temperature=0,
    api_key=OPENAI_KEY,
    base_url=OPENAI_URL,
    model=OPENAI_MODEL,
)

# 3. 准备零件三：输出解析器（把模型返回的一长串复杂 JSON 对象，直接提取成纯文本字符串）
parser = StrOutputParser()

# 4. 【核心】用 LCEL 管道符 | 串成一条链（Chain）
chain = prompt | model | parser

# 5. 运行整条流水线！
result = chain.invoke({"topic": "量子纠缠"})
print(result)
