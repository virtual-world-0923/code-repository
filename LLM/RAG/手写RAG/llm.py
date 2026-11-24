from openai import OpenAI

from LLM.http.httpx截取 import http_client
from LLM.RAG.手写RAG.config import OPENAI_KEY, OPENAI_MODEL, OPENAI_URL


class LLM:

    def __init__(self):

        self.client = OpenAI(
            api_key=OPENAI_KEY,
            base_url=OPENAI_URL,
            http_client=http_client
        )

    def generate(self, question, context):

        prompt = f"""
你是一个知识库问答助手。

请严格根据下面提供的参考资料回答问题。

如果参考资料中没有答案，请明确说明不知道。
不要编造信息。

参考资料：
{context}

用户问题：
{question}

请给出准确、清晰的回答。
"""

        response = self.client.chat.completions.create(

            model=OPENAI_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的知识库问答助手。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0
        )

        return response.choices[0].message.content