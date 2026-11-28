"""提供基于智谱 GLM 的 RAG 回答生成功能。"""

import os  # 导入操作系统接口，用于从环境变量读取 API Key。
from collections.abc import Sequence  # 导入 Sequence，用于声明只读文档序列参数。

from config import LLM_API_KEY_ENV, LLM_MODEL  # 导入智谱 API Key 环境变量名称和模型名称。
from langchain_core.documents import Document  # 导入 Document 类型，作为回答上下文文档的类型。

from LLM.http.httpx截取 import http_client


def build_prompt(query: str, retrieved_docs: Sequence[Document]) -> str:
    """将问题和检索到的资料组合成限制模型回答范围的提示词。

    Args:
        query (str): 用户提出的问题。
        retrieved_docs (Sequence[Document]): 用作回答依据的检索文档序列。

    Returns:
        str: 包含系统规则、参考资料和用户问题的完整提示词。
    """
    context_parts: list[str] = []
    for position, document in enumerate(retrieved_docs, 1):
        context_parts.append(f"--- 资料{position}（{document.metadata['source']}）---\n{document.page_content}")
    context = "\n\n".join(context_parts)
    prompt = f"""【系统指令】
你是一个严谨的问答助手。请严格基于下面提供的资料回答问题。

规则：
1. 只使用资料中的信息回答，不要编造或补充资料外的内容
2. 如果资料中没有明确答案，必须回答"不知道"
3. 回答要简洁，直接给结论，必要时引用资料原文
4. 如果多份资料有矛盾，指出矛盾并说明各资料内容
5. 回答前，先列出资料中的相关条件，再逐一比对用户情况，最后给出结论

【参考资料】
{context}

【问题】
{query}

【回答】"""

    return prompt


def answer_with_llm(query: str, retrieved_docs: Sequence[Document]) -> str:
    """使用检索资料构造提示词并调用智谱模型生成回答。

    Args:
        query (str): 用户提出的问题。
        retrieved_docs (Sequence[Document]): 作为回答依据的检索文档序列。

    Returns:
        str: 智谱模型生成的回答；没有检索资料时返回“不知道”提示。

    Raises:
        ImportError: 未安装 zhipuai 客户端依赖时抛出。
        RuntimeError: API Key 未配置，或模型没有返回文本内容时抛出。
    """

    if not retrieved_docs:
        message = "不知道。资料库中没有找到与您问题相关的信息。"
        return message

    prompt = build_prompt(query, retrieved_docs)  # 以用户问题和检索文档构造模型提示词。
    print(f"\n   ⏳ 正在调用 {LLM_MODEL} 生成回答...")

    from zai import ZhipuAiClient

    api_key = os.getenv(LLM_API_KEY_ENV)

    client = ZhipuAiClient(api_key=api_key, http_client=http_client)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[  # 设置聊天消息列表，依次提供系统约束和包含资料的问题。
            {"role": "system", "content": "你是一个严谨的问答助手。严格基于用户提供的资料回答问题，不编造。资料中没有的答案就说'不知道'。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # 设置较低采样温度，以降低回答随机性。
        max_tokens=1024,  # 限制模型本次回答最多生成的 token 数量。
    )
    answer = response.choices[0].message.content

    print("\n   ✅ LLM 返回回答:")
    print(f"   {'─' * 50}")
    print(f"   {answer}")
    print(f"   {'─' * 50}")
    return answer
