"""串联文档读取、向量检索和模型回答的 RAG 示例入口。"""

from .config import DATA_FILE, DEFAULT_QUERY, FAISS_PATH  # 导入语料路径、默认问题和 LlamaIndex 专属索引目录。
from .documents import load_and_split_documents  # 导入语料解析和递归字符切块函数。
from .llm import answer_with_llm  # 导入基于检索资料生成回答的函数。
from .vector_store import create_embedding_model, load_or_create_vector_store, retrieve  # 导入嵌入模型、索引和检索函数。


def run_rag() -> str:  # 声明运行完整 RAG 流程的函数并标注字符串返回类型。
    """读取语料、建立或载入 FAISS 索引、检索相关资料并生成回答。

    Returns:
        str: 基于检索资料生成的最终回答。
    """
    nodes = load_and_split_documents(DATA_FILE)  # 调用语料解析函数读取 DATA_FILE 并生成带来源的字符切分节点。
    embeddings = create_embedding_model()  # 调用嵌入模型工厂创建索引与查询共用的 OpenAI 嵌入模型。
    index = load_or_create_vector_store(nodes, embeddings, FAISS_PATH)  # 调用索引加载或创建函数复用本地 FAISS 索引并保持独立存储。
    retrieved_nodes = retrieve(index, DEFAULT_QUERY)  # 调用检索函数按默认问题获取并筛选相关节点。
    return answer_with_llm(DEFAULT_QUERY, retrieved_nodes)  # 调用回答函数根据检索节点生成并返回最终答案。


if __name__ == "__main__":  # 仅在模块作为程序入口执行时运行示例流程。
    run_rag()  # 调用 run_rag 执行完整的检索增强生成流程。
