from config import *
from documents import load_and_split_documents
from vector_store import create_embedding_model, load_or_create_vector_store, retrieve

from llm import answer_with_llm


def run_rag() -> str:

    # 读取文档，并切分
    documents = load_and_split_documents(DATA_FILE)

    # 加载文本向量模型
    embeddings = create_embedding_model()

    # 创建文本向量索引，并保存到磁盘
    vector_store = load_or_create_vector_store(documents, embeddings, FAISS_PATH)

    #根据问题搜索相关文档
    retrieved_docs = retrieve(vector_store, DEFAULT_QUERY)  # 调用检索函数，根据示例问题从向量库获取相关文档。

    answer_with_llm(DEFAULT_QUERY, retrieved_docs)  # 调用模型回答函数并返回基于检索资料生成的最终回答。


if __name__ == "__main__":
    run_rag()
