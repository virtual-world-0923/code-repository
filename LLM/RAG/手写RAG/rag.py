from document_loader import load_documents
from embedding import EmbeddingModel
from text_splitter import split_data, split_documents
from vector_store import VectorStore

from llm import LLM
from LLM.RAG.手写RAG.config import *


class RAG:
    def __init__(self):

        self.embedding = EmbeddingModel()

        self.vector_store = VectorStore()

        self.llm = LLM()

    def delete_knowledge_base(self):
        print("删除已有知识库")
        if VECTOR_DB_DIR.exists():
            self.vector_store.delete()
        print(f"🗑️ 已删除旧向量库: {VECTOR_DB_DIR}")

    def build_knowledge_base(self):

        if VECTOR_DB_DIR.exists():
            print("加载已有知识库")

            self.vector_store.load()

            print("知识库加载成功！")
        else:

            """构建知识库"""

            print("1. 加载文档")

            documents = load_documents(DATA_DIR)

            print(f"加载文档数量：{len(documents)}")

            print("2. 文本切分")

            chunks = split_data(documents, CHUNK_SIZE, CHUNK_OVERLAP)

            print(f"文本块数量：{len(chunks)}")

            print("3. Embedding")

            texts = [chunk["text"] for chunk in chunks]

            vectors = self.embedding.encode(texts)

            print("4. 构建向量数据库")

            self.vector_store.add(vectors, chunks)

            print("5. 保存向量数据库")

            self.vector_store.save()

            print("知识库构建完成！")

    def ask(self, question):
        """RAG 问答"""

        print("\n1. 用户问题 Embedding")

        query_vector = self.embedding.encode([question])

        print("2. 向量检索")

        results = self.vector_store.search(query_vector, TOP_K)

        print("检索结果：")

        for result in results:
            print(f"相似度：{result['score']:.4f}")

            print(f"来源：{result['source']}")

            print(f"内容：{result['text'][:100]}")

        print("3. 拼接上下文")

        context = "\n\n".join(result["text"] for result in results)

        print("4. 调用大模型")

        answer = self.llm.generate(question, context)

        return answer


rag = RAG()

# 删除知识库
# rag.delete_knowledge_base()

# 构建知识库,存在加载，不存在构建
rag.build_knowledge_base()

question = "入职两年，年假怎么休？"
answer = rag.ask(question)

print(answer)