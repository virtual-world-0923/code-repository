from sentence_transformers import SentenceTransformer
from LLM.RAG.手写RAG.config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(EMBEDDING_MODEL,device="cuda")

    def encode(self, texts):

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True, # 对生成的向量进行归一化
            convert_to_numpy=True # 将生成的向量转换成 NumPy 数组
        )

        return vectors.astype("float32")