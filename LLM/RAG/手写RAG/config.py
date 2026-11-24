import os
from pathlib import Path

current_file_path = Path(__file__).resolve()

# 文档目录
DATA_DIR = current_file_path.parent.parent / "data"

# 向量数据库目录
VECTOR_DB_DIR = current_file_path.parent / "vector_db"
FAISS_INDEX_PATH = str(VECTOR_DB_DIR / "faiss.index")
METADATA_JSON_PATH = str(VECTOR_DB_DIR / "metadata.json")

# Embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 文本切分配置
CHUNK_SIZE = 100 # 每个文本块的最大字符数。
CHUNK_OVERLAP = 10 # 相邻文本块之间重复的字符数。
TOP_K = 1 # 每次检索返回最相关的几个文本块。
"""
    例如：
        原始文本：ABCDEFGHIJKLMN
    假设：
        CHUNK_SIZE = 6
        CHUNK_OVERLAP = 2
    切分结果：
        Chunk 1: ABCDEF
        Chunk 2: EFGHIJ
        Chunk 3: IJKLMN

    为什么要有重叠？
    因为直接切分可能破坏句子之间的语义关联。适当重叠可以减少这种信息损失。
"""

# 大模型配置
OPENAI_KEY=os.environ.get("OPENAI_KEY")
OPENAI_URL=os.environ.get("OPENAI_URL")
OPENAI_MODEL=os.environ.get("OPENAI_MODEL")