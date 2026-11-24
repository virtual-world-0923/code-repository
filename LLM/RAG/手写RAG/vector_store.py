import json  # 调用 Python 标准库 json；作用：在 Python 对象与 JSON 文本之间转换；这里用于保存/读取元数据
import os
import shutil
import tempfile
from pathlib import Path

import faiss  # 导入 FAISS 向量检索库；作用：创建向量索引、添加向量、执行相似度搜索、读写索引文件

from LLM.RAG.手写RAG.config import FAISS_INDEX_PATH, METADATA_JSON_PATH, VECTOR_DB_DIR


class VectorStore:
    """
    FAISS 索引操作和文本元数据管理
    """

    def __init__(self):

        self.index = None  # FAISS 索引
        self.metadata = []  # 元数据列表；每个元素保存一个文本块及其 source、page 等信息；列表顺序必须与 FAISS 向量顺序对应

    def add(self, vectors, documents):
        """
        添加向量和对应的文本元数据

        vectors 是待添加的向量矩阵
        documents 是每个向量对应的文本/元数据列表
        """

        # 读取向量矩阵的第二个维度，也就是一个词用多少维表示
        dimension = vectors.shape[1]

        if (self.index is None):  # 判断当前对象是否还没有 FAISS 索引；首次添加向量时需要先创建索引
            self.index = faiss.IndexFlatIP(dimension)  # 创建精确内积检索索引；dimension 是每条向量的维度

        self.index.add(vectors)  #  vectors 二维向量矩阵，形状为 (数量, dimension)，把这些向量加入索引

        self.metadata.extend(documents)  # documents 是与 vectors 一一对应的元数据列表；将其逐项追加到 metadata，保持向量位置与元数据位置一致

    def search(self, query_vector, top_k=3):
        """
        向量相似度检索

        query_vector 是用户问题的向量矩阵，通常形状为 (1, dimension)
        top_k 是最多返回几个最相似结果，默认 3

        """

        scores, indices = (
            self.index.search(  # 返回 scores（相似度分数）和 indices（对应向量在索引中的位置）
                query_vector,  # 要搜索的查询向量；必须与索引中的向量维度一致
                top_k,  # 每条查询向量返回的近邻数量；这里最多返回 top_k 条结果
            )
        )  # scores 和 indices 通常形状都是 (查询数量, top_k)

        results = []  # 用于保存检索结果及其文本、来源、页码

        for score, index in zip(scores[0], indices[0]):  # 将第一条查询的分数和索引逐个配对；scores[0]/indices[0] 表示第一条查询的结果
            if (index == -1):  # FAISS 用 -1 表示没有找到有效结果（例如索引中的向量数量少于 top_k）；检查并跳过无效位置
                continue

            results.append(
                {
                    "score": float(score),  # 把 FAISS 返回的分数转换为 Python 浮点数
                    "text": self.metadata[index]["text"],  # 用 FAISS 返回的 index 作为位置，从 metadata 中取回对应文本块的 text 字段
                    "source": self.metadata[index]["source"],  # 从相同元数据项中取出 source 字段；通常是原始 PDF/TXT 文件路径
                    "page": self.metadata[index]["page"],  # 从相同元数据项中取出 page 字段；通常是 PDF 页码，TXT 可以是 None
                }
            )

        return results

    def save(self):
        """保存文档向量和元数据"""

        # 创建目录
        # parents=True. 对应 os.makedirs 的递归创建功能，当父级目录不存在时会自动一并创建。
        # 与 os.makedirs 中的作用一致，如果目标目录已经存在，不会抛出 FileExistsError 异常
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


        # 保存文档向量
        _faiss_write_safe(self.index, FAISS_INDEX_PATH)

        # 保存文档元数据
        with open(
            os.path.join(METADATA_JSON_PATH),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(  # 把 Python 对象序列化为 JSON 并写入文件
                self.metadata,
                f,
                ensure_ascii=False,  # 设为 False 时保留中文等非 ASCII 字符，不转成 \uXXXX 形式
                indent=2,  # 设置 JSON 缩进空格数为 2，使文件格式更易阅读
            )

    def load(self):
        """加载向量索引和元数据"""

        self.index = _faiss_read_safe(FAISS_INDEX_PATH)  # 读取索引文件

        with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)  # 读取文件中的 JSON 并转换成 Python 对象

def _faiss_write_safe(index, final_path: str):
    """
    安全写入 FAISS 索引，彻底绕开 Windows 中文/长路径问题。

    策略：在 %TEMP% 下用纯英文短路径写入 → shutil.move 到最终位置。
    Python 的 shutil.move 对中文路径兼容性远好于 faiss 内部。
    """
    folder = os.path.dirname(final_path) or "."
    os.makedirs(folder, exist_ok=True)

    # 在系统临时目录创建纯英文临时文件
    fd, tmp_path = tempfile.mkstemp(suffix=".index", prefix="faiss_tmp_")
    os.close(fd)

    try:
        # faiss 写到临时文件（纯英文路径，100% 稳定）
        faiss.write_index(index, tmp_path)
        # 用 Python 移动到最终位置（处理中文路径没问题）
        shutil.move(tmp_path, final_path)
    except Exception:
        # 清理临时文件
        if os.path.exists(tmp_path): # 如果临时文件存在
            os.unlink(tmp_path) # 删除
        raise

def _faiss_read_safe(final_path: str):
    """
    安全读取 FAISS 索引。
    复制到临时文件再加载，避开 Windows 路径问题。
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".index", prefix="faiss_tmp_")
    os.close(fd)

    try:
        shutil.copy2(final_path, tmp_path)
        index = faiss.read_index(tmp_path)
        return index
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
