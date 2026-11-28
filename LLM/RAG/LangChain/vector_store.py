"""提供 FAISS 索引检查、嵌入模型创建和向量库加载功能。"""

from pathlib import Path

from config import *
from langchain_community.vectorstores import FAISS  # 导入 FAISS 向量库实现，用于持久化和相似度检索。
from langchain_core.documents import Document  # 导入 Document 类型，作为待索引文本块的类型。
from langchain_huggingface import HuggingFaceEmbeddings  # 导入 Hugging Face 嵌入模型包装器，将文本编码为向量。


def _is_faiss_index_available(index_path: Path) -> bool:  # 声明检查本地 FAISS 索引是否完整的函数及其类型。
    """判断指定目录中是否存在完整的 FAISS 本地索引文件。

    Args:
        index_path (Path): 需要检查的 FAISS 索引目录。

    Returns:
        bool: 当目录、index.faiss 和 index.pkl 均存在时返回 True，否则返回 False。
    """
    faiss_file = index_path / "index.faiss"  # 使用 Path 运算符拼接 FAISS 二进制索引文件路径。
    pkl_file = index_path / "index.pkl"  # 使用 Path 运算符拼接文档元数据序列化文件路径。
    return index_path.is_dir() and faiss_file.is_file() and pkl_file.is_file()  # 调用 Path.is_dir 和 Path.is_file 检查目录及两个索引文件。


def create_embedding_model() -> HuggingFaceEmbeddings:
    """根据项目配置创建 Hugging Face 文本嵌入模型。

    Returns:
        HuggingFaceEmbeddings: 已配置模型名称、计算设备和向量归一化选项的嵌入模型。
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,  # 指定将文本编码为向量所使用的预训练模型。
        model_kwargs={"device": DEVICE},  # 模型推理设备，例如 cuda 或 cpu。
        encode_kwargs={"normalize_embeddings": True},  # 对输出向量进行归一化
    )
    return embeddings


def load_or_create_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    index_path: Path,
) -> FAISS:
    """加载完整的本地 FAISS 索引；不存在时根据文档块新建并保存索引。

    Args:
        documents (list[Document]): 需要写入索引的新切分文档列表。
        embeddings (HuggingFaceEmbeddings): 用于加载索引和编码新文档的嵌入模型。
        index_path (Path): FAISS 索引的本地目录。

    Returns:
        FAISS: 已从磁盘加载或新建并保存的 FAISS 向量库。
    """
    if _is_faiss_index_available(index_path):
        return FAISS.load_local(
            str(index_path),
            embeddings,  # 传入嵌入模型，以便对查询文本进行与索引一致的向量化。
            allow_dangerous_deserialization=True,  # 允许加载本地 index.pkl；只应加载可信来源的索引文件。
        )

    vector_store = FAISS.from_documents(documents, embeddings)  # 将文档编码并创建新向量索引。
    print(f"FAISS 索引类型: {type(vector_store.index).__name__}")  # 调用 print 和 type 显示新建索引的底层类型。
    print("✅ 向量库构建完成")
    vector_store.save_local(str(index_path))  # 将索引及文档元数据保存到 index_path。
    print(f"💾 已保存 FAISS 库到：{index_path}（共 {vector_store.index.ntotal} 条向量）")
    return vector_store


def retrieve(vector_store: FAISS, query: str, top_k: int = TOP_K) -> list[Document]:  # 声明向量检索函数的参数和返回类型。
    """从 FAISS 向量库检索问题相关文档，并筛除低于相似度阈值的结果。

    Args:
        vector_store (FAISS): 用于执行相似度搜索的向量库。
        query (str): 用户提出的检索问题。
        top_k (int): 最多获取的候选文档数量，默认使用配置中的 TOP_K。

    Returns:
        list[Document]: 相似度达到配置阈值的文档，按检索相关性排序。
    """
    docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)

    retrieved: list[Document] = []  # 创建列表，用于保留达到采纳阈值的文档。

    for position, (document, score) in enumerate(docs_with_scores, 1):
        similarity = 1 / (1 + score)
        print("检索到的文档：")
        print(position, "\n 分数：", similarity, "\n 文档：\n", document)
        if similarity >= SIMILARITY_THRESHOLD:  # 判断当前结果是否达到配置的相似度下限。
            retrieved.append(document)  # 调用 list.append 将达标文档加入最终检索结果。

    if not retrieved:
        print(f"\n   ❌ 没有找到超过阈值({SIMILARITY_THRESHOLD})的片段")  # 调用 print 提示没有可采纳的检索片段。
    return retrieved
