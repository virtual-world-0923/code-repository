"""提供嵌入模型创建、FAISS 索引持久化和相似度检索功能。"""

from pathlib import Path  # 导入 Path，以便检查和传递本地索引目录。

import faiss  # 导入 FAISS，用于创建底层 L2 向量索引。
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage  # 导入 LlamaIndex 索引构建、持久化上下文和加载函数。
from llama_index.core.schema import TextNode  # 导入 TextNode，声明待索引节点的类型。
from llama_index.embeddings.openai import OpenAIEmbedding  # 导入 OpenAIEmbedding，将文本和查询编码为向量。
from llama_index.vector_stores.faiss import FaissVectorStore  # 导入 LlamaIndex 的 FAISS 适配器。

from .config import (  # 从配置模块导入当前向量索引需要的模型和检索参数。
    EMBEDDING_DIMENSION,  # 导入 FAISS 索引所需的向量维度。
    OPENAI_EMBEDDING_MODEL,  # 导入 OpenAI 嵌入模型名称。
    SIMILARITY_THRESHOLD,  # 导入采纳检索结果的相似度阈值。
    TOP_K,  # 导入默认候选文档数量。
    get_openai_api_key,  # 导入读取并校验 OpenAI API Key 的配置函数。
)  # 完成向量库配置项导入。


def _is_faiss_index_available(index_path: Path) -> bool:  # 声明检查 LlamaIndex FAISS 持久化文件的辅助函数和类型。
    """判断指定目录中是否存在可供 LlamaIndex 加载的完整索引文件。

    Args:
        index_path (Path): 需要检查的本地 FAISS 索引目录。

    Returns:
        bool: 当 FAISS 向量文件、节点文档库和索引元数据均存在时返回 True。
    """
    vector_file = index_path / "default__vector_store.json"  # 拼接 FAISS 适配器持久化的底层索引文件路径。
    docstore_file = index_path / "docstore.json"  # 拼接 LlamaIndex 节点文档库文件路径。
    index_store_file = index_path / "index_store.json"  # 拼接 LlamaIndex 索引结构元数据文件路径。
    return (  # 只有目录和所有必需持久化文件均存在时才判定索引可加载。
        index_path.is_dir()  # 调用 Path.is_dir 检查索引目录是否存在。
        and vector_file.is_file()  # 调用 Path.is_file 检查 FAISS 向量文件是否存在。
        and docstore_file.is_file()  # 调用 Path.is_file 检查节点文档库文件是否存在。
        and index_store_file.is_file()  # 调用 Path.is_file 检查索引结构文件是否存在。
    )  # 返回完整索引文件检查结果。


def create_embedding_model() -> OpenAIEmbedding:  # 声明嵌入模型工厂函数的返回类型。
    """根据项目配置创建 OpenAI 文本嵌入模型。

    Returns:
        OpenAIEmbedding: 使用配置的模型名称及向量维度的嵌入模型。
    """
    api_key = get_openai_api_key()  # 调用配置函数读取 OpenAI API Key，未配置时会给出明确错误。
    embeddings = OpenAIEmbedding(  # 调用 OpenAIEmbedding 构造函数创建 LlamaIndex 嵌入模型。
        model=OPENAI_EMBEDDING_MODEL,  # 传入 OpenAI 嵌入模型名称。
        dimensions=EMBEDDING_DIMENSION,  # 传入 FAISS 索引要求的向量维度。
        api_key=api_key,  # 传入从 OPENAI_API_KEY 环境变量读取的凭据。
    )  # 完成嵌入模型创建。
    return embeddings  # 返回供索引构建和查询复用的嵌入模型。


def load_or_create_vector_store(
    nodes: list[TextNode],  # 声明待建立索引的 LlamaIndex 文本节点列表。
    embeddings: OpenAIEmbedding,  # 声明用于生成文本及查询向量的嵌入模型。
    index_path: Path,  # 声明 FAISS 索引的本地持久化目录。
) -> VectorStoreIndex:  # 声明返回已加载或新建的 LlamaIndex 向量索引。
    """加载完整的本地 FAISS 索引；不存在时根据节点创建并保存新索引。

    Args:
        nodes (list[TextNode]): 需要写入新索引的切分文本节点。
        embeddings (OpenAIEmbedding): 用于加载索引和编码节点及查询的嵌入模型。
        index_path (Path): FAISS 索引及其 LlamaIndex 元数据的本地目录。

    Returns:
        VectorStoreIndex: 已从磁盘加载或新建并保存的 LlamaIndex 向量索引。

    Raises:
        ValueError: 索引不存在且没有任何文本节点可用于构建新索引时抛出。
    """
    if _is_faiss_index_available(index_path):  # 调用辅助函数检查持久化目录中是否有完整的索引文件。
        vector_store = FaissVectorStore.from_persist_dir(str(index_path))  # 调用 from_persist_dir 从指定目录读取 FAISS 二进制向量索引。
        storage_context = StorageContext.from_defaults(  # 调用 from_defaults 构造并载入节点库、索引结构和 FAISS 向量库上下文。
            vector_store=vector_store,  # 传入从磁盘载入的 FAISS 向量存储。
            persist_dir=str(index_path),  # 传入保存节点文档库和索引结构元数据的目录。
        )  # 完成持久化上下文加载。
        return load_index_from_storage(storage_context, embed_model=embeddings)  # 调用 load_index_from_storage 恢复索引并绑定查询嵌入模型。

    if not nodes:  # 检查当前没有持久化索引时是否仍有节点可用于构建。
        raise ValueError("索引不存在且没有可用于构建索引的文本节点。")  # 明确报告无法构建空索引的原因。

    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)  # 调用 IndexFlatL2 创建指定维数的精确 L2 距离索引。
    vector_store = FaissVectorStore(faiss_index=faiss_index)  # 调用 FaissVectorStore 将底层 FAISS 索引接入 LlamaIndex。
    storage_context = StorageContext.from_defaults(vector_store=vector_store)  # 调用 from_defaults 创建包含新 FAISS 向量库的存储上下文。
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embeddings)  # 调用 VectorStoreIndex 为文本节点生成嵌入并写入 FAISS。
    storage_context.persist(persist_dir=str(index_path))  # 调用 persist 将 FAISS 向量文件、节点库和索引结构写入独立目录。
    print(f"FAISS 索引构建并保存完成：{index_path}（共 {faiss_index.ntotal} 条向量）")  # 调用 print 显示索引目录及底层 FAISS 向量数。
    return index  # 返回刚创建并持久化的向量索引。


def retrieve(index: VectorStoreIndex, query: str, top_k: int = TOP_K) -> list[TextNode]:  # 声明向量检索函数的参数和返回类型。
    """从 FAISS 向量索引检索相关节点，并过滤低于相似度阈值的结果。

    Args:
        index (VectorStoreIndex): 用于执行向量相似度搜索的 LlamaIndex 索引。
        query (str): 用户提出的检索问题。
        top_k (int): 最多获取的候选节点数量，默认使用配置中的 TOP_K。

    Returns:
        list[TextNode]: 达到配置相似度阈值的节点，按检索相关性排序。

    Raises:
        RuntimeError: 检索器没有为 FAISS 搜索结果返回距离分数时抛出。
    """
    retriever = index.as_retriever(similarity_top_k=top_k)  # 调用 as_retriever 创建至多返回 top_k 个候选项的 LlamaIndex 检索器。
    scored_nodes = retriever.retrieve(query)  # 调用 retrieve 将 query 向量化并获取候选节点及 FAISS L2 距离。
    retrieved_nodes: list[TextNode] = []  # 创建列表，用于保留达到相似度采纳阈值的节点。

    for position, scored_node in enumerate(scored_nodes, start=1):  # 调用 enumerate 按检索顺序遍历带分数的节点。
        distance = scored_node.score  # 读取 FAISS 返回的 L2 距离分数。
        if distance is None:  # 检查检索结果是否缺少用于阈值判断的距离分数。
            raise RuntimeError("FAISS 检索结果缺少距离分数，无法计算相似度。")  # 明确报告缺失分数，避免静默采用结果。

        similarity = 1.0 / (1.0 + distance)  # 将非负 L2 距离转换为与原流程一致的 0 到 1 相似度。
        print(f"检索结果 {position}：相似度 {similarity:.4f}，来源 {scored_node.node.metadata.get('source', '未知来源')}")  # 调用 print 显示排序、转换后的相似度和来源元数据。
        if similarity >= SIMILARITY_THRESHOLD:  # 判断当前候选是否达到配置的相似度下限。
            retrieved_nodes.append(scored_node.node)  # 调用 list.append 将达标的 TextNode 加入最终结果。

    if not retrieved_nodes:  # 检查是否没有候选节点达到采纳阈值。
        print(f"没有找到相似度达到阈值 {SIMILARITY_THRESHOLD} 的文本块。")  # 调用 print 显示无可采纳检索结果的原因。
    return retrieved_nodes  # 返回按检索相关性排序且已过滤的节点列表。
