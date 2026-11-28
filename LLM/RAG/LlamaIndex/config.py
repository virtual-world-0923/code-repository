"""集中管理 LlamaIndex RAG 示例的路径和模型参数。"""

import os  # 导入 os，以便从环境变量读取 OpenAI API Key。
from pathlib import Path  # 导入 Path，用路径对象定位语料和索引目录。

current_file_path = Path(__file__).resolve()  # 调用 Path.resolve 获取当前配置文件的绝对路径。
DATA_FILE = current_file_path.parent.parent / "data" / "1.txt"  # 拼接现有 RAG 语料文件路径，不复制语料。
TOP_K = 3  # 设置每次检索返回的候选文档数量。
SIMILARITY_THRESHOLD = 0.6  # 设置采纳检索结果的最低相似度阈值。
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # 指定 OpenAI API Key 的环境变量名称。
OPENAI_LLM_MODEL = "gpt-4o-mini"  # 设置用于生成回答的 OpenAI 轻量模型。
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # 设置用于文本向量化的 OpenAI 嵌入模型。
EMBEDDING_DIMENSION = 1536  # 设置嵌入维度，与 text-embedding-3-small 的默认输出维度一致。
FAISS_PATH = current_file_path.parent / "faiss_llamaindex_index"  # 为 LlamaIndex 索引指定独立持久化目录。
CHUNK_SIZE = 50  # 设置文本切块的目标最大字符数。
CHUNK_OVERLAP = 10  # 设置相邻文本块之间保留的重叠字符数。
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "，", ""]  # 按段落、换行、中文标点到单字符的顺序递归切分。
DEFAULT_QUERY = "我入职两年，可以休几天年假？"  # 设置示例运行时使用的默认问题。


def get_openai_api_key() -> str:  # 声明读取 OpenAI API Key 的辅助函数及其返回类型。
    """读取 OpenAI API Key 环境变量并在未配置时明确报错。

    Returns:
        str: 用于 OpenAI 嵌入和文本生成请求的 API Key。

    Raises:
        RuntimeError: 指定环境变量不存在或为空时抛出。
    """
    api_key = os.getenv(OPENAI_API_KEY_ENV)  # 调用 os.getenv 按配置名称读取 OpenAI API Key 环境变量。
    if not api_key:  # 检查环境变量是否未设置或仅包含空值。
        raise RuntimeError(f"请先设置环境变量 {OPENAI_API_KEY_ENV}。")  # 明确指出调用 OpenAI 服务前必须配置的变量名称。
    return api_key  # 返回已配置的 API Key，供 LlamaIndex OpenAI 集成使用。
