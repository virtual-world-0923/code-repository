"""集中管理 LangChain RAG 示例使用的路径和模型参数。"""

from pathlib import Path

# DATA_FILE = Path(os.getcwd()) / "rag_data.txt"  # 使用当前工作目录下的 rag_data.txt 作为语料文件。
current_file_path = Path(__file__).resolve()
DATA_FILE = current_file_path.parent.parent / "data/1.txt"
TOP_K = 3  # 设置每次检索返回的候选文档数量。
SIMILARITY_THRESHOLD = 0.6  # 设置采纳检索结果的最低相似度阈值。
LLM_API_KEY_ENV = "OPENAI_KEY"  # 指定智谱 API Key 所在的环境变量名称。
LLM_MODEL = "glm-4-flash"  # 指定生成回答时调用的智谱模型名称。
FAISS_PATH = current_file_path.parent / "faiss_zh_index"  # 指定本地 FAISS 索引的保存目录。
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 指定用于文本向量化的 Hugging Face 模型。
DEVICE = "cuda"  # 无可用 GPU 时可改为 cpu。
CHUNK_SIZE = 50  # 设置文本切块的目标最大字符数。
CHUNK_OVERLAP = 10  # 设置相邻文本块之间保留的重叠字符数。
DEFAULT_QUERY = "我入职两年，可以休几天年假？"  # 设置示例运行时使用的默认问题。
