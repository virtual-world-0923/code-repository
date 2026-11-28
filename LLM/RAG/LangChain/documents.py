"""提供语料读取、文档封装和递归文本切分功能。"""

from pathlib import Path  # 导入 Path，以便通过路径对象读取 UTF-8 语料文件。

from config import *  # 导入文本切分长度和重叠长度配置。
from langchain_core.documents import Document  # 导入 LangChain 文档类型，用于保存正文和来源元数据。
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入递归文本切分器，用于按优先级拆分长文档。


def load_and_split_documents(data_file: Path) -> list[Document]:
    """读取语料文件，将文档标记解析为 LangChain 文档并切分为检索块。

    Args:
        data_file (Path): UTF-8 编码的语料文件路径；文档以“【文档”作为起始标记。

    Returns:
        list[Document]: 已切分的文档列表，每个文档都保留首行来源元数据。
    """
    content = data_file.read_text(encoding="utf-8")  # 调用 Path.read_text 读取 data_file 指定的 UTF-8 文本内容。
    raw_chunks = content.split("【文档")  # 调用 str.split 按文档起始标记拆分整份语料。
    documents: list[Document] = []  # 创建列表，用于收集解析后的 LangChain 文档对象。

    for chunk in raw_chunks: 
        chunk = chunk.strip()  # 调用 str.strip 去除当前片段首尾空白字符。
        if not chunk:
            continue
        text = "【文档" + chunk  # 将拆分时去掉的文档起始标记补回正文。
        first_line = text.split("\n")[0]  # 取首行作为来源标识。
        document = Document(page_content=text, metadata={"source": first_line}) 
        documents.append(document) 

    # 设置切分器 
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，", ""],  # 设置优先切割顺序
        chunk_size=CHUNK_SIZE,  # 设置每个文本块的目标最大字符数。
        chunk_overlap=CHUNK_OVERLAP,  # 设置相邻文本块重叠字符数，以保留跨块上下文。
        length_function=len,  # 指定内置 len 计算文本长度
    ) 
    splits = text_splitter.split_documents(documents)
    return splits
