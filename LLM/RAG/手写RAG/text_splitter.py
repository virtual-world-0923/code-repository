def _split_text(text: str, chunk_size: int, overlap: int) -> list:
    """按照字符数量切分文本

    Args:
        text: 是待切分的字符串
        chunk_size: 是每块最大字符数
        overlap: 是相邻块重叠字符数

    Returns:
    ```
        严格按照最大字符数切分好的 list
        例如：
            原始文本：ABCDEFGHIJKLMN
        假设：
            CHUNK_SIZE = 6
            CHUNK_OVERLAP = 2
        切分结果：
            [
                "ABCDEF",
                "EFGHIJ",
                "IJKLMN",
            ]
    ```
    """

    chunks = []  # 创建空列表；用于依次保存切分得到的文本块（Chunk）

    start = 0  # 设置当前文本块的起始字符下标；Python 字符串下标从 0 开始

    while start < len(text):  # 调用内置函数 len(text) 获取字符串字符数；只要起始位置还没到文本末尾，就继续切分
        end = start + chunk_size  # 计算当前文本块的结束下标；例如 start=0、chunk_size=500，则 end=500；切片不包含 end 位置

        chunk = text[start:end]  # 使用 Python 字符串切片 [start:end] 取出当前文本块；最多取得 chunk_size 个字符

        if chunk.strip():  # 去掉首尾空白字符；判断切分结果是否不是纯空白内容
            chunks.append(chunk)  # 将当前非空文本块追加到 chunks 列表

        start += chunk_size - overlap  # 更新下一块起始位置；例如 500-100=400，下一块从 400 开始，因此与上一块重叠 100 个字符

    return chunks  # 返回切分后的文本块列表；每个元素都是一个字符串


def split_documents(documents: list, chunk_size: int, overlap: int) -> list:
    """完全按照最大字符切分

    Args:
        documents: 准备切分的文档
        chunk_size: 每个文本块的最大字符数
        overlap: 相邻文本块之间重复的字符数

    Returns:
    ```
        一条记录是切分好的文本块，同时有这文本块的原信息

        [
            {"text": 数据块, "source": 属于哪个文件, "page": 页码},
            ...
        ]
    ```
    """

    results = []  # 创建空列表；用于保存所有文档切分后的 Chunk 字典

    # 使用 for 循环遍历 documents；每次取出一个文档字典并赋值给 doc
    for doc in documents:
        chunks = _split_text(  # 把当前文档正文切分成字符串列表
            doc["text"], chunk_size, overlap
        )

        for chunk in chunks:  # 遍历当前文档的所有文本块；每次取出一个字符串并赋值给 chunk
            results.append(
                {  # 将一个包含文本和元数据的字典追加到 results
                    "text": chunk,  # 保存当前 Chunk 的正文；后续将对它进行 Embedding 和向量检索
                    "source": doc["source"],  # 从原始文档字典复制 source；记录这个 Chunk 来自哪个文件
                    "page": doc["page"],  # 从原始文档字典复制 page；记录这个 Chunk 来自 PDF 的哪一页，TXT 通常为 None
                }
            )

    return results  # 返回所有文档切分后的结果；每个元素都包含 text、source、page


def _charact_split(text: str, separators: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    ```
    递归字符切分器
        优先级：
            按行切分
            按句号切分
            按逗号切分
            按单个字符切分
    每次切分完会查看当前切分数据块是否大于 chunk_size，如果大于，继续切分
    最后会合并相邻的数据块，如果合并完小于 chunk_size，否则不合并
    ```
    Args:
        text: 准备切分的数据
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻文本块之间重复的字符数

    Returns:
    ```
        按照字符优先级逐个切分数据，每一个数据块不大于 chunk_size
        [
            数据块,
            数据块,
            ...
        ]
    ```
    """

    def _split(text: str, seps: list) -> list:
        """递归切分：用当前最高优先级分隔符切，超长的段降级到下一个分隔符"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if not seps:
            # 兜底：按允许最大字符硬切
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = seps[0]
        if sep == "":
            # 空字符串 = 按字符切
            parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            return [p for p in parts if p.strip()]

        # 用当前分隔符切
        raw_parts = text.split(sep)
        # 拼接回带分隔符的片段（保留语义）
        parts = []
        for i, p in enumerate(raw_parts):
            if i < len(raw_parts) - 1:
                p = p + sep
            if p.strip():
                parts.append(p)

        # 检查每个片段是否超长
        result = []
        for p in parts:
            if len(p) > chunk_size:
                # 超长 → 降级用下一个分隔符
                result.extend(_split(p, seps[1:]))
            else:
                result.append(p)
        return result

    # 递归切分
    chunks = _split(text, separators)

    # 合并过短的相邻块
    merged = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
        elif len(merged[-1]) + len(chunk) <= chunk_size:
            merged[-1] = merged[-1] + chunk
        else:
            overlap_str = merged[-1][-abs(chunk_overlap) :]  # 从后往前切chunk_overlap个字符
            merged.append(overlap_str + chunk)

    return merged


def _paragraph_split(text: str, paragraph_basis: str) -> list:
    """按段落粗切

    Args:
        text: 被切分数据
        paragraph_basis: 段落字符

    Returns:
        _description_
    """

    paragraphs = text.split(paragraph_basis)
    coarse_docs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        text = paragraph_basis + paragraph
        coarse_docs.append(text)

    print(f"   粗切完成，共 {len(coarse_docs)} 个大块")
    return coarse_docs


def split_data(documents: list, chunk_size: int, overlap: int) -> list:
    """先根据段落分，每个段落在根据符号分

    Args:
        documents: 准备切分的文档
        chunk_size: 每个文本块的最大字符数
        overlap: 相邻文本块之间重复的字符数

    Returns:
    ```
        切分好的文档列表
        [
            {"text": 切分好的文本,"source": 文件来源,"page": 页码}
            ...
        ]
    ```
    """

    separators = ["】", "\n", "。", "，", ""]
    final_docs = []
    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        page = doc["page"]
        sentence_docs = _charact_split(text, separators, chunk_size, overlap)
        for s in sentence_docs:
            final_docs.append(
                {
                    "text": s,
                    "source": source,
                    "page": page,
                }
            )

    return final_docs
