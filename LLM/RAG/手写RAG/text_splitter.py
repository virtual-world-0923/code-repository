def _split_text(text, chunk_size, overlap): 

    """
        按照字符数量切分文本

        text 是待切分的字符串
        chunk_size 是每块最大字符数，默认 500
        overlap 是相邻块重叠字符数，默认 100
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


def split_documents(documents, chunk_size, overlap):  

    """
    切分文档，并保留元数据
    
    documents 是文档字典列表
    chunk_size 和 overlap 的含义与 split_text 相同
    """ 

    results = []  # 创建空列表；用于保存所有文档切分后的 Chunk 字典

    for doc in documents:  # 使用 for 循环遍历 documents；每次取出一个文档字典并赋值给 doc

        chunks = _split_text(  # 把当前文档正文切分成字符串列表
            doc["text"],
            chunk_size,
            overlap
        ) 

        for chunk in chunks:  # 遍历当前文档的所有文本块；每次取出一个字符串并赋值给 chunk

            results.append({ # 将一个包含文本和元数据的字典追加到 results
                "text": chunk,  # 保存当前 Chunk 的正文；后续将对它进行 Embedding 和向量检索
                "source": doc["source"],  # 从原始文档字典复制 source；记录这个 Chunk 来自哪个文件
                "page": doc["page"]  # 从原始文档字典复制 page；记录这个 Chunk 来自 PDF 的哪一页，TXT 通常为 None
            })

    return results  # 返回所有文档切分后的结果；每个元素都包含 text、source、page
