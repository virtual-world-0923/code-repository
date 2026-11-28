"""提供语料读取、来源标注和递归字符切块功能。"""

from pathlib import Path  # 导入 Path，以便通过路径对象读取 UTF-8 语料文件。

from llama_index.core.schema import TextNode  # 导入 TextNode，用 LlamaIndex 节点保存文本和来源元数据。

from .config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE  # 导入字符切块长度、重叠长度和分隔符配置。


def _merge_text_splits(text_splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:  # 声明合并分段的辅助函数和类型。
    """将较短的文本片段合并为带有指定重叠长度的文本块。

    Args:
        text_splits (list[str]): 按选定分隔符拆分后、需要依次合并的文本片段。
        chunk_size (int): 每个输出文本块的最大字符数。
        chunk_overlap (int): 相邻输出文本块保留的最大重叠字符数。

    Returns:
        list[str]: 去除首尾空白且长度不超过 chunk_size 的文本块列表。
    """
    merged_chunks: list[str] = []  # 创建列表，用于收集合并完成的文本块。
    current_splits: list[str] = []  # 创建列表，暂存当前正在拼接的片段。
    current_length = 0  # 记录当前片段组合的字符总数。

    for text_split in text_splits:  # 遍历分隔符切出的片段，按顺序构造文本块。
        if current_splits and current_length + len(text_split) > chunk_size:  # 调用 len 计算新片段长度并判断拼接后是否超限。
            merged_text = "".join(current_splits).strip()  # 调用 str.join 拼接当前片段并用 str.strip 去除首尾空白。
            if merged_text:  # 检查清理后的文本块是否包含有效内容。
                merged_chunks.append(merged_text)  # 调用 list.append 保存已完成的文本块。

            while current_splits and (  # 保留必要的尾部上下文，直到满足重叠长度和下个片段的空间要求。
                current_length > chunk_overlap  # 判断当前保留内容是否仍超过配置的重叠长度。
                or (current_length + len(text_split) > chunk_size and current_length > 0)  # 调用 len 检查下一个片段是否仍放不进当前组合。
            ):  # 结束重叠片段清理条件。
                removed_split = current_splits.pop(0)  # 调用 list.pop 移除最早片段，为重叠内容腾出空间。
                current_length -= len(removed_split)  # 调用 len 更新剩余片段的字符总数。

        current_splits.append(text_split)  # 调用 list.append 将当前片段放入正在构建的文本块。
        current_length += len(text_split)  # 调用 len 更新当前文本块的字符总数。

    if current_splits:  # 检查循环结束后是否还有未输出的片段。
        merged_text = "".join(current_splits).strip()  # 调用 str.join 拼接剩余片段并用 str.strip 去除首尾空白。
        if merged_text:  # 检查最后一个文本块是否包含有效内容。
            merged_chunks.append(merged_text)  # 调用 list.append 保存最后一个有效文本块。

    return merged_chunks  # 返回完成合并的文本块列表。


def _split_text_recursively(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:  # 声明递归切分函数及其类型。
    """按优先级递归切分超长文本，并尽可能保留原有分隔符边界。

    Args:
        text (str): 需要切分的原始文本。
        separators (list[str]): 按优先级排列的分隔符；空字符串表示逐字符切分。
        chunk_size (int): 每个文本块的最大字符数。
        chunk_overlap (int): 相邻文本块之间保留的重叠字符数。

    Returns:
        list[str]: 按分隔符优先级切分并合并后的文本块列表。
    """
    if len(text) <= chunk_size:  # 调用 len 判断文本是否已经不超过目标字符数。
        return [text]  # 原文本已符合长度要求，直接作为单个文本块返回。

    separator_position = len(separators) - 1  # 默认选用分隔符列表中的最后一项作为最低优先级分隔符。
    for position, separator in enumerate(separators):  # 调用 enumerate 按顺序检查每个候选分隔符。
        if separator == "" or separator in text:  # 选择空分隔符或当前文本实际包含的第一个分隔符。
            separator_position = position  # 记录当前命中的分隔符位置。
            break  # 找到最高优先级的有效分隔符后停止检查。

    separator = separators[separator_position]  # 读取本轮递归使用的分隔符。
    remaining_separators = separators[separator_position + 1 :]  # 保留优先级更低的分隔符供超长片段递归使用。
    if separator == "":  # 判断是否已降级到逐字符切分。
        text_splits = list(text)  # 调用 list 将文本展开为单字符片段。
    else:  # 对普通分隔符采用保留边界的切分方式。
        raw_splits = text.split(separator)  # 调用 str.split 按当前分隔符拆分文本。
        text_splits = [  # 创建片段列表，并把分隔符前置到后续片段以保留原文边界。
            (piece if position == 0 else separator + piece)  # 第一段不添加分隔符，其余段在开头补回分隔符。
            for position, piece in enumerate(raw_splits)  # 调用 enumerate 获取片段位置以恢复分隔符。
            if piece  # 跳过空片段，避免生成无内容节点。
        ]  # 完成保留分隔符的片段列表构造。

    final_chunks: list[str] = []  # 创建列表，用于收集本层递归生成的文本块。
    merge_candidates: list[str] = []  # 创建列表，暂存长度合适、可直接合并的片段。
    for text_split in text_splits:  # 遍历当前层级分隔符生成的片段。
        if len(text_split) <= chunk_size:  # 调用 len 检查片段是否能直接参与当前层的合并。
            merge_candidates.append(text_split)  # 调用 list.append 暂存未超长片段以便批量合并。
            continue  # 处理下一个片段，避免进入超长片段递归分支。

        if merge_candidates:  # 检查是否有尚未合并的普通片段。
            final_chunks.extend(_merge_text_splits(merge_candidates, chunk_size, chunk_overlap))  # 调用辅助函数合并普通片段并写入结果列表。
            merge_candidates = []  # 清空暂存列表，避免跨越超长片段合并文本。

        if remaining_separators:  # 检查是否还存在更低优先级的分隔符。
            final_chunks.extend(  # 将递归切分产生的子块加入本层结果。
                _split_text_recursively(text_split, remaining_separators, chunk_size, chunk_overlap)  # 递归调用本函数并传入剩余分隔符及长度设置。
            )  # 完成递归子块的结果追加。
        else:  # 没有更多分隔符时保留该片段，避免丢失文本内容。
            final_chunks.append(text_split)  # 调用 list.append 收集无法进一步拆分的文本片段。

    if merge_candidates:  # 检查循环结束后是否还留有待合并的普通片段。
        final_chunks.extend(_merge_text_splits(merge_candidates, chunk_size, chunk_overlap))  # 调用辅助函数合并最后一组片段并追加到结果列表。

    return final_chunks  # 返回本层递归切分并合并后的文本块。


def load_and_split_documents(data_file: Path) -> list[TextNode]:  # 声明语料加载函数的参数和返回类型。
    """读取语料，将“【文档”分隔的内容转换为带来源的 LlamaIndex 节点并切块。

    Args:
        data_file (Path): UTF-8 编码的语料文件路径；各文档以“【文档”作为起始标记。

    Returns:
        list[TextNode]: 已切分的文本节点列表，每个节点都保留所属文档的首行来源。
    """
    content = data_file.read_text(encoding="utf-8")  # 调用 Path.read_text 以 UTF-8 编码读取指定语料文件。
    raw_documents = content.split("【文档")  # 调用 str.split 按文档起始标记拆分整份语料。
    nodes: list[TextNode] = []  # 创建列表，用于收集已标注来源的 LlamaIndex 文本节点。

    for raw_document in raw_documents:  # 按原始顺序逐个处理语料中的文档片段。
        cleaned_document = raw_document.strip()  # 调用 str.strip 去除当前片段首尾空白字符。
        if not cleaned_document:  # 检查当前片段是否为空。
            continue  # 跳过空片段，不为其创建检索节点。

        document_text = "【文档" + cleaned_document  # 将拆分时去掉的文档起始标记补回正文。
        source = document_text.split("\n", maxsplit=1)[0]  # 调用 str.split 取得首行作为该文档的来源标识。
        chunks = _split_text_recursively(  # 调用递归切分函数按配置的字符数、重叠长度和分隔符生成文本块。
            document_text,  # 传入当前完整文档正文。
            CHUNK_SEPARATORS,  # 传入从段落边界到单字符的分隔符优先级列表。
            CHUNK_SIZE,  # 传入每个文本块允许的最大字符数。
            CHUNK_OVERLAP,  # 传入相邻文本块之间保留的重叠字符数。
        )  # 完成当前文档的递归字符切分调用。
        for chunk in chunks:  # 遍历当前文档生成的各个文本块。
            nodes.append(TextNode(text=chunk, metadata={"source": source}))  # 调用 TextNode 创建携带正文与来源元数据的 LlamaIndex 节点并加入列表。

    return nodes  # 返回所有文档转换得到的文本节点。
