# %%
import json
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

DATA_FILE = "./rag_data.txt"
INDEX_DIR   = "./faiss_index/"
INDEX_FILE  = INDEX_DIR + "faiss.index"
DOCS_FILE   = INDEX_DIR + "documents.pkl"
META_FILE   = INDEX_DIR + "meta.json"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 10
CHUNK_OVERLAP = 5
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3   # 余弦相似度阈值（归一化后范围 -1~1）
LLM_API_KEY_ENV = "ZHIPU_API_KEY"
LLM_MODEL = "glm-4-flash"
DEVICE = "cuda"   # 没 GPU改 "cpu"

# %%    
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
        if os.path.exists(tmp_path): # 如果临时目录存在
            os.unlink(tmp_path) # 删除
        raise


# %%
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


# %%
def save_vector_store(index, documents, model_name: str):
    """把 FAISS 索引 + 文档 + 元信息写入磁盘"""
    print(f"\n💾 正在保存向量库到 {INDEX_DIR}...")

    # 1. 保存 FAISS 索引（安全写法）
    _faiss_write_safe(index, str(INDEX_FILE))
    print(f"   ✅ 索引: {INDEX_FILE}")

    # 2. 保存文档（含 page_content + metadata）
    with open(str(DOCS_FILE), "wb") as f:
        pickle.dump(documents, f)
    print(f"   ✅ 文档: {DOCS_FILE}")

    # 3. 保存元信息
    meta = {
        "model_name":    model_name,
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_documents": len(documents),
        "saved_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(str(META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 元信息: {META_FILE}")

    print("✅ 向量库已保存")


# %%
def load_vector_store():
    """从磁盘加载 FAISS 索引 + 文档，校验一致性"""
    print(f"\n📂 正在加载已有向量库: {INDEX_DIR}")

    # 1. 加载 FAISS 索引（安全读法）
    index = _faiss_read_safe(str(INDEX_FILE))

    # 2. 加载文档
    with open(str(DOCS_FILE), "rb") as f:
        documents = pickle.load(f)

    # 3. 加载并校验元信息
    with open(str(META_FILE), "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"   📋 模型: {meta['model_name']}")
    print(f"   📋 文档数: {meta['num_documents']}")
    print(f"   📋 保存时间: {meta['saved_at']}")
    print(f"   📋 chunk_size={meta['chunk_size']}, overlap={meta['chunk_overlap']}")

    # 校验：embedding 模型是否和当前配置一致
    if meta["model_name"] != EMBED_MODEL_NAME:
        print(f"   ⚠️ 模型不匹配！已存={meta['model_name']}, 当前={EMBED_MODEL_NAME}")
        print(f"   → 将重建向量库")
        return None, None

    # 校验：chunk 参数是否一致
    if meta["chunk_size"] != CHUNK_SIZE or meta["chunk_overlap"] != CHUNK_OVERLAP:
        print(f"   ⚠️ 切分参数不匹配！已存=({meta['chunk_size']},{meta['chunk_overlap']})")
        print(f"   → 将重建向量库")
        return None, None

    print(f"   FAISS 索引类型: {type(index).__name__}")
    print("✅ 向量库加载完成")
    return index, documents


# %%
def delete_vector_store():
    """删除已有向量库，强制下次重建"""
    shutil.rmtree(str(INDEX_DIR))
    print(f"🗑️ 已删除旧向量库: {INDEX_DIR}")


# %%
def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    递归字符切分器（复刻 LangChain RecursiveCharacterTextSplitter 逻辑）。

    优先级：先按 "\\n" 切 → 还大就按 "。" 切 → ... → 最后按字符硬切。
    """
    separators = ["\n", "。", "，", ""]

    def _split(text: str, seps: list) -> list:
        """递归切分：用当前最高优先级分隔符切，超长的段降级到下一个分隔符"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if not seps:
            # 兜底：按允许最大字符硬切
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        sep = seps[0]
        if sep == "":
            # 空字符串 = 按字符切
            parts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
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
        if merged and len(merged[-1]) + len(chunk) <= chunk_size:
            merged[-1] = merged[-1] + chunk
        else:
            merged.append(chunk)

    return merged

# %%
def load_and_split_data(file_path: Path) -> list:
    """读取资料文件，按【文档N】粗切，再细切到 chunk_size，返回 List[dict]"""
    print(f"📂 正在读取资料文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按【文档N】粗切
    raw_chunks = content.split("【文档")
    coarse_docs = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        text = "【文档" + chunk.strip()
        first_line = text.split("\n")[0]
        coarse_docs.append({
            "page_content": text,
            "metadata": {"source": first_line[:30]}
        })

    print(f"   粗切完成，共 {len(coarse_docs)} 个大块")
    for i, doc in enumerate(coarse_docs, 1):
        print(f"   [{i}] {doc['metadata']['source']}  ({len(doc['page_content'])} 字)")

    # 细切（递归字符切分）
    print(f"\n🔧 正在细切（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）...")
    final_docs = []
    for doc in coarse_docs:
        text = doc["page_content"]
        source = doc["metadata"]["source"]
        sub_chunks = recursive_split(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for j, sub in enumerate(sub_chunks):
            final_docs.append({
                "page_content": sub,
                "metadata": {"source": source, "chunk": j+1, "total_chunks": len(sub_chunks)}
            })

    print(f"   细切完成，共 {len(final_docs)} 个块")
    return final_docs

# %%
def build_vector_store(documents: list) -> faiss.Index:
    """把文档列表编码成向量，构建 FAISS 索引（IndexFlatIP + 归一化 = 余弦相似度）"""
    print(f"\n🔧 正在构建向量库...")

    # 提取文本
    texts = [doc["page_content"] for doc in documents]
    print(f"   正在编码 {len(texts)} 段文本...")

    # 加载 embedding 模型
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    # 编码 + 归一化（L2 归一化后，内积 = 余弦相似度）
    vectors = embed_model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    print(f"   向量维度: {vectors.shape[1]}")

    # 创建 FAISS 索引（IndexFlatIP = 内积，配合归一化 = 余弦相似度）
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    print(f"   FAISS 索引类型: {type(index).__name__}（内积/余弦）")
    print("✅ 向量库构建完成")
    return index

# %%
def retrieve(index, documents: list, query: str,
             embed_model, top_k: int = TOP_K) -> list:
    """用 FAISS 检索最相关的文档"""
    # 1. 编码 query
    query_vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # 2. FAISS 搜索
    '''
        查询向量 query_vec，在 FAISS 索引里找最相似的 top_k 个向量，返回它们的“相似度分数”和“在索引中的位置编号
        query_vec：查询向量,形状：(1, dim) 的二维 NumPy 数组。FAISS 支持 一次查多个问题
        top_k：返回最相似的向量数量
        scores：相似度分数矩阵，形状：(1, top_k) 的 NumPy 数组，1完全一样，0完全不相关，负数相反。
        indices：向量在索引中的位置编号，形状：(1, top_k) 的 NumPy 数组
    '''
    scores, indices = index.search(query_vec, top_k)

    print(f"\n{'='*60}")
    print(f"🔍 检索结果（Top {top_k}）：")
    print(f"{'='*60}")

    retrieved = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        cosine = float(score)
        similarity_pct = (cosine + 1) / 2  # 映射到 0~1
        marker = "✅ 采纳" if cosine >= SIMILARITY_THRESHOLD else "⚠️ 低于阈值"
        doc = documents[idx]
        print(f"   [{i}] {marker}  余弦={cosine:.3f} ({similarity_pct:.1%})")
        print(f"       来源: {doc['metadata']['source']} (块{doc['metadata'].get('chunk',1)})")
        print(f"       内容: {doc['page_content']}")
        if cosine >= SIMILARITY_THRESHOLD:
            retrieved.append(doc)

    if not retrieved:
        print(f"\n   ❌ 没有找到超过阈值({SIMILARITY_THRESHOLD})的片段")
    return retrieved

# %%
def build_prompt(query: str, retrieved_docs: list) -> str:
    """构造 RAG Prompt，并打印完整内容"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        src = doc["metadata"]["source"]
        context_parts.append(f"--- 资料{i}（{src}）---\n{doc['page_content']}")
    context = "\n\n".join(context_parts)

    prompt = f"""【系统指令】
你是一个严谨的问答助手。请严格基于下面提供的资料回答问题。

规则：
1. 只使用资料中的信息回答，不要编造或补充资料外的内容
2. 如果资料中没有明确答案，必须回答"不知道"
3. 回答要简洁，直接给结论，必要时引用资料原文
4. 如果多份资料有矛盾，指出矛盾并说明各资料内容
5. 回答前，先列出资料中的相关条件，再逐一比对用户情况，最后给出结论

【参考资料】
{context}

【问题】
{query}

【回答】"""

    print(f"\n{'='*60}")
    print(f"📝 准备传给 LLM 的完整 Prompt（{len(prompt)} 字符）：")
    print(f"{'='*60}")
    print(prompt)
    print(f"{'='*60}")
    return prompt

# %%
def mode_rag_with_llm(query: str, retrieved_docs: list) -> str:
    """拼接 Prompt 后调用智谱 GLM-4-Flash 生成回答"""
    print(f"\n{'='*60}")
    print(f"🤖 模式2：RAG + LLM（智谱 {LLM_MODEL}）")
    print(f"{'='*60}")

    if not retrieved_docs:
        msg = "不知道。资料库中没有找到与您问题相关的信息。"
        print(f"\n   → {msg}")
        return msg

    # 打印检索到的原文
    print(f"\n   📄 检索到的原文片段：")
    for i, doc in enumerate(retrieved_docs, 1):
        src = doc["metadata"]["source"]
        print(f"\n   ┌─ 片段{i}（{src}）")
        for line in doc["page_content"].split("\n"):
            print(f"   │  {line}")
        print(f"   └─")

    # 拼接 Prompt
    prompt = build_prompt(query, retrieved_docs)

    # 检查 API Key
    api_key = os.getenv(LLM_API_KEY_ENV)
    if not api_key:
        print(f"\n   ❌ 环境变量 {LLM_API_KEY_ENV} 未设置")
        print(f"   Windows CMD:  set {LLM_API_KEY_ENV}=你的key")
        print(f"   PowerShell:   $env:{LLM_API_KEY_ENV}='你的key'")
        return f"API Key 未设置（环境变量 {LLM_API_KEY_ENV}）"

    print(f"\n   🔑 API Key 前5位: {api_key[:5]}...")
    print(f"   ⏳ 正在调用 {LLM_MODEL} 生成回答...")
    sys.stdout.flush()

    # 调用智谱 API（zai-sdk 官方写法）
    try:
        from zai import ZhipuAiClient
    except ImportError:
        print("   ❌ 缺少依赖，请安装: pip install zai-sdk")
        return "依赖缺失：请运行 pip install zai-sdk"

    client = ZhipuAiClient(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个严谨的问答助手。严格基于用户提供的资料回答问题，不编造。资料中没有的答案就说'不知道'。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"调用 LLM 出错: {type(e).__name__}: {e}"
        print(f"   ❌ {answer}")
        import traceback
        traceback.print_exc()
        return answer

    print(f"\n   ✅ LLM 返回回答:")
    print(f"   {'─'*50}")
    for line in answer.split("\n"):
        print(f"   {line}")
    print(f"   {'─'*50}")

    return answer

# %%
delete_vector_store()

# %%
load_vector_store()

# %%
documents = load_and_split_data(DATA_FILE)
index = build_vector_store(documents)
save_vector_store(index, documents, EMBED_MODEL_NAME)

# %%
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

# %%
query = "我入职二年，可以休几天年假？"

# %%
docs_retrieved = retrieve(index, documents, query, embed_model)

# %%
mode_rag_with_llm(query, docs_retrieved)


