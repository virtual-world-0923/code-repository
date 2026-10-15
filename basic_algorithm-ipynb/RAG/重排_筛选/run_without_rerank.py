from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from kb_data import load_kb, QUERY

TEXT, META = load_kb()

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
db = FAISS.from_texts(TEXT, emb, metadatas=META)
# 把已经建好的 FAISS 向量库 db，包装成一个“接收问题、吐出 4 条最相关文本”的标准接口 retriever。​
# 以后你只管 retriever.invoke("问题")，它自动去向量库里找 Top-4 片段回来，不用再手动写 similarity_search。
retriever = db.as_retriever(search_kwargs={"k": 4}) # 把向量库包装成“提问即返 Top-4 最相似文本”的标准检索接口

print("🟡 用户问题：", QUERY, "\n")
print("=" * 80)
print("【无 Rerank】纯向量 Top-4 ")
print("=" * 80)

hits = retriever.invoke(QUERY)
for i, d in enumerate(hits, 1):
    m = d.metadata

    print(f"\n#{i} {m['doc_id']} | {m['date']}| {m['kind']}")
    print(f"   {d.page_content}")

# import call_glm4_flash as glm4
# raw = glm4.rag_ask(QUERY,hits)
# print(raw)