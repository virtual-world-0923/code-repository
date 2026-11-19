
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
import jieba
from kb_data import load_kb, QUERY

TEXT, META = load_kb()

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vec_db = FAISS.from_texts(TEXT, emb, metadatas=META)
bm25 = BM25Okapi([jieba.lcut(t) for t in TEXT]) # 输入分词后的列表的列表。List[List[词]]

def embed_recall(q, k=4):
    return vec_db.similarity_search(q, k=k)

def bm25_recall(q, k=4):
    query_fenci = jieba.lcut(q)
    sc = bm25.get_scores(query_fenci)
    idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
    return [Document(page_content=TEXT[i], metadata=META[i]) for i in idx]

vec_hits = embed_recall(QUERY, 4)
bm_hits  = bm25_recall(QUERY, 4)

seen, union = set(), []
for d in vec_hits + bm_hits:
    if d.metadata["doc_id"] not in seen: # seen存在的意义就是查找速度快，为了去重操作
        seen.add(d.metadata["doc_id"])
        union.append(d)

print("🔀 混合并集：", [d.metadata["doc_id"] for d in union], "\n")

rerank = CrossEncoder("BAAI/bge-reranker-base") # 加载一个“交叉编码器重排序模型”，专门用来给 (问题, 文档) 这对文本打“相关度分数”。
rel = rerank.predict([[QUERY, d.page_content] for d in union])

def time_boost(m):
    # 时间冲突消解：2024后新policy +0.35，旧policy -0.30
    if m["kind"] != "政策": return 0.0
    return +1 if m["date"] >= "2024-01-01" else -0

scored = sorted([(float(s)+time_boost(d.metadata), d) for d,s in zip(union,rel)],
                key=lambda x: -x[0])

final_docs = [(s,d) for s,d in scored if d.metadata["kind"]=="政策"][:3]

for score,doc in final_docs:
    print(f"得分：{score}")
    print(doc.metadata["doc_id"])
    print(doc.metadata["date"])
    print(doc.metadata["kind"])
    print(doc.page_content)
    print("\n\n")
