# call_glm4_flash.py
"""
调用 智谱 GLM-4-Flash 做 RAG 受控生成
- 输入: query + final_docs（已过 混合检索 / Rerank / 时间消解 / policy过滤）
- 输出: 严格带 [DOC_ID@DATE] 引文的文本（也可切 JSON 模式）
"""
import os
import json
import re
from openai import OpenAI

# =========================================================
# 1. 智谱客户端（OpenAI 兼容协议）
# =========================================================
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY", "ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

MODEL = "glm-4-flash"


# =========================================================
# 2. 把 final_docs 拼成「带锚点」的上下文
# =========================================================
def build_context(final_docs):
    blocks = []
    for i, d in enumerate(final_docs, 1):
        m = d.metadata
        blocks.append(
            f"[{i}. {m['doc_id']}@{m['date']} | kind={m['kind']}]\n{d.page_content}"
        )
    return "\n\n".join(blocks)


# =========================================================
# 3. System Prompt：锁死行为
# =========================================================
SYSTEM_PROMPT = """你是企业制度问答模型，规则极严：
1) 只依据【用户消息中的参考资料】回答，禁止用训练常识、禁止推测、禁止补条款。
2) 每条事实结论句尾必须带来源：[DOC_ID@DATE]，例如 [POLICY_2025_REV@2025-01-01]。
3) 资料不够就写【未覆盖】并说明“现行制度未规定”，严禁编数字/编流程。
4) 不要“可能/一般来说”，要直接、确定、可执行。
5) 操作手册、福利通知、IT公告 不是制度依据，绝对不许引用。
"""


# =========================================================
# 4. 主函数：发请求
# =========================================================
def rag_ask(query: str, final_docs: list, as_json: bool = False) -> str:
    ctx = build_context(final_docs)

    if as_json:
        user_msg = f"""问题：
{query}

参考资料（仅此 {len(final_docs)} 条生效制度）：
{ctx}

【严格要求】
只输出一个 JSON，不要写任何解释，不要加 ```json 标记。
格式必须完全如下：
{{
  "answer": "一句话结论，句尾带 [DOC_ID@DATE]",
  "citations": [
    {{"doc_id": "POLICY_2025_REV", "date": "2025-01-01", "quote": "原文摘录"}}
  ],
  "uncovered": "无"   // 若缺信息则写清楚缺什么
}}
"""
    else:
        user_msg = f"""问题：
{query}

参考资料（仅此 {len(final_docs)} 条生效制度，别信别的）：
{ctx}

请按下面格式输出：
【答案】
...（每句事实结论句尾带 [DOC_ID@DATE]）

【依据】
- [DOC_ID@DATE] 原文摘录...

【未覆盖】
（无则写“无”）
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,  # RAG 生成必须 0，别让模型发挥
        top_p=0.85,
        max_tokens=1024,
    )

    raw = resp.choices[0].message.content

    # ---- 如果让它输出 JSON，但模型偷偷包了 ```json ... ```，这里剥掉 ----
    if as_json:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        try:
            obj = json.loads(raw)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return raw  # 解析失败就原样返回，外层再兜底拒答

    return raw


# =========================================================
# 5. 单独跑这个文件也能看效果（不依赖前面脚本）
# =========================================================
if __name__ == "__main__":
    # 模拟前面 Rerank 之后喂进来的 final_docs（同结构）
    from langchain_core.documents import Document

    final_docs = [
        Document(
            page_content="《差旅制度2025修订版》第7条：研发岗国内出差可报高铁一等座，需在OA提‘特殊seat申请’并经部门总监审批；非研发岗仍不可报。",
            metadata={"doc_id": "POLICY_2025_REV", "date": "2025-01-01", "kind": "policy"},
        ),
        Document(
            page_content="《境外差旅补充规定》：境外出差商务舱需总裁特批；国内研发岗一等座按《2025修订版》执行、部门总监审批即可。",
            metadata={"doc_id": "POLICY_OVERSEAS", "date": "2025-02-01", "kind": "policy"},
        ),
    ]

    QUERY = "我们研发岗国内出差，能报高铁一等座吗？要不要走审批？"

    print("=" * 82)
    print("🤖 调用 GLM-4-Flash（文本+强制引文版）")
    print("=" * 82)
    print(rag_ask(QUERY, final_docs, as_json=False))

    print("\n" + "=" * 82)
    print("🤖 JSON 版（方便前端渲染成卡片）")
    print("=" * 82)
    print(rag_ask(QUERY, final_docs, as_json=True))
