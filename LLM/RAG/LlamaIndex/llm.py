"""提供基于 LlamaIndex OpenAI 集成的检索增强回答生成功能。"""

from collections.abc import Sequence  # 导入 Sequence，用于声明只读文本节点序列参数。

from llama_index.core.schema import TextNode  # 导入 TextNode，声明生成回答所使用的检索资料类型。
from llama_index.llms.openai import OpenAI  # 导入 LlamaIndex OpenAI 封装，用于生成模型回答。

from .config import OPENAI_LLM_MODEL, get_openai_api_key  # 导入 OpenAI 模型名称及 API Key 校验函数。


def build_prompt(query: str, retrieved_nodes: Sequence[TextNode]) -> str:  # 声明提示词构造函数的参数和返回类型。
    """将问题及检索资料组合为限制回答范围的提示词。

    Args:
        query (str): 用户提出的问题。
        retrieved_nodes (Sequence[TextNode]): 用作回答依据的检索文本节点序列。

    Returns:
        str: 包含回答规则、参考资料和用户问题的完整提示词。
    """
    context_parts: list[str] = []  # 创建列表，用于按检索顺序收集带来源标记的资料文本。
    for position, node in enumerate(retrieved_nodes, start=1):  # 调用 enumerate 为每个检索节点生成从 1 开始的资料序号。
        source = node.metadata.get("source", "未知来源")  # 调用 dict.get 读取节点来源元数据并为缺失来源提供明确标签。
        context_parts.append(f"--- 资料{position}（{source}）---\n{node.text}")  # 调用 list.append 将来源和节点正文加入参考资料列表。
    context = "\n\n".join(context_parts)  # 调用 str.join 以空行分隔每份参考资料。
    prompt_lines = [  # 创建逐行提示词，保持内容本身不被行尾注释污染。
        "【系统指令】",  # 添加系统指令标题。
        "你是一个严谨的问答助手。请严格基于下面提供的资料回答问题。",  # 约束模型只根据当前检索资料回答。
        "",  # 为规则标题与规则正文保留空行。
        "规则：",  # 添加回答规则标题。
        "1. 只使用资料中的信息回答，不要编造或补充资料外的内容",  # 禁止引入检索资料之外的信息。
        "2. 如果资料中没有明确答案，必须回答“不知道”",  # 指定资料不足时的回答方式。
        "3. 回答要简洁，直接给结论，必要时引用资料原文",  # 指定回答的简洁程度和引用原则。
        "4. 如果多份资料有矛盾，指出矛盾并说明各资料内容",  # 指定多来源冲突时的处理方式。
        "5. 回答前，先列出资料中的相关条件，再逐一比对用户情况，最后给出结论",  # 指定比较资料条件与用户情况的回答步骤。
        "",  # 为规则与参考资料之间保留空行。
        "【参考资料】",  # 添加参考资料区标题。
        context,  # 插入按检索顺序组合的来源和文档正文。
        "",  # 为资料与用户问题之间保留空行。
        "【问题】",  # 添加用户问题区标题。
        query,  # 插入用户提出的问题。
        "",  # 为问题与模型回答位置保留空行。
        "【回答】",  # 指示模型从此处开始生成最终回答。
    ]  # 完成提示词各行内容构造。
    return "\n".join(prompt_lines)  # 调用 str.join 以换行符拼接提示词各行并返回完整文本。


def answer_with_llm(query: str, retrieved_nodes: Sequence[TextNode]) -> str:  # 声明回答生成函数的参数和返回类型。
    """使用检索节点构造提示词并调用 OpenAI 模型生成回答。

    Args:
        query (str): 用户提出的问题。
        retrieved_nodes (Sequence[TextNode]): 作为回答依据的检索文本节点序列。

    Returns:
        str: OpenAI 模型生成的回答；没有检索资料时返回“不知道”提示。

    Raises:
        RuntimeError: OpenAI 没有返回有效文本内容时抛出。
    """
    if not retrieved_nodes:  # 检查向量检索是否没有返回可采纳的文本节点。
        return "不知道。资料库中没有找到与您问题相关的信息。"  # 无可靠参考资料时直接返回明确提示，不调用模型。

    prompt = build_prompt(query, retrieved_nodes)  # 调用 build_prompt 组合用户问题和检索节点，生成约束回答范围的提示词。
    print(f"正在调用 {OPENAI_LLM_MODEL} 生成回答...")  # 调用 print 显示当前将请求的模型名称。
    api_key = get_openai_api_key()  # 调用配置函数读取 OpenAI API Key，未配置时会给出明确错误。
    llm = OpenAI(model=OPENAI_LLM_MODEL, temperature=0.1, max_tokens=1024, api_key=api_key)  # 调用 OpenAI 构造函数设置模型、凭据、低随机性和最大回答 token 数。
    response = llm.complete(prompt)  # 调用 LlamaIndex complete 将完整提示词发送给 OpenAI 并获取补全文本。
    answer = response.text.strip()  # 调用 str.strip 清理模型回答首尾空白。
    if not answer:  # 检查模型响应是否为空字符串。
        raise RuntimeError("OpenAI 模型未返回有效回答文本。")  # 明确报告模型响应缺少文本，不返回成功形态的空答案。

    print("LLM 返回回答：")  # 调用 print 显示模型回答区标题。
    print("-" * 50)  # 调用 print 输出分隔线，区分回答正文。
    print(answer)  # 调用 print 显示生成的回答文本。
    print("-" * 50)  # 调用 print 输出回答区结束分隔线。
    return answer  # 返回模型生成的回答供调用方复用。
