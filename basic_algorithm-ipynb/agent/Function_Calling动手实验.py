# -*- coding: utf-8 -*-
"""
=====================================================================
Function Calling / Tool Use 动手实验脚本
从「手写 JSON Schema」一路做到「多工具 Agent 自动路由」
=====================================================================

【环境准备】在 Anaconda 里执行（本地 CPU 即可跑，不需要显卡）：
    pip install -U openai                  # 用 OpenAI SDK 调智谱（协议兼容）
    pip install -U langchain langchain-core langchain-openai
    pip install -U jsonschema              # PART 1 校验用（可选）
    pip install -U python-dotenv           # 读 .env 里的 API Key（可选）

【API Key】推荐写进 .env 文件，不要硬编码进代码：
    ZHIPU_API_KEY=你的智谱key
    或 OPENAI_API_KEY=sk-xxx

【怎么跑】每个 PART 都是独立函数，改最下面的 RUN_PARTS 即可单独执行：
    RUN_PARTS = {1}        # 只想看 JSON Schema
    RUN_PARTS = {1, 2}     # 只跑离线 + 原生协议
    RUN_PARTS = {1, 2, 3, 4}

【模型选择提示（很重要）】
    glm-4-flash 永久免费，做入门练习完全够用，但它体积小，
    工具调用的「选得准不准、参数填得对不对」会明显弱于大模型。
    练多工具路由（PART 4）时如果频繁选错工具，把 MODEL 换成
    glm-4-plus / glm-4.6（或 DeepSeek-V3 等）再跑，效果差别很大。
"""

import json
import os
from typing import Any, Callable, Dict

# ======================================================================
# 第 0 部分：全局配置
# ======================================================================

# 智谱是 OpenAI 协议兼容的：只改 base_url 和 api_key，代码一行不动
# （DeepSeek、通义、Kimi 等国产模型同理，都是换这两个参数）
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"          # 免费模型，练习首选
# MODEL = "glm-4-plus"         # 工具调用更稳，适合 PART 4 多工具路由

API_KEY = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY", "")


# ======================================================================
# PART 1：JSON Schema —— 工具定义的"地基"（纯离线，不需要 API Key）
# ======================================================================
# 背景：模型本身不知道你有哪些函数。你只能用"一段结构化文本"告诉它：
#       函数叫什么、干什么用、收什么参数、哪些必填、参数有哪些合法取值。
#       这段结构化文本就是 JSON Schema。它本质是一份"给模型看的接口文档"。
#
# 关键关键字：
#   type        数据类型：object / string / number / integer / boolean / array / enum 载体
#   properties  对象里有哪些字段，每个字段再套一层 mini schema
#   required    必填字段数组 —— 不写就是可选，模型可能不传
#   enum        枚举，把参数的取值空间"钉死"，是提升准确率最省事的手段
#   description 最重要的字段：模型靠它判断"什么时候该用这个工具"
#   default     默认值（可选参数用它兜底）

WEATHER_SCHEMA: Dict[str, Any] = {
    "type": "object",                       # 参数整体是一个对象（OpenAI 约定顶层必须是 object）
    "properties": {
        "city": {                           # 参数 1：城市名
            "type": "string",               # 类型是字符串
            "description": "城市名称，例如：北京、上海、深圳。不要带'市'字以外的修饰。",
        },
        "unit": {                           # 参数 2：温度单位（可选，有枚举）
            "type": "string",
            "enum": ["celsius", "fahrenheit"],   # 枚举：只能是这两个值之一
            "description": "温度单位，celsius 为摄氏度，fahrenheit 为华氏度。",
            "default": "celsius",           # 模型不传时按摄氏度处理
        },
        "days": {                           # 参数 3：预报天数（数值 + 范围约束）
            "type": "integer",
            "description": "预报天数，1 表示今天，最大 7 天。",
            "minimum": 1,
            "maximum": 7,
            "default": 1,
        },
    },
    "required": ["city"],                   # 只有 city 必填；unit / days 可选
}

# 完整的一个 tool 描述对象（OpenAI 协议的"包装层"）
# 注意结构是：{"type": "function", "function": {name / description / parameters}}
# 中间的 "function" 这层是历史遗留，但所有框架都认，别写错
WEATHER_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        # description 是"模型选工具"的第一依据，要写清三件事：干什么 / 什么时候用 / 返回什么
        "description": (
            "查询指定城市的天气情况。"
            "当用户询问天气、气温、是否下雨、穿衣建议时使用本工具。"
            "不要用它查询空气质量或股票。"
        ),
        "parameters": WEATHER_SCHEMA,       # 上面那份 schema 挂在这里
    },
}


def part1_json_schema():
    """离线演示：schema 怎么校验模型吐出来的参数。"""
    print("=" * 70)
    print("PART 1  JSON Schema 基础")
    print("=" * 70)

    print("\n[1] 一份完整的工具定义（这就是真正发给模型的东西）：")
    print(json.dumps(WEATHER_TOOL_DEF, ensure_ascii=False, indent=2))

    # 模拟三次"模型填参数"的结果，看哪些合法、哪些非法
    samples = [
        {"city": "北京"},                                  # 合法：只填必填项
        {"city": "上海", "unit": "fahrenheit", "days": 3},  # 合法：全填
        {"city": "广州", "unit": "开尔文"},                 # 非法：unit 不在枚举里
        {"unit": "celsius"},                               # 非法：缺必填 city
    ]

    try:
        import jsonschema                                  # 可选依赖，没装就跳过校验
    except ImportError:
        print("\n[2] 未安装 jsonschema，跳过自动校验（pip install jsonschema 后可看效果）")
        print("    手动判断：第 3 个样例 unit 不在枚举内，第 4 个样例缺必填 city。")
        return

    print("\n[2] 用 jsonschema 校验模型填出来的参数：")
    for i, args in enumerate(samples, 1):
        try:
            jsonschema.validate(instance=args, schema=WEATHER_SCHEMA)   # 校验：不抛异常即通过
            print(f"    样例{i} {args} -> 合法")
        except jsonschema.ValidationError as e:
            # 真跑起来时，这里就是"模型参数填错了"的拦截点
            print(f"    样例{i} {args} -> 非法：{e.message}")

    print(
        "\n[3] 工程意义：生产环境一定要在『本地执行工具前』做一次校验。\n"
        "    校验失败有两种处理：(a) 把错误信息作为 tool 结果回传给模型让它重填；\n"
        "                        (b) 直接重试一次。这是 Agent 稳定性的关键防线。"
    )


# ======================================================================
# PART 2：原生 OpenAI 协议 —— 完整的工具调用闭环（最该先学懂的部分）
# ======================================================================
# 完整链路（背下来，面试和讲课都靠它）：
#   ① 你定义 tools（JSON Schema）  →  ② 连同用户问题一起发给模型
#   ③ 模型不直接回答，而是返回 tool_calls：要调哪个函数 + 参数（JSON 字符串）
#   ④ 你的代码解析参数、本地执行函数，拿到结果
#   ⑤ 把结果包装成 role="tool" 的消息，连同 ③ 的 assistant 消息一起追加进 messages
#   ⑥ 再次请求模型 → 模型基于工具结果生成自然语言答案
# 关键点：模型永远只是"提议调用"，真正执行的是你的代码。这是安全的根源。

def part2_native_protocol():
    """用 OpenAI SDK（打智谱端点）跑一遍原生工具调用闭环。"""
    print("\n" + "=" * 70)
    print("PART 2  原生 OpenAI 协议：从定义到闭环")
    print("=" * 70)

    if not API_KEY:
        print("!! 未检测到 ZHIPU_API_KEY / OPENAI_API_KEY 环境变量，跳过本部分。")
        return

    from 手动_JSON_Schema import OpenAI                      # 用官方 SDK，靠 base_url 切换厂商
    client = OpenAI(api_key=API_KEY, base_url=ZHIPU_BASE_URL)

    # ---------- 真实要被执行的业务函数（本地逻辑，可以是查库、调 API、算数）----------
    def get_weather(city: str, unit: str = "celsius", days: int = 1) -> str:
        """假数据：真实场景这里换成天气 API 请求即可。"""
        fake_db = {"北京": "晴，26℃", "上海": "多云，23℃", "广州": "雷阵雨，31℃"}
        base = fake_db.get(city, "晴，25℃")
        return f"{city} 未来{days}天：{base}（单位：{unit}）"

    def calculator(expression: str) -> str:
        """只允许算数学表达式，演示第二个工具。"""
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))   # 生产环境别用 eval，用 ast 或 numexpr
        except Exception as e:
            return f"计算失败：{e}"

    # ---------- 工具注册表：名字 -> 可调用对象（多工具管理的核心数据结构）----------
    TOOL_REGISTRY: Dict[str, Callable] = {
        "get_weather": get_weather,
        "calculator": calculator,
    }

    # ---------- 给模型看的工具清单 ----------
    tools = [
        WEATHER_TOOL_DEF,                       # 复用 PART 1 手写的那一份
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式，例如 '(12+8)*3'。只在需要算数时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要求值的数学表达式，只能包含数字和运算符。",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    user_question = "北京今天天气怎么样？顺便帮我算一下 (12+8)*3 等于多少"
    messages = [{"role": "user", "content": user_question}]

    # ---------- 第一轮请求：让模型决定要不要调工具 ----------
    print(f"\n[用户] {user_question}")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        # tool_choice 控制"是否必须调工具"：
        #   "auto"     默认，模型自己决定（对话场景用它）
        #   "required"/"any"  强制至少调一个（信息抽取场景常用，有些国产模型不支持，会报错）
        #   {"type":"function","function":{"name":"xxx"}}  锁定某一个工具
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    print(f"\n[第一轮] finish_reason = {resp.choices[0].finish_reason}")   # stop=直接答；tool_calls=要调工具
    print(f"[第一轮] tool_calls = {msg.tool_calls}")

    # ---------- 没有工具调用：直接返回文本答案，闭环结束 ----------
    if not msg.tool_calls:
        print(f"\n[最终答案] {msg.content}")
        return

    # ---------- 有工具调用：把 assistant 的"调用提议"先放进历史（必须放！）----------
    messages.append(msg)                     # 这步漏掉，第二轮模型会报"找不到对应 tool_call_id"

    # ---------- 逐个执行工具，把结果以 role="tool" 回传 ----------
    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name                        # 模型选中的函数名
        fn_args = json.loads(tool_call.function.arguments)        # 参数是 JSON 字符串，需反序列化
        print(f"\n[执行] {fn_name}({fn_args})")

        fn = TOOL_REGISTRY.get(fn_name)                           # 从注册表里取真实函数
        if fn is None:
            result = f"错误：未注册的工具 {fn_name}"               # 兜底：不能让程序崩
        else:
            result = fn(**fn_args)                                # 本地执行，模型碰不到你的系统

        messages.append({                                         # 工具结果消息：三要素缺一不可
            "role": "tool",
            "tool_call_id": tool_call.id,                         # 与上面的调用一一对应
            "name": fn_name,
            "content": json.dumps(result, ensure_ascii=False),     # content 必须是字符串
        })

    # ---------- 第二轮请求：模型拿到工具结果，组织成自然语言 ----------
    resp2 = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    print(f"\n[最终答案] {resp2.choices[0].message.content}")


# ======================================================================
# PART 3：LangChain —— @tool 装饰器 / StructuredTool / bind_tools
# ======================================================================
# 核心认知：LangChain 没发明新东西，它只是"自动帮你生成 PART 1 那份 JSON Schema"。
#   函数名      -> schema 里的 name
#   docstring   -> schema 里的 description（所以 docstring 不是注释，是给模型看的提示词！）
#   类型注解    -> schema 里的 type
#   Pydantic    -> 每个字段的 description / 默认值 / 约束

def part3_langchain():
    """用 LangChain 定义工具、绑定模型、手写执行循环。"""
    print("\n" + "=" * 70)
    print("PART 3  LangChain 工具注册")
    print("=" * 70)

    if not API_KEY:
        print("!! 未检测到 API Key，跳过本部分。")
        return

    from langchain_core.tools import StructuredTool, tool
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    # ---------- 写法 A：@tool 装饰器（最快，函数即工具）----------
    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """查询指定城市的天气。当用户问天气、气温、是否下雨时使用。"""   # ← 这句就是 description
        fake_db = {"北京": "晴，26℃", "上海": "多云，23℃"}
        return f"{city}：{fake_db.get(city, '晴，25℃')}（{unit}）"

    # 打印 LangChain 自动生成的 schema —— 和 PART 1 手写的那份对比一下，结构几乎一样
    print("\n[A] @tool 自动生成的 schema：")
    print(json.dumps(get_weather.args_schema.model_json_schema(), ensure_ascii=False, indent=2))

    # ---------- 写法 B：StructuredTool + Pydantic（生产推荐）----------
    class SearchInput(BaseModel):
        """工具入参模型：Field 里的 description 会原样进 schema，是"参数填对"的关键。"""
        query: str = Field(description="搜索关键词，简短精确，不要写完整句子")
        top_k: int = Field(default=3, ge=1, le=10, description="返回结果条数，1-10")

    def _search(query: str, top_k: int = 3) -> str:
        return f"【模拟搜索】'{query}' 的前 {top_k} 条结果…"

    search_tool = StructuredTool.from_function(
        func=_search,                 # 同步实现
        name="web_search",            # 显式命名：模型看到的名字（不依赖函数名）
        description="联网搜索公开信息。当问题涉及时事、最新资料、模型知识截止后的内容时使用。",
        args_schema=SearchInput,      # 注意：传类本身 SearchInput，不是实例 SearchInput()
    )

    print("\n[B] StructuredTool 的 schema：")
    print(json.dumps(search_tool.args_schema.model_json_schema(), ensure_ascii=False, indent=2))

    # ---------- 绑定工具到模型 ----------
    llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=ZHIPU_BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools([get_weather, search_tool])   # 把 schema 注入每次请求
    # 想强制调用：llm.bind_tools([...], tool_choice="any")

    # ---------- 手写执行循环（和 PART 2 完全同构，只是换成了 LangChain 的对象）----------
    messages = [{"role": "user", "content": "北京天气如何？"}]
    ai_msg = llm_with_tools.invoke([("human", "北京天气如何？")])     # 第一轮：拿 tool_calls
    print(f"\n[C] 模型返回的 tool_calls = {ai_msg.tool_calls}")

    tool_map = {t.name: t for t in [get_weather, search_tool]}       # 名字 -> 工具对象
    if ai_msg.tool_calls:
        collected = [ai_msg]                                          # 先存 assistant 消息
        for tc in ai_msg.tool_calls:
            selected = tool_map[tc["name"]]                           # 按名字查注册表
            tool_msg = selected.invoke(tc)                            # LangChain 帮你执行 + 封装 ToolMessage
            print(f"    执行 {tc['name']}({tc['args']}) -> {tool_msg.content}")
            collected.append(tool_msg)
        # 第二轮：把工具结果喂回去，让模型总结
        final = llm_with_tools.invoke([("human", "北京天气如何？")] + collected)
        print(f"\n[D] 最终答案 = {final.content}")


# ======================================================================
# PART 4：多工具管理 —— 一个 Agent 注册 N 个工具，让模型按需选择
# ======================================================================
# 让模型"选对工具"的四条实战经验（面试讲这个很加分）：
#   1. description 写"适用边界"，不只是功能。比如"不要用它查天气"能显著减少误调。
#   2. 工具数量控制在 10 个以内。太多就先做"工具检索"（先 embedding 召回 top-k 再绑定）。
#   3. 名字要有区分度。get_weather / get_air_quality 比 get_info_1 / get_info_2 强得多。
#   4. 必填参数尽量少。可选参数给 default 和 enum，模型填错的概率直线下降。

def part4_multi_tool_agent():
    """注册 4 个工具，跑一个多轮 ReAct 式循环（最多 5 步）。"""
    print("\n" + "=" * 70)
    print("PART 4  多工具注册与自动路由")
    print("=" * 70)

    if not API_KEY:
        print("!! 未检测到 API Key，跳过本部分。")
        return

    from 手动_JSON_Schema import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=ZHIPU_BASE_URL)

    # ---------- 4 个业务函数：天气 / 计算器 / 单位换算 / 查订单 ----------
    def get_weather(city: str) -> str:
        return f"{city}：晴，26℃"

    def calculator(expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"计算失败：{e}"

    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        rate = {"USD": 7.1, "EUR": 7.8, "CNY": 1.0}
        return f"{amount}{from_currency} ≈ {amount * rate.get(from_currency, 1) / rate.get(to_currency, 1):.2f}{to_currency}"

    def query_order(order_id: str) -> str:
        return f"订单 {order_id}：已发货，预计明天送达"

    # ---------- 统一注册表：新增工具只需在这里加一行 + 下面 tools 加一项 ----------
    REGISTRY: Dict[str, Callable] = {
        "get_weather": get_weather,
        "calculator": calculator,
        "convert_currency": convert_currency,
        "query_order": query_order,
    }

    # ---------- 对应的 4 份 JSON Schema（description 里都写了"适用边界"）----------
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气。仅用于天气/气温/降雨相关问题，不要用于空气质量。",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式。仅用于纯算术，货币换算请用 convert_currency。",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "数学表达式，如 12*3+5"}},
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "货币换算。涉及汇率、美元欧元人民币金额转换时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "金额数值"},
                        "from_currency": {"type": "string", "enum": ["CNY", "USD", "EUR"], "description": "源币种"},
                        "to_currency": {"type": "string", "enum": ["CNY", "USD", "EUR"], "description": "目标币种"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_order",
                "description": "查询订单物流状态。用户提供订单号询问物流时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "description": "订单号，字母+数字"}},
                    "required": ["order_id"],
                },
            },
        },
    ]

    def run_agent(user_input: str, max_steps: int = 5):
        """一个最小的 Agent 循环：反复『模型决策 → 执行 → 回填』直到模型不再要工具。"""
        messages = [{"role": "user", "content": user_input}]
        print(f"\n[用户] {user_input}")
        for step in range(1, max_steps + 1):                     # 防死循环：最多 max_steps 轮
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, tool_choice="auto",
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:                               # 模型不再要工具 → 输出最终答案
                print(f"[答案] {msg.content}")
                return msg.content

            messages.append(msg)                                 # 先记录 assistant 的调用提议
            for tc in msg.tool_calls:                            # 支持一次返回多个调用（并行调用）
                name, args = tc.function.name, json.loads(tc.function.arguments)
                fn = REGISTRY.get(name)
                result = fn(**args) if fn else f"错误：未注册工具 {name}"
                print(f"  步骤{step} 调用 {name}({args}) -> {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        print("[答案] 已达到最大步数，强制结束")
        return None

    # 三个问题分别考察：单工具 / 多工具串联 / 边界区分（算术 vs 汇率）
    run_agent("北京今天天气怎么样？")
    run_agent("帮我查一下订单 A12345 到哪了")
    run_agent("100 美元能换多少人民币？再帮我算一下换成人民币后买 3 件 89 元的商品够不够")


# ======================================================================
# PART 5：其他框架速查（对照用，理解"万变不离其宗"）
# ======================================================================
FRAMEWORK_CHEATSHEET = """
┌────────────────┬──────────────────────────────┬───────────────────────────────┐
│ 框架           │ 定义工具的写法               │ 绑定到模型                    │
├────────────────┼──────────────────────────────┼───────────────────────────────┤
│ OpenAI 原生    │ 手写 tools=[{type:function…}]│ create(..., tools=tools)      │
│ LangChain      │ @tool / StructuredTool       │ llm.bind_tools([...])         │
│                │ (Pydantic 做 args_schema)    │ + create_tool_calling_agent   │
│ LlamaIndex     │ FunctionTool.from_defaults() │ ReActAgent / FunctionAgent    │
│                │ (Annotated 写参数描述)        │ (llm=..., tools=[...])        │
│ SemanticKernel │ @kernel_function 装饰类方法  │ kernel.add_plugin() +         │
│ (微软,偏C#)    │ 类 = Plugin，方法 = Function │ function_choice_behavior=Auto │
└────────────────┴──────────────────────────────┴───────────────────────────────┘

LangChain 版本提示（2025 后变化较大，别照抄老教程）：
  老写法：from langchain.agents import create_tool_calling_agent, AgentExecutor
  新版本：上面这套已迁入 langchain-classic 包；官方新推荐 langchain.agents.create_agent
          （底层是 LangGraph 的 StateGraph）。初学先用 bind_tools + 手写循环把原理吃透，
          再上框架封装，否则出了问题不知道往哪查。

LlamaIndex 最小示例：
    from llama_index.core.tools import FunctionTool
    from llama_index.core.agent.workflow import ReActAgent
    def get_weather(location: str) -> str:
        \"\"\"查询指定地点的天气。\"\"\"
        return "晴，26℃"
    tool = FunctionTool.from_defaults(get_weather)      # docstring → description
    agent = ReActAgent(llm=llm, tools=[tool])

Semantic Kernel 最小示例（Python）：
    from semantic_kernel.functions import kernel_function
    class WeatherPlugin:
        @kernel_function(description="查询城市天气")
        def get_weather(self, city: str) -> str:
            return "晴，26℃"
    kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
"""


# ======================================================================
# 入口
# ======================================================================
if __name__ == "__main__":
    RUN_PARTS = {1, 2, 3, 4}          # ← 想跑哪部分就改这里，例如 {1} 只看 schema

    if 1 in RUN_PARTS:
        part1_json_schema()            # 离线，建议第一个跑
    if 2 in RUN_PARTS:
        part2_native_protocol()        # 原生协议闭环（最核心）
    if 3 in RUN_PARTS:
        part3_langchain()              # LangChain 封装
    if 4 in RUN_PARTS:
        part4_multi_tool_agent()       # 多工具路由
    if 5 in RUN_PARTS:
        print(FRAMEWORK_CHEATSHEET)    # 框架对照速查
