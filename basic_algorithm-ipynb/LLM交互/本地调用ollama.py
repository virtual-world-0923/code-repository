# ============================================================
# 功能：通过 OpenAI 官方 Python 库，调用本地 Ollama 服务中的模型，并以流式输出（打字机效果）打印回复。
#
# 背景知识：
#   - Ollama 是一个在本地运行大模型的工具，启动后会在本机开一个 HTTP 服务（默认端口 11434）。
#   - Ollama 提供了与 OpenAI 接口兼容的 API（路径以 /v1 结尾），
#     所以可以直接用 openai 库去调用本地模型，不需要真正的 OpenAI 账号，也不会联网收费。
#   - 与前两份代码的区别：模型由 Ollama 服务在另一个进程里加载和运行，
#     这段代码只负责“发请求、收结果”，自己不加载模型。
#   - 使用前提：1) 已安装并启动 Ollama；2) 已创建/下载好名为 my-qwen 的模型；
#               3) 已安装 openai 库：pip install openai
# ============================================================

# OpenAI：openai 库提供的客户端类，用来向兼容 OpenAI 协议的服务发请求
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionMessageParam,
)

# ---------- 1. 初始化客户端，指向 Ollama 本地服务器 ----------
client = OpenAI(
    base_url="http://localhost:11434/v1",  # API 的基础地址。
                                           # 默认是 OpenAI 官方地址，这里改成本机的 Ollama 服务：
                                           #   localhost：本机；11434：Ollama 默认端口；/v1：OpenAI 兼容接口路径
                                           # 如果 Ollama 跑在别的机器上，把 localhost 换成对方 IP 即可
    api_key="ollama",                      # API 密钥。openai 库要求必须传，
                                           # 但 Ollama 不校验，所以填任意字符串都可以
)

# ---------- 2. 准备对话内容 ----------
# messages 是消息列表，每条消息包含：
#   role    ：说话者角色
#             - "system"    系统提示词，设定模型的身份/风格/规则
#             - "user"      用户的提问
#             - "assistant" 模型之前的回复（多轮对话时需要带上历史）
#   content ：消息的具体文本
# 使用联合类型 ChatCompletionMessageParam
messages: list[ChatCompletionMessageParam] = [
    ChatCompletionSystemMessageParam(
        role="system",
        content="幽默的脱口秀演员"
    ),
    ChatCompletionUserMessageParam(
        role="user",
        content="你的角色是什么？"
    ),
]
# 和下面的写法是一样的效果。检查器只能检查出是 str，不能保证类型一定正确，可以消除IDE的传入类型警告
# messages = [
#     {"role": "system", "content": "幽默的脱口秀演员"},   # 让模型扮演幽默的脱口秀演员
#     {"role": "user", "content": "你的角色是什么？"}       # 用户提的问题
# ]

# 打印提示信息；"="*40 表示重复 40 个等号，用作分割线
# 注：提示里写的“已通过 GPU 加速”只是文字，代码本身并不能保证用了 GPU，
# 是否使用 GPU 由 Ollama 服务端决定
print("正在向本地 Ollama 发送请求（已通过 GPU 加速）...\n" + "="*40)

# ---------- 3. 调用模型（流式输出） ----------
# client.chat.completions.create：向服务端发送一次“对话补全”请求
completion = client.chat.completions.create(
    model="my-qwen",       # 要调用的模型名称，必须是 Ollama 中已存在的模型。
                           # 这里是用 `ollama create` 自定义创建的模型名；
                           # 可以在命令行用 `ollama list` 查看本机已有的模型
    messages=messages,     # 上面准备好的对话消息列表
    stream=True,           # True：流式输出，模型每生成一小段就立刻返回一个片段（chunk），
                           #       返回值是一个可迭代对象，可以用 for 循环逐个取；
                           # False：等全部生成完，一次性返回完整结果
    temperature=0.3,       # 采样温度：越低输出越稳定、保守；越高越随机、有创意。
                           # 0.3 偏稳定，适合需要比较准确的回答
    max_tokens=512         #max_tokens：最多生成多少个 token
    # 另外还有一些常用但本例没有设置的参数：
    #   max_tokens：最多生成多少个 token，回复被截断时可以加上
    #   top_p     ：核采样阈值，只从累计概率达到该值的候选词中采样
    #   stop      ：遇到指定字符串时停止生成
)

# ---------- 4. 实时打印 AI 的回复 ----------
# completion 是一个流，每次循环拿到一个 chunk（一小段增量结果）
for chunk in completion:
    # 新版 openai 库返回的是对象而不是字典，所以用“点号”取属性，而不是 ["..."]：
    #   chunk.choices[0]       ：候选回复列表中的第 1 条（这里只生成 1 条）
    #   .delta                 ：本次新增的增量内容（流式模式用 delta；非流式用 message）
    #   .content               ：增量里的文字片段
    content = chunk.choices[0].delta.content

    # 第一个 chunk 通常只有 role 没有文字，最后一个 chunk 的 content 是 None，
    # 所以要先判断是否有内容，避免打印出 "None"
    if content:
        # end=""     ：不换行，让后面的文字接在同一行
        # flush=True ：立即把内容输出到屏幕，才能看到逐字出现的效果
        print(content, end="", flush=True)

# 输出结束后换行，并打印分割线和结束提示
print("\n" + "="*40 + "\n程序测试运行完毕！")