# ============================================================
# 功能：使用 llama-cpp-python 加载本地 GGUF 格式的量化模型（Qwen2.5-1.5B-Instruct，q4_k_m），
#       并以“流式输出（打字机效果）”的方式完成一次对话。
#
# 背景知识：
#   - GGUF 是 llama.cpp 使用的模型文件格式，通常已经做过量化，体积小、可在 CPU 上运行。
#   - q4_k_m 表示 4-bit 量化中的 K-quant、Medium 档：体积小，精度损失较小，是常用的折中选择。
#   - 与上一份代码（transformers）不同，这里不需要单独加载分词器，
#     GGUF 文件内部已经包含了分词器和对话模板。
# ============================================================

# Llama：llama-cpp-python 提供的核心类，负责加载 GGUF 模型并进行推理
# 安装方式：pip install llama-cpp-python
from llama_cpp import Llama

# ---------- 1. 初始化并加载本地模型 ----------
# model_path：GGUF 模型文件的完整路径（请替换为你自己下载的文件，如换成 q8_0 精度更高但体积更大）
# 路径里的 / 在 Windows 上也可以正常使用；如果写 \ 则需要写成 \\ 或在字符串前加 r
model_path = "C:/Users/Administrator/.cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/master/qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("正在加载模型中，请稍候...")
llm = Llama(
    model_path=model_path,   # 要加载的 GGUF 模型文件路径
    n_ctx=2048,              # 上下文窗口大小（单位：token）。
                             # 指“输入 + 输出”总共最多能容纳多少 token，超出会报错或被截断。
                             # 1.5B 小模型一般设 2048 或 4096；越大越占内存
    n_gpu_layers=0,          # 把多少层模型放到 GPU 上加速。
                             #   0  ：全部在 CPU 上运行（没有显卡或未安装 GPU 版 llama-cpp-python 时使用）
                             #   -1 ：所有层都放进 GPU
                             #   N  ：只放前 N 层到 GPU（显存不够时可以部分卸载）
                             # 注意：必须安装支持 CUDA/Metal 等的 GPU 版本，此参数才会真正生效
    verbose=False            # 是否输出 llama.cpp 底层的详细日志；False 关闭，保持控制台整洁
)
print("模型加载成功！\n" + "="*30)

# ---------- 2. 准备对话内容（通用 Chat 格式） ----------
# messages 是一个列表，每个元素是一条消息，包含：
#   role    ：说话者角色
#             - "system"    系统提示词，设定模型的身份、语气、规则
#             - "user"      用户提问
#             - "assistant" 模型之前的回复（做多轮对话时需要把历史回复也放进来）
#   content ：消息的具体文本
messages = [
    {
        "role": "system",
        "content": "你是一个由阿里云开发的超强人工智能助手。请用清晰、准确的中文回答用户的问题。"
    },
    {
        "role": "user",
        "content": "请用一段话解释：什么是大语言模型的量化？"
    }
]

# ---------- 3. 生成回复 ----------

# create_chat_completion：以“对话”形式调用模型。
# 它会自动使用 GGUF 文件内置的对话模板，把 messages 拼成模型认识的格式再生成回复。
response = llm.create_chat_completion(
    messages=messages,   # 上面准备好的对话消息列表
    stream=False,         # True：流式输出，模型每生成一小段就立刻返回一个片段（chunk），
                         #       返回值是一个生成器，可以用 for 循环逐个取出；
                         # False：等全部生成完才一次性返回完整结果
    temperature=0.7,      # 采样温度：越低越保守、稳定（0 近乎固定输出）；越高越随机、有创意。
                         # 0.7 是兼顾稳定性与多样性的常用值
    max_tokens=512,   # 本次回复最多生成 512 个 token
    # 另外还有一些常用但本例没有设置的参数：
    #   top_p     ：核采样阈值，只从累计概率达到该值的候选词中采样
    #   stop      ：遇到指定字符串时停止生成
)

#--------------- 非流式输出 ------------------------
# response 是一个字典，不是生成器，不需要 for 循环
reply = response["choices"][0]["message"]["content"]
usage = response["usage"]
finish_reason = response["choices"][0]["finish_reason"]
print("回复：", reply)
print("用量：", usage)
print("完成原因：", finish_reason)
print("-------：", response["choices"])

# # ---------- 4. 循环打印流式输出的内容 ----------
# # response 是生成器
# # 每个 chunk 是一个字典，结构大致为：
# #   {"choices": [{"delta": {...}, "finish_reason": ...}], ...}
# # end=""      ：print 默认结尾换行，这里设为空字符串，让后面的流式文字接在同一行
# # flush=True  ：立即把缓冲区内容输出到屏幕，否则文字可能攒着不显示，看不到“逐字”效果
# print("AI 回复：", end="", flush=True)
# for chunk in response:
#     # chunk["choices"]：候选回复列表；这里只生成 1 条，所以取下标 [0]
#     # ["delta"]       ：本次新增的增量内容（流式模式下用 delta，非流式模式用 message）
#     delta = chunk["choices"][0]["delta"]
#
#     # 流式输出的第一个 chunk 往往只包含 {"role": "assistant"}，没有 content；
#     # 最后一个 chunk 也可能是空的，所以要先判断 "content" 是否存在，避免 KeyError
#     if "content" in delta:
#         # 打印本次新增的文字片段，不换行并立即刷新，形成打字机效果
#         print(delta["content"], end="", flush=True)
