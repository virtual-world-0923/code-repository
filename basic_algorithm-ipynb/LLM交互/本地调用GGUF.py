from llama_cpp import Llama

# 1. 初始化并加载本地模型（请将 model_path 替换为你挑选的文件名，如 q8_0）
model_path = "C:/Users/Administrator/.cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/master/qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("正在加载模型中，请稍候...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,      # 上下文窗口大小（1.5B模型通常设为 2048 或 4096 即可）
    n_gpu_layers=0, # -1 代表将所有模型层放进显卡加速。如果只有CPU，写 0 即可
    verbose=False    # 关闭底层的冗长日志输出，保持控制台整洁
)
print("模型加载成功！\n" + "="*30)

# 2. 准备对话内容（采用通用的 Chat 格式）
messages = [
    {
        "role": "system",
        "content": "你是一个由阿里云开发的超强人工智能助手。请用清晰、准确的中文回答用户的问题。"
    },
    {
        "role": "user",
        "content": "请用一段话简要解释：什么是大语言模型的量化？"
    }
]

# 3. 生成回复（开启 stream=True 可以像 ChatGPT 一样逐字打字输出）
print("AI 回复：", end="", flush=True)

response_stream = llm.create_chat_completion(
    messages=messages,
    stream=True,        # 开启流式输出（打字机效果）
    temperature=0.7     # 采样温度，0.7 代表兼顾创造力与稳定性
)

# 4. 循环打印流式输出的内容
for chunk in response_stream:
    # 提取当前字词片段
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)

print("\n" + "="*30)
