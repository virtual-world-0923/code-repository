from openai import OpenAI

# 1. 初始化客户端，指向 Ollama 本地默认启动的服务器地址
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama 的本地 API 接口
    api_key="ollama",                      # 必须填，但 Ollama 填任意字符均可
)

messages = [
    {"role": "system", "content": "幽默的脱口秀演员"},
    {"role": "user", "content": "你的角色是什么？"}
]

print("正在向本地 Ollama 发送请求（已通过 GPU 加速）...\n" + "="*40)

# 3. 调用模型（支持打字机流式输出）
completion = client.chat.completions.create(
    model="my-qwen",       # 填写你在 Ollama 中 create 的自定义模型名称
    messages=messages,
    stream=True,           # 开启流式打字效果
    temperature=0.3,       # 采样温度
)

# 4. 实时打印 AI 的回复
for chunk in completion:
    # 新版 openai 库的提取标准格式
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print("\n" + "="*40 + "\n程序测试运行完毕！")
