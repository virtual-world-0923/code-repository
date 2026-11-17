import json
import httpx

from langchain_openai import ChatOpenAI
from langchain.tools import tool


# ==================================================
# 1. 定义 HTTP 请求日志函数
# ==================================================

def log_request(request: httpx.Request):

    print("\n" + "=" * 70)
    print("发送给模型的原始 HTTP 请求")
    print("=" * 70)

    print("请求地址：", request.url)
    print("请求方法：", request.method)

    # 读取请求体
    body = request.content

    if body:
        try:
            # 将 JSON 字符串格式化输出
            data = json.loads(body)

            print(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=4
                )
            )

        except Exception:
            print(body.decode("utf-8", errors="replace"))


# ==================================================
# 2. 定义 HTTP 响应日志函数
# ==================================================

def log_response(response: httpx.Response):

    # 读取响应体
    # 注意：read() 会将响应体加载到内存
    response.read()

    print("\n" + "=" * 70)
    print("模型返回的原始 HTTP 响应")
    print("=" * 70)

    print("HTTP 状态码：", response.status_code)

    body = response.content

    if body:
        try:
            data = json.loads(body)

            print(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=4
                )
            )

        except Exception:
            print(body.decode("utf-8", errors="replace"))


# ==================================================
# 3. 创建 HTTPX 客户端
# ==================================================

http_client = httpx.Client(
    event_hooks={
        "request": [log_request],
        "response": [log_response]
    }
)


# ==================================================
# 4. 定义工具
# ==================================================

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气。

    Args:
        city: 城市名称。
    """
    return f"{city}今天晴天，25度"


# ==================================================
# 5. 创建模型
# ==================================================

model = ChatOpenAI(
    model="GLM-4-Flash",
    temperature=0,
    api_key="5e4e165df1c24405873fcb3621b5d7a0",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    http_client=http_client
)


# ==================================================
# 6. 绑定工具
# ==================================================

model_with_tools = model.bind_tools([
    get_weather
])


# ==================================================
# 7. 调用模型
# ==================================================

response = model_with_tools.invoke(
    "北京今天的天气怎么样？"
)


# ==================================================
# 8. 查看 LangChain 解析后的响应
# ==================================================

print("\n" + "=" * 70)
print("LangChain 解析后的 AIMessage")
print("=" * 70)

print(response)

print("\ncontent：")
print(response.content)

print("\ntool_calls：")
print(response.tool_calls)

print("\nusage_metadata：")
print(response.usage_metadata)


# 关闭 HTTP 客户端
http_client.close()