import json

import httpx


def log_request(request: httpx.Request):
    """
    定义 HTTP 请求日志函数
    """

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

            print(json.dumps(data, ensure_ascii=False, indent=4))

        except Exception:
            print(body.decode("utf-8", errors="replace"))


def log_response(response: httpx.Response):
    """
    定义 HTTP 响应日志函数
    """

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

            print(json.dumps(data, ensure_ascii=False, indent=4))

        except Exception:
            print(body.decode("utf-8", errors="replace"))


# 创建 HTTPX 客户端
http_client = httpx.Client(
    event_hooks={"request": [log_request], "response": [log_response]}
)
