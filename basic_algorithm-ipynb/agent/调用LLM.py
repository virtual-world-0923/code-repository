import os
from openai import OpenAI
from pprint import pprint

# -------------------------------------------------------------------
# 配置真实免费 LLM 客户端（这里默认使用智谱 GLM-4-Flash，全接口免费）
# 注册地址：https://open.bigmodel.cn/ 免费获取 API Key
# -------------------------------------------------------------------
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY_HERE")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4-flash"  # 支持 Function Calling 的全免费模型

# 初始化 OpenAI 兼容客户端
_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# -------------------------------------------------------------------
# 代码七：纯粹的请求/接收处理函数 (没有业务逻辑)
# -------------------------------------------------------------------
def call_llm(messages: list, tools: list = None, model: str = DEFAULT_MODEL):
    """
    代码七：无业务逻辑的纯粹 LLM 传输函数。

    接收传入参数 -> 发送 API 请求 -> 返回 LLM 原始 Message 对象

    参数说明:
    - messages (list): 准备发给 LLM 的完整上下文消息列表
    - tools (list, optional): 注册给模型的 JSON Schema 工具列表，默认为 None
    - model (str): 使用的模型标识符，默认使用免费的 glm-4-flash

    返回值:
    - ChatCompletionMessage: LLM 返回的原始 Message 对象 (包含 content 或 tool_calls)
    """
    # 组装请求参数字典
    request_params = {
        "model": model,
        "messages": messages,
    }

    # 只有当传入了工具列表时，才添加 tools 参数
    if tools:
        request_params["tools"] = tools

    # 发送真实 HTTP 请求给大模型 API
    pprint("------------ 发送模型 --------------")
    pprint(request_params)
    pprint("------------ 发送模型 --------------")
    response = _client.chat.completions.create(**request_params)

    # 直接提取并返回模型的第一个回答 Message 对象，不处理任何业务逻辑
    mess = response.choices[0].message
    pprint("------------ 模型返回 --------------")
    pprint(mess)
    pprint("------------ 模型返回 --------------")
    return mess