import os
from openai import OpenAI
from pprint import pprint

def _call_openAI(API_KEY: str, BASE_URL: str, model: str, messages: list, tools: list = None, isStream : bool = False):
    # 初始化 OpenAI 兼容客户端
    _client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 组装请求参数字典
    request_params = {
        "model": model, # 使用的模型标识符
        "messages": messages, # 准备发给 LLM 的完整上下文消息列表
        "stream": isStream,  # 在这里设置：False 为非流式，True 为流式
    }

    # 只有当传入了工具列表时，才添加 tools 参数
    if tools:
        request_params["tools"] = tools

    # 发送真实 HTTP 请求给大模型 API
    response = _client.chat.completions.create(**request_params)
    mess = _process_response(response, isStream)
    return mess
'''
    此函数有 yield ，python会把函数会变成一个生成器 Generator，调用方可以通过 for 循环实时接收）
'''
def _process_response(response,isStream : bool):

    # isStream True 流式，isStream False 非流式

    mess = None
    if isStream: # 流式输出需要用一个 for 循环来遍历每一个数据块 (chunk)
        for chunk in response:
            # 提取每一个片段的内容
            delta_content = chunk.choices[0].delta.content
            yield delta_content
    else: # 非流式
        mess = response.choices[0].message.content
        yield mess

def zhipu_llm(messages: list, model_name: str = "glm-4-flash", tools: list = None, isStream: bool = False):
    # -------------------------------------------------------------------
    # 配置真实免费 LLM 客户端（这里默认使用智谱 GLM-4-Flash，全接口免费）
    # 注册地址：https://open.bigmodel.cn/ 免费获取 API Key
    # -------------------------------------------------------------------
    API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY_HERE")
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    response = _call_openAI(API_KEY, BASE_URL, model_name, messages, tools, isStream)
    return response