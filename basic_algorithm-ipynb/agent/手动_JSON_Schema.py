import json
from 调用LLM import call_llm


# 1. 本地实际调用的函数
def get_current_weather(location: str, unit: str = "celsius") -> str:
    return json.dumps({"location": location, "temperature": "22", "unit": unit, "condition": "晴朗"},
                      ensure_ascii=False)


# 2. 手动拼写 JSON Schema
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称，例如：北京、上海"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，如果不指定则默认使用 celsius"
                }
            },
            "required": ["location"]
        }
    }
}]

# 3. 构造消息，通过代码七发送真实请求
messages = [{"role": "user", "content": "北京今天天气怎么样？请用摄氏度回答。"}]
print("【第一次调用 LLM】...")
response_message = call_llm(messages=messages, tools=tools)

tool_calls = response_message.tool_calls

# 4. 解析并执行本地工具
if tool_calls:
    messages.append(response_message)  # 追加模型中间状态

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        if func_name == "get_current_weather":
            func_res = get_current_weather(**func_args)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": func_res,
            })

    # 5. 将工具运行结果发送回代码七，获取最终回答
    print("【第二次调用 LLM 生成答案】...")
    final_message = call_llm(messages=messages)  # 调用代码七
    print("最终回答：", final_message.content)