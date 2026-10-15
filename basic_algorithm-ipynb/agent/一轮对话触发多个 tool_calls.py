import json
from 调用LLM import call_llm

# 1. 本地业务函数
def get_current_weather(location: str) -> str:
    return json.dumps({"location": location, "temperature": "22°C", "condition": "晴朗"}, ensure_ascii=False)


def calculate(expression: str) -> str:
    try:
        return json.dumps({"expression": expression, "result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_stock_price(ticker: str) -> str:
    prices = {"AAPL": "225.50 USD", "NVDA": "120.80 USD"}
    return json.dumps({"ticker": ticker, "price": prices.get(ticker.upper(), "未找到")})


# 2. 配置 3 个工具 Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                           "required": ["expression"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询股票价格",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    }
]

func_map = {"get_current_weather": get_current_weather, "calculate": calculate, "get_stock_price": get_stock_price}

# 3. 发送包含多需求的请求
messages = [{"role": "user", "content": "帮我看看北京的天气，算一下 23 乘以 45 等于多少，再查下 AAPL 的股价。"}]
print("【发送复合请求给 LLM】...")
response_message = call_llm(messages=messages, tools=tools)  # 调用代码七

tool_calls = response_message.tool_calls

# 4. 真实并发/顺序执行所有返回的 tool_calls
if tool_calls:
    print(f"真实 LLM 成功返回了 {len(tool_calls)} 个工具调用任务！")
    messages.append(response_message)

    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        print(f" -> 执行: {fn_name}, 参数: {fn_args}")

        fn_res = func_map[fn_name](**fn_args)
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": fn_name,
            "content": fn_res,
        })

    # 5. 将搜集到的全部工具结果汇总二次发给代码七
    final_message = call_llm(messages=messages)  # 调用代码七
    print("\n最终汇总回答：\n", final_message.content)