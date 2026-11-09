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

# 直到模型不再返回 tool_calls 为止。因为有的模型一次调用一个工具，需要多次调用
while True:
  print("【发送复合请求给 LLM】...")
  response_message = call_llm(messages=messages, tools=tools)   # 大模型 API 是无状态的，所以每次都要带上工具列表
  tool_calls = response_message.tool_calls

  if not tool_calls:
    print("\n最终汇总回答：\n", response_message.content)
    break  # 模型没有调用工具了，说明任务完成，跳出循环

  print(f"LLM 成功返回了 {len(tool_calls)} 个工具调用任务！")
  messages.append(response_message)
  for tool_call in tool_calls:
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments)
    fn_res = func_map[fn_name](**fn_args)
    messages.append({
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": fn_name,
        "content": fn_res,
    })

    