import json
from 调用LLM import call_llm


def get_weather(location: str) -> str:
    return json.dumps({"location": location, "weather": "多云", "temp": "18°C"}, ensure_ascii=False)


def search_restaurant(cuisine: str) -> str:
    return json.dumps({"cuisine": cuisine, "recommended": ["老北京烤鸭店", "蜀香阁火锅"]}, ensure_ascii=False)


tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询天气",
                                      "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                                                     "required": ["location"]}}},
    {"type": "function", "function": {"name": "search_restaurant", "description": "查找推荐餐厅",
                                      "parameters": {"type": "object", "properties": {"cuisine": {"type": "string"}},
                                                     "required": ["cuisine"]}}}
]
func_map = {"get_weather": get_weather, "search_restaurant": search_restaurant}


# 多轮对话管理函数
def run_conversation_step(user_input: str, conversation_history: list) -> str:
    conversation_history.append({"role": "user", "content": user_input})

    while True:
        # 调用代码七，纯粹交互
        msg = call_llm(messages=conversation_history, tools=tools)

        if msg.tool_calls:
            conversation_history.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn_res = func_map[fn_name](**fn_args)

                conversation_history.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": fn_res
                })
        else:
            conversation_history.append({"role": "assistant", "content": msg.content})
            return msg.content


# 真实多轮测试
history = []
print("--- 第一轮 ---")
print("Agent:", run_conversation_step("上海今天天气如何？", history))

print("\n--- 第二轮 ---")
print("Agent:", run_conversation_step("那附近有什么好的川菜推荐吗？", history))