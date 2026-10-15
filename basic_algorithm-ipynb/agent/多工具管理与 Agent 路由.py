import json
from typing import Callable, Dict, Any, List
from 调用LLM import call_llm


# 1. 工具注册器
class ToolRegistry:
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, name: str, description: str, parameters: dict):
        def decorator(func: Callable):
            self._functions[name] = func
            self._schemas.append({
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters}
            })
            return func

        return decorator

    def get_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    def execute(self, name: str, arguments: dict) -> str:
        func = self._functions[name]
        return json.dumps(func(**arguments), ensure_ascii=False)


registry = ToolRegistry()


# 2. 注入业务工具
@registry.register(
    name="get_user_balance",
    description="查询指定用户 ID 的账户余额信息",
    parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
)
def get_user_balance(user_id: str) -> dict:
    return {"user_id": user_id, "balance": 10500.50, "currency": "CNY"}


# 3. 路由 Agent，通过调用代码七做决策
class RouterAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, user_prompt: str):
        print(f"\n======== 用户指令: '{user_prompt}' ========")
        messages = [{"role": "user", "content": user_prompt}]

        # 1. 调用代码七传递包含全量注册 Schema 的请求
        msg = call_llm(messages=messages, tools=self.registry.get_schemas())

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"【路由抉择】: LLM 自动匹配工具 -> {fn_name}")
                exec_res = self.registry.execute(fn_name, fn_args)

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": exec_res
                })

            # 2. 再次调用代码七生成总结
            final_msg = call_llm(messages=messages)
            print(f"【Agent 最终回答】:\n{final_msg.content}")


agent = RouterAgent(registry=registry)
agent.run("帮我查一下用户 U10086 的账户余额是多少？")