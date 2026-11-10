import json
from typing import Callable, Dict, Any, List
from 调用LLM import call_llm

# Callable：表示一个函数/方法等可调用对象
# Any：表示任意类型，关闭类型检查

# 1. 工具注册器
class ToolRegistry:
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, name: str, description: str, parameters: dict) -> Callable[..., Callable[..., Any]]:
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


# 3. 路由 Agent
class RouterAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, user_prompt: str):
        print(f"\n======== 用户指令: '{user_prompt}' ========")
        messages = [{"role": "user", "content": user_prompt}]

        # 1. 调用代码七传递包含全量注册 Schema 的请求
        msg = call_llm(messages=messages, tools=self.registry.get_schemas())
        messages.append(msg)
        while(True):
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    print(f"【路由抉择】: 调用工具 -> {fn_name}")
                    exec_res = self.registry.execute(fn_name, fn_args)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": exec_res
                    })

                msg = call_llm(messages=messages, tools=self.registry.get_schemas())
                messages.append(msg)
            else:
                print(f"【Agent 最终回答】:\n{msg.content}")
                break

            
registry = ToolRegistry()

# 2. 注入业务工具
@registry.register(
    name="get_user_balance",
    description="查询指定用户 ID 的账户余额信息",
    parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
)
def get_user_balance(user_id: str) -> dict:
    re = {"user_id": user_id, "balance": 10500.50, "currency": "CNY"}
    return re

agent = RouterAgent(registry=registry)
agent.run("帮我查一下用户 U10086 的账户余额是多少？")
