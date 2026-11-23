"""
文件：agent/manual_agent_llamaindex.py

功能：不使用 ReActAgent 或 FunctionCallingAgent，
      手动实现 LlamaIndex Tool Calling 循环。
"""
import os
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike
from LLM.agent.httpx截取 import http_client

# ==================================================
# 1. 初始化模型与工具 (示例配置)
# ==================================================

llm = OpenAILike(
    model=os.environ.get("OPENAI_MODEL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_base=os.environ.get("OPENAI_BASE_URL"),
    is_chat_model=True, # 调用没有支持的第三方模型时必须写
    is_function_calling_model=True,# 调用没有支持的第三方模型时必须写
    http_client=http_client,
)

# 定义示例工具函数
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}：晴天，25°C"

def get_current_time() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 将函数包装为 LlamaIndex 的 FunctionTool
tools: list[FunctionTool] = [
    FunctionTool.from_defaults(fn=get_weather),
    FunctionTool.from_defaults(fn=get_current_time),
]

# 构建工具名称到工具对象的映射字典，方便后续查找
tool_map = {tool.metadata.name: tool for tool in tools}


# ==================================================
# 2. 定义 Agent 执行函数
# ==================================================

def run_agent(
    user_input: str,
    max_iterations: int = 5
) -> str:
    """
    手动执行 LlamaIndex Agent 工具调用流程。

    参数：
        user_input: 用户输入的自然语言问题。
        max_iterations: 最大模型调用轮数，避免无限循环。

    返回：
        str: Agent 最终回答。
    """

    # ------------------------------------------
    # 第一步：创建消息历史
    # ------------------------------------------
    messages = [
        ChatMessage(role=MessageRole.USER, content=user_input)
    ]

    # ------------------------------------------
    # 第二步：进入 Agent 循环
    # ------------------------------------------
    for iteration in range(max_iterations):
        print(f"\n========== 第 {iteration + 1} 轮模型调用 ==========")

        # --------------------------------------
        # 2.1 调用大模型 (使用 chat_with_tools 传入工具)
        # --------------------------------------
        response = llm.chat_with_tools(
            tools=tools,
            chat_history=messages,
            allow_parallel_tool_calls=True  # 允许单轮调用多个工具
        )

        # 将助手的回复 (ChatMessage) 加入消息历史
        assistant_message = response.message
        messages.append(assistant_message)

        print("模型内容：", assistant_message.content)

        # --------------------------------------
        # 2.2 提取模型请求的工具调用
        # --------------------------------------
        # LlamaIndex 通过 get_tool_calls_from_response 提取标准化的 ToolSelection 对象列表
        tool_calls = llm.get_tool_calls_from_response(
            response, 
            error_on_no_tool_call=False
        )

        # 如果没有工具调用请求，说明模型已经生成最终回答
        if not tool_calls:
            print("\nAgent 执行结束")
            return assistant_message.content or ""

        # --------------------------------------
        # 2.3 遍历模型请求的工具
        # --------------------------------------
        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_args = tool_call.tool_kwargs
            tool_call_id = tool_call.tool_id

            print("\n模型请求调用工具：", tool_name)
            print("工具参数：", tool_args)
            print("工具调用 ID：", tool_call_id)

            # ----------------------------------
            # 2.4 根据名称查找工具
            # ----------------------------------
            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                tool_result = f"工具不存在：{tool_name}"
            else:
                # ----------------------------------
                # 2.5 执行工具
                # ----------------------------------
                try:
                    # LlamaIndex 的 FunctionTool 使用 __call__ 或 .call() 执行
                    tool_output = selected_tool(**tool_args)
                    tool_result = str(tool_output.content)
                except Exception as e:
                    tool_result = f"工具执行失败：{str(e)}"

            print("工具执行结果：", tool_result)

            # ----------------------------------
            # 2.6 创建工具返回消息 (MessageRole.TOOL)
            # ----------------------------------
            tool_message = ChatMessage(
                role=MessageRole.TOOL,
                content=tool_result,
                additional_kwargs={
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                }
            )

            # ----------------------------------
            # 2.7 将工具结果加入消息历史
            # ----------------------------------
            messages.append(tool_message)

    # ==================================================
    # 3. 超过最大循环次数
    # ==================================================
    return (
        "Agent 已达到最大工具调用轮数，"
        "请尝试简化问题或提高 max_iterations。"
    )


if __name__ == "__main__":
    answer = run_agent(user_input="北京天气，现在日期时间", max_iterations=5)
    print("\n最终回答：", answer)
