
"""
文件：agent/manual_agent.py

功能：不使用 create_agent，
      手动实现 LangChain Tool Calling 循环。
"""

from langchain_core.messages import HumanMessage, ToolMessage

from LLM.agent.langchain.模块练习.model_config import model
from LLM.agent.langchain.模块练习.tools import tool_map, tools

# ==================================================
# 1. 将工具绑定到模型
# ==================================================

model_with_tools = model.bind_tools(tools)


# ==================================================
# 2. 定义 Agent 执行函数
# ==================================================

def run_agent(
    user_input: str,
    max_iterations: int = 5
) -> str:
    """
    手动执行 Agent 工具调用流程。

    参数：

        user_input:
            用户输入的自然语言问题。

        max_iterations:
            最大模型调用轮数。
            用于避免 Agent 无限循环。

    返回：

        str:
            Agent 最终回答。
    """

    # ------------------------------------------
    # 第一步：创建消息历史
    # ------------------------------------------

    messages = [

        HumanMessage(
            content=user_input
        )

    ]

    # messages 用于保存完整对话过程
    # 后面每一次模型调用都会使用它


    # ------------------------------------------
    # 第二步：进入 Agent 循环
    # ------------------------------------------

    for iteration in range(max_iterations):

        print(
            f"\n========== 第 {iteration + 1} 轮模型调用 =========="
        )

        # --------------------------------------
        # 2.1 调用大模型
        # --------------------------------------

        response = model_with_tools.invoke(
            messages
        )

        # 将 AIMessage 保存到消息历史
        messages.append(response)

        # 查看模型返回的文本
        print("模型内容：", response.content)


        # --------------------------------------
        # 2.2 判断模型是否请求调用工具
        # --------------------------------------

        tool_calls = response.tool_calls

        # 如果没有工具调用请求
        # 说明模型已经准备好直接回答
        if not tool_calls:

            print("\nAgent 执行结束")

            return response.content


        # --------------------------------------
        # 2.3 遍历模型请求的工具
        # --------------------------------------

        for tool_call in tool_calls:

            # 获取工具名称
            tool_name = tool_call["name"]

            # 获取工具参数
            tool_args = tool_call["args"]

            # 获取工具调用 ID
            tool_call_id = tool_call["id"]


            print("\n模型请求调用工具：", tool_name)

            print("工具参数：", tool_args)

            print("工具调用 ID：", tool_call_id)


            # ----------------------------------
            # 2.4 根据名称查找工具
            # ----------------------------------

            selected_tool = tool_map.get(tool_name)

            # 判断工具是否存在
            if selected_tool is None:

                tool_result = (
                    f"工具不存在：{tool_name}"
                )

            else:

                # ----------------------------------
                # 2.5 执行工具
                # ----------------------------------

                try:

                    tool_result = selected_tool.invoke(tool_args)

                except Exception as e:

                    # 工具执行失败时
                    # 将异常信息返回给模型
                    tool_result = (
                        f"工具执行失败：{str(e)}"
                    )


            print("工具执行结果：", tool_result)


            # ----------------------------------
            # 2.6 创建 ToolMessage
            # ----------------------------------

            tool_message = ToolMessage(

                # 工具执行结果
                content=str(tool_result),

                # 必须对应模型请求中的工具调用 ID
                tool_call_id=tool_call_id,

                # 工具名称
                name=tool_name
            )


            # ----------------------------------
            # 2.7 将工具结果加入消息历史
            # ----------------------------------

            messages.append(tool_message)


        # 当前这一轮工具全部执行完毕
        # 回到循环顶部，再次调用大模型
        #
        # 大模型就可以读取 ToolMessage
        # 根据工具执行结果决定下一步操作


    # ==================================================
    # 3. 超过最大循环次数
    # ==================================================

    return (
        "Agent 已达到最大工具调用轮数，"
        "请尝试简化问题或提高 max_iterations。"
    )

answer = run_agent(user_input="北京天气，现在日期时间", max_iterations=5)
print(answer)