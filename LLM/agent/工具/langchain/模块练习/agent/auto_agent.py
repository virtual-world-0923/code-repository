"""
文件：agent/auto_agent.py

功能：使用 LangChain create_agent
      自动完成工具调用。
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from LLM.agent.工具.langchain.模块练习.model_config import model
from LLM.agent.工具.langchain.模块练习.tools import tools

# ==================================================
# 1. 定义 System Prompt
# ==================================================

SYSTEM_PROMPT = """
你是一个智能助手。

你可以使用以下工具：

1. get_weather：
   查询指定城市的当前天气。

2. calculator：
   执行数学计算。

3. get_current_time：
   获取指定时区的当前时间。

请遵守以下规则：

1. 根据用户问题判断是否需要调用工具。

2. 需要天气数据时，必须调用天气工具。

3. 需要精确计算时，必须调用计算工具。

4. 需要当前时间时，必须调用时间工具。

5. 可以根据任务需要调用多个工具。

6. 必须根据工具真实返回的结果回答。

7. 不允许编造工具执行结果。

8. 如果工具执行失败，应如实说明。
"""


# ==================================================
# 2. 创建 Agent
# ==================================================

agent = create_agent(
    # 指定大语言模型
    model=model,
    # 注册所有工具
    tools=tools,
    # 指定系统提示词
    system_prompt=SYSTEM_PROMPT, # "role": "system" 消息
)


# ==================================================
# 3. 定义 Agent 执行函数
# ==================================================


def run_auto_agent(user_input: str) -> str:
    """
    使用 create_agent 执行工具调用。

    参数：

        user_input:
            用户输入的问题。

    返回：

        str:
            Agent 最终回答。
    """

    # ------------------------------------------
    # 构造 Agent 输入
    # ------------------------------------------

    # agent_input = {"messages": [{"role": "user", "content": user_input}]}
    # 或
    agent_input = {"messages": [HumanMessage(content=user_input)]}

    # ------------------------------------------
    # 调用 Agent
    # ------------------------------------------

    result = agent.invoke(
        agent_input,
        # 限制图执行的最大步骤数
        config={"recursion_limit": 15},
    )

    # ------------------------------------------
    # 获取消息历史
    # ------------------------------------------

    messages = result["messages"]

    print(type(result))
    print(result)
    print("\n")

    # 最后一条消息通常是最终 AI 回复
    final_message = messages[-1]

    return final_message.content

answer = run_auto_agent(user_input="北京天气，现在日期时间")
print(answer)