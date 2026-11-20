"""
文件：main.py

功能：
    项目主入口。

    支持两种运行模式：

    1. 手动 Agent：自己实现 Tool Calling 循环。
    2. 自动 Agent：使用 create_agent。
"""

from agent.auto_agent import run_auto_agent
from agent.manual_agent import run_agent

# ==================================================
# 1. 定义主程序
# ==================================================


def main():

    print("=" * 50)

    print("LangChain Tool Calling 智能助手")

    print("=" * 50)

    print("请选择运行模式：")

    print("1. 手动 Agent（bind_tools）")

    print("2. 自动 Agent（create_agent）")

    print("输入 exit 退出程序")

    # ------------------------------------------
    # 2. 选择 Agent 类型
    # ------------------------------------------

    mode = input("\n请选择模式：").strip()

    if mode not in ["1", "2"]:
        print("退出程序")

        return

    # ------------------------------------------
    # 3. 循环接收用户输入
    # ------------------------------------------

    while True:
        user_input = input("\n用户：").strip()

        # 判断是否退出
        if user_input.lower() in ["exit", "quit"]:
            print("程序结束")

            break

        if not user_input:
            print("请输入问题")

            continue

        # --------------------------------------
        # 4. 根据模式执行 Agent
        # --------------------------------------

        try:
            if mode == "1":
                print("\n正在使用手动 Agent...")

                answer = run_agent(user_input=user_input, max_iterations=5)

            else:
                print("\n正在使用自动 Agent...")

                answer = run_auto_agent(user_input=user_input)

            # ----------------------------------
            # 5. 输出最终答案
            # ----------------------------------

            print("\nAI：")

            print(answer)

        except Exception as e:
            print("\n程序执行失败：", str(e))


# ==================================================
# 6. Python 程序入口
# ==================================================

if __name__ == "__main__":
    main()
