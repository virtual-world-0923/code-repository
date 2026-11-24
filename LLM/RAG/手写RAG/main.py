from rag import RAG


def main():

    rag = RAG()

    # 第一次运行：构建知识库
    # rag.build_knowledge_base()

    # 如果已经构建过知识库，
    # 下次运行可以改成：
    rag.load_knowledge_base()

    while True:

        question = input("\n请输入问题（输入 exit 退出）：")

        if question.lower() == "exit":
            break

        answer = rag.ask(question)

        print("\n最终回答：")

        print(answer)


if __name__ == "__main__":

    main()