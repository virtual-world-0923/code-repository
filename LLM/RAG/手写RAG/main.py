from rag import RAG


def main():

    rag = RAG()

    # 删除知识库
    rag.delete_knowledge_base()

    # 构建知识库,存在加载，不存在构建
    rag.build_knowledge_base()

    while True:

        question = input("\n请输入问题（输入 exit 退出）：")

        if question.lower() == "exit":
            break

        answer = rag.ask(question)

        print("\n最终回答：")

        print(answer)


if __name__ == "__main__":

    main()