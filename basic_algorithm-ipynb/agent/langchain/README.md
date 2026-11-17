目录结构：
langchain_tool_demo/
│
├── .env
├── requirements.txt
├── main.py
│
├── tools/
│   ├── __init__.py
│   ├── weather_tool.py
│   ├── calculator_tool.py
│   └── time_tool.py
│
└── agent/
    ├── __init__.py
    ├── manual_agent.py
    └── auto_agent.py


每个文件的职责：
文件	职责
.env	保存 API Key 和接口地址
requirements.txt	保存项目依赖
main.py	项目入口
tools/	存放所有自定义工具
manual_agent.py	手动实现 Agent 循环
auto_agent.py	使用 create_agent 创建 Agent
