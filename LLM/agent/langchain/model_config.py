"""
文件：model_config.py

功能：统一初始化大语言模型
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from LLM.agent.httpx截取 import http_client

# ==================================================
# 1. 加载 .env 文件
# ==================================================

load_dotenv()


# ==================================================
# 2. 读取环境变量
# ==================================================

API_KEY = os.getenv("OPENAI_API_KEY")

BASE_URL = os.getenv("OPENAI_BASE_URL")

MODEL_NAME = os.getenv("OPENAI_MODEL", "GLM-4-Flash")


# ==================================================
# 3. 检查 API Key
# ==================================================

if not API_KEY:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件")


# ==================================================
# 4. 创建 ChatOpenAI 模型
# ==================================================

model = ChatOpenAI(
    # 指定模型名称
    model=MODEL_NAME,
    # 控制生成随机性
    temperature=0,
    # API 密钥
    api_key=API_KEY,
    # 模型服务地址
    # None 表示使用默认 OpenAI 地址
    base_url=BASE_URL,

    http_client=http_client
)
