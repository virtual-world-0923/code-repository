import os

OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_URL = os.environ.get("OPENAI_URL")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

from LLM.http.httpx截取 import http_client

model = ChatOpenAI(
    temperature=0,
    api_key=OPENAI_KEY,
    base_url=OPENAI_URL,
    model=OPENAI_MODEL,
    http_client=http_client,
)