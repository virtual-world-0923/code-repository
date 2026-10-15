import os
import re
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params import ResponseFormatJSONObject

API_KEYz = os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY_HERE")
API_KEYg = os.environ.get("SILICONFLOW_API_KEY", "YOUR_ZHIPU_API_KEY_HERE")

client = OpenAI(
    api_key=API_KEYg,
    base_url="https://api.siliconflow.cn/v1"
)

# 1. 用 Pydantic 定义想要的 JSON 格式
class UserInfo(BaseModel):
    name: str = Field(description="人员姓名")
    age: int = Field(description="年龄")
    company: str = Field(description="任职公司名称")
    skills: list[str] = Field(description="技能列表")

# 生成 JSON Schema 并转为字符串存入 Prompt
schema_json = json.dumps(UserInfo.model_json_schema(), ensure_ascii=False)
# mes: list[ChatCompletionMessageParam] = [
#     ChatCompletionSystemMessageParam(
#         role="system",
#         content = f"你是一个信息提取助手。请严格按照以下 JSON Schema 输出数据，不得改动字段名：\n{schema_json}"
#     ),
#     ChatCompletionUserMessageParam(
#         role="user",
#         content="张伟今年 29 岁，目前在腾讯担任高级后端开发工程师，平时主要使用 Go 语言、Docker 和 Redis 进行微服务开发。"
#     )
# ]

mes: list[ChatCompletionMessageParam] = [
    ChatCompletionSystemMessageParam(
        role="system",
        content = "你是一个信息提取助手。请从用户输入的文本中精准提取关键数据并按要求输出。"
    ),
    ChatCompletionUserMessageParam(
        role="user",
        content="张伟今年 29 岁，目前在腾讯担任高级后端开发工程师，平时主要使用 Go 语言、Docker 和 Redis 进行微服务开发。"
    )
]

user_info_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "人员姓名"
        },
        "age": {
            "type": "integer",
            "description": "年龄"
        },
        "company": {
            "type": "string",
            "description": "任职公司名称"
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "掌握的技能或技术栈"
        }
    },
    "required": ["name", "age", "company", "skills"],
    "additionalProperties": False
}

# 不是所有模型都支持json_schema，如果不支持，返回json_object，并手动解析。或者在prompt中显示规定返回json字段
rf = ResponseFormatJSONSchema(
    type="json_schema",
    json_schema={
        "name": "user_profile_extraction",
        "schema": user_info_schema,
        "strict": True,  # 可选，如果模型支持 Strict Mode 可以开启
    }
)
# rf = ResponseFormatJSONObject(type = "json_object")

response = client.chat.completions.create(
    model = "deepseek-ai/DeepSeek-V4-Pro",
    messages = mes,
    response_format = rf,
)

# 5. 取出 content 并增加空值防护
raw_content = response.choices[0].message.content
print(raw_content)
if not raw_content:
    raise ValueError("大模型未返回任何文本内容！")

# 6. 正则脱壳：彻底切除 ```json 和 ``` 标记
json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
if json_match:
    clean_content = json_match.group(1).strip()
else:
    clean_content = raw_content.strip()

# 7. 使用 Pydantic 解析清洗后的字符串（兼顾强类型校验）
user_info = UserInfo.model_validate_json(clean_content)
print("--- 解析成功，提取出的 Pydantic 对象 ---")
print(f"姓名: {user_info.name}")
print(f"年龄: {user_info.age}")
print(f"公司: {user_info.company}")
print(f"技能: {user_info.skills}")


