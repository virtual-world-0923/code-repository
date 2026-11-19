import json
import os
import re

from openai import OpenAI

API_KEYz = os.environ.get("ZHIPU_API_KEY", "YOUR_ZHIPU_API_KEY_HERE")
API_KEYg = os.environ.get("SILICONFLOW_API_KEY", "YOUR_ZHIPU_API_KEY_HERE")

def _call_request(mes,rf):
    client = OpenAI(
        # api_key=API_KEYg,
        # base_url="https://api.siliconflow.cn/v1" # 硅基流动
        api_key=API_KEYz,
        base_url="https://open.bigmodel.cn/api/paas/v4/" # 智谱
    )

    response = client.chat.completions.create(
        # model="deepseek-ai/DeepSeek-V4-Pro",
        model="GLM-4-Flash", # 智谱免费模型
        messages=mes,
        response_format=rf
    )
    return response
    

def _process_response(response):
    raw_content = response.choices[0].message.content
    print("--- 模型返回的原始字符串 ---")
    print(raw_content)

    # 去掉 ``` 和 json 字符。返回 re.Match 对象
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_content)
    if json_match:
        clean_content = json_match.group(1) # 返回第一个捕获组匹配到的字符串，就是提取出来的纯 JSON
    else:
        clean_content = raw_content.strip()

    print("--- 去掉无效字符后：--- ")
    print(clean_content)
    return clean_content

if __name__ == "__main__":

    # 定义自定义的 JSON Schema 契约
    # 目标：从非结构化段落中提取人员和公司信息
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

    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "user_profile_extraction",
            "schema": user_info_schema
        }
    }

    rf = {"type": "json_object"}

    mes = [
    {
        "role": "system",
        "content": "你是一个信息提取助手。请从用户输入的文本中精准提取关键数据并按要求输出"
    },
    {
        "role": "user",
        "content": "小明今年29岁，目前在腾讯担任高级后端开发工程师，平时主要使用Go语言、Docker和Redis进行微服务开发。"
    }
]

    response = _call_request(mes,rf)
    res = _process_response(response)

    try:
        parsed_data = json.loads(res) # 把 json 字符串转成 dict

        # dumps：把 dict 字符串转成 json,并格式化输出
        # indent=4 表示缩进4个空格；ensure_ascii=False 保证中文不乱码
        formatted_json = json.dumps(parsed_data, indent=4, ensure_ascii=False)
        print(formatted_json)
    except json.JSONDecodeError:
        print("返回的不是 json 字符串")

    

