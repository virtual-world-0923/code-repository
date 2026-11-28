from langchain.tools import tool
from pydantic import BaseModel, Field


class get_weather_Input(BaseModel):
    city: str = Field(description=("查询指定城市未来几天的天气"))

    days: int = Field(description="查询未来几天，范围为1到7天", default=1, ge=1, le=7)


@tool(args_schema=get_weather_Input)
# @tool(parse_docstring=True) # 读取 Docstring
def get_weather(city: str, days: int = 1) -> str:
    """
    查询指定城市未来几天的天气。

    Args:
        city: 城市名称，例如北京、上海。
        days: 查询未来几天，范围为1到7天。
    """
    return f"{city}未来{days}天天气晴朗"


# 1. 查看工具类型
print(" \n type(get_weather)")
print(type(get_weather))

# 2. 查看工具名称
print(" \n get_weather.name")
print(get_weather.name)

# 3. 查看工具整体说明
print(" \n get_weather.description")
print(get_weather.description)

# 4. 查看参数说明
print(" \n get_weather.args")
print(get_weather.args)

# 5. 查看完整参数 JSON Schema
print(" \n get_weather.args_schema.model_json_schema()")
print(get_weather.args_schema.model_json_schema())

# 6. 查看转换成 OpenAI 工具定义之后的格式
from langchain_core.utils.function_calling import convert_to_openai_tool

print(" \n convert_to_openai_tool(get_weather)")
print(convert_to_openai_tool(get_weather))
