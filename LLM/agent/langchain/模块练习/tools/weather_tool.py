'''
这个工具的作用是：
    1. 接收大模型传入的城市名称。
    2. 调用 Open-Meteo 地理编码 API。
    3. 根据城市名称获取经纬度。
    4. 调用天气 API。
    5. 返回天气信息。
'''


"""
文件：tools/weather_tool.py

功能：天气查询工具
"""

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ==================================================
# 1. 定义工具输入参数
# ==================================================

class WeatherInput(BaseModel):
    """
    定义天气工具需要接收的参数。

    BaseModel：
        Pydantic 提供的数据模型基类。

    city：
        城市名称。

    Field：
        用于设置字段描述、默认值和校验规则。
    """

    city: str = Field(
        description="需要查询天气的城市名称，例如北京、上海、深圳"
    )


# ==================================================
# 2. 创建天气工具
# ==================================================


def get_weather1(city: str) -> str:
    """
    查询指定城市的当前天气。

    参数：
        city:
            需要查询天气的城市名称。

    返回：
        str:
            天气查询结果。
    """

    try:

        # ------------------------------------------
        # 第一步：获取城市经纬度
        # ------------------------------------------

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geo_params = {
            "name": city,
            "count": 1,
            "language": "zh",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        # HTTP 状态码异常时抛出异常
        geo_response.raise_for_status()

        # 将 JSON 字符串转换为 Python 字典
        geo_data = geo_response.json()

        # 判断城市是否存在
        if not geo_data.get("results"):
            return f"没有找到城市：{city}"

        # 获取第一个匹配城市
        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location["name"]


        # ------------------------------------------
        # 第二步：查询当前天气
        # ------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {

            # 城市纬度
            "latitude": latitude,

            # 城市经度
            "longitude": longitude,

            # 需要返回的天气字段
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "precipitation,"
                "wind_speed_10m,"
                "weather_code"
            ),

            # 使用城市所在时区
            "timezone": "auto"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        # 获取当前天气数据
        current = weather_data["current"]


        # ------------------------------------------
        # 第三步：整理返回结果
        # ------------------------------------------

        result = (
            f"城市：{city_name}\n"
            f"当前温度：{current['temperature_2m']}℃\n"
            f"体感温度：{current['apparent_temperature']}℃\n"
            f"相对湿度：{current['relative_humidity_2m']}%\n"
            f"降水量：{current['precipitation']}毫米\n"
            f"风速：{current['wind_speed_10m']}公里/小时\n"
            f"天气代码：{current['weather_code']}"
        )

        return result

    except requests.RequestException as e:

        return f"天气 API 请求失败：{str(e)}"

    except (KeyError, IndexError, ValueError) as e:

        return f"天气数据解析失败：{str(e)}"

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气。

    参数：
        city:
            需要查询天气的城市名称。

    返回：
        str:
            天气查询结果。
    """

    return f"{city} 255度"
