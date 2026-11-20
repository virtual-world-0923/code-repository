'''
这个工具相对简单，主要学习如何定义带默认参数的 Tool。
'''


"""
文件：tools/time_tool.py

功能：获取指定时区的当前时间
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ==================================================
# 1. 定义工具参数
# ==================================================

class TimeInput(BaseModel):

    timezone: str = Field(
        default="Asia/Taipei",
        description=(
            "需要查询的时区，"
            "例如 Asia/Taipei、Asia/Shanghai、UTC"
        )
    )


# ==================================================
# 2. 创建时间工具
# ==================================================

def get_current_time1(timezone: str = "Asia/Taipei") -> str:
    """
    获取当前日期和时间。

    当用户询问现在几点、今天日期、
    当前星期几时调用。

    参数：
        timezone:
            时区名称，默认为 Asia/Taipei。

    返回：
        str:
            当前日期、时间和星期。
    """

    try:

        # 根据时区名称创建时区对象
        tz = ZoneInfo(timezone)

        # 获取指定时区的当前时间
        now = datetime.now(tz)

        # 星期映射
        weekdays = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日"
        ]

        weekday = weekdays[now.weekday()]

        return (
            f"时区：{timezone}\n"
            f"日期：{now.strftime('%Y-%m-%d')}\n"
            f"时间：{now.strftime('%H:%M:%S')}\n"
            f"星期：{weekday}"
        )

    except ZoneInfoNotFoundError:

        return f"不支持的时区：{timezone}"

@tool(args_schema=TimeInput,description="需要查询的时区")
def get_current_time(timezone: str = "Asia/Taipei") -> str:
    return f"{timezone} 2026-11-20 19:15:07"