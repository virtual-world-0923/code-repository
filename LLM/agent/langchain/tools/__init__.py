"""
文件：tools/__init__.py

功能：统一管理所有 LangChain 工具
"""

"""
文件：tools/__init__.py

功能：统一管理所有 LangChain 工具
"""


# 导入数学计算工具
from .calculator_tool import calculator

# 导入时间工具
from .time_tool import get_current_time

# 导入天气工具
from .weather_tool import get_weather

# 统一工具列表
tools = [get_weather, get_current_time]


# 根据工具名称建立映射关系
tool_map = {tool.name: tool for tool in tools}
