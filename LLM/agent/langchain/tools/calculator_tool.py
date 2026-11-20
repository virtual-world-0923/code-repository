'''
这个工具的作用是让大模型能够执行真正的数学计算，而不是让大模型自己估算结果。

我们使用 Python AST 解析表达式，不使用不安全的 eval()。
'''


"""
文件：tools/calculator_tool.py

功能：数学计算工具
"""

import ast
import operator

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ==================================================
# 1. 定义工具参数
# ==================================================

class CalculatorInput(BaseModel):

    expression: str = Field(
        description=(
            "需要计算的数学表达式，"
            "例如 128 * 256、100 / 5 + 20"
        )
    )


# ==================================================
# 2. 定义允许使用的运算符
# ==================================================

OPERATORS = {

    ast.Add: operator.add,       # 加法 +

    ast.Sub: operator.sub,       # 减法 -

    ast.Mult: operator.mul,      # 乘法 *

    ast.Div: operator.truediv,   # 除法 /

    ast.Pow: operator.pow,       # 幂运算 **

    ast.Mod: operator.mod,       # 取模 %

    ast.USub: operator.neg,      # 负号 -

    ast.UAdd: operator.pos       # 正号 +
}


# ==================================================
# 3. 递归解析数学表达式
# ==================================================

def safe_calculate(node):
    """
    解析 AST 节点并执行数学计算。

    参数：
        node:
            Python AST 语法树节点。

    返回：
        计算结果。
    """

    # ------------------------------------------
    # 情况一：数字节点
    # ------------------------------------------

    if isinstance(node, ast.Constant):

        # 只允许整数和浮点数
        if type(node.value) not in (int, float):

            raise ValueError("只允许数字")

        return node.value


    # ------------------------------------------
    # 情况二：二元运算
    # ------------------------------------------

    if isinstance(node, ast.BinOp):

        # 获取运算符类型
        op_type = type(node.op)

        # 检查是否属于允许的运算符
        if op_type not in OPERATORS:

            raise ValueError("不支持的运算符")

        # 递归计算左边表达式
        left = safe_calculate(node.left)

        # 递归计算右边表达式
        right = safe_calculate(node.right)

        # 限制幂运算指数
        if op_type is ast.Pow and abs(right) > 10:

            raise ValueError("指数不能超过10")

        # 执行对应的数学运算
        result = OPERATORS[op_type](left, right)

        # 限制计算结果
        if isinstance(result, (int, float)):

            if abs(result) > 10**15:

                raise ValueError("计算结果超出允许范围")

        return result


    # ------------------------------------------
    # 情况三：一元运算
    # ------------------------------------------

    if isinstance(node, ast.UnaryOp):

        op_type = type(node.op)

        if op_type not in OPERATORS:

            raise ValueError("不支持的一元运算符")

        value = safe_calculate(node.operand)

        return OPERATORS[op_type](value)


    # ------------------------------------------
    # 其他节点不允许执行
    # ------------------------------------------

    raise ValueError("不允许的数学表达式")


# ==================================================
# 4. 创建 LangChain Tool
# ==================================================

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    执行数学计算。

    支持：
        加法 +
        减法 -
        乘法 *
        除法 /
        幂运算 **
        取模 %

    当用户需要数学计算时调用。

    参数：
        expression:
            数学表达式。

    返回：
        str:
            计算结果。
    """

    try:

        # 限制表达式长度
        if len(expression) > 100:

            raise ValueError("表达式长度不能超过100个字符")

        # 将字符串解析成 AST
        tree = ast.parse(
            expression,
            mode="eval"
        )

        # 限制语法树节点数量
        if sum(1 for _ in ast.walk(tree)) > 50:

            raise ValueError("表达式过于复杂")

        # 获取表达式主体并执行计算
        result = safe_calculate(tree.body)

        return str(result)

    except Exception as e:

        return f"计算失败：{str(e)}"
