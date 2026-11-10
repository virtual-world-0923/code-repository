import sys

def my_decorator(func):
    func_name = sys._getframe().f_code.co_name
    print(f"当前执行的函数是: {func_name}")
    def wrapper(q,w):
        func_name = sys._getframe().f_code.co_name
        print(f"当前执行的函数是: {func_name}")
        
        # 执行原函数，并保存返回值
        result = func(q, w) 
        
        print(f"{func_name}函数调用结束")
        return result  # 返回原函数的返回值

    print(f"{func_name}函数调用结束")
    return wrapper

@my_decorator
def add(a, b):
    func_name = sys._getframe().f_code.co_name
    print(f"当前执行的函数是: {func_name}")
    print("正在计算加法...")
    c = a + b
    print(f"{func_name}函数调用结束")
    return c

func_name = sys._getframe().f_code.co_name
print(f"当前执行的函数是: {func_name}")
# 调用
sum_result = add(3, 5)
print(f"结果是: {sum_result}")
print(f"{func_name}函数调用结束")