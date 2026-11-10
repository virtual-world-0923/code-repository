import sys

def a(num_times):
    code_obj = sys._getframe().f_code
    print(f"本函数名称: {code_obj.co_name}")

    print(f"动态运行时---hex(id(code_obj)): {hex(id(code_obj))}")
    print(f"函数地址---hex(id(a))：{hex(id(a))}")
    print(f"代码地址---hex(id(a.__code__)): {hex(id(a.__code__))}")

    print(f"参数：{num_times}")

    def b(func):
        code_obj = sys._getframe().f_code
        print(f"本函数名称: {code_obj.co_name}")

        print(f"动态运行时---hex(id(code_obj)): {hex(id(code_obj))}")
        print(f"函数地址---hex(id(b))：{hex(id(b))}")
        print(f"代码地址---hex(id(b.__code__)): {hex(id(b.__code__))}")

        print(f"参数：{func}")
        print(f"传入的函数名称是: {func.__name__}")
        print(f"函数地址---hex(id(func))：{hex(id(func))}")
        print(f"代码地址---hex(id(func.__code__)): {hex(id(func.__code__))}")

        def c(*args, **kwargs):
            code_obj = sys._getframe().f_code
            print(f"本函数名称: {code_obj.co_name}")

            print(f"动态运行时---hex(id(code_obj)): {hex(id(code_obj))}")
            print(f"函数地址---hex(id(c))：{hex(id(c))}")
            print(f"代码地址---hex(id(c.__code__)): {hex(id(c.__code__))}")

            print(f"参数：{args},{kwargs}")

            for _ in range(num_times):
                print(f"c里调用func---hex(id(func))：{hex(id(func))}")
                print(f"c里调用func---hex(id(func.__code__))：{hex(id(func.__code__))}")
                result = func(*args, **kwargs)

            print(f"{code_obj.co_name}函数调用结束，返回{result}")
            return result

        print(f"{code_obj.co_name}函数调用结束，返回函数地址：{hex(id(c))}，返回代码地址: {hex(id(c.__code__))}")
        return c

    print(f"{code_obj.co_name}函数调用结束，返回函数地址：{hex(id(b))}，返回代码地址: {hex(id(b.__code__))}")
    return b

@a(num_times=1)
def greet(name):
    code_obj = sys._getframe().f_code
    print(f"本函数名称: {code_obj.co_name}")

    print(f"动态运行时---hex(id(code_obj)): {hex(id(code_obj))}")
    print(f"函数地址---hex(id(greet))：{hex(id(greet))}")
    print(f"代码地址---hex(id(greet.__code__)): {hex(id(greet.__code__))}")

    print(f"参数：{name}")

    print(f"Hello, {name}!")

    print(f"{code_obj.co_name}函数调用结束")


code_obj = sys._getframe().f_code
print(f"主函数名称: {code_obj.co_name}，主函数地址: {hex(id(code_obj))}")
print(f"主调用hex(id(greet)----{hex(id(greet))}")
print(f"主调用hex(id(greet.__code__)----{hex(id(greet.__code__))}")
greet("Alice")
print(f"{code_obj.co_name}主函数调用结束")