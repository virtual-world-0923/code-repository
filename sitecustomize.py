import builtins

try:
    from rich import print as rprint
    from rich.pretty import pprint as rpprint
    from rich.traceback import install

    # 1. 全局替换内置 print
    builtins.print = rprint

    # 2. 注册全局内置 pprint（任意文件无需 import 即可直接调用）
    builtins.pprint = rpprint

    # 3. 全局美化异常报错信息（show_locals=False 避免打印过多冗余局部变量）
    install(show_locals=False)
except ImportError:
    # 若当前环境未安装 rich，静默跳过，保证环境正常启动
    pass
    
from dotenv import load_dotenv, find_dotenv

# 启动任何 python 脚本前自动寻找并加载 .env
env_file = find_dotenv(usecwd=True)
if env_file:
    load_dotenv(env_file, override=True)
