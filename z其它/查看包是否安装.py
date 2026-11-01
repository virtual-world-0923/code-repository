from importlib.metadata import version, PackageNotFoundError

requirements = r"C:\Repositories\basic_algorithm-ipynb\RAG\重排_筛选\requirements.txt"
def check(requirements_file=requirements):
    for raw in open(requirements_file, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()        # 去掉行内注释
        if not line:
            continue
        name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
        try:
            print(f"[OK]   {name:25s} {version(name)}")
        except PackageNotFoundError:
            print(f"[MISS] {name:25s} 未安装")

if __name__ == "__main__":
    check()