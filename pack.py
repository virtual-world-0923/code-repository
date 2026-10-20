import os
import zipfile
from pathlib import Path

# ===== 配置区 =====
PROJECT_ROOT = Path(__file__).resolve().parent  # 项目根目录（脚本所在目录）
OUTPUT_ZIP = PROJECT_ROOT / "project_py_ipynb.zip"  # 输出 zip 路径

INCLUDE_EXTS = {".py", ".ipynb"}  # 只打包这两类文件

# 需要排除的目录名（任意层级匹配）
EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode",
    "__pycache__", ".ipynb_checkpoints",
    "venv", ".venv", "env", ".env",
    "node_modules", "dist", "build",
}


# =================


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in EXCLUDE_DIRS


def main(keep_structure: bool = True):
    """
    全平台标准 UTF-8 打包函数（完美兼容 Windows/Drive/Colab 中文显示）
    """
    count = 0
    existing_arcnames = set()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # 原地修改 dirs，跳过排除目录
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]

            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in INCLUDE_EXTS:
                    continue

                if keep_structure:
                    # 【保留层级】
                    arcname_obj = fpath.relative_to(PROJECT_ROOT)
                    arcname_str = str(arcname_obj)
                else:
                    # 【不保留层级】
                    base_name = fpath.stem
                    suffix = fpath.suffix
                    proposed_name = f"{base_name}{suffix}"

                    version = 1
                    while proposed_name in existing_arcnames:
                        proposed_name = f"{base_name}_{version}{suffix}"
                        version += 1

                    arcname_str = proposed_name
                    existing_arcnames.add(proposed_name)

                # ==================== ✨ 核心修复：全平台标准 UTF-8 编码 ====================
                # 1. 统一路径分隔符为正斜杠 '/'
                arcname_str = arcname_str.replace(os.path.sep, '/')

                # 2. 创建标准的 ZipInfo 对象
                zinfo = zipfile.ZipInfo(filename=arcname_str)
                zinfo.compress_type = zipfile.ZIP_DEFLATED

                # 3. 💥 最关键的一步：开启 Flag 11 (通用标志位第11位)，通知解压软件“此文件名是 UTF-8 编码”
                # 这样不管是 Windows 的 WinRAR、7-Zip，还是 Linux/Colab 都会强行用 UTF-8 解析，绝无乱码
                zinfo.flag_bits |= 0x800

                # 4. 读取原文件并写入
                with open(fpath, 'rb') as f:
                    zf.writestr(zinfo, f.read())
                # ============================================================================

                count += 1
                print(f"added: {arcname_str}")

    print(f"\n打包完成：{OUTPUT_ZIP}")
    print(f"共打包 {count} 个文件")


if __name__ == "__main__":
    # 依然可以控制是否平铺（False 为平铺，True 为保留目录结构）
    main(keep_structure=False)
