import os
import hashlib
from collections import Counter
from pathlib import Path

def get_dir_summary(dir_path):
    """统计目录下的文件类型数量及所有相对路径"""
    dir_path = Path(dir_path)
    extensions = []
    file_set = set()
    
    for root, _, files in os.walk(dir_path):
        for file in files:
            full_path = Path(root) / file
            # 获取相对于根目录的相对路径，用于后续对比
            rel_path = full_path.relative_to(dir_path)
            file_set.add(rel_path)
            
            # 统计后缀名（统一转小写）
            ext = full_path.suffix.lower()
            if ext == '':
                extensions.append('无后缀文件')
            else:
                extensions.append(ext)
                
    return file_set, Counter(extensions)

def calculate_md5(file_path):
    """计算文件的 MD5 值（适用于任何文件，包括 md 文本）"""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # 分块读取，防止大文件内存溢出
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return f"Error_{e}"

def compare_directories(dir1, dir2):
    path1, path2 = Path(dir1), Path(dir2)
    
    if not path1.exists() or not path2.exists():
        print("❌ 错误：其中一个或两个目录路径不存在！")
        return

    print("=" * 50)
    print("【步骤 1：统计目录文件类型】")
    files1, ext_count1 = get_dir_summary(path1)
    files2, ext_count2 = get_dir_summary(path2)
    
    print(f"\n📂 目录 A ({path1.name}) 的文件类型统计：")
    for ext, count in ext_count1.items():
        print(f"  - {ext}: {count} 个")
        
    print(f"\n📂 目录 B ({path2.name}) 的文件类型统计：")
    for ext, count in ext_count2.items():
        print(f"  - {ext}: {count} 个")
    print("=" * 50)

    print("【步骤 2：正在对比文件内容...】")
    all_relative_paths = files1.union(files2)
    
    only_in_a = []
    only_in_b = []
    content_mismatch = []
    
    for rel_path in all_relative_paths:
        file_a = path1 / rel_path
        file_b = path2 / rel_path
        
        # 1. 检查文件是否存在
        if not file_a.exists():
            only_in_b.append(rel_path)
            continue
        if not file_b.exists():
            only_in_a.append(rel_path)
            continue
            
        # 2. 文件都存在，进行大小和内容对比（包含 .md 和其他类型）
        if file_a.stat().st_size != file_b.stat().st_size:
            content_mismatch.append(rel_path)
        else:
            # 大小一致时，通过 MD5 确保内容绝对一致
            if calculate_md5(file_a) != calculate_md5(file_b):
                content_mismatch.append(rel_path)

    print("\n【步骤 3：对比结果报告】")
    
    # 判断是否完全一致
    is_perfect_match = not only_in_a and not only_in_b and not content_mismatch
    
    if is_perfect_match:
        print("✨ 结果：两个目录的文件以及内容【完全一致】！")
    else:
        print("⚠️ 结果：两个目录【不一致】，差异如下：")
        
        if only_in_a:
            print(f"\n🛑 只在目录 A ({path1.name}) 中存在的文件：")
            for f in only_in_a:
                print(f"  - {f}")
                
        if only_in_b:
            print(f"\n🛑 只在目录 B ({path2.name}) 中存在的文件：")
            for f in only_in_b:
                print(f"  - {f}")
                
        if content_mismatch:
            print(f"\n❌ 两个目录都存在，但【内容不一致】的文件（包含 md 文本）：")
            for f in content_mismatch:
                print(f"  - {f}")
    print("=" * 50)

# 💡 使用示例：在这里输入你的两个目录路径
if __name__ == "__main__":
    # 请替换为你的实际目录路径
    dir_a = r"C:\Users\Administrator\Desktop\123\ben"
    dir_b = r"C:\Users\Administrator\Desktop\123\yun"
    dir_c = r"C:\Users\Administrator\Desktop\pd\yun_ArcanumPalimpsest_backup"
    
    compare_directories(dir_a, dir_b)
