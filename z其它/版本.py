import sys
import os
import psutil
import torch

print("Python版本:", sys.version) # 查看 Python 版本
print("安装位置:", sys.executable) # 查看 Python 安装路径

# 1. 检测算力平台来源
if 'KAGGLE_URL_BASE' in os.environ or os.path.exists('/kaggle'):
    print("🚀 算力来源：【 Kaggle 托管算力 】")
elif 'COLAB_GPU' in os.environ or os.path.exists('/content'):
    print("🛸 算力来源：【 Google Colab 原生算力 】")
else:
    print("💻 算力来源：【 本地电脑 / 其他环境 】")

# 2. 检测计算核心（CPU 还是 GPU）
if torch.cuda.is_available():
    print("🔥 计算核心：【 GPU 加速模式 】")
    print(f"   显卡型号：{torch.cuda.get_device_name(0)}")
    print(f"   显存总量：{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("❄️ 计算核心：【 CPU 模式 】")

# 3. 查看系统磁盘容量
total_disk = psutil.disk_usage('/').total / (1024**3)
print(f"📦 磁盘容量：系统总空间 {total_disk:.2f} GB")
print("="*74)
