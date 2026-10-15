import sklearn
import numpy
import pandas
import torch
import tensorflow as tf
import keras
import sys

print(sys.executable)
print(sys.version)
print(sys.path)
print(sys.version)

print("="*100)

print("scikit-learn:", sklearn.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)

print("="*100)

print("PyTorch:", torch.__version__)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("Keras Active backend:", keras.backend.backend())
r"""
keras默认调用TensorFlow,修改方法如下：
1、修改环境变量
    $env:KERAS_BACKEND="torch"
2、代码中设置（必须在 import keras 之前）
    import os
    os.environ["KERAS_BACKEND"] = "torch"   # 可选: "tensorflow", "jax"
    import keras
3、修改配置文件
    编辑 ~/.keras/keras.json（Windows 在 C:\Users\<用户名>\.keras\keras.json）
    {
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
    }
"""
print("-"*100)

print("-"*50 + "显卡" + "-"*50)

import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout[:500])
except:
    print("nvidia-smi 不可用")

print("-"*50 + "torch 是否支持 GPU" + "-"*50)
#是否支持CUDA显卡
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("torch cuda: {torch.version.cuda}")
    print(f"显卡名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")


print("-"*50 + "TensorFlow 是否支持 GPU" + "-"*50)
#2.10以后不支持Windows下GPU训练，需要使用 WSL2 代替
# 2. 列出所有 GPU 设备
gpus = tf.config.list_physical_devices('GPU')
print("GPU 设备:", gpus)

if gpus:
    print(f"✅ 找到 {len(gpus)} 个 GPU，GPU 加速可用！")
    for gpu in gpus:
        print(f"   - {gpu.name}")
else:
    print("❌ 未检测到 GPU，将使用 CPU 运行")

# 3. 查看构建信息（是否包含 CUDA 支持）
print(tf.sysconfig.get_build_info())

print("-"*100)

import torch, torchvision
print(torch.__version__)
print(torch.cuda.is_available())
print(torchvision.__version__)
print(torch.ops.torchvision.nms)
print("-"*100)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))












