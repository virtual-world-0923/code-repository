# ===================== 导入依赖库 =====================
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ===================== 配置参数（必须和训练代码保持一致） =====================
# 和训练代码完全相同的网络超参
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
# 训练时保存的模型文件路径
MODEL_PATH = r"mnist_fc_model.pth"
# 图像预处理必须和训练时一模一样！
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 推理设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 网络模型定义（和训练文件完全一致，加载模型必须有） =====================
class MNISTNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# ===================== 单张图片预测函数 =====================
def predict_digit(img_file_path: str):
    """
    输入手写数字图片路径，返回预测数字
    img_file_path：你自己手写数字png/jpg图片路径
    return：预测数字 int 0~9
    """
    # 如果训练时模型只保存了参数，调用时要重现训练模型的网络架构
    # 1. 初始化网络结构
    # model = MNISTNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    # # 2. 加载训练保存的权重参数
    # if not os.path.exists(MODEL_PATH):
    #     raise FileNotFoundError(f"模型文件不存在，请先运行train.py训练！路径：{MODEL_PATH}")
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    #如果训练时模型保存了参数+架构，调用时直接加载模型使用
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False).to(device) #weights_only=False  关闭安全机制。默认开启，不能加载完整的本地模型

    # 3. 模型设为评估模式，关闭梯度、dropout等训练层
    model.eval()

    # 4. 打开图片并预处理
    img = Image.open(img_file_path)
    img_tensor = transform(img)
    # 扩充batch维度 [1,1,28,28]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 5. 无梯度推理预测
    with torch.no_grad():
        output = model(img_tensor)
        pred_num = torch.argmax(output, dim=1).item()
    return pred_num

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 修改这里为你自己手写数字图片的路径
    test_image_path = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/843.png"

    try:
        result = predict_digit(test_image_path)
        print(f"图片 {test_image_path} 预测结果数字：{result}")
    except FileNotFoundError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"预测出错：{e}")