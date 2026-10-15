# 模型结构必须一致：PyTorch 的 .pth 文件只保存了权重（参数），不保存网络结构。因此加载时必须重新定义与训练时一模一样的 SimpleNN 类。
# 设备映射 (map_location)：如果模型是在 GPU 上训练的，但当前运行环境没有 GPU，必须加上 map_location=device，否则会报找不到 CUDA 设备的错误。
# 评估模式 (model.eval())：预测前务必调用 model.eval()，这会让模型内部的 Dropout 等层停止随机丢弃神经元，保证每次预测结果一致。
# 批次维度 (unsqueeze(0))：神经网络默认接收的是批量数据 [Batch_Size, Channel, Height, Width]，单张图片预测时必须手动增加一个批次维度。

# ========================== 1. 导入必要的库 ==========================
import torch  # 导入 PyTorch 核心库，用于加载模型和处理张量
from torchvision import transforms  # 导入图像变换模块，用于对预测图片进行预处理
from PIL import Image  # 导入图像处理库，用于打开本地的 PNG 图片


# ========================== 2. 重新定义模型结构（必须与训练时完全一致） ==========================
# 注意：加载模型时，PyTorch 只保存了权重参数，没有保存网络结构，因此必须重新定义模型类
class SimpleNN(torch.nn.Module):
    """
    简单全连接神经网络（与训练时的结构完全相同）
    """

    def __init__(self):
        super(SimpleNN, self).__init__()  # 调用父类初始化方法
        self.flatten = torch.nn.Flatten()  # 展平层：将二维图像展平为一维向量
        self.fc1 = torch.nn.Linear(28 * 28, 128)  # 第一层全连接层：784 -> 128
        self.fc2 = torch.nn.Linear(128, 128)  # 第二层全连接层：128 -> 128
        self.fc3 = torch.nn.Linear(128, 10)  # 输出层：128 -> 10（对应0-9十个数字）
        self.relu = torch.nn.Tanh()  # Tanh 激活函数

    def forward(self, x):
        """
        定义前向传播过程
        参数:
            x (Tensor): 输入图像张量
        返回:
            logits (Tensor): 模型输出的原始预测值
        """
        x = self.flatten(x)  # 将图像展平
        x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = self.relu(self.fc2(x))  # 第二层全连接 + ReLU 激活
        logits = self.fc3(x)  # 输出层，得到10个类别的得分
        return logits  # 返回预测结果


# ========================== 3. 加载模型权重 ==========================
# 自动检测当前可用的计算设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 实例化一个空的网络结构
model = SimpleNN()
# 加载之前训练保存的模型权重字典（state_dict）
# map_location=device 确保模型加载到正确的设备上，避免 GPU 训练 CPU 加载时报错
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
# 将模型移动到计算设备（GPU/CPU）上
model.to(device)
# 将模型切换为评估模式（关闭 Dropout 等训练专属操作，确保预测结果稳定）
model.eval()

print(f"模型加载成功，当前运行在设备: {device}")  # 打印模型加载状态和设备信息


# ========================== 4. 定义单张图片预测函数 ==========================
def predict_single_image(model, image_path, device):
    """
    使用加载好的模型预测一张外部 PNG 图片的数字类别
    参数:
        model (nn.Module): 加载好权重的神经网络模型
        image_path (str): 待预测图片的完整路径
        device (torch.device): 计算设备（GPU/CPU）
    返回:
        predicted_label (int): 预测出的数字 (0-9)
    """
    # 定义单张图片的预处理流程：转灰度 -> 转张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图片是单通道灰度图
        transforms.ToTensor(),  # 将 PIL 图片转为 Tensor，并自动归一化到 [0,1]
    ])

    image = Image.open(image_path)  # 使用 PIL 打开指定路径的图片
    image = transform(image)  # 应用预处理规则
    # unsqueeze(0) 增加批次维度，将 [1, 28, 28] 变为 [1, 1, 28, 28]，并移至计算设备
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():  # 上下文管理器：预测时禁用梯度计算，节省显存并加速推理
        output = model(image)  # 前向传播，将图片输入模型
        _, predicted = torch.max(output, 1)  # 获取输出中概率最大的类别索引
        predicted_label = predicted.item()  # 将张量转为普通的 Python 整数

    print(f"图片 '{image_path}' 的预测数字是: {predicted_label}")  # 打印预测结果
    return predicted_label  # 返回预测的数字


# ========================== 5. 执行预测 ==========================
# 请替换为您本地实际存在的图片路径（建议图片尺寸为 28x28 像素）
img_path = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/843.png"
predict_single_image(model, img_path, device)