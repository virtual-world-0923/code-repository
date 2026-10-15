# ===================== 导入依赖库 =====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# ===================== 全局路径与超参数配置 =====================
# 训练集路径
TRAIN_IMG_DIR = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/train_sampled/images"
TRAIN_CSV_PATH = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/train_sampled/labels_summary.csv"
# 测试集路径
TEST_IMG_DIR = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/test_sampled/images"
TEST_CSV_PATH = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/test_sampled/labels_summary.csv"
# 模型保存路径
MODEL_SAVE_PATH = r"mnist_fc_model.pth"

# 训练超参
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

# 图像预处理流水线
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ===================== 自定义本地数据集类 =====================
class MNISTLocalDataset(Dataset):
    """
    读取本地png图片+csv标签文件
    img_dir：图片文件夹路径
    csv_path：标签csv文件路径
    transform：图像预处理操作
    """
    def __init__(self, img_dir: str, csv_path: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        img_name = f"{str(row['Original_Index'])}.png"
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row["Label"]), dtype=torch.long)
        return img, label

# ===================== 全连接神经网络模型 =====================
class MNISTNet(nn.Module):
    """两层隐藏层全连接手写数字识别网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [batch,1,28,28] -> [batch,784]
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# ===================== 单轮训练函数 =====================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # 梯度清零，每个batch必须执行
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += torch.sum(preds == labels).item()
        total += imgs.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# ===================== 测试评估函数 =====================
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total += imgs.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# ===================== 主训练入口 =====================
if __name__ == "__main__":
    print(f"训练设备：{device}")
    # 构建数据集与加载器
    train_ds = MNISTLocalDataset(TRAIN_IMG_DIR, TRAIN_CSV_PATH, transform)
    test_ds = MNISTLocalDataset(TEST_IMG_DIR, TEST_CSV_PATH, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集样本数：{len(train_ds)}，测试集样本数：{len(test_ds)}")

    # 初始化模型、损失、优化器
    model = MNISTNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 循环训练
    print("\n===== 开始训练 =====")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"第{epoch:2d}轮 | 训练损失:{train_loss:.4f} 训练准确率:{train_acc:.4f} | 测试损失:{test_loss:.4f} 测试准确率:{test_acc:.4f}")
    print("===== 训练完成 =====")

    # ========== 保存训练好的模型权重 ==========
    # 方式：仅保存模型参数（推荐，体积更小、加载更灵活）
    # torch.save(model.state_dict(), MODEL_SAVE_PATH) #只保留参数，调用时要创建和训练一样的架构
    torch.save(model, MODEL_SAVE_PATH) #保存全部模型（参数+架构）
    print(f"模型已保存至：{MODEL_SAVE_PATH}")