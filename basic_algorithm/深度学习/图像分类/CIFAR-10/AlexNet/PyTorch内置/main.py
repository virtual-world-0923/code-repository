import runpy
from torchvision import models
import torch
import os

#数据处理
result = runpy.run_path("数据处理.py")
train_loader = result["train_loader"]
test_loader = result["test_loader"]

#网络搭建
result = runpy.run_path("网络搭建.py")
model = result["model"]

#训练模型
result = runpy.run_path("训练模型.py")
criterion = result["criterion"] #获取模块级变量
train_model = result["train_model"] #获取模块级函数

print("\n" + "=" * 70)                                    # 打印大分隔线
print("           开始训练AlexNet模型")      # 打印训练开始标题
print("=" * 70)                                          # 打印分隔线

# 调用训练函数，传入所有必要的参数
trained_model, train_losses, train_accs = train_model(   # 调用训练函数，接收训练后的模型及训练历史数据
    model=model,                                         # 传入已修改为10分类的AlexNet模型
    train_loader=train_loader,                           # 传入训练数据加载器
    num_epochs=1                                        # 设置训练总轮数为j几个epoch
)

#评估模型
from 评估模型 import evaluate_model
# 调用评估函数获取测试损失和准确率
test_loss, test_acc = evaluate_model(  # 调用评估函数
    model=trained_model,               # 传入训练完成后的模型
    test_loader=test_loader,           # 传入测试数据加载器
    criterion = criterion              # 传入损失函数
)

#保存模型
# 定义模型保存路径
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(DATA_DIR, "alexnet_cifar10_trained.pth")  # 拼接模型保存的完整文件路径

# 保存模型的状态字典（state_dict），只保存模型参数，不保存模型结构
torch.save(trained_model.state_dict(), save_path)

#预测模型
from 预测模型 import predicted_model
# 创建与保存时结构相同的模型
alexnet_model = models.alexnet(weights=None)
alexnet_model.classifier[6] = torch.nn.Linear(4096, 10)  # 和训练时一样的修改

# 加载权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet_model.load_state_dict(torch.load(save_path, map_location=device))

#取出一条数据
sample_iter = iter(test_loader)
sample_images, sample_labels = next(sample_iter)
sample_images = sample_images[0:1]
# sample_images = sample_images[0].unsqueeze(0)
#进行预测
highest_score, class_index = predicted_model(alexnet_model, sample_images)
print(highest_score)
print(class_index)