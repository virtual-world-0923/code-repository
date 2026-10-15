import keras
import runpy
import os
import torch

# 固定随机种子，保证可复现
keras.utils.set_random_seed(42)

######################## 数据处理 #############################################
result = runpy.run_path("数据处理.py")
train_loader = result["train_loader"]
test_loader = result["test_loader"]
fenci = result["fenci"]

######################## 主干网络 #############################################
result = runpy.run_path("主干网络.py")
backbone_network = result["backbone_network"]
model = backbone_network(len(fenci["vocab"]), fenci["word_to_idx"]["<PAD>"])

######################## 训练模型 #############################################
result = runpy.run_path("训练模型.py")
train_model = result["train_model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_model(model, train_loader, device)

######################## 评估模型 #############################################
result = runpy.run_path("评估模型.py")
evaluate_model = result["evaluate_model"]
train_precision = evaluate_model(model, train_loader, device)
test_precision = evaluate_model(model, test_loader, device)
print("训练精度：{train_precision}")
print("测试精度：{test_precision}")

######################## 保存模型 #############################################
MODEL_PATH = "sentiment_rnn"
needed = {
    "model_state_dict": model.state_dict(),
    "fenci":fenci
}
# 保存模型参数
torch.save(needed, MODEL_PATH)
print(f"模型已保存 → {os.path.abspath(MODEL_PATH)}")
print("-" * 60)

######################## 预测模型 #############################################

# result = runpy.run_path("预测模型.py")
# predicted_model = result["predicted_model"]
# predicted_model(MODEL_PATH)



