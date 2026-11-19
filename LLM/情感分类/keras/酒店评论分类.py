import keras
import runpy
import os

# 固定随机种子，保证可复现
keras.utils.set_random_seed(42)

######################## 数据处理 #############################################
result = runpy.run_path("数据处理.py")
X_train_int = result["X_train_int"]
X_test_int = result["X_test_int"]
y_train = result["y_train"]
y_test = result["y_test"]
vectorize_layer = result["vectorize_layer"]

######################## 主干网络 #############################################
result = runpy.run_path("主干网络.py")
model = result["model"]

######################## 训练模型 #############################################
result = runpy.run_path("训练模型.py")
train_model = result["train_model"]
combined_model, history = train_model(model, X_train_int, y_train)

######################## 保存模型 #############################################
MODEL_PATH = "sentiment_rnn.keras"
combined_model.save(MODEL_PATH)
print(f"模型已保存 → {os.path.abspath(MODEL_PATH)}")
print("-" * 60)

######################## 评估模型 #############################################
result = runpy.run_path("评估模型.py")
evaluate_model = result["evaluate_model"]
evaluate_model(model, X_train_int, y_train, X_test_int, y_test)

######################## 预测模型 #############################################
result = runpy.run_path("预测模型.py")
predicted_model = result["predicted_model"]
predicted_model(MODEL_PATH,vectorize_layer)