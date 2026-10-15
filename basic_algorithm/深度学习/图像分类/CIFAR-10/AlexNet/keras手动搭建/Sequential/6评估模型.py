
# 打印评估结果标题
print("\n" + "=" * 70)
print("模型评估结果:")
print("=" * 70)

def evaluate_model(model, x_train,y_train, x_test,y_test):
      # model.evaluate: 在给定数据集上计算模型的损失值和所有编译时指定的指标
      # 参数:
      #   x — 输入数据，形状(N, H, W, C)
      #   y — 真实标签，形状(N,)
      #   verbose=0 — 不输出进度条(评估速度快，无需进度显示)
      # 返回:
      #   list — [损失值, 指标1值, 指标2值, ...]
      #     这里返回 [loss, accuracy] 两个值
      # 解包赋值: 将列表中的两个元素分别赋给 train_loss 和 train_accuracy
      train_loss, train_accuracy = model.evaluate(x_train,y_train, verbose=0)

      # 打印训练集上的损失和准确率
      # :.4f — 格式化为保留4位小数的浮点数
      # :.2% — 格式化为保留2位小数的百分比(如 0.8523 → 85.23%)
      print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_accuracy:.4f} "
            f"({train_accuracy * 100:.2f}%)")

      # 在测试集上评估模型
      test_loss, test_accuracy = model.evaluate(x_test,y_test, verbose=0)

      # 打印测试集上的损失和准确率
      print(f"测试集 - 损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f} "
            f"({test_accuracy * 100:.2f}%)")

