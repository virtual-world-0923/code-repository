
def evaluate_model(model, x_train, y_train, x_test, y_test):

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss,  test_acc  = model.evaluate(x_test,  y_test,  verbose=0)

    print(f"训练集精度 (accuracy): {train_acc:.4f}   loss: {train_loss:.4f}")
    print(f"测试集精度 (accuracy): {test_acc:.4f}   loss: {test_loss:.4f}")
    print("-" * 60)