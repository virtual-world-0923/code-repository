def train_model(combined_model, x_train,y_train, x_test,y_test,BATCH_SIZE,EPOCHS):
    combined_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    history = combined_model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )
    # model.summary(): 打印模型的详细结构摘要表格
    combined_model.summary(show_trainable=True, expand_nested=True)
    return history,combined_model