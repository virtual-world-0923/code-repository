from tqdm import tqdm
import torch
import torch.nn as nn

def train_model(model, train_loader, device):

    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 训练循环
    EPOCHS = 1
    best_acc = 0
    for epoch in range(EPOCHS):
        running_loss = 0.0

        #tqdm给循环加一个进度条，desc 是进度条前面显示的提示文字
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs,"给自定义model的forward传参实验,训练模型")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
