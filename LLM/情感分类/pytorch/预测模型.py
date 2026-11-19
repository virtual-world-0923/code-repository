import jieba
import torch
import runpy

######################## 主干网络 #############################################
result = runpy.run_path("主干网络.py")
backbone_network = result["backbone_network"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process(text, word_to_idx):
    tokens = list(jieba.cut(text.strip()))
    wi = [word_to_idx.get(char, word_to_idx['<UNK>']) for char in tokens]
    return wi

def predicted_model(MODEL_PATH):
    needed = torch.load(MODEL_PATH, map_location=device)
    model_state_dict = needed["model_state_dict"]
    fenci = needed["fenci"]
    vocab_size = len(fenci["vocab"])
    word_to_idx = fenci["word_to_idx"]
    pad_idx = word_to_idx["<PAD>"]
    model = backbone_network(vocab_size, pad_idx)
    model.load_state_dict(model_state_dict)
    model.eval()

    t1 = "这家酒店环境优雅，服务态度非常好，房间干净舒适，下次还会再来！"
    t2 = "房间太小，隔音差，服务态度恶劣，不会再入住。"
    wi = process(t2, word_to_idx)
    tensor = torch.tensor(wi, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(tensor, "模型预测").item()

    sentiment = "满意 ✅" if prob >= 0.5 else "不满意 ❌"
    print(f"评论: {t1}")
    print(f"预测: {sentiment} (正向置信度: {prob:.4f})\n")

MODEL_PATH = "sentiment_rnn"
predicted_model(MODEL_PATH)