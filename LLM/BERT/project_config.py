import random                  # 随机数：用于设置随机种子保证可复现
import numpy as np             # 数值计算库：用于设置随机种子
import torch                   # PyTorch 深度学习框架：张量计算与自动求导

from transformers.trainer_utils import set_seed  # 设置 transformers 全局随机种子

# -------------------- 模型与数据路径 --------------------
# 方式 A/B 使用的模型：谷歌官方中文 BERT（字级别分词，12 层）
MODEL_NAME_BERT = "bert-base-chinese"
# 方式 C 使用的模型：哈工大中文 RoBERTa-wwm-ext（整词掩码，效果更好）
MODEL_NAME_ROBERTA = "hfl/chinese-roberta-wwm-ext"

# 原始数据集 CSV 路径（逗号分隔，label,review）
CSV_PATH = r"C:\Repositories\basic_algorithm\database\ChnSentiCorp_htl_all.csv"

# 模型保存目录（不同方式保存到不同子目录，互不覆盖）
SAVE_DIR_BERT = r"C:\Repositories\basic_algorithm\深度学习\BERT\models\bert_dir_sentiment"           # 方式 A
SAVE_TRAINER_BERT = r"C:\Repositories\basic_algorithm\深度学习\BERT\models\bert_trainer_sentiment"   # 方式 B
SAVE_DIR_ROBERTA = r"C:\Repositories\basic_algorithm\深度学习\BERT\models\roberta_wwm_sentiment"  # 方式 C

# -------------------- 训练超参数 --------------------
MAX_LEN    = 128    # 每条评论最大 token 长度；超过截断，不足补 pad
BATCH_SIZE = 16     # 每个 batch 的样本数（显存不够可改 16）
LR         = 2e-5   # AdamW 初始学习率（BERT 微调经典值）
EPOCHS    = 1      # 训练轮数
WEIGHT_DECAY = 0.01 # AdamW 权重衰减（L2 正则）
WARMUP_STEPS = 100 #
SEED       = 42     # 随机种子，保证可复现

# -------------------- 设备选择 --------------------
# 优先使用 GPU（CUDA），否则回退到 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. 设置随机种子（保证每次运行结果一致）
# ==============================================================================
set_seed(SEED)              # transformers 内部使用的种子
random.seed(SEED)           # Python 内置随机
np.random.seed(SEED)        # numpy 随机
torch.manual_seed(SEED)     # PyTorch CPU 随机
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  # PyTorch 所有 GPU 随机
