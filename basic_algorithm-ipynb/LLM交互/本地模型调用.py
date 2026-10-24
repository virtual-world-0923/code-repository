# ============================================================
# 功能：在本地加载 Qwen2.5-1.5B-Instruct 对话模型，输入一段对话并返回模型的回复
# ============================================================

# AutoTokenizer：自动根据模型名称加载对应的分词器（文本 <-> token ID 互相转换）
# AutoModelForCausalLM：自动加载“因果语言模型”（从左到右逐个预测下一个 token，即常见的 GPT 式生成模型）
# BitsAndBytesConfig：bitsandbytes 库的量化配置类，用来做 4-bit / 8-bit 量化加载以节省显存
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch  # PyTorch，提供张量运算、数据类型（如 float16）和 GPU 支持


def call_local_model(messages, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    调用本地（或从 HuggingFace 自动下载到本地缓存）的大模型，生成一次回复。

    参数:
        messages (list[dict]):
            对话消息列表，格式为 [{"role": "...", "content": "..."}, ...]。
            role 常见取值：
              - "system"    ：系统提示词，设定模型的身份/行为
              - "user"      ：用户说的话
              - "assistant" ：模型之前的回复（多轮对话时使用）
        model_name (str):
            模型名称或本地路径。
            默认 "Qwen/Qwen2.5-1.5B-Instruct"（阿里通义千问 2.5，15 亿参数，指令微调版）。
            可以填 HuggingFace 仓库名，也可以填本地模型文件夹路径。

    返回:
        str: 模型生成的回复文本（不包含输入的提示词部分）。
    """

    # ---------- 1. 加载分词器 ----------
    # from_pretrained：从指定模型名/路径加载分词器及其配置
    # trust_remote_code=True：允许执行模型仓库里自带的自定义 Python 代码。
    #   某些模型需要它才能正确加载；注意：只对可信来源的模型开启，否则有安全风险。
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ---------- 2. 加载模型（float16 半精度） ----------
    # 使用 float16 精度加载模型：显存占用约为 float32 的一半，速度更快，精度损失很小
    model = AutoModelForCausalLM.from_pretrained(
        model_name,                # 模型名称或本地路径
        dtype=torch.float16,       # 权重加载的数据类型；float16 = 半精度。
                                   # （旧版 transformers 中此参数名为 torch_dtype，新版改为 dtype）
        device_map="auto",         # 自动分配模型放到哪些设备上（GPU / CPU），
                                   # 显存不够时会自动把部分层放到 CPU；需要安装 accelerate 库
        trust_remote_code=True,    # 同上，允许运行模型自带的自定义代码
    )

    # ---------- （可选）4-bit 量化加载方案：目前被注释，未启用 ----------
    # 如果显存不足，可以取消注释下面这段，用 4-bit 量化加载模型（显存占用大幅下降，精度略有损失）。
    # 注意：使用时要把上面“float16 加载”的那段代码注释掉，避免重复加载。

    # # 1. 定义 4-bit 量化配置
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,                       # 开启 4-bit 量化加载（权重用 4 位整数存储）
    #     bnb_4bit_compute_dtype=torch.float16,    # 实际计算时，把 4-bit 权重“反量化”回此精度来运算
    #                                              # （建议 fp16 或 bf16）
    #     bnb_4bit_quant_type="nf4",               # 量化格式：nf4（NormalFloat4）效果一般优于 fp4
    #     bnb_4bit_use_double_quant=True           # 双重量化：连量化参数本身也再量化一次，进一步省显存
    # )
    # # 使用 4-bit 精度加载模型
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=quant_config,        # 传入上面定义的量化配置
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    # ---------- 3. 检查分词器是否带有对话模板 ----------
    # getattr(对象, 属性名, 默认值)：安全读取属性，不存在则返回 None
    # chat_template 是一段 Jinja 模板，规定了如何把 messages 拼成模型认识的文本格式
    # 注：这里的 has_template 后面并没有被使用，仅作检查用途，可以删除
    has_template = (getattr(tokenizer, "chat_template", None) is not None)

    # ---------- 4. 把对话消息按模板转成文本 ----------
    text_prompt = tokenizer.apply_chat_template(
        messages,                       # 对话消息列表
        tokenize=False,                 # False：只返回拼好的文本字符串，不直接转成 token ID
        add_generation_prompt=True,     # True：在末尾追加“assistant 开始回答”的标记，
                                        # 提示模型接下来该轮到它回复了
    )
    # 对 Qwen 来说，结果大致形如：
    # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n自我介绍<|im_end|>\n<|im_start|>assistant\n

    # ---------- 5. 文本 -> token ID ----------
    # 调用分词器，把文本切分并映射成数字 ID
    input_ids = tokenizer(
        text_prompt,
        return_tensors="pt"             # "pt" = 返回 PyTorch 张量
    )
    # 返回对象是一个字典式结构，包含：
    #   input_ids      ：token ID 张量，形状 [batch, 序列长度]
    #   attention_mask ：注意力掩码，1 表示真实 token，0 表示填充（padding）

    # ---------- 6. 把数据和模型放到同一个设备上 ----------
    # 有 NVIDIA GPU 就用第 0 块显卡，否则退回 CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 把模型移动到该设备。
    # 注意：前面已用 device_map="auto" 由 accelerate 自动放置模型，这里再 .to(device) 通常是多余的，
    # 在多卡/显存不足需要分层卸载的情况下，甚至可能报错或破坏自动分配。
    model = model.to(device)
    # 输入张量也必须与模型在同一设备上，否则会报设备不一致的错误
    pure_input_ids = input_ids.input_ids.to(device)
    pure_attention_mask = input_ids.attention_mask.to(device)

    # ---------- 7. 生成回复 ----------
    model.eval()  # 切换到推理（评估）模式：关闭 Dropout 等仅训练时使用的随机行为
    with torch.no_grad():  # 不计算梯度：推理时不需要反向传播，可节省显存和时间
        output_ids = model.generate(
            input_ids=pure_input_ids,            # 输入 prompt 的 token ID
            attention_mask=pure_attention_mask,  # 注意力掩码，告诉模型哪些位置是有效 token
            num_return_sequences=1,              # 每个输入生成几条候选回复；这里只要 1 条
            temperature=0.1,                     # 温度：越低输出越确定、保守；越高越随机、有创意。
                                                 # ⚠️ 仅在 do_sample=True 时才生效；
                                                 # 默认 do_sample=False（贪心解码），此参数会被忽略并给出警告
            max_new_tokens=100,                  # 最多新生成 100 个 token（不含输入部分）。
                                                 # 回复较长时会被截断，如需完整自我介绍可调大，如 512
        )
    # output_ids 的内容 = 输入的 prompt token + 新生成的 token，形状 [batch, 总长度]

    # ---------- 8. 截掉 prompt 部分，只保留新生成的内容 ----------
    # zip(...) 把每条输入与对应的输出配对；
    # output_ids[len(input_ids):] 跳过前面与输入等长的部分，只取模型新生成的 token。
    # （注意：推导式里的 input_ids 是循环变量，指“某一条输入的 token 序列”，
    #   与外层的 input_ids 同名但含义不同，容易混淆，建议改名为 in_ids / out_ids）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids.input_ids, output_ids)
    ]

    # ---------- 9. token ID -> 文本 ----------
    # batch_decode：批量解码；skip_special_tokens=True 会去掉 <|im_end|> 等特殊标记
    # [0] 取第一条（也是唯一一条）结果
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 下面这行解码结果（保留特殊标记）
    return_str = tokenizer.decode(generated_ids[0])

    # 返回最终的回复文本
    return response                                        # 打印模型的回复