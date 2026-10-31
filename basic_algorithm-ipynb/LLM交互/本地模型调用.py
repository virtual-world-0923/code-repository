# ============================================================
# 功能：在本地加载 Qwen2.5-1.5B-Instruct 对话模型，输入一段对话并返回模型的回复
# ============================================================

# AutoTokenizer：自动根据模型名称加载对应的分词器（文本 <-> token ID 互相转换）
# AutoModelForCausalLM：自动加载“因果语言模型”（从左到右逐个预测下一个 token，即常见的 GPT 式生成模型）
# BitsAndBytesConfig：bitsandbytes 库的量化配置类，用来做 4-bit / 8-bit 量化加载以节省显存
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor, \
    AutoModelForImageTextToText
import torch  # PyTorch，提供张量运算、数据类型（如 float16）和 GPU 支持
from PIL import Image  # Pillow：图像库；用于打开本地图片并转成模型可处理的格式
from dataclasses import dataclass
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # 开启离线模式，强制只从本地缓存加载模型，禁止联网下载
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 开启Transformers离线模式，跳过版本检查直接使用本地文件


@dataclass
class _conveyor:
    tokenizer = None
    model = None
    processor = None
    messages_prompt_ids = None
    streamer = None
    messages = None
    isStream = None
    model_name = None
    image_path = None
    generate_kwargs = {
        "max_new_tokens": 128,  # 最多新生成 128 个 token（防止无限生成）
        "top_p": 0.9,  # 核采样：只从累积概率 90% 的 token 中采样
        "num_return_sequences": 1,  # 每个输入生成几条候选回复；这里只要 1 条
        "temperature": 0.1,  # 温度：越低输出越确定、保守；越高越随机、有创意。注意：如果要让 temperature 生效，建议同时开启 do_sample=True
        "do_sample": True,  # 开启采样后 temperature 才会生效

    }


# 有 NVIDIA GPU 就用第 0 块显卡，否则退回 CPU
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- 检查分词器是否带有对话模板 ----------
# getattr(对象, 属性名, 默认值)：安全读取属性，不存在则返回 None
# chat_template 是一段 Jinja 模板，规定了如何把 messages 拼成模型认识的文本格式
# 注：这里的 has_template 后面并没有被使用，仅作检查用途，可以删除
# has_template = (getattr(tokenizer, "chat_template", None) is not None)

def _use_4bit(model_name):
    # ---------- （可选）4-bit 量化加载方案：目前被注释，未启用 ----------
    # 如果显存不足，可以取消注释下面这段，用 4-bit 量化加载模型（显存占用大幅下降，精度略有损失）。
    # 注意：使用时要把上面“float16 加载”的那段代码注释掉，避免重复加载。

    # pip install -U bitsandbytes accelerate

    # 1. 定义 4-bit 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 开启 4-bit 量化加载（权重用 4 位整数存储）
        bnb_4bit_compute_dtype=torch.float16,  # 实际计算时，把 4-bit 权重“反量化”回此精度来运算
        # （建议 fp16 或 bf16）
        bnb_4bit_quant_type="nf4",  # 量化格式：nf4（NormalFloat4）效果一般优于 fp4
        bnb_4bit_use_double_quant=True  # 双重量化：连量化参数本身也再量化一次，进一步省显存
    )
    # 使用 4-bit 精度加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,  # 传入上面定义的量化配置
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def _multimedis_model(_conveyor):
    model_name = _conveyor.model_name
    messages = _conveyor.messages
    image_path = _conveyor.image_path

    # Processor = tokenizer(文本) + image_processor(图像) 的组合
    #   它封装了 chat_template 渲染、tokenizer.encode、图像 resize/归一化
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 加载模型权重（safetensors 分片文件）
    #   torch_dtype="auto" : 自动沿用模型原始精度（通常 bfloat16/float16）
    #   device_map="auto"  : 自动把不同层分配到可用 GPU/CPU（需安装 accelerate）
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,  # GPU 上常用 bfloat16 节省显存
        device_map="auto",
        trust_remote_code=True,
    )

    # 应用 chat_template 渲染成带特殊 token 的文本
    #   add_generation_prompt=True : 在末尾追加 "assistant" 起始标记，告诉模型"该你说了"
    messages_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,  # 先不切 token，返回纯字符串
        add_generation_prompt=True,
    )

    # 加载图像
    #   Qwen2.5-VL 的 processor 对图像参数名为 "images"
    image = Image.open(image_path).convert("RGB") if image_path else None

    #  tokenize + (图像预处理) -> token ID
    #   return_tensors="pt" : 返回 PyTorch 张量
    # 返回对象是一个字典式结构，包含：
    #   input_ids      ：token ID 张量，形状 [batch, 序列长度]
    #   attention_mask ：注意力掩码，1 表示真实 token，0 表示填充（padding）
    messages_prompt_ids = processor(
        text=[messages_prompt],  # 渲染后的 prompt 文本（列表，支持 batch）
        images=image,  # 图像对象；纯文本传 None
        return_tensors="pt",  # 返回张量格式："pt"=PyTorch
        # 加这两行，限制图片分辨率，防止 CPU 内存爆炸。强制把图片的 token 数（不是像素数）限制在这个范围内
        min_pixels=256 * 28 * 28,  # 约 200×200
        max_pixels=512 * 28 * 28,  # 约 280×280
    )

    # ---------- 把数据和模型放到同一个设备上 ----------
    '''
    # 把模型移动到该设备。
    # 注意：前面已用 device_map="auto" 由 accelerate 自动放置模型，这里不要再用 model.to(device)
    # 在多卡/显存不足需要分层卸载的情况下，甚至可能报错或破坏自动分层。
    # model = model.to(device)
    '''

    # 输入张量也必须与模型在同一设备上，否则会报设备不一致的错误
    # model.device，它会自动指向模型第一层所在的设备。把数据放在模型第一层（即输入层/Embedding 层所在的设备）上即可
    messages_prompt_ids.to(model.device)

    # ---------- 生成回复 ----------
    model.eval()  # 切换到推理（评估）模式：关闭 Dropout 等仅训练时使用的随机行为

    _conveyor.model = model
    _conveyor.processor = processor
    _conveyor.tokenizer = processor.tokenizer
    _conveyor.messages_prompt_ids = messages_prompt_ids
    return _conveyor


def _causa_model(_conveyor):
    model_name = _conveyor.model_name
    messages = _conveyor.messages

    # ---------- 加载分词器 ----------
    # from_pretrained：从指定模型名/路径加载分词器及其配置
    # trust_remote_code=True：允许执行模型仓库里自带的自定义 Python 代码。
    #   某些模型需要它才能正确加载；注意：只对可信来源的模型开启，否则有安全风险。
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ---------- 加载模型（float16 半精度） ----------
    # 使用 float16 精度加载模型：显存占用约为 float32 的一半，速度更快，精度损失很小
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # 模型名称或本地路径
        dtype=torch.float16,  # 权重加载的数据类型；float16 = 半精度。
        # （旧版 transformers 中此参数名为 torch_dtype，新版改为 dtype）
        device_map="auto",  # 自动分配模型放到哪些设备上（GPU / CPU），
        # 显存不够时会自动把部分层放到 CPU；需要安装 accelerate 库
        trust_remote_code=True,  # 同上，允许运行模型自带的自定义代码
    )

    # 把对话消息按模板渲染成带特殊 token 的文本
    # 对 Qwen 来说，结果大致形如：
    # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n自我介绍<|im_end|>\n<|im_start|>assistant\n
    #   add_generation_prompt=True : 在末尾追加 "assistant" 起始标记，告诉模型"该你说了"
    messages_prompt = tokenizer.apply_chat_template(
        messages,  # 对话消息列表
        tokenize=False,  # False：只返回拼好的文本字符串，不直接转成 token ID
        add_generation_prompt=True,  # True：在末尾追加“assistant 开始回答”的标记，提示模型接下来该轮到它回复了
    )

    # ---------- 文本 -> token ID ----------
    # 调用分词器，把文本切分并映射成数字 ID
    # 返回对象是一个字典式结构，包含：
    #   input_ids      ：token ID 张量，形状 [batch, 序列长度]
    #   attention_mask ：注意力掩码，1 表示真实 token，0 表示填充（padding）
    messages_prompt_ids = tokenizer(
        messages_prompt,
        return_tensors="pt"  # "pt" = 返回 PyTorch 张量
    )

    # ---------- 把数据和模型放到同一个设备上 ----------
    '''
    # 把模型移动到该设备。
    # 注意：前面已用 device_map="auto" 由 accelerate 自动放置模型，这里不要再用 model.to(device)
    # 在多卡/显存不足需要分层卸载的情况下，甚至可能报错或破坏自动分层。
    # model = model.to(device)
    '''

    # 输入张量也必须与模型在同一设备上，否则会报设备不一致的错误
    # model.device，它会自动指向模型第一层所在的设备。把数据放在模型第一层（即输入层/Embedding 层所在的设备）上即可
    messages_prompt_ids.to(model.device)

    # ---------- 生成回复 ----------
    model.eval()  # 切换到推理（评估）模式：关闭 Dropout 等仅训练时使用的随机行为

    _conveyor.model = model
    _conveyor.tokenizer = tokenizer
    _conveyor.messages_prompt_ids = messages_prompt_ids
    return _conveyor


def _processing_model_output_stream(_conveyor):
    tokenizer = _conveyor.tokenizer
    model = _conveyor.model
    generate_kwargs = _conveyor.generate_kwargs
    messages_prompt_ids = _conveyor.messages_prompt_ids

    from threading import Thread
    from transformers import TextIteratorStreamer

    # ② 创建流式器：负责把模型新生成的 token 缓存成可迭代的文本流
    #   skip_prompt=True : 跳过 prompt 部分，只流式输出回答
    #   skip_special_tokens=True : 输出时丢弃特殊 token
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs["streamer"] = streamer

    '''
    启动后台线程执行生成任务
    因为 model.generate 是阻塞的，必须在子线程中运行，主线程才能实时读取 streamer
    子线程并不会继承主线程的 no_grad 状态
    直接这样在主线程里调用 thread = Thread(target=model.generate, kwargs=generation_kwargs)，子线程还是会计算梯度，影响性能
    '''

    # 定义一个带有 torch.no_grad() 的子线程执行函数
    def target_func():
        with torch.no_grad():
            model.generate(**messages_prompt_ids, **generate_kwargs)

    # 把 target_func 传给子线程
    thread = Thread(target=target_func)
    thread.start()

    # ④ 主线程：从 streamer 逐块读取并打印
    for chunk in streamer:  # chunk: str，每次若干字符/token
        yield chunk
    # thread.join()  # 等待生成线程结束


def _processing_model_output_not_stream(_conveyor):
    messages_prompt_ids = _conveyor.messages_prompt_ids
    model = _conveyor.model
    tokenizer = _conveyor.tokenizer
    generate_kwargs = _conveyor.generate_kwargs

    with torch.no_grad():  # 关闭梯度计算，节省显存与计算
        output_ids = model.generate(**messages_prompt_ids, **generate_kwargs)

        # output_ids 的内容 = 输入的 prompt token + 新生成的 token，形状 [batch, 总长度]
        # ---------- 截掉 prompt 部分，只保留新生成的内容 ----------
        # zip(...) 把每条输入与对应的输出配对；
        # output_ids[len(input_ids):] 跳过前面与输入等长的部分，只取模型新生成的 token。
        # （注意：推导式里的 input_ids 是循环变量，指“某一条输入的 token 序列”，
        #   与外层的 input_ids 同名但含义不同，容易混淆，建议改名为 in_ids / out_ids）
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(messages_prompt_ids.input_ids, output_ids)
        ]

        # ---------- token ID -> 文本 ----------
        # batch_decode：批量解码；skip_special_tokens=True 会去掉 <|im_end|> 等特殊标记
        # [0] 取第一条（也是唯一一条）结果
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # decode: 解码一条
        return_str = tokenizer.decode(generated_ids[0])

        return response


def local_model_qwen(messages, image_path=None, isStream=False):
    _conveyor.messages = messages
    _conveyor.image_path = image_path
    _conveyor.isStream = isStream

    if image_path:  # 多媒体模型
        model_name = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master"
        _conveyor.model_name = model_name
        _multimedis_model(_conveyor)
    else:  # 因果模型
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        _conveyor.model_name = model_name
        _causa_model(_conveyor)

    if isStream:
        output_generator = _processing_model_output_stream(_conveyor)
    else:
        output_generator = _processing_model_output_not_stream(_conveyor)
    return output_generator
