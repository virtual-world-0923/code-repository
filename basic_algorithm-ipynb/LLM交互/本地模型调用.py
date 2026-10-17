from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def call_local_model(messages, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 使用 float16 精度加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 1. 定义 4-bit 量化配置
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # 开启 4-bit 量化加载
    #     bnb_4bit_compute_dtype=torch.float16,  # 计算时展开使用的精度（建议 fp16 或 bf16）
    #     bnb_4bit_quant_type="nf4",  # 量化格式，推荐 nf4（效果比 fp4 更好）
    #     bnb_4bit_use_double_quant=True  # 开启双重量化，进一步节省显存
    # )
    # # 使用 4-bit 精度加载模型
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=quant_config,  # 传入量化配置
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    has_template = (getattr(tokenizer, "chat_template", None) is not None)

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = tokenizer(
        text_prompt,
        return_tensors="pt"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    pure_input_ids = input_ids.input_ids.to(device)
    pure_attention_mask = input_ids.attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=pure_input_ids,
            attention_mask=pure_attention_mask,
            num_return_sequences=1,
            temperature=0.1,
            max_new_tokens=100,
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids.input_ids, output_ids)
    ]

    # 7. 解码为中文并返回
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return_str = tokenizer.decode(generated_ids[0])
    return response