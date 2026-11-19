from model_config import model

response = model.invoke(
    "你好，请介绍一下自己"
)

print(response.content)