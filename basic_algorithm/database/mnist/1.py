from PIL import Image

img = Image.open("test.jpg")     # 从文件打开
print(type(img))                 # <class 'PIL.Image.Image'>
print(img.size)                  # (宽度, 高度)，比如 (227, 227)
print(img.mode)                  # 'RGB'、'L'（灰度）、'RGBA' 等