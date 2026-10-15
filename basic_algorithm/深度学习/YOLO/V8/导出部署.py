from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")

# 导成 ONNX（最通用，ONNXRuntime/TensorRT/OpenVINO 都能吃）
model.export(format="onnx", imgsz=640, dynamic=True)

# 其他常用：
# format="engine"  -> TensorRT（NVIDIA 加速）
# format="openvino"-> OpenVINO（Intel CPU/GPU）
# format="tflite"  -> 安卓/树莓派
# format="coreml"  -> iOS


#导出后推理可以完全不依赖 torch
model_onnx = YOLO("best.onnx")
results = model_onnx("../test_image.jpg")
results[0].show()