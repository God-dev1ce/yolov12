import cv2
from ultralytics import YOLO

# 1. 加载模型
model = YOLO('run/voc/yolov12/weights/best.pt')

# 2. 执行推理并保存结果
results = model('ultralytics/assets/bus.jpg', save=True)  # 自动保存到runs/detect目录

# 3. 打印结果
for result in results:
    print(result.boxes)