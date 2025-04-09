from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

model.train(data="CDHD.yaml", epochs=2, batch=32, imgsz=640)

