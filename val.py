from ultralytics import YOLO

model = YOLO('OptiSAR-Net.pt')

model.val(data='CDHD.yaml', batch=1)