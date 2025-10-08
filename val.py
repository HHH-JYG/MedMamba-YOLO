from ultralytics import YOLO

# Load a model
model = YOLO("runs/BCCD/train/weights/best.pt")

# Customize validation settings
model.val(data="BCCD.yaml", split="test")






       


