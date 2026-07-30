from ultralytics import YOLO

# Load a pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs",
    name="aircraft_detector"
)