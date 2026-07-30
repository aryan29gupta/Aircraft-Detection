from ultralytics import YOLO
import numpy as np

model = YOLO("runs/detect/runs/aircraft_detector/weights/best.pt")

results = model.predict(
    source="extras",
    save=True,
    conf=0.25
)

confidences = []

for result in results:
    for box in result.boxes:
        conf = box.conf.item()
        confidences.append(conf)

print("\n----------- Statistics -----------")

print("Number of detections :", len(confidences))
print("Mean Confidence      :", np.mean(confidences))
print("Maximum Confidence   :", np.max(confidences))
print("Minimum Confidence   :", np.min(confidences))
print("Variance             :", np.var(confidences))
print("Standard Deviation   :", np.std(confidences))