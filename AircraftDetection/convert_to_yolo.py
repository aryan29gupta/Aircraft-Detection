import os
import ast
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
IMAGE_SIZE = 2560

CSV_FILE = "AircraftDetection/annotations.csv"
LABELS_DIR = "AircraftDetection/labels"

# Create labels folder if it doesn't exist
os.makedirs(LABELS_DIR, exist_ok=True)

# Read annotations
annotations = pd.read_csv(CSV_FILE)

print(f"Total annotations: {len(annotations)}")

# -----------------------------
# Convert every annotation
# -----------------------------
for _, row in annotations.iterrows():

    image_name = row["image_id"]
    geometry = row["geometry"]

    # Convert string into Python list
    points = ast.literal_eval(geometry)

    # Separate x and y coordinates
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Bounding box
    xmin = min(x_coords)
    xmax = max(x_coords)

    ymin = min(y_coords)
    ymax = max(y_coords)

    # Convert to YOLO format
    center_x = ((xmin + xmax) / 2) / IMAGE_SIZE
    center_y = ((ymin + ymax) / 2) / IMAGE_SIZE

    width = (xmax - xmin) / IMAGE_SIZE
    height = (ymax - ymin) / IMAGE_SIZE

    # Create label filename
    label_name = image_name.replace(".jpg", ".txt")
    label_path = os.path.join(LABELS_DIR, label_name)

    # Write one aircraft annotation
    with open(label_path, "a") as f:
        f.write(
            f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
        )

print("Conversion Complete!")