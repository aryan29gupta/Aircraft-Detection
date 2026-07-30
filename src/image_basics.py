import cv2
import numpy as np

image = cv2.imread("dataset/large_images/satellite.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")

print("Original shape:",image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Gray shape:", gray.shape)

print("\nValue of pixel at (100, 200):")
print(image[100, 200])

print("\nImage Height:", image.shape[0])
print("Image Width:", image.shape[1])
print("Channels:", image.shape[2])

crop = image[500:1000, 800:1500]
cv2.imwrite("outputs/crop.jpg", crop)