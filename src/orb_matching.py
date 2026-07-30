import cv2
import numpy as np

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
image = cv2.imread("dataset/large_images/satellite.jpg")

# Read the template image
template = cv2.imread("dataset/templates/airfield.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create()
#keypoints = orb.detect(gray_image, None)
keypoints, descriptors = orb.detectAndCompute(gray_image, None)
orb_image = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=0
)
first = keypoints[0]

print("X, Y:", first.pt)
print("Size:", first.size)
print("Angle:", first.angle)
print("Response:", first.response)
print("Octave:", first.octave)
print("Number of keypoints:", len(keypoints))
print("Descriptor Shape:", descriptors.shape)
print("First Descriptor:")
print(descriptors[0])
display_image = cv2.resize(orb_image, (1100, 700))
cv2.imshow("ORB Keypoints", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()