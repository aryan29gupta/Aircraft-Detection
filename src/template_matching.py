import cv2
import numpy as np
# Read the large satellite image
image = cv2.imread("dataset/large_images/satellite.jpg")

# Read the template image
template = cv2.imread("dataset/templates/airfield.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
height, width = gray_template.shape
# matchTemplate(where to search, what to search, method)
result = cv2.matchTemplate(
    gray_image,
    gray_template,
    cv2.TM_CCOEFF_NORMED
)

threshold = 0.8

locations = np.where(result >= threshold)

for point in zip(*locations[::-1]):

    top_left = point

    bottom_right = (
        point[0] + width,
        point[1] + height
    )

    cv2.rectangle(
        image,
        top_left,
        bottom_right,
        (0,255,0),
        2
    )


print("locations are:  ",locations)

# Find minimum and maximum similarity values and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print("Minimum Similarity:", min_val)
print("Maximum Similarity:", max_val)
print("Best Match Location:", max_loc)

print("Template Height:", height)
print("Template Width:", width)


print(type(result))
print(result.shape)
print(result.max())

# Top-left corner
top_left = max_loc

# Bottom-right corner
bottom_right = (top_left[0] + width, top_left[1] + height)

# Draw rectangle
cv2.rectangle(
    image,
    top_left,
    bottom_right,
    (0, 255, 0),   # Green color
    3              # Thickness
)

# Display image
# Resize image only for display
display_image = cv2.resize(image, (1100, 700))
print("Image Shape:", image.shape)
print("Template Shape:", template.shape)

print("Maximum Similarity:", max_val)
print("Best Match Location:", max_loc)
cv2.imshow("Detected Airfield", display_image)
cv2.imwrite("outputs/detected_airfield.jpg", image)


cv2.waitKey(0)
cv2.destroyAllWindows()