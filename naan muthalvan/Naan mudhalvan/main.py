import cv2
import numpy as np
image = cv2.imread(r"D:\naan muthalvan\Naan mudhalvan\Input\brain.jpeg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error loading image.")
    exit()
blurred = cv2.GaussianBlur(image, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)


original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
closing_bgr = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)


height = 400
original_bgr = cv2.resize(original_bgr, (height, height))
closing_bgr = cv2.resize(closing_bgr, (height, height))
output = cv2.resize(output, (height, height))


font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.7
thickness = 2
color = (0, 255, 0)

cv2.putText(original_bgr, "Original", (10, 30), font, scale, color, thickness)
cv2.putText(closing_bgr, "Thresholded", (10, 30), font, scale, color, thickness)
cv2.putText(output, "Tumor Detected", (10, 30), font, scale, color, thickness)

combined = np.hstack((original_bgr, closing_bgr, output))
cv2.imshow("MRI Analysis - Split View", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
