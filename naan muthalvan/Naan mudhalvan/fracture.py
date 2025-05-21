import cv2
import numpy as np

# Load and resize image
image = cv2.imread(r'D:\Projects\Naan mudhalvan\Input\brain.jpeg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found.")
    exit()
image = cv2.resize(image, (400, 400))

# Step 1: Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Canny Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Step 3: Dilate edges
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Step 4: Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert images to BGR for display
original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
output = original_bgr.copy()

# Step 5: Highlight possible fractures
for cnt in contours:
    area = cv2.contourArea(cnt)
    length = cv2.arcLength(cnt, True)
    if area < 1000 and length > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, "Possible Fracture", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Add labels to each image
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.7
thickness = 2
color = (0, 255, 0)
cv2.putText(original_bgr, "Original X-ray", (10, 30), font, scale, color, thickness)
cv2.putText(edges_bgr, "Edge Detection", (10, 30), font, scale, color, thickness)
cv2.putText(output, "Fracture Detection", (10, 30), font, scale, color, thickness)

# Combine images horizontally
combined = np.hstack((original_bgr, edges_bgr, output))

# Show the result
cv2.imshow("Hand Fracture Analysis - Split View", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
