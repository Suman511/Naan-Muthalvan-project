import cv2
import numpy as np

image = cv2.imread(r'D:\Projects\Naan mudhalvan\Input\brain.jpeg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found.")
    exit()

blurred = cv2.GaussianBlur(image, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    if cv2.contourArea(cnt) > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, "Possible Anomaly", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow("X-ray Anomaly Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
