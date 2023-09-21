import cv2
import numpy as np

with open('girl5.jpg', 'rb') as f:
    buffer = np.frombuffer(f.read(), dtype=np.uint8)
img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)

contours = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Result", contours)
cv2.waitKey(0)
cv2.imshow("1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
