import cv2 as cv
import numpy as np

img = cv.imread('images/coins.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv.medianBlur(gray, 5)
circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img.shape[0]/10, param1=100, param2=9, minRadius=2, maxRadius=13)
# Draw circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    radius = []
    for i in range(0, len(circles[0, :])):
        radius.append(circles[0, :][i][2])
    print(radius)
    mean = np.mean(radius)
    print(mean)
    for i in circles[0, :]:
        # Draw bigger circles
        if i[2] > mean:
            cv.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # Draw smaller circles
        else:
            cv.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # Draw inner circles
        cv.circle(img, (i[0], i[1]), 2, (0, 255, 255), 3)

cv.imshow('Final', img)
cv.waitKey(0)
