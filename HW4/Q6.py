import numpy as np
import cv2

img = cv2.imread('images/jump.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, 17)
ret2, thresh2 = cv2.threshold(gray, 0, 255, 17)
ret3, thresh3 = cv2.threshold(gray, 0, 255, 8)

cv2.imshow('image', thresh)

kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_DILATE,
                           kernel, iterations=10)
closing2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE,
                           kernel, iterations=10)
closing3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE,
                           kernel, iterations=95)

closing = cv2.bitwise_and(closing, closing2)
closing = cv2.bitwise_and(closing, closing3)
bg = cv2.dilate(closing, kernel, iterations=1)

dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
ret, fg = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0)

print(fg.shape, img.shape)

new_image = cv2.imread('images/jump.jpg')

for i in range(0, fg.shape[0]):
    for j in range(0, fg.shape[1]):
        sum = int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])
        if sum == 0 or fg[i][j] != 255.0 or img[i][j][0] / sum > 0.5:
            new_image[i][j] = [0, 0, 0]

cv2.imshow('image', new_image)
cv2.waitKey(0)
