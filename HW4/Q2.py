import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_with_plot(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


img1 = cv.imread('images/cygnus.tif', 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
opening = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
show_with_plot(closing, 'Opening-Closing(1)')

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
show_with_plot(closing, 'Opening-Closing(3)')

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
show_with_plot(closing, 'Opening-Closing(5)')

img2 = cv.imread('images/brain.tif', 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_dilation = cv.dilate(img2, kernel, iterations=1)
img_erosion = cv.erode(img2, kernel, iterations=1)
img_gradient = img_dilation - img_erosion
show_with_plot(img_gradient, 'Gradient')

img3 = cv.imread('images/rice.tif', 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 40))
img_top_hat = cv.morphologyEx(img3, cv.MORPH_TOPHAT, kernel)
_, thresh = cv.threshold(img_top_hat, 25, 255, cv.THRESH_BINARY)

plt.subplot(211)
plt.title('Top-hat')
plt.imshow(img_top_hat, cmap='gray')
plt.subplot(212)
plt.title('Binary!')
plt.imshow(thresh, cmap='gray')
plt.show()

