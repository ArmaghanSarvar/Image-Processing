import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_with_plot(img, title):
    plt.figure()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


img = cv.imread('images/object.jpg', 0)
kernel = np.ones((3, 3), np.uint8)

img_erosion = cv.erode(img, kernel, iterations=1)
img_dilation = cv.dilate(img, kernel, iterations=1)
img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
img_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

show_with_plot(img, 'Original')
show_with_plot(img_erosion, 'Erosion')
show_with_plot(img_dilation, 'Dilation')
show_with_plot(img_closing, 'CLosing')
show_with_plot(img_opening, 'Opening')

