import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_with_plot(image, title):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


img = cv.imread('images/MRI1.png')
rows, cols, ch = img.shape
# The point to point feature extraction - here : the butterfly wings in the image
pts1 = np.float32([[66, 80], [115, 81], [104, 134]])
pts2 = np.float32([[257, 154], [341, 193], [262, 260]])

m = cv.getAffineTransform(pts1, pts2)
X = [0., 0., 1.]
affine = np.vstack((m, X))
print(affine)

# check correctness
new_image = cv.warpAffine(img, m, (3 * cols, 2 * rows))

show_with_plot(new_image, 'Final')
