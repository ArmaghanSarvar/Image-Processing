import cv2 as cv
import numpy as np

img = cv.imread('T.jpg')
cv.imshow('Original', img)

scaled_img = cv.resize(img, None, fx=0.7, fy=0.7)
# cv.imshow('scaled-factor=0.7', scaled_img)

rows, cols, channels = img.shape

M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated = cv.warpAffine(img, M, (cols, rows))
# cv.imshow('rotated image', rotated)

tx = cols / 3
ty = rows / 3
T = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv.warpAffine(img, T, (cols, rows))
# cv.imshow('translated', translated)

vertical_shear_T = np.float32([[1, 0, 0], [0.5, 1, 0]])
vertical_shear = cv.warpAffine(img, vertical_shear_T, (2 * cols, 2 * rows))
# cv.imshow('Vertically Sheared', vertical_shear)

horizontal_shear_T = np.float32([[1, 0.5, 0], [0, 1, 0]])
horizontal_shear = cv.warpAffine(img, horizontal_shear_T, (2 * cols, 2 * rows))
# cv.imshow('Horizontally Sheared', horizontal_shear)


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

degree = -45

tx = (1 - np.cos(np.deg2rad(degree))) * rows / 2 - np.sin(np.deg2rad(degree)) * cols / 2
ty = np.sin(np.deg2rad(degree)) * rows / 2 + (1 - np.cos(np.deg2rad(degree))) * cols / 2

T_matrix = np.float32([[np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
                       [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0], [tx, ty, 1]])
img_using_forward = np.zeros((rows, cols))

for i in range(rows):
    for j in range(cols):
        orig = np.array([[i, j, 1]])
        res = orig.dot(T_matrix)[0]
        try:
            img_using_forward[int(res[0]), int(res[1])] = img[i, j]
        except:  # more than borders
            pass
img_using_forward = img_using_forward.astype('uint8')
cv.imshow('img_using_forward', img_using_forward)

inverse_T_matrix = np.linalg.inv(T_matrix)
img_using_backward = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        res = np.array([[i, j, 1]])
        orig = np.dot(res, inverse_T_matrix)[0]
        try:
            img_using_backward[i, j] = img[int(orig[0]), int(orig[1])]
        except: # more than borders
            pass

img_using_backward = img_using_backward.astype('uint8')
cv.imshow('img_using_backward', img_using_backward)

cv.waitKey(0)
cv.destroyAllWindows()