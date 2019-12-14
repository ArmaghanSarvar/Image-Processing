import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('spine.tif')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.subplot(121)
plt.title('original image')
plt.imshow(img, cmap='gray')
plt.subplot(122)
histogram = cv.calcHist([img], [0], None, [256], [0, 256])
plt.title('original histogram')
plt.plot(histogram, color='r')
plt.show()


# np.log
def logarithmic_function(n):
    c = n/(np.log(1 + np.max(img)))
    log_transformed = np.uint8(c * np.log1p(img))
    plt.subplot(121)
    title = 'n= ' + str(n) + ' transformed image'
    plt.title(title)
    plt.imshow(log_transformed, cmap='gray')
    plt.subplot(122)
    histogram = cv.calcHist([log_transformed], [0], None, [256], [0, 256])
    plt.title('transformed histogram')
    plt.plot(histogram, color='r')
    plt.show()


logarithmic_function(255)
logarithmic_function(100)
logarithmic_function(50)
# np.log10
n = 255
to_be_changed = img.copy().astype('uint64')
c = n/(np.log10(1 + np.max(img)))
transformed = c * np.log10(to_be_changed + 1)
log_transformed = np.array(transformed, dtype=np.uint8)
plt.subplot(121)
title = 'log10 transformed image'
plt.title(title)
plt.imshow(log_transformed, cmap='gray')
plt.subplot(122)
histogram = cv.calcHist([log_transformed], [0], None, [256], [0, 256])
plt.title('transformed histogram')
plt.plot(histogram, color='b')
plt.show()

# b
img = cv.imread('kidney.tif')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.show()

figure1 = img.copy()
figure2 = img.copy()
# print(type(figure1))
# print(figure1.shape[0])

for pix in range(0, figure1.shape[0]):
    for pixc in range(0, figure1.shape[1]):
        if figure1[pix][pixc] <= 160 or figure1[pix][pixc] >= 240:
            figure1[pix][pixc] = 20
        if 160 < figure1[pix][pixc] < 240:
            figure1[pix][pixc] = 150

plt.title('figure1')
plt.imshow(figure1, cmap='gray', vmin=0, vmax=255)
plt.show()


for pix in range(0, figure2.shape[0]):
    for pixc in range(0, figure2.shape[1]):
        if 100 <= figure2[pix][pixc] <= 165:
            figure2[pix][pixc] = 200

plt.title('figure2')
plt.imshow(figure2, cmap='gray', vmin=0, vmax=255)
plt.show()
