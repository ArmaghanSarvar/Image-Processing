import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

low_contrast = cv.imread('Lowcontrast.tif')
low_contrast = cv.cvtColor(low_contrast, cv.COLOR_BGR2GRAY)
dark = cv.imread('Dark.tif')
dark = cv.cvtColor(dark, cv.COLOR_BGR2GRAY)
white = cv.imread('White.tif')
white = cv.cvtColor(white, cv.COLOR_BGR2GRAY)


# a
def get_histogram(image):
    my_hist = []
    for i in range(0, 256):
        my_hist.append(0)
    unique, indices = np.unique(image, return_counts=True)
    for i in range(len(unique)):
        my_hist[unique[i]] = indices[i]
    arrayed = np.array(my_hist).astype('float64')
    return arrayed


# b
low_hist = get_histogram(low_contrast) / (low_contrast.shape[0] * low_contrast.shape[1])
white_hist = get_histogram(white) / (white.shape[0] * white.shape[1])
dark_hist = get_histogram(dark) / (dark.shape[0] * dark.shape[1])

plt.subplot(131)
plt.title('low_contrast')
plt.imshow(low_contrast, cmap='gray', vmin=0, vmax=255)
plt.subplot(132)
plt.title('white')
plt.imshow(white, cmap='gray', vmin=0, vmax=255)
plt.subplot(133)
plt.title('dark')
plt.imshow(dark, cmap='gray', vmin=0, vmax=255)
plt.show()


def plt_hist(image, name, color):
    title = name + ' histogram'
    histogram = cv.calcHist([image], [0], None, [256], [0, 256])
    plt.title(title)
    plt.plot(histogram, color=color)
    plt.show()


# plt_hist(low_contrast, 'low_contrast', 'b')
# plt_hist(white, 'white', 'r')
# plt_hist(dark, 'dark', 'gold')

def scale(hist, image):
    return 255 * hist / (image.shape[1] * image.shape[0])


# c
def equalize_histogram(image, hist):
    lhist = len(hist)
    limg = len(image)
    for i in range(1, lhist):
        hist[i] = hist[i] + hist[i-1]
    new_levels = (scale(hist, image)).astype('uint8')
    for i in range(limg):
        image[i] = new_levels[image[i]]
    new_result = image.reshape(image.shape[0], image.shape[1])
    return new_result


low_cop = low_contrast.copy()
white_cop = white.copy()
dark_cop = dark.copy()
low_eq = equalize_histogram(low_cop, get_histogram(low_cop.ravel()))
white_eq = equalize_histogram(white_cop, get_histogram(white_cop.ravel()))
dark_eq = equalize_histogram(dark_cop, get_histogram(dark_cop.ravel()))
# plt_hist(low_eq, 'low_equalized', 'c')
# plt_hist(white_eq, 'white_equalized', 'g')
# plt_hist(dark_eq, 'dark_equalized', 'm')

plt.subplot(141)
plt.title('Before low_contrast')
plt.imshow(low_contrast, cmap='gray', vmin=0, vmax=255)

plt.subplot(142)
histogram = cv.calcHist([low_contrast], [0], None, [256], [0, 256])
plt.title('Normalized hist')
plt.plot(histogram, color='r')

plt.subplot(143)
plt.title('After low_contrast')
plt.imshow(low_eq, cmap='gray', vmin=0, vmax=255)

plt.subplot(144)
histogram = cv.calcHist([low_eq], [0], None, [256], [0, 256])
plt.title('Equalized hist')
plt.plot(histogram, color='r')
plt.show()


plt.subplot(141)
plt.title('Before dark')
plt.imshow(dark, cmap='gray', vmin=0, vmax=255)

plt.subplot(142)
histogram = cv.calcHist([dark], [0], None, [256], [0, 256])
plt.title('Normalized hist')
plt.plot(histogram, color='c')

plt.subplot(143)
plt.title('After dark')
plt.imshow(dark_eq, cmap='gray', vmin=0, vmax=255)

plt.subplot(144)
histogram = cv.calcHist([dark_eq], [0], None, [256], [0, 256])
plt.title('Equalized hist')
plt.plot(histogram, color='c')
plt.show()


plt.subplot(141)
plt.title('Before white')
plt.imshow(white, cmap='gray', vmin=0, vmax=255)
plt.subplot(142)
histogram = cv.calcHist([white], [0], None, [256], [0, 256])
plt.title('Normalized hist')
plt.plot(histogram, color='m')

plt.subplot(143)
plt.title('After white')
plt.imshow(white_eq, cmap='gray', vmin=0, vmax=255)

plt.subplot(144)
histogram = cv.calcHist([white_eq], [0], None, [256], [0, 256])
plt.title('Equalized hist')
plt.plot(histogram, color='m')
plt.show()
