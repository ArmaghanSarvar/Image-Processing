import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img = cv.imread('bone-scan.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.title('Original Image')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

kernel_lap = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
g_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
g_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def median_convolve(image, kernel):
    output_img = np.zeros(image.shape)
    kernel_height = kernel.shape[0] // 2
    kernel_weight = kernel.shape[1] // 2
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            window = image[i - kernel_height: i + kernel_height + 1, j - kernel_weight: j + kernel_weight + 1]
            output_img[i, j] = np.median(window.ravel())
    final = output_img[1: output_img.shape[0] - 1, 1:output_img.shape[1] - 1]
    return final


def convolve(image, kernel, change_padding=0):
    output_img = np.zeros(image.shape)
    kernel_height = kernel.shape[0] // 2
    kernel_weight = kernel.shape[1] // 2
    if change_padding == 0:
        padding = 1
    else:
        padding = 2
    for i in range(padding, image.shape[0] - padding):
        for j in range(padding, image.shape[1] - padding):
            window = image[i - kernel_height: i + kernel_height + 1, j - kernel_weight: j + kernel_weight + 1]
            product = np.multiply(kernel, window)
            output_img[i, j] = np.sum(product)
    final = output_img[padding: output_img.shape[0] - padding, padding:output_img.shape[1] - padding]
    return final


def scale(image):
    image = image * (255.0 / np.max(image))
    return image


def fit(image):
    image[image < 0] = 0
    image[image > 255] = 255
    return image


kernel = np.ones((3, 3))
restored = median_convolve(img, kernel)
restored = restored - np.min(restored)
restored = scale(restored)
plt.title('After Median Filter')
plt.imshow(restored, cmap='gray', vmin=0, vmax=255)
plt.show()
after_padding = np.zeros((restored.shape[0] + 2, restored.shape[1] + 2))
after_padding[1:restored.shape[0] + 1, 1: restored.shape[1] + 1] = restored
normal_laplacian = convolve(after_padding, kernel_lap)
plt.title('Laplacian Without Scaling')
plt.imshow(normal_laplacian, cmap='gray', vmin=0, vmax=255)
plt.show()


laplacian_with_scaling = normal_laplacian - np.min(normal_laplacian)
laplacian_with_scaling = scale(laplacian_with_scaling)
plt.title('Laplacian With Scaling')
plt.imshow(laplacian_with_scaling, cmap='gray', vmin=0, vmax=255)
plt.show()


sharpened_laplacian = fit(restored + normal_laplacian)
plt.title('Sharpened Laplacian')
plt.imshow(sharpened_laplacian, cmap='gray', vmin=0, vmax=255)
plt.show()


def get_mag(g):
    return np.abs(g)


after_padding = np.zeros((restored.shape[0] + 2, restored.shape[1] + 2))
after_padding[1:restored.shape[0] + 1, 1: restored.shape[1] + 1] = restored
gx_img = convolve(after_padding, g_x)
gy_img = convolve(after_padding, g_y)
M_img = get_mag(gx_img) + get_mag(gy_img)
plt.title('Sobel Image')
plt.imshow(M_img, cmap='gray')
plt.show()


smoothing_kernel = 1/25 * (np.ones((5, 5)))
res = np.zeros((img.shape[0] + 4, img.shape[1] + 4))
after_padding = np.zeros((restored.shape[0] + 4, restored.shape[1] + 4))
after_padding[2:img.shape[0] + 2, 2:img.shape[1] + 2] = img
smoothed = convolve(after_padding, smoothing_kernel, 2)
plt.title('Smooth Sobel Image')
plt.imshow(smoothed, cmap='gray')
plt.show()

scaled_ = scale(smoothed - np.min(smoothed))
mask = np.multiply(sharpened_laplacian.astype('uint64'), scaled_.astype('uint64'))
mask = mask - np.min(mask)
scaled_mask = scale(mask)
added_mask = restored.astype('uint64') + scaled_mask.astype('uint64')
plt.subplot(121)
plt.title('Sum of Mask and Restored')
plt.imshow(added_mask, cmap='gray', vmin=0, vmax=255)
plt.subplot(122)
plt.title('Mask')
plt.imshow(scaled_mask, cmap='gray', vmin=0, vmax=255)
plt.show()

power_law_transformed = 1 * (added_mask ** 0.3)
power_law_transformed = power_law_transformed - np.min(power_law_transformed)
power_law_transformed = scale(power_law_transformed)
plt.title('Final Result-powerLaw-gamma0.3')
plt.imshow(power_law_transformed, cmap='gray')
plt.show()