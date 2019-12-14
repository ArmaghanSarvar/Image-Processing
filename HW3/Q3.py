import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


def show_with_plot(img, title):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


def main_func(img, is_low, filter, filter_parameters):
    my_window = np.zeros(img.size, img.dtype)
    if is_low:
        if filter == 'ideal':
            my_window = low_ideal(filter_parameters)
        elif filter == 'gaussian':
            my_window = low_gaussian(filter_parameters)
        elif filter == 'butterworth':
            my_window = low_butterWorth(filter_parameters)
    else:
        if filter == 'ideal':
            my_window = high_ideal(filter_parameters)
        elif filter == 'gaussian':
            my_window = high_gaussian(filter_parameters)
        elif filter == 'butterworth':
            my_window = high_butterWorth(filter_parameters)
    img_f = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_f)
    img_f_ishift = np.fft.ifftshift(masking(dft_shift, my_window))
    image_fi = cv.idft(img_f_ishift)
    show_with_plot(crop(cv.magnitude(image_fi[:, :, 0], image_fi[:, :, 1])), "Final")


def crop(to_be_cropped):
    cropped = np.zeros((h, w))
    for i in range(0, p):
        for j in range(0, q):
            if i < h and j < w:
                cropped[i, j] = to_be_cropped[i, j]
    return cropped


def distance(x, y):
    return np.sqrt(math.pow((x - p / 2), 2) + math.pow((y - q / 2), 2))


def low_ideal(cutoff):
    H_uv = np.zeros((p, q), padded_img.dtype)
    for u in range(0, p):
        for v in range(0, q):
            if distance(u, v) <= cutoff[0]:
                H_uv[u, v] = 1
    return H_uv


def high_ideal(cutoff):
    H_uv = np.zeros((p, q), padded_img.dtype)
    for u in range(0, p):
        for v in range(0, q):
            if distance(u, v) > cutoff[0]:
                H_uv[u, v] = 1
    return H_uv


def low_gaussian(parameter):
    H_uv = np.zeros((p, q), padded_img.dtype)
    for u in range(0, p):
        for v in range(0, q):
            H_uv[u, v] = np.exp((-(math.pow(distance(u, v), 2))) / (2 * (math.pow(parameter[0], 2))))
    return H_uv


def high_gaussian(parameter):
    H_uv = np.zeros((p, q), padded_img.dtype)
    for u in range(0, p):
        for v in range(0, q):
            H_uv[u, v] = 1 - np.exp((-(distance(u, v)) ** 2) / (2 * ((parameter[0]) ** 2)))
    return H_uv


def low_butterWorth(parameter_list):
    H_uv = np.zeros((p, q), padded_img.dtype)
    for u in range(0, p):
        for v in range(0, q):
            H_uv[u, v] = 1 / (1 + ((distance(u, v) / parameter_list[0]) ** (2 * parameter_list[1])))
    return H_uv


def high_butterWorth(parameter):
    H_uv = np.zeros((p, q), padded_img.dtype)
    for u in range(0, p):
        for v in range(0, q):
            if distance(u, v) != 0:
                H_uv[u, v] = 1 / (1 + ((parameter[0] / distance(u, v)) ** (2 * parameter[1])))
    return H_uv


def masking(F_uv, H_uv):
    for i in range(0, p):
        for j in range(0, q):
            F_uv[i, j, 0] = F_uv[i, j, 0] * H_uv[i, j]
            F_uv[i, j, 1] = F_uv[i, j, 1] * H_uv[i, j]
    return F_uv


img = cv.imread("a.tif", 0)
h = img.shape[0]
w = img.shape[1]
p = 2 * h
q = 2 * w
padded_img = np.zeros((p, q))
for i in range(0, p):
    for j in range(0, q):
        if i < h and j < w:
            padded_img[i][j] = img[i][j]

main_func(padded_img, True, "gaussian", [50])
main_func(padded_img, True, "gaussian", [100])
main_func(padded_img, True, "gaussian", [200])
