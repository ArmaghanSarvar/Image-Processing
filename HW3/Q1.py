import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_with_plot(img, title):
    plt.figure()
    if 'gray' in title:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


img = cv.imread('chest.tif', 0)
show_with_plot(img, 'chest gray')
img_32 = np.float32(img)
dft = cv.dft(img_32, flags = cv.DFT_COMPLEX_OUTPUT)
f_shift = np.fft.fftshift(dft)
# real = f_shift[:, :, 0]
# img = f_shift[:, :, 1]
magnitude_spectrum = 20 * np.log(cv.magnitude(f_shift[:, :, 0], f_shift[:, :, 1]))
show_with_plot(magnitude_spectrum, 'magnitude_spectrum gray')

magnitude, phase = cv.cartToPolar(f_shift[:,:,0], f_shift[:,:,1])
show_with_plot(phase, 'phase_spectrum gray')

shifted_back = np.fft.ifftshift(f_shift)
img_idft = cv.idft(shifted_back)
img_idft = cv.magnitude(img_idft[:,:,0], img_idft[:,:,1])
show_with_plot(img_idft, 'img_inverse_transform gray')

h = img.shape[0]
w = img.shape[1]
mirrored_f = np.zeros((h, w, 2))
for i in range(0, h):
    for j in range(0, w):
        mirrored_f[i, j, 0] = np.multiply(magnitude[i, j], np.exp((-1) * phase[i, j] * 1j)).real
        mirrored_f[i, j, 1] = np.multiply(magnitude[i, j], np.exp((-1) * phase[i, j] * 1j)).imag

f_ishift = np.fft.ifftshift(mirrored_f)
mirrored = cv.idft(f_ishift)
mirrored = cv.magnitude(mirrored[:, :, 0], mirrored[:, :, 1])
show_with_plot(mirrored, "mirrored gray")
