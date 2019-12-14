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


def read_images(name1, name2):
    img_mandrill = cv.imread(name1, 0)
    img_clown = cv.imread(name2, 0)
    show_with_plot(img_mandrill, name1 + ' gray')
    show_with_plot(img_clown, name2 + ' gray')
    return img_mandrill, img_clown


def combine(img_mandrill, img_clown):
    img_mandrill_f = np.fft.fft2(img_mandrill)
    img_mandrill_shift = np.fft.fftshift(img_mandrill_f)
    phase_spectrum1 = np.angle(img_mandrill_shift)
    magnitude_spectrum1 = 20 * np.log(np.abs(img_mandrill_shift))

    img_clown_f = np.fft.fft2(img_clown)
    img_clown_shift = np.fft.fftshift(img_clown_f)
    phase_spectrum2 = np.angle(img_clown_shift)
    magnitude_spectrum2 = 20 * np.log(np.abs(img_clown_shift))

    mandrill_mag_and_clown_phase = np.multiply(np.abs(img_mandrill_f), np.exp(1j*np.angle(img_clown_f)))
    combined1 = np.real(np.fft.ifft2(mandrill_mag_and_clown_phase))
    combined1 = np.abs(combined1)
    clown_mag_and_mandrill_phase = np.multiply(np.abs(img_clown_f), np.exp(1j*np.angle(img_mandrill_f)))
    combined2 = np.real(np.fft.ifft2(clown_mag_and_mandrill_phase))
    combined2 = np.abs(combined2)

    show_with_plot(combined1, 'mandrill_mag_and_clown_phase gray')
    show_with_plot(combined2, 'clown_mag_and_mandrill_phase gray')
    plt.subplot(221)
    plt.title('phase_spectrum of img_mandrill')
    plt.imshow(phase_spectrum1, cmap='gray')
    plt.subplot(222)
    plt.title('magnitude_spectrum of img_mandrill')
    plt.imshow(magnitude_spectrum1, cmap='gray')
    plt.subplot(223)
    plt.title('phase_spectrum of img_clown')
    plt.imshow(phase_spectrum2, cmap='gray')
    plt.subplot(224)
    plt.title('magnitude_spectrum of img_clown')
    plt.imshow(magnitude_spectrum2, cmap='gray')
    plt.show()


if __name__ == "__main__":
    img1, img2 = read_images('mandrill.tif', 'clown.tif')
    combine(img1, img2)
