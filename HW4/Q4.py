import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_with_plot(img, title):
    plt.figure()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype('uint8')
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


img = cv.imread('images/new_echo.jpg', 0)
# show_with_plot(img, "Original Image")

img_gaussian = cv.GaussianBlur(img, (5, 5), 0)

img_sobel_x = cv.Sobel(img_gaussian, cv.CV_8U, 1, 0, ksize=5)
img_sobel_y = cv.Sobel(img_gaussian, cv.CV_8U, 0, 1, ksize=5)
img_sobel = img_sobel_x + img_sobel_y

# plt.subplot(221)
# plt.title('Sobel X')
# plt.imshow(img_sobel_x, cmap='gray')
# plt.subplot(222)
# plt.title('Sobel Y')
# plt.imshow(img_sobel_y, cmap='gray')
# plt.subplot(223)
# plt.title('Sobel')
# plt.imshow(img_sobel, cmap='gray')
# plt.show()

img_canny = cv.Canny(img, 100, 200)
# show_with_plot(img_canny, "Canny")

kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewitt_x = cv.filter2D(img_gaussian, -1, kernelx)
img_prewitt_y = cv.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewitt_x + img_prewitt_y
# plt.subplot(221)
# plt.title('Prewitt X')
# plt.imshow(img_prewitt_x, cmap='gray')
# plt.subplot(222)
# plt.title('Prewitt Y')
# plt.imshow(img_prewitt_y, cmap='gray')
# plt.subplot(223)
# plt.title('Prewitt')
# plt.imshow(img_prewitt, cmap='gray')
# plt.show()

robert_x = np.array([[-1, 0], [0, 1]])
robert_y = np.array([[0, -1], [1, 0]])
img_robert_x = cv.filter2D(img, -1, robert_x)
img_robert_y = cv.filter2D(img, -1, robert_y)
img_robert = img_robert_x + img_robert_y
# show_with_plot(img_robert, 'Robert')

kernel = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
img_LoG = cv.filter2D(img, -1, kernel)
# show_with_plot(img_LoG, "LoG")

blur = cv.GaussianBlur(img, (3, 3), 0)
laplacian = cv.Laplacian(blur, cv.CV_64F)
zeroscrossing_img = np.zeros(laplacian.shape)
for i in range(1, laplacian.shape[0] - 1):
    for j in range(1, laplacian.shape[1] - 1):
        negative = 0
        positive = 0
        neighbours = [laplacian[i + 1, j - 1], laplacian[i + 1, j], laplacian[i + 1, j + 1], laplacian[i, j - 1], laplacian[i, j + 1],
         laplacian[i - 1, j - 1], laplacian[i - 1, j], laplacian[i - 1, j + 1]]
        for neighbour in neighbours:
            if neighbour > 0:
                positive += 1
            elif neighbour < 0:
                negative += 1
        if (negative > 0) and (positive > 0):
            if laplacian[i, j] > 0:
                zeroscrossing_img[i, j] = laplacian[i, j] + np.abs(min(neighbours))
            elif laplacian[i, j] < 0:
                zeroscrossing_img[i, j] = np.abs(laplacian[i, j]) + max(neighbours)

zerocrossing = np.uint8((zeroscrossing_img / np.max(zeroscrossing_img)) * 255)
# show_with_plot(zerocrossing, "Zero Crossing")


plt.subplot(231)
plt.title('Sobel')
plt.imshow(img_sobel, cmap='gray')
plt.subplot(232)
plt.title('Canny')
plt.imshow(img_canny, cmap='gray')
plt.subplot(233)
plt.title('Prewitt')
plt.imshow(img_prewitt, cmap='gray')
plt.subplot(234)
plt.title('Robert')
plt.imshow(img_robert, cmap='gray')
plt.subplot(235)
plt.title('LoG')
plt.imshow(img_LoG, cmap='gray')
plt.subplot(236)
plt.title('Zero Crossing')
plt.imshow(zerocrossing, cmap='gray')
plt.show()
