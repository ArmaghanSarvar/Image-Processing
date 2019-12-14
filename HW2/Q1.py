import matplotlib.pyplot as plt
import cv2 as cv
from bitplane import bit_plane_slicing

img = cv.imread('head.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
slices = bit_plane_slicing(img)

fig, axs = plt.subplots(2, 4)
cnt = 7
for i in range(0, 2):
    for j in range(0, 4):
        axs[i, j].imshow(slices[cnt], cmap='gray', vmin=0, vmax=1)
        axs[i, j].set_title(cnt)
        cnt -= 1
plt.show()

four_MSB_bits = 0
for i in range(0, 4):
    four_MSB_bits += slices[7 - i]
plt.imshow(four_MSB_bits, cmap='gray', vmin=0, vmax=255)
plt.show()


two_MSB_bits = 0
for i in range(0, 2):
    two_MSB_bits += slices[7 - i]
plt.imshow(two_MSB_bits, cmap='gray', vmin=0, vmax=255)
plt.show()