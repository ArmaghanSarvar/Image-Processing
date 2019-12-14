import cv2 as cv
import matplotlib.pyplot as plt


def save_as_png(name, image):
    cv.imwrite(name + '.png', image)


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
    

# 1
img = cv.imread('brains.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 2
show_with_plot(img, 'Brains')

# 3
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show_with_plot(gray, "gray")

# 4
gray_64 = gray//4
gray_16 = gray//16
gray_2 = gray//128
show_with_plot(gray_64 * 4, 'gray 64')
show_with_plot(gray_16 * 17, 'gray 16')
show_with_plot(gray_2 * 255, 'gray 2')

# 5
ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
show_with_plot(binary, 'gray binary')
double_img = cv.normalize(gray.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
show_with_plot(double_img, 'gray double')

# 6
save_as_png('graypng', gray)

# 7
xax, yax, rgb = img.shape
width = yax//3
left_img = img[:, :width, :]
middle_img = img[:, width:2 * width, :]
right_img = img[:, 2*width:, :]
plt.subplot(131)
plt.title('left brain')
plt.imshow(left_img)
plt.subplot(132)
plt.title('middle brain')
plt.imshow(middle_img)
plt.subplot(133)
plt.title('right brain')
plt.imshow(right_img)
plt.show()

# 8
if img.shape[1] % 3 is not 0:
    right_img = img[:, 2 * width: 3 * width, :]
left_img_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
middle_img_gray = cv.cvtColor(middle_img, cv.COLOR_BGR2GRAY)
right_img_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
total_img = (left_img_gray + middle_img_gray + right_img_gray)//3
show_with_plot(total_img, 'total gray image')
save_as_png('totalpng', total_img)

# 9
scale = 4
new_shape = (xax * scale, yax * scale)
resized_1 = cv.resize(img, new_shape, interpolation=cv.INTER_NEAREST)
show_with_plot(resized_1, 'gray resized nearest-4')
resized_2 = cv.resize(img, new_shape, interpolation=cv.INTER_LINEAR)
show_with_plot(resized_2, 'gray resized linear-4')
resized_3 = cv.resize(img, new_shape, interpolation=cv.INTER_AREA)
show_with_plot(resized_3, 'gray resized area-4')


scale = 0.25
new_shape = (int(xax * scale), int(yax * scale))
resized_1 = cv.resize(img, new_shape, interpolation=cv.INTER_NEAREST)
show_with_plot(resized_1, 'gray resized nearest-0.25')
resized_2 = cv.resize(img, new_shape, interpolation=cv.INTER_LINEAR)
show_with_plot(resized_2, 'gray resized linear-0.25')
resized_3 = cv.resize(img, new_shape, interpolation=cv.INTER_AREA)
show_with_plot(resized_3, 'gray resized area-0.25')


# 10
flip_hor = cv.flip(middle_img, 1)
show_with_plot(flip_hor, 'h flipped image')
flip_ver = cv.flip(middle_img, 0)
show_with_plot(flip_ver, 'v flipped image')

cv.waitKey(0)
cv.destroyAllWindows()