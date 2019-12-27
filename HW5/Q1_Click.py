import cv2 as cv
import numpy as np

x1, y1 = -1, -1
points_1 = []
points_2 = []


def getPoints(path, point_list):
    def draw_circle(event, x, y, flags, param):
        global x1, y1
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(img, (x, y), 2, (255, 0, 255), -1)
            x1, y1 = x, y

    img = cv.imread(path)
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)

    while True:
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('b'):
            print(x1, y1)
            point_list.append((x1, y1))

    cv.destroyAllWindows()


getPoints('images/MRI1.png', points_1)
print("----------------------------------")
getPoints('images/MRI2.png', points_2)

pts1 = np.float32([points_1[0], points_1[1], points_1[2]])
pts2 = np.float32([points_2[0], points_2[1], points_2[2]])

Affine = cv.getAffineTransform(pts2, pts1)
print(Affine)


test = cv.imread('images/MRI1.png')
height, width = test.shape[:2]
final = cv.warpAffine(test, Affine, (width, height))
cv.imshow('Result', final)
cv.waitKey(0)
