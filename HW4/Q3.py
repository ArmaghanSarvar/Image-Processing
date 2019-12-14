import cv2 as cv
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

fourcc = VideoWriter_fourcc(*'XVID')
cap = cv.VideoCapture('images/HW.avi')
ret, first_frame = cap.read()
average_value = np.float32(first_frame)
out = VideoWriter('output.avi', fourcc, 20.0, (first_frame.shape[1], first_frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    cv.accumulateWeighted(frame, average_value, 0.05)
    resulting_avg = cv.convertScaleAbs(average_value)
    resulting_avg_gray = cv.cvtColor(resulting_avg, cv.COLOR_BGR2GRAY)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    difference = cv.absdiff(resulting_avg_gray, frame_gray)
    _, difference = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)
    cv.imshow('Difference', difference)
    frame[np.where(difference == 255)] = (0, 0, 255)
    out.write(frame)

    if ret:
        cv.imshow('Frame', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
