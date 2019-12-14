import cv2 as cv

cap = cv.VideoCapture('sky1.avi')
counter = 1
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if counter == 1:
            frame_avg = frame
        else:
            frame_avg = frame_avg_bef * ((counter - 1)/counter) + 1/counter * frame
        if counter == 20:
            cv.imshow('20th frame', frame_avg.astype('uint8'))
        if counter == 40:
            cv.imshow('40th frame', frame_avg.astype('uint8'))
            cv.waitKey(0)
        frame_avg_bef = frame_avg
        counter += 1
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()