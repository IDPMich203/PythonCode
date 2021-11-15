import cv2
import time
cap = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
curr = 0
while cap.isOpened():
    print(time.time() - curr)
    curr = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
