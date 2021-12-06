
import cv2
import urllib.request

import numpy as np

stream = urllib.request.urlopen('http://localhost:8081/stream/video.mjpeg')
bytes = b''
while True:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
        cv2.imshow('i', i)
        if cv2.waitKey(1) == 27:
            exit(0)

img = cv2.imread('src.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
