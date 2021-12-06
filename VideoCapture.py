import cv2
import queue
import threading
# import multiprocessing as mp
import time
from cv2 import CAP_PROP_BUFFERSIZE

# bufferless VideoCapture


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(CAP_PROP_BUFFERSIZE, 3)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        time.sleep(1)
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.cap.isOpened():
            # print("N:frames: ", cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            # ret = self.cap.read()
            # if not ret:
            #     continue

    def IsOpened(self):
        return self.cap.isOpened()

    def read(self):
        # return self.cap.read()
        return True, self.q.get()
