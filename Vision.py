import cv2
import numpy as np
import pickle
import utils
import sys
import time
import dbscan


class Vision():
    """Class to wrap all the vision stuff"""

    def __init__(self, send_queue, recieve_queue) -> None:

        self.send_queue = send_queue
        self.recieve_queue = recieve_queue

        cv2.namedWindow('frame')

        # Green thresholds
        self.lower = 46
        self.upper = 80

        # Image height and width
        self.maxHeight = 720
        self.maxWidth = 720

        # Sampels for DBScan
        self.N_samples = 50
        # Reference image
        im1 = cv2.imread('calibresult.png', 0)

        # Scuffed but bottom right, top right, top left, bottom left
        points = [[836, 667],
                  [764, 54],
                  [190, 109],
                  [199, 682]]

        d = pickle.load(open("prop.b", "rb"))
        self.mtx = d["mtx"]
        self.dist = d["dist"]
        self.newcameramtx = d["newcameramtx"]
        roi = d["roi"]

        self.cap = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

        # Capture frame-by-frame
        ret, frame = self.cap.read()

        # # Check if frame is not empty
        if not ret:
            sys.exit()

        im2 = cv2.undistort(frame, self.mtx, self.dist,
                            None, self.newcameramtx)
        matrix, mask = utils.find_homography(im1, im2)

        # stencil = np.zeros(im2.shape).astype(im2.dtype)
        dst = cv2.perspectiveTransform(
            np.float32(points).reshape(-1, 1, 2), matrix)

        # cv2.fillPoly(stencil, [np.int32(dst)], (255, 255, 255))

        output_pts = np.float32([[0, 0],
                                [0, self.maxHeight - 1],
                                [self.maxWidth - 1, self.maxHeight - 1],
                                [self.maxWidth - 1, 0]])

        self.M = cv2.getPerspectiveTransform(dst, output_pts)

        self.searchstencil = np.zeros(
            (self.maxHeight, self.maxWidth, 3)).astype(im2.dtype)

        a = np.int32(
            [[0, 0], [self.maxWidth, self.maxHeight], [
                self.maxWidth, 0]]).reshape(-1, 1, 2)

        cv2.fillPoly(self.searchstencil, [a], (255, 255, 255))

        self.points = np.zeros((self.N_samples, 2))

    def run(self):
        # Keep track of time
        curr = 0

        # Keep track of samples
        curr_sample = 0

        # Only start DBScan after some data collected
        to_go = False

        self.last_known_position = None
        self.last_known_rotation = None

        self.closest_dummy = None
        while self.cap.isOpened():

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            print(time.time() - curr)
            curr = time.time()

            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Check if frame is not empty
            if not ret:
                continue
            orig = cv2.undistort(frame, self.mtx, self.dist,
                                 None, self.newcameramtx)

            # cv2.imshow("frame", frame)
            # continue
            im2 = cv2.warpPerspective(
                orig, self.M, (self.maxWidth, self.maxHeight), flags=cv2.INTER_LINEAR)
            # cv2.imshow('frame', out)
            # continue
            # im2 = cv2.bitwise_and(orig, arenastencil)
            search = cv2.bitwise_and(im2, self.searchstencil)
            # cv2.imshow("frame", search)
            # continue
            keypoints, green = utils.detect_blobs(
                search, low=self.lower, high=self.upper)
            # green = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
            # cv2.imshow("frame", green)
            # continue
            print(len(keypoints))

            corners, ids = utils.find_markers(im2)

            centers = []
            for x in corners:
                x = x[0]
                center: np.ndarray = x.sum(axis=0)
                center /= 4
                center = center.astype(np.int32)
                centers.append(center)
                self.last_known_position = center
                self.last_known_rotation = 90

            for point in centers:
                cv2.circle(im2, (int(point[0]), int(
                    point[1])), 10, (0, 255, 0), 3)

            for x in range(len(keypoints)):
                self.points[curr_sample][0] = keypoints[x].pt[0]
                self.points[curr_sample][1] = keypoints[x].pt[1]
                cv2.circle(im2, (int(keypoints[x].pt[0]), int(
                    keypoints[x].pt[1])), int(10), (255, 0, 255), 1)
                curr_sample += 1
                if curr_sample >= self.N_samples:
                    curr_sample = 0
                    to_go = True

            if to_go:
                clusters = dbscan.dbscan(self.points.T, 10, 5)
                print(clusters)
                num_clusters = len(set(clusters))
                print(num_clusters)
                if dbscan.NOISE in clusters:
                    num_clusters -= 1
                print(num_clusters)

                cluster_positions = np.zeros((num_clusters, 2))
                cluster_lengths = [0] * num_clusters
                for i, cluster in enumerate(clusters):
                    if cluster == dbscan.NOISE:
                        continue

                    cluster_positions[cluster - 1] += self.points[i]
                    cluster_lengths[cluster - 1] += 1

                print(cluster_lengths)

                dummy_positions = []
                # God awful workaround
                mindist = 10000000000
                for i, pos in enumerate(cluster_positions):
                    if not cluster_lengths[i]:
                        continue
                    pos /= cluster_lengths[i]
                    if(self.last_known_position):
                        dist = np.linalg.norm(self.last_known_position - pos)
                        if dist < mindist:
                            mindist = dist
                            self.closest_dummy = [pos[0], pos[1]]

                    cv2.circle(im2, (int(pos[0]), int(
                        pos[1])), int(20), (0, 0, 255), 3)

                # if closest_dummy:
                #     self.closest_dummy = closest_dummy

            cv2.imshow('frame', im2)

            if not self.recieve_queue.qsize():
                continue

            data = self.recieve_queue.get()
            data_str = "F"
            if(data == "robotcoords"):
                if(self.last_known_position is not None):
                    data_str = f"{self.last_known_position[0]}, {self.last_known_position[1]} \n"
                self.send_queue.put(data_str)
            elif(data == "InSearch"):
                if(self.last_known_position is not None):
                    data_str = "y"
                    if self.last_known_position[0] > self.last_known_position[1]:
                        data_str = "n"
                self.send_queue.put(data_str)
            elif(data == "dummycoords"):
                if self.closest_dummy is not None:
                    data_str = f"{self.closest_dummy[0]},{self.closest_dummy[1]} \n"
                self.send_queue.put(data_str)

        self.cap.release()
        print("System shut down")
