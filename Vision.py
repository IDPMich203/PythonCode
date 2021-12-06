import cv2
import numpy as np
import pickle
import utils
import sys
import time
import dbscan


class Vision():
    """Class to wrap all the vision stuff"""

    def __init__(self, from_arduino, to_arduino) -> None:

        self.to_arduino = to_arduino
        self.from_arduino = from_arduino

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

        # Measured arena corners in pixel coordinates in calibresult.png
        points = [[836, 667],
                  [764, 54],
                  [190, 109],
                  [199, 682]]

        # Load distortion coefficients
        d = pickle.load(open("prop.b", "rb"))
        self.mtx = d["mtx"]
        self.dist = d["dist"]
        self.newcameramtx = d["newcameramtx"]
        roi = d["roi"]

        # Open capture
        self.cap = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

        # Capture a frame
        ret, frame = self.cap.read()

        # # Check if frame is not empty
        if not ret:
            sys.exit()

        # Undistort frame
        im2 = cv2.undistort(frame, self.mtx, self.dist,
                            None, self.newcameramtx)

        # Determine a homography matrix between the two images
        matrix, mask = utils.find_homography(im1, im2)

        # Work out pixel coordinates of arena corners in recieved image
        dst = cv2.perspectiveTransform(
            np.float32(points).reshape(-1, 1, 2), matrix)

        # Work out pixel coordinates of target squares in recieved image
        redbluewhite = [[550, 177], [663, 276], [253, 620]]
        redbluewhite = cv2.perspectiveTransform(
            np.float32(redbluewhite).reshape(-1, 1, 2), matrix)

        # Transform arena to region of dimension (maxWidth, maxHeight)corner points

        output_pts = np.float32([[0, 0],
                                [0, self.maxHeight - 1],
                                [self.maxWidth - 1, self.maxHeight - 1],
                                [self.maxWidth - 1, 0]])

        self.M = cv2.getPerspectiveTransform(dst, output_pts)

        # Correct target square coordinates
        self.corrected_rbw = cv2.perspectiveTransform(redbluewhite, self.M)

        # A mask to mask out regions that are not of interest to dummy location (e.g only leaves search area)
        self.searchstencil = np.zeros(
            (self.maxHeight, self.maxWidth, 3)).astype(im2.dtype)

        # Fill the search stencil with white over the search area
        a = np.int32(
            [[0, 0], [self.maxWidth, self.maxHeight], [
                self.maxWidth, 0]]).reshape(-1, 1, 2)

        cv2.fillPoly(self.searchstencil, [a], (255, 255, 255))

        # DBScan points
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

            # If it takes less that 10ms to get a frame it was probably buffered so we skip it to get the latest one
            curr = time.time()
            ret, frame = None, None
            while (time.time() - curr) < 0.01:
                curr = time.time()
                ret, frame = self.cap.read()

            # Check if frame is not empty
            if not ret:
                continue

            # Undistort frame
            orig = cv2.undistort(frame, self.mtx, self.dist,
                                 None, self.newcameramtx)

            # Project arena to rectangular region
            im2 = cv2.warpPerspective(
                orig, self.M, (self.maxWidth, self.maxHeight), flags=cv2.INTER_LINEAR)

            """ROBOT LOCATION AND POSTIION"""

            # Find aruco markers
            corners, ids = utils.find_markers(im2)

            # Find average of all the corners to determine center of Aruco marker
            centers = []
            for x in corners:
                x = x[0]
                center: np.ndarray = x.sum(axis=0)
                center /= 4
                center = center.astype(np.int32)
                centers.append(center)
                self.last_known_position = center
                self.last_known_rotation = 90

            # Determine rotation from offset between the center and corner[0]
            if corners:
                corner = corners[0][0][0]
                self.last_known_rotation = utils.angle_between(
                    np.array([0, 1]), corner - center)

            # Draw debug circle
            for point in centers:
                cv2.circle(im2, (int(point[0]), int(
                    point[1])), 10, (0, 255, 0), 3)

            """DUMMY DETECTION"""

            # Determine search area
            search = cv2.bitwise_and(im2, self.searchstencil)

            # Detect blobs in the image
            keypoints, green = utils.detect_blobs(
                search, low=self.lower, high=self.upper)

            # Draw detected blobs and add them samples
            for x in range(len(keypoints)):
                self.points[curr_sample][0] = keypoints[x].pt[0]
                self.points[curr_sample][1] = keypoints[x].pt[1]
                cv2.circle(im2, (int(keypoints[x].pt[0]), int(
                    keypoints[x].pt[1])), int(10), (255, 0, 255), 1)
                curr_sample += 1

                # If we have enough samples start DBSCAN
                if curr_sample >= self.N_samples:
                    curr_sample = 0
                    to_go = True

            if to_go:
                # Run DBSCAN on blobs
                clusters = dbscan.dbscan(self.points.T, 10, 5)

                # Determine number of actual clusters (not noise)
                num_clusters = len(set(clusters))
                if dbscan.NOISE in clusters:
                    num_clusters -= 1

                # Add all the positions in each cluster to average them and keep track of length
                cluster_positions = np.zeros((num_clusters, 2))
                cluster_lengths = [0] * num_clusters
                for i, cluster in enumerate(clusters):
                    if cluster == dbscan.NOISE:
                        continue

                    cluster_positions[cluster - 1] += self.points[i]
                    cluster_lengths[cluster - 1] += 1

                # God awful workaround - starting maximum value that distance should never be greater than
                mindist = 10000000000

                # Find average point and then determine if it is closest by looping through each distance as we compute them
                for i, pos in enumerate(cluster_positions):
                    if not cluster_lengths[i]:
                        continue
                    pos /= cluster_lengths[i]
                    if(self.last_known_position is not None):
                        dist = np.linalg.norm(self.last_known_position - pos)
                        if dist < mindist:
                            mindist = dist
                            self.closest_dummy = [pos[0], pos[1]]

                    # Draw a thicker circle for detected dummies
                    cv2.circle(im2, (int(pos[0]), int(
                        pos[1])), int(20), (0, 0, 255), 3)

            # Draw circles on target squares
            cv2.circle(im2, (int(self.corrected_rbw[0][0][0]), int(
                self.corrected_rbw[0][0][1])), 30, (255, 0, 0), 4)

            cv2.circle(im2, (int(self.corrected_rbw[1][0][0]), int(
                self.corrected_rbw[1][0][1])), 30, (0, 0, 255), 4)

            cv2.circle(im2, (int(self.corrected_rbw[2][0][0]), int(
                self.corrected_rbw[2][0][1])), 30, (255, 255, 255), 4)

            # Display image
            cv2.imshow('frame', im2)

            # No data to do anything with
            if not self.from_arduino.qsize():
                continue

            # Check if command recieved matches a known command
            data: str = self.from_arduino.get()
            print("Got: ", data)
            data = data.strip()
            data_str = "F"

            # Send robot x and y values seperately (not used)
            if(data == "robotx"):
                if(self.last_known_position is not None):
                    data_str = str(self.last_known_position[0])
            elif(data == "roboty"):
                if(self.last_known_position is not None):
                    data_str = str(self.last_known_position[1])

            # Send robot position as comma separated integers
            elif(data == "robotloc"):
                if(self.last_known_position is not None):
                    data_str = ",".join(self.last_known_position)

            # Send robot rotation as a float
            elif(data == "robotrot"):
                if(self.last_known_rotation is not None):
                    data_str = str(self.last_known_rotation)

            # Send blue/red/white box coordinates as comma separated integers
            elif(data == "boxb"):
                data_str = self.rbwtostr(0)

            elif(data == "boxr"):
                data_str = self.rbwtostr(1)

            elif(data == "boxw"):
                data_str = self.rbwtostr(2)

            # The same but separately (not used)
            elif(data == "boxbx"):
                data_str = str(self.corrected_rbw[0][0][0])
            elif(data == "boxby"):
                data_str = str(self.corrected_rbw[0][0][1])
            elif(data == "boxrx"):
                data_str = str(self.corrected_rbw[1][0][0])
            elif(data == "boxry"):
                data_str = str(self.corrected_rbw[1][0][1])
            elif(data == "boxwx"):
                data_str = str(self.corrected_rbw[2][0][0])
            elif(data == "boxwy"):
                data_str = str(self.corrected_rbw[2][0][1])

            # Y if robot is in search area, n if not - used to check if we are seeing a dummy or the ramp
            elif(data == "InSearch"):
                if(self.last_known_position is not None):
                    data_str = "y"
                    if self.last_known_position[0] > self.last_known_position[1]:
                        data_str = "n"

            # Send coordinates of the closest dummy
            elif(data == "dummyloc"):
                if(self.closest_dummy is not None):
                    data_str = ",".join(self.closest_dummy)

            # Send x and y separately (not used)
            elif(data == "dummyx"):
                if self.closest_dummy is not None:
                    data_str = str(self.closest_dummy[0])
            elif(data == "dummyy"):
                if self.closest_dummy is not None:
                    data_str = str(self.closest_dummy[1])

            # Put to queue to be sent
            print("Sending: ", data_str)
            self.to_arduino.put(data_str)

        self.cap.release()
        print("System shut down")

    # Helper function to send box coordinates
    def rbwtostr(self, index: int) -> str:
        return str(int(self.corrected_rbw[index][0][0])) + "," + str(int(self.corrected_rbw[index][0][1]))
