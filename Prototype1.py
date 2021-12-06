import cv2
import numpy as np
import pickle
import utils
import sys
import time
import dbscan
from VideoCapture import VideoCapture

# TODO: tweak blob detection parameters
# TODO: robot rotation

lower = 0
upper = 255


def set_lower(x):
    global lower
    lower = x


def set_upper(x):
    global upper
    upper = x


cv2.namedWindow('frame')
# cv2.createTrackbar('low', 'frame', 0, 255, set_lower)
# cv2.createTrackbar('high', 'frame', 0, 255, set_upper)
lower = 46
upper = 80

im1 = cv2.imread('calibresult.png', 0)


# Scuffed but bottom right, top right, top left, bottom left
points = [[836, 667],
          [764, 54],
          [190, 109],
          [199, 682]]

redbluewhite = [[550, 177], [663, 276], [715, 112]]


d = pickle.load(open("prop.b", "rb"))
mtx = d["mtx"]
dist = d["dist"]
newcameramtx = d["newcameramtx"]
roi = d["roi"]


cap = cv2.VideoCapture(
    "http://localhost:8081/stream/video.mjpeg")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
# cap = cv2.VideoCapture(0)

ret = False
while not ret:
    ret, frame = cap.read()
# print(frame)
im2 = cv2.undistort(frame, mtx, dist,
                    None, newcameramtx)
matrix, mask = utils.find_homography(im1, im2)

# stencil = np.zeros(im2.shape).astype(im2.dtype)
dst = cv2.perspectiveTransform(
    np.float32(points).reshape(-1, 1, 2), matrix)

redbluewhite = cv2.perspectiveTransform(
    np.float32(redbluewhite).reshape(-1, 1, 2), matrix)

maxHeight = 720
maxWidth = 720

output_pts = np.float32([[0, 0],
                         [0, maxHeight - 1],
                         [maxWidth - 1, maxHeight - 1],
                         [maxWidth - 1, 0]])

print(output_pts.shape)
M = cv2.getPerspectiveTransform(dst, output_pts)

corrected_rbw = cv2.perspectiveTransform(redbluewhite, M)
print(corrected_rbw)

# arenastencil = np.zeros((1080,1080)).astype(im2.dtype)
searchstencil = np.zeros((maxHeight, maxWidth, 3)).astype(im2.dtype)

# cv2.fillPoly(arenastencil, [np.int32(dst)], (255, 255, 255))

a = np.int32(
    [[0, 0], [maxWidth, maxHeight], [
        maxWidth, 0]]).reshape(-1, 1, 2)

cv2.fillPoly(searchstencil, [a], (255, 255, 255))

# dst = cv2.perspectiveTransform(np.float32(


#     [points[0], points[2], points[3]]).reshape(-1, 1, 2), homomatrix)


numpoints = 0
confidence = None
curr = 0

N_samples = 50
points = np.zeros((N_samples, 2))
curr_sample = 0
to_go = False
while cap.isOpened():

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # print(time.time() - curr)
    curr = time.time()
    ret, frame = None, None
    while (time.time() - curr) < 0.01:
        curr = time.time()
        ret, frame = cap.read()
    # print("N frames, ", cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Capture frame-by-frame

    # while ret:
    # cap.read()
    # cap.read()
    # ret, frame = cap.read()
    # ret = False

    # Check if frame is not empty
    if not ret:
        continue
    orig = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # cv2.imshow("frame", frame)
    # continue
    im2 = cv2.warpPerspective(
        orig, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # cv2.imshow('frame', out)
    # continue
    # im2 = cv2.bitwise_and(orig, arenastencil)
    search = cv2.bitwise_and(im2, searchstencil)
    # cv2.imshow("frame", search)
    # continue
    keypoints, green = utils.detect_blobs(search, low=lower, high=upper)
    # green = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
    # cv2.imshow("frame", green)
    # continue
    # print(len(keypoints))

    corners, ids = utils.find_markers(im2)

    centers = []
    for x in corners:
        x = x[0]
        center: np.ndarray = x.sum(axis=0)
        center /= 4
        center = center.astype(np.int32)
        centers.append(center)

    if corners:
        corner = corners[0][0][0]
        # print(corner)
        print(utils.angle_between(np.array([0, 1]), corner - center))
    for point in centers:
        cv2.circle(im2, (int(point[0]), int(point[1])), 10, (0, 255, 0), 3)

    for x in range(len(keypoints)):
        points[curr_sample][0] = keypoints[x].pt[0]
        points[curr_sample][1] = keypoints[x].pt[1]
        cv2.circle(im2, (int(keypoints[x].pt[0]), int(
            keypoints[x].pt[1])), int(10), (255, 0, 255), 1)
        curr_sample += 1
        if curr_sample >= N_samples:
            curr_sample = 0
            to_go = True
    if to_go:
        clusters = dbscan.dbscan(points.T, 10, 5)
        num_clusters = len(set(clusters))
        if dbscan.NOISE in clusters:
            num_clusters -= 1

        cluster_positions = np.zeros((num_clusters, 2))
        cluster_lengths = [0] * num_clusters
        for i, cluster in enumerate(clusters):
            if cluster == dbscan.NOISE:
                continue

            cluster_positions[cluster - 1] += points[i]
            cluster_lengths[cluster - 1] += 1

        dummy_positions = []
        for i, pos in enumerate(cluster_positions):
            if not cluster_lengths[i]:
                continue
            pos /= cluster_lengths[i]
            # print(pos)
            cv2.circle(im2, (int(pos[0]), int(
                pos[1])), int(20), (0, 0, 255), 3)

    cv2.circle(im2, (int(corrected_rbw[0][0][0]), int(
        corrected_rbw[0][0][1])), 30, (255, 0, 0), 4)

    cv2.circle(im2, (int(corrected_rbw[1][0][0]), int(
        corrected_rbw[1][0][1])), 30, (0, 0, 255), 4)

    cv2.circle(im2, (int(corrected_rbw[2][0][0]), int(
        corrected_rbw[2][0][1])), 30, (255, 255, 255), 4)

    # pos -= bottom_left

    cv2.imshow('frame', im2)


cap.release()
