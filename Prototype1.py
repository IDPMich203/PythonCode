from typing import no_type_check
import cv2
import numpy as np
import pickle
import utils
import sys
import time
import dbscan

# TODO: mask before feature detect
# TODO: arena-space coordinates
# TODO: tweak blob detection parameters


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


im1 = cv2.imread('calibresult.png', 0)

points = [[836, 667],
          [764, 54],
          [196, 111],
          [196, 680]]

d = pickle.load(open("prop.b", "rb"))
mtx = d["mtx"]
dist = d["dist"]
newcameramtx = d["newcameramtx"]
roi = d["roi"]


cap = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
# cap = cv2.VideoCapture(0)

# Capture frame-by-frame
ret, frame = cap.read()

# # Check if frame is not empty
if not ret:
    sys.exit()

im2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
matrix, mask = utils.find_homography(im1, im2)

stencil = np.zeros(im2.shape).astype(im2.dtype)
dst = cv2.perspectiveTransform(
    np.float32(points).reshape(-1, 1, 2), matrix)
cv2.fillPoly(stencil, [np.int32(dst)], (255, 255, 255))

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
    print(time.time() - curr)
    curr = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue
    im2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    im2 = cv2.bitwise_and(im2, stencil)
    keypoints = utils.detect_blobs(im2, low=47, high=68)
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

            cluster_positions[cluster - 1] += points[i]
            cluster_lengths[cluster - 1] += 1

        print(cluster_lengths)
        for i, pos in enumerate(cluster_positions):
            if not cluster_lengths[i]:
                continue
            pos /= cluster_lengths[i]
            print(pos)
            cv2.circle(im2, (int(pos[0]), int(
                pos[1])), int(20), (0, 0, 255), 3)

    # print(dst)
    for point in dst:
        pnt = point[0]
        cv2.circle(im2, (int(pnt[0]), int(pnt[1])), 5, (255, 0, 0), 2)

    cv2.imshow('frame', im2)


cap.release()
