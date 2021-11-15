import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
img1 = cv2.imread('calibresult.png', 0)          # queryImage


points = [[836, 667],
          [764, 54],
          [196, 111],
          [196, 680]]
img2 = cv2.imread('opencv_frame_0.png', 0)  # trainImage

d = pickle.load(open("prop.b", "rb"))
mtx = d["mtx"]
dist = d["dist"]
newcameramtx = d["newcameramtx"]
roi = d["roi"]
sift = cv2.SIFT_create(1000)
kp1, des1 = sift.detectAndCompute(img1, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()   # or pass empty dictionary


flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue

    img2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # d = {"mtx": mtx, "dist": dist, "newcameramtx": newcameramtx, "roi": roi}

    # Initiate SIFT detector
    # sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in matches]

    good_matches = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)

    query_pts = np.float32([kp1[m.queryIdx]
                            .pt for m in good_matches]).reshape(-1, 1, 2)

    train_pts = np.float32([kp2[m.trainIdx]
                            .pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    dst = cv2.perspectiveTransform(
        np.float32(points).reshape(-1, 1, 2), matrix)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # while(1):
    #     # cv2.imshow('test', img1)
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    #     cv2.imshow("Homography", homography)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                           good_matches, None, **draw_params)

    cv2.imshow('frame', img3)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
