import cv2
import numpy as np
import cv2.aruco as aruco


def find_homography(im1, im2):

    # find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()

    # Find matches
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in matches]
    good_matches = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)

    # Find points and their relations
    query_pts = np.float32([kp1[m.queryIdx]
                            .pt for m in good_matches]).reshape(-1, 1, 2)

    train_pts = np.float32([kp2[m.trainIdx]
                            .pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

    # matchesMask = mask.ravel().tolist()
    return matrix, mask


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 50
params.maxThreshold = 150
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.3

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)


def detect_blobs(frame, low=60, high=90):
    hsv = cv2.blur(frame, (7, 7))
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (low, 0, 0), (high, 255, 255))
    imask = mask > 0
    green = np.zeros_like(frame, np.uint8)
    green[imask] = frame[imask]
    green = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY) * 10
    keypoints = detector.detect(green)

    # return keypoints, green
    return keypoints


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()


def find_markers(im):
    corners, ids, _ = aruco.detectMarkers(
        im, aruco_dict, parameters=parameters)
    # frame = aruco.drawDetectedMarkers(im, corners, ids)
    return corners, ids
