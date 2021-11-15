"""
An example of detecting ArUco markers with OpenCV.
"""

import cv2
import sys
import cv2.aruco as aruco
import numpy as np

device = 0  # Front camera

cap = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


params = cv2.SimpleBlobDetector_Params()


# Change thresholds

params.minThreshold = 10

params.maxThreshold = 200


params.filterByArea = True

params.minArea = 10

# Filter by Circularity

params.filterByCircularity = True

params.minCircularity = 0.01


# Filter by Convexity

params.filterByConvexity = True

params.minConvexity = 0.01


# Filter by Inertia

params.filterByInertia = True

params.minInertiaRatio = 0.01


# Create a detector with the parameters

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

confidence = None
numpoints = 0
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 10, 10), (70, 255, 255))

    # slice the green
    imask = mask > 0
    green = np.zeros_like(frame, np.uint8)
    green[imask] = frame[imask]
    green = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY) * 10
    keypoints = detector.detect(green)
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

    numpoints += len(keypoints)
    if confidence is None:
        print("A")
        height, width, _ = frame.shape
        print(width, height)
        confidence = np.zeros((height, width))

    # im_with_keypoints = cv2.drawKeypoints(confidence, keypoints, np.array(
    #     []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for x in range(len(keypoints)):
        a = create_circular_mask(confidence.shape[0], confidence.shape[1], center=(
            keypoints[x].pt[0], keypoints[x].pt[1]), radius=10)
        confidence += a * 5

    heat = confidence / numpoints
    # heat = confidence
    print(numpoints)
    print(confidence.max())
    heat = heat.astype(np.uint8)
    heat *= 255
    heat = cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    added_image = cv2.addWeighted(frame, 0.5, heat, 1, 0)
    cv2.imshow('frame', added_image)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
