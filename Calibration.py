import numpy as np
import cv2 as cv
import glob
import os
import pickle
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
path = os.path.dirname(__file__)
images = glob.glob(path + '\Images\*.png')
print(path + '\Images\*.jpg')
print(len(images))
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv.imread('opencv_frame_0.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

d = {"mtx": mtx, "dist": dist, "newcameramtx": newcameramtx, "roi": roi}
with open("prop.b", "wb") as f:
    pickle.dump(d, f)
# crop the image
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
cv.destroyAllWindows()
