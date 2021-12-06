"""
An example of detecting ArUco markers with OpenCV.
"""

import cv2
import sys
import cv2.aruco as aruco


device = 0  # Front camera

cap = cv2.VideoCapture(device)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue

    # Auto rotate camera
    # frame = cv2.autorotate(frame, device)

    # Convert from BGR to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        frame, aruco_dict, parameters=parameters)
    frame = aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the resulting frame

    cv2.imshow('frame', frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
