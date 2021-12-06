# Python Code

This is a repository to store the python side of IDP M203's codebase.

There are a lot of files in the repository, but only 5 of them were used together for the final competition. The rest were created during testing
and prototyping to explore various methods and techniques of using the computer vision. 

The important files are as follows:

- **Competition.py**: The main file that provides complete functionality. It calls upon the following files:

- **Vision.py**: Most of the computer vision and main logic is contained within this file. The video stream is handled and analysed here.

- **Connection.py**: This file contains the code for the wifi connection to the Arduino. The code is designed to run in a separate thread, in parallel with vision code.

- **Utils.py**: Computer vision related utility functions that have been moved out to improve readability of Vision.py

- **dbscan.py**: An implementation of dbscan by Choffstein from [here](https://github.com/choffstein/dbscan)

Other files include:

- **ArucoDetection.py**: A file to detect and draw aruco markers - used when evaluating if there were a good solution for robot location. Most code is now in Vision and utils

- **Benchmark.py**: A file that simply reads and displays frames from the video stream, to compare performance to.

- **Calibration.py**: Used to calibrate a camera from a series of images of checkerboards

- **CalibrationSampling.py**: A helper file to capture said images of checkerboards from a stream

- **ConnectionTesting.py**: A file to help test connection with the Arduino without computer vision running - it simply reads from the queue and prints, and writes user input

- **Featuredetect.py**: A file to test and prototype feature matching and detection as a method of finding arena bounds. Most code from this file has been integrated into Vision or utils.

- **LineDetection.py**: A file to test and prototype line detection and classification as a method of finding arena bounds. This method was quickly abandoned as it was far harder to implement than feature matching.

- **PointLabel.py**: A helper file to print out pixel coordinates of a location on an image when clicked. Used to determine coordinates of corners and squares in calibresult.png

- **Prototype1.py**: Initial full working prototype of computer vision. Almost identical to Vision.py, save for some quality of code improvements and extra features.

- **SocketTesting.py**: A file to test and evaluate sockets as a method of communication with the Arduino, independent of the computer vision system.

- **UrllibStream.py**: A file to test reading from a stream with urllib. This approach was abandoned in favour of OpenCV's internal video stream support.

- **VideoCapture.py**: A file to test video capture being on a separate thread with frames being requested as and when needed. This would have solved the problem of a growing input frame buffer, but in the end it was not used due to performance degradation.
