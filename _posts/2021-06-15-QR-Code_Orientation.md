---
layout: post
title:  "Orientation of QR code using OpenCV"
date:   2021-06-15 10:00:00 +0000
categories: Python Computer_vision
---

Determining the relative position of an object with respect to a camera viewport is useful in many applications. For example, AR applications need to define a coordinates system before applying augmentations. In our lab, we use QR code or AR markers to identify objects and determine their positions in our work space. We will use openCV in this post to check if a QR code exists in camera frame and if it does, determine the rotation matrix and translation vector from QR coordinate system to camera coordinate system. The full code can be found on my repository here: [link](https://github.com/TemugeB/QR_code_orientation_OpenCV).

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/f000.gif?raw=true">
</p>
<p align="center">
Figure 1. Example output.
</p>

Before we start, you will need to have the intrinsic camera matrix and distortion parameters of your camera. To make this post more complete, a sample video clip and the correponding camera intrinsic parameters are provided. In this demo, the RGB channels of an Intel RealSense D435 camera is used. Some manufacturers will provide intrinsic camera parameters in product description. If not, you can follow my camera calibration guide for the single camera setup here: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html), or follow the OpenCV calibration guide here: [link](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).

First, we add imports and simple code to read the camera parameters. If you're using your own intrinsic parameters, make sure to follow the format in my example or rewrite this code. 
```python
import cv2 as cv
import numpy as np
import sys


def read_camera_parameters(filepath = 'camera_parameters/intrinsic.dat'):

    inf = open(filepath, 'r')

    cmtx = []
    dist = []

    #ignore first line
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    #ignore line that says "distortion"
    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    #cmtx = camera matrix, dist = distortion parameters
    return np.array(cmtx), np.array(dist)

if __name__ == '__main__':

    cmtx, dist = read_camera_parameters()
```

Next, we will define input stream source. If you want to use a webcam, simply call the code with webcam id from command line. Otherwise, the default behavior is to use provided sample video clip. 
```python
if __name__ == '__main__':

    #read camera intrinsic parameters.
    cmtx, dist = read_camera_parameters()

    input_source = 'test.mp4'
    if len(sys.argv) > 1:
        input_source = int(sys.argv[1])

    show_axes(cmtx, dist, input_source)
```

Next, we will create a QR code reader object. Additionally, we will open the input stream source and read frames one by one and show the output.
```python
def show_axes(cmtx, dist, in_source):

    cap = cv.VideoCapture(in_source)

    qr = cv.QRCodeDetector()

    while True:
        ret, img = cap.read()
        if ret == False: break

        cv.imshow('frame', img)

        k = cv.waitKey(20)
        if k == 27: break #27 is ESC key.

    cap.release()
    cv.destroyAllWindows()
```

Next, we run the QR detection algorithm. This can be done with a single call.
```python
ret_qr, points = qr.detect(img)
```
The first returned value indicates if QR code was found or not, and has a boolean value. The second returned variable provides the four corners of the QR code as pixel values in the image. These are shown in Figure 2. If you want to detect QR and decode the value, you can call qr.detectAndDecode(). 


<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/QR_points.png?raw=true">
</p>
<p align="center">
Figure 2. Points returned by QR detector.
</p>

If QR code is found, then we can use the four detected corners to define a coordinate system. In this demo, point #1 is defined as origin. OpenCV uses a right hand coordinate system. To have the QR coordinate axes to point up, we have to chose x axis to be pointing from point #1 to #4 and y axis to point from #1 to #2. This will point z axis up from the QR code. To get this coordinate system, we have to assign coordinate values to each point (#1 through #4). The assigned coordinate values are shown in Figure 2. Once we have the pixel values of the corners and our assigned coordinate for each point, we can determine the rotation matrix and translation vector using OpenCV functionality. 
```python
def get_qr_coords(cmtx, dist, points):

    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)
```

The internal method used by cv.solvePnP to get rotation and translation data can be found here: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html). If rotation matrix and translation vector are successfully found, we simply reproject unit x,y,z vectors to camera pixel values to draw them. Note however that the returned rotation matrix is in Rodrigues vector form. If you want to matrix form, call cv.Rodrigues() to convert between the two.
```python
def get_qr_coords(cmtx, dist, points):

    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)

    #Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    if ret:
        points, jac = cv.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        #the returned points are pixel coordinates of each unit vector.
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else: return [], [], []
```

To finish things up, we simply call the get_qr_coords() function whenever QR code is detected. This returns the unit vector positions that define the QR code coordinate system. Additionally, we return rotation and translation information for the camera from this defined coordinate system.
```python
    while True:
        ret, img = cap.read()
        if ret == False: break

        ret_qr, points = qr.detect(img)

        if ret_qr:
            axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)

            #BGR color format
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,0,0)]

            #check axes points are projected to camera view.
            if len(axis_points) > 0:
                axis_points = axis_points.reshape((4,2))

                origin = (int(axis_points[0][0]),int(axis_points[0][1]) )

                for p, c in zip(axis_points[1:], colors[:3]):
                    p = (int(p[0]), int(p[1]))

                    #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases. 
                    if origin[0] > 5*img.shape[1] or origin[1] > 5*img.shape[1]:break
                    if p[0] > 5*img.shape[1] or p[1] > 5*img.shape[1]:break

                    cv.line(img, origin, p, c, 5)

        cv.imshow('frame', img)

        k = cv.waitKey(20)
        if k == 27: break #27 is ESC key.
```

If you need the position of the QR code with respect to the camera, then the position is saved in tvec. That is, the location of the QR code from the camera is tvec. On the other hand, if you need to location of the camera from the new QR code coordinates, then this can be calculated as:
```python
rvec, jacobian = cv.Rodrigues(rvec)
camera_position = -rvec.transpose() @ tvec
```

For closing notes, in this demo, we've used QR code to obtain the corner points. But in general, QR code is not necessary. As long as we can determine four in-plane points, the above method will work in determining orientation and location. For example, AR marker could be used to obtain the four corners, from which we can again define a coordinate system. 

The accuracy of the QR detector is determined by how well the camera can focus on the QR code. In practice, focus of the camera should change as we get closer to the QR code. When the camera or the QR code moves fast, then the image is blurry and we observe poor performance from the QR detector. Additionally, if focal point of the camera changes, then the intrinsic parameters of the camera also changes. This mean the code above must dynamically create the intrinic parameters. 

Finally, if you don't care about the encoded message in the QR code, I find that ARUCO markers are much more relaiably detected than QR code. The ARUCO markers can give you a marker index, which you can use to look up external information. OpenCV supports ARUCO markers so minimal changes are required to support ARUCO markers. 
