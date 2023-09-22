---
layout: post
title:  "Stereo Camera Calibration and Triangulation with OpenCV and Python"
date:   2021-02-02 10:00:00 +0000
categories: OpenCV Python
---

2022.06 UPDATE: The code in this post has be updated and turned into a package. If you just want the calibration results, go here: [Stereo Camera Calibration](https://github.com/TemugeB/python_stereo_camera_calibrate)

In this post, I show how to calibrate two cameras looking at the same view using a checkerboard pattern. Next, I show how to triangulate a 3D point based on the observed pixel coordinates of the two cameras.

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/triangulate.png?raw=true">
</p>

The flow of this demo will be:

  1. Calibrate each camera separately using the checkerboard pattern.
  2. Calibrate stereo camera setup.
  3. Use direct linear transform(DLT) to triangulate camera pixels to 3D coordinates.

I assume that you have already taken checkerboard pattern videos by both cameras. Make sure that your cameras are synchronized so that both frames see the same checkerboard pattern at the same time. If you don’t have video available, you can download my calibration frames here: [link](https://drive.google.com/file/d/1o6OVbxi6dT0kDinmUQaBrEZBF3vz_kDp/view?usp=sharing). Put each folder in the zip file next to your code script.  

**Calibrating Single View Cameras**

The cameras are first calibrated individually. This is recommended because the number of parameters that need to be fitted are large for the stereo calibration case.

We first read in the calibration frames:

```python
import cv2 as cv
import glob
import numpy as np
 
images_folder = 'D2/*'
images_names = sorted(glob.glob(images_folder))
images = []
for imname in images_names:
    im = cv.imread(imname, 1)
    images.append(im)    

```

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/frames.png?raw=true">
</p>
<p align="center">
Sample frames.
</p>

Next, we need to detect the checkerboard patterns. This can be done by using opencv builtin functions. Opencv assumes that the bottom left of the checkerboard pattern is the world coordinate. This means each frame we use to calibrate gets a separate world coordinate. However, in the coordinate space of the checkerboard, we can easily define the coordinates of each of the square corners. Make sure to set the number of rows and columns as in the image below. Notice that the rows and columns are not equal to the actual number of squares in the checkerboard.


```python
#criteria used by checkerboard pattern detector.
#Change this if the code can't find the checkerboard
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
rows = 5 #number of checkerboard rows.
columns = 8 #number of checkerboard columns.
world_scaling = 1. #change this to the real world square size. Or not.
 
#coordinates of squares in the checkerboard world space
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
objp = world_scaling* objp
 
 
#frame dimensions. Frames should be the same size.
width = images[0].shape[1]
height = images[0].shape[0]
 
#Pixel coordinates of checkerboards
imgpoints = [] # 2d points in image plane.
 
#coordinates of the checkerboard in checkerboard world space.
objpoints = [] # 3d point in real world space
 
 
for frame in images:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    #find the checkerboard
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
    if ret == True:
 
        #Convolution size used to improve corner detection. Don't make this too large.
        conv_size = (11, 11)
 
        #opencv can attempt to improve the checkerboard coordinates
        corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
        cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
        cv.imshow('img', frame)
        k = cv.waitKey(500)
 
        objpoints.append(objp)
        imgpoints.append(corners)
```

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/checkboard_detect.png?raw=true">
</p>
<p align="center">
Detected 5x8 checkerboard pattern.
</p>

Then camera calibration can be done with a single function call. The returned values are: RMSE, camera matrix, distortion coefficients, per frame rotations and translations. Of importance is the RMSE value. This gives the per pixel projection error. Anything under 1 is very good. You might be able to get away with up to 2 or 3. My provided images return a value of 0.44 for camera#1 and 0.30 for camera#2.

```python
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)
```

Next, we simply wrap the above code into a function and call it twice for both camera frames. This gives us the camera matrices and distortion coefficients of both cameras. We can now keep the calibrated camera matrices and pass it to stereo calibration function. The full single camera calibration code is:

```python
import cv2 as cv
import glob
import numpy as np
 
def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist
 
mtx1, dist1 = calibrate_camera(images_folder = 'D2/*')
mtx2, dist2 = calibrate_camera(images_folder = 'J2/*')
```

**Stereo Calibration**

We now attempt stereo calibration. I assume you have the camera matrices and distortion coefficients of both cameras from last step. Our first step is to read synchronized frames from both cameras. If you’re using the images I provide, they are stored in the ‘synched’ folder. Make sure to read the images in correct order. Otherwise, the calibration will not work.

```python
#read the synched frames
images_names = glob.glob('synched/*')
images_names = sorted(images_names)
c1_images_names = images_names[:len(images_names)//2]
c2_images_names = images_names[len(images_names)//2:]
 
c1_images = []
c2_images = []
for im1, im2 in zip(c1_images_names, c2_images_names):
    _im = cv.imread(im1, 1)
    c1_images.append(_im)
 
    _im = cv.imread(im2, 1)
    c2_images.append(_im)
```

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/stereo_view.png?raw=true">
</p>
<p align="center">
Synched frames from 2 cameras.
</p>


Next, we again find the checkerboard patterns on the two camera frames. If your frames are already synchronized in the previous step, then the image coordinates will work without having to find them again. If you’re using the images I provide, then the frames are not synchronized in the previous step and you need to read the images from the ‘synched’ folder. Otherwise, the code is the same as before.

```python 
#change this if stereo calibration not good.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
rows = 5 #number of checkerboard rows.
columns = 8 #number of checkerboard columns.
world_scaling = 1. #change this to the real world square size. Or not.
 
#coordinates of squares in the checkerboard world space
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
objp = world_scaling* objp
 
#frame dimensions. Frames should be the same size.
width = c1_images[0].shape[1]
height = c1_images[0].shape[0]
 
#Pixel coordinates of checkerboards
imgpoints_left = [] # 2d points in image plane.
imgpoints_right = []
 
#coordinates of the checkerboard in checkerboard world space.
objpoints = [] # 3d point in real world space
 
for frame1, frame2 in zip(c1_images, c2_images):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
    c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
 
    if c_ret1 == True and c_ret2 == True:
        corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
        cv.drawChessboardCorners(frame1, (5,8), corners1, c_ret1)
        cv.imshow('img', frame1)
 
        cv.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
        cv.imshow('img2', frame2)
        k = cv.waitKey(0)
 
        objpoints.append(objp)
        imgpoints_left.append(corners1)
        imgpoints_right.append(corners2)
```

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/stereo_checkerboard_detect.png?raw=true">
</p>


Once we have the pixel coordinates of the checkerboard in each image, we simply calibrate the stereo camera setup with a single function call. But we have to tell the function call to keep the camera matrices constant.

```python
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
```

The return values of the function call are: RMSE, C1 camera matrix (unchanged), C1 distortion coefficients, C2 camera matrix (unchanged), C2 distortion coefficients, rotation matrix, translation vector, essential matrix and fundamental matrix. The rotation matrix obtained here is the coordinate rotation matrix to go from C1 coordinate system to C2 coordinate system. The translation vector is also the location of C2 from C1. So it does not contain world coordinate rotation and translation vectors, it only provides C2 position and rotation with respect to C1. To obtain world coordinate to C2 rotation and translation, calculate as:

**R**2 = **R** * **R**1 and **T**2 = **R**T1 + **T**

One way to get **R**2 and **T**2 is to use the rotation and translation matrices obtained in single camera calibration from previous step as **R**1 and **T**1. This means your world coordinate will overlap the bottom left grid of the checkerboard in that frame. If you need absolute world coordinates, you need to determine **R**1 and **T**1 somehow. However, an even simpler choice for **R**1 and **T**1 exists. We simply overlap world coordinates with the coordinates of the first camera. This means **R**1 = eye(3), **T**1 = zeros(3) and **R**2 = **R**, **T**2 = **T**. Therefore, all triangulated 3D points are measured from the C1 camera position in the world.

One additional note here is the value of RMSE. Using the frames I provide, we get a value of 3.3. This is not great but also not too bad. Because both checkerboard gird patterns has uncertainties in the coordinates, the uncertainty gets compounded when we perform stereo calibration. In the next step, we will perform triangulation and we will see that the triangulated result is not bad. However, if you need a better calibration error, I suggest you use sharper images at higher resolutions.

The full stereo calibration code is wrapped into a function and provided below.

```python
def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (5,8), corners1, c_ret1)
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T
 
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')
```

**Stereo Triangulation**

We are now ready to triangulate pixel coordinates from two frames into 3D coordinates. As noted in the previous section, by selecting **R**1 = eye(3) and **T**1 = zeros(3), our triangulated points will measured from the position and orientation of camera #1.

Additionally, I had a bug such that using opencv and matplotlib in the same script caused segmentation errors. The developers of opencv are aware of the bug but if you also experience this, I suggest you save the camera matrices, rotation matrix and translation to disk and proceed by starting a new script.

Our first task is to select several points from both cameras to triangulate. For the images provided in the testing folder, the keypoints are hard coded.

```python
uvs1 = [[458, 86], [451, 164], [287, 181],
        [196, 383], [297, 444], [564, 194],
        [562, 375], [596, 520], [329, 620],
        [488, 622], [432, 52], [489, 56]]
 
uvs2 = [[540, 311], [603, 359], [542, 378],
        [525, 507], [485, 542], [691, 352],
        [752, 488], [711, 605], [549, 651],
        [651, 663], [526, 293], [542, 290]]
 
uvs1 = np.array(uvs1)
uvs2 = np.array(uvs2)
 
 
frame1 = cv.imread('testing/_C1.png')
frame2 = cv.imread('testing/_C2.png')
 
plt.imshow(frame1[:,:,[2,1,0]])
plt.scatter(uvs1[:,0], uvs1[:,1])
plt.show()
 
plt.imshow(frame2[:,:,[2,1,0]])
plt.scatter(uvs2[:,0], uvs2[:,1])
plt.show()
```

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/keypoints.png?raw=true">
</p>
<p align="center">
Keypoints for triangulation.
</p>


The next step is to obtain the projection matrices. This is done simply by multiplying the camera matrix by the rotation and translation matrix.

```python
#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix for C2
```

All that is left is to triangulate each point using direct linear transform(DLT). Here I provide my code for DLT without explanation. If you want to know how DLT works, please see here for my post: [link](https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html).

```python
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]
```

We simply pass the keypoints from the two camera views to the DLT and obtain the triangulated point.

```python
p3ds = []
for uv1, uv2 in zip(uvs1, uvs2):
    _p3d = DLT(P1, P2, uv1, uv2)
    p3ds.append(_p3d)
p3ds = np.array(p3ds)
```

We can visualize the 3D triangulation using Matplotlib. If you don’t see the pose, make sure to rotate the axis until the pose appears. If you’re also using cv.imshow(), the plt.show() call will cause a crash. I suggest you comment out the cv.imshow() calls to see the triangulation.

```python
from mpl_toolkits.mplot3d import Axes3D
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-15, 5)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(10, 30)
 
connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
for _c in connections:
    print(p3ds[_c[0]])
    print(p3ds[_c[1]])
    ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
 
plt.show()
```

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/triangulation.png?raw=true">
</p>
<p align="center">
3D coordinates are triangulated using the camera matrices.
</p>


So that is it. The full code is below. You can copy and paste for the full run through. Make sure you also download and put the frames I use in this demo in the same working folder. The frames can be downloaded here: [link](https://drive.google.com/file/d/1o6OVbxi6dT0kDinmUQaBrEZBF3vz_kDp/view?usp=sharing). If you observe a segmentation fault crash, then comment out cv.imshow() or plt.show() calls.

```python 
import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
 
 
def calibrate_camera(images_folder):
    images_names = glob.glob(images_folder)
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    # plt.figure(figsize = (10,10))
    # ax = [plt.subplot(2,2,i+1) for i in range(4)]
    #
    # for a, frame in zip(ax, images):
    #     a.imshow(frame[:,:,[2,1,0]])
    #     a.set_xticklabels([])
    #     a.set_yticklabels([])
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist
 
def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (5,8), corners1, c_ret1)
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
            cv.imshow('img2', frame2)
            cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T
 
def triangulate(mtx1, mtx2, R, T):
 
    uvs1 = [[458, 86], [451, 164], [287, 181],
            [196, 383], [297, 444], [564, 194],
            [562, 375], [596, 520], [329, 620],
            [488, 622], [432, 52], [489, 56]]
 
    uvs2 = [[540, 311], [603, 359], [542, 378],
            [525, 507], [485, 542], [691, 352],
            [752, 488], [711, 605], [549, 651],
            [651, 663], [526, 293], [542, 290]]
 
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
 
 
    frame1 = cv.imread('testing/_C1.png')
    frame2 = cv.imread('testing/_C2.png')
 
    plt.imshow(frame1[:,:,[2,1,0]])
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show() #this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.
 
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(uvs2[:,0], uvs2[:,1])
    plt.show()#this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this
 
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
 
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
 
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
 
        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]
 
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
 
    from mpl_toolkits.mplot3d import Axes3D
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)
 
    connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
    for _c in connections:
        print(p3ds[_c[0]])
        print(p3ds[_c[1]])
        ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
    ax.set_title('This figure can be rotated.')
    #uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
    plt.show()
 
 
mtx1, dist1 = calibrate_camera(images_folder = 'D2/*')
mtx2, dist2 = calibrate_camera(images_folder = 'J2/*')
 
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')
 
#this call might cause segmentation fault error. This is due to calling cv.imshow() and plt.show()
triangulate(mtx1, mtx2, R, T)
```

References:

1. OpenCV camera calibration documentation: [link](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5)
2. For a good theory on photogrammetry, watch the youtube lessons here: [link](https://www.youtube.com/watch?v=_mOG_lpPnpY&list=PLgnQpQtFTOGRsi5vzy9PiQpNWHjq-bKN1&ab_channel=CyrillStachniss)
3. For the workings of DLT: [link](http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf)
