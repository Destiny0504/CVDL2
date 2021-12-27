import numpy as np
from cv2 import cv2
import os
from matplotlib import pyplot as plt

def corner_detection(image_path, board_row=11, board_col=8):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_col*board_row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_row, 0:board_col].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.

    # for image_path in images_path:
    #     print(image_path)
    img = cv2.imread(image_path)
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(grayimage, (board_row, board_col), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # objpoints.append(objp)

        corners2 = cv2.cornerSubPix(grayimage, corners, (11, 11), (-1, -1), criteria)
        # imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (board_row, board_col), corners2, ret)

    return img, grayimage, objp, corners2

def all_corner_detection():
    #The code is from opencv official tutorial
    #Each row has 11 corner
    board_row = 11
    #Each column has 8 corner
    board_col = 8

    images_path = []
    for i in range(1,16):
        images_path.append(os.path.join('Q2_Image', str(i)+'.bmp'))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_col * board_row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_row, 0:board_col].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    all_image = []
    for image_path in images_path:
        # print(image_path)
        img = cv2.imread(image_path)
        grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(grayimage, (board_row, board_col), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(grayimage, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

        # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (board_row, board_col), corners2, ret)
            all_image.append(img)

    return all_image, objpoints, imgpoints

def find_intrinsics(select_image = 1):
    image_path = os.path.join('Q2_Image', str(select_image)+'.bmp')
    img = cv2.imread(image_path)
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_img, objpoints, imgpoints= all_corner_detection()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        grayimage.shape[::-1],
        None,
        None
    )
    print(mtx)
    return mtx


def find_distortion(select_image = 1):
    image_path = os.path.join('Q2_Image', str(select_image)+'.bmp')
    img = cv2.imread(image_path)
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_img, objpoints, imgpoints= all_corner_detection()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        grayimage.shape[::-1],
        None,
        None
    )
    print(dist)
    return dist

def find_extrinsics(index):
    image_path = os.path.join('Q2_Image', str(index)+'.bmp')
    img = cv2.imread(image_path)
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_img, objpoints, imgpoints= all_corner_detection()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        grayimage.shape[::-1],
        None,
        None
    )
    extrinsic_matrices = []
    for i in range(15):
        R, _ = cv2.Rodrigues(rvecs[i])
        nR = np.asarray(R)
        nT = np.asarray(tvecs[i])
        extrinsic_matrix = np.concatenate((nR, nT), axis=1)
        extrinsic_matrices.append(extrinsic_matrix)
    print(extrinsic_matrices[index-1])

def undistortion():
    #load the image
    # image_path = os.path.join('Q2_Image', str(1)+'.bmp')
    # img = cv2.imread(image_path)

    all_image = []

    all_undistorted_image = []

    #find the distortion matrix
    dist = find_distortion(1)
    #find the intrinsic matrix
    mtx = find_intrinsics(1)
    for i in range(1,16):
        img = cv2.imread(os.path.join('Q2_Image', str(i)+'.bmp'))


        h,  w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        undistorted_image = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        img = cv2.resize(img, (img.shape[0]//3, img.shape[1]//3))
        undistorted_image = cv2.resize(undistorted_image, (undistorted_image.shape[0]//3, undistorted_image.shape[1]//3))

        all_image.append(img)
        all_undistorted_image.append(undistorted_image)

    return all_image, all_undistorted_image

def Q21():
    all_image, _, _ = all_corner_detection()
    for i in all_image:
        i = cv2.resize(i, (i.shape[0]//3, i.shape[1]//3))
        cv2.imshow('image', i)
        cv2.waitKey(500)
def Q22():
    find_intrinsics()
def Q23(index):
    find_extrinsics(index)
def Q24():
    find_distortion()
def Q25():
    all_image, all_undistorted = undistortion()
    for i in range(len(all_image)):
        img = cv2.resize(all_image[i], (all_image[i].shape[0], all_image[i].shape[1]))
        undistorted_img = cv2.resize(all_undistorted[i], (all_undistorted[i].shape[0], all_undistorted[i].shape[1]))
        cv2.imshow('image', img)
        cv2.imshow('undistorted_image', undistorted_img)
        cv2.waitKey(500)
# if __name__ == "__main__":
#     Q24()

if __name__ =='__main__':
    all_corner_detection()