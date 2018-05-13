import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
dist_pickle = pickle.load(open( "calibrated_data.p", "rb" ))

#img = cv2.imread('test_image2.png')
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def cal_undistort(img, mtx, dist):
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist


def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
