import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
dist_pickle = pickle.load(open( "calibrated_data.p", "rb" ))

def cal_undistort(img, mtx=dist_pickle['mtx'], dist=dist_pickle['dist']):
    return cv2.undistort(img, mtx, dist, None, mtx)


def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
