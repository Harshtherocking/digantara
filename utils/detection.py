import cv2 as cv
import torch 
import numpy as np
import os 
import tqdm 

from utils.visualize import viz_blob

SIZE =  64
KERNEL = 3
SIGMA = 0.5


def star_detection (img : np.ndarray) : 
    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = 5
    params.maxThreshold = 255

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 2      
    params.maxArea = 8000    

    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.maxCircularity = 1

    params.filterByConvexity = False

    detector = cv.SimpleBlobDetector_create(params)
    keypoints  = detector.detect(img)
    return keypoints



def streak_detection (img : np.ndarray) : 
    params =  cv.SimpleBlobDetector_Params()

    params.minThreshold = 5
    params.maxThreshold = 255

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 2      
    params.maxArea = 8000    

    params.filterByInertia = True
    params.minInertiaRatio = 0.00001
    params.maxInertiaRatio = 0.5

    params.filterByCircularity = True
    params.minCircularity = 0.001
    params.maxCircularity = 0.7

    params.filterByConvexity = False

    detector  = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return keypoints



    

def blob_detection (img : np.ndarray | torch.Tensor, plot : bool = False) : 
    # if given tensor 
    if isinstance(img, torch.Tensor) :
        img = img.numpy()
        img = img.astype(np.uint8)
        img = img * 255

    assert(isinstance(img, np.ndarray)), "cant find numpy array"

    if len(img.shape) ==  4 and img.shape[1] == 1 :
        img = img[0]


    images = img 
    for img in images  :
        assert(isinstance(img,  np.ndarray)), "cant convert to nd array"
        img = cv.GaussianBlur(img, (KERNEL, KERNEL), SIGMA)
        stars = star_detection(img)
        streaks = streak_detection(img)
        print(f"Found Stars : {len(stars)} Streaks : {len(streaks)}")
    
        if plot : 
            viz_blob(img, stars)
            viz_blob(img, streaks)
            
    return stars, streaks
        








if __name__ == "__main__" : 
    pass