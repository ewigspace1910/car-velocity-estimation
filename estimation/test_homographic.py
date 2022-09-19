from ast import parse
import cv2
import time
import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import yaml
import sys
import os
import math

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("x: {} y: {}".format(x, y))
        

def LineGen(X1=0, Y1=0, X2=1, Y2=1, ab=None):
    '''
    Args:
    ab: tuple
    X1,Y1,X2,Y2 : int - scalars

    create a line by 2 way:
    - use 2 point(X1,Y1) and (X2, Y2) 
    - use (a,b) in y=ax+b
    '''
    a = b = 1
    if ab is None:
        a = (Y1 - Y2) / (X1 - X2)
        b = Y1 - a * X1
    else: a, b = ab
    def _line(x=None, y=None, only_x=False, only_ab=False):

        if only_x: return int(a * x + b)
        if only_ab: return a, b
        return y - (a * x + b)
    return _line





def main(cfg):
    #initial-----
    lseg_points_up = cfg['lseg-points-up']
    lseg_points_dw = cfg['lseg-points-dw']  # [282, 592] #(x,y) is a point in parallel line


    _lsegpoints_up = [tuple(x) for x in lseg_points_up]
    _lsegpoints_dw = [tuple(x) for x in lseg_points_dw]

    pline1 = LineGen(X1=_lsegpoints_up[0][0], Y1=_lsegpoints_up[0][1], 
                        X2=_lsegpoints_dw[0][0], Y2=_lsegpoints_dw[0][1])
    pline2 = LineGen(X1=_lsegpoints_up[1][0], Y1=_lsegpoints_up[1][1], 
                        X2=_lsegpoints_dw[1][0], Y2=_lsegpoints_dw[1][1])
    a1, b1 = pline1(only_ab=True)
    a2, b2 = pline2(only_ab=True)
    xv = (b2-b1) / (a1-a2)
    yv = a1 * xv + b1
    _VPoitn = tuple((int(xv), int(yv)))
    print(f"Vanishing point -> ({xv}, {yv})")

    A = np.array(cfg['bounding_box']['A'] + [1])
    B = np.array(cfg['bounding_box']['B'] + [1])
    D = np.array(cfg['bounding_box']['D'] + [1])
    C = np.array(cfg['bounding_box']['C'] + [1])
    path = "../_data/img/homography.png" 
    if not os.path.exists(path) : 
        print(f"{path} khong ton tai")
        print(os.getcwd())
        exit()
    vs = cv2.imread(path)
    pts_src = np.array([list(D), list(C), list(A), list(B), [_VPoitn[0], _VPoitn[1], 1]])
    pts_dst = np.array([list(D), [C[0], D[1], 1], [D[0], A[1], 1], [C[0], A[1], 1] , [1, 0, 0]])

    #_H, status = cv2.findHomography(pts_src, pts_dst)
    _H, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
    im_out = cv2.warpPerspective(vs, _H, (vs.shape[1],vs.shape[0]))
    #--------------draw----------------
    #for  x in _lsegpoints_dw :cv2.line(vs, _VPoitn, x, (59, 255, 235), 2)
    A = np.matmul(_H, A.T)
    B = np.matmul(_H, B.T)
    C = np.matmul(_H, C.T)
    D = np.matmul(_H, D.T)
    A = tuple((A / A[-1])[:2].astype(int))
    print(A)
    B = tuple((B / B[-1])[:2].astype(int))
    C = tuple((C / C[-1])[:2].astype(int))
    D = tuple((D / D[-1])[:2].astype(int))
    cv2.line(im_out, A, B, (0, 0, 255), 2) #image, start_point, end_point, color, thickness
    cv2.line(im_out, D, C, (0, 0, 255), 2)
    #for  x, y in zip(_lsegpoints_up, _lsegpoints_dw) :cv2.line(frame, x, y, (59, 255, 235), 2)
    #cv2.line(frame, _VPoitn, _ipoint, (59, 59, 235), 2)


    #show
    
    #cv2.imshow("hahah", vs)
    cv2.imshow("homography", im_out)

    #cv2.imwrite(vs, "test.png")
    cv2.waitKey(10000)
    time.sleep(100)

    # if key == ord("l"): #esc
    #     print("====interrupted by quit key====")
    #     cv2.destroyAllWindows() 
    #     exit()

if __name__ == '__main__':
    with open("../_data/cfg.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        if not os.path.exists(cfg['source']): assert False, "Not found video."
    main(cfg)


