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
sys.path.insert(0, './yolov7')
sys.path.append('.')

from yolov7.models.experimental import attempt_load
# from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, TracedModel

from tracker.mc_bot_sort import BoTSORT
# from tracker.tracking_utils.timer import Timer

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

def dist2point(X, Y):
    '''Args: 
    - X: (int,int) the first point
    - Y: (int,int) the second point
    Returns: a Ecul distance between X and Y
    '''
    if not isinstance(X, tuple) or not isinstance(Y, tuple): 
        print(X, Y)
        assert False, "X, Y must tuple"
    return math.sqrt((X[0]-Y[0]) ** 2 + (X[1] - Y[1]) ** 2)

def main(opt, cfg):
    path = cfg['source']
    vs = cv2.VideoCapture(path)
    vs.set(cv2.CAP_PROP_FPS, cfg['fps'])

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    #Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_path = Path(save_dir / path.split("/")[-1])
    if os.path.exists(save_path): os.remove(save_path)
    #dectect
    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
    if opt.trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()  # to FP16
    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
   
    #tracker
    tracker = BoTSORT(opt, frame_rate=10.0) #fr depend on your vid
    #======================================================================
    #setup video
    windown_name = "TEST"
    cv2.namedWindow(windown_name)
    cv2.setMouseCallback("TEST", click_and_crop)
    vid_path, vid_writer = None, None
    _fps = int(vs.get(cv2.CAP_PROP_FPS))
    _w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #system parameter
    _SE = float(cfg['se'])

    #setup bounding box
    _A = tuple(cfg['bounding_box']['A'])
    _B = tuple(cfg['bounding_box']['B'])
    _C = tuple(cfg['bounding_box']['C'])
    _D = tuple(cfg['bounding_box']['D'])
    _RealDistance = cfg['real_distances']['A-D'];print("real distance: ", _RealDistance, " m")
    
    #point(x,y) is down the line if y-upline(x) > 0, conversely up the line if y-upline(x) < 0. (do truc y bi dao)
    upline   = LineGen(_A[0], _A[1], _B[0], _B[1])
    downline = LineGen(_D[0], _D[1], _C[0], _C[1])
    righline = LineGen(_B[0], _B[1], _C[0], _C[1])
    downEline = LineGen(_D[0], _D[1] * 1.2 , _C[0], _C[1] * 1.2) #endline detect

    if opt.mode == 2:
        _RealDistance = cfg['real_distances']['A-D']
        _lsegpoints_up = [tuple(x) for x in cfg['lseg-points-up']]
        _lsegpoints_dw = [tuple(x) for x in cfg['lseg-points-dw']]
        
        pline1 = LineGen(X1=_lsegpoints_up[0][0], Y1=_lsegpoints_up[0][1], 
                         X2=_lsegpoints_dw[0][0], Y2=_lsegpoints_dw[0][1])
        pline2 = LineGen(X1=_lsegpoints_up[1][0], Y1=_lsegpoints_up[1][1], 
                         X2=_lsegpoints_dw[1][0], Y2=_lsegpoints_dw[1][1])
        a1, b1 = pline1(only_ab=True)
        a2, b2 = pline2(only_ab=True)
        xv = (b2-b1) / (a1-a2)
        yv = a1 * xv + b1
        _VPoitn = tuple((int(xv), int(yv)))
        del a1, b1, xv, yv, pline1, pline2
        _interval = cfg['interval']
        
        pts_src = np.array([list(_D) + [1], list(_C) + [1], list(_A) + [1], list(_B) + [1] , [_VPoitn[0], _VPoitn[1], 1]])
        pts_dst = np.array([list(_D) + [1], [_C[0], _D[1], 1], [_D[0], _A[1], 1], [_C[0], _A[1], 1] , [1, 0, 0]])
        _H, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)

        _homoA = np.matmul(_H, np.array([_A[0], _A[1], 1]).T);_homoD = np.matmul(_H, np.array([_D[0], _D[1], 1]).T)
        _homoA, _homoD = _homoA / _homoA[-1], _homoD / _homoD[-1]
        
        

    
    #runvideo
    speed_track= {} #depend on type config
                    #       SOLUTION1 :
                    #id : [x, y , [dD/frame_t, dD/frame_t-1, dD/frame_t-2], sum_distance, [start, now], Flag] 
                    #       with (x,y is  top left point coordinates)
                    #
                    #       SOLUTION2:
                    #id : [] 


    _fcount = 0
    while True:
        _, frame = vs.read()
        if frame is None: break
        _fcount += 1
        if _fcount > 1000: break
    
        rsframe = cv2.resize(frame,(1920, 1088), interpolation=cv2.INTER_CUBIC) #1080, 1920 => 1088, 1920
        img = torch.from_numpy(rsframe).to(device) 
        img = img.half() if half else img.float() # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0).permute(0, 3, 1, 2)

        # Inference
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # Run tracker
            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], frame.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            #----------------------------------#
            # discard objs are out boundingbox #
            ignores = [i for i in range(detections.shape[0]) \
                    if upline((detections[i][0] + detections[i][2]) /2, (detections[i][3] + detections[i][1]) /2) < 0 \
                    or downEline(detections[i][0], detections[i][1]) > 0
                    #righline(detections[i][0], detections[i][1]) < 0 \
                    ]
            detections = np.delete(detections, ignores, axis=0)
            #----------------------------------#
            #          update tracker          #
            online_targets = tracker.update(detections, frame)
            #----------------------------------#
            #       update speed tracker       #
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                # tcls = t.cls                
                if tlwh[2] * tlwh[3] < opt.min_box_area: continue
                v = "---"
                #Option1: Scanline - only trigger estimation when car cross all boundingbox 
                if opt.mode == 1: 
                    if tid in speed_track: 
                        pp = speed_track[tid] 
                        cp = (int((tlbr[0] + tlbr[2]) / 2), int((tlbr[1] + tlbr[3])/2)) 
                        dD = max(round(math.sqrt((pp[0] - cp[0]) ** 2 + (pp[1] - cp[1]) ** 2), 2) - _SE , 0)
        
                        ##estimate speed
                        if pp[2][1] * pp[2][0] * dD  == 0: 
                            v = f"Stop!!!"
                            if  dD != 0 and pp[2][0] != 0: pp[4][1] = _fcount
                            else: pp[4][0] = pp[4][1] = _fcount
                            pp[3] = (1 - (upline(cp[0], cp[1], True) - cp[1]) / (upline(cp[0], cp[1], True) - downline(cp[0], cp[1], True)))  * _RealDistance
                        else:
                            if downline(cp[0], cp[1]) > 0 and pp[4][0] > 0:
                                if pp[4][1] != pp[4][0]:
                                    v = f"{pp[3] * _fps / (pp[4][1] - pp[4][0]) * 3.6} kmh"  #khoang cach di duoc
                                pp[5] = False
                        #update stage  
                        pp[0], pp[1] = cp
                        pp[2].pop();pp[2] = [dD] + pp[2]    #state
                        if pp[5]: pp[4][1] = _fcount        #number of frames 
                        speed_track[tid] = pp
                    else :
                        speed_track[tid] = [(tlbr[0] + tlbr[2]) / 2, 
                                            (tlbr[1] + tlbr[3])/2, 
                                            [-1, -1, -1], 
                                            _RealDistance, 
                                            [_fcount, _fcount], True]
                
                #Option2: A Semi-Automatic 2D Solution
                elif opt.mode == 2:
                    #Scale Recovery
                    sx = 1
                    sy = 1

                    #speed optimation
                    if tid in speed_track : 
                        pp = speed_track[tid] 
                        if pp[-1]:
                            cp = (int((tlbr[0] + tlbr[2]) / 2), int((tlbr[1] + tlbr[3])/2)) 
                            
                            _old = np.matmul(_H, np.array([pp[0], pp[1], 1]).T)
                            _new = np.matmul(_H, np.array([cp[0], cp[1], 1]).T)
                            _old = _old / _old[-1]
                            _new = _new / _new[-1]

                            homoD = math.sqrt((sx * (_old[0] - _new[0])) ** 2 + (sy * (_new[1] - _old[1])) ** 2)
                            #smood distant
                            homoD *= _RealDistance / np.sqrt((_homoA[0] - _homoD[0]) ** 2 + (_homoA[1] - _homoD[1]) ** 2)
                            pp[2] += homoD
                          
                            ##estimate speed     
                            if (_fcount - pp[4][0]) % _interval == 0 :     
                                if homoD * _fps * 3.6 < 3: #noise  
                                    v = f"Stop!!!"
                                    pp[2] = pp[3] = 0
                                    pp[4][0] = _fcount                       
                                else:
                                    pp[3] = pp[2] * _fps / (_fcount - pp[4][0]) * 3.6  #khoang cach di duoc
                                    v = f'{pp[3]} kmh'
                            else:
                                #print(pp[2])
                                v = f'{pp[3]} kmh' if pp[3] > 0 else f'---'
                            
                            if downline(cp[0], cp[1]) > 0: pp[-1] = False
                            #update stage  
                            pp[0], pp[1] = cp
                            speed_track[tid] = pp
                        else: v = f'{pp[3]} kmh' if pp[3] > 0 else f'---'
                    else :
                        speed_track[tid] = [(tlbr[0] + tlbr[2]) / 2, # X center
                                            (tlbr[1] + tlbr[3]) / 2, # Y center
                                            0,                       #denta D
                                            0,                       #V
                                            [_fcount, _fcount + 1],  #frame
                                            True]                    #flag


                #NonOption
                elif opt.mode == -1: pass
                else: assert "chiu deyyy!"
  
                # Add bbox to image
                #print(f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
                label = f'{tid}, {v}'
                plot_one_box(tlbr, frame, label=label, color=colors[int(tid) % len(colors)], line_thickness=1)

        #########################
        #upper line + lowerline
        cv2.line(frame, _A, _B, (0, 0, 255), 2) #image, start_point, end_point, color, thickness
        cv2.line(frame, _D, _C, (0, 0, 255), 2)
        #============control presentation=============
        print("{}th-frame".format(_fcount))
        if not opt.save:
            cv2.imshow("TEST", frame)
            key = cv2.waitKey(1) & 0xFF #100 millisecond
        
            if key == ord(cfg['quit']): #esc
                print(key)
                print("====interrupted by quit key====")
                break
        else:
            if vid_writer is None:  # new video
                print(save_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(str(save_path), apiPreference=0,fourcc=fourcc, fps=_fps, frameSize=(_w, _h))
            vid_writer.write(frame)


    #end
    if isinstance(vid_writer, cv2.VideoWriter): vid_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #detect
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    #velocity-estimation
    parser.add_argument("--cfg", type=str, help="path to config file contain coordinates", default="./cfg.yaml")
    parser.add_argument("--fps", type=int, help="fps of video", default=30)
    parser.add_argument("--mode", type=int, help="type of solutions", default=1, choices=[-1,1,2,3])


    #########################################
    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    # print("opt: --> ", opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    if not os.path.exists(opt.cfg): assert False, "Not found configure file."

    else:
        with open(opt.cfg, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            if not os.path.exists(cfg['source']): assert False, "Not found video."
            # print("config box --> ",cfg)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                main(opt, cfg)
                strip_optimizer(opt.weights)
        else:
            main(opt, cfg)


