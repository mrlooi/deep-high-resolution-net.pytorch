from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import argparse
# import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# import _init_paths
from lib.config import cfg, update_config
config = cfg
from lib.core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import *

import cv2
import dataset
import models
import numpy as np
import time


def _box2cs(box, image_width, image_height, scale=1.25):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height, scale)

def _xywh2cs(x, y, w, h, image_width, image_height, scale=1.25):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    
    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    size = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        size = size * scale

    return center, size

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
vid_formats = ['.mov', '.avi', '.mp4']

def is_img_file(file):
    return os.path.splitext(file)[-1].lower() in img_formats

def is_video_file(file):
    return os.path.splitext(file)[-1].lower() in vid_formats

def input_to_tensor(input, device='cpu'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    x = transform(input)
    if x.ndimension() == 3:
        x = x.unsqueeze(0)
    return x.to(device)

def crop_and_scale_image(image, bbox, out_size, scale=1.1):
    """
    image: (H,W,3)
    bbox: [x1,y1,w,h]
    scale: multiplicative padding around bbox
    out_size: (out_H, out_W)
    """
    # H, W = image.shape[:2]
    # out_H, out_W = out_size
    # x1, y1, w, h = bbox
    # half_w, half_h = w * 0.5, h * 0.5
    # cx = x1 + half_w
    # cy = y1 + half_h
    # c = np.array([cx, cy])
    # # cx, cy = int(round(cx)), int(round(cy))
    # # w, h = int(w), int(h)
    # xx1 = int(round(cx - half_w * scale))
    # yy1 = int(round(cy - half_h * scale))
    # xx2 = int(round(cx + half_w * scale))
    # yy2 = int(round(cy + half_h * scale))
    # w = xx2 - xx1 + 1
    # h = yy2 - yy1 + 1
    # cropped = np.zeros((h, w, 3), dtype=image.dtype)
    # extra_x1 = -min(0, xx1)
    # extra_y1 = -min(0, yy1)
    # extra_x2 = max(W, xx2)
    # extra_y2 = max(H, yy2)
    # cropped[extra_y1:extra_y2, extra_x1:extra_x2] = image[yy1-extra_y1:yy2-extra_y2+1,xx1-extra_x1:xx2-extra_x2+1]

    # out = cv2.resize(cropped, (out_W, out_H))

    c, s = _box2cs(bbox, image.shape[0], image.shape[1], scale)
    r = 0
    trans = get_affine_transform(c, s, r, out_size)
    out = cv2.warpAffine(
        ori_img, trans, (int(out_size[0]), int(out_size[1])),
        flags=cv2.INTER_LINEAR)
    return out, c, s

def get_frames_from_video(video_file):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_file)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    frames = {}

    # Read until video is completed
    ix = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            frames[ix] = frame

        # Break the loop
        else: 
            break

        ix += 1
     
    # When everything done, release the video capture object
    cap.release()

    fk = frames.keys()
    total = max(fk) + 1
    print("Processed video: %d frames"%(total))

    data = np.zeros((total, frame_height, frame_width, 3), dtype=np.uint8)
    for k in fk:
        data[k] = frames[k]

    return data


def inference(model, input, flip_pairs, flip_test=True, shift_heatmap=False):
    if not flip_test:
        output = model(input)
    else:
        input_flipped = input.flip(-1) # flip last channel (width)
        inputs = torch.stack((input, input_flipped)).squeeze(1)
        output = model(inputs)
        output = output.cpu().numpy()

        output_flipped = flip_back(output[1:], flip_pairs)
        output = output[0:1]

        # feature is not aligned, shift flipped heatmap for higher accuracy
        if shift_heatmap:
            output_flipped[:, :, :, 1:] = \
                output_flipped.copy()[:, :, :, 0:-1]

        output = (output + output_flipped) * 0.5 # average heuristic. could use max
    return output


if __name__ == '__main__':
    class Args():
        def __init__(self):
            # self.cfg = "experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml"
            # self.model_file = "models/pytorch/pose_coco/pose_resnet_50_384x288.pth.tar"
            self.cfg = "experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml"
            self.model_file = "models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth"

            # self.source = '/home/vincent/hd/datasets/MPII/images/000001163.jpg'
            # self.source_bbox = None
            self.source = '/home/vincent/Documents/py/ml/yolov3/data/videos/ice_skate_girl.mp4'
            self.source_bbox = '/home/vincent/Documents/py/ml/yolov3/ice_skate_girl.npy'
            
            self.flip_test = True
            self.shift_heatmap = True

            self.modelDir = None
            self.dataDir = None
            self.logDir = "./log"

            self.device = "cuda"

            self.opts = []
            # self.gpus = [0] # use this for dataparallel

    args = Args() # parse_args()
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = False
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    joint_names = [
        "Nose", # 0
        "LEye", "REye", "LEar", "REar",  # 1, 2, 3, 4
        "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", # 5,6,7,8,9,10
        "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle" # 11,12,13,14,15,16
    ]
    parents = [-1,0,0,1,2,0,0,5,6,7,8,0,0,11,12,13,14]
    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                  [9, 10], [11, 12], [13, 14], [15, 16]]

    # create logger
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    # init model
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    logger.info('=> loading model from {}'.format(args.model_file))
    model.load_state_dict(torch.load(args.model_file))
    model = model.to(args.device)
    # gpus = args.gpus
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()


    ## Load data
    source = args.source
    if is_video_file(source):
        images = get_frames_from_video(source)
        data = np.load(args.source_bbox)
        bboxes = data[:, :5]
        assert len(images) == len(bboxes)
    else:
        ori_img = cv2.imread(source)
        if ori_img is None:
            raise ValueError('Fail to read {}'.format(source))
        images = [ori_img]
        box = [470, 85, 605, 464]  # x, y, x2, y2
        bboxes = np.array([box])

    N = len(images)

    # switch to evaluate mode
    model.eval()

    keypoints_2d = np.zeros((N, 17, 3))

    with torch.no_grad():
        scale = 1.25
        for ix in range(N):
            ori_img = images[ix]
            bbox = bboxes[ix, :4]
            box = bbox.copy()
            box[2:] -= box[:2]

            cropped_img, c, s = crop_and_scale_image(ori_img, box, config.MODEL.IMAGE_SIZE, scale)

            cv2.imshow('cropped_img', cropped_img)

            # inference
            ts = time.time()
            img_tensor = input_to_tensor(cropped_img, device=args.device)
            output = inference(model, img_tensor, flip_pairs, args.flip_test, args.shift_heatmap)
            print("Time taken: %.3fs"%(time.time() - ts))

            # compute coordinate
            preds, maxvals = get_final_preds(
                config, output.copy(), np.asarray([c]), np.asarray([s]))
            maxvals = maxvals[..., 0]

            # visualize
            preds = np.round(preds).astype(np.int)
            image = ori_img.copy()

            score = maxvals[0]
            pred = preds[0]

            keypoints_2d[ix,:,:2] = pred
            keypoints_2d[ix,:,2] = score

            score_thresh = 0.1

            bbox_int = np.round(bbox).astype(int)
            cv2.rectangle(image, tuple(bbox_int[:2]), tuple(bbox_int[2:]), (0,255,0))
            for j, mat in enumerate(pred):
                if score[j] < score_thresh:
                    print("%s: %.3f"%(joint_names[j], score[j]))
                    continue
                px = tuple(mat)
                cv2.circle(image, px, 2, (255, 0, 0), 2)
                cv2.putText(image, "%d"%(j), px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

                if parents[j] > -1:
                    cv2.line(image, px, tuple(pred[parents[j]]), (0,0,255))

            # vis result
            cv2.imshow('res', image)
            cv2.waitKey(0)
            print(ix)

    out_file = os.path.splitext(source)[0].split("/")[-1] + ".npz"
    out_file = "keypoints_2dX_" + out_file
    # custom gay style for VideoPose3D
    bboxes_g = []
    keypoints_g = []
    for ix in range(N):
        bboxes_g.append([[], [bboxes[ix]]])
        kkk = np.zeros((4, 17))
        kkk[:2] = keypoints_2d[ix,:,:2].T.copy()
        keypoints_g.append([[], [kkk]])
    np.savez_compressed(out_file, boxes=bboxes_g, keypoints=keypoints_g, metadata={'w': ori_img.shape[1], 'h': ori_img.shape[0]})
