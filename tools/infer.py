from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import argparse
# import pprint

import math

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
from lib.core.inference import transform_preds #, get_final_preds
# from utils.utils import create_logger
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
    c, s = _box2cs(bbox, image.shape[0], image.shape[1], scale)
    r = 0
    trans = get_affine_transform(c, s, r, out_size)
    out = cv2.warpAffine(
        image, trans, (int(out_size[0]), int(out_size[1])),
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


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale, post_process=False):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # for pair in flip_pairs:
    #     l_id, r_id = pair
    #     pair_name = joint_names[l_id].replace("L","")
    #     hm_img = np.zeros((heatmap_height, heatmap_width*2), dtype=np.uint8)
    #     for i in range(2):
    #         hm0 = batch_heatmaps[0,pair[i]].copy()
    #         hm0 = np.clip(hm0, 0, 1)
    #         hm_img[:,i*heatmap_width: (i+1)*heatmap_width] = hm0 * 255
    #     hm_img = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
    #     hm_img[:,heatmap_width] = 255 # border
    #     cv2.imshow("%s"%(pair_name), hm_img)

    # post-processing
    if post_process:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def save_data(out_file):
    # custom gay style for VideoPose3D
    bboxes_g = []
    keypoints_g = []
    for ix in range(N):
        bboxes_g.append([[], [bboxes[ix]]])
        kkk = np.zeros((4, 17))
        kkk[:2] = keypoints_2d[ix,:,:2].T.copy()
        keypoints_g.append([[], [kkk]])
    np.savez_compressed(out_file, boxes=bboxes_g, keypoints=keypoints_g, metadata={'w': ori_img.shape[1], 'h': ori_img.shape[0]})


class KeypointPredictor:
    def __init__(self, args):
        update_config(config, args)

        cudnn.benchmark = False
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        self.args = args
        self.model = self.load_model()
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                  [9, 10], [11, 12], [13, 14], [15, 16]]

    def load_model(self):
        # init model
        model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, is_train=False
        )
        print('=> loading model from {}'.format(self.args.model_file))
        model.load_state_dict(torch.load(self.args.model_file))
        model = model.to(self.args.device)
        return model.eval()

    def predict(self, image, bbox=None):
        scale = 1.25
        args = self.args

        H, W = image.shape[:2]
        if bbox is None:
            # x, y, w, h
            bbox = np.array([0,0,W,H])
        # else:
        #     assert 0 < bbox[0] < (bbox[0] + bbox[2]) <= W and 0 < bbox[1] < (bbox[1] + bbox[3]) <= H

        cropped_img, c, s = crop_and_scale_image(image, bbox, config.MODEL.IMAGE_SIZE, scale)

        # inference
        ts = time.time()
        with torch.no_grad():
            img_tensor = input_to_tensor(cropped_img, device=args.device)
            output = inference(self.model, img_tensor, self.flip_pairs, args.flip_test, args.shift_heatmap)
        print("Time taken: %.3fs"%(time.time() - ts))

        # compute coordinate
        preds, maxvals = get_final_preds(
            config, output.copy(), np.asarray([c]), np.asarray([s]), config.TEST.POST_PROCESS)
        maxvals = maxvals[..., 0]

        return preds, maxvals, cropped_img


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.backbone = "hrnet" # "resnet"
            # self.cfg = "experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml"
            # self.model_file = "models/pytorch/pose_coco/pose_resnet_50_384x288.pth.tar"
            self.cfg = "experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml"
            self.model_file = "models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth"
            self.flip_test = True
            self.shift_heatmap = False
            self.device = "cuda"
            self.opts = []

            self.modelDir = None
            self.dataDir = None
            self.logDir = "./log"

            # self.source = '/home/vincent/hd/datasets/MPII/images/000001163.jpg'
            # self.source_bbox = None
            self.source = "manako"
            self.source_file = 'data/%s.mp4'%(self.source)
            self.source_bbox = 'data/%s.npy'%(self.source)

            # self.gpus = [0] # use this for dataparallel

    args = Args() # parse_args()

    joint_names = [
        "Nose", # 0
        "LEye", "REye", "LEar", "REar",  # 1, 2, 3, 4
        "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", # 5,6,7,8,9,10
        "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle" # 11,12,13,14,15,16
    ]
    parents = [-1,0,0,1,2,0,0,5,6,7,8,0,0,11,12,13,14]

    # # create logger
    # logger, final_output_dir, tb_log_dir = create_logger(
    #     config, args.cfg, 'valid')

    # init model
    predictor = KeypointPredictor(args)

    ## Load data
    source = args.source_file
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

    keypoints_2d = np.zeros((N, 17, 3))
    start_frame = 0
    end_frame = N

    with torch.no_grad():
        scale = 1.25
        for ix in range(start_frame, end_frame):
            ori_img = images[ix]
            bbox = bboxes[ix, :4]
            box = bbox.copy()
            box[2:] -= box[:2]

            # compute coordinate
            preds, maxvals, cropped_img = predictor.predict(ori_img, box)

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
            cv2.imshow('cropped_img', cropped_img)
            cv2.imshow('res', image)
            cv2.waitKey(1)
            print(ix)

    # out_file = os.path.splitext(source)[0].split("/")[-1] + ".npz"
    # out_file = "keypoints_2dX_" + out_file
    # save_data(out_file)

    def filter_keypoints(keypoints_2d, window_size=10):
        """
        keypoints_2d.shape == (N, 17, 3) # x, y, score

        Use moving window, neighboring frames as priors to refine keypoints with pairs
        # the hips and legs are mainly the problem; sometimes ankles cross wrongly
        # sometimes arms as well
        # usually, face is correct

        Heuristic:
        Use trajectory of keypoints in moving window, and compare with L/R consistency
        If there is a sudden switch in trajectory, compare with L/R to see if they're flipped
        """
        k2d = keypoints_2d.copy()
        N = k2d.shape[0]
        pad = window_size // 2
        window_is_odd = int(window_size % 2 == 1)

        kps = keypoints_2d[:,:,:2]
        traj = kps[1:] - kps[:-1]

        for idx in range(len(traj)):
            start = max(0, idx - pad)
            end = min(N, idx + pad + window_is_odd)
            traj_window = traj[start:end]

            # to solve for legs, start with ankles crossing

            # to solve for arms, start with wrists crossing

            # detect sudden, temporary switch for hips (we assume this is a wrong flip)
            # and check consistency with shoulders.
            # shoulders and hips cannot cross, except for certain side body views
            # maybe voting scheme?

            # if hips are flipped, flip all leg keypoints
            # if shoulders are flipped, flip all arm keypoints


            # update any flips for traj and k2d
            # traj[idx]

        return k2d

    keypoints_2d = filter_keypoints(keypoints_2d)
    out_file = "data/%s_keypoints_hrnet.npy"%(args.source)
    np.save(out_file, keypoints_2d)
    print("Saved to %s"%(out_file))

    def visualize():
        for ix in range(start_frame, end_frame):
            ori_img = images[ix]
            img = ori_img.copy()
            canvas = np.zeros_like(ori_img)
            kps = keypoints_2d[ix, :, :2].astype(int)

            for j, kp in enumerate(kps):
                px = tuple(kp)
                cv2.circle(img, px, 2, (255, 0, 0), 2)
                cv2.putText(img, "%d"%(j), px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

                cv2.circle(canvas, px, 2, (255, 0, 0), 2)
                cv2.putText(canvas, "%d"%(j), px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

                parent = parents[j]
                # skip nose to hip
                if parent > -1:
                    kp_parent = tuple(kps[parent])
                    if j in [11,12]: # rewire line from hip to shoulder
                        kp_parent = tuple(kps[j-6])
                    cv2.line(img, px, kp_parent, (0,0,255))
                    cv2.line(canvas, px, kp_parent, (0,0,255))

            # view shoulder to hip lines
            cv2.line(img, tuple(kps[5]), tuple(kps[11]), (0,0,255))
            cv2.line(img, tuple(kps[6]), tuple(kps[12]), (0,0,255))
            cv2.line(canvas, tuple(kps[5]), tuple(kps[11]), (0,0,255))
            cv2.line(canvas, tuple(kps[6]), tuple(kps[12]), (0,0,255))

            cv2.imshow("kp img", img)
            cv2.imshow("kp", canvas)
            cv2.waitKey(0)

    visualize()
