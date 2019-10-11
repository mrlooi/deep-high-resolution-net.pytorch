from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test

    # if args.gpus:
    #     config.GPUS = args.gpus
    # if args.workers:
    #     config.WORKERS = args.workers
    # if args.use_detect_bbox:
    #     config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    # if args.post_process:
    #     config.TEST.POST_PROCESS = args.post_process
    # if args.shift_heatmap:
    #     config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    # if args.coco_bbox_file:
    #     config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    
    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.cfg = "experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml"
            self.model_file = "models/pytorch/pose_coco/pose_resnet_50_384x288.pth.tar"
            self.image_file = '/home/vincent/hd/datasets/MPII/images/000001163.jpg'
            self.flip_test = True

            self.modelDir = None
            self.dataDir = None
            self.logDir = "./log"

            self.opts = []
            # self.gpus = 1

    args = Args() # parse_args()
    reset_config(config, args)
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    logger.info('=> loading model from {}'.format(args.model_file))
    model.load_state_dict(torch.load(args.model_file))
    # else:
    #     model_state_file = os.path.join(final_output_dir,
    #                                     'final_state.pth.tar')
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model.load_state_dict(torch.load(model_state_file))

    gpus = [0]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    ## Load an image
    image_file = args.image_file
    ori_img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if ori_img is None:
        logger.error('=> fail to read {}'.format(image_file))
        raise ValueError('Fail to read {}'.format(image_file))

    # object detection box
    # box = [450, 160, 350, 560] # x, y, w, h
    box = [470, 85, 605, 464] # x, y, x2, y2
    box = np.array(box); box[2:] -= box[:2]
    c, s = _box2cs(box, ori_img.shape[0], ori_img.shape[1])
    r = 0

    trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
    input = cv2.warpAffine(
        ori_img,
        trans,
        (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    # vis transformed image
    cv2.imshow('image', input)
    cv2.waitKey(100)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    input = transform(input).unsqueeze(0)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = model(input)
        # for i in range(10):
        #     ts = time.time()
        #     output = model(input)
        #     print("Time taken: %.3fs"%(time.time() - ts))

        # compute coordinate
        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))
        maxval = maxvals[0].squeeze()
        # plot
        image = ori_img.copy()
        for j, mat in enumerate(preds[0]):
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            cv2.putText(image, "%d"%(maxval[j] * 100), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

        # vis result
        cv2.imshow('res', image)
        cv2.waitKey(0)