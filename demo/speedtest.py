#!/usr/bin/env python

# maskrcnn_benchmark/demo/speedtest.py

import argparse
import time

import cv2
import six
import skimage.io
import torch

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def bench_pytorch(img_file, gpu, times):
    print('==> Testing Mask R-CNN ResNet-C4 with PyTorch')
    torch.cuda.set_device(gpu)

    config_file = '../configs/caffe2/e2e_mask_rcnn_R_50_C4_1x_caffe2.yaml'

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    image = skimage.io.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in six.moves.range(5):
        predictions = coco_demo.compute_prediction(image)
        coco_demo.select_top_predictions(predictions)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in six.moves.range(times):
        predictions = coco_demo.compute_prediction(image)
        coco_demo.select_top_predictions(predictions)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--times', type=int, default=1000)
    args = parser.parse_args()

    img_file = 'https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/33823288584_1d21cf0a26_k.jpg'  # NOQA

    print('==> Benchmark: gpu=%d, times=%d' % (args.gpu, args.times))
    print('==> Image file: %s' % img_file)
    bench_pytorch(img_file, args.gpu, args.times)


if __name__ == '__main__':
    main()
