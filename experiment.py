import os
import random
import FrameStackExtractor
import SlideWindowExtractor
import numpy as np
import cv2

# CHECKPOINT_PATHS
#     'rgb': 'data/checkpoints/rgb_imagenet/model.ckpt'
#     'flow': 'data/checkpoints/flow_imagenet/model.ckpt'
# need libCppInterface.so

def experiment(mode, videos, dest, experiment_name):
    # mode: string, 16frame or slide_window
    # videos: array, filenames
    # dest: dest to save features
    # experiment_name: filename to save log
    if mode == '16frame':
        FrameStackExtractor.main(videos, dest, experiment_name=experiment_name)
    elif mode == 'slide_window':
        SlideWindowExtractor.main(videos, dest_path=dest, experiment_name=experiment_name)
        # need i3d.py
    else:
        print('Please select a mode between 16frame and slide_window!')
