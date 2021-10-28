import libCppInterface
import numpy as np
import os
import os.path as osp
from PIL import Image
_IMAGE_SIZE = 224
_NUM_CLASSES = 400

class Video:
    def __init__(self, file_name, temporal_window, batch_size, clip_optical_flow_at, is_only_for_rgb):
        self.file_name = file_name
        self.temporal_window = temporal_window
        self.batch_size = batch_size
        self.rgb_data = []
        self.flow_data = []
        self.batch_id = 0
        self.clip_optical_flow_at=int(clip_optical_flow_at)
        self.features = [] #np.array([])
        self.is_only_for_rgb = is_only_for_rgb
        self.loader = libCppInterface.LazyLoader()
        self.loader.initializeLazy(self.file_name, self.batch_size, self.temporal_window, self.is_only_for_rgb)

    def has_data(self):
        return self.loader.hasNextBatch()

    def get_batch(self):
        if self.loader.hasNextBatch():
            result_rgb = self.loader.nextBatchFrames()
            if not self.is_only_for_rgb:
                result_flow = self.loader.nextBatchFlows()
                return result_rgb, result_flow
            else:
                return result_rgb, None
        else:
            return None, None

    def append_feature(self, rgb_features, flow_features):
        for i in range(rgb_features.shape[0]):
            self.features.append( np.concatenate([rgb_features[i,:], flow_features[i,:]]) )
    '''
    def append_feature(self, rgb_features):
        for i in range(rgb_features.shape[0]):
            self.features.append(rgb_features[i,:])
    '''
    def finalize(self, dest_path):
        print(dest_path)
        data = np.array( self.features )
        print(data.shape)
        np.save(osp.join(dest_path, self.file_name[self.file_name.rfind('/')+1:]), data)

class ActiveVideo:
	def __init__(self, file_name, clip_optical_flow_at, fps=None):
		self.file_name = file_name
		self.clip_optical_flow_at=int(clip_optical_flow_at)
		if fps is None:
			self.fps = 0
		else:
			self.fps = fps

	def get_batch(self):
		loader = libCppInterface.ActiveLoader()
		loader.initialize(self.file_name, self.fps)

		rgb = loader.getFrames()
		flow = loader.getOpticalFlows(self.clip_optical_flow_at)

		if rgb.shape[1]<10:
			raise Exception('Video is very small')

		return rgb, flow

v = Video("/data/changjian/v_CricketShot_g04_c01.avi", 1, 1, 20, False)
# rgb, flow = v.get_batch()
# print(rgb.shape)
# print(flow.shape)
all_flow = []
all_rgb = []
while v.has_data():
    rgb, flow = v.get_batch()
    all_rgb.append(rgb)
    all_flow.append(flow)

active_v = ActiveVideo("/data/changjian/v_CricketShot_g04_c01.avi", 20)
rgb, flow = active_v.get_batch()

active_v2 = ActiveVideo("/data/changjian/v_CricketShot_g04_c01.avi", 20, 12)
rgb2, flow2 = active_v2.get_batch()

# for i in range(79):
#     a_flow = all_flow[i]
#     a_rgb = all_rgb[i]
#     b_flow = flow[:, i:i+1, :, :, :]
#     b_rgb = rgb[:, i:i+1, :, :, :]
#     err_flow = ((a_flow - b_flow)**2).sum()
#     err_rgb = ((a_rgb - b_rgb)**2).sum()
#     if err_flow > 0.0001 or err_rgb > 0.0001:
#         print(i, err_flow, err_rgb)
#         import IPython; IPython.embed(); exit()

# for idx, flow in enumerate(al):
#     flow = flow[0,0]
#     flow = flow[:, :, 0]
#     data = flow
#     print("{}-{}-{}-{}".format(data[0][0], 
#                         data[0][1], 
#                         data[1][0], 
#                         data[1][1]))
#     data = flow.astype(np.uint8)
#     data = data[:, :, np.newaxis].repeat(axis=2, repeats=3)
#     im = Image.fromarray(data)
#     im.save("data/{}_x.jpg".format(idx+1))

import IPython; IPython.embed(); exit()