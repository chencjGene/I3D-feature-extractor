import numpy as np
import tensorflow as tf
import os
import utils_feature as uf
import argparse
import libCppInterface
import os.path as osp
from PIL import Image
import i3d
import sonnet as snt
import cv2


IMAGE_SIZE = 224
INPUT_VIDEO_FRAMES = 16
NUM_CLASSES = 400

CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow': 'data/checkpoints/flow_imagenet/model.ckpt',
}

def change(input_file, output_file):
    vid = cv2.VideoCapture(input_file)
    w, h = [int(vid.get(3)), int(vid.get(4))]
    fps = vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 25, (w,h))
    while(True):
        ret, frame = vid.read()
        if ret:
            out.write(frame)
        else:
            break
    out.release()

class BatchVideo:
    def __init__(self, file_name, batch_size, clip_optical_flow_at=20):
        self.file_name = file_name
        self.temporal_window = 1
        self.batch_size = batch_size
        self.rgb_data = []
        self.flow_data = []
        self.batch_id = 0
        self.clip_optical_flow_at=int(clip_optical_flow_at)
        self.features = [] #np.array([])
        self.is_only_for_rgb = False
        self.loader = libCppInterface.LazyLoader()
        self.loader.initializeLazy(self.file_name, self.batch_size, self.temporal_window, self.is_only_for_rgb)

    def has_data(self):
        return self.loader.hasNextBatch()

    def get_batch(self):
        if self.loader.hasNextBatch():
            result_rgb = self.loader.nextBatchFrames()
            if not self.is_only_for_rgb:
                result_flow = self.loader.nextBatchFlows()
                return result_rgb.squeeze(axis=1), result_flow.squeeze(axis=1)
            else:
                return result_rgb, None
        else:
            return None, None


class FeatureExtractor:
    def __init__(self, vid_path):
        v = BatchVideo(vid_path, 50)
        all_flow = []
        all_rgb = []
        while v.has_data():
            rgb, flow = v.get_batch()
            all_rgb.append(rgb)
            all_flow.append(flow)
        self.rgb = np.concatenate(all_rgb)
        self.flow = np.concatenate(all_flow)
        # np.save('rgb.npy', self.rgb)
        # np.save('flow.npy', self.flow)
        # read rgb and flow from ActiveVideo
        # self.rgb = np.load('rgb.npy')
        # self.flow = np.load('flow.npy')
        # self.rgb = self.rgb[0]
        # self.flow = self.flow[0]

        print('rgb shape', self.rgb.shape)
        print('flow shape', self.flow.shape)
        print('###############################')
        self.num_vid_frame = self.rgb.shape[0]
        print('frames', self.num_vid_frame)
        print('###############################')
        self.num_segments = int(self.num_vid_frame / INPUT_VIDEO_FRAMES)
        print('segments', self.num_segments)
        print('###############################')
        # number of total segments (ex. 1 feature vector per 16 frames, so total segments will be (total frames / 16))
        self.num_hundred_segments = int(self.num_segments / 100)

        init = tf.global_variables_initializer()

    def extractor_rgb(self):
        channel = 3
        stream = 'rgb'
        X_feature = np.zeros((self.num_segments, 1024))
        X_feature_v2 = np.zeros((self.num_segments, 1024))

        vid = self.rgb
        vid = vid.astype(np.float32)
        # vid = vid[:, :, :, ::-1]
        # convert BGR to RGB

        for j in range(self.num_hundred_segments + 1):
            if j == self.num_hundred_segments:
                extract_size = self.num_segments - self.num_hundred_segments * 100
                if extract_size == 0:
                    break
            else:
                extract_size = 100

            tf.reset_default_graph()
            feature_saver, feature_input, model_logits = uf.get_model(stream, extract_size)
            frame_inputs = np.zeros((extract_size, INPUT_VIDEO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, channel))
            for k in range(extract_size):
                frame_inputs[k] = vid[j * INPUT_VIDEO_FRAMES * 100 + k * INPUT_VIDEO_FRAMES:
                                      j * INPUT_VIDEO_FRAMES * 100 + (k + 1) * INPUT_VIDEO_FRAMES]
            
            X_feature[j * 100: j * 100 + extract_size] = uf.get_feature(frame_inputs, stream, extract_size,
                                                                        feature_saver, feature_input, model_logits)

        return X_feature


    def extractor_flow(self):
        channel = 2
        stream = 'flow'
        X_feature = np.zeros((self.num_segments, 1024))
        X_feature_v2 = np.zeros((self.num_segments, 1024))

        vid = self.flow
        vid = vid.astype(np.float32)

        for j in range(self.num_hundred_segments + 1):
            if j == self.num_hundred_segments:
                extract_size = self.num_segments - self.num_hundred_segments * 100
                if extract_size == 0:
                    break
            else:
                extract_size = 100

            tf.reset_default_graph()
            feature_saver, feature_input, model_logits = uf.get_model(stream, extract_size)
            frame_inputs = np.zeros((extract_size, INPUT_VIDEO_FRAMES, IMAGE_SIZE, IMAGE_SIZE, channel))
            for k in range(extract_size):
                frame_inputs[k] = vid[j * INPUT_VIDEO_FRAMES * 100 + k * INPUT_VIDEO_FRAMES:
                                      j * INPUT_VIDEO_FRAMES * 100 + (k + 1) * INPUT_VIDEO_FRAMES]
            X_feature[j * 100: j * 100 + extract_size] = uf.get_feature(frame_inputs, stream, extract_size,
                                                                        feature_saver, feature_input, model_logits)

        return X_feature


def main(videos, dest, experiment_name):
    error_str = ''
    for video in videos:
        # if os.path.exists(dest + 'rgb/' + filename + '.npy'):
        #     continue
        try:
            change(video, './' + experiment_name + '.avi')
            extractor = FeatureExtractor('./' + experiment_name + '.avi')
            rgb = extractor.extractor_rgb()
            flow = extractor.extractor_flow()
            filename = video[video.rfind('/')+1:]
            np.save(os.path.join(dest, 'rgb/' + filename + '.npy'), rgb)
            np.save(os.path.join(dest, 'flow/' + filename + '.npy'), flow)
            os.remove('./' + experiment_name + '.avi')
        except Exception as e:
            print(str(e))
            error_str += video
            error_str += '\n'
    name = experiment_name + '.txt'
    with open(name, 'w') as file:
        file.write(error_str)

