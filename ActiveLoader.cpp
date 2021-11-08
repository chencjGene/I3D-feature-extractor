#include "ActiveLoader.h"

void ActiveLoader::initialize(const char* video, const uint fps) {
	frames.clear();
	flows.clear();
	if (!video) {
		PyErr_SetString(PyExc_TypeError, "Video File shouldn't be null or empty");
		p::throw_error_already_set();
	}
	if (fps == 0){
		populateFrames(video);
	}
	else{
		populateFrames_with_fps(video, fps);
	}
	if (frames.size() == 0) {
		PyErr_SetString(PyExc_TypeError, "Unable to Process video");
		p::throw_error_already_set();	
	}

	populateOpticalFlows();
	if (flows.size() == 0) {
		PyErr_SetString(PyExc_TypeError, "Unable to Process video");
		p::throw_error_already_set();			
	}
}

void ActiveLoader::populateFrames_with_fps(const char* video, const uint fps) {
	VideoCapture capture;
	capture.open(video);
	if (!capture.isOpened()) {
		PyErr_SetString(PyExc_TypeError, "Unable to open the video");
		p::throw_error_already_set();
	}

	int frame_rate = 1000 / fps;
	int sec = 0;
	
	Mat frame;
	int frameNum = 0;
	while(true) {
		capture.set(CAP_PROP_POS_MSEC, sec);
		capture>>frame;
		if (frame.empty()) {
			break;
		}
		sec = sec + frame_rate;
		frameNum += 1;
		frames.push_back(frame.clone());
	}
}

void ActiveLoader::populateFrames(const char* video) {
	VideoCapture capture;
	capture.open(video);
	if (!capture.isOpened()) {
		PyErr_SetString(PyExc_TypeError, "Unable to open the video");
		p::throw_error_already_set();
	}

	int step = 1;

	Mat frame;
	int frameNum = 0;
	while(true) {
		capture>>frame;
		if (frame.empty()) {
			break;
		}
		capture.set(CAP_PROP_POS_FRAMES, frameNum + step);
		frameNum += step;
		frames.push_back(frame.clone());
	}
}

void ActiveLoader::populateOpticalFlows() {
	Utils::populateOpticalFlow(frames, flows);
}

np::ndarray ActiveLoader::getOpticalFlows(float bound) {
	return Utils::convertOpticalFlowsToNPArray(flows, bound);
}

np::ndarray ActiveLoader::getFrames() {
	return Utils::convertRGBFramesToNPArray(frames);
}

