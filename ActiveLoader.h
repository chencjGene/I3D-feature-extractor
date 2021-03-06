#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <Python.h>
#include <vector>
#include "Utils.h"
#include <boost/python.hpp>
using namespace boost::python;
using namespace std;
using namespace cv;
using namespace cv::cuda;
namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef _ACTIVELOADER_H_
#define _ACTIVELOADER_H_

class ActiveLoader {
private:
	//std::vector<Mat> frames;
	//std::vector<Mat> flows;
	std::list<Mat> frames;
	std::list<Mat> flows;

	void populateFrames(const char* video);
	void populateFrames_with_fps(const char* video, const uint fps);
	void populateOpticalFlows();
	
public:
	ActiveLoader() {}
	~ActiveLoader() { }
	void initialize(const char* videoFile, const uint fps);

	np::ndarray getOpticalFlows(float bound);
	np::ndarray getFrames();
	
};
#endif