#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"
#include "enclasses.h"

class surf {
private:
	bool surf_Extended;
	double surf_Threshold;
	int surf_NOctaveLayers;
	int surf_NOctaves;
	bool surf_Upright;

public:
	surf(void)
	{
		surf_Extended = INIT_SURF_EXTENDED;
		surf_Threshold = INIT_SURF_THRESHOLD;
		surf_NOctaveLayers = INIT_SURF_NOctaveLayers;
		surf_NOctaves = INIT_SURF_NOctaves;
		surf_Upright = INIT_SURF_Upright;
	}
	void set_surfTh(double th = INIT_SURF_THRESHOLD);
	double get_surfTh(void) const;
	void featuredetection(featureDetectionType fd, source_data& srcd, destination_data& dstd);
	//void set_parameters(cv::Ptr<cv::xfeatures2d::SURF>& detector);
};

