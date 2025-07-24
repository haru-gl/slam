#pragma once
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "enclasses.h"
#include "sac_main.h"
#include "classes.h"


class reinforcementlearning : public sac
{
public:
	double alpha;
	size_t maxIteration;				//繰り返しの最大数
	double confidence;				//確度％
	double maxDistance;				//最大距離
	double ave, stddv;               //Average and standard deviation of the cordinate
	double med, medad;               //Median and Median absoute deviation
	double csd;
	reinforcementlearning(void)
	{
		alpha = INIT_RL_ALPHA;					//for Reinforcement Learning
		maxIteration = INIT_MAXITERATION;	//Basic ransac
		confidence = INIT_CONFIDENCE;		//Basic ransac
		maxDistance = INIT_MAXDISTANCE;		//Basic ransac
		ave = stddv = med = medad = 0.0;
		csd = 1.0;
	}
	void set_rflearning_al(double v = INIT_RL_ALPHA)
	{
		alpha = v;
	}
	double get_rflearning_al(void) const
	{
		return alpha;
	}
	size_t computeLoopNumbers(size_t numPts, size_t inlierNum, size_t sampleSize);
	bool matrixestimation(const matchingType mt, const matrixcalType ct, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform);

	void positionestimation_grfl(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rs);

	bool sac_grfl(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, cv::Mat& tform,
				  std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd);

	bool a_ransac_grfl(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);

};

