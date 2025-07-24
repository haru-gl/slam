﻿#pragma once
#include "parameters.h"
#include "enclasses.h"
#include "classes.h"

class sac {
public:
	size_t maxIteration;				//繰り返しの最大数
	double confidence;				//確度％
	double maxDistance;				//最大距離
	double ave, stddv;               //Average and standard deviation of the cordinate
	double med, medad;               //Median and Median absoute deviation
	double csd;

	sac(void) {
		maxIteration = INIT_MAXITERATION;	//Basic ransac
		confidence = INIT_CONFIDENCE;		//Basic ransac
		maxDistance = INIT_MAXDISTANCE;		//Basic ransac
		ave = stddv = med = medad = 0.0;
		csd = 1.0;
	}

	//void set_maxIteration(size_t th = INIT_MAXITERATION);
	//int get_maxIteration(void) const;
	//void set_confidence(double th = INIT_CONFIDENCE);
	//double get_confidence(void) const;
	//void set_maxDistance(double th = INIT_MAXDISTANCE);
	//double get_maxDistance(void) const;
	size_t computeLoopNumbers(size_t numPts, size_t inlierNum, size_t sampleSize);
	bool matrixestimation(const matchingType mt, const matrixcalType ct, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform);
	bool a_ransac(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	//bool a_ransac_nd(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);

	bool sac_normal(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, cv::Mat& tform,
					std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd);
	bool sac_normal_opencv(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, cv::Mat& tform,
								std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd);
	void positionestimation_normal(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rs);

	//bool sac_normal_dr(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	//void positionestimation_normal_dr(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);

	//bool sac_dr_only(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	//bool sac_dr(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	//void positionestimation_dr(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);

	////	bool sac_drransac(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	////	void positionestimation_drransac(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);

	//bool dr_ransac(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	////	bool dr_ransac_nd(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);

	
	// 描画系
	void draw_RANSAC(source_data srcd, destination_data dstd, std::vector<cv::Point2d> selectedcm, std::vector<cv::Point2d> selectedtd);
	void draw_RANSAC_elegant(source_data srcd, destination_data dstd, std::vector<cv::Point2d> selectedcm, std::vector<cv::Point2d> selectedtd);
};
