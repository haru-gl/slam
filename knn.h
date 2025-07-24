﻿#pragma once
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "classes.h"
#include "enclasses.h"


class knn {
private:
	int knn_normType;
	int knn_k;
	double knn_matchratio;
	size_t knn_fixn;
	double knn_spDist;
	bool knn_sort;
public:
	knn(void)
	{
		//knn_normType = cv::NORM_HAMMING;
		knn_normType = -1;	// SURFの場合に対応できるよう，ノルムの種類に応じた使い分けを実装．場所はknn::match(@knn.cpp).
		knn_k = INIT_KNNK;
		knn_matchratio = INIT_NNMATCHRATIO;
		knn_fixn = INIT_MAXNUM;
		knn_spDist = INIT_SAMEPOINT;
		knn_sort = INIT_KNNSORT;
	}
	//void set_knn_k(int k = INIT_KNNK);
	//int get_knn_k(void) const;
	//void set_knn_matchratio(double mr = INIT_NNMATCHRATIO);
	//double get_knn_matchratio(void) const;
	void set_knn_sortflag(bool st = INIT_KNNSORT);
	//bool get_knn_sortflag(void) const;
	//void set_knn_fixn(size_t maxn);
	//size_t get_knn_fixn(void) const;
	//void set_knn_spDist(double spd = INIT_SAMEPOINT);
	//double get_knn_spDist(void) const;
	size_t matchiratiocheck(std::vector<std::vector<cv::DMatch>>& nn_matches, source_data& srcd, destination_data& dstd, std::vector<cv::Point2d>& srcts, cv::Mat& srcdes, std::vector<cv::Point2d>& dstpts, cv::Mat& dstdes, std::vector<float>& ratio, std::vector<std::vector<cv::DMatch>>& good_matches);
	//size_t detect_samepoint(std::vector<cv::Point2d>& pts, std::vector<float>& ratio);
	size_t match(featureType ft, knnType kt, source_data& srcd, destination_data& dstd);
};