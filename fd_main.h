#pragma once
#include "classes.h"
#include "enclasses.h"
#include "akaze.h"
#include "surf.h"
#include "orb.h"
#include "sift.h"
#include "brisk.h"
#include "pca.h"

#include "functions.h"

#include <algorithm>



void featurepointdetection(featureDetectionType fd, source_data& srcd, destination_data& dstd);


enum sort_order_list {
	small2large, // 【昇順】　値が「小さいもの」から「大きいもの」へ
	large2small  // 【降順】　値が「大きいもの」から「小さいもの」へ
};

void sort_oc(const std::vector<double> src,			// 入力
			 std::vector<double>& order,		    // 出力1
			 std::vector<int>& count, 				// 出力2
			 sort_order_list sort_order);			// 設定項目（どの順番で並べるか）



void sq_make_list(sr sr, double th, int round_dpn,
	std::vector<cv::KeyPoint> oPts, cv::Mat oFeatures, std::string name, featureDetectionType fd,
	std::vector<bool>& mask);



void sq(std::vector<cv::KeyPoint>& oPts, cv::Mat& oFeatures, std::string name, featureDetectionType fd,
	std::vector<bool> mask);