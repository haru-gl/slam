#pragma once
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "enclasses.h"


class featuremap_data {
public:
	//cv::Mat oImage;
	int mps_x;	// Map size x
	int mps_y;	// Map size y
	std::vector<cv::KeyPoint> oPts;
	cv::Mat oFeatures;
	std::vector<cv::Point2d> oMatchedPts;
	cv::Mat oMatchedFeatures;
	int dimFeatures;
	cv::Mat curr_tform;
	int newest_taken_idx;

	// 特徴点の色分けのため
	std::vector<cv::KeyPoint> prev_oPts;
	std::vector<cv::KeyPoint> new_oPts;



	featuremap_data(void)
	{
		//oImage = cv::Mat();
		mps_x = SZ;
		mps_y = SZ;
		oPts.clear();
		oFeatures = cv::Mat();
		oMatchedPts.clear();
		oMatchedFeatures = cv::Mat();
		dimFeatures = -1;
		curr_tform = cv::Mat();
		newest_taken_idx = 0;

		prev_oPts.clear();
	}
};


class source_data {
public:
	cv::Mat oImage;
	bool oImage_dummy;
	std::vector<cv::KeyPoint> oPts;
	cv::Mat oFeatures;
	std::vector<cv::Point2d> oMatchedPts;
	cv::Mat oMatchedFeatures;

	std::vector<int> selectedcm_srcdoPts;




	source_data(void)
	{
		oImage = cv::Mat();
		oImage_dummy = false;
		oPts.clear();
		oFeatures = cv::Mat();
		oMatchedPts.clear();
		oMatchedFeatures = cv::Mat();
	}

	~source_data(void)
	{
		oImage = cv::Mat();
		oPts.clear(); oPts.shrink_to_fit();
		oFeatures = cv::Mat();
		oMatchedPts.clear(); oMatchedPts.shrink_to_fit();
		oMatchedFeatures = cv::Mat();
	}

	void selectedcm_srcdoPts_init(int size)
	{
		selectedcm_srcdoPts.clear();
		selectedcm_srcdoPts.resize(size);
	}
};

class destination_data {
public:
	cv::Mat oImage;
	std::vector<cv::KeyPoint> oPts;
	cv::Mat oFeatures;
	std::vector<cv::Point2d> oMatchedPts;
	cv::Mat oMatchedFeatures;

	std::vector<int> selectedtd_dstdoPts;



	destination_data(void)
	{
		oImage = cv::Mat();
		oPts.clear();
		oFeatures = cv::Mat();
		oMatchedPts.clear();
		oMatchedFeatures = cv::Mat();
	}

	~destination_data(void)
	{
		oImage = cv::Mat();
		oPts.clear(); oPts.shrink_to_fit();
		oFeatures = cv::Mat();
		oMatchedPts.clear(); oMatchedPts.shrink_to_fit();
		oMatchedFeatures = cv::Mat();
	}

	void selectedtd_dstdoPts_init(int size) 
	{
		selectedtd_dstdoPts.clear();
		selectedtd_dstdoPts.resize(size);  // reserveでは、.size()関数でサイズ取得時に0扱いになってしまい不具合が生じる！
	}
};


class analysis_results {
public:
	int total_image_num;
	//double estimatedCenter2dx, estimatedCenter2dy;	//推定座標
	//double scale;									//推定スケール
	//double estimatedHeight;							//推定高度
	//long long elapsedTime;							//処理時間
	//size_t map_ptsNum, target_ptsNum;				//検出特徴点数
	size_t goodPairsNum;							//有効対応点数
	int status;										//状態：0:正検出、2:マッチング未了、3:インライア不足、4：その他))
	cv::Point2d c00, c01, c11, c10;					//地図における撮影画像の四隅の座標（左上c00、右上c01、右下c11、左下c10）


	// 以下，fmp用
	long long elapsed_time_formap;
	long long elapsed_time_formap_sum;
	long long elapsed_time_foraccuracy;

	// 以下、Accuracy用
	double norm_sum;
	int norm_size;
	double norm_ave;
	double norm_max;
	double norm_min;
	double norm_3sigma;

	// 地図生成時のオプションを，最後にfinals.csvへ書き出すため．
	featureDetectionType fd;

	analysis_results()
	{
		total_image_num = -1;
		goodPairsNum = -1;
		status = -1;

		// 以下，fmp用
		elapsed_time_formap      = 0.0;
		elapsed_time_formap_sum  = 0.0;
		elapsed_time_foraccuracy = 0.0;

		// 以下、Accuracy用
		norm_sum = 0;
		norm_size = 0;
		norm_ave = 0;
		norm_max = 0;
		norm_min = 0;
		norm_3sigma = 0;

		fd = featureDetectionType();
	}

	~analysis_results()
	{
		goodPairsNum = -1;
		status = -1;

		// 以下、Accuracy用
		norm_sum = 0;
		norm_size = 0;
	}
};


