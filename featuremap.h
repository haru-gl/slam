#pragma once

#include <opencv2/opencv.hpp>

#include "classes.h"
#include "functions.h"
#include "parameters.h"
#include "enclasses.h"
#include "tmatrix.h"

class FeatureMap
{
public:
	MappingType mt;
	bool reDetection;	// 撮影画像をtform行列を利用して変形し，改めて特徴点検出を実施して，検出された特徴点，特徴量情報をfmpdにコピーするオプション
	int interpolation;	// 変換手法

	FeatureMap(void) {
		
	}

	void feature_mapping(featureDetectionType fd, FeatureMap fm, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rst);

	const char* get_MappingType()
	{
		switch (mt)
		{
		case MappingType::mALL:
			return "mALL";
			break;
		case MappingType::mSIMILAR:
			return "mSIMILAR";
			break;
		case MappingType::mOLDER:
			return "mOLDER";
			break;
		case MappingType::mNEWER:
			return "mNEWER";
			break;
		default:
			error_log("[featuremap.h] get_Mapping_type: 想定外の型です。\n");
			exit(-1);			
		}
		
	}
};



void Correspond_selectedcm_selectedtd(source_data& srcd, destination_data& dstd, std::vector<cv::Point2d> selectedcm, std::vector<cv::Point2d> selectedtd);

