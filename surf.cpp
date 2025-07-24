#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "surf.h"
#include "fd_main.h"


void surf::set_surfTh(double th)
{
	surf_Threshold = th;
}

double surf::get_surfTh(void) const
{
	return surf_Threshold;
}

void set_parameters(cv::Ptr<cv::xfeatures2d::SURF>& detector) {
	//detector->setExtended(surf_Extended);
	//detector->setHessianThreshold(surf_Threshold);
	//detector->setNOctaveLayers(surf_NOctaveLayers);
	//detector->setNOctaves(surf_NOctaves);
	//detector->setUpright(surf_Upright);

	detector->setExtended(INIT_SURF_EXTENDED);
	detector->setHessianThreshold(INIT_SURF_THRESHOLD);
	detector->setNOctaveLayers(INIT_SURF_NOctaveLayers);
	detector->setNOctaves(INIT_SURF_NOctaves);
	detector->setUpright(INIT_SURF_Upright);
}

void surf::featuredetection(featureDetectionType fd, source_data& srcd, destination_data& dstd)
{
	cv::Ptr<cv::xfeatures2d::SURF> surfDetector = cv::xfeatures2d::SURF::create();
	//surfDetector->setHessianThreshold(surf_Threshold);
	set_parameters(surfDetector);

	// srcdについて検出（詳細は`akaze.cpp`参照のこと）
	if (!srcd.oImage_dummy) {
		surfDetector->detectAndCompute(srcd.oImage, cv::Mat(), srcd.oPts, srcd.oFeatures);

		std::vector<bool> mask_src(srcd.oPts.size(), true); // 各特徴点について，size面，response面で「良い点」かを判断．どちらか一つの軸で「ダメ」扱いされたら即falseとなり，二度とtrueには直さない．

		switch (fd.sq_order)
		{
		case sqOrderType::size_sq_response_sq:
			if (fd.sq_size)		sq_make_list(sr::size, SQ_SIZE_TH, SQ_SIZE_ROUND_DPN, srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			if (fd.sq_size || fd.sq_response) sq(srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			mask_src.clear();
			mask_src.resize(srcd.oPts.size(), true);
			if (fd.sq_response) sq_make_list(sr::response, SQ_RESPONSE_TH, SQ_RESPONSE_ROUND_DPN, srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			if (fd.sq_size || fd.sq_response) sq(srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			break;

		case sqOrderType::response_sq_size_sq:
			if (fd.sq_response) sq_make_list(sr::response, SQ_RESPONSE_TH, SQ_RESPONSE_ROUND_DPN, srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			if (fd.sq_size || fd.sq_response) sq(srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			mask_src.clear();
			mask_src.resize(srcd.oPts.size(), true);
			if (fd.sq_size)		sq_make_list(sr::size, SQ_SIZE_TH, SQ_SIZE_ROUND_DPN, srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			if (fd.sq_size || fd.sq_response) sq(srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			break;

		case sqOrderType::size_response_sq:
			if (fd.sq_size)		sq_make_list(sr::size, SQ_SIZE_TH, SQ_SIZE_ROUND_DPN, srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			if (fd.sq_response) sq_make_list(sr::response, SQ_RESPONSE_TH, SQ_RESPONSE_ROUND_DPN, srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			if (fd.sq_size || fd.sq_response) sq(srcd.oPts, srcd.oFeatures, "srcd", fd, mask_src);
			break;

		default:
			printf("[akaze, surf,sift,....cpp] Error: そのタイプのsq_orderは定義されていません．\n");
			exit(EXIT_FAILURE);
			break;
		}
	}
	


	
	surfDetector->detectAndCompute(dstd.oImage, cv::Mat(), dstd.oPts, dstd.oFeatures);

	std::vector<bool> mask_dst(dstd.oPts.size(), true);

	switch (fd.sq_order)
	{
	case sqOrderType::size_sq_response_sq:
		if (fd.sq_size)		sq_make_list(sr::size, SQ_SIZE_TH, SQ_SIZE_ROUND_DPN, dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		if (fd.sq_size || fd.sq_response) sq(dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		mask_dst.clear();
		mask_dst.resize(dstd.oPts.size(), true);
		if (fd.sq_response) sq_make_list(sr::response, SQ_RESPONSE_TH, SQ_RESPONSE_ROUND_DPN, dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		if (fd.sq_size || fd.sq_response) sq(dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		break;

	case sqOrderType::response_sq_size_sq:
		if (fd.sq_response) sq_make_list(sr::response, SQ_RESPONSE_TH, SQ_RESPONSE_ROUND_DPN, dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		if (fd.sq_size || fd.sq_response) sq(dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		mask_dst.clear();
		mask_dst.resize(dstd.oPts.size(), true);
		if (fd.sq_size)		sq_make_list(sr::size, SQ_SIZE_TH, SQ_SIZE_ROUND_DPN, dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		if (fd.sq_size || fd.sq_response) sq(dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		break;

	case sqOrderType::size_response_sq:
		if (fd.sq_size)		sq_make_list(sr::size, SQ_SIZE_TH, SQ_SIZE_ROUND_DPN, dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		if (fd.sq_response) sq_make_list(sr::response, SQ_RESPONSE_TH, SQ_RESPONSE_ROUND_DPN, dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		if (fd.sq_size || fd.sq_response) sq(dstd.oPts, dstd.oFeatures, "dstd", fd, mask_dst);
		break;

	default:
		printf("[akaze, surf,sift,....cpp] Error: そのタイプのsq_orderは定義されていません．\n");
		exit(EXIT_FAILURE);
		break;
	}
}

