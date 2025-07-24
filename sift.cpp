#include <opencv2/opencv.hpp>
#include "sift.h"
#include "fd_main.h"

void sift::set_sift_contrastTh(double th)
{
	contrastThreshold = th;
}

double sift::get_sift_contrastTh(void) const
{
	return contrastThreshold;
}

void sift::set_sift_egdeTh(double th)
{
	edgeThreshold = th;
}

double sift::get_sift_edgeTh(void) const
{
	return edgeThreshold;
}

void sift::set_sift_sigma(double th)
{
	sigma = th;
}

double sift::get_sift_sigma(void) const
{
	return sigma;
}

void sift::featuredetection(featureDetectionType fd, source_data& srcd, destination_data& dstd)
{

	cv::Ptr<cv::SIFT> detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);


	if (!srcd.oImage_dummy) {
		detector->detectAndCompute(srcd.oImage, cv::Mat(), srcd.oPts, srcd.oFeatures);


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
	


	detector->detectAndCompute(dstd.oImage, cv::Mat(), dstd.oPts, dstd.oFeatures);

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

