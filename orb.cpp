#include <opencv2/opencv.hpp>
#include "orb.h"
#include "fd_main.h"

void orb::set_orb_maxfeatures(int mf)
{
	orb_maxfeatures = mf;
}

int orb::get_orb_maxfeatures(void) const
{
	return orb_maxfeatures;
}

void orb::set_orb_scalefactor(double mf)
{
	orb_scalefactor = mf;
}

double orb::get_orb_scalefactor(void) const
{
	return orb_scalefactor;
}

void orb::set_orb_edgethreshold(int mf)
{
	orb_edgethreshold = mf;
}

int orb::get_orb_edgethreshold(void) const
{
	return orb_edgethreshold;
}

void orb::set_orb_fastthreshold(int mf)
{
	orb_fastthreshold = mf;
}

int orb::get_orb_fastthreshold(void) const
{
	return orb_fastthreshold;
}

void orb::set_parameters(cv::Ptr<cv::ORB>& detector)
{
	detector->setMaxFeatures(orb_maxfeatures);		//500:The maximum number of features to retain
	detector->setScaleFactor(orb_scalefactor);		//1.2f:Pyramid decimation ratio, greater than 1.
	detector->setNLevels(3);					//3:The number of pyramid levels.
	detector->setEdgeThreshold(orb_edgethreshold);	//31:Size of the border where the features are not detected.
	detector->setFirstLevel(0);					//0:It should be 0.
	detector->setWTA_K(2);						//
	detector->setScoreType(cv::ORB::HARRIS_SCORE);//HARRIS_SCORE, FAST_SCORE
	detector->setPatchSize(31);					//31:Size of the patch used by the oriented BRIEF descriptor.
	detector->setFastThreshold(orb_fastthreshold);	//20:
}

void orb::featuredetection(featureDetectionType fd, source_data& srcd, destination_data& dstd)
{
	cv::Ptr<cv::ORB> detector = cv::ORB::create();
	set_parameters(detector);

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
