#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "classes.h"
#include "enclasses.h"
#include "knn.h"
#include "classes.h"    // For clipedmap_data, target_data
#include "enclasses.h"  // For knnType enum
#include "flann_lsh_matcher.h"


void matrow_swap(cv::Mat& des, int i, int j)
{
	unsigned char tmp;
	for (int k = 0; k < des.cols; k++) {
		tmp = des.at<unsigned char>(i, k);
		des.at<unsigned char>(i, k) = des.at<unsigned char>(j, k);
		des.at<unsigned char>(j, k) = tmp;
	}
}


void knn::set_knn_sortflag(bool st)
{
	knn_sort = st;
}



size_t knn::matchiratiocheck(std::vector<std::vector<cv::DMatch>>& nn_matches, source_data& srcd, destination_data& dstd, std::vector<cv::Point2d>& srcts, cv::Mat& srcdes, std::vector<cv::Point2d>& dstpts, cv::Mat& dstdes, std::vector<float>& ratio, std::vector<std::vector<cv::DMatch>>& good_matches)
{
	size_t candnum = 0;
	dstpts.clear(); dstdes = cv::Mat(); srcts.clear(); srcdes = cv::Mat(); ratio.clear();
	for (size_t n = 0; n < nn_matches.size(); n++) {
		cv::DMatch first = nn_matches[n][0];
		if ((double)(nn_matches[n][0].distance) < knn_matchratio * (double)(nn_matches[n][1].distance)) {
			candnum++;
			dstpts.push_back((cv::Point2d)dstd.oPts[first.queryIdx].pt);
			dstdes.push_back(dstd.oFeatures.row(first.queryIdx));
			srcts.push_back((cv::Point2d)srcd.oPts[first.trainIdx].pt);
			srcdes.push_back(srcd.oFeatures.row(first.trainIdx));
			ratio.push_back(fabs(nn_matches[n][0].distance / nn_matches[n][1].distance));

			good_matches.push_back(nn_matches[n]);
		}
	}
	return candnum;
}




void kp_sort(std::vector<cv::Point2d>& srcpts, cv::Mat& srcdes, std::vector<cv::Point2d>& dstpts, cv::Mat& dstdes, std::vector<float>& ratio)
{
	size_t num_kpts = srcpts.size();
	for (size_t i = 0; i < num_kpts - 1; i++)
		for (size_t j = i + 1; j < num_kpts; j++)
			if (ratio[i] > ratio[j]) {
				std::swap(ratio[i], ratio[j]);
				std::swap(srcpts[i], srcpts[j]);
				matrow_swap(srcdes, (int)i, (int)j);
				std::swap(dstpts[i], dstpts[j]);
				matrow_swap(dstdes, (int)i, (int)j);
			}
}



size_t knn::match(featureType ft, knnType kt, source_data& srcd, destination_data& dstd)
{
	bool use_flann = false;

	//	k-NN
	cv::BFMatcher matcher;
	cv::Ptr<cv::DescriptorMatcher> flann_matcher;

	const bool isCrossCheck = false;
	switch (ft)
	{
	case featureType::fAKAZE:
	case featureType::fORB:
	case featureType::fBRISK:
	case featureType::fPCA_uschar:
		//matcher = cv::BFMatcher(knn_normType, isCrossCheck);
		if (use_flann) flann_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
		else           matcher = cv::BFMatcher(cv::NORM_HAMMING, isCrossCheck);
		break;
	case featureType::fSURF:
	case featureType::fSIFT:
	case featureType::fPCA_float:
		if (use_flann) flann_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); 
		else           matcher = cv::BFMatcher(cv::NORM_L2, isCrossCheck);
		break;
	case featureType::fPCA_16bin:
		matcher = cv::BFMatcher(cv::NORM_HAMMING, isCrossCheck);
		break;
	default:
		error_log("[%s]:[%s] そのfeatureTypeは実装されていません．\n", __FILE__, __FUNCTION__);
		exit(EXIT_FAILURE);
		break;
	}
	
	std::vector<std::vector<cv::DMatch>> nn_matches;
	std::vector<std::vector<cv::DMatch>> good_matches;
	
	if (use_flann) flann_matcher->knnMatch(dstd.oFeatures, srcd.oFeatures, nn_matches, knn_k);
	else           matcher.knnMatch(dstd.oFeatures, srcd.oFeatures, nn_matches, knn_k);
	//								tgt							cmp


		// 対応点draw
	if (IS_IMSHOW) {
		cv::Mat match_img_init;
		cv::drawMatches(dstd.oImage, dstd.oPts, srcd.oImage, srcd.oPts, nn_matches, match_img_init);
		cv::resize(match_img_init, match_img_init, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
		cv::imshow("[k-NN]match_img_init", match_img_init);
		cv::waitKey(1);
	}








	size_t goodPairsNum = 0;
	size_t candnum = 0;
	std::vector<cv::Point2d> dstpts, srcpts;
	cv::Mat dstdes, srcdes;
	std::vector<float> ratio;
	size_t rejt = 0, rejm = 0;
	size_t rfixn;


	srcd.oMatchedPts.clear(); srcd.oMatchedFeatures = cv::Mat(); dstd.oMatchedPts.clear(); dstd.oMatchedFeatures = cv::Mat();
	srcpts.clear(); srcdes = cv::Mat(); dstpts.clear(); dstdes = cv::Mat();


	switch (kt) {
	case knnType::kNORMAL:
		goodPairsNum = matchiratiocheck(nn_matches, srcd, dstd, srcd.oMatchedPts, srcd.oMatchedFeatures, dstd.oMatchedPts, dstd.oMatchedFeatures, ratio, good_matches);

		if (knn_sort) kp_sort(srcd.oMatchedPts, srcd.oMatchedFeatures, dstd.oMatchedPts, dstd.oMatchedFeatures, ratio);
		break;

	default:
		warn_log("[knn.cpp（96行目）] k-NNの設定例外。終了します。\n");
		exit(EXIT_FAILURE);
	}

	dstpts.clear(); dstpts.shrink_to_fit();
	srcpts.clear(); srcpts.shrink_to_fit();
	ratio.clear(); ratio.shrink_to_fit();
	dstdes = cv::Mat(); srcdes = cv::Mat();

	//nn_matches.clear();
	//nn_matches.shrink_to_fit();
	info_log("goodPairsNum after kNN=%d\n", goodPairsNum);



	// 対応点draw
	if (IS_IMSHOW) {
		cv::Mat match_img;
		cv::drawMatches(dstd.oImage, dstd.oPts, srcd.oImage, srcd.oPts, good_matches, match_img);
		cv::resize(match_img, match_img, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
		cv::imshow("[k-NN]match_img", match_img);
		cv::waitKey(1);
	}


	return goodPairsNum;
}