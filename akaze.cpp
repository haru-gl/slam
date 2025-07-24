#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "akaze.h"
#include "classes.h"
#include "fd_main.h"




void akaze::set_parameters(cv::Ptr<cv::AKAZE>& detector)
{
	detector->setDescriptorType(cv::AKAZE::DESCRIPTOR_MLDB);
	detector->setDiffusivity(cv::KAZE::DIFF_PM_G2);
	detector->setThreshold(akaze_threshold);
	detector->setDescriptorSize(akaze_descriptor_size);
	detector->setDescriptorChannels(akaze_descriptor_channels);
	detector->setNOctaves(akaze_nOctaves);
	detector->setNOctaveLayers(akaze_nOctaveLayers);
}


void akaze::featuredetection(featureDetectionType fd, source_data &srcd, destination_data &dstd)
{
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	set_parameters(detector);
	// srcdについて検出
	// もしsrcdのoImageが空ならば、二回目以降の実施であることが明らか。
	// 画像がない場合は、srcdに対してdetectandcomputeできないし、する必要もないので、dstdに対してだけ行うことにする。
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
		

		
		//std::cout << "srcd | Row=" << srcd.oFeatures.rows << ",Col=" << srcd.oFeatures.cols << ",type=" << srcd.oFeatures.type() << std::endl;
		//std::cout << "       ptsNum=" << srcd.oPts.size() << "\n";
	}



	
	//for (int y = 0; y < srcd.oFeatures.rows; y++) {
	//	printf("=====キーポイントNo. [%3d]=====\n", y);
	//	printf("    座標：\n");
	//	printf("        x: %f\n", srcd.oPts[y].pt.x);
	//	printf("        y: %f\n", srcd.oPts[y].pt.y);
	//	printf("    ");
	//	for (int x = 0; x < srcd.oFeatures.cols; x++) {
	//		printf("%4d, ", (int)srcd.oFeatures.at<unsigned char>(y, x));
	//		//std::cout << (int)cm.oFeatures.at<unsigned char>(y, x) << ", ";	// 記述子の中身を表示
	//	}
	//	putchar('\n\n');
	//}
	//putchar('\n\n');


	// dstdについて検出
	detector->detectAndCompute(dstd.oImage, cv::Mat(), dstd.oPts, dstd.oFeatures);
	//std::cout << "dstd | Row=" << dstd.oFeatures.rows << ",Col=" << dstd.oFeatures.cols << ",type=" << dstd.oFeatures.type() << std::endl;
	//std::cout << "       ptsNum=" << dstd.oPts.size() << "\n";


	// size()が小さい点だけに絞り込む場合
	//if (fd.sq_size)		sq_size(dstd.oPts, dstd.oFeatures, "dstd", fd);
	//if (fd.sq_response) sq_response(dstd.oPts, dstd.oFeatures, "dstd", fd);
	

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