#include <opencv2/opencv.hpp>
#include "pca.h"
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include "parameters.h"
#include "fd_main.h"
#include <omp.h>

#define PI 3.14159265358979


cv::Mat pca::cnv_pca2img(std::vector<std::vector<double>>& pcav)
{
	double pmax, pmin;
	pmax = pmin = pcav[0][0];
	for (int iy = 0; iy < pcav.size(); iy++)
		for (int ix = 0; ix < pcav[0].size(); ix++)
			if (pmax < pcav[iy][ix]) pmax = pcav[iy][ix];
			else if (pmin > pcav[iy][ix]) pmin = pcav[iy][ix];

	double scl = 255.0 / (pmax - pmin);
	//std::cout << "max=" << pmax << ",min=" << pmin << ",scl=" << scl << std::endl;
	cv::Mat pca_img = cv::Mat::zeros((int)pcav.size(), (int)pcav[0].size(), CV_8UC1);
	int ic = 0;
	for (int iy = 0; iy < (int)pcav.size(); iy++)
		for (int ix = 0; ix < (int)pcav[0].size(); ix++)
			pca_img.at<unsigned char>(iy, ix) = (unsigned char)((pcav[iy][ix] - pmin) * scl);
	return pca_img;
}

bool pca::pcas_load(const std::string fnm, std::vector<std::vector<std::vector<double>>>& uvec, int numvec)
{
	std::ifstream ifile(fnm);
	if (!ifile) {
		std::cerr << "Load file not open!" << std::endl;
		return false;
	}
	std::cout << "file=" << fnm;
	std::string line;
	if (!std::getline(ifile, line)) {
		ifile.close(); return false;
	}
	int num = std::stoi(line);
	std::cout << " number of vectors=" << num;
	if (num != numvec) {
		ifile.close(); return false;
	}
	if (!std::getline(ifile, line)) {
		ifile.close(); return false;
	}
	int numy = std::stoi(line);
	std::cout << " numy=" << numy;
	if (!std::getline(ifile, line)) {
		ifile.close(); return false;
	}
	int numx = std::stoi(line);
	std::cout << " numx=" << numx << std::endl;

	uvec.resize(numvec, std::vector <std::vector<double>>(numy, std::vector<double>(numx)));
	for (int i = 0; i < num; i++) {
		for (int iy = 0; iy < numy; iy++) {
			for (int ix = 0; ix < numx; ix++) {
				if (!std::getline(ifile, line)) return false;
				uvec[i][iy][ix] = std::stod(line);
				//std::cout << " uvec[" << iy << "][" << ix << "]=" << uvec[i][iy][ix] << std::endl;
			}
		}
	}
	ifile.close();
	return true;
}

std::vector<std::vector<std::vector<double>>> pca::get_uvec_ad0(void)
{
	// ====================主成分のテンプレート読込====================
	std::string pcafile = "./pcas_ad0_15.txt";
	std::vector<std::vector<std::vector<double>>> uvec_ad0;
	int vectorn = 10;//確認する主成分ベクトル数
	if (!pcas_load(pcafile, uvec_ad0, vectorn)) {
		printf("PCA load Failed.\n");
	}
	else {
		printf("PCA load Successed.\n");

		if (IS_IMSHOW) {
			for (int i = 0; i < vectorn; i++) {
				cv::Mat pca_img = cnv_pca2img(uvec_ad0[i]);
				cv::resize(pca_img, pca_img, cv::Size(uvec_ad0[0].size() * 10, uvec_ad0[0][0].size() * 10), -1, -1, cv::INTER_AREA);
				char nam[100];
				snprintf(nam, sizeof(nam), "PCA_AD0_%02d", i); // sprintf_s関数はLinuxで使用できないので変更．引数が異なるので注意（sizeofが必要）．
				cv::imshow(nam, pca_img);
				cv::imwrite("./PCA_" + std::to_string(i) + ".bmp", pca_img);
			}
			cv::waitKey(1);
		}

	}

	return uvec_ad0;
}

std::vector<std::vector<std::vector<double>>> pca::get_luvecs2(void)
{
	// ====================アングル計算用の学習済み？主成分の読込====================
	int vectorln = 2;//mskn使用
	std::string lpcafile = "./learnedpca_15.txt";
	std::vector<std::vector<std::vector<double>>> luvecs2;// (vectorln, std::vector<std::vector<double>>(psize, std::vector<double>(psize)));
	if (!pcas_load(lpcafile, luvecs2, vectorln)) {
		printf("PCA load Failed.\n");
	}
	else {
		printf("PCA load Successed.\n");
	}
	return luvecs2;
}

double pca::cal_angle(const cv::Mat& cworg, float pkx, float pky)
{
	int ma_y = (int)pca_degree_x.size(), ma_x = (int)pca_degree_x[0].size();
	int ix = (int)pkx - ma_x / 2, iy = (int)pky - ma_y / 2;
	double sum_x = 0.0, sum_y = 0.0;
	for (int py = 0; py < ma_y; py++)
		for (int px = 0; px < ma_x; px++) {
			double pix = (double)cworg.at<unsigned char>(iy + py, ix + px);
			sum_x += pix * pca_degree_x[py][px];
			sum_y += pix * pca_degree_y[py][px];
		}
	return atan2(sum_y, sum_x) / PI * 180.0;
}



double pca::cal_corr(const cv::Mat& image, const std::vector<std::vector<double>>& pca_xyz, cv::Point2d center, int size, int row, int col)
{
	return 
		(double)image.at<unsigned char>(center - cv::Point2d(size/2, size/2) + cv::Point2d(col, row))
		* pca_xyz[row][col];
}


void pca::detectAndCompute(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor, featureDetectionType fd)
{
	// 初期化
	keypoint.clear();


	// ====================マッチングの下準備====================
	int mcy = (int)pca_x.size(), mcx = (int)pca_x[0].size();
	int wwy = image.rows - mcy, wwx = image.cols - mcx;
	std::vector<std::vector<double>> corr_r(wwy, std::vector<double>(wwx));


	//double rate = 1.25; // 1.25倍の拡大までを見据えて，大きな配列をメモリ上に確保しておく．（scale評価時 用）

	//// メモリ解放・再確保は鬱陶しいのでstatic．
	//static std::vector<std::vector<std::vector<std::vector<float>>>> feats_x(
	//	(int)(wwy * rate), std::vector<std::vector<std::vector<float>>>(
	//		(int)(wwx * rate), std::vector<std::vector<float>>(
	//			mcy, std::vector<float>(
	//				mcx))));
	//static std::vector<std::vector<std::vector<std::vector<float>>>> feats_y(
	//	(int)(wwy * rate), std::vector<std::vector<std::vector<float>>>(
	//		(int)(wwx * rate), std::vector<std::vector<float>>(
	//			mcy, std::vector<float>(
	//				mcx))));

	//static std::vector<std::vector<std::vector<std::vector<float>>>> feats_z(
	//	(int)(wwy * rate), std::vector<std::vector<std::vector<float>>>(
	//		(int)(wwx * rate), std::vector<std::vector<float>>(
	//			mcy, std::vector<float>(
	//				mcx))));






	// ====================相関値計算====================
	//  Calculation of Correlation Value
#pragma omp parallel for 
	for (int iy = 0; iy < wwy; iy++) {
		for (int ix = 0; ix < wwx; ix++) {
			double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
			for (int py = 0; py < mcy; py++)
				for (int px = 0; px < mcx; px++) {
					double pix = (double)image.at<unsigned char>(iy + py, ix + px);
					//sum_x += cal_corr(image, pca_x, cv::Point2d(ix + 7, iy + 7), 15, py, px);
					//sum_y += cal_corr(image, pca_y, cv::Point2d(ix + 7, iy + 7), 15, py, px);
					//sum_z += cal_corr(image, pca_z, cv::Point2d(ix + 7, iy + 7), 15, py, px);
					sum_x += pix * pca_x[py][px];
					sum_y += pix * pca_y[py][px];
					sum_z += pix * pca_z[py][px];

					//feats_x[iy][ix][py][px] = pix * pca_x[py][px];
					//feats_y[iy][ix][py][px] = pix * pca_y[py][px];
					//feats_z[iy][ix][py][px] = pix * pca_z[py][px];
				}
			corr_r[iy][ix] = sqrt(sum_x * sum_x + sum_y * sum_y);
		}
	}



	// ====================ピーク検出並列化のためのOpenMP実装====================
	// OpenMPで並列化しながら.push_back()はできないので，threadごとに.push_back()し，
	// 最後に全体を結合する．
	int max_threads = omp_get_max_threads();
	std::vector<std::vector<cv::KeyPoint>> kp_all(max_threads);




	// ====================ピーク検出====================
	//  Correlation peak detection
	//std::vector<cv::KeyPoint> kpoints;
#pragma omp parallel for 
	for (int iy = 1; iy < corr_r.size() - 1; iy++)
		for (int ix = 1; ix < corr_r[0].size() - 1; ix++) {
			double ct = corr_r[iy][ix]; if (ct < pca_Threshold) continue;
			int bix = ix - 1, nix = ix + 1;
			if (corr_r[iy][bix] < ct && ct > corr_r[iy][nix]) {
				int biy = iy - 1, niy = iy + 1;
				if (corr_r[biy][ix] < ct && ct > corr_r[niy][ix]) {
					if (corr_r[biy][bix] < ct && ct > corr_r[niy][nix]) {
						if (corr_r[niy][bix] < ct && ct > corr_r[biy][nix]) {
							if (ENABLE_SUBPIXEL_ESTIMATION) {
								//--------------------------------
								std::vector<cv::Point2f> points = {
									{-1, -1}, {0, -1}, {1, -1},
									{-1,  0}, {0,  0}, {1,  0},
									{-1,  1}, {0,  1}, {1,  1}
								};
								std::vector<double> values = {
									corr_r[biy][bix], corr_r[biy][ix], corr_r[biy][nix],
									corr_r[iy][bix], corr_r[iy][ix], corr_r[iy][nix],
									corr_r[niy][bix], corr_r[niy][ix], corr_r[niy][nix]
								};

								cv::Mat A(points.size(), 6, CV_32F);
								cv::Mat b(points.size(), 1, CV_32F);

								for (int i = 0; i < points.size(); ++i) {
									float x = points[i].x;
									float y = points[i].y;
									float z = values[i];
									A.at<float>(i, 0) = x * x;
									A.at<float>(i, 1) = y * y;
									A.at<float>(i, 2) = x * y;
									A.at<float>(i, 3) = x;
									A.at<float>(i, 4) = y;
									A.at<float>(i, 5) = 1;
									b.at<float>(i, 0) = z;
								}
								cv::Mat p;
								cv::solve(A, b, p, cv::DECOMP_SVD);

								float a = p.at<float>(0, 0);
								float b_ = p.at<float>(1, 0);
								float c = p.at<float>(2, 0);
								float d = p.at<float>(3, 0);
								float e = p.at<float>(4, 0);
								float f = p.at<float>(5, 0);

								float x_peak = -d / (2 * a);
								float y_peak = -e / (2 * b_);
								//--------------------------------

								float pkx = (float)(ix + mcx / 2), pky = (float)(iy + mcy / 2);
								float pkx2 = (float)(ix + mcx / 2 + x_peak), pky2 = (float)(iy + mcy / 2 + y_peak);
								float cangle = (float)cal_angle(image, pkx, pky);
								float csize = 10.0;// cal_size(cworg, pkx, pky);

								// intheareaを流用．
								cv::Point2i center = cv::Point2i(image.cols / 2.0, image.rows / 2.0);
								double rr = std::max(image.cols, image.rows) / 2.0 - 10.0;
								double xx = (double)pkx2 - (double)center.x;
								double yy = (double)pky2 - (double)center.y;
								if (DETECT_POINTS_ONLY_WITHIN_CIRCLE) {
									if (sqrt(xx * xx + yy * yy) < rr)
										//座標(float)、サイズ(一旦全て10)、推定角度、相関値
										kp_all[omp_get_thread_num()].push_back(cv::KeyPoint(pkx2, pky2, csize, cangle, (float)ct, 0, 0));
										//keypoint.push_back(cv::KeyPoint(pkx2, pky2, csize, cangle, (float)ct, 0, 0));
								}
								else {
									kp_all[omp_get_thread_num()].push_back(cv::KeyPoint(pkx2, pky2, csize, cangle, (float)ct, 0, 0));
									//keypoint.push_back(cv::KeyPoint(pkx2, pky2, csize, cangle, (float)ct, 0, 0));
								}								
							}
							else {
								float pkx = (float)(ix + mcx / 2), pky = (float)(iy + mcy / 2);
								float cangle = (float)cal_angle(image, pkx, pky);
								float csize = 10.0;// cal_size(cworg, pkx, pky);

								// intheareaを流用．
								cv::Point2i center = cv::Point2i(image.cols / 2.0, image.rows / 2.0);
								double rr = std::max(image.cols, image.rows) / 2.0 - 10.0;
								double xx = (double)pkx - (double)center.x;
								double yy = (double)pky - (double)center.y;
								if (DETECT_POINTS_ONLY_WITHIN_CIRCLE) {
									//if (sqrt(xx * xx + yy * yy) < rr)
									kp_all[omp_get_thread_num()].push_back(cv::KeyPoint(pkx, pky, csize, cangle, (float)ct, 0, 0));
									//keypoint.push_back(cv::KeyPoint(pkx, pky, csize, cangle, (float)ct, 0, 0));  // なぜかAKAZEを動作させるにはclass_idが存在しないといけない．かつデフォルトの-1ではダメだった．
								}
								else {
									kp_all[omp_get_thread_num()].push_back(cv::KeyPoint(pkx, pky, csize, cangle, (float)ct, 0, 0));
									//keypoint.push_back(cv::KeyPoint(pkx, pky, csize, cangle, (float)ct, 0, 0));  // なぜかAKAZEを動作させるにはclass_idが存在しないといけない．かつデフォルトの-1ではダメだった．
								}
								
							}
						}
					}
				}
			}
		}

	
	// スレッド毎に入れいていた特徴点の情報を一つの配列に結合
	for (const auto& buffer : kp_all) {
		keypoint.insert(keypoint.end(), buffer.begin(), buffer.end());
	}


	// ====================特徴量の記述====================
	 switch (fd.ft)
	 {
	 case featureType::fPCA_float:
		 //create_features_miura_multi_circle7(image, keypoint, descriptor, 416);
		 //create_features_miura_around_quadrant(image, keypoint, descriptor, 20);
		 create_features_miura_pixel(image, keypoint, descriptor, 32);
		 break;
	 case featureType::fPCA_16bin:
		 create_features_16bin(image, keypoint, descriptor);
		 break;
	 case featureType::fPCA_uschar:		 
		 break;
	 default:
		 break;
	 }

	//int sz = descriptor.rows;

}





void pca::featuredetection(featureDetectionType fd, source_data& srcd, destination_data& dstd)
{
	//cv::Ptr<cv::xfeatures2d::SURF> descriptor = cv::xfeatures2d::SURF::create();//#include <opencv2/xfeatures2d.hpp>
	//set_pca_parameters(detector);


	pca detector;
	if (!srcd.oImage_dummy) {	
		detector.detectAndCompute(srcd.oImage, cv::Mat(), srcd.oPts, srcd.oFeatures, fd);
		//descriptor->detectAndCompute(srcd.oImage, cv::Mat(), srcd.oPts, srcd.oFeatures);


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
	detector.detectAndCompute(dstd.oImage, cv::Mat(), dstd.oPts, dstd.oFeatures, fd);
	//descriptor->detectAndCompute(dstd.oImage, cv::Mat(), dstd.oPts, dstd.oFeatures);


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
