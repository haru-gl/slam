#include "pca.h"

void pca::create_features_miura_pixel(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim)
{
    // 特徴量記述
    oFeatures = cv::Mat::zeros(kpoints.size(), dim, CV_32F);


    std::vector<std::vector<char>> pmsk = {
        {0,0,0,0,0,1,1,1,1,1,0,0,0,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,0,0,0,1,1,1,1,1,0,0,0,0,0}
    };
    
#pragma omp parallel for
	for (int i = 0; i < kpoints.size(); i++) {        
		cv::Point2i pos_idx = cv::Point2i(std::round(kpoints[i].pt.x + 0), std::round(kpoints[i].pt.y + 0));
        
		// 近くの画素の輝度値を取得
		std::vector<unsigned char> enums;
        int sum = 0;
        unsigned char v;
        int cnt = 0;
        int range = 15, st = -range / 2, ed = range / 2, circle_diameter = range/2;

		for (int r = st; r <= ed; r++) {
			for (int c = st; c <= ed; c++) {
                if (std::sqrt(c*c + r*r) <= circle_diameter) {//if (pmsk[r + range / 2][c + range / 2] == 1) {
                    v = image.at<unsigned char>(pos_idx.y + r, pos_idx.x + c);
                    enums.push_back(v);
                    sum += (int)v;
                    cnt++;
                }
			}
		}


        // 正規化
        std::vector<float> enums_float; for (auto v : enums) enums_float.push_back((float)v);
        std::vector<float> nd_enums;
        

        //// ========== 標準偏差 ==========
        //for (auto v : enums) nd_enums.push_back(((float)v - ((float)sum/cnt)) / calc_stdev_p(enums_float));
        //// ヒストグラム化
        //std::vector<float> feat_stdev;
        //create_histgram(nd_enums, feat_stdev, 32, find_max(nd_enums), find_min(nd_enums));
        //// ベクトルとして格納
        //int ptr = 0;
        //for (auto v : feat_stdev) oFeatures.at<float>(i, ptr++) = v;


        // ========== min-max正規化 ==========
        nd_enums.clear();
        for (auto v : enums) nd_enums.push_back(((float)v - (float)find_min(enums_float)) / ((float)find_max(enums_float) - (float)find_min(enums_float))*255.0);
		
        //

        // ヒストグラム化
        std::vector<float> feat_minmax;
        create_histgram<float>(nd_enums, feat_minmax, 32, 255, 0);
        int ptr = 0;
        
        //int mx = 0;
        //for (auto v : feat_minmax) {
        //    if (mx < v) mx = (int)v;
        //}
        //std::cout << "max:" << mx << "\n";


        for (auto v : feat_minmax) oFeatures.at<float>(i, ptr++) = v;
        //oFeatures.at<unsigned char>(i, 0) = 0;
        //oFeatures.at<unsigned char>(i, oFeatures.cols-1) = 0;
	}
}
