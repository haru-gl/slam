#pragma once
#include <opencv2/opencv.hpp>
#include "enclasses.h"
#include "classes.h"
#include <numeric>

class pca {
private:
	int pca_Threshold;
	bool pcas_load(const std::string fnm, std::vector<std::vector<std::vector<double>>>& uvec, int numvec);
	std::vector<std::vector<std::vector<double>>> get_uvec_ad0(void);
	std::vector<std::vector<std::vector<double>>> get_luvecs2(void);

public:
	std::vector<std::vector<std::vector<double>>> uvec_ad0;
	std::vector<std::vector<std::vector<double>>> luvecs2;
	std::vector<std::vector<double>> pca_x;
	std::vector<std::vector<double>> pca_y;
	std::vector<std::vector<double>> pca_z; // 三浦追加
	std::vector<std::vector<double>> pca_degree_x;
	std::vector<std::vector<double>> pca_degree_y;
	pca(void)
	{
		// std::chrono::system_clock::time_point st, ed;
		// st = std::chrono::system_clock::now();

		pca_Threshold = INIT_PCA_THRESHOLD;
		uvec_ad0 = get_uvec_ad0();
		luvecs2 = get_luvecs2();

		// ed = std::chrono::system_clock::now();

		// long long durt = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
		// std::cout  << "PCAのLoadにかかった時間：" << durt << " [ms]\n";

		pca_x = uvec_ad0[0];
		pca_y = uvec_ad0[1];
		pca_z = uvec_ad0[2];
		pca_degree_x = luvecs2[0];
		pca_degree_y = luvecs2[1];
	}

	cv::Mat cnv_pca2img(std::vector<std::vector<double>>& pcav);
	double cal_angle(const cv::Mat& cworg, float pkx, float pky);
	void detectAndCompute(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor, featureDetectionType fd);
	void featuredetection(featureDetectionType fd, source_data& srcd, destination_data& dstd);

	double cal_corr(const cv::Mat& image, const std::vector<std::vector<double>>& pca_xyz, cv::Point2d center, int size, int row, int col);


	void create_features_miura_multi_circle7(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim);
	void create_features_miura_around_quadrant(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim);
	void create_features_miura_pixel(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim);
	void create_features_miura_pixel_angle(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim);
	void create_features_16bin(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures);




	// StandardDeviation (母集団を与えられた値全てと考えた標準偏差)  #include <numeric>
	float calc_stdev_p(const std::vector<float>& input) {
		// 入力が空の場合は0を返す
		if (input.empty()) {
			return 0.0;
		}

		// 平均を計算
		double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();

		// 偏差の平方を計算して合計
		double variance = std::accumulate(input.begin(), input.end(), 0.0,
			[mean](double sum, double value) {
				return sum + (value - mean) * (value - mean);
			}) / input.size();

		// 標準偏差を計算
		return std::sqrt(variance);
	}

	// maxとminの間をn分割したヒストグラムを作成する．enumsの各要素が，その範囲に何個ずつあるかを数える関数．
	template <typename T>
	void create_histgram(const std::vector<float>& enums, std::vector<T>& histgram, int n, float max, float min)
	{
		// featの要素数をnに設定し、0で初期化
		histgram.assign(n, 0);

		// 分割幅を計算
		float bin_width = (max - min) / n;

		// enumsの各要素について
		for (float val : enums) {
			// 値が範囲外の場合は無視
			if (val < min || val > max) {
				continue;
			}

			// 適切なビンのインデックスを計算
			int bin_index = std::min(static_cast<int>((val - min) / bin_width), n - 1);

			// 対応するビンのカウントを増加
			histgram[bin_index] += 1.0;

			//for (auto v : histgram) printf("%3d ", v);
		}
		//printf("\n");
	}

	double find_max(const std::vector<float>& input) {
		if (input.empty()) {
			throw std::invalid_argument("Input vector is empty.");
		}
		return *std::max_element(input.begin(), input.end());
	}
	double find_min(const std::vector<float>& input) {
		if (input.empty()) {
			throw std::invalid_argument("Input vector is empty.");
		}
		return *std::min_element(input.begin(), input.end());
	}

};
