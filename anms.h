#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Adaptive Non-Maximal Suppression (ANMS) を適用します。
 * 与えられたキーポイントを画像全体に均一に分布するように間引き、
 * 対応する特徴量ディスクリプタも同様にフィルタリングします。
 * @param keypoints [in/out] フィルタリング対象のキーポイント。処理後、間引かれた結果で上書きされます。
 * @param descriptors [in/out] フィルタリング対象の特徴量。処理後、間引かれた結果で上書きされます。
 * @param numBestPoints [in] 保持したいキーポイントの最大数。
 */
void applyAnms(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int numBestPoints);