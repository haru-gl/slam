#include "anms.h"
#include <algorithm>
#include <vector>
#include <numeric> // std::iota のために必要

void applyAnms(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int numBestPoints) {
    // 特徴点数が目標数以下、または特徴量と数が合わない場合は何もしない
    if (keypoints.size() <= numBestPoints || keypoints.size() != descriptors.rows) {
        if (keypoints.size() != descriptors.rows) {
            printf("[ANMS Warning] Keypoints size (%zu) and descriptors rows (%d) do not match. Skipping.\n", keypoints.size(), descriptors.rows);
        }
        return;
    }

    // 1. 各キーポイントの抑制半径を計算
    std::vector<float> suppressionRadii(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        float minRadius = std::numeric_limits<float>::max();
        for (size_t j = 0; j < keypoints.size(); ++j) {
            // 応答値がより大きい(または等しく、かつインデックスが小さい)キーポイントjに対してのみ距離を計算
            if (keypoints[j].response > keypoints[i].response) {
                float dist = cv::norm(keypoints[i].pt - keypoints[j].pt);
                if (dist < minRadius) {
                    minRadius = dist;
                }
            }
        }
        suppressionRadii[i] = minRadius;
    }

    // 2. キーポイントを抑制半径の降順でソートするためのインデックスを作成
    std::vector<int> indices(keypoints.size());
    std::iota(indices.begin(), indices.end(), 0); // 0, 1, 2, ...

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return suppressionRadii[a] > suppressionRadii[b];
        });

    // 3. 上位N個のキーポイントと特徴量を新しいコンテナに格納
    std::vector<cv::KeyPoint> bestKeypoints;
    bestKeypoints.reserve(numBestPoints);

    // 特徴量のデータ型と次元数を維持して新しいMatを準備
    cv::Mat bestDescriptors = cv::Mat::zeros(numBestPoints, descriptors.cols, descriptors.type());

    for (int i = 0; i < numBestPoints; ++i) {
        int original_index = indices[i];
        bestKeypoints.push_back(keypoints[original_index]);
        // 元のインデックスを使って、対応する特徴量の行をコピー
        descriptors.row(original_index).copyTo(bestDescriptors.row(i));
    }

    // 4. 元のコンテナをフィルタリング後の結果で上書き
    keypoints = bestKeypoints;
    descriptors = bestDescriptors;
}