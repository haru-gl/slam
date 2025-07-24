#include "pca.h"
#include <algorithm>
#include <cmath>

void pca::create_features_16bin(const cv::Mat& image,
    const std::vector<cv::KeyPoint>& kpts,
    cv::Mat& desc)
{
    constexpr int DIM = 32;
    desc = cv::Mat::zeros((int)kpts.size(), DIM, CV_8U);



    // ==== corr[61] を求める処理を実装 ====

    for (int i = 0; i < kpts.size(); ++i) {
        float corr[61] = { 0.f };  // ← 実演用。必ず計算で埋める
        // パッチ中心座標
        int cx = int(kpts[i].pt.x);
        int cy = int(kpts[i].pt.y);

        // 例: 7×7円形マスクを 3 本の基底ごとに 61 セルへ集約
        int idx = 0;
        for (int row = -3; row <= 3; ++row)
            for (int col = -3; col <= 3; ++col)
            {
                if (row * row + col * col > 9) continue;          // 半径 3 の円内
                float pix = image.at<uchar>(cy + row, cx + col);
                float v = pix * (pca_x[row + 7][col + 7] +
                    pca_y[row + 7][col + 7] +
                    pca_z[row + 7][col + 7]);
                corr[idx++] = v;                            // idx は 0..60
            }

        std::partial_sort(corr, corr + 16, corr + 61,
            [](float a, float b) { return std::fabs(a) > std::fabs(b); });

        for (int k = 0; k < DIM; ++k) {
            float v = corr[k];
            //float vn = std::tanh(v / 128.f);
            float vn = v / 512.f; //緩め
            int   q = int((vn + 1.f) * 127.5f);
            desc.at<uchar>(i, k) = static_cast<uchar>(q);
        }
    }
}