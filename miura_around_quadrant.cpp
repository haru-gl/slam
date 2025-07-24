#include <opencv2/opencv.hpp>
#include "pca.h"


// 距離を計算するヘルパー関数
double calculateDistance(const cv::Point2f& pt1, const cv::Point2f& pt2) {
    return std::sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
}

// 特徴点を近い順に返す関数
std::vector<int> findClosestKeypoints(const std::vector<cv::KeyPoint>& keypoints, const cv::Point2f& pt, int n) {
    // 各特徴点までの距離とインデックスのペアを作成
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < keypoints.size(); ++i) {
        double distance = calculateDistance(keypoints[i].pt, pt);
        distances.emplace_back(distance, i);
    }

    // 距離に基づいてソート
    std::sort(distances.begin(), distances.end());

    // 近い順にn個のインデックスを取り出す
    std::vector<int> closestIndices;
    for (int i = 0; i < std::min(n, static_cast<int>(distances.size())); ++i) {
        closestIndices.push_back(distances[i].second);
    }

    return closestIndices;
}



// 渡されたキーポイントが，注目している特徴点に対して第n象限にいることを返却する関数
int judge_quadrant(const cv::KeyPoint& focus_kp, const cv::KeyPoint& description_ni_tsukau_kp)
{
    cv::Point2d center = focus_kp.pt;
    cv::Point2d outer  = description_ni_tsukau_kp.pt;


}

void pca::create_features_miura_around_quadrant(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim)
{
    // 特徴量記述
    oFeatures = cv::Mat::zeros(kpoints.size(), dim, CV_32F);
    int ptr = 0;

    // それぞれの特徴点について

    std::chrono::system_clock::time_point st, ed;
    st = std::chrono::system_clock::now();




    for (int i = 0; i < kpoints.size(); i++) {
        // 近傍10点を探索（10では...）--> 全体にした．
        std::vector<int> nid = findClosestKeypoints(kpoints, kpoints[i].pt, kpoints.size() - 1);


        // 第n象限であるかを計算し，各象限について，【th個の点】を発見するまで繰り返す．
        int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
        int th = 5;
        int ptr = 0;
        do
        {
            // 近い順に近傍点の箇所をあたる．
            kpoints[nid[ptr]];
            ptr++;
            break;
        } while (q1 < th && q2 < th && q3 < th && q4 < th);
        std::cout << "繰り返し率：" << (double)ptr/kpoints.size() * 100 << " %." << std::endl;



        // =====近傍の点情報から特徴量を記述する．=====
        // 1. 角度の絶対値の差の割合
        // ★4C2みたいな，選択はどのくらい？くみあわせ　．
        // 2. 
        oFeatures.at<float>(i, ptr++) = kpoints[nid[0]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[1]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[2]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[3]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[4]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[5]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[6]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[7]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[8]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[9]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[10]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[11]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[12]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[13]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[14]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[15]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[16]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[17]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[18]].response;
        oFeatures.at<float>(i, ptr++) = kpoints[nid[19]].response;
        ptr = 0;
    }

    ed = std::chrono::system_clock::now();
    long long durt = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();

    std::cout << durt << " [ms]\n";



}