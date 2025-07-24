#include "pca.h"
#include "masks.h"
#include <cmath>
#include <numeric>
#include <stdexcept> // 例外処理用
#include <algorithm> // std::max_element用



void normalize(std::vector<std::vector<float>>& data, float min, float max) {
    // Initialize min and max
    float minv = std::numeric_limits<float>::max();
    float maxv = std::numeric_limits<float>::lowest();


    // 最小・最大値探索
    for (const auto& row : data) {
        for (const auto& value : row) {
            if (value < minv) minv = value;
            if (value > maxv) maxv = value;
        }
    }

    //// 正規化（下限値は無視）
    //for (auto& row : data) {
    //    for (auto& value : row) {
    //        value = (value - minv) / (maxv - minv) * max;
    //    }
    //}

     // 正規化係数を計算
    float scale = (max - min) / (maxv - minv);

    // 各要素を -1 から 1 の範囲に正規化
    for (auto& row : data) {
        for (auto& val : row) {
            val = (val - minv) * scale + min;
        }
    }
}



// maxとminの間をn分割したヒストグラムを作成する．enumsの各要素が，その範囲に何個ずつあるかを数える関数．
void create_histgram(const std::vector<float>& enums, std::vector<float>& histgram, int n, float max, float min)
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
        histgram[bin_index]+= 1.0;
    }
}



void pca::create_features_miura_multi_circle7(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim)
{
    // 特徴量記述
    oFeatures = cv::Mat::zeros(kpoints.size(), dim, CV_32F);

    // マスクを作成
    masks mk_data;
    std::vector<std::vector<std::vector<unsigned char>>> mk;

    mk.push_back(mk_data.outer0);
    mk.push_back(mk_data.outer1);
    mk.push_back(mk_data.outer2);
    mk.push_back(mk_data.outer3);
    mk.push_back(mk_data.outer4);
    mk.push_back(mk_data.outer5);
    mk.push_back(mk_data.outer6);
    mk.push_back(mk_data.vec2OR(mk_data.outer1, mk_data.outer2));   // 複数の円形マスクのOR和を取ることも可能
    mk.push_back(mk_data.vec2OR(mk_data.outer2, mk_data.outer3));
    mk.push_back(mk_data.vec2OR(mk_data.outer3, mk_data.outer4));
    mk.push_back(mk_data.vec2OR(mk_data.outer3, mk_data.outer4));
    mk.push_back(mk_data.vec2OR(mk_data.outer4, mk_data.outer5));
    mk.push_back(mk_data.vec2OR(mk_data.outer5, mk_data.outer6));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer4, mk_data.outer5), mk_data.outer6));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer3, mk_data.outer4), mk_data.outer5));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer2, mk_data.outer3), mk_data.outer4));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer1, mk_data.outer2), mk_data.outer3));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer0, mk_data.outer1), mk_data.outer2));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer0, mk_data.outer1), mk_data.outer2), mk_data.outer3));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer1, mk_data.outer2), mk_data.outer3), mk_data.outer4));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer2, mk_data.outer3), mk_data.outer4), mk_data.outer5));
    mk.push_back(mk_data.vec2OR(mk_data.vec2OR(mk_data.vec2OR(mk_data.outer3, mk_data.outer4), mk_data.outer5), mk_data.outer6));
    

    // マスクを使用した，回転不変特徴量の作成
// #pragma omp parallel for
    for (int i = 0; i < kpoints.size(); i++) {
        cv::Point2i pos_idx = cv::Point2i(kpoints[i].pt.x + 0, kpoints[i].pt.y + 0);//---------------------------------7であってる？
        // x, y方向の相関値の合計をとる
        std::vector<std::vector<float>> summed(15, std::vector<float>(15, 0.0));
        for (int row = 0; row < summed.size(); row++)
            for (int col = 0; col < summed[0].size(); col++)
                // 二乗ルートの方が圧倒的に精度がよかった
                //summed[row][col] = feats_x[pos_idx.y][pos_idx.x][row][col] + feats_y[pos_idx.y][pos_idx.x][row][col];
        //        summed[row][col] = sqrt(std::pow(feats_x[pos_idx.y][pos_idx.x][row][col], 2) + std::pow(feats_y[pos_idx.y][pos_idx.x][row][col], 2));
                summed[row][col] = sqrt(std::pow(cal_corr(image, pca_x, pos_idx, 15, row, col), 2) + 
                                        std::pow(cal_corr(image, pca_y, pos_idx, 15, row, col), 2) +
                                        std::pow(cal_corr(image, pca_z, pos_idx, 15, row, col), 2)) ;
        // 正規化など
        // 正規化しない方が精度がよかった
        //len1(summed); sum0(summed);   // これではない．
        //normalize(summed, 0.0, 1.0);

        // ========== 各マスクで1の部分を探し出し，その合計値を特徴量ベクトルに書き込む ==========
        int ptr = 0;
        

        std::vector<float> enums;
        std::vector<float> nd_enums;
        for (int is = 0; is < mk.size(); is++) {
            
            
            double summation = 0.0;
            int cnt = 0;
            float hensuu=0.0;

            enums.clear();
            nd_enums.clear();

            // 合計や個数を算出
            for (int row = 0; row < mk[is].size(); row++)
                for (int col = 0; col < mk[is][row].size(); col++)
                    if (mk[is][row][col] == 1) {                        
                        summation += summed[row][col];   // 各マスクで1の部分を探し出し，
                        enums.push_back(summed[row][col]);
                        cnt++;
                    }


            // 合計や個数を算出
            
            for (int row = 0; row < mk[is].size(); row++)
                for (int col = 0; col < mk[is][row].size(); col++)
                    if (mk[is][row][col] == 1) {
                        // hensuu += (summed[row][col] - (summation/cnt)) / calc_stdev_p(nd_enums); 
                        // hensuu += (summed[row][col] - (float)find_min(enums)) / ((float)find_max(enums) - (float)find_min(enums));
                        
                        // 標準偏差で正規化
                        nd_enums.push_back((summed[row][col] - (summation/cnt)) / calc_stdev_p(enums));

                        // min-maxで正規化
                        //nd_enums.push_back((summed[row][col] - (float)find_min(enums)) / ((float)find_max(enums) - (float)find_min(enums))*255.0);

                        
                    }

            
            
            std::vector<float> feat;
            //create_histgram<float>(nd_enums, feat, 16, 255.0, 0.0);
            create_histgram<float>(nd_enums, feat, 8, find_max(nd_enums), find_min(nd_enums));
            for (auto v : feat) oFeatures.at<float>(i, ptr++) = v;

             // oFeatures.at<float>(i, ptr++) = (float)(summation / (float)cnt);           // その合計値を特徴量ベクトルに書き込む（平均して）
            // for (auto v : nd_enums) printf("%3.1f ", v); printf("\n -->%f\n", calc_stdev_p(nd_enums));
             //oFeatures.at<float>(i, ptr++) = ((summation - (summation/cnt)) / calc_stdev_p(nd_enums));
            // oFeatures.at<float>(i, ptr++) = (summation / cnt)-(calc_stdev_p(nd_enums));
            // oFeatures.at<float>(i, ptr++) = hensuu;

           
            

            //printf("[%3d/%3d]: %.10f\n", is, (int)mk.size(), hensuu);
            //if (is == mk.size()-1) exit(0);
        }

        //std::vector<float> nd_enums; 
        //for (auto v : enums) nd_enums.push_back((v - (float)find_min(enums)) / ((float)find_max(enums) - (float)find_min(enums))*255.0);

        //std::vector<float> feat;
        //create_histgram<float>(nd_enums, feat, 32, 255.0, 0.0);
        //for (auto v : feat) oFeatures.at<float>(i, ptr++) = v;

        //oFeatures.at<float>(i, 0) = 0.0;
        
    }
}