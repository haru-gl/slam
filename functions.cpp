#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

#include "functions.h"
#include "classes.h"



void drawKeyPoints(source_data& srcd, destination_data& dstd)
{
    if (IS_IMSHOW) {
        cv::Mat src_key, dst_key;
        cv::drawKeypoints(srcd.oImage, srcd.oPts, src_key, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(dstd.oImage, dstd.oPts, dst_key, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::resize(src_key, src_key, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
        cv::resize(dst_key, dst_key, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
        cv::imshow("[drawKeyPoints] src_key", src_key);
        cv::imshow("[drawKeyPoints] dst_key", dst_key);

        cv::waitKey(1);
    }
}

// 0で青、max値で赤の値を返す
cv::Scalar getColorLevel(int brt, int max) {
    // 輝度を0.0〜1.0に変換
    double xbrt = (double)brt / (double)max;

    int b, g, r;

    //カラー変換
    if (xbrt >= 0 && xbrt <= 0.25) {
        b = 255;      //B
        g = (int)((double)255 * sin(xbrt * 2 * M_PI));        //G
        r = 0;        //R
    }
    else if (xbrt > 0.25 && xbrt <= 0.5) {
        b = (int)((double)255 * sin(xbrt * 2 * M_PI));  //B
        g = 255;      //G
        r = 0;        //R
    }
    else if (xbrt > 0.5 && xbrt <= 0.75) {
        b = 0;        //B
        g = 255;      //G
        r = (int)(-(double)255 * sin(xbrt * 2 * M_PI));  //R
    }
    else {
        b = 0;        //B
        g = (int)(-(double)255 * sin(xbrt * 2 * M_PI));  //G
        r = 255;      //R
    }

    return cv::Scalar(b,g,r);
}


// 特徴点の情報（srcd, dstd）に[追記]のみする関数
void appendNewFeaturesInfo(std::vector<cv::KeyPoint>  newoPts, cv::Mat  newoFeatures,
                           featuremap_data& fmpd, bool forceOverRide, featureDetectionType fd)
{
    // 大きさの比較を行う。サイズが異なる場合に追加すると重大なエラーにつながるので排除する。
    //if (fmpd.oPts.size() == fmpd.oFeatures.rows) {}
    //else {
    //    std::cout << "[fmpd]既存のキーポイントと、既存の特徴量の大きさが等しくありません。強制終了。\n";
    //    exit(EXIT_FAILURE);
    //}
    //if (newoPts.size() == newoFeatures.rows) {}
    //else {
    //    std::cout << "新しく追加するキーポイントと、新しく追加する特徴量の大きさが等しくありません。強制終了。\n";
    //    exit(EXIT_FAILURE);
    //}
    warn_log("[functions.cpp] デバッグ：サイズの比較処理は保留...\n");



    // この部分に、「もし座標値がマイナスになりうるのなら、それを補正（全体シフト）の最大値を算出する」が必要？
    // 最大値を探索
    int x_max = fmpd.mps_x; //intでは、情報が失われるぞ？。
    int y_max = fmpd.mps_y;
    for (int i = 0; i < newoPts.size(); i++) {
        if (x_max < newoPts[i].pt.x) x_max = newoPts[i].pt.x;
        if (y_max < newoPts[i].pt.y) y_max = newoPts[i].pt.y;
    }

    // 変更があったか確認
    if (fmpd.mps_x < x_max) {
        debug_log("地図サイズ（x軸方向）が拡張されました：%d\n", x_max);
        fmpd.mps_x = x_max;
    }
    if (fmpd.mps_y < y_max) {
        debug_log("地図サイズ(y軸方向）が拡張されました：%d\n", y_max);
        fmpd.mps_y = y_max;
    }






    // oFeaturesの移植の下準備
    // サイズの確認、拡張、.atでのアクセスによる書き込み、の順かな？
    int cur_fts_s = (int)fmpd.oFeatures.rows;       // current_features_size  例：現在118ポイントあり、
    int new_frs_s = (int)newoPts.size();            //     new_features_size   　 新たに36点を追加する、といった状況。
    const int init_num = cur_fts_s;
    const int idx_diff = new_frs_s;                 // 拡張前と拡張後でidx番号にどれだけ差があるか
    const int final_num = cur_fts_s + new_frs_s;

    debug_log("%d,  %d\n", cur_fts_s, new_frs_s);

    //if (fmpd.oPts.capacity() < new_frs_s + cur_fts_s) fmpd.oPts.reserve(new_frs_s + cur_fts_s+1); // 次の行でVectorを拡張するとき，capacity以上の値が設定されると異常終了するため
    fmpd.oPts.resize(new_frs_s + cur_fts_s);
    fmpd.oFeatures.resize(new_frs_s + cur_fts_s);
    debug_log("拡張後の行数（サイズ）：%d\n", new_frs_s + cur_fts_s);



    if (forceOverRide) {
        fmpd.oPts = newoPts;
        fmpd.oFeatures = newoFeatures.clone();
    }
    else {
        // 特徴点のコピー
//#pragma omp parallel for
        for (int i = 0; i < newoPts.size(); i++) {
            fmpd.oPts[init_num + i] = newoPts[i];
        }

        // 特徴量のコピー
        for (int i = 0; i < newoPts.size(); i++) {
//#pragma omp parallel for
            for (int c = 0; c < fmpd.dimFeatures; c++) {
                copy2features(fd, fmpd.oFeatures, newoFeatures, cur_fts_s, c, i, c);
            }
            cur_fts_s++;
        }
    }





    //for (int i = 0; i < newoPts.size(); i++) {
    //    // oPtsの移植
    //    fmpd.oPts.push_back(newoPts[i]);
    //    
    //    if (forceOverRide) {
    //        fmpd.oFeatures = newoFeatures.clone();
    //    }
    //    else {
    //        // oFeaturesの移植
    //        for (int c = 0; c < fmpd.dimFeatures; c++) {
    //            copy2features(fd, fmpd.oFeatures, newoFeatures, cur_fts_s, c, i, c);
    //            
    //            
    //            //fmpd.oFeatures.data[fmpd.oFeatures.cols * cur_fts_s + c] = newoFeatures.data[newoFeatures.cols * i + c];
    //            //std::cout << (double)fmpd.oFeatures.data[fmpd.oFeatures.cols * cur_fts_s + c] << ", ";
    //            
    //            /*std::cout << "[from]" << (float)newoFeatures.data[newoFeatures.cols * i + c] << std::endl;*/
    //            //std::cout << "[  to]" << (float)fmpd.oFeatures.data[fmpd.oFeatures.cols * cur_fts_s + c] << std::endl;
    //        }
    //        cur_fts_s++;
    //    }
    //}
    warn_log("[functions.cpp] featuresの追加は保留...\n");
}
    

// 二つの十進数の値同士で、二進数時のハミング距離を返す
int calcHumming(unsigned char x, unsigned char y)
{
    int humm = 0;
    unsigned char res = x ^ y;

    /*std::cout << (int)res << std::endl;*/

    for (int it = 0; it < 8; it++) {
        humm += (res >> it) & 1;
    }
    return humm;
}



// 小数点以下のある桁数で丸める関数（decimal place number）
double roundd(double num, int dpn)
{
    int d = 1;
    for (int i = 0; i < dpn; i++) {
        d *= 10;
    }
    return (double)std::round(num * d) / (double)d;
}



void copy2features(featureDetectionType fd,
    cv::Mat& dst, cv::Mat& src, int dst_row, int dst_col, int src_row, int src_col)
{
    switch (fd.ft)
    {
    case featureType::fAKAZE:
    case featureType::fORB:
    case featureType::fBRISK:
    case featureType::fPCA_uschar:
        dst.at<unsigned char>(dst_row, dst_col) = src.at<unsigned char>(src_row, src_col);
        break;
    case featureType::fSIFT:
    case featureType::fSURF:
    case featureType::fPCA_float:
        dst.at<float>(dst_row, dst_col) = src.at<float>(src_row, src_col);
        break;
    case featureType::fPCA_16bin:
        dst.at<uchar>(dst_row, dst_col) = src.at<uchar>(src_row, src_col);
        break;
    default:
        error_log("[copy2features @ functions.cpp] featureTypeが予期せぬ入力です．\n");
        exit(EXIT_FAILURE);
        break;
    }
}
