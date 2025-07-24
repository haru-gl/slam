#include <opencv2/opencv.hpp>
#include "sac_main.h"
#include "tmatrix.h"
#include "functions.h"
#include "featuremap.h"

#include <bitset>




bool sac::matrixestimation(const matchingType mt, const matrixcalType ct, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform)
{
    bool status = true;
    cv::Mat x0, xn1;
    switch (ct) {
    case matrixcalType::cSVD:
        tform = computematrix(mt, selectedtd, selectedcm);
        break;
    case matrixcalType::cSVD_EIGEN:
        tform = computematrix_byEigen(mt, selectedtd, selectedcm);
        break;
    case matrixcalType::cGAUSSNEWTON:
        std::cout << "ガウスニュートン法は実装されていません。\n";
        exit(EXIT_FAILURE);

        //tform = computematrix(mt, selectedtd, selectedcm);
        //status = checkFunc(tform);
        //x0 = cnv_mt2vc(mt, tform);
        //xn1 = gauss_newton(mt, selectedtd, selectedcm, x0);
        //tform = cnv_vc2mt(mt, xn1);
        break;
    case matrixcalType::cGAUSSNEWTON_EIGEN:
        std::cout << "ガウスニュートン法は実装されていません。\n";
        exit(EXIT_FAILURE);
        //tform = computematrix_byEigen(mt, selectedtd, selectedcm);
        //status = checkFunc(tform);
        //x0 = cnv_mt2vc(mt, tform);
        //xn1 = gauss_newton(mt, selectedtd, selectedcm, x0);
        //tform = cnv_vc2mt(mt, xn1);
        break;
    case matrixcalType::cTAUBIN:
        std::cout << "タービン法は実装されていません。\n";
        exit(EXIT_FAILURE);
        //if (mt != matchingType::mPROJECTIVE && mt != matchingType::mPROJECTIVE3) {
        //    std::cout << "MatcingType is not Projective" << std::endl;
        //    tform = cv::Mat();
        //    break;
        //}
        //x0 = compute_x0(selectedtd, selectedcm);
        //xn1 = hyper_renormalization(selectedtd, selectedcm, x0);
        //tform = cnv_vc2mt(mt, xn1);
        break;
    }
    status = checkFunc(tform);
    return status;
}


// basic ransac  `sac::sac_normal`から呼び出される関数
bool sac::a_ransac(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    double minErr = DBL_MAX;
    std::vector<size_t> bestInliersIdx(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    double tsd;
    std::vector<double> exy;
    std::vector<double> bestInliersExy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }

    for (it = 1; it <= maxIteration; it++) {
        indices = randperm(numPts, sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        double err = 0.0;
        size_t iidx = 0;
        exy.clear();
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                exy.push_back(norm);
            }
            else
                norm = maxDistance;
            err += norm;
        }
        //save best fit model & iterrationEvaluation
        if (err < minErr) {
            minErr = err;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            bestInliersExy = exy;
            //iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
            //printf("現在の回数 / 必要と推定される回数：    [%3d / %3d]\n", (int)it, (int)iterNum);
            //if (iterNum <= it) break;
        }
        // 2025.01.20: この評価式は，上記if文の外側に存在するべきなので移動した．
        iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        printf("現在の回数 / 必要と推定される回数：    [%3d / %3d]\n", (int)it, (int)iterNum);
        if (iterNum <= it) break;
    }
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    printf("sac後の有効対応点数：%d\n", (int)bestInliersNum);
    for (size_t i = 0; i < bestInliersNum; i++) {
        if (true) {//bestInliersExy[inliersIdx[i]] < 0.00001
            selectedtd.push_back(td[bestInliersIdx[i]]);
            selectedcm.push_back(cm[bestInliersIdx[i]]);
        }
    }
    printf("sac後の有効対応点数：%d\n", (int)selectedtd.size());

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}



//`sac::positionestimation_normal`から呼び出される関数
bool sac::sac_normal(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, cv::Mat& tform,
                     std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd)
{
    bool status = true;

    if (pe.ndon == false) status = a_ransac(pe.mt, pe.rm, srcd.oMatchedPts, dstd.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else
    {
        printf("ndon == trueでの実行はサポートされていません。");
        exit(EXIT_FAILURE);
    }
    //else status = a_ransac_nd(pe.mt, pe.rm, srcd.oMatchedPts, dstd.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }

    if (IS_INFO) std::cout << tform << std::endl;


    //// 例えば、マッチ0番目のsrcdとdstdで、RAWのAKAZE特徴量がどのように似ているかを表示する
    //int y = 19;
    //for (int x = 0; x < srcd.oFeatures.cols; x++) {
    //    printf("%4d, ", (int)srcd.oFeatures.at<unsigned char>(y, x));
    //}

    //printf("\n\n");

    //y = 0;
    //for (int x = 0; x < dstd.oFeatures.cols; x++) {
    //    printf("%4d, ", (int)dstd.oFeatures.at<unsigned char>(y, x));
    //}

    return status;
}




//#include <omp.h>

// 出力は，入力したoPtsと同じ座標を持つsrc/dstのインデックス番号．
// 
// =====================
// この関数は正常に動作するか検証していない！！！！！
// ひとまず．
// =====================
//template <typename T>
//int srcdstIdx_from_position(cv::Point2f pts, T srcdst)
//{
//    int sv;     // share value
//    bool found = false;
//#pragma omp parallel for
//    for (int i = 0; i < srcdst.oPts.size(); i++) {
//        if ((srcdst.oPts[i].pt.x == pts.x) && (srcdst.oPts[i].pt.y == pts.y)) {
//            sv = i;
//            found = true;
//#pragma omp cancel for if (found) // foundがtrueなら，ループ脱出
//        }
//    }
//    return sv;
//}



bool sac::sac_normal_opencv(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, cv::Mat& tform,
                            std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd)
{
    bool status = true;

    // OpenCVの関数を使ったRANSAC
    cv::Mat mask;
    std::vector<cv::Point2d> srcpts(srcd.oMatchedPts.size()); // この二つは同じ値になるはず
    std::vector<cv::Point2d> dstpts(dstd.oMatchedPts.size()); // この二つは同じ値になるはず


#pragma omp parallel for
    for (int i = 0; i < srcd.oMatchedPts.size(); i++) {
        srcpts[i] = srcd.oMatchedPts[i];
    }
#pragma omp parallel for
    for (int i = 0; i < dstd.oMatchedPts.size(); i++) {
        dstpts[i] = dstd.oMatchedPts[i];
    }
    tform = cv::findHomography(dstpts, srcpts, mask, cv::RANSAC, 5.0); // 入力する対応点は「たがいに対応」するようにする！
    std::cout << tform << std::endl;


    // 有効対応点数の計上
    int candnum = 0;
    for (int row = 0; row < mask.rows; row++) {
        unsigned char* v = mask.ptr<unsigned char>(row);
        debug_log("[%3d / %3d]: %d\n", row, mask.rows, v[0]);
        if (v[0] == 1) {
            candnum++;
            selectedcm.push_back(srcd.oMatchedPts[row]);
            selectedtd.push_back(dstd.oMatchedPts[row]);
        }
    }
    printf("有効対応点（After RANSAC）: %d\n", candnum);



 
    //
    //for (int i = 0; i < candnum; i++) {
    //    selectedcm = ;
    //}

    //int itr = 0;
    //for (int row = 0; row < candnum; row++) {
    //    unsigned char* v = mask.ptr<unsigned char>(row);
    //    if (v[0] == 1) {
    //        srcd.oPts[itr] = srcd.oPts
    //            srcd.oPts[itr].pt = srcd.oMatchedPts[row];
    //        dstd.oPts[itr].pt = dstd.oMatchedPts[row];  // この感じだと，座標情報だけになってしまう．angle, response,etc.もコピーする必要ある．
    //        // ここで，matchedPtsの情報を持っている元のPtsを探す．

    //        // ここには，oFeaturesのコピーが必要．
    //        // この行も！
    //        // oMatchedFeaturesはいらないのか？？？
    //        itr++;
    //    }
    //}

    //srcd.oPts.resize(candnum);  // Vectorなので，要素数が調整される
    //dstd.oPts.resize(candnum);
    //srcd.oFeatures.resize(candnum); // cv::Matなので列数は現状ママ，行数が調整される
    //dstd.oFeatures.resize(candnum);




    




    status = false;////////////////###############################################################################
    return status;
}



void sac::positionestimation_normal(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt);	 if (minGoodPairs == 0) { rs.status = 4; return; }

    if (rs.goodPairsNum < minGoodPairs) {
        rs.status = 3;//Insufficient inlayer
        error_log("\nInsufficien inlayer: 3. Exitting...\n");
        exit(EXIT_FAILURE);
    }
    else {
        cv::Mat tform;
        bool stat = true;
        std::vector<cv::Point2d> selectedcm, selectedtd;
        if (pe.use_OpenCV_findHomography) {
            // OpenCV内臓のRANSAC
            stat = sac_normal_opencv(pe, fmpd, srcd, dstd, tform, selectedcm, selectedtd);
            if (stat == false) {
                fmpd.curr_tform = tform.clone();
                rs.status = 0;
            }
            else {
                rs.status = 2;//Matching not completed
                std::cout << "マッチング未了。異常終了。\n";
                exit(EXIT_FAILURE);
            }
        }
        else {
            // 自作のRANSAC
            stat = sac_normal(pe, fmpd, srcd, dstd, tform, selectedcm, selectedtd);
            if (stat == false) {
                //rs.scale = cal_scale(tform, td.szcenter);
                //cv::Point2d estimated;
                //estimated = transform2d(cv::Point2d(td.szcenter, td.szcenter), tform);
                //rs.estimatedCenter2dx = estimated.x + cm.lux;
                //rs.estimatedCenter2dy = estimated.y + cm.luy;
                //rs.estimatedHeight = cm.averageDistance * rs.scale + cm.averageHeight - 600.0;//The meaning of "600" is unknown


                // 保存しておく
                fmpd.curr_tform = tform;

                rs.status = 0;
            }
            else {
                rs.status = 2;//Matching not completed
                std::cout << "マッチング未了。異常終了。\n";
                exit(EXIT_FAILURE);
            }
        }
        if (IS_IMSHOW || IS_IMSHOW_SAC) draw_RANSAC(srcd, dstd, selectedcm, selectedtd);
        Correspond_selectedcm_selectedtd(srcd, dstd, selectedcm, selectedtd);   // この中で、綺麗な対応点Matcing画像も生成・表示される。
    }
}



size_t sac::computeLoopNumbers(size_t numPts, size_t inlierNum, size_t sampleSize)
{
    double eps = 1.0e-15;
    double inlierProbability = 1.0;//initial value=1
    size_t nn;
    double factor = (double)inlierNum / numPts;
    for (size_t i = 0; i < sampleSize; i++)
        inlierProbability *= factor;

    if (inlierProbability < eps) nn = INT_MAX;
    else {
        double conf = confidence / 100.0;
        double numerator = log10(1 - conf);
        double denominator = log10(1 - inlierProbability);
        nn = (size_t)(numerator / denominator);
    }
    return nn;
}



void sac::draw_RANSAC(source_data srcd, destination_data dstd, std::vector<cv::Point2d> selectedcm, std::vector<cv::Point2d> selectedtd)
{
    // 左にtd、右にcmが表示されるようにする
    std::vector<std::vector<cv::DMatch>> empty_matches;
    std::vector<cv::KeyPoint> empty_pts;

    cv::Mat match_img;
    cv::drawMatches(dstd.oImage, empty_pts, srcd.oImage, empty_pts, empty_matches, match_img);

    //std::cout << selectedcm.size() << "\n";

    for (int itr = 0; itr < selectedcm.size(); itr++) {
        cv::line(match_img, cv::Point(selectedtd[itr].x, selectedtd[itr].y), cv::Point(dstd.oImage.rows + selectedcm[itr].x, selectedcm[itr].y), cv::Scalar(20, 250, 30));// , 2);// , cv::LINE_4);
    }

    cv::imwrite("./match_img.jpg", match_img);
    cv::resize(match_img, match_img, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
    cv::imshow("[ransac後]match_img", match_img);
    cv::waitKey(1);
}

void sac::draw_RANSAC_elegant(source_data srcd, destination_data dstd, std::vector<cv::Point2d> selectedcm, std::vector<cv::Point2d> selectedtd)
{

}