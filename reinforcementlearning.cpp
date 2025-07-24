#include "tmatrix.h"

#include <opencv2/opencv.hpp>
#include "functions.h"
#include "featuremap.h"
#include "sac_main.h"
#include "reinforcementlearning.h"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>
#include <bitset>

void disprawprop(const std::vector<double>& prop);
void dispexpprop(const std::vector<double>& prop);
void dispsqrtprop(const std::vector<double>& prop);
std::vector<size_t> softmax_y(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_y1(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_y2(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_y3(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_select(size_t numPts, const std::vector<double>& prop, size_t& sampleNum);
std::vector<size_t> softmax_org(size_t numPts, const std::vector<double>& prop, size_t& sampleNum, size_t cnt);
std::vector<size_t> softmax_n(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_n1(size_t numPts, const std::vector<double>& prop, size_t sampleSize, int& status);
std::vector<size_t> softmax_dc(size_t numPts, const std::vector<double>& disvalue, size_t sampleSize);

bool reinforcementlearning::matrixestimation(const matchingType mt, const matrixcalType ct, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform)
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


void reinforcementlearning::positionestimation_grfl(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rs)
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
            error_log("[reinforcementlearning.cpp] 現状では，OpenCV内蔵のRANSACを使用した強化学習は実装されていません．\n");
            exit(EXIT_FAILURE);
        }
        else {
            // 強化学習RANSAC（大森）
            stat = sac_grfl(pe, fmpd, srcd, dstd, tform, selectedcm, selectedtd);
            if (stat == false) {                
                fmpd.curr_tform = tform.clone();    // 保存
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


bool reinforcementlearning::sac_grfl(const posestType& pe, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, cv::Mat& tform,
                                     std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd)
{
    printf("★★★★★強化学習TD(0)法★★★★★\n");
    bool status = true;

    if (pe.ndon == false) status = a_ransac_grfl(pe.mt, pe.rm, srcd.oMatchedPts, dstd.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    //else status = a_ransac_grfl(pe.mt, pe.rm, srcd.oMatchedPts, dstd.oMatchedPts, selectedcm, selectedtd, maxDistance, true); ndon == trueでの実行はサポートされていません。
    else
    {
        printf("a");
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


double fuzzyE(double sdv, double x)
{
    return (sdv - x) / sdv;
}

double gaussE(double sdv, double x)
{
    return exp(-x * x / (2.0 * sdv * sdv));// / sqrt(2.0 * 3.14159265358979 * sdv * sdv);
}

double gaussm0E(double sdv, double x)
{
    return exp(-x * x / (0.5 * sdv * sdv));
}

double gaussm1E(double sdv, double x)
{
    return exp(-x / (0.5 * sdv));
}
double exE(double sdv, double x, double p)
{
    return 1.0 - pow(x / sdv, p);
}

double circle(double sdv, double x)
{
    return sdv - sqrt(sdv * sdv - (x - sdv) * (x - sdv));
}

// ransac with reinforcemant learning
bool reinforcementlearning::a_ransac_grfl(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;

    std::vector<size_t> indices = randperm(numPts, sampleSize);


    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), prop1(numPts), prop2(numPts), disvalue(numPts), disvalue1(numPts), disvalue2(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    double tsd;
    std::vector<double> exy;

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
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    for (it = 1; it <= maxIteration; it++) {
        indices = softmax_y(numPts, prop, sampleSize);


        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        //td = 撮影画像, cm = 地図画像
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2); //変換に応じたサイズのベクトルで行列を作成

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++; //インライアの数のみ数える
                inliersIdx[iidx++] = j;
                //disvalue[j] = exE(tsd, norm, 0.965);
                disvalue[j] = fuzzyE(tsd, norm);
                //disvalue[j] = circle(tsd, norm);
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }

        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }

        for (int j = 0; j < numPts; j++) {
            int k;
            for (k = 0; k < sampleSize; k++)
                if (indices[k] == j) break;
            if (k != sampleSize) continue;

            if (inlierNum > 4) prop[j] += sumvalue / maxvalue * alpha * (disvalue1[j] + 0.9 * prop1[j] - prop[j]);
            else prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);

        }

        if (inlierNum > 4)
        {
            //TD(0)法の1step先の計算(予測)
            indices = softmax_y(numPts, prop, sampleSize);
            //for (size_t i = 0; i < 4; i++) // indices確認
            //    printf("indices = %zu\n" ,indices[i]); 

            for (int i = 0; i < sampleSize; i++) {
                samplePts1[i] = td[indices[i]];
                samplePts2[i] = cm[indices[i]];
            }
            //td = 撮影画像, cm = 地図画像
            cv::Mat tform1 = computematrix(mt, samplePts1, samplePts2); //変換に応じたサイズのベクトルで行列を作成

            for (size_t j = 0; j < numPts; j++) {
                cv::Point2d invPts = transform2d(td[j], tform1);
                cv::Point2d dist = invPts - cm[j];
                double norm = cv::norm(dist);
                if (norm < tsd) {
                    disvalue1[j] = fuzzyE(tsd, norm);
                }
                else {
                    norm = tsd;
                    disvalue1[j] = 0.0;
                }
                prop1[j] = prop[j] + sumvalue / maxvalue * alpha * (disvalue1[j] - prop[j]);
            }
        }
        if (iterNum <= it) break;
    }


    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    printf("sac後の有効対応点数：%d\n", (int)(selectedtd.size()));

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

size_t reinforcementlearning::computeLoopNumbers(size_t numPts, size_t inlierNum, size_t sampleSize)
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