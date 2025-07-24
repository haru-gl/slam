#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>



#include "enclasses.h"
#include "parameters.h"
#include "akaze.h"
#include "surf.h"
#include "classes.h"
#include "knn.h"
#include "sac_main.h"
#include "functions.h"
#include "featuremap.h"
#include "csv.h"
#include "fd_main.h"
#include "functions.h"
#include "config.h"
#include "reinforcementlearning.h"
#include "results.h"



bool IS_FIRST_RUN = true;


void featurempap_generation_main(featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rst)
{
    //Setting of Feature Points Detection
    featureDetectionType fd;
    fd.ft = FT_SHARED; // この設定は`accuracy_evaluation`と同一でなければならない．
    fd.sq_size = false;
    fd.sq_response = false;
    
    fd.st = sqType::sqPERCENT;
    fd.sq_size_sl     = sl::small_is_better;
    fd.sq_response_sl = sl::large_is_better;

    fd.sq_order = sqOrderType::size_sq_response_sq;

    rst.fd = fd;    // 設定情報を「analysis_results」に記録

    
    //Setting of Position Estimation
    posestType pe;
    pe.mt = matchingType::mPROJECTIVE;//SIMILARITY;
    pe.rm = ransacMode::dNORMAL;
    pe.ct = matrixcalType::cSVD;    // 特異値分解
    pe.ndon = false;
    pe.use_OpenCV_findHomography = false;

    //Setting of kNN
    knnType kt = knnType::kNORMAL;
    knn knnt; knnt.set_knn_sortflag(false);

    //Setting of RANSAC
#if RANSAC_NORMAL 
    sac sc; //normal RANSAC  
#elif RANSAC_grfl
    reinforcementlearning sc; //強化学習RANSAC
#endif
          
  

    
    // Setting of Mapping
    FeatureMap fm;
    fm.mt = MappingType::mOLDER;
    fm.reDetection = false;  // trueの場合，dstd.oImageが変形されてしまうので注意．
    fm.interpolation = cv::INTER_LANCZOS4;

    //fm.get_MappingType()




    featurepointdetection(fd, srcd, dstd);

    
    // --- 変更後（16B 記述子対応版）--------------------
    if (IS_FIRST_RUN) {
        if (fd.ft == featureType::fPCA_16bin)      // ★ 追加: 新タイプ判定
            fmpd.dimFeatures = 16;                 //   → 固定 16
        else
            fmpd.dimFeatures = srcd.oFeatures.cols;

        appendNewFeaturesInfo(srcd.oPts, srcd.oFeatures, fmpd, true, fd);
    }
    

    if (IS_IMSHOW) {
        drawKeyPoints(srcd, dstd);
    }
    



    rst.goodPairsNum = knnt.match(fd.ft, kt, srcd, dstd);
    printf("[DEBUG] goodPairsNum = %zu\n", rst.goodPairsNum);
    printf("[DEBUG] minGoodPairs = %zu\n", get_minGP(pe.mt));   // projective なら 4



#if RANSAC_NORMAL 
    sc.positionestimation_normal(pe, fmpd, srcd, dstd, rst); //normal RANSAC
#elif RANSAC_grfl
    sc.positionestimation_grfl(pe, fmpd, srcd, dstd, rst);  //強化学習RANSAC 
#endif
    


    

    // ========== 再検出処理 ==========
    if (fm.reDetection) {
        cv::Mat re_before_tform = fmpd.curr_tform.clone();

        // 画像の変形
        cv::Point2d after_size = transform2d(cv::Point2d(SZ, SZ), re_before_tform);//cv::Point2d(SZ, SZ);
        cv::warpPerspective(dstd.oImage, dstd.oImage, fmpd.curr_tform, cv::Size(after_size), fm.interpolation);//dstd.oImage.size()
        if (IS_IMSHOW) {
            cv::Mat afp;
            cv::resize(dstd.oImage, afp, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
            cv::imshow("射影変換後", afp);
        }

        // 特徴点・特徴量の検出
        featurepointdetection(fd, srcd, dstd);
        rst.goodPairsNum = knnt.match(fd.ft, kt, srcd, dstd);        

#if RANSAC_NORMAL 
        sc.positionestimation_normal(pe, fmpd, srcd, dstd, rst); //normal RANSAC
#elif RANSAC_grfl
        sc.positionestimation_grfl(pe, fmpd, srcd, dstd, rst);  //強化学習RANSAC 
#endif

        fmpd.curr_tform = re_before_tform.clone();
    }

    // dst画像をfmpdに描画する際の位置を計算
    rst.c00 = transform2d(cv::Point2d(0.0, 0.0), fmpd.curr_tform);
    rst.c01 = transform2d(cv::Point2d((double)512.0, 0.0), fmpd.curr_tform);
    rst.c11 = transform2d(cv::Point2d((double)512.0, (double)512.0), fmpd.curr_tform);
    rst.c10 = transform2d(cv::Point2d(0.0, (double)512.0), fmpd.curr_tform);

    fm.feature_mapping(fd, fm, fmpd, srcd, dstd, rst);



 



    info_log("\n");
    info_log("%5f   %5f\n", rst.c00.x, rst.c01.x);
    info_log("%5f   %5f\n", rst.c00.y, rst.c01.y);
    info_log("\n");
    info_log("%5f   %5f\n", rst.c10.x, rst.c11.x);
    info_log("%5f   %5f\n", rst.c10.y, rst.c11.y);

    

}


// 生成された地図画像の精度評価を実施
void accuracy_evaluation(featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rst, bool is_create_detail_file)
{
#if _WIN64 || _WIN32
    srcd.~source_data();
    dstd.~destination_data();
#endif

    // srcのデータ（fmpdからコピー）
    {
        srcd.oImage = cv::Mat(cv::Size(fmpd.mps_x, fmpd.mps_y), CV_8UC3, cv::Scalar(0, 0, 0));
        //                                                      ***** srcdの画像は、2回目以降は
        //														fmpdのコピーになるので、空っぽ。
        //														このままだとエラーになるので、
        //														空を与える。
        srcd.oImage_dummy = true;                               // 上の情報。有効にしておけば、srcd「画像」から特徴点を取得しなくなり、既存のpts情報を流用するようになる。
        srcd.oPts = fmpd.oPts;
        srcd.oFeatures = fmpd.oFeatures;
        srcd.oMatchedPts = fmpd.oMatchedPts;
        srcd.oMatchedFeatures = fmpd.oMatchedFeatures;
    }

    // dstに地図画像を読み込ませる
    {
        dstd.oImage = cv::imread(JAXA_database_map_img_path, cv::IMREAD_GRAYSCALE);
    }



    //Setting of Feature Points Detection
    featureDetectionType fd;
    fd.ft = FT_SHARED; // この設定は`map_generation_main`と同一でなければならない．
    fd.sq_size = false;
    fd.sq_response = false;

    fd.st = sqType::sqPERCENT;

    fd.sq_size_sl = sl::small_is_better;
    fd.sq_response_sl = sl::large_is_better;

    fd.sq_order = sqOrderType::size_sq_response_sq;

    //Setting of Position Estimation
    posestType pe;
    pe.mt = matchingType::mPROJECTIVE;//SIMILARITY;
    pe.rm = ransacMode::dNORMAL;
    pe.ct = matrixcalType::cSVD;
    pe.ndon = false;
    pe.use_OpenCV_findHomography = true;

    //Setting of kNN
    knnType kt = knnType::kNORMAL;
    knn knnt; knnt.set_knn_sortflag(false);
    
    //Setting of RANSAC
#if RANSAC_NORMAL 
    sac sc; //normal RANSAC  
#elif RANSAC_grfl
    reinforcementlearning sc; //強化学習RANSAC
#endif



    featurepointdetection(fd, srcd, dstd);
    drawKeyPoints(srcd, dstd);    
    rst.goodPairsNum = knnt.match(fd.ft, kt, srcd, dstd);


#if RANSAC_NORMAL 
    sc.positionestimation_normal(pe, fmpd, srcd, dstd, rst); //normal RANSAC
#elif RANSAC_grfl
    sc.positionestimation_grfl(pe, fmpd, srcd, dstd, rst);  //強化学習RANSAC 
#endif



    CSVHandler csv_allpoints("./allpoints.csv");
    double norm_sum = 0;
    std::vector<double> l2_norms(srcd.selectedcm_srcdoPts.size());
    for (int i = 0; i < srcd.selectedcm_srcdoPts.size(); i++) {
        int srcd_idx = srcd.selectedcm_srcdoPts[i];
        int dstd_idx = dstd.selectedtd_dstdoPts[i];
        double diff_x = srcd.oPts[srcd_idx].pt.x - dstd.oPts[dstd_idx].pt.x;
        double diff_y = srcd.oPts[srcd_idx].pt.y - dstd.oPts[dstd_idx].pt.y;
        double l2_norm = sqrt(diff_x * diff_x + diff_y * diff_y);
        // info_log("fmpd_x: %6f, fmpd_y: %6f, diff_x: %6f, diff_y: %6f, l2_norm%6f\n", srcd.oPts[srcd_idx].pt.x, srcd.oPts[srcd_idx].pt.y, 
        //                                                                             diff_x, diff_y, l2_norm);
        l2_norms[i] = l2_norm;

        if (is_create_detail_file) {
            // 結果はCSVに書き出しながら進める
            csv_allpoints.setValue(std::to_string(i), "srcd_idx", std::to_string(srcd_idx));
            csv_allpoints.setValue(std::to_string(i), "dstd_idx", std::to_string(dstd_idx));
            csv_allpoints.setValue(std::to_string(i), "srcd_x", std::to_string(srcd.oPts[srcd_idx].pt.x));
            csv_allpoints.setValue(std::to_string(i), "srcd_y", std::to_string(srcd.oPts[srcd_idx].pt.y));
            csv_allpoints.setValue(std::to_string(i), "dstd_x", std::to_string(dstd.oPts[dstd_idx].pt.x));
            csv_allpoints.setValue(std::to_string(i), "dstd_y", std::to_string(dstd.oPts[dstd_idx].pt.y));
            csv_allpoints.setValue(std::to_string(i), "l2_norm", std::to_string(l2_norm));
        }
        norm_sum += l2_norm;

        if (rst.norm_max < l2_norm) {
            rst.norm_max = l2_norm;
        }

        if (l2_norm < rst.norm_min) {
            rst.norm_min = l2_norm;
        }
    }
    csv_allpoints.saveChanges();  // 保存を忘れないこと！（＊逐次保存を行わず，ここでまとめて保存しているので，保存されたcsvファイルの行方向の順番はランダム...。）

    rst.norm_sum = norm_sum;
    rst.norm_size = (int)srcd.selectedcm_srcdoPts.size();
    rst.norm_ave = rst.norm_sum / rst.norm_size;
    
    double sq_in_sum = 0;
    for (int i = 0; i < srcd.selectedcm_srcdoPts.size(); i++) {
        sq_in_sum += (l2_norms[i] - rst.norm_ave) * (l2_norms[i] - rst.norm_ave);
    }
    double sq_out = sqrt(sq_in_sum / srcd.selectedcm_srcdoPts.size());
    double sigma3 = 3 * sq_out;
    rst.norm_3sigma = sigma3;
    

    


    



    if (is_create_detail_file) {
        // 矢印を引いてみる
        cv::Mat arrowed_map;
        arrowed_map = cv::imread(JAXA_database_map_img_path);

        for (int i = 0; i < srcd.selectedcm_srcdoPts.size(); i++) {
            int srcd_idx = srcd.selectedcm_srcdoPts[i];
            int dstd_idx = dstd.selectedtd_dstdoPts[i];
            double st_x = dstd.oPts[dstd_idx].pt.x;
            double st_y = dstd.oPts[dstd_idx].pt.y;

            double ed_x = srcd.oPts[srcd_idx].pt.x;
            double ed_y = srcd.oPts[srcd_idx].pt.y;
            cv::arrowedLine(arrowed_map, cv::Point(st_x, st_y), cv::Point(ed_x, ed_y), cv::Scalar(0, 250, 0), 1, cv::LINE_8, 0, 0.3);
        }

        char fname[256] = "";
        snprintf(fname, sizeof(fname), "./arrowed_map_%03d.jpg", fmpd.newest_taken_idx);

        cv::imwrite(fname, arrowed_map);
        if (IS_IMSHOW) {
            cv::resize(arrowed_map, arrowed_map, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
            cv::imshow(fname, arrowed_map);
        }
    }
}


void load_images(std::string imgfile, std::vector<cv::Mat>& takenImgs)
{
    // include <fstream>
    std::ifstream file(imgfile);  // 読み込むファイルのパスを指定
    std::string line;

    // 画像の連続読込
    std::vector<std::string> takenImgPath;
    while (std::getline(file, line)) {  // 1行ずつ読み込む
        //std::cout << line << std::endl;
        takenImgPath.push_back(RootPath + DistNum + line);
    }

    
    for (int i = 0; i < takenImgPath.size(); i++) {
        debug_log("%s\n", takenImgPath[i].c_str());
        takenImgs.push_back(cv::imread(takenImgPath[i].c_str(), cv::IMREAD_GRAYSCALE));       // g++ではc_strが必須？ // グレースケールで読み込まないと，PCAでは死ぬ
        if (IS_IMSHOW) {
            cv::Mat tmp;
            tmp = takenImgs[i].clone();
            cv::resize(tmp, tmp, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
            cv::imshow("takenImgs", tmp);
        }
        cv::waitKey(1);
    }
    if (IS_IMSHOW) {
        cv::destroyWindow("takenImgs");
    }

}


int main()
{   
    // 全所要時間の計測
    std::chrono::system_clock::time_point whole_st = std::chrono::system_clock::now();


    featuremap_data fmpd;
    source_data srcd;
    destination_data dstd;
    analysis_results rst;


    std::string imgfile;
    imgfile.append(RootPath);
    imgfile.append(DistNum);
    imgfile.append("imgfile.txt");

    std::vector<cv::Mat> takenImgs;


    // 画像読込
    load_images(imgfile, takenImgs); 
    rst.total_image_num = takenImgs.size(); // 総枚数を記録
    

            
    //	Measurement of time
    std::chrono::system_clock::time_point st, ed;

    // 探査機による撮影を模擬
    for (int i = 0; i < takenImgs.size()-1; i++) {
        if (DISPLAY_CLEAR_EACH) printf("\033[2J \033[0;0H"); // 画面消去，カーソルを(0, 0)へ移動
        else progress_log("\n");
        progress_log("==================== Image has now been taken in [%3d / %3d] ====================\n", i + 1, (int)takenImgs.size());


        // ==========　src, rstについて ==========
#if _WIN64 || _WIN32  // Linuxではdouble free or corruption`になることがある
        rst.~analysis_results();
        srcd.~source_data();    
#endif

        srcd.oImage = takenImgs[i];
        
        if (IS_FIRST_RUN){
        }
        else {
            srcd.oImage = cv::Mat(cv::Size(fmpd.mps_x, fmpd.mps_y), CV_8UC3, cv::Scalar(0, 0, 0));
            //                                                      ***** srcdの画像は、2回目以降は
            //                                                        fmpdのコピーになるので、空っぽ。
            //                                                        このままだとエラーになるので、
            //                                                        空を与える。
            srcd.oImage_dummy = true;                               // 上の情報。有効にしておけば、srcd「画像」から特徴点を取得しなくなり、既存のpts情報を流用するようになる。
            srcd.oPts = fmpd.oPts;
            srcd.oFeatures = fmpd.oFeatures;
            srcd.oMatchedPts = fmpd.oMatchedPts;
            srcd.oMatchedFeatures = fmpd.oMatchedFeatures;
        }

        
        // ========== dstdについて ==========
#if _WIN64 || _WIN32
        dstd.~destination_data();       // Linuxでは`double free or corruption`になることがある
#endif
        dstd.oImage = takenImgs[i + 1];

        fmpd.newest_taken_idx = i + 1;


        if (IS_IMSHOW) {
            cv::Mat src_tmp = srcd.oImage.clone();
            cv::Mat dst_tmp = dstd.oImage.clone();
            cv::resize(src_tmp, src_tmp, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
            cv::resize(dst_tmp, dst_tmp, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
            cv::imshow("source", src_tmp);
            cv::imshow("destination", dst_tmp);
        }


        st = std::chrono::system_clock::now();
        featurempap_generation_main(fmpd, srcd, dstd, rst);
        ed = std::chrono::system_clock::now();

             
        rst.elapsed_time_formap      = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
        rst.elapsed_time_formap_sum += rst.elapsed_time_formap;
        



        // 特徴点マップの生成精度評価
        st = std::chrono::system_clock::now();
        if (ACCURACY_EVALUATION_IN_EACH) {
            accuracy_evaluation(fmpd, srcd, dstd, rst, false);  // 詳細なファイルは、ここでは出力しなくてよい。
        }
        ed = std::chrono::system_clock::now();

        rst.elapsed_time_foraccuracy = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();


        save_statistics(i, fmpd, rst);

        debug_log("srcd.oPts.size(): %d\n", (int)srcd.oPts.size());
        debug_log("dstd.oPts.size(): %d\n", (int)dstd.oPts.size());
        debug_log("fmpd.oPts.size(): %d\n", (int)fmpd.oPts.size());
        debug_log("");
        debug_log("srcd.oFeatures.rows: %d\n", srcd.oFeatures.rows);
        debug_log("dstd.oFeatures.rows: %d\n", dstd.oFeatures.rows);
        debug_log("fmpd.oFeatures.rows: %d\n", fmpd.oFeatures.rows);
        debug_log("");
        debug_log("srcd.oMatchedPts.size(): %d\n", (int)srcd.oMatchedPts.size());
        debug_log("dstd.oMatchedPts.size(): %d\n", (int)dstd.oMatchedPts.size());
        debug_log("fmpd.oMatchedPts.size(): %d\n", (int)fmpd.oMatchedPts.size());
        debug_log("");
        debug_log("srcd.oMatchedFeatures.rows: %d\n", srcd.oMatchedFeatures.rows);
        debug_log("dstd.oMatchedFeatures.rows: %d\n", dstd.oMatchedFeatures.rows);
        debug_log("fmpd.oMatchedFeatures.rows: %d\n", fmpd.oMatchedFeatures.rows);

        

        if (STOP_EACH) {
            cv::waitKey(0);
        }
        else {
            cv::waitKey(1);
        }
        
        IS_FIRST_RUN = false;
    }


#if _WIN64 || _WIN32
    rst.~analysis_results();
#endif

    if (ACCURACY_EVALUATION_FINAL) {
        // 最終的に出来上がった特徴点マップの生成精度評価
        st = std::chrono::system_clock::now();
        accuracy_evaluation(fmpd, srcd, dstd, rst, true);
        ed = std::chrono::system_clock::now();

        // 特徴点マップの生成時間評価
        rst.elapsed_time_foraccuracy = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
    }

    // csvファイルへの書き込み
    save_statistics(INT_MAX, fmpd, rst);    // INT_MAX means "final".
    save_finals(fmpd, rst, DistNum, FT_SHARED, RANSAC_TYPE);



    // 全所要時間の計測
    std::chrono::system_clock::time_point whole_ed = std::chrono::system_clock::now();

    // オーバーヘッドの計算
    long long whole_time = std::chrono::duration_cast<std::chrono::milliseconds>(whole_ed - whole_st).count();
    float percent_overhead = (whole_time - rst.elapsed_time_formap_sum) / (float)whole_time;


    progress_log("========== RESULTS ==========\n");
    progress_log("ノルムの和：%f\n", rst.norm_sum);
    progress_log("合計の個数：%d\n", rst.norm_size);
    progress_log("平均：%f\n", rst.norm_ave);
    progress_log("最大値：%f\n", rst.norm_max);
    progress_log("最小値：%f\n", rst.norm_min);
    progress_log("3sigma：%f\n", rst.norm_3sigma);
    progress_log("地図生成の総所要時間　：%d [ms]\n", (int)rst.elapsed_time_formap_sum);// long long型
    progress_log("本プログラムの動作時間：%d [ms]\n", (int)whole_time);
    progress_log("地図生成以外のオーバーヘッド：%f %%\n", percent_overhead * 100);


    cv::waitKey(1);
    return 0;
}
