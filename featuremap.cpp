#include "featuremap.h"
#include "fd_main.h"
#include "parameters.h"

void Correspond_selectedcm_selectedtd(source_data& srcd, destination_data& dstd, std::vector<cv::Point2d> selectedcm, std::vector<cv::Point2d> selectedtd)
{

    //std::vector<int> selectedcm_srcdoPts(selectedcm.size());
    //std::vector<int> selectedtd_dstdoPts(selectedtd.size());

    srcd.selectedcm_srcdoPts_init(selectedcm.size());
    dstd.selectedtd_dstdoPts_init(selectedtd.size());

    //初期検出の全キーポイントから、最後まで残ったdstdに居るものと逐次比較
    for (int y = 0; y < srcd.oFeatures.rows; y++) {

        debug_log("=====[srcd]キーポイントNo. [%3d]=====\n", y);
        debug_log("    座標：\n");
        debug_log("        x: %f\n", srcd.oPts[y].pt.x);
        debug_log("        y: %f\n", srcd.oPts[y].pt.y);
        debug_log("    探索結果：");



        bool found_pts = false;
        for (int srcd_i = 0; srcd_i < selectedcm.size(); srcd_i++) {
            if (srcd.oPts[y].pt.x == selectedcm[srcd_i].x && srcd.oPts[y].pt.y == selectedcm[srcd_i].y) {
                found_pts = true;
                debug_log("〇 --> selectedcm[%d]ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー\n", srcd_i);

                srcd.selectedcm_srcdoPts[srcd_i] = y;
                //selectedcm_srcdoPts.push_back(y);

                //debug_log("    記述子（srcd）：");
                for (int x = 0; x < srcd.oFeatures.cols; x++) {
                    //printf("%4d, ", (int)srcd.oFeatures.at<unsigned char>(y, x));
                    //std::cout << (int)cm.oFeatures.at<unsigned char>(y, x) << ", ";	// 記述子の中身を表示
                }
                debug_log("\n");
                continue;
            }
        }
        if (!found_pts) {
            debug_log("[srcd] Not matched.\n");
        }
        debug_log("\n\n");
    }
    debug_log("\n\n");




    // selectedtdは、dstdの何番かを探索
    for (int y = 0; y < dstd.oFeatures.rows; y++) {
        debug_log("=====[dstd]キーポイントNo. [%3d]=====\n", y);
        debug_log("    座標：\n");
        debug_log("        x: %f\n", dstd.oPts[y].pt.x);
        debug_log("        y: %f\n", dstd.oPts[y].pt.y);
        debug_log("    探索結果：");

        bool found_pts = false;
        for (int dstd_i = 0; dstd_i < selectedtd.size(); dstd_i++) {
            if (dstd.oPts[y].pt.x == selectedtd[dstd_i].x && dstd.oPts[y].pt.y == selectedtd[dstd_i].y) {
                found_pts = true;
                debug_log("〇 --> selectedtd[%d]ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー\n", dstd_i);

                //selectedcm_srcdoPts[srcd_i] = y;
                dstd.selectedtd_dstdoPts[dstd_i] = y;
                //selectedtd_dstdoPts.push_back(y);

                //printf("    記述子（dstd）：");
                //for (int x = 0; x < dstd.oFeatures.cols; x++) {
                //    printf("%4d, ", (int)dstd.oFeatures.at<unsigned char>(y, x));
                //    std::cout << (int)td.oFeatures.at<unsigned char>(y, x) << ", ";	// 記述子の中身を表示
                //}
                debug_log("\n");
                continue;
            }
        }
        if (!found_pts) {
            debug_log("[dstd] Not matched.\n");
        }
        debug_log("\n\n");
    }
    debug_log("\n\n");






    //for (int dstd_i = 0; dstd_i < selectedtd.size(); dstd_i++) {
    //    if (dstd.oPts[y].pt.x == selectedtd[dstd_i].x && dstd.oPts[y].pt.y == selectedtd[dstd_i].y) {

    //    }
    //}

    //for (int x = 0; x < srcd.oFeatures.cols; x++) {
    //    printf("%4d, ", (int)srcd.oFeatures.at<unsigned char>(y, x));
    //    std::cout << (int)cm.oFeatures.at<unsigned char>(y, x) << ", ";	// 記述子の中身を表示
    //}



    
    // 関係表
    if (IS_INFO || (IS_IMSHOW || IS_IMSHOW_SAC)) {
        std::vector<int> hummings;
        int humming_max = -1;  //8bit x 61dims-----------------------------------------------------------
        bool abs_humm = false;
        if (abs_humm) humming_max = 8 * 61;

        info_log("[selected{cm / td}]: srcd.oPts dstd.oPts  humm     norm_d\n");
        for (int i = 0; i < srcd.selectedcm_srcdoPts.size(); i++) {
            // ハミング距離を計算
            int humm = 0;
            for (int col = 0; col < dstd.oFeatures.cols; col++) {
                unsigned char src = (int)srcd.oFeatures.at<unsigned char>(srcd.selectedcm_srcdoPts[i], col);
                unsigned char dst = (int)dstd.oFeatures.at<unsigned char>(dstd.selectedtd_dstdoPts[i], col);

                //printf("%d, ", calcHumming(src, dst));
                humm += calcHumming(src, dst);
            }
            hummings.push_back(humm);
            if (!abs_humm) {
                if (humm > humming_max) humming_max = humm;
            }
            //putchar('\n');
            //printf("総延長：%d", humm);

            // ノルムを計算（キーポイント位置同士の）
            double norm_d = -1.0;
            norm_d = (selectedcm[i].x - selectedtd[i].x) * (selectedcm[i].x - selectedtd[i].x) + (selectedcm[i].y - selectedtd[i].y) * (selectedcm[i].y - selectedtd[i].y);
            norm_d = sqrt(norm_d);

            info_log("[%17d]: %9d %9d %5d %10f\n", i, srcd.selectedcm_srcdoPts[i], dstd.selectedtd_dstdoPts[i], humm, norm_d);
        }


        if (IS_IMSHOW || IS_IMSHOW_SAC) {
            // 左にtd、右にcmが表示されるようにする
            std::vector<std::vector<cv::DMatch>> empty_matches_2;
            std::vector<cv::KeyPoint> empty_pts_2;

            cv::Mat match_img_with_calced_data;
            cv::drawMatches(dstd.oImage, empty_pts_2, srcd.oImage, empty_pts_2, empty_matches_2, match_img_with_calced_data);

            //std::cout << selectedcm.size() << "\n";

            for (int itr = 0; itr < selectedcm.size(); itr++) {
                //cv::line(match_img_with_calced_data, cv::Point(selectedtd[itr].x,                      selectedtd[itr].y)                       , cv::Point(dstd.oImage.rows + selectedcm[itr].x                       , selectedcm[itr].y),                        cv::Scalar(20, 250, 30));// , 2);// , cv::LINE_4);
                cv::line(match_img_with_calced_data, cv::Point(dstd.oPts[dstd.selectedtd_dstdoPts[itr]].pt.x, dstd.oPts[dstd.selectedtd_dstdoPts[itr]].pt.y), cv::Point(dstd.oImage.cols + srcd.oPts[srcd.selectedcm_srcdoPts[itr]].pt.x, srcd.oPts[srcd.selectedcm_srcdoPts[itr]].pt.y), getColorLevel(hummings[itr], humming_max));// , 2);// , cv::LINE_4);

                std::string line = std::to_string(itr);
                //line += "(";
                //line += std::to_string(hummings[itr]);
                //line += ")";


                cv::putText(
                    match_img_with_calced_data,
                    line,
                    cv::Point(dstd.oPts[dstd.selectedtd_dstdoPts[itr]].pt.x - 25, dstd.oPts[dstd.selectedtd_dstdoPts[itr]].pt.y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1
                );

                cv::putText(
                    match_img_with_calced_data,
                    line,
                    cv::Point(dstd.oImage.cols + srcd.oPts[srcd.selectedcm_srcdoPts[itr]].pt.x + 5, srcd.oPts[srcd.selectedcm_srcdoPts[itr]].pt.y + 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1
                );
            }

            cv::imwrite("./match_img_with_calced_data.jpg", match_img_with_calced_data);
            cv::resize(match_img_with_calced_data, match_img_with_calced_data, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
            cv::imshow("[ransac後]match_img_with_calced_data", match_img_with_calced_data);
        }
    }
}


void FeatureMap::feature_mapping(featureDetectionType fd, FeatureMap fm, featuremap_data& fmpd, source_data& srcd, destination_data& dstd, analysis_results& rst)
{
    // 一つ前のステップまでの特徴点と、今回追加した特徴点を色分け表示する機能
    fmpd.prev_oPts.clear();
    fmpd.prev_oPts = fmpd.oPts;


    // すでにfmpd(1回目はsrcd)にあって、今回マッチング成功した対応点は不要なので、
    // それ以外を逐次作業する。
    // |
    // |  1) dst画像から得た特徴点の持つ座標を、fmpd座標系に変換。
    // |  2) 最後まで残った有効な対応点の組（dstdの、じゃないの？）をfmpdに保存ーーーーーーーーーー
    std::vector<cv::KeyPoint> newoPts;
    cv::Mat newoFeatures = dstd.oFeatures;
    
    newoFeatures.resize(0);
    newoFeatures.resize(dstd.oFeatures.rows);
    int Feat_idx = 0;

    for (int i = 0; i < dstd.oPts.size(); i++) {
        bool append = false;    // 全てで使用
        bool found = false;     // mOLDERで使用
        switch (fm.mt)
        {
        case MappingType::mALL:
            append = true;
            break;

        case MappingType::mSIMILAR:
            // 難しそう。（対応点の組が存在しない特徴点は新しいマップに追加できない、ということになってしまう、、。
            //// 各特徴点について、「61次元のAKAZE特徴量のハミング距離」を計算。指定範囲以内であれば地図に追加
            //// ハミング距離が離れすぎている場合は、特徴量記述子をコピーしてしまうのが問題になりそうなので後日検討
            //int humm = 0;
            //for (int col = 0; col < dstd.oFeatures.cols; col++) {
            //    unsigned char src = (int)srcd.oFeatures.at<unsigned char>(i, col);
            //    unsigned char dst = (int)dstd.oFeatures.at<unsigned char>(i, col);
            //}
            error_log("[featuremap.cpp] 実装されていない地図生成タイプ\n");
            exit(EXIT_FAILURE);
            break;


        case MappingType::mOLDER:
            // dstd.oPtsのインデックスと、selectedtd_dstdoPts[i]の値が一致するなら、追加しない（重複になる）。一致しないなら追加する。            
            for (int td = 0; td < dstd.selectedtd_dstdoPts.size(); td++) {
                if (i == dstd.selectedtd_dstdoPts[td]) {
                    // 対応点の組が成立している！-->追加しない。
                    found = true;
                    break;
                }
            }
            // 対応点の組が成立していない！==>未知の特徴点なので、地図に追加する。
            if (!found) {
                append = true;
            }            
            break;


        case MappingType::mNEWER:
            error_log("[featuremap.cpp] 実装されていない地図生成タイプ\n");
            exit(EXIT_FAILURE);
            break;

        default:
            error_log("[featuremap.cpp] 意図されていない入力：地図生成タイプ\n");
            exit(EXIT_FAILURE);
        }

        if (append) {
            // ==========キーポイントの追加==========
            cv::Point2f px;
            cv::KeyPoint tmp;

            if (fm.reDetection) {
                // 再検出されているのであれば，座標位置はコピーでOK．
                px = dstd.oPts[i].pt;
            }
            else {
                // 再検出をしないのであれば，座標変換が必要
                px = transform2f(dstd.oPts[i].pt, fmpd.curr_tform);
            }               

            tmp = dstd.oPts[i];
            tmp.pt = px;

            newoPts.push_back(tmp);


            // ==========特徴量の追加==========
            for (int col = 0; col < dstd.oFeatures.cols; col++) {
                copy2features(fd, newoFeatures, dstd.oFeatures, Feat_idx, col, i, col); // 特徴量の中身がfloat型かunsigned char型かでコピー方法が異なるので，その分岐を処理する関数
            }
            Feat_idx++;
        }
        newoFeatures.resize(Feat_idx + 1);      // もし3回追加したなら、Feat_idxは2を示す。resizeは行数を指定するので、+1して3個の要素は「消してしまわないように」する。
    }

    fmpd.new_oPts.clear();
    fmpd.new_oPts = newoPts;
    appendNewFeaturesInfo(newoPts, newoFeatures, fmpd, false, fd);



    // SLAMの核心である地図画像を描画
    // まず前回の更新までで存在していた特徴点を白で描画し，その上から新たに追加された特徴点を緑で描画．   
    cv::Mat fmp_img_empty(cv::Size(fmpd.mps_x, fmpd.mps_y), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat fmp_img;
    cv::drawKeypoints(fmp_img_empty, fmpd.prev_oPts, fmp_img, cv::Scalar::all(250));//, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    cv::drawKeypoints(fmp_img, fmpd.new_oPts, fmp_img, cv::Scalar(20, 250, 30));//, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    
    // 変換行列により推定した画像の位置を枠で描画
    cv::line(fmp_img, rst.c00, rst.c01, cv::Scalar(0, 250, 0), cv::LINE_4);
    cv::line(fmp_img, rst.c01, rst.c11, cv::Scalar(0, 250, 0), cv::LINE_4);
    cv::line(fmp_img, rst.c11, rst.c10, cv::Scalar(0, 250, 0), cv::LINE_4);
    cv::line(fmp_img, rst.c10, rst.c00, cv::Scalar(0, 250, 0), cv::LINE_4);

    // 地図画像を保存（最後は必ず．途中経過は必要に応じて．）
    if (SAVE_SEQUENTIAL_FMP || rst.total_image_num - 1 == fmpd.newest_taken_idx) { // 添え字の関係でマイナス1
        char fname[256] = "";
        snprintf(fname, sizeof(fname), "./final_fmp_%03d.jpg", fmpd.newest_taken_idx);
        cv::imwrite(fname, fmp_img);
    }

    // featureMapの表示
    if (IS_IMSHOW || IS_IMSHOW_FMP) {
        // 表示用に縮小
        cv::resize(fmp_img, fmp_img, cv::Size(), IMAGE_SMALLEN_SCALE, IMAGE_SMALLEN_SCALE);
        cv::imshow("fmp_img", fmp_img);
    }

    printf("===== 今回の地図拡張に伴う特徴点数の変化：\n");
    printf("   --> before\t: %6d\n", (int)fmpd.prev_oPts.size());
    printf("   --> 今回増分\t:%6d\n", (int)fmpd.new_oPts.size());
    printf("   --> after\t: %6d\n", (int)fmpd.oPts.size());
}


