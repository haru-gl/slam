#include "results.h"

void save_statistics(int img_num, featuremap_data fmpd, analysis_results rst)
{
    CSVHandler csv("./statistics.csv");
    std::string col_name = "";
    if (img_num == INT_MAX) col_name = "final";
    else              col_name = std::to_string(img_num);


    csv.setValue(col_name, "newest_taken_idx", std::to_string(fmpd.newest_taken_idx));
    csv.setValue(col_name, "fmpd.oPts.size()", std::to_string(fmpd.oPts.size()));
    csv.setValue(col_name, "elapsed_time_formap", std::to_string(rst.elapsed_time_formap));
    csv.setValue(col_name, "<--sum", std::to_string(rst.elapsed_time_formap_sum));

    csv.setValue(col_name, "rst.norm_sum", std::to_string(rst.norm_sum));
    csv.setValue(col_name, "rst.norm_size", std::to_string(rst.norm_size));
    csv.setValue(col_name, "rst.norm_min", std::to_string(rst.norm_min));
    csv.setValue(col_name, "rst.norm_max", std::to_string(rst.norm_max));
    csv.setValue(col_name, "rst.norm_ave", std::to_string(rst.norm_ave));
    csv.setValue(col_name, "rst.norm_3sigma", std::to_string(rst.norm_3sigma));
    csv.setValue(col_name, "elapsed_time_foraccuracy", std::to_string(rst.elapsed_time_foraccuracy));

    csv.saveChanges();
    //csv.readCSV();   // これは，逐次保存＆読込を挟まないと，行方向の順番がめちゃくちゃになるため．
}


void save_finals(featuremap_data fmpd, analysis_results rst, std::string DistNum, featureType ft, ransacType rt)
{
    // 今回実行の最終結果のみをまとめたファイルに追記
    CSVHandler csv("../finals.csv");
    time_t now = std::time(nullptr);      // 現在時刻
    std::string cut = std::to_string(now); // currnt_unix_time

    csv.setValue(cut, "newest_taken_idx", "-");
    csv.setValue(cut, "fmpd.oPts.size()", std::to_string(fmpd.oPts.size()));
    csv.setValue(cut, "elapsed_time_formap", std::to_string(0));
    csv.setValue(cut, "<--sum", std::to_string(rst.elapsed_time_formap_sum));
    csv.setValue(cut, "rst.norm_sum", std::to_string(rst.norm_sum));
    csv.setValue(cut, "rst.norm_size", std::to_string(rst.norm_size));
    csv.setValue(cut, "rst.norm_min", std::to_string(rst.norm_min));
    csv.setValue(cut, "rst.norm_max", std::to_string(rst.norm_max));
    csv.setValue(cut, "rst.norm_ave", std::to_string(rst.norm_ave));
    csv.setValue(cut, "rst.norm_3sigma", std::to_string(rst.norm_3sigma));
    csv.setValue(cut, "elapsed_time_foraccuracy", std::to_string(rst.elapsed_time_foraccuracy));

    csv.setValue(cut, "DistNum", DistNum);

    switch (ft)
    {
    case featureType::fAKAZE:
        csv.setValue(cut, "featureType", "fAKAZE");
        csv.setValue(cut, "threshold", std::to_string(INIT_AKAZE_THRESHOLD));
        break;
    case featureType::fKAZE:
        //csv.setValue(cut, "featureType", "fKAZE");
        //csv.setValue(cut, "threshold", std::to_string(xxx));
        break;
    case featureType::fSURF:
        csv.setValue(cut, "featureType", "fSURF");
        csv.setValue(cut, "threshold", std::to_string(INIT_SURF_THRESHOLD));
        break;
    case featureType::fSIFT:
        //csv.setValue(cut, "featureType", "fSIFT");
        //csv.setValue(cut, "threshold", std::to_string(xxx));
        printf("[results.cpp line:67] 特徴点検出法が記録されない！（想定外の入力）\n");
        exit(EXIT_FAILURE);
        break;
    case featureType::fBRISK:
        csv.setValue(cut, "featureType", "fBRISK");
        csv.setValue(cut, "threshold", std::to_string(INIT_BRISK_THRESHOLD));
        break;
    case featureType::fORB:
        csv.setValue(cut, "featureType", "fORB");
        csv.setValue(cut, "threshold", std::to_string(INIT_ORB_MAXFEATURES));
        break;
    case featureType::fPCA_float:
        csv.setValue(cut, "featureType", "fPCA_float");
        csv.setValue(cut, "threshold", std::to_string(INIT_PCA_THRESHOLD));
        csv.setValue(cut, "ENABLE_SUBPIXEL_ESTIMATION", ENABLE_SUBPIXEL_ESTIMATION ? "true" : "false");
        break;
    case featureType::fPCA_uschar:
        csv.setValue(cut, "featureType", "fPCA_uschar");
        csv.setValue(cut, "threshold", std::to_string(INIT_PCA_THRESHOLD));
        csv.setValue(cut, "ENABLE_SUBPIXEL_ESTIMATION", ENABLE_SUBPIXEL_ESTIMATION ? "true" : "false");
        break;
    case featureType::fPCA_16bin:
        csv.setValue(cut, "featureType", "fPCA_16bin");
        csv.setValue(cut, "threshold", std::to_string(INIT_PCA_THRESHOLD));
        csv.setValue(cut, "ENABLE_SUBPIXEL_ESTIMATION", ENABLE_SUBPIXEL_ESTIMATION ? "true" : "false");
        break;
    case featureType::fDAMMY:
        csv.setValue(cut, "featureType", "fDAMMY");
        break;
    default:
        error_log("[results.cpp] 特徴点検出法の例外（92行）\n");
        exit(EXIT_FAILURE);
        break;
    }


    switch (rt)
    {
    case ransacType::rNORMAL:
        csv.setValue(cut, "ransacType", "normal");
        break;
    case ransacType::rTD0:
        csv.setValue(cut, "ransacType", "td0");
        break;
    default:
        error_log("[results.cpp] 予期しないransacType．");
        exit(EXIT_FAILURE);
}


    csv.setValue(cut, "SQ_SIZE", rst.fd.sq_size ? "true" : "false");
    csv.setValue(cut, "SQ_RESPONSE", rst.fd.sq_response ? "true" : "false");
    csv.setValue(cut, "SQ_SIZE_TH", std::to_string(SQ_SIZE_TH));
    csv.setValue(cut, "SQ_RESPONSE_TH", std::to_string(SQ_RESPONSE_TH));
    csv.setValue(cut, "SQ_SIZE_ROUND_DPN", std::to_string(SQ_SIZE_ROUND_DPN));
    csv.setValue(cut, "SQ_RESPONSE_ROUND_DPN", std::to_string(SQ_RESPONSE_ROUND_DPN));

    csv.setValue(cut, "SQ_SIZE_SL", nameof_enum::nameof(rst.fd.sq_size_sl));
    csv.setValue(cut, "SQ_RESPONSE_SL", nameof_enum::nameof(rst.fd.sq_response_sl));
    csv.setValue(cut, "sqOrderType", nameof_enum::nameof(rst.fd.sq_order));

    csv.saveChanges();
}