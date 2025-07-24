#pragma once

#if __APPLE__
// ----- Apple -----
std::string JAXA_database_map_img_path = "/Users/hiroaki/Pictures/JAXA_database/mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp";
std::string RootPath = "/Users/hiroaki/Pictures/20230718_splitted_moon_images_newest/";
#else
#if __GNUC__
// ----- Linux(Raspberry Pi) -----
#if __ARM_ARCH
std::string JAXA_database_map_img_path = "/home/hiroaki/Pictures/JAXA_database/mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp";
std::string RootPath = "/home/hiroaki/Pictures/20230718_splitted_moon_images_newest/";
#else
//  ----- Linux(CentOSなど) -----
std::string JAXA_database_map_img_path = "/Jaxa_database/mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp";
std::string RootPath = "/home/miura22/ピクチャ/20230718_splitted_moon_images_newest/";
#endif
#else
// ----- Windows -----
#define WINDOWS
std::string JAXA_database_map_img_path = "C:/JAXA_database/mapimg/CST1/TCO_CST1_TM_SIM_a7351_i3438_h36000_lanczos3.bmp";
std::string RootPath = "C:/JAXA_database/FeatureMapping/";
//std::string RootPath = "D:/JAXA_database_splitted_moon/";
#endif //GNUC
#endif // APPLE


std::string DistNum = "dist[400]_seed[0]/";
//std::string DistNum = "400/";


// ==================== Shared Settings ========================
featureType FT_SHARED = featureType::fPCA_16bin; // fPCA_16bin ←16B 記述子を使用
// =============================================================

// RANSAC  (使うものだけTrueにすること)
#define RANSAC_NORMAL true	
#define RANSAC_grfl   false


// 変更不要
#if RANSAC_NORMAL
ransacType RANSAC_TYPE = ransacType::rNORMAL;
#elif RANSAC_grfl
ransacType RANSAC_TYPE = ransacType::rTD0;
#endif
