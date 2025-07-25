#pragma once

#define IS_PROGRESS true		// progress_log
#define IS_ERROR	true		// error_log
#define IS_WARNING	true		// warn_log
#define IS_INFO		false		// info_log
#define IS_DEBUG	false		// debug_log

#define USE_ANMS true   // trueにするとANMS（特徴点の均一化）が有効

#define IS_IMSHOW false			// 全体スイッチ（個別スイッチが有効だと，それは優先される）
#define IS_IMSHOW_FMP true		// 個別スイッチ
#define IS_IMSHOW_SAC false		// 個別スイッチ
#define IMAGE_SMALLEN_SCALE 0.5	// 画像をFHDディスプレイにも収まるよう，「表示のみ」0.5xなどに変更（保存される画像の画質には影響しない）
#define SAVE_SEQUENTIAL_FMP false // 地図生成の過程を全て画像として保存するか

#define DISPLAY_CLEAR_EACH true				// 地図を拡張するたびに，出力画面をクリアするか
#define STOP_EACH false						// 地図を拡張するたびに，動作を停止してキー入力を待つか
#define ACCURACY_EVALUATION_IN_EACH false	// 地図を拡張するたびに，精度評価を行うか

#define ACCURACY_EVALUATION_FINAL true	// 最終的な精度評価を行うか（通常は行うべき）

#define CV_KEYPOINT_DSTD_TO_CSV false		// csvファイルに，dstd（撮影画像）上で検出された特徴点の全情報（cv::KeyPoint）を出力するか（※sq_size, responseによる絞り込みのあと，k-NNの手前でデータが取得される．必要に応じてsq_*はfalseにすること．）（なおcsvファイルが同名で作成されるので，初回の1回しか動かない．）

#define CSV_OVERRIDE_STOP false	// csvファイル書き込み時に，既存データを上書きしてしまう場合に，その前にエラー吐いて停止させるか．


//Setting of the target images
// 正方形を仮定している．
#define SZ 512



// Setting of sq_size,response  sqTypeの設定により意味は変化する．（割合または絶対個数）
//AKAZE の場合，< 0.2, 0.7 >，SURF の場合< 0.3, 0.6 >
#define SQ_SIZE_TH 0.2
#define SQ_RESPONSE_TH 0.7

#define SQ_SIZE_ROUND_DPN 16			// size値で小数点以下を刻んでくる場合に，何桁目までで四捨五入して同一視するか
#define SQ_RESPONSE_ROUND_DPN 16



//Setting of AKAZE
#define INIT_AKAZE_DESCRIPTOR_SIZE 0
#define INIT_AKAZE_DESCRIPTOR_CHANNELS 3
#define INIT_AKAZE_THRESHOLD 1e-4 // おおよそ1万点が取得される．//6e-6//0.0010//0.000001
//-0.000005    0.00001
#define INIT_AKAZE_NOCTAVES 4
#define INIT_AKAZE_NOCTARVELAYERS 4


//Setting of SURF
#define INIT_SURF_EXTENDED false
#define INIT_SURF_THRESHOLD 50 //50で1万点くらい．
#define INIT_SURF_NOctaveLayers  3
#define INIT_SURF_NOctaves 4
#define INIT_SURF_Upright false


//Setting of BRISK
#define INIT_BRISK_OCTAVES 3
#define INIT_BRISK_THRESHOLD 20		// 9で4万点くらい．20で1万点くらい．小さい方がたくさん取ってくる．


//Setting of SIFT
#define INIT_SIFT_NFEATURES 0
#define INIT_SIFT_NOCTAVELAYERS 3
#define INIT_SIFT_CONTRASTTH 0.04
#define INIT_SIFT_EDGETH 10
#define INIT_SIFT_SIGMA 1.6


//Setting of ORB
#define INIT_ORB_MAXFEATURES 4000
#define INIT_ORB_SCALEFACTOR 1.2
#define INIT_ORB_THRESHOLD 31
#define INIT_ORB_FASTTHRESHOLD 18  // 18で1万点くらい．※小さい方がたくさん取れる．


//Setting of PCA
#define INIT_PCA_THRESHOLD 100.0
#define ENABLE_SUBPIXEL_ESTIMATION true
#define DETECT_POINTS_ONLY_WITHIN_CIRCLE false

//Setting of k-NN Method
#define INIT_KNNK 2
#define INIT_NNMATCHRATIO 0.90//0.84//0.93
#define INIT_DISTTH 3.0
#define INIT_MAXNUM 200
#define INIT_SAMEPOINT 0.5
#define INIT_KNNSORT true





//Setting of Basic RANSAC
#define INIT_MAXITERATION 5000
#define INIT_CONFIDENCE 99.99
#define INIT_MAXDISTANCE 3.0




//Setting of SVD_Eigen
#define EG_MAXITR 1000
#define EG_CNV1 1.0e-16
#define EG_CNV2 1.0e-36
#define EG_CNV3 1.0e-50


//Setting of Reinforcement learning
#define INIT_RL_ALPHA 0.1




#define progress_log(format, ...) \
	do { \
		if (IS_PROGRESS) {\
			printf(format, ##__VA_ARGS__); \
		} \
	} while(0)

#define error_log(format, ...) \
	do { \
		if (IS_ERROR) {\
			printf(format, ##__VA_ARGS__); \
		} \
	} while(0)


#define warn_log(format, ...) \
	do { \
		if (IS_WARNING) {\
			printf(format, ##__VA_ARGS__); \
		} \
	} while(0)

#define info_log(format, ...) \
	do { \
		if (IS_INFO) {\
			printf(format, ##__VA_ARGS__); \
		} \
	} while(0)

#define debug_log(format, ...) \
	do { \
		if (IS_DEBUG) {\
			printf(format, ##__VA_ARGS__); \
		} \
	} while(0)