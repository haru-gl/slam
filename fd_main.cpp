#include "fd_main.h"
#include "csv.h"
#include "anms.h" // ★ 追加

void featurepointdetection(featureDetectionType fd, source_data& srcd, destination_data& dstd)
{
	// === 特徴点検出 (既存のコード) ===
	switch (fd.ft) {
	case featureType::fAKAZE:
	{
		akaze fdt_akaze;
		fdt_akaze.featuredetection(fd, srcd, dstd);
	}
	break;
	case featureType::fSURF:
	{
		surf fdt_surf;
		fdt_surf.featuredetection(fd, srcd, dstd);
	}
	break;
	case featureType::fSIFT:
	{
		sift fdt_sift;
		fdt_sift.featuredetection(fd, srcd, dstd);
	}
	break;
	case featureType::fORB:
	{
		orb fdt_orb;
		fdt_orb.featuredetection(fd, srcd, dstd);
	}
	break;
	case featureType::fBRISK:
	{
		brisk fdt_brisk;
		fdt_brisk.featuredetection(fd, srcd, dstd);
	}
	break;
	case featureType::fPCA_float:
	case featureType::fPCA_uschar:
	case featureType::fPCA_16bin:
	{
		pca fdt_pca;
		fdt_pca.featuredetection(fd, srcd, dstd);
	}
	break;
	default:
		error_log("[%s]:[%s] そのfeatureTypeは実装されていません．\n", __FILE__, __FUNCTION__);
		exit(EXIT_FAILURE);
	}

	// ★ 変更: USE_ANMSがtrueのときだけANMS処理が実行されるようにする ★
#if USE_ANMS
	const int ANMS_POINTS = 500; // ANMSで残したい特徴点の数 //250 100
	const float ANMS_MULTIPLIER = 2.0f; // ANMSの中間ステップで保持する点の倍率

	// source_data (地図側) に適用 (初回実行時のみ)
	if (!srcd.oImage_dummy) {
		printf("Applying ANMS to source_data...\n");
		progress_log("  - Before ANMS (srcd): %zu points\n", srcd.oPts.size());
		applyAnms(srcd.oPts, srcd.oFeatures, ANMS_POINTS, ANMS_MULTIPLIER);
		progress_log("  - After ANMS (srcd): %zu points\n", srcd.oPts.size());
	}

	// destination_data (カメラ側) に適用
	printf("Applying ANMS to destination_data...\n");
	progress_log("  - Before ANMS (dstd): %zu points\n", dstd.oPts.size());
	applyAnms(dstd.oPts, dstd.oFeatures, ANMS_POINTS, ANMS_MULTIPLIER);
	progress_log("  - After ANMS (dstd): %zu points\n", dstd.oPts.size());
#else
	printf("ANMS is disabled. Skipping.\n");
#endif
	// ★ 変更ここまで ★

	if (CV_KEYPOINT_DSTD_TO_CSV) {
		CSVHandler csv("./featurepoints_dstd.csv");

		for (int i = 0; i < dstd.oPts.size(); i++) {
			csv.setValue(std::to_string(i), "x", std::to_string(dstd.oPts[i].pt.x));
			csv.setValue(std::to_string(i), "y", std::to_string(dstd.oPts[i].pt.y));
			csv.setValue(std::to_string(i), "size", std::to_string(dstd.oPts[i].size));
			csv.setValue(std::to_string(i), "angle", std::to_string(dstd.oPts[i].angle));
			csv.setValue(std::to_string(i), "response", std::to_string(dstd.oPts[i].response));
			csv.setValue(std::to_string(i), "octave", std::to_string(dstd.oPts[i].octave));
			csv.setValue(std::to_string(i), "class_id", std::to_string(dstd.oPts[i].class_id));
		}

		csv.saveChanges();
	}
}




void sort_oc(const std::vector<double> src,			// 入力
	std::vector<double>& order,		    // 出力1
	std::vector<int>& count, 				// 出力2
	sort_order_list sort_order)			// 設定項目（どの順番で並べるか）
{
	// 入力配列のもつ値が，小さい順に入る．（重複はしない！）ex: [1,3,4,9, 15,30, 99]など．
	order.clear();	order.push_back(DBL_MAX);
	count.clear();	count.push_back(INT_MAX);



	for (int i = 0; i < src.size(); i++) {
		auto it_o = order.begin();
		auto it_c = count.begin();
		for (int a = 0; a < order.size(); a++) {
			//printf("確認するのは：，%f < %f", pts_size[i], little_order[a]);

			if (src[i] < order[a]) {
				order.insert(it_o, src[i]);
				count.insert(it_c, 1);
				break;
			}
			if (src[i] == order[a]) {	// src[i] == order[a]
				//printf(" = \n");
				count[a] += 1;
				break;
			}
			//printf("等しくない，%f < %f\n", pts_size[i], little_order[a]);
			//printf(" > \n");
			++it_o;
			++it_c;
		}
	}



	order.pop_back(); // ここまでは昇順で処理されている．最初に入れておいたDBL_MAXの値を削除する．昇順なので，大きな値は末尾に入る．
	count.pop_back();


	if (sort_order == sort_order_list::small2large) {
		// 昇順
		// なにもしない．すでに昇順で処理している．
	}
	else if (sort_order == sort_order_list::large2small) {
		// 降順
		// 並び順を反転させればよい．
		std::reverse(order.begin(), order.end());
		std::reverse(count.begin(), count.end());
	}






	//for (int i = 0; i < order.size(); i++) {
	//	printf("order[%3d], %20.10f, %d\n", i, order[i], count[i]);
	//}
}



//// KeyPointのsize，responseによる絞り込みのためのマスク配列を生成する関数
void sq_make_list(sr sr, double th, int round_dpn,
	std::vector<cv::KeyPoint> oPts, cv::Mat oFeatures, std::string name, featureDetectionType fd,
	std::vector<bool>& mask)
{

	switch (sr)
	{
	case sr::size:
		std::cout << "----- < sq_* 絞り込みモード [size]  > -----" << std::endl;
		break;
	case sr::response:
		std::cout << "----- < sq_* 絞り込みモード [response]  > -----" << std::endl;
		break;
	default:
		break;
	}

	// 割合で絞り込むモードの場合のエラー検出
	if (fd.st == sqType::sqPERCENT) {
		if (th == 1.0) return;	// 絞り込みを行わないのと同義（1.0 === 全通過のため）
		else if (th < 0 || 1.0 < th) {
			printf("[fd_main.cpp] Error! 「sqType::sqPERCENT」モードの場合，SQ_{SIZE, RESPONSE}_THの値は次の範囲で有効です： 0.0 < SQ_{SIZE, RESPONSE}_TH <= 1.0\n");
			exit(EXIT_FAILURE);
		}
	}
	// 個数で絞り込むモードの場合のエラー検出
	if (fd.st == sqType::sqNUM) {
		if (th <= 1.0) {		// 閾値が1.0以下の場合 --> 設定ミスの可能性が大きいのでエラーで終了
			printf("[fd_main.cpp] Error! SQ_{SIZE, RESPONSE}_THの設定値が小さすぎます．\n");
			exit(EXIT_FAILURE);
		}
	}


	int  start_size = oPts.size();	// 削減前の特徴点数
	int  end_size = -1;				// 削減後の特徴点数




	printf("事前の数: %s.oPts     ：%5d\n", name.c_str(), (int)oPts.size());
	printf("事前の数: %s.oFeatures：%5d\n", name.c_str(), oFeatures.rows);



	// キーポイントの持つsize, responseの値を調査する．
	std::vector<double> pts(oPts.size());
	std::vector<double> little_order;	// size,responseの値が，小さい順に入る（重複はしない！）．ex: [1,3,4,9, 15,30, 99]など．
	std::vector<int>	little_count;	// そのsize,responseが，それぞれ合計で何個あるかが入る．

	switch (sr)
	{
	case sr::size:
		for (int i = 0; i < oPts.size(); i++) {
			double tmp = oPts[i].size;
			tmp = roundd(tmp, round_dpn);	// 小数点第 {SQ_SIZE_ROUND_DPN} 位以下は四捨五入し，そこまで同じ値であれば同一とみなす．
			pts[i] = tmp;
		}
		break;
	case sr::response:
		for (int i = 0; i < oPts.size(); i++) {
			double tmp = oPts[i].response;
			tmp = roundd(tmp, round_dpn);	// 小数点第 {SQ_SIZE_ROUND_DPN} 位以下は四捨五入し，そこまで同じ値であれば同一とみなす．
			pts[i] = tmp;
		}
		break;
	default:
		break;
	}



	double max_size, min_size;
	switch (sr == sr::size ? fd.sq_size_sl : fd.sq_response_sl)
	{
	case sl::small_is_better:
		sort_oc(pts, little_order, little_count, sort_order_list::small2large);
		max_size = little_order[little_order.size() - 1];
		min_size = little_order[0];
		break;
	case sl::large_is_better:
		sort_oc(pts, little_order, little_count, sort_order_list::large2small);
		max_size = little_order[0];
		min_size = little_order[little_order.size() - 1];
		break;
	default:
		printf("[fd_main.cpp line:186] sq_{size, response}_slに関して，その分岐は定義されていません．\n");
		exit(EXIT_FAILURE);
	}


	//double max_size = *std::max_element(pts_size.begin(), pts_size.end());		// #include <algorithm>
	//double min_size = *std::min_element(pts_size.begin(), pts_size.end());		// #include <algorithm>




	switch (sr)
	{
	case sr::size:
		printf("キーポイントの持つsizeの最大値：%f\n", max_size);
		printf("キーポイントの持つsizeの最小値：%f\n", min_size);
		break;
	case sr::response:
		printf("キーポイントの持つresponseの最大値：%f\n", max_size);
		printf("キーポイントの持つresponseの最小値：%f\n", min_size);
		break;
	default:
		break;
	}




	double use_max = DBL_MAX;
	double use_min = DBL_MAX;	// 利用するキーポイントが持つ，最小のsizeの値を決定する．(仮決め．今度アルゴリズムを考える．)
	// AKAZEの場合は，最小のものたちだけを利用すれば最高精度のように見えるが，ＳＵＲＦの場合は，dstd側から<<< 1点でも >>>>>減るだけで，不具合が発生する．
	// 地図側に多数の特徴点が追加されてしまい，mOLDERが機能しない．というか，ＲＡＮＳＡＣのマッチングがおかしくなる．

	double edge1 = DBL_MAX;
	double edge2 = DBL_MAX;
	int sum_num = 0;

	switch (fd.st)
	{
	case sqType::sqPERCENT:
		// 配列の中で指定された【割合】を達成する最低限の要素数を，「index 0から順に見て」取り出すには，どこまで進めばよいかを判定する．

		// 絞り込みの端点の値は　【little_order[0]】と【下のbreak時の little_order[i]】に挟まれる間のものたち．
		edge1 = little_order[0];
		for (int i = 0; i < little_order.size(); i++) {
			sum_num += little_count[i];
			if ((double)sum_num / (double)oPts.size() > th) {
				printf("%f * 100%%超過．終了 [%3d]\n", th, i);
				edge2 = little_order[i];
				break;
			}
			//printf("%3f < 0.1 続行...\n", (double)sum_num / srcdst.oPts.size());
		}
		break;
	case sqType::sqNUM:
		// 配列の中で指定された【個数】を達成する最低限の要素数を，「index 0から順に見て」取り出すには，どこまで進めばよいかを判定する．

		// 絞り込みの端点の値は　【little_order[0]】と【下のbreak時の little_order[i]】に挟まれる間のものたち．
		edge1 = little_order[0];
		for (int i = 0; i < little_order.size(); i++) {
			sum_num += little_count[i];
			if (sum_num > th) {
				printf("%f 個以上に到達．終了 [%3d]\n", th, i);
				edge2 = little_order[i];
				break;
			}
			//printf("%3f < 0.1 続行...\n", (double)sum_num / srcdst.oPts.size());
		}
		break;
	default:
		printf("[fd_main.cpp] Error! 不正なインデックス：fd.st\n");
		exit(EXIT_FAILURE);
	}

	// edge1, edge2のうち，小さい方がuse_min, 大きい方がuse_max．
	// （もしサイズが両方同じなら，別に入れ替えなくてもそのままでOK．)
	if (edge1 < edge2) {
		use_min = edge1;
		use_max = edge2;
	}
	else {
		use_min = edge2;
		use_max = edge1;
	}


	switch (sr)
	{
	case sr::size:
		printf("使用するsizeの最小値：%f\n", use_min);
		printf("使用するsizeの最大値：%f\n", use_max);
		break;
	case sr::response:
		printf("使用するresponseの最小値：%f\n", use_min);
		printf("使用するresponseの最大値：%f\n", use_max);
		break;
	default:
		break;
	}



	int itr = 0;
	for (int i = 0; i < oPts.size(); i++) {
		double kp_size;
		switch (sr)
		{
		case sr::size:
			kp_size = oPts[i].size;
			break;
		case sr::response:
			kp_size = oPts[i].response;
			break;
		default:
			break;
		}
		kp_size = roundd(kp_size, round_dpn);

		if (use_min <= kp_size && kp_size <= use_max) {
			// 残留させるべき特徴点なので「残留させるべき」を意味するtrueのままから変更しない．
			itr++;
		}
		else
		{
			mask[i] = false;	// 残留させるべき特徴点ではないので，falseへ変更．
		}

	}
	end_size = itr;

	double sqed_rt = (double)end_size / (double)start_size * 100; // squed rate
	printf("事後の数: %s.oPts/oFeatures：%5d (事前比：%5f %%)\n", name.c_str(), end_size, sqed_rt);
}



void sq(std::vector<cv::KeyPoint>& oPts, cv::Mat& oFeatures, std::string name, featureDetectionType fd,
	std::vector<bool> mask)
{
	std::cout << "----- 絞り込み開始 -----" << std::endl;
	int  start_size = oPts.size();	// 削減前の特徴点数
	int  end_size = -1;				// 削減後の特徴点数

	std::vector<cv::KeyPoint> newoPts(oPts.size());
	cv::Mat newoFeatures = oFeatures.clone();		// 今仮に書き込んでいるが，それはサイズを合わせるため．newoFeaturesの中身は現時点で全消去可であり，のちに上書きされていく．




	int itr = 0;
	for (int i = 0; i < oPts.size(); i++) {
		if (mask[i]) {  // この中には [ture]/[false] が格納されている． trueはコピーせよ，の意味．
			//　キーポイントのコピー
			newoPts[itr] = oPts[i];

			// 特徴量のコピー
			for (int col = 0; col < oFeatures.cols; col++) {
				copy2features(fd, newoFeatures, oFeatures, itr, col, i, col); //newoFeatures.at<unsigned char>(itr, col) = srcdst.oFeatures.at<unsigned char>(i, col);
			}
			itr++;
		}
	}
	end_size = itr;

	oPts.clear();
	newoPts.resize(end_size);
	//newoPts.shrink_to_fit();	// 機能してる？
	oPts = newoPts;

	oFeatures.resize(0);
	newoFeatures.resize(end_size);	// 機能してる？
	oFeatures = newoFeatures.clone();

	double sqed_rt = (double)end_size / (double)start_size * 100; // squed rate
	printf("絞り込み完了：事後の数: %s.oPts     ：%5d --> %5d\n", name.c_str(), start_size, (int)oPts.size());
	printf("絞り込み完了：事後の数: %s.oFeatures：%5d --> %5d (事前比：%5f %%)\n", name.c_str(), start_size, oFeatures.rows, sqed_rt);
}
