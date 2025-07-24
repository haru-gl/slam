#pragma once
#include <iostream>


enum class featureType { fAKAZE, fKAZE, fSURF, fSIFT, fBRISK, fORB, fPCA_float, fPCA_uschar, fPCA_16bin, fDAMMY };
enum class ransacType { rNORMAL, rTD0 };

//Type of sq
enum class sqType { 
	sqPERCENT,	// 残数を割合値で指定（個数可変）
	sqNUM		// 残数を絶対数で指定（個数指定）
};

//Type of sqのsl
enum class sl {			// Small or Largeの略
	small_is_better,	// sq_size, sq_responseにおいて，値が小さいほど良いとする．
	large_is_better		// sq_size, sq_responseにおいて，値が大きいほど良いとする．
};


enum class sr {
	size,		// sizeによる絞り込み
	response	// responseによる絞り込み
};
// ※これ以外の値で絞り込みたい場合は， fd_main.cppの190行目付近を書き換える必要がある．「switch (sr==sr::size? fd.sq_size_sl: fd.sq_response_sl)」の箇所．


enum class sqOrderType {
	size_sq_response_sq,	// サイズで絞り込み配列を作成して絞り込み実行後，さらにレスポンスで絞り込み配列を作成して絞り込み実行
	response_sq_size_sq,	// size_sq_response_sqの逆順
	size_response_sq		// サイズ・レスポンスそれぞれで絞り込み配列を作成し，AND積を取る形で絞り込みを実行
};


//Type of kNN Method
enum class knnType { kNORMAL, kDTSFP, kNNFIXN, kNNFIXNDTSFP, kDAMMY, kFLANN_LSH };


//Action Mode of RANSAC
enum class ransacMode { dNORMAL, dSTDDEV, dHAMPLEI, dDAMMY };
// sacType=sNORMAL:    ransacMode=dNORMAL
// sacType=sRANSACNOR: ransacMode=dNORMAL
// sacType=sFILTERS1:  ransacMode=dSTDDEV
// sacType=sFILTERH1:  ransacMode=dHAMPLEI
// sacType=sFILTERS2:  ransacMode=dSTDDEV
// sacType=sFILTERS2:  ransacMode=dHAMPLEI
// sacType=sFILTERSH:  ransacMode=dSTDDEV and dHAMPLEI
// sacType=sFILTERHS:  ransacMode=dHAMPLEI and dSTDDEV
ransacMode begin(ransacMode);
ransacMode end(ransacMode);
ransacMode operator*(ransacMode ft);
ransacMode operator++(ransacMode& ft);
std::ostream& operator<<(std::ostream& os, ransacMode ft);



//Type of Transformation Matrix
enum class matchingType { mSIMILARITY, mAFFINE, mPROJECTIVE, mPROJECTIVE3, mPROJECTIVE_EV, mDAMMY };
matchingType begin(matchingType);
matchingType end(matchingType);
matchingType operator*(matchingType ft);
matchingType operator++(matchingType& ft);
std::ostream& operator<<(std::ostream& os, matchingType ft);



//Type of Estimation Method for Transformation Matrix
enum class matrixcalType { cSVD, cGAUSSNEWTON, cSVD_EIGEN, cGAUSSNEWTON_EIGEN, cTAUBIN, cVBAYES, cDAMMY };
matrixcalType begin(matrixcalType);
matrixcalType end(matrixcalType);
matrixcalType operator*(matrixcalType ft);
matrixcalType operator++(matrixcalType& ft);
std::ostream& operator<<(std::ostream& os, matrixcalType ft);



// 特徴点マップ生成時の古い対応点の扱い方
enum class MappingType {
	mALL,		// 何もせず全て追加
	mSIMILAR,	// 特徴量のハミング距離が似ている場合のみ追加（範囲は別指定）
	mOLDER,		// 古い点情報があるなら、その箇所には新しい点は追加しない
	mNEWER		// 古い点情報があっても、新しい点が来るならその情報で上書きする
}; 			

////////////////MappingType& operator<<(std::ostream& os, MappingType fm)
////////////////{
////////////////	//fm = static_cast<MappingType>(fm);
////////////////	os << fm.mt;
////////////////	return fm;
////////////////};


// sac_main利用時に転記
class posestType {
public:
	matchingType mt;
	ransacMode rm;
	bool ndon;
	matrixcalType ct;
	bool use_OpenCV_findHomography;
};




class featureDetectionType {
public:
	featureType ft;
	bool sq_size;
	bool sq_response;

	sqType st;
	sl sq_size_sl;
	sl sq_response_sl; 

	sqOrderType sq_order;
};