#include <random>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "enclasses.h"
#include "tmatrix.h"
#include "parameters.h"

Eigen::VectorXd projmatrixestimation(const matchingType mt, std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2);


void get_ave_stddev(const std::vector<double>& exy, double& nave, double& nstddv)
{
	double ave = 0.0, dev = 0.0;
	for (size_t j = 0; j < exy.size(); j++) {
		ave += exy[j];
		dev += exy[j] * exy[j];
	}
	ave /= (double)exy.size();
	dev /= (double)exy.size();
	dev -= ave * ave;
	nave = ave; nstddv = sqrt(dev);
}


double get_median(std::vector<double>& exy)
{
	double med;
	std::sort(exy.begin(), exy.end());
	size_t ecentor = exy.size() / 2;
	if (exy.size() % 2 == 1)
		med = exy[ecentor];
	else
		med = (exy[ecentor] + exy[ecentor + 1]) / 2.0;
	return med;
}

cv::Point2d transform2d(const cv::Point2d pts, const cv::Mat& tform)
{
	cv::Point2d dstPts;
	cv::Mat ptsMat(3, 1, CV_64FC1);
	ptsMat.at<double>(0, 0) = pts.x;
	ptsMat.at<double>(1, 0) = pts.y;
	ptsMat.at<double>(2, 0) = 1.0;
	cv::Mat dstPtsMat = tform * ptsMat;
	dstPts.x = dstPtsMat.at<double>(0, 0) / dstPtsMat.at<double>(2, 0);
	dstPts.y = dstPtsMat.at<double>(1, 0) / dstPtsMat.at<double>(2, 0);
	return dstPts;
}

cv::Point2f transform2f(const cv::Point2f pts, const cv::Mat& tform)
{
	cv::Point2f dstPts;
	cv::Mat ptsMat(3, 1, CV_64FC1);
	ptsMat.at<double>(0, 0) = pts.x;
	ptsMat.at<double>(1, 0) = pts.y;
	ptsMat.at<double>(2, 0) = 1.0;
	cv::Mat dstPtsMat = tform * ptsMat;
	dstPts.x = dstPtsMat.at<double>(0, 0) / dstPtsMat.at<double>(2, 0);
	dstPts.y = dstPtsMat.at<double>(1, 0) / dstPtsMat.at<double>(2, 0);
	return dstPts;
}

int get_minGP(const matchingType mt) //matchingType mt)
{
	int val = 0;
	switch (mt) {
	case matchingType::mSIMILARITY:
		val = 2;
		break;
	case matchingType::mAFFINE:
		val = 3;
		break;
	case matchingType::mPROJECTIVE:
	case matchingType::mPROJECTIVE3:
	case matchingType::mPROJECTIVE_EV:
		val = 4;
		break;
	}
	return val;
}


std::vector<size_t> randperm(size_t size, size_t returnNum)
{
	std::random_device rd;			//必要： `#include <random>`
	std::mt19937 rng(rd());
	std::vector<size_t> idxArray(size);
	std::vector<size_t> returnArray(returnNum);
	for (size_t i = 0; i < size; i++)
		idxArray[i] = i;
	std::shuffle(idxArray.begin(), idxArray.end(), rng);
	for (size_t i = 0; i < returnNum; i++)
		returnArray[i] = idxArray[i];
	return returnArray;
}



bool checkFunc(const cv::Mat& tform)
{
	bool isInfinite = false;
	for (int i = 0; i < tform.rows; i++)
		for (int j = 0; j < tform.cols; j++)
			if (isinf(tform.at<double>(i, j)) || isnan(tform.at<double>(i, j)))
				isInfinite = true;
	return isInfinite;
}



cv::Mat computematrix(const matchingType mt, std::vector<cv::Point2d>& tgpts1, std::vector<cv::Point2d>& cmpts2)
{
	cv::Mat tform(3, 3, CV_64FC1);
	switch (mt) {
	case matchingType::mSIMILARITY:
	case matchingType::mAFFINE:
	case matchingType::mPROJECTIVE:
	case matchingType::mPROJECTIVE3:
	{
		cv::Mat jxn = set_jacobian(mt, tgpts1, cmpts2);
		cv::Mat u;
		cv::SVD::solveZ(jxn, u);
		tform = cnv_vc2mt(mt, u);
	}
	break;
	case matchingType::mPROJECTIVE_EV:
	{
		projmatrixestimation(mt, tgpts1, cmpts2);

	}
	}
	return tform;
}

cv::Mat computematrix_byEigen(const matchingType mt, std::vector<cv::Point2d>& tgpts1, std::vector<cv::Point2d>& cmpts2)
{
	cv::Mat jxn = set_jacobian(mt, tgpts1, cmpts2);
	cv::Mat u;
	int stat = solveZ_eigen(jxn, u);
	if (stat != 0) std::cout << stat << std::endl;
	cv::Mat tform(3, 3, CV_64FC1);
	tform = cnv_vc2mt(mt, u);
	return tform;
}




int solveZ_eigen(const cv::Mat& constraints, cv::Mat& u)//, double *err)
{
	Eigen::MatrixXd m(constraints.rows, constraints.cols);
	for (int i = 0; i < constraints.rows; i++)
		for (int j = 0; j < constraints.cols; j++)
			m(i, j) = constraints.at<double>(i, j);

	Eigen::MatrixXd ata = m.transpose() * m;
	Eigen::VectorXd vv = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(ata).eigenvectors().col(0);

	Eigen::MatrixXd ai = ata.inverse();
	Eigen::VectorXd vh(constraints.cols), vhp(constraints.cols);
	double rambda;
	vhp = vv;
	double er, per = 10000.0;
	int stat = 2;
	for (int i = 0; i < EG_MAXITR; i++) {
		vh = ai * vhp;
		rambda = vh.norm();
		vh /= rambda;
		if (1.0 / rambda < EG_CNV1) {
			stat = 0;
			break;
		}
		Eigen::VectorXd ss = vh - vhp;
		er = ss.norm();
		if (er < EG_CNV2) {//				printf("convergence%d:", i); 
			stat = 1;
			break;
		}
		if (fabs(per - er) < EG_CNV3) {//	printf("Without error-improvement%d:",i); 
			stat = 2;
			break;
		}
		vhp = vh; per = er;
	}
	u = cv::Mat::zeros(constraints.cols, 1, CV_64FC1);
	for (int i = 0; i < constraints.cols; i++)
		u.at<double>(i, 0) = vh(i);
	return stat;
}





cv::Mat set_jacobian(const matchingType mt, std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2)
{
	int cl = (int)p1.size();
	int rw = get_xcol(mt);
	int ct = get_yraw(mt);
	cv::Mat jxn = cv::Mat::zeros(ct * cl, rw, CV_64FC1);
	switch (mt) {
	case matchingType::mSIMILARITY:
		for (int i = 0; i < cl; i++) {
			int idx1 = 2 * i;
			int idx2 = 2 * i + 1;
			jxn.at<double>(idx1, 0) = p1[i].x;
			jxn.at<double>(idx1, 1) = p1[i].y;
			jxn.at<double>(idx1, 2) = 1.0;
			jxn.at<double>(idx1, 3) = 0.0;
			jxn.at<double>(idx1, 4) = -p2[i].x;

			jxn.at<double>(idx2, 0) = -p1[i].y;
			jxn.at<double>(idx2, 1) = p1[i].x;
			jxn.at<double>(idx2, 2) = 0.0;
			jxn.at<double>(idx2, 3) = -1.0;
			jxn.at<double>(idx2, 4) = p2[i].y;
		}
		break;
	case matchingType::mAFFINE:
		for (int i = 0; i < cl; i++) {
			int idx1 = 2 * i;
			int idx2 = 2 * i + 1;
			jxn.at<double>(idx1, 0) = -p1[i].x;
			jxn.at<double>(idx1, 1) = -p1[i].y;
			jxn.at<double>(idx1, 2) = -1.0;
			jxn.at<double>(idx1, 3) = 0.0;
			jxn.at<double>(idx1, 4) = 0.0;
			jxn.at<double>(idx1, 5) = 0.0;
			jxn.at<double>(idx1, 6) = p2[i].x;

			jxn.at<double>(idx2, 0) = 0.0;
			jxn.at<double>(idx2, 1) = 0.0;
			jxn.at<double>(idx2, 2) = 0.0;
			jxn.at<double>(idx2, 3) = -p1[i].x;
			jxn.at<double>(idx2, 4) = -p1[i].y;
			jxn.at<double>(idx2, 5) = -1.0;
			jxn.at<double>(idx2, 6) = p2[i].y;
		}
		break;
	case matchingType::mPROJECTIVE:
		for (int i = 0; i < cl; i++) {
			int idx1 = 2 * i;
			int idx2 = 2 * i + 1;
			jxn.at<double>(idx1, 0) = p1[i].x;
			jxn.at<double>(idx1, 1) = p1[i].y;
			jxn.at<double>(idx1, 2) = 1.0;
			jxn.at<double>(idx1, 3) = 0.0;
			jxn.at<double>(idx1, 4) = 0.0;
			jxn.at<double>(idx1, 5) = 0.0;
			jxn.at<double>(idx1, 6) = -p1[i].x * p2[i].x;
			jxn.at<double>(idx1, 7) = -p1[i].y * p2[i].x;
			jxn.at<double>(idx1, 8) = -p2[i].x;

			jxn.at<double>(idx2, 0) = 0.0;
			jxn.at<double>(idx2, 1) = 0.0;
			jxn.at<double>(idx2, 2) = 0.0;
			jxn.at<double>(idx2, 3) = p1[i].x;
			jxn.at<double>(idx2, 4) = p1[i].y;
			jxn.at<double>(idx2, 5) = 1.0;
			jxn.at<double>(idx2, 6) = -p1[i].x * p2[i].y;
			jxn.at<double>(idx2, 7) = -p1[i].y * p2[i].y;
			jxn.at<double>(idx2, 8) = -p2[i].y;
		}
		break;
	case matchingType::mPROJECTIVE3:
		for (int i = 0; i < cl; i++) {
			int idx1 = 3 * i;
			int idx2 = 3 * i + 1;
			int idx3 = 3 * i + 2;
			jxn.at<double>(idx1, 0) = p1[i].x;
			jxn.at<double>(idx1, 1) = p1[i].y;
			jxn.at<double>(idx1, 2) = 1.0;
			jxn.at<double>(idx1, 3) = 0.0;
			jxn.at<double>(idx1, 4) = 0.0;
			jxn.at<double>(idx1, 5) = 0.0;
			jxn.at<double>(idx1, 6) = -p1[i].x * p2[i].x;
			jxn.at<double>(idx1, 7) = -p1[i].y * p2[i].x;
			jxn.at<double>(idx1, 8) = -p2[i].x;

			jxn.at<double>(idx2, 0) = 0.0;
			jxn.at<double>(idx2, 1) = 0.0;
			jxn.at<double>(idx2, 2) = 0.0;
			jxn.at<double>(idx2, 3) = p1[i].x;
			jxn.at<double>(idx2, 4) = p1[i].y;
			jxn.at<double>(idx2, 5) = 1.0;
			jxn.at<double>(idx2, 6) = -p1[i].x * p2[i].y;
			jxn.at<double>(idx2, 7) = -p1[i].y * p2[i].y;
			jxn.at<double>(idx2, 8) = -p2[i].y;

			jxn.at<double>(idx3, 0) = -p1[i].x * p2[i].y;
			jxn.at<double>(idx3, 1) = -p1[i].y * p2[i].y;
			jxn.at<double>(idx3, 2) = -p2[i].y;
			jxn.at<double>(idx3, 3) = p1[i].x * p2[i].x;
			jxn.at<double>(idx3, 4) = p1[i].y * p2[i].x;
			jxn.at<double>(idx3, 5) = p2[i].x;
			jxn.at<double>(idx3, 6) = 0.0;
			jxn.at<double>(idx3, 7) = 0.0;
			jxn.at<double>(idx3, 8) = 0.0;
		}
		break;
	}
	return jxn;
}


int get_xcol(const matchingType mt)
{
	int numcol = 5;
	switch (mt) {
	case matchingType::mSIMILARITY:
		numcol = 5; break;
	case matchingType::mAFFINE:
		numcol = 7; break;
	case matchingType::mPROJECTIVE:
	case matchingType::mPROJECTIVE3:
		numcol = 9; break;
	}
	return numcol;
}

int get_yraw(const matchingType mt)
{
	int numraw = 2;
	switch (mt) {
	case matchingType::mSIMILARITY:
	case matchingType::mAFFINE:
	case matchingType::mPROJECTIVE:
		numraw = 2; break;
	case matchingType::mPROJECTIVE3:
		numraw = 3; break;
	}
	return numraw;
}



cv::Mat cnv_vc2mt(const matchingType mt, const cv::Mat& xn)
{
	cv::Mat tform(3, 3, CV_64FC1);
	switch (mt) {
	case matchingType::mSIMILARITY:
		tform.at<double>(0, 0) = xn.at<double>(0, 0);	tform.at<double>(0, 1) = xn.at<double>(1, 0);	tform.at<double>(0, 2) = xn.at<double>(2, 0);
		tform.at<double>(1, 0) = -xn.at<double>(1, 0);	tform.at<double>(1, 1) = xn.at<double>(0, 0);	tform.at<double>(1, 2) = xn.at<double>(3, 0);
		tform.at<double>(2, 0) = 0.0;					tform.at<double>(2, 1) = 0.0;					tform.at<double>(2, 2) = xn.at<double>(4, 0);
		break;
	case matchingType::mAFFINE:
		tform.at<double>(0, 0) = xn.at<double>(0, 0);	tform.at<double>(0, 1) = xn.at<double>(1, 0);	tform.at<double>(0, 2) = xn.at<double>(2, 0);
		tform.at<double>(1, 0) = xn.at<double>(3, 0);	tform.at<double>(1, 1) = xn.at<double>(4, 0);	tform.at<double>(1, 2) = xn.at<double>(5, 0);
		tform.at<double>(2, 0) = 0.0;					tform.at<double>(2, 1) = 0.0;					tform.at<double>(2, 2) = xn.at<double>(6, 0);
		break;
	case matchingType::mPROJECTIVE:
	case matchingType::mPROJECTIVE3:
		tform.at<double>(0, 0) = xn.at<double>(0, 0);	tform.at<double>(0, 1) = xn.at<double>(1, 0);	tform.at<double>(0, 2) = xn.at<double>(2, 0);
		tform.at<double>(1, 0) = xn.at<double>(3, 0);	tform.at<double>(1, 1) = xn.at<double>(4, 0);	tform.at<double>(1, 2) = xn.at<double>(5, 0);
		tform.at<double>(2, 0) = xn.at<double>(6, 0); 	tform.at<double>(2, 1) = xn.at<double>(7, 0);	tform.at<double>(2, 2) = xn.at<double>(8, 0);
		break;
	}
	double xn22 = tform.at<double>(2, 2);
	tform /= xn22;
	return tform;

}



