#pragma once
#include "classes.h"
#include "enclasses.h"

void drawKeyPoints(source_data& srcd, destination_data& dstd);

cv::Scalar getColorLevel(int brt, int max);
void appendNewFeaturesInfo(std::vector<cv::KeyPoint>  newoPts, cv::Mat  newoFeatures,
                           featuremap_data& fmpd, bool forceOverRide, featureDetectionType fd);

int calcHumming(unsigned char x, unsigned char y);

double roundd(double num, int dpn); // decimal place number


void copy2features(featureDetectionType fd,
	cv::Mat& dst, cv::Mat& src, int dst_row, int dst_col, int src_row, int src_col);
