#pragma once
#include "classes.h"
#include "csv.h"
#include "nameof_enum.h"

void save_statistics(int i, featuremap_data fmpd, analysis_results rst);
void save_finals(featuremap_data fmpd, analysis_results rst, std::string DistNum, featureType ft, ransacType rt);