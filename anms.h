#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Adaptive Non-Maximal Suppression (ANMS) ��K�p���܂��B
 * �^����ꂽ�L�[�|�C���g���摜�S�̂ɋψ�ɕ��z����悤�ɊԈ����A
 * �Ή���������ʃf�B�X�N���v�^�����l�Ƀt�B���^�����O���܂��B
 * @param keypoints [in/out] �t�B���^�����O�Ώۂ̃L�[�|�C���g�B������A�Ԉ����ꂽ���ʂŏ㏑������܂��B
 * @param descriptors [in/out] �t�B���^�����O�Ώۂ̓����ʁB������A�Ԉ����ꂽ���ʂŏ㏑������܂��B
 * @param numBestPoints [in] �ێ��������L�[�|�C���g�̍ő吔�B
 */
void applyAnms(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int numBestPoints);