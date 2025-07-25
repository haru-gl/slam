#include "anms.h"
#include <algorithm>
#include <vector>
#include <numeric> // std::iota �̂��߂ɕK�v

void applyAnms(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int numBestPoints) {
    // �����_�����ڕW���ȉ��A�܂��͓����ʂƐ�������Ȃ��ꍇ�͉������Ȃ�
    if (keypoints.size() <= numBestPoints || keypoints.size() != descriptors.rows) {
        if (keypoints.size() != descriptors.rows) {
            printf("[ANMS Warning] Keypoints size (%zu) and descriptors rows (%d) do not match. Skipping.\n", keypoints.size(), descriptors.rows);
        }
        return;
    }

    // 1. �e�L�[�|�C���g�̗}�����a���v�Z
    std::vector<float> suppressionRadii(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        float minRadius = std::numeric_limits<float>::max();
        for (size_t j = 0; j < keypoints.size(); ++j) {
            // �����l�����傫��(�܂��͓������A���C���f�b�N�X��������)�L�[�|�C���gj�ɑ΂��Ă̂݋������v�Z
            if (keypoints[j].response > keypoints[i].response) {
                float dist = cv::norm(keypoints[i].pt - keypoints[j].pt);
                if (dist < minRadius) {
                    minRadius = dist;
                }
            }
        }
        suppressionRadii[i] = minRadius;
    }

    // 2. �L�[�|�C���g��}�����a�̍~���Ń\�[�g���邽�߂̃C���f�b�N�X���쐬
    std::vector<int> indices(keypoints.size());
    std::iota(indices.begin(), indices.end(), 0); // 0, 1, 2, ...

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return suppressionRadii[a] > suppressionRadii[b];
        });

    // 3. ���N�̃L�[�|�C���g�Ɠ����ʂ�V�����R���e�i�Ɋi�[
    std::vector<cv::KeyPoint> bestKeypoints;
    bestKeypoints.reserve(numBestPoints);

    // �����ʂ̃f�[�^�^�Ǝ��������ێ����ĐV����Mat������
    cv::Mat bestDescriptors = cv::Mat::zeros(numBestPoints, descriptors.cols, descriptors.type());

    for (int i = 0; i < numBestPoints; ++i) {
        int original_index = indices[i];
        bestKeypoints.push_back(keypoints[original_index]);
        // ���̃C���f�b�N�X���g���āA�Ή���������ʂ̍s���R�s�[
        descriptors.row(original_index).copyTo(bestDescriptors.row(i));
    }

    // 4. ���̃R���e�i���t�B���^�����O��̌��ʂŏ㏑��
    keypoints = bestKeypoints;
    descriptors = bestDescriptors;
}