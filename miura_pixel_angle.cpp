#include "pca.h"


template <typename T>
T cal_dist(cv::Point2f pos1, cv::Point2f pos2)
{
    return std::sqrt(std::pow(pos1.x - pos2.x, 2) + std::pow(pos1.y - pos2.y, 2));
}

void pca::create_features_miura_pixel_angle(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpoints, cv::Mat& oFeatures, const int dim)
{
    // �����ʋL�q
    oFeatures = cv::Mat::zeros(kpoints.size(), dim, CV_32F);


    std::vector<std::vector<char>> pmsk = {
        {0,0,0,0,0,1,1,1,1,1,0,0,0,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        {0,0,1,1,1,1,1,1,1,1,1,1,1,0,0},
        {0,0,0,1,1,1,1,1,1,1,1,1,0,0,0},
        {0,0,0,0,0,1,1,1,1,1,0,0,0,0,0}
    };

#pragma imp parallel for
    for (int i = 0; i < kpoints.size(); i++) {
        cv::Point2i pos_idx = cv::Point2i(std::round(kpoints[i].pt.x + 0), std::round(kpoints[i].pt.y + 0));

        // �߂��̉�f�̋P�x�l���擾		
        int sum = 0;
        unsigned char v;
        int cnt = 0;
        int range = 15, st = -range / 2, ed = range / 2, circle_diameter = range / 2;

        // ��ʒu
        int divn = 2; // ������
        float theta = (kpoints[i].angle + 45.0) * CV_PI / 180; // deg����rad��
        std::vector<cv::Point2f> hanbetsu_p;
        float x = (float)range / 4, y = 0;
        hanbetsu_p.push_back(cv::Point2f(cos(theta) * x - sin(theta) * y,
            sin(theta) * x + cos(theta) * y));
        //hanbetsu_p.push_back(cv::Point2f(-hanbetsu_p[0].x,  hanbetsu_p[0].y));
        //hanbetsu_p.push_back(cv::Point2f( hanbetsu_p[0].x, -hanbetsu_p[0].y));
        hanbetsu_p.push_back(cv::Point2f(-hanbetsu_p[0].x, -hanbetsu_p[0].y));
        //for (auto v : hanbetsu_p) std::cout << v.x << " " << v.y << "\n";


        std::vector<std::vector<float>> enums(divn);
        for (int r = st; r <= ed; r++) {
            for (int c = st; c <= ed; c++) {
                // �~���ɓ����Ă����
                if (pmsk[r + range / 2][c + range / 2] == 1) {//if (std::sqrt(c*c + r*r) <= circle_diameter) {//
                    // N�������������̂ǂ̗̈�ɑ�����̂����v�Z�D                    
                    int min_idx = -1; float min_dist = INT_MAX;
                    for (int j = 0; j < hanbetsu_p.size(); j++) {
                        if (cal_dist<float>(hanbetsu_p[j], cv::Point2f(c, r)) < min_dist) {
                            min_idx = j; min_dist = cal_dist<float>(hanbetsu_p[j], cv::Point2f(c, r));
                        }
                    }

                    // ���̑�����G���A�ɒǉ��D
                    v = image.at<unsigned char>(pos_idx.y + r, pos_idx.x + c);
                    enums[min_idx].push_back((float)v);
                    sum += (int)v;
                    cnt++;
                }
            }
        }
        printf("��f�̑傫���F%d\n", cnt);


        // ���K��        
        //std::vector<std::vector<float>> enums_float;
        //int idx = 0;
        //for (auto en : enums) {
        //    for (auto v : en) {
        //        enums_float[idx].push_back((float)v);
        //        idx++;
        //    }
        //}



        //// ========== �W���΍� ==========
        //for (auto v : enums) nd_enums.push_back(((float)v - ((float)sum/cnt)) / calc_stdev_p(enums_float));
        //// �q�X�g�O������
        //std::vector<float> feat_stdev;
        //create_histgram(nd_enums, feat_stdev, 32, find_max(nd_enums), find_min(nd_enums));
        //// �x�N�g���Ƃ��Ċi�[
        //int ptr = 0;
        //for (auto v : feat_stdev) oFeatures.at<float>(i, ptr++) = v;


        // ========== min-max���K�� ==========
        std::vector<std::vector<float>> nd_enums(divn);
        int idx = 0;
        for (auto en : enums) {
            for (auto v : en) {
                nd_enums[idx].push_back(((float)v - (float)find_min(enums[idx])) / ((float)find_max(enums[idx]) - (float)find_min(enums[idx])) * 255.0);
            }
            idx++;
            //printf("%d\n", idx-1);
        }
        // ���̕��z���m�F
        printf("[0]%d, [1]%d, [2]%d, [3]%d\n", nd_enums[0].size(), nd_enums[1].size(), nd_enums[2].size(), nd_enums[3].size());

        //

        // �q�X�g�O������
        printf("�q�X�g�O�����X�^�[�g\n");
        int ptr = 0;
        for (int j = 0; j < nd_enums.size(); j++) {
            printf("j=%d\n", j);

            std::vector<float> feat_minmax;
            create_histgram<float>(nd_enums[j], feat_minmax, 8, 255, 0);

            //int mx = 0;
            //for (auto v : feat_minmax) {
            //    if (mx < v) mx = (int)v;
            //}
            //std::cout << "max:" << mx << "\n";


            for (auto v : feat_minmax) oFeatures.at<float>(i, ptr++) = v;
            printf("ptr%d ", ptr);
            //oFeatures.at<unsigned char>(i, 0) = 0;
            //oFeatures.at<unsigned char>(i, oFeatures.cols-1) = 0;
        }
    }
}
