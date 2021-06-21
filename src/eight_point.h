//
// Created by gabriel on 21.06.21.
//

#ifndef STEREO_RECONSTRUCTION_EIGHT_POINT_H
#define STEREO_RECONSTRUCTION_EIGHT_POINT_H

#include "Eigen.h"
void eightPointAlgorithm(const std::vector<cv::KeyPoint>& keypointsLeft,
                         const std::vector<cv::KeyPoint>& keypointsRight,
                         const std::vector<cv::DMatch>& matches,
                         const Matrix3f& cameraLeft,
                         const Matrix3f& cameraRight,
                         Matrix4f& pose);

#endif //STEREO_RECONSTRUCTION_EIGHT_POINT_H
