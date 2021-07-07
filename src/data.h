//
// Created by gabriel on 07.07.21.
//

#ifndef STEREO_RECONSTRUCTION_DATA_H
#define STEREO_RECONSTRUCTION_DATA_H

#include "Eigen.h"
#include <opencv4/opencv2/opencv.hpp>


class Data {
/**
 * Structure of a training or test scenario
 */
public:
    Data(cv::Mat imageLeft,
         cv::Mat imageRight,
         Matrix3f cameraLeft,
         Matrix3f cameraRight);

    // TODO: Add second data constructor including disparities

    /**
     * Getters
     */

    const cv::Mat &getImageLeft() const;

    const cv::Mat &getImageRight() const;

    const Matrix3f &getCameraMatrixLeft() const;

    const Matrix3f &getCameraMatrixRight() const;

    // TODO: get disparity maps

private:
    cv::Mat imageLeft;
    cv::Mat imageRight;
    Matrix3f cameraLeft;
    Matrix3f cameraRight;
};


#endif //STEREO_RECONSTRUCTION_DATA_H
