//
// Created by gabriel on 07.07.21.
//

#include "data.h"


Data::Data(cv::Mat imageLeft, cv::Mat imageRight, Matrix3f cameraLeft, Matrix3f cameraRight, cv::Mat disparityGTLeft,
           cv::Mat disparityGTRight)
        : imageLeft{std::move(imageLeft)}, imageRight{std::move(imageRight)}, cameraLeft{std::move(cameraLeft)},
          cameraRight{std::move(cameraRight)}, disparityGTRight{std::move(disparityGTRight)}, disparityGTLeft{std::move(disparityGTLeft)} {

}

Data::Data(cv::Mat imageLeft, cv::Mat imageRight, Matrix3f cameraLeft, Matrix3f cameraRight)
: imageLeft{std::move(imageLeft)}, imageRight{std::move(imageRight)}, cameraLeft{std::move(cameraLeft)},
cameraRight{std::move(cameraRight)}
{
    disparityGTRight = NULL;
    disparityGTLeft = NULL;
}


const cv::Mat &Data::getImageLeft() const {
    return imageLeft;
}

const cv::Mat &Data::getImageRight() const {
    return imageRight;
}

const Matrix3f &Data::getCameraMatrixLeft() const {
    return cameraLeft;
}

const Matrix3f &Data::getCameraMatrixRight() const {
    return cameraRight;
}

const cv::Mat &Data::getDisparityLeft() const {
    return disparityGTLeft;
}

const cv::Mat &Data::getDisparityRight() const {
    return disparityGTRight;
}

