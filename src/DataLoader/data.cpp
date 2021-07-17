
#include "data.h"


Data::Data(cv::Mat imageLeft, cv::Mat imageRight, Matrix3f cameraLeft, Matrix3f cameraRight, cv::Mat disparityGTLeft,
           cv::Mat disparityGTRight, cv::Mat maskNonOccludedLeft, cv::Mat maskNonOccludedRight)
        : imageLeft{std::move(imageLeft)}, imageRight{std::move(imageRight)}, cameraLeft{std::move(cameraLeft)},
          cameraRight{std::move(cameraRight)}, disparityGTRight{std::move(disparityGTRight)}, disparityGTLeft{std::move(disparityGTLeft)},
          maskNonOccludedRight{std::move(maskNonOccludedRight)}, maskNonOccludedLeft{std::move(maskNonOccludedLeft)}{

}

Data::Data(cv::Mat imageLeft, cv::Mat imageRight, Matrix3f cameraLeft, Matrix3f cameraRight)
: imageLeft{std::move(imageLeft)}, imageRight{std::move(imageRight)}, cameraLeft{std::move(cameraLeft)},
cameraRight{std::move(cameraRight)}
{
    disparityGTRight = NULL;
    disparityGTLeft = NULL;
    maskNonOccludedLeft = NULL;
    maskNonOccludedRight = NULL;
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

const cv::Mat &Data::getMaskNonOccludedLeft() const {
    return maskNonOccludedLeft;
}

const cv::Mat &Data::getMaskNonOccludedRight() const {
    return maskNonOccludedRight;
}

