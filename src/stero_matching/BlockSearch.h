#ifndef STEREO_RECONSTRUCTION_BLOCKSEARCH_H
#define STEREO_RECONSTRUCTION_BLOCKSEARCH_H

#include "../Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

class BlockSearch {
    cv::Mat leftImage;
    cv::Mat rightImage;
    uint8_t blockSize;
public:
    BlockSearch(cv::Mat &leftImage, cv::Mat &rightImage, uint8_t blockSize);

    cv::Mat computeDisparityMap();
};
#endif //STEREO_RECONSTRUCTION_BLOCKSEARCH_H
