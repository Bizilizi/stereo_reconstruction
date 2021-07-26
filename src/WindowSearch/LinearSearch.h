#ifndef STEREO_RECONSTRUCTION_LINEARSEARCH_H
#define STEREO_RECONSTRUCTION_LINEARSEARCH_H

#include "../Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

class LinearSearch {
    cv::Mat leftImage;
    cv::Mat rightImage;
public:
    LinearSearch(cv::Mat &leftImage, cv::Mat &rightImage);
    /**
     * compute the disparity map of right image
     * @param smoothFactor
     * @return disparity map
     */
    cv::Mat computeDisparityMap(double smoothFactor);
};


#endif //STEREO_RECONSTRUCTION_LINEARSEARCH_H
