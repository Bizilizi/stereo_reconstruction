#ifndef STEREO_RECONSTRUCTION_BLOCKSEARCH_H
#define STEREO_RECONSTRUCTION_BLOCKSEARCH_H

#include "../Eigen.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

class BlockSearch {
public:
    BlockSearch(cv::Mat &leftImage,
                cv::Mat &rightImage,
                int blockSize,
                int minDisparity,
                int maxDisparity);

    BlockSearch(cv::Mat &leftImage,
                cv::Mat &rightImage,
                std::vector<int> &blockSizes,
                int minDisparity,
                int maxDisparity);

    /**
     * Computes the disparity of left image
     * @param smoothFactor: a factor between 0 anf 1 to smooth the result,
     * @return disparity map
     */
    cv::Mat computeDisparityMapLeft(double smoothFactor);

    /**
     * Computes the disparity of right image
     * @param smoothFactor: a factor between 0 anf 1 to smooth the result,
     * @param varBlock: if using vary window size
     * @param thres: threshold only used for varying window size
     * @return disparity map
     */
    cv::Mat computeDisparityMapRight(double smoothFactor, bool varBlock = false, double thres = 19.0);

private:
    cv::Mat leftImage_;
    cv::Mat rightImage_;
    int blockSize_;
    int maxDisparity_;
    int minDisparity_;
    std::vector<int> blockSizes_;
};

#endif //STEREO_RECONSTRUCTION_BLOCKSEARCH_H
