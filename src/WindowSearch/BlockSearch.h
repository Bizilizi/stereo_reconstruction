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

  cv::Mat computeDisparityMapLeft(double smoothFactor);
  cv::Mat computeDisparityMapRight(double smoothFactor, bool gaussianBlur=false);
 private:
  cv::Mat leftImage_;
  cv::Mat rightImage_;
  int blockSize_;
  int maxDisparity_;
  int minDisparity_;
  std::vector<int> blockSizes_;
};
#endif //STEREO_RECONSTRUCTION_BLOCKSEARCH_H
