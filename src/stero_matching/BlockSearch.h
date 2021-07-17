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
			  int maxDisparity);

  cv::Mat computeDisparityMap();
 private:
  cv::Mat leftImage_;
  cv::Mat rightImage_;
  int blockSize_;
  int maxDisparity_;
};
#endif //STEREO_RECONSTRUCTION_BLOCKSEARCH_H
