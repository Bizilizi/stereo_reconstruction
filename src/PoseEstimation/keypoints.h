
#ifndef STEREO_RECONSTRUCTION_KEYPOINTS_H
#define STEREO_RECONSTRUCTION_KEYPOINTS_H

#define N_KEYPOINTS_SIFT 500

#include <opencv4/opencv2/opencv.hpp>
#include "../Eigen.h"
#include "eight_point.h"

void featureMatching(const cv::Mat &featuresLeft, const cv::Mat &featuresRight, std::vector<cv::DMatch> &outputMatches,
                     float ratio_thresh = 0.7f);

/**
 *
 * @param image
 * @param outputKeypoints
 * @param outputDescriptor
 * @param edgeThreshold: the larger, the more features (standard is 10)
 * @param contrastThreshold : the smaller, the more features (standard is 0.04)
 */
void SIFTKeypointDetection(const cv::Mat &image, std::vector<cv::KeyPoint> &outputKeypoints, cv::Mat &outputDescriptor,
                           const float edgeThreshold = 5, const float contrastThreshold = 0.01);


void showExtrinsicsReconstruction(const std::string &filename, const Matrix4f &pose,
                                  const Matrix3Xf &pointsLeftReconstructed, const Matrix3Xf &pointsRightReconstructed);



#endif //STEREO_RECONSTRUCTION_KEYPOINTS_H
