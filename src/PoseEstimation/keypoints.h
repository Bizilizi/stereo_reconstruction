
#ifndef STEREO_RECONSTRUCTION_KEYPOINTS_H
#define STEREO_RECONSTRUCTION_KEYPOINTS_H

#define N_KEYPOINTS_SIFT 500

#include <opencv4/opencv2/opencv.hpp>
#include "../Eigen.h"
#include "eight_point.h"

/**
 * Match features using the ratio from the first nearest neighbor to the second nearest neighbor. RootSIFT is used as similarity metric for two feature descritors.
 * @param featuresLeft: feature descriptors in the left image
 * @param featuresRight: feature descriptors in the second image
 * @param outputMatches: output is put here
 * @param ratio_thresh: kNN threshold as described before
 */
void featureMatching(const cv::Mat &featuresLeft, const cv::Mat &featuresRight, std::vector<cv::DMatch> &outputMatches,
                     float ratio_thresh = 0.7f);

/**
 * Computes SIFT features for given input image.
 * @param image: input image
 * @param outputKeypoints: output for storing keypoints in pixel coordinates
 * @param outputDescriptor: output for the computed SIFT feature descriptors
 * @param edgeThreshold: the larger, the more features (standard is 10)
 * @param contrastThreshold : the smaller, the more features (standard is 0.04)
 */
void SIFTKeypointDetection(const cv::Mat &image, std::vector<cv::KeyPoint> &outputKeypoints, cv::Mat &outputDescriptor,
                           const float edgeThreshold = 5, const float contrastThreshold = 0.01);


/**
 * Creates an .OFF-file that visualizes the relative pose between two cameras and draws 3D points as detectetd from both views.
 * @param filename: Relative or absolute path where to store the image.
 * @param pose: relative pose defining rotation and translation from left to right camera frame
 * @param pointsLeftReconstructed: 3D points observed in left frame
 * @param pointsRightReconstructed: 3D points observed in right frame
 */
void showExtrinsicsReconstruction(const std::string &filename, const Matrix4f &pose,
                                  const Matrix3Xf &pointsLeftReconstructed, const Matrix3Xf &pointsRightReconstructed);



#endif //STEREO_RECONSTRUCTION_KEYPOINTS_H
