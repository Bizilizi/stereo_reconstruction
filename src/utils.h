//
// Created by tim on 14.07.21.
//

#ifndef STEREO_RECONSTRUCTION_UTILS_H
#define STEREO_RECONSTRUCTION_UTILS_H

#include "Eigen.h"
#include <opencv4/opencv2/opencv.hpp>

/**
 * Converts a vector to its corresponding skew symmetric matrix.
 * @param vec Vector to b e converted
 * @return Skew symmetric matrix
 */
Matrix3f vectorAsSkew(const Vector3f &vec);

/**
 * Kronecker product for two vectors.
 * @param vec1
 * @param vec2
 * @return Kronecker product
 */
VectorXf kron(const VectorXf &vec1, const VectorXf &vec2);

/**
 * Convert the matched keypoint pairs given to two Eigen matrices.
 * @param keypointsLeft vector of keypoints in left image
 * @param keypointsRight vector of keypoints in right image
 * @param matches cv::DMatch instance that contains the indices of matched keypoints
 * @param outLeft output matrix for keypoints in left picture
 * @param outRight output matrix for keypoints in right picture
 * @param filterDuplicates filters out duplicates in matches
 */
void transformMatchedKeypointsToEigen(const std::vector<cv::KeyPoint> &keypointsLeft,
                                      const std::vector<cv::KeyPoint> &keypointsRight,
                                      const std::vector<cv::DMatch> &matches,
                                      Matrix3Xf &outLeft,
                                      Matrix3Xf &outRight,
                                      bool filterDuplicates = true);

/**
 * Returns the column indices corresponding to unique elements.
 * @param pointMat: (3,n)-matrix containing column-wise 3d vectors
 * @param tol: max. tolerance for comparing elements
 * @return vector of unique elements id
 */
std::vector<int> uniqueColumnsInMatrix(const Matrix3Xf &pointMat, float tol=0.1f);

/**
 * Computes the reprojection error.
 * @param matchesLeft
 * @param matchesRight
 * @param intrinsicsLeft
 * @param intrinsicsRight
 * @param rotation
 * @param translation
 * @param reconstructedPointsLeft
 * @return error
 */
float averageReconstructionError(const Matrix3Xf& matchesLeft, const Matrix3Xf& matchesRight,
                                 const Matrix3f& intrinsicsLeft, const Matrix3f& intrinsicsRight,
                                 const Matrix3f& rotation, const Vector3f& translation,
                                 const Matrix3Xf& reconstructedPointsLeft);

#endif //STEREO_RECONSTRUCTION_UTILS_H
