//
// Created by gabriel on 21.06.21.
//

#ifndef STEREO_RECONSTRUCTION_EIGHT_POINT_H
#define STEREO_RECONSTRUCTION_EIGHT_POINT_H

#include "Eigen.h"
void eightPointAlgorithm(const Matrix3Xf& matchesLeft,
                         const Matrix3Xf& matchesRight,
                         const Matrix3f& cameraLeft,
                         const Matrix3f& cameraRight,
                         Matrix4f& pose,
                         Matrix3f& essentialMatrix);

/**
 * Reconstructs depth of corresponding 2D points in two views by triangulation.
 * @param R (3,3) Rotation matrix
 * @param T (3) Translation vector
 * @param xLeft (3, N) homogenous coordinates for matched points in left picture
 * @param xRight (3, N) homogenous coordinates for matched points in left picture
 * @param xLeftReconstructed output
 * @param xRightReconstructed output
 * @return success boolean indicating whether all depths are positive (otherwise reconstruction failed)
 */
bool structureReconstruction(const Matrix3f& R, const Vector3f& T, const MatrixXf& xLeft, const MatrixXf& xRight, MatrixXf& xLeftReconstructed, MatrixXf& xRightReconstructed);

/**
 * Converts a vector to its corresponding skew symmetric matrix.
 * @param vec Vector to b e converted
 * @return Skew symmetric matrix
 */
Matrix3f vectorAsSkew(const Vector3f& vec);

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
 */
void transformMatchedKeypointsToEigen(const std::vector<cv::KeyPoint>& keypointsLeft,
                                      const std::vector<cv::KeyPoint>& keypointsRight,
                                      const std::vector<cv::DMatch>& matches,
                                      Matrix3Xf& outLeft,
                                      Matrix3Xf& outRight);

#endif //STEREO_RECONSTRUCTION_EIGHT_POINT_H
